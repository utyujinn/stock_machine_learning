"""特徴量・ターゲット生成とStudy Period分割.

論文の方法論:
- 日次リターンを計算
- 各日の全銘柄リターンの中央値以上 → 1, それ以外 → 0 (二値分類ターゲット)
- LSTM用: 過去90日間のリターン系列を標準化して入力特徴量とする
- 各Study Periodで独立に訓練セットの統計量で標準化
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import SEQUENCE_LENGTH, STUDY_PERIODS


@dataclass
class StudyPeriodData:
    """1つのStudy Periodの学習・検証・テストデータ."""
    sp_id: int
    # 訓練データ
    X_train: np.ndarray  # (n_samples, 90, 1)
    y_train: np.ndarray  # (n_samples,)
    # 検証データ
    X_val: np.ndarray
    y_val: np.ndarray
    # テストデータ
    X_test: np.ndarray
    y_test: np.ndarray
    # テスト期間のメタデータ (バックテスト用)
    test_dates: list[pd.Timestamp]
    test_coin_ids: list[str]
    test_returns: pd.DataFrame     # テスト期間の日次リターン (日付 × 銘柄)
    # 全銘柄のテスト期間予測用データ (日付ごと)
    test_samples_by_day: dict      # {日付: {coin_id: X_sample}}


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """日次リターンを計算: r_t = p_t / p_{t-1} - 1."""
    return price_df.pct_change()


def compute_targets(returns_df: pd.DataFrame) -> pd.DataFrame:
    """二値ターゲットを計算.

    各日の全銘柄リターンの中央値以上 → 1, それ以外 → 0.
    """
    daily_median = returns_df.median(axis=1)
    targets = returns_df.ge(daily_median, axis=0).astype(int)
    return targets


def prepare_study_period(
    price_df: pd.DataFrame,
    sp_config: dict,
) -> StudyPeriodData:
    """1つのStudy Periodの学習データを準備.

    Args:
        price_df: 全銘柄の日次価格データ (日付 × 銘柄)
        sp_config: Study Period設定 (train/val/test の日付範囲)
    """
    sp_id = sp_config["id"]
    train_start, train_end = sp_config["train"]
    val_start, val_end = sp_config["val"]
    test_start, test_end = sp_config["test"]

    # 利用可能な銘柄: 訓練開始日時点で価格データが存在する銘柄
    # かつ、Study Period 全体で十分なデータがある銘柄
    available_coins = []
    for col in price_df.columns:
        series = price_df[col].dropna()
        if len(series) == 0:
            continue
        # 訓練開始の90日前からテスト終了日までデータがあること
        from datetime import timedelta
        buffer_start = pd.Timestamp(train_start) - timedelta(days=SEQUENCE_LENGTH + 10)
        if series.index.min() <= buffer_start and series.index.max() >= pd.Timestamp(test_end):
            available_coins.append(col)

    if len(available_coins) == 0:
        raise ValueError(f"SP{sp_id}: 利用可能な銘柄がありません")

    print(f"  SP{sp_id}: {len(available_coins)} 銘柄が利用可能")

    # 利用可能な銘柄のみの価格データ
    prices = price_df[available_coins].copy()

    # 日次リターン
    returns = compute_returns(prices)
    # ターゲット (二値分類)
    targets = compute_targets(returns)

    # --- 各セットのデータ作成 ---
    def _make_sequences(start: str, end: str, mean: float, std: float):
        """指定期間のサンプルを作成."""
        period_dates = returns.loc[start:end].index
        X_list = []
        y_list = []

        for date in period_dates:
            date_idx = returns.index.get_loc(date)
            if date_idx < SEQUENCE_LENGTH:
                continue

            for coin in available_coins:
                # 過去90日間のリターン系列
                seq = returns[coin].iloc[date_idx - SEQUENCE_LENGTH:date_idx].values
                target = targets.loc[date, coin]

                if np.isnan(seq).any() or np.isnan(target):
                    continue

                # 標準化
                seq_normalized = (seq - mean) / (std + 1e-8)
                X_list.append(seq_normalized)
                y_list.append(target)

        if not X_list:
            return np.array([]).reshape(0, SEQUENCE_LENGTH, 1), np.array([])

        X = np.array(X_list).reshape(-1, SEQUENCE_LENGTH, 1)
        y = np.array(y_list)
        return X, y

    def _make_test_samples_by_day(start: str, end: str, mean: float, std: float):
        """テスト期間の日付ごと・銘柄ごとのサンプルを作成 (バックテスト用)."""
        period_dates = returns.loc[start:end].index
        samples_by_day = {}

        for date in period_dates:
            date_idx = returns.index.get_loc(date)
            if date_idx < SEQUENCE_LENGTH:
                continue

            day_samples = {}
            for coin in available_coins:
                seq = returns[coin].iloc[date_idx - SEQUENCE_LENGTH:date_idx].values
                if np.isnan(seq).any():
                    continue
                seq_normalized = (seq - mean) / (std + 1e-8)
                day_samples[coin] = seq_normalized.reshape(SEQUENCE_LENGTH, 1)

            if day_samples:
                samples_by_day[date] = day_samples

        return samples_by_day

    # 訓練セットの統計量を計算 (標準化用)
    train_returns = returns.loc[train_start:train_end]
    train_mean = train_returns.values[~np.isnan(train_returns.values)].mean()
    train_std = train_returns.values[~np.isnan(train_returns.values)].std()

    print(f"    訓練セット統計: mean={train_mean:.6f}, std={train_std:.6f}")

    # 各セットの作成
    print(f"    訓練データ作成中...")
    X_train, y_train = _make_sequences(train_start, train_end, train_mean, train_std)
    print(f"    検証データ作成中...")
    X_val, y_val = _make_sequences(val_start, val_end, train_mean, train_std)
    print(f"    テストデータ作成中...")
    X_test, y_test = _make_sequences(test_start, test_end, train_mean, train_std)

    # テスト期間のバックテスト用データ
    print(f"    テスト期間のバックテスト用データ作成中...")
    test_samples_by_day = _make_test_samples_by_day(
        test_start, test_end, train_mean, train_std
    )

    # テスト期間の日次リターン (バックテスト用)
    test_returns = returns.loc[test_start:test_end][available_coins]

    print(f"    サンプル数 - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"    テスト期間: {len(test_samples_by_day)} 日")

    return StudyPeriodData(
        sp_id=sp_id,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_dates=sorted(test_samples_by_day.keys()),
        test_coin_ids=available_coins,
        test_returns=test_returns,
        test_samples_by_day=test_samples_by_day,
    )


def prepare_all_study_periods(price_df: pd.DataFrame) -> list[StudyPeriodData]:
    """全Study Periodのデータを準備."""
    results = []
    for sp_config in STUDY_PERIODS:
        print(f"\nStudy Period {sp_config['id']} を準備中...")
        sp_data = prepare_study_period(price_df, sp_config)
        results.append(sp_data)
    return results
