"""特徴量・ターゲット生成とStudy Period分割.

論文の方法論:
- 日次リターンを計算
- 各日の全銘柄リターンの中央値以上 → 1, それ以外 → 0 (二値分類ターゲット)
- LSTM用: 過去90日間のリターン系列を標準化して入力特徴量とする
- 各Study Periodで独立に訓練セットの統計量で標準化
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from config import SEQUENCE_LENGTH, STUDY_PERIODS


@dataclass
class StudyPeriodData:
    """1つのStudy Periodの学習・検証・テストデータ."""
    sp_id: int
    X_train: np.ndarray   # (n_samples, 90, 1)
    y_train: np.ndarray   # (n_samples,)
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_dates: list[pd.Timestamp]
    test_coin_ids: list[str]
    test_returns: pd.DataFrame
    test_samples_by_day: dict


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """日次リターンを計算: r_t = p_t / p_{t-1} - 1."""
    returns = price_df.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    # 極端なリターンをクリップ (±100%)
    # 暗号資産でも1日±100%を超えるのはデータ異常 or 取引不可能な動き
    returns = returns.clip(-1.0, 1.0)
    return returns


def compute_targets(returns_df: pd.DataFrame) -> pd.DataFrame:
    """二値ターゲットを計算.

    各日の全銘柄リターンの中央値以上 → 1, それ以外 → 0.
    NaN のリターンは NaN のまま保持 (学習データに含めない).
    """
    daily_median = returns_df.median(axis=1)  # NaN は自動でスキップ
    # ge() は NaN に対して False を返すが、NaN を NaN として保持したい
    targets = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
    for col in returns_df.columns:
        mask = returns_df[col].notna()
        targets.loc[mask, col] = (returns_df.loc[mask, col] >= daily_median[mask]).astype(float)
    return targets


def prepare_study_period(
    price_df: pd.DataFrame,
    sp_config: dict,
) -> StudyPeriodData:
    """1つのStudy Periodの学習データを準備."""
    sp_id = sp_config["id"]
    train_start, train_end = sp_config["train"]
    val_start, val_end = sp_config["val"]
    test_start, test_end = sp_config["test"]

    # 全銘柄でリターンとターゲットを計算
    returns = compute_returns(price_df)
    targets = compute_targets(returns)

    # 利用可能な銘柄: 各日ごとに動的に判定 (NaNでない銘柄のみ参加)
    # ただし、全体のコインリストは価格データが一定量ある銘柄に限定
    all_coins = [
        col for col in price_df.columns
        if price_df[col].loc[train_start:test_end].notna().sum() > 100
    ]
    print(f"  SP{sp_id}: {len(all_coins)} 銘柄が利用可能")

    # 訓練セットの統計量 (標準化用) - NaN/inf除去
    train_vals = returns.loc[train_start:train_end][all_coins].values.flatten()
    train_vals = train_vals[np.isfinite(train_vals)]
    train_mean = float(np.mean(train_vals))
    train_std = float(np.std(train_vals))

    if not np.isfinite(train_mean) or not np.isfinite(train_std) or train_std < 1e-10:
        print(f"    警告: 統計量が異常 (mean={train_mean}, std={train_std})、デフォルト値使用")
        train_mean = 0.0
        train_std = 0.05

    print(f"    訓練セット統計: mean={train_mean:.6f}, std={train_std:.6f}")

    def _make_sequences(start: str, end: str):
        """指定期間のサンプルを作成."""
        period_dates = returns.loc[start:end].index
        X_list, y_list = [], []

        for date in period_dates:
            date_idx = returns.index.get_loc(date)
            if date_idx < SEQUENCE_LENGTH:
                continue

            for coin in all_coins:
                # 当日のリターンとターゲットがNaNなら スキップ
                ret_val = returns.loc[date, coin]
                tgt_val = targets.loc[date, coin]
                if not np.isfinite(ret_val) or not np.isfinite(tgt_val):
                    continue

                # 過去90日間のリターン系列
                seq = returns[coin].iloc[date_idx - SEQUENCE_LENGTH:date_idx].values
                if not np.all(np.isfinite(seq)):
                    continue

                seq_normalized = (seq - train_mean) / train_std
                X_list.append(seq_normalized)
                y_list.append(tgt_val)

        if not X_list:
            return np.array([]).reshape(0, SEQUENCE_LENGTH, 1), np.array([])

        X = np.array(X_list, dtype=np.float32).reshape(-1, SEQUENCE_LENGTH, 1)
        y = np.array(y_list, dtype=np.float32)
        return X, y

    def _make_test_samples_by_day(start: str, end: str):
        """テスト期間の日付ごと・銘柄ごとのサンプルを作成."""
        period_dates = returns.loc[start:end].index
        samples_by_day = {}

        for date in period_dates:
            date_idx = returns.index.get_loc(date)
            if date_idx < SEQUENCE_LENGTH:
                continue

            day_samples = {}
            for coin in all_coins:
                seq = returns[coin].iloc[date_idx - SEQUENCE_LENGTH:date_idx].values
                if not np.all(np.isfinite(seq)):
                    continue
                seq_normalized = (seq - train_mean) / train_std
                day_samples[coin] = seq_normalized.reshape(SEQUENCE_LENGTH, 1).astype(np.float32)

            if day_samples:
                samples_by_day[date] = day_samples

        return samples_by_day

    print(f"    訓練データ作成中...")
    X_train, y_train = _make_sequences(train_start, train_end)
    print(f"    検証データ作成中...")
    X_val, y_val = _make_sequences(val_start, val_end)
    print(f"    テストデータ作成中...")
    X_test, y_test = _make_sequences(test_start, test_end)

    print(f"    テスト期間のバックテスト用データ作成中...")
    test_samples_by_day = _make_test_samples_by_day(test_start, test_end)

    test_returns = returns.loc[test_start:test_end][all_coins]

    # ラベルバランス確認
    if len(y_train) > 0:
        pos_rate = y_train.mean()
        print(f"    サンプル数 - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"    ラベル分布 - Train正例率: {pos_rate:.3f}")
    print(f"    テスト期間: {len(test_samples_by_day)} 日")

    return StudyPeriodData(
        sp_id=sp_id,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        test_dates=sorted(test_samples_by_day.keys()),
        test_coin_ids=all_coins,
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
