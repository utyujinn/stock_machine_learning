"""LSTM Trading Advisor - 売買推奨スクリプト.

バックテストで検証済みのLSTMロング・ショート戦略に基づき、
現在のマーケットデータからポジション推奨を表示する。
自動執行は行わず、ユーザーが手動でトレードする前提。

使い方:
  uv run python advisor.py                    # 4hで推奨 (デフォルト)
  uv run python advisor.py --timeframe daily  # 日足で推奨
  uv run python advisor.py --retrain          # モデル再訓練→推奨
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR,
    PORTFOLIO_K,
    SEQUENCE_LENGTH,
    STUDY_PERIODS,
)
from data.preprocessor import compute_returns, prepare_study_period
from models.lstm_model import (
    ensemble_predict_ranks,
    load_ensemble,
    save_ensemble,
    select_best_units,
    train_ensemble,
)

# --- 定数 ---
MODELS_DIR = os.path.join(DATA_DIR, "models")
HOLD_THRESHOLD = 10
SP3_CONFIG = STUDY_PERIODS[2]  # 最新のStudy Period


# ============================================================
# データ取得 (軽量版: 推論用に直近100本のみ)
# ============================================================

def fetch_recent_4h(n_candles: int = 100) -> pd.DataFrame:
    """Binance APIから各銘柄の直近4hキャンドルを取得."""
    from data.collector_4h import BINANCE_TICKERS, BINANCE_API_URL, _ticker_to_column
    import requests

    print("  最新4hデータを Binance から取得中...")
    all_data = {}
    failed = []

    for i, ticker in enumerate(BINANCE_TICKERS):
        col_name = _ticker_to_column(ticker)
        params = {
            "symbol": ticker,
            "interval": "4h",
            "limit": n_candles,
        }
        for attempt in range(3):
            try:
                resp = requests.get(BINANCE_API_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    failed.append(ticker)
                    break
                time.sleep(2)
        else:
            continue

        if ticker in failed:
            continue

        data = resp.json()
        if len(data) < 10:
            failed.append(ticker)
            continue

        timestamps = [datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).replace(tzinfo=None) for k in data]
        closes = [float(k[4]) for k in data]
        all_data[col_name] = pd.Series(closes, index=timestamps)

        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(BINANCE_TICKERS)} 銘柄取得完了")
        time.sleep(0.05)

    if failed:
        print(f"  {len(failed)} 銘柄が取得失敗")

    price_df = pd.DataFrame(all_data).sort_index()
    price_df = price_df[~price_df.index.duplicated(keep="first")]
    print(f"  {len(price_df.columns)} 銘柄 × {len(price_df)} 本取得完了")
    return price_df


def fetch_recent_daily(n_days: int = 180) -> pd.DataFrame:
    """yfinanceから各銘柄の直近日足データを取得."""
    import yfinance as yf
    from data.collector import CRYPTO_TICKERS

    print("  最新日足データを yfinance から取得中...")
    tickers_str = " ".join(CRYPTO_TICKERS)
    raw = yf.download(
        tickers_str,
        period=f"{n_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        price_df = raw["Close"]
    else:
        price_df = raw[["Close"]]
        price_df.columns = CRYPTO_TICKERS[:1]

    price_df = price_df.dropna(how="all")
    print(f"  {len(price_df.columns)} 銘柄 × {len(price_df)} 日取得完了")
    return price_df


# ============================================================
# 特徴量生成
# ============================================================

def build_current_features(
    price_df: pd.DataFrame,
    train_mean: float,
    train_std: float,
) -> tuple[pd.Timestamp, dict[str, np.ndarray], dict[str, float]]:
    """最新時点の特徴量を構築.

    Returns:
        (latest_timestamp, coin_samples, current_prices)
    """
    returns = compute_returns(price_df)
    latest_date = returns.index[-1]

    coin_samples = {}
    current_prices = {}

    for coin in returns.columns:
        seq = returns[coin].iloc[-SEQUENCE_LENGTH:].values
        if len(seq) < SEQUENCE_LENGTH:
            continue
        if not np.all(np.isfinite(seq)):
            continue

        seq_norm = (seq - train_mean) / train_std
        coin_samples[coin] = seq_norm.reshape(SEQUENCE_LENGTH, 1).astype(np.float32)

        # 現在価格
        last_price = price_df[coin].dropna().iloc[-1]
        if np.isfinite(last_price):
            current_prices[coin] = float(last_price)

    return latest_date, coin_samples, current_prices


# ============================================================
# 動的リバランス
# ============================================================

def load_positions(path: str) -> dict | None:
    """前回のポジションファイルを読込."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_positions(path: str, data: dict):
    """ポジションファイルを保存."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def dynamic_rebalance(
    ranks: dict[str, float],
    prev_longs: list[str] | None,
    prev_shorts: list[str] | None,
    k: int = PORTFOLIO_K,
    hold_threshold: int = HOLD_THRESHOLD,
) -> tuple[list[str], list[str], dict]:
    """動的リバランスを適用.

    Returns:
        (new_longs, new_shorts, changes_dict)
        changes_dict: {coin: "NEW_LONG"|"KEEP_LONG"|"NEW_SHORT"|"KEEP_SHORT"|"CLOSE_LONG"|"CLOSE_SHORT"}
    """
    sorted_coins = sorted(ranks.keys(), key=lambda c: ranks[c])
    ideal_longs = sorted_coins[:k]
    ideal_shorts = sorted_coins[-k:]
    changes = {}

    if prev_longs is None or prev_shorts is None:
        # 初回
        for c in ideal_longs:
            changes[c] = "NEW_LONG"
        for c in ideal_shorts:
            changes[c] = "NEW_SHORT"
        return ideal_longs, ideal_shorts, changes

    # 動的リバランス
    top_set = set(sorted_coins[:hold_threshold])
    bottom_set = set(sorted_coins[-hold_threshold:])

    # ロング側
    kept_longs = [c for c in prev_longs if c in top_set and c in ranks]
    needed_long = k - len(kept_longs)
    candidates_long = [c for c in ideal_longs if c not in kept_longs]
    new_longs = kept_longs + candidates_long[:needed_long]

    # ショート側
    kept_shorts = [c for c in prev_shorts if c in bottom_set and c in ranks]
    needed_short = k - len(kept_shorts)
    candidates_short = [c for c in ideal_shorts if c not in kept_shorts]
    new_shorts = kept_shorts + candidates_short[:needed_short]

    # 変更を記録
    new_long_set = set(new_longs)
    new_short_set = set(new_shorts)
    prev_long_set = set(prev_longs)
    prev_short_set = set(prev_shorts)

    for c in new_longs:
        changes[c] = "KEEP_LONG" if c in prev_long_set else "NEW_LONG"
    for c in new_shorts:
        changes[c] = "KEEP_SHORT" if c in prev_short_set else "NEW_SHORT"
    for c in prev_longs:
        if c not in new_long_set:
            changes[c] = "CLOSE_LONG"
    for c in prev_shorts:
        if c not in new_short_set:
            changes[c] = "CLOSE_SHORT"

    return new_longs, new_shorts, changes


# ============================================================
# 表示
# ============================================================

def _format_price(price: float) -> str:
    """価格をフォーマット."""
    if price >= 1000:
        return f"${price:,.0f}"
    elif price >= 1:
        return f"${price:.2f}"
    elif price >= 0.01:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"


def _action_label(action: str) -> str:
    """アクションを日本語ラベルに変換."""
    labels = {
        "NEW_LONG": "★ 新規買い",
        "KEEP_LONG": "  継続保有",
        "NEW_SHORT": "★ 新規売り",
        "KEEP_SHORT": "  継続保有",
        "CLOSE_LONG": "→ 売却",
        "CLOSE_SHORT": "→ 買戻し",
    }
    return labels.get(action, action)


def display_recommendations(
    longs: list[str],
    shorts: list[str],
    changes: dict[str, str],
    ranks: dict[str, float],
    prices: dict[str, float],
    timestamp: pd.Timestamp,
    timeframe: str,
    model_info: str,
    n_coins: int,
):
    """推奨を表示."""
    tf_label = "4h candles" if timeframe == "4h" else "daily"
    ts_str = timestamp.strftime("%Y-%m-%d %H:%M UTC")

    print(f"\n{'='*60}")
    print(f"  LSTM Trading Advisor")
    print(f"  {ts_str} | {tf_label} | MEXC")
    print(f"{'='*60}")
    print(f"  銘柄数: {n_coins} | {model_info}")

    # LONG
    print(f"\n  {'─'*50}")
    print(f"  LONG (買い推奨 Top {len(longs)})")
    print(f"  {'─'*50}")
    print(f"  {'#':>3}  {'銘柄':<14} {'ランク':>6}  {'アクション':<12} {'価格':>12}")
    for i, coin in enumerate(longs):
        rank = ranks.get(coin, 0)
        price = prices.get(coin, 0)
        action = changes.get(coin, "")
        print(f"  {i+1:>3}  {coin:<14} {rank:>6.1f}  {_action_label(action):<12} {_format_price(price):>12}")

    # SHORT
    print(f"\n  {'─'*50}")
    print(f"  SHORT (空売り推奨 Bottom {len(shorts)})")
    print(f"  {'─'*50}")
    print(f"  {'#':>3}  {'銘柄':<14} {'ランク':>6}  {'アクション':<12} {'価格':>12}")
    for i, coin in enumerate(shorts):
        rank = ranks.get(coin, 0)
        price = prices.get(coin, 0)
        action = changes.get(coin, "")
        print(f"  {i+1:>3}  {coin:<14} {rank:>6.1f}  {_action_label(action):<12} {_format_price(price):>12}")

    # CLOSE
    close_actions = {c: a for c, a in changes.items() if a.startswith("CLOSE")}
    if close_actions:
        print(f"\n  {'─'*50}")
        print(f"  クローズ推奨")
        print(f"  {'─'*50}")
        for coin, action in close_actions.items():
            rank = ranks.get(coin, 0)
            direction = "LONG" if action == "CLOSE_LONG" else "SHORT"
            label = _action_label(action)
            print(f"  • {coin:<14} ({direction}→ランク{rank:.0f}) {label}")

    # サマリー
    n_new = sum(1 for a in changes.values() if a.startswith("NEW"))
    n_keep = sum(1 for a in changes.values() if a.startswith("KEEP"))
    n_close = sum(1 for a in changes.values() if a.startswith("CLOSE"))
    total_positions = len(longs) + len(shorts)

    print(f"\n  {'─'*50}")
    if timeframe == "4h":
        # 次の4h足の時刻を計算
        current_hour = timestamp.hour
        next_4h = (current_hour // 4 + 1) * 4
        if next_4h >= 24:
            next_time = (timestamp + timedelta(days=1)).replace(hour=0, minute=0)
        else:
            next_time = timestamp.replace(hour=next_4h, minute=0)
        next_str = next_time.strftime("%H:%M UTC")
        print(f"  ターンオーバー: {n_new}/{total_positions} 入替 | 次回: {next_str}")
    else:
        print(f"  ターンオーバー: {n_new}/{total_positions} 入替 | 次回: 明日")
    print(f"{'='*60}")


# ============================================================
# 訓練
# ============================================================

def retrain_models(timeframe: str):
    """SP3のデータでモデルを訓練し保存."""
    print(f"\n[モデル訓練] {timeframe}足 SP3 モデルを訓練中...")

    # データ取得
    if timeframe == "4h":
        from data.collector_4h import collect_4h_data
        price_df = collect_4h_data()
    else:
        from data.collector import collect_all_data
        price_df = collect_all_data()

    print(f"  価格データ: {price_df.shape[1]} 銘柄 × {price_df.shape[0]} 本")

    # SP3 の訓練/検証データ準備
    sp_data = prepare_study_period(price_df, SP3_CONFIG)

    # ハイパーパラメータ探索
    print("\n  ハイパーパラメータ探索...")
    best_units = select_best_units(
        sp_data.X_train, sp_data.y_train,
        sp_data.X_val, sp_data.y_val,
    )

    # アンサンブル訓練
    print(f"\n  アンサンブル訓練 (units={best_units})...")
    models = train_ensemble(
        best_units,
        sp_data.X_train, sp_data.y_train,
        sp_data.X_val, sp_data.y_val,
    )

    # 訓練統計を取得 (特徴量標準化に使用)
    returns = compute_returns(price_df)
    train_start, train_end = SP3_CONFIG["train"]
    all_coins = [
        col for col in price_df.columns
        if price_df[col].loc[train_start:SP3_CONFIG["test"][1]].notna().sum() > 100
    ]
    train_vals = returns.loc[train_start:train_end][all_coins].values.flatten()
    train_vals = train_vals[np.isfinite(train_vals)]
    train_mean = float(np.mean(train_vals))
    train_std = float(np.std(train_vals))

    # 保存
    model_path = os.path.join(MODELS_DIR, timeframe)
    config = {
        "best_units": best_units,
        "train_mean": train_mean,
        "train_std": train_std,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sp_id": 3,
    }
    save_ensemble(models, model_path, config)

    # メモリ解放
    import tensorflow as tf
    for m in models:
        del m
    tf.keras.backend.clear_session()

    print("  訓練完了!\n")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LSTM Trading Advisor")
    parser.add_argument(
        "--timeframe", choices=["4h", "daily"], default="4h",
        help="時間足 (default: 4h)",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="モデルを再訓練",
    )
    args = parser.parse_args()

    tf = args.timeframe
    model_path = os.path.join(MODELS_DIR, tf)
    positions_path = os.path.join(DATA_DIR, f"positions_{tf}.json")

    # --- A. モデル準備 ---
    if args.retrain or not os.path.exists(os.path.join(model_path, "config.json")):
        if not args.retrain:
            print(f"保存済みモデルが見つかりません。訓練を開始します...")
        retrain_models(tf)

    print(f"[モデル読込] {model_path}")
    models, config = load_ensemble(model_path)
    train_mean = config["train_mean"]
    train_std = config["train_std"]
    model_info = f"SP{config.get('sp_id', 3)} (units={config['best_units']}, ensemble×{len(models)})"

    # --- B. 最新データ取得 ---
    print(f"\n[データ取得]")
    if tf == "4h":
        price_df = fetch_recent_4h()
    else:
        price_df = fetch_recent_daily()

    # --- C. 特徴量生成 ---
    print(f"\n[特徴量生成]")
    latest_date, coin_samples, current_prices = build_current_features(
        price_df, train_mean, train_std,
    )
    print(f"  {len(coin_samples)} 銘柄の特徴量を生成 (時点: {latest_date})")

    if len(coin_samples) < 2 * PORTFOLIO_K:
        print(f"エラー: 有効な銘柄が {len(coin_samples)} しかありません (最低{2*PORTFOLIO_K}必要)")
        sys.exit(1)

    # --- D. 予測 ---
    print(f"\n[予測]")
    ranks = ensemble_predict_ranks(models, coin_samples)
    print(f"  {len(ranks)} 銘柄のランク予測完了")

    # --- E. 動的リバランス ---
    prev = load_positions(positions_path)
    prev_longs = prev["longs"] if prev else None
    prev_shorts = prev["shorts"] if prev else None

    if prev:
        prev_time = prev.get("timestamp", "不明")
        print(f"\n[リバランス] 前回ポジション: {prev_time}")
    else:
        print(f"\n[リバランス] 初回実行 (前回ポジションなし)")

    new_longs, new_shorts, changes = dynamic_rebalance(
        ranks, prev_longs, prev_shorts,
    )

    # --- F. 推奨表示 ---
    display_recommendations(
        new_longs, new_shorts, changes, ranks, current_prices,
        latest_date, tf, model_info, len(coin_samples),
    )

    # --- G. ポジション保存 ---
    position_data = {
        "timestamp": latest_date.isoformat(),
        "timeframe": tf,
        "longs": new_longs,
        "shorts": new_shorts,
        "all_ranks": {k: float(v) for k, v in ranks.items()},
    }
    save_positions(positions_path, position_data)
    print(f"\nポジション保存: {positions_path}")


if __name__ == "__main__":
    main()
