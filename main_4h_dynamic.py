"""4時間足 動的リバランス実験.

3つのコストシナリオを比較:
  A) 0 bps (手数料0) + 動的リバランス
  B) 5 bps (スリッページのみ) + 動的リバランス
  C) 15 bps フルターンオーバー (前回の4h実験、参考)

--dynamic フラグ (ポイントインタイム方式):
  固定リスト + 現在の動的リストの和集合を候補とし、
  各SPの訓練開始時点での Binance 出来高でフィルタ。
  ルックアヘッドバイアスを排除した公正な比較。
"""

import argparse
import os
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(line_buffering=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DYNAMIC_TICKER_MIN_VOLUME_USD,
    PERIODS_PER_YEAR_4H,
    PORTFOLIO_K,
    SEQUENCE_LENGTH,
    STUDY_PERIODS,
)
from data.collector_4h import collect_4h_data
from data.preprocessor import prepare_all_study_periods, prepare_study_period
from models.lstm_model import (
    ensemble_predict_ranks,
    select_best_units,
    train_ensemble,
)
from backtest.engine import (
    BacktestResult,
    calculate_metrics,
    print_metrics,
    run_backtest,
    run_backtest_dynamic,
)


SCENARIOS = [
    {"name": "0bps + 動的リバランス", "cost_bps": 0, "hold_threshold": 10, "dynamic": True},
    {"name": "5bps + 動的リバランス", "cost_bps": 5, "hold_threshold": 10, "dynamic": True},
    {"name": "15bps フルターンオーバー", "cost_bps": 15, "hold_threshold": 0, "dynamic": False},
]


def run_study_period(sp_data, sp_config):
    """1つのStudy Periodの訓練・予測を実行し、ランクを返す."""
    sp_id = sp_data.sp_id
    print(f"\n{'#'*60}")
    print(f"  Study Period {sp_id}")
    print(f"  Test: {sp_config['test'][0]} ~ {sp_config['test'][1]}")
    print(f"{'#'*60}")

    print(f"\n  [1/3] ハイパーパラメータ探索...")
    best_units = select_best_units(
        sp_data.X_train, sp_data.y_train,
        sp_data.X_val, sp_data.y_val,
    )

    print(f"\n  [2/3] アンサンブル訓練 (ユニット数={best_units})...")
    models = train_ensemble(
        best_units,
        sp_data.X_train, sp_data.y_train,
        sp_data.X_val, sp_data.y_val,
    )

    print(f"\n  [3/3] テスト期間の予測...")
    period_ranks = {}
    for date in sp_data.test_dates:
        if date not in sp_data.test_samples_by_day:
            continue
        coin_samples = sp_data.test_samples_by_day[date]
        if len(coin_samples) < 2 * PORTFOLIO_K:
            continue
        ranks = ensemble_predict_ranks(models, coin_samples)
        period_ranks[date] = ranks

    print(f"    予測完了: {len(period_ranks)} 期間分")

    # メモリ解放
    import tensorflow as tf
    for m in models:
        del m
    tf.keras.backend.clear_session()

    return period_ranks


def run_scenario_backtest(scenario, period_ranks, test_returns):
    """1つのシナリオでバックテストを実行."""
    if scenario["dynamic"]:
        return run_backtest_dynamic(
            period_ranks,
            test_returns,
            cost_bps=scenario["cost_bps"],
            hold_threshold=scenario["hold_threshold"],
        )
    else:
        return run_backtest(
            period_ranks,
            test_returns,
            cost_bps=scenario["cost_bps"],
        )


def plot_comparison(scenario_results: dict, output_dir: str):
    """シナリオ比較グラフを出力."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 全期間NAV比較
    ax = axes[0]
    for name, results in scenario_results.items():
        all_returns = pd.concat([r.daily_returns for r in results]).sort_index()
        nav = (1 + all_returns).cumprod()
        ax.plot(nav.index, nav.values, label=name, linewidth=1.5)
    ax.set_title("Scenario Comparison: Portfolio NAV (4h candles, k=5)")
    ax.set_ylabel("Performance Index")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # シナリオ別シャープレシオ
    ax = axes[1]
    names = list(scenario_results.keys())
    sharpes = []
    for name, results in scenario_results.items():
        all_returns = pd.concat([r.daily_returns for r in results]).sort_index()
        m = calculate_metrics(all_returns, periods_per_year=PERIODS_PER_YEAR_4H)
        sharpes.append(m.sharpe_ratio)
    bars = ax.bar(range(len(names)), sharpes, color=["green", "orange", "red"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.set_title("Sharpe Ratio by Scenario")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "scenario_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n比較グラフ保存: {fig_path}")


# ============================================================
# ポイントインタイム動的銘柄選定
# ============================================================

def collect_pointintime_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """候補全銘柄の4h価格+出来高データを収集.

    固定リスト + 現在の動的リストの和集合をユニバースとし、
    価格と出来高を同時に取得する。

    Returns:
        (price_df, volume_df)
    """
    from data.collector_4h import BINANCE_TICKERS, _ticker_to_column, _fetch_klines
    from data.ticker_discovery import get_dynamic_tickers

    # 候補 = 固定リスト ∪ 動的リスト
    print("\n  動的ティッカー選定中...")
    dynamic = get_dynamic_tickers()
    all_candidates = sorted(set(BINANCE_TICKERS) | set(dynamic))
    print(f"  候補銘柄: {len(all_candidates)} (固定{len(BINANCE_TICKERS)} ∪ 動的{len(dynamic)})")

    # 全SPをカバーする期間を計算 (余裕をもって)
    earliest = min(sp["train"][0] for sp in STUDY_PERIODS)
    earliest_dt = datetime.strptime(earliest, "%Y-%m-%d")
    now_dt = datetime.now(timezone.utc)
    days_needed = (now_dt - earliest_dt.replace(tzinfo=timezone.utc)).days + 60 + SEQUENCE_LENGTH
    n_candles = days_needed * 6 + 100

    now_ms = int(now_dt.timestamp() * 1000)
    start_ms = now_ms - n_candles * 4 * 3600 * 1000

    print(f"  ヒストリカルデータ取得 ({n_candles}本 × {len(all_candidates)}銘柄)...")

    price_data = {}
    volume_data = {}
    failed = []

    for i, ticker in enumerate(all_candidates):
        col_name = _ticker_to_column(ticker)
        data = _fetch_klines(ticker, "4h", start_ms, now_ms)

        if len(data) < 10:
            failed.append(ticker)
            continue

        timestamps = [
            datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).replace(tzinfo=None)
            for k in data
        ]
        closes = [float(k[4]) for k in data]
        volumes = [float(k[7]) for k in data]  # quote volume (USDT)

        price_data[col_name] = pd.Series(closes, index=timestamps)
        volume_data[col_name] = pd.Series(volumes, index=timestamps)

        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(all_candidates)} 銘柄取得完了")

    if failed:
        print(f"  {len(failed)} 銘柄が取得失敗")

    price_df = pd.DataFrame(price_data).sort_index()
    price_df = price_df[~price_df.index.duplicated(keep="first")]
    volume_df = pd.DataFrame(volume_data).sort_index()
    volume_df = volume_df[~volume_df.index.duplicated(keep="first")]

    print(f"  {len(price_df.columns)} 銘柄 × {len(price_df)} 本取得完了")
    return price_df, volume_df


def filter_by_volume_at_date(
    volume_df: pd.DataFrame,
    cutoff_date: str,
    min_daily_volume_usd: float,
    lookback_candles: int = 180,
) -> list[str]:
    """指定日以前の出来高で銘柄をフィルタ.

    Args:
        volume_df: 4h出来高DataFrame (USDT建て)
        cutoff_date: この日以前のデータのみ使用 (ルックアヘッド防止)
        min_daily_volume_usd: 日次最低出来高 (USDT)
        lookback_candles: 出来高計算期間 (4h足本数, 180=30日)

    Returns:
        フィルタ通過したカラム名リスト
    """
    window = volume_df.loc[:cutoff_date].iloc[-lookback_candles:]
    if len(window) < 6:
        return list(volume_df.columns)

    # 4h足の平均出来高 × 6 = 日次平均出来高
    avg_4h = window.mean()
    daily_avg = avg_4h * 6
    liquid = daily_avg[daily_avg >= min_daily_volume_usd].dropna().index.tolist()
    return liquid


def main():
    parser = argparse.ArgumentParser(description="4h 動的リバランス実験")
    parser.add_argument(
        "--dynamic", action="store_true",
        help="ポイントインタイム方式: 各SP開始時の出来高で銘柄を動的選定",
    )
    args = parser.parse_args()

    ticker_mode = "ポイントインタイム動的" if args.dynamic else "固定"
    print("=" * 60)
    print(f"  4時間足 動的リバランス実験 [{ticker_mode}銘柄]")
    print("  シナリオ比較: 0bps / 5bps / 15bps")
    print("=" * 60)

    # --- データ収集 ---
    print(f"\n[Step 1] 4hデータ収集...")
    if args.dynamic:
        price_df, volume_df = collect_pointintime_data()
    else:
        price_df = collect_4h_data()
    print(f"  価格データ: {price_df.shape[1]} 銘柄 × {price_df.shape[0]} 本")

    # --- 前処理 ---
    print("\n[Step 2] データ前処理...")
    if args.dynamic:
        # ポイントインタイム: 各SPの訓練開始時点で出来高フィルタ
        sp_data_list = []
        for sp_config in STUDY_PERIODS:
            sp_id = sp_config["id"]
            train_start = sp_config["train"][0]
            liquid = filter_by_volume_at_date(
                volume_df, train_start, DYNAMIC_TICKER_MIN_VOLUME_USD,
            )
            filtered = price_df[[c for c in liquid if c in price_df.columns]]
            print(f"  SP{sp_id}: {len(filtered.columns)} 銘柄 "
                  f"(出来高 > ${DYNAMIC_TICKER_MIN_VOLUME_USD/1e6:.0f}M @ {train_start})")
            sp_data = prepare_study_period(filtered, sp_config)
            sp_data_list.append(sp_data)
    else:
        sp_data_list = prepare_all_study_periods(price_df)

    # --- 訓練・予測 (全シナリオで共通) ---
    print("\n[Step 3] 訓練・予測...")
    all_ranks = []  # SP毎のランク辞書
    for sp_data, sp_config in zip(sp_data_list, STUDY_PERIODS):
        period_ranks = run_study_period(sp_data, sp_config)
        all_ranks.append(period_ranks)

    # --- シナリオ別バックテスト ---
    print("\n" + "=" * 60)
    print(f"  シナリオ別バックテスト [{ticker_mode}銘柄]")
    print("=" * 60)

    scenario_results = {}

    for scenario in SCENARIOS:
        name = scenario["name"]
        print(f"\n{'─'*50}")
        print(f"  シナリオ: {name}")
        print(f"{'─'*50}")

        sp_results = []
        for i, (sp_data, sp_config) in enumerate(zip(sp_data_list, STUDY_PERIODS)):
            sp_id = sp_config["id"]
            print(f"\n  SP{sp_id} バックテスト ({name})...")
            result = run_scenario_backtest(scenario, all_ranks[i], sp_data.test_returns)
            result.sp_id = sp_id
            sp_results.append(result)

            metrics = calculate_metrics(result.daily_returns, periods_per_year=PERIODS_PER_YEAR_4H)
            if len(result.daily_accuracy) > 0:
                metrics.accuracy = result.daily_accuracy.mean()
            print_metrics(metrics, f"SP{sp_id} {name}")

        scenario_results[name] = sp_results

        # 全期間統合
        all_returns = pd.concat([r.daily_returns for r in sp_results]).sort_index()
        overall = calculate_metrics(all_returns, periods_per_year=PERIODS_PER_YEAR_4H)
        all_acc = pd.concat([r.daily_accuracy for r in sp_results])
        if len(all_acc) > 0:
            overall.accuracy = all_acc.mean()
        print_metrics(overall, f"全期間 {name}")

    # --- 比較サマリー ---
    print("\n" + "=" * 60)
    print(f"  シナリオ比較サマリー [{ticker_mode}銘柄]")
    print("=" * 60)
    print(f"\n  {'シナリオ':<25s} {'Sharpe':>8s} {'精度':>8s} {'総リターン':>10s} {'最大DD':>8s}")
    print(f"  {'─'*60}")
    for name, results in scenario_results.items():
        all_returns = pd.concat([r.daily_returns for r in results]).sort_index()
        m = calculate_metrics(all_returns, periods_per_year=PERIODS_PER_YEAR_4H)
        all_acc = pd.concat([r.daily_accuracy for r in results])
        acc = all_acc.mean() if len(all_acc) > 0 else 0
        print(f"  {name:<25s} {m.sharpe_ratio:>8.2f} {acc:>7.1%} {m.total_return:>9.1%} {m.max_drawdown:>7.1%}")

    # --- グラフ ---
    suffix = "_dynamic_tickers" if args.dynamic else ""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"output_4h_dynamic{suffix}")
    plot_comparison(scenario_results, output_dir)

    print("\n完了!")


if __name__ == "__main__":
    main()
