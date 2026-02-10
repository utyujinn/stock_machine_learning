"""4時間足 動的リバランス実験.

3つのコストシナリオを比較:
  A) 0 bps (手数料0) + 動的リバランス
  B) 5 bps (スリッページのみ) + 動的リバランス
  C) 15 bps フルターンオーバー (前回の4h実験、参考)
"""

import os
import sys

sys.stdout.reconfigure(line_buffering=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PORTFOLIO_K, STUDY_PERIODS, PERIODS_PER_YEAR_4H
from data.collector_4h import collect_4h_data
from data.preprocessor import prepare_all_study_periods
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


def main():
    print("=" * 60)
    print("  4時間足 動的リバランス実験")
    print("  シナリオ比較: 0bps / 5bps / 15bps")
    print("=" * 60)

    # --- データ収集 ---
    print("\n[Step 1] 4hデータ収集...")
    price_df = collect_4h_data()
    print(f"  価格データ: {price_df.shape[1]} 銘柄 × {price_df.shape[0]} 本")

    # --- 前処理 ---
    print("\n[Step 2] データ前処理...")
    sp_data_list = prepare_all_study_periods(price_df)

    # --- 訓練・予測 (全シナリオで共通) ---
    print("\n[Step 3] 訓練・予測...")
    all_ranks = []  # SP毎のランク辞書
    for sp_data, sp_config in zip(sp_data_list, STUDY_PERIODS):
        period_ranks = run_study_period(sp_data, sp_config)
        all_ranks.append(period_ranks)

    # --- シナリオ別バックテスト ---
    print("\n" + "=" * 60)
    print("  シナリオ別バックテスト")
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
    print("  シナリオ比較サマリー")
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
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_4h_dynamic")
    plot_comparison(scenario_results, output_dir)

    print("\n完了!")


if __name__ == "__main__":
    main()
