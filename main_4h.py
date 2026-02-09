"""4時間足パイプライン: データ収集 → 前処理 → 訓練 → バックテスト → 評価.

Jaquart et al. (2022) のLSTM戦略を4時間足に適用して有効性を検証する。
日足版 (main.py) との比較が目的。
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
)


def run_study_period(sp_data, sp_config):
    """1つのStudy Periodの完全パイプラインを実行."""
    sp_id = sp_data.sp_id
    print(f"\n{'#'*60}")
    print(f"  Study Period {sp_id}")
    print(f"  Test: {sp_config['test'][0]} ~ {sp_config['test'][1]}")
    print(f"{'#'*60}")

    # --- ハイパーパラメータ探索 ---
    print(f"\n  [1/3] ハイパーパラメータ探索...")
    best_units = select_best_units(
        sp_data.X_train, sp_data.y_train,
        sp_data.X_val, sp_data.y_val,
    )

    # --- アンサンブル訓練 ---
    print(f"\n  [2/3] アンサンブル訓練 (ユニット数={best_units})...")
    models = train_ensemble(
        best_units,
        sp_data.X_train, sp_data.y_train,
        sp_data.X_val, sp_data.y_val,
    )

    # --- テスト期間の予測 ---
    print(f"\n  [3/3] テスト期間の予測・バックテスト...")
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

    # --- バックテスト ---
    result = run_backtest(period_ranks, sp_data.test_returns)
    result.sp_id = sp_id

    # メモリ解放
    import tensorflow as tf
    for m in models:
        del m
    tf.keras.backend.clear_session()

    return result


def plot_performance(all_results: list[BacktestResult], output_dir: str):
    """全Study Period統合のパフォーマンスグラフを出力."""
    os.makedirs(output_dir, exist_ok=True)

    all_returns = pd.concat([r.daily_returns for r in all_results]).sort_index()
    nav = (1 + all_returns).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.plot(nav.index, nav.values, label="LSTM Long-Short (4h)", linewidth=1.5)
    ax.set_title("Portfolio Performance Index (4h candles, k=5)")
    ax.set_ylabel("Performance Index")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(all_returns.values, bins=100, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("4h Return Distribution")
    ax.set_xlabel("4h Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "performance.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nパフォーマンスグラフ保存: {fig_path}")

    # Study Period別グラフ
    fig, axes = plt.subplots(len(all_results), 1, figsize=(14, 4 * len(all_results)))
    if len(all_results) == 1:
        axes = [axes]

    for i, result in enumerate(all_results):
        ax = axes[i]
        sp_nav = result.nav
        ax.plot(sp_nav.index, sp_nav.values, linewidth=1.5)
        ax.set_title(f"Study Period {result.sp_id} (4h)")
        ax.set_ylabel("Performance Index")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "performance_by_sp.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"SP別グラフ保存: {fig_path}")


def main():
    print("=" * 60)
    print("  暗号資産 LSTM 統計的裁定取引システム (4時間足)")
    print("  Jaquart et al. (2022) 手法の4h適用実験")
    print("=" * 60)

    # --- Step 1: データ収集 ---
    print("\n[Step 1] 4hデータ収集...")
    price_df = collect_4h_data()
    print(f"  価格データ: {price_df.shape[1]} 銘柄 × {price_df.shape[0]} 本 (4h足)")

    # --- Step 2: 前処理 ---
    print("\n[Step 2] データ前処理・Study Period分割...")
    sp_data_list = prepare_all_study_periods(price_df)

    # --- Step 3-5: 各Study Period の訓練・予測・バックテスト ---
    all_results = []
    for sp_data, sp_config in zip(sp_data_list, STUDY_PERIODS):
        result = run_study_period(sp_data, sp_config)
        all_results.append(result)

        # SP別の評価 (4h足用の年率化係数を使用)
        metrics = calculate_metrics(result.daily_returns, periods_per_year=PERIODS_PER_YEAR_4H)
        if len(result.daily_accuracy) > 0:
            metrics.accuracy = result.daily_accuracy.mean()
        print_metrics(metrics, f"LSTM SP{result.sp_id} (4h)")

    # --- Step 6: 全体評価 ---
    print("\n" + "=" * 60)
    print("  全Study Period統合評価 (4時間足)")
    print("=" * 60)

    all_returns = pd.concat([r.daily_returns for r in all_results]).sort_index()
    overall_metrics = calculate_metrics(all_returns, periods_per_year=PERIODS_PER_YEAR_4H)
    all_accuracy = pd.concat([r.daily_accuracy for r in all_results])
    if len(all_accuracy) > 0:
        overall_metrics.accuracy = all_accuracy.mean()

    print_metrics(overall_metrics, "LSTM 全期間 4h (k=5)")

    # 日足との比較
    print("\n--- 日足実験との比較 ---")
    print(f"  シャープレシオ: {overall_metrics.sharpe_ratio:.4f} (日足: 3.5662)")
    print(f"  ソルティノレシオ: {overall_metrics.sortino_ratio:.4f} (日足: 5.1993)")
    print(f"  精度: {overall_metrics.accuracy:.4f} (日足: 0.5773)")
    print(f"  総リターン: {overall_metrics.total_return:.4f} (日足: 130.4361)")

    # --- グラフ出力 ---
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_4h")
    plot_performance(all_results, output_dir)

    print("\n完了!")


if __name__ == "__main__":
    main()
