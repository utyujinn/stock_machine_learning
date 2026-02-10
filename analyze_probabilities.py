"""予測確率分析スクリプト.

LSTMアンサンブルの生の予測確率（sigmoid出力）を抽出し、
ドローダウン期間中のZEC・DASHの確率を可視化する。

また、全銘柄の平均確率に基づいてショート比率を動的に調整する
戦略改善案をシミュレーションする。
"""

import os
import sys

sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PORTFOLIO_K, STUDY_PERIODS, PERIODS_PER_YEAR_4H, DATA_DIR
from data.collector_4h import collect_4h_data
from data.preprocessor import prepare_study_period
from models.lstm_model import load_ensemble
from backtest.engine import run_backtest_dynamic, calculate_metrics

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_4h_dynamic")


def ensemble_predict_with_probs(
    models, coin_samples: dict[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, float]]:
    """ランクと生の予測確率の両方を返す.

    Returns:
        (ranks_dict, probs_dict)
        - ranks_dict: {coin_id: average_rank}
        - probs_dict: {coin_id: average_probability}
    """
    coin_ids = list(coin_samples.keys())
    if not coin_ids:
        return {}, {}

    X_batch = np.array([coin_samples[cid] for cid in coin_ids])

    all_ranks = []
    all_probs = []
    for model in models:
        probs = model.predict(X_batch, verbose=0).flatten()
        all_probs.append(probs)
        order = np.argsort(-probs)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        all_ranks.append(ranks)

    avg_ranks = np.mean(all_ranks, axis=0)
    avg_probs = np.mean(all_probs, axis=0)

    ranks_dict = {cid: rank for cid, rank in zip(coin_ids, avg_ranks)}
    probs_dict = {cid: prob for cid, prob in zip(coin_ids, avg_probs)}
    return ranks_dict, probs_dict


def run_backtest_dynamic_with_prob_filter(
    period_ranks, period_probs, test_returns,
    cost_bps=0, hold_threshold=10,
    prob_threshold=0.6, min_shorts=2,
):
    """確率ベースのショート削減戦略でバックテスト.

    全銘柄の平均確率が prob_threshold を超えた場合、
    ショート数を減らす（最低 min_shorts まで）。
    ロングはk固定、ショートは動的。
    """
    k = PORTFOLIO_K
    dates = sorted(period_ranks.keys())
    returns_list = []
    prev_longs = None
    prev_shorts = None
    long_positions = {}
    short_positions = {}

    for date in dates:
        if date not in test_returns.index:
            continue

        ranks = period_ranks[date]
        probs = period_probs[date]

        # 全銘柄の平均確率
        avg_prob = np.mean(list(probs.values()))

        # ショート数を動的に決定
        if avg_prob > prob_threshold:
            # 強気相場 → ショート削減
            # avg_prob が 0.6→k, 0.8→min_shorts に線形補間
            ratio = min(1.0, (avg_prob - prob_threshold) / (0.8 - prob_threshold))
            n_shorts = max(min_shorts, int(k * (1 - ratio)))
        else:
            n_shorts = k

        sorted_coins = sorted(ranks.keys(), key=lambda c: ranks[c])
        new_longs = sorted_coins[:k]
        new_shorts = sorted_coins[-n_shorts:]

        # 動的リバランス (hold_threshold)
        if prev_longs is not None and hold_threshold > 0:
            # ロング: 既存保持ルール
            keep_longs = [c for c in prev_longs if c in sorted_coins[:k + hold_threshold]]
            add_longs = [c for c in new_longs if c not in keep_longs]
            need = k - len(keep_longs)
            final_longs = keep_longs + add_longs[:need]

            # ショート: 既存保持ルール
            n_short_threshold = len(sorted_coins) - n_shorts - hold_threshold
            keep_shorts = [c for c in prev_shorts
                          if c in sorted_coins[n_short_threshold:]]
            add_shorts = [c for c in new_shorts if c not in keep_shorts]
            need_s = n_shorts - len(keep_shorts)
            final_shorts = keep_shorts + add_shorts[:need_s]
        else:
            final_longs = new_longs[:k]
            final_shorts = new_shorts[-n_shorts:]

        long_positions[date] = final_longs
        short_positions[date] = final_shorts

        # リターン計算
        day_returns = test_returns.loc[date]
        long_ret = day_returns[[c for c in final_longs if c in day_returns.index]].mean()
        if len(final_shorts) > 0:
            short_ret = -day_returns[[c for c in final_shorts if c in day_returns.index]].mean()
        else:
            short_ret = 0.0

        # ロングとショートの加重平均 (ショート数が少ない場合、ロング比率が上がる)
        total_positions = k + len(final_shorts)
        if total_positions > 0:
            portfolio_ret = (k * long_ret + len(final_shorts) * short_ret) / total_positions
        else:
            portfolio_ret = 0.0

        # コスト計算 (簡略化)
        if prev_longs is not None:
            turnover_l = len(set(final_longs) - set(prev_longs))
            turnover_s = len(set(final_shorts) - set(prev_shorts))
            cost = (turnover_l + turnover_s) * cost_bps / 10000 / total_positions
            portfolio_ret -= cost

        returns_list.append((date, portfolio_ret, len(final_shorts)))

        prev_longs = final_longs
        prev_shorts = final_shorts

    if not returns_list:
        return None, None

    result_df = pd.DataFrame(returns_list, columns=["date", "return", "n_shorts"])
    result_df = result_df.set_index("date")
    return result_df["return"], result_df["n_shorts"]


def main():
    print("=" * 60)
    print("  予測確率分析 & 確率ベース戦略シミュレーション")
    print("  SP3 Test: 2025-09-13 ~ 2026-02-09")
    print("=" * 60)

    # --- 1. データ読込 ---
    print("\n[1] 4hデータ読込...")
    price_df = collect_4h_data()
    print(f"  {price_df.shape[1]} 銘柄 × {price_df.shape[0]} 本")

    # --- 2. SP3前処理 ---
    print("\n[2] SP3データ前処理...")
    sp3_config = STUDY_PERIODS[2]
    sp3_data = prepare_study_period(price_df, sp3_config)

    # --- 3. モデル読込 ---
    print("\n[3] モデル読込...")
    model_path = os.path.join(DATA_DIR, "models", "4h")
    models, config = load_ensemble(model_path)
    print(f"  SP{config.get('sp_id', 3)} (units={config['best_units']}, ensemble×{len(models)})")

    # --- 4. 予測確率抽出 ---
    print("\n[4] 全期間の予測確率を抽出...")
    period_ranks = {}
    period_probs = {}

    for date in sp3_data.test_dates:
        if date not in sp3_data.test_samples_by_day:
            continue
        coin_samples = sp3_data.test_samples_by_day[date]
        if len(coin_samples) < 2 * PORTFOLIO_K:
            continue
        ranks, probs = ensemble_predict_with_probs(models, coin_samples)
        period_ranks[date] = ranks
        period_probs[date] = probs

    print(f"  {len(period_ranks)} 期間の予測完了")

    # メモリ解放
    import tensorflow as tf
    for m in models:
        del m
    tf.keras.backend.clear_session()

    # --- 5. 確率分析 ---
    print("\n[5] 予測確率の分析...")

    # 全期間の平均確率時系列
    dates = sorted(period_probs.keys())
    avg_probs_ts = pd.Series(
        {d: np.mean(list(period_probs[d].values())) for d in dates}
    )
    median_probs_ts = pd.Series(
        {d: np.median(list(period_probs[d].values())) for d in dates}
    )
    max_probs_ts = pd.Series(
        {d: np.max(list(period_probs[d].values())) for d in dates}
    )
    min_probs_ts = pd.Series(
        {d: np.min(list(period_probs[d].values())) for d in dates}
    )

    print(f"\n  全期間の平均確率統計:")
    print(f"    平均: {avg_probs_ts.mean():.4f}")
    print(f"    標準偏差: {avg_probs_ts.std():.4f}")
    print(f"    最大: {avg_probs_ts.max():.4f}")
    print(f"    最小: {avg_probs_ts.min():.4f}")

    # --- 6. ZEC・DASHの確率 ---
    print("\n[6] ZEC・DASHの予測確率 (ドローダウン期間)...")

    # ドローダウン期間特定 (analyze_drawdown.pyの結果から)
    result = run_backtest_dynamic(
        period_ranks, sp3_data.test_returns,
        cost_bps=0, hold_threshold=10,
    )
    nav = (1 + result.daily_returns).cumprod()
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    dd_end = drawdown.idxmin()
    dd_start = nav.loc[:dd_end].idxmax()
    print(f"  ドローダウン期間: {dd_start} ~ {dd_end}")

    # ZEC, DASHの確率推移
    target_coins = ["ZEC-USD", "DASH-USD"]
    for coin in target_coins:
        print(f"\n  === {coin} ===")
        coin_probs = []
        coin_ranks = []
        for d in dates:
            if coin in period_probs[d]:
                coin_probs.append((d, period_probs[d][coin]))
                coin_ranks.append((d, period_ranks[d][coin]))

        if not coin_probs:
            print(f"    データなし")
            continue

        prob_series = pd.Series({d: p for d, p in coin_probs})
        rank_series = pd.Series({d: r for d, r in coin_ranks})
        n_coins = len(period_ranks[dates[0]])

        # ドローダウン期間中の統計
        dd_probs = prob_series.loc[dd_start:dd_end]
        dd_ranks = rank_series.loc[dd_start:dd_end]
        if len(dd_probs) > 0:
            print(f"    ドローダウン期間中:")
            print(f"      平均確率: {dd_probs.mean():.4f}")
            print(f"      最小確率: {dd_probs.min():.4f}")
            print(f"      最大確率: {dd_probs.max():.4f}")
            print(f"      平均ランク: {dd_ranks.mean():.1f} / {n_coins}")
            print(f"      最良ランク: {dd_ranks.min():.1f}")
            print(f"      最悪ランク: {dd_ranks.max():.1f}")

            # ショートに入っていた期間
            in_short = (dd_ranks > n_coins - PORTFOLIO_K).sum()
            print(f"      ショートポジション入り: {in_short}/{len(dd_ranks)} 期間")

        # 全期間
        print(f"    全期間:")
        print(f"      平均確率: {prob_series.mean():.4f}")
        print(f"      平均ランク: {rank_series.mean():.1f} / {n_coins}")

    # --- 7. ドローダウン期間の全銘柄確率分布 ---
    print("\n[7] ドローダウン期間の確率分布...")

    dd_dates = [d for d in dates if dd_start <= d <= dd_end]
    dd_avg = avg_probs_ts.loc[dd_start:dd_end]
    other_avg = avg_probs_ts.loc[~avg_probs_ts.index.isin(dd_dates)]

    print(f"\n  ドローダウン期間中の全銘柄平均確率:")
    print(f"    平均: {dd_avg.mean():.4f}")
    print(f"    標準偏差: {dd_avg.std():.4f}")
    print(f"\n  ドローダウン期間外の全銘柄平均確率:")
    print(f"    平均: {other_avg.mean():.4f}")
    print(f"    標準偏差: {other_avg.std():.4f}")

    # 確率>0.5の銘柄割合
    dd_above50 = []
    for d in dd_dates:
        probs = list(period_probs[d].values())
        dd_above50.append(np.mean([p > 0.5 for p in probs]))
    other_above50 = []
    for d in dates:
        if d not in dd_dates:
            probs = list(period_probs[d].values())
            other_above50.append(np.mean([p > 0.5 for p in probs]))

    print(f"\n  確率 > 0.5 の銘柄割合:")
    print(f"    ドローダウン中: {np.mean(dd_above50):.1%}")
    print(f"    ドローダウン外: {np.mean(other_above50):.1%}")

    # --- 8. 確率ベース戦略シミュレーション ---
    print("\n[8] 確率ベース戦略シミュレーション...")
    print("  ショート数を全銘柄平均確率に応じて動的調整")

    thresholds = [0.55, 0.60, 0.65]
    min_shorts_options = [0, 1, 2, 3]

    print(f"\n  {'閾値':>6s} {'最小Short':>10s} {'Sharpe':>8s} {'総リターン':>10s} {'最大DD':>8s} {'平均Short数':>10s}")
    print(f"  {'─'*60}")

    # ベースライン (通常戦略)
    baseline_result = run_backtest_dynamic(
        period_ranks, sp3_data.test_returns,
        cost_bps=0, hold_threshold=10,
    )
    baseline_metrics = calculate_metrics(baseline_result.daily_returns, periods_per_year=PERIODS_PER_YEAR_4H)
    print(f"  {'基準':>6s} {'k=5':>10s} {baseline_metrics.sharpe_ratio:>8.2f} {baseline_metrics.total_return:>9.1%} {baseline_metrics.max_drawdown:>7.1%} {'5.0':>10s}")

    best_sharpe = baseline_metrics.sharpe_ratio
    best_config = None

    for threshold in thresholds:
        for min_s in min_shorts_options:
            ret_series, n_shorts_series = run_backtest_dynamic_with_prob_filter(
                period_ranks, period_probs, sp3_data.test_returns,
                cost_bps=0, hold_threshold=10,
                prob_threshold=threshold, min_shorts=min_s,
            )
            if ret_series is None:
                continue
            metrics = calculate_metrics(ret_series, periods_per_year=PERIODS_PER_YEAR_4H)
            avg_shorts = n_shorts_series.mean()
            print(f"  {threshold:>6.2f} {min_s:>10d} {metrics.sharpe_ratio:>8.2f} {metrics.total_return:>9.1%} {metrics.max_drawdown:>7.1%} {avg_shorts:>10.1f}")

            if metrics.sharpe_ratio > best_sharpe:
                best_sharpe = metrics.sharpe_ratio
                best_config = (threshold, min_s)

    if best_config:
        print(f"\n  → 最良: 閾値={best_config[0]}, 最小Short={best_config[1]}, Sharpe={best_sharpe:.2f}")
    else:
        print(f"\n  → ベースライン (通常k=5) が最良")

    # --- 9. グラフ出力 ---
    print("\n[9] グラフ出力...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(16, 22), sharex=False)

    # Panel 1: 全銘柄平均確率の推移
    ax = axes[0]
    ax.plot(avg_probs_ts.index, avg_probs_ts.values, 'b-', linewidth=0.8, label='Mean probability')
    ax.fill_between(min_probs_ts.index, min_probs_ts.values, max_probs_ts.values,
                    alpha=0.15, color='blue', label='Min-Max range')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='0.5 threshold')
    ax.axvspan(dd_start, dd_end, alpha=0.2, color='red', label='Max DD period')
    ax.set_ylabel('Prediction Probability')
    ax.set_title('Average Prediction Probability (all coins)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: ZEC・DASHの確率推移
    ax = axes[1]
    colors = {'ZEC-USD': 'green', 'DASH-USD': 'purple'}
    for coin in target_coins:
        coin_data = [(d, period_probs[d].get(coin)) for d in dates if coin in period_probs[d]]
        if coin_data:
            ds, ps = zip(*coin_data)
            ax.plot(ds, ps, linewidth=0.8, color=colors[coin], label=coin)
    ax.plot(avg_probs_ts.index, avg_probs_ts.values, 'b--', linewidth=0.5, alpha=0.5, label='All-coin mean')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax.axvspan(dd_start, dd_end, alpha=0.2, color='red', label='Max DD period')
    ax.set_ylabel('Prediction Probability')
    ax.set_title('ZEC & DASH Prediction Probabilities')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: ZEC・DASHのランク推移
    ax = axes[2]
    for coin in target_coins:
        coin_data = [(d, period_ranks[d].get(coin)) for d in dates if coin in period_ranks[d]]
        if coin_data:
            ds, rs = zip(*coin_data)
            n_coins = len(period_ranks[dates[0]])
            ax.plot(ds, rs, linewidth=0.8, color=colors[coin], label=coin)
    ax.axhline(y=PORTFOLIO_K, color='green', linestyle='--', alpha=0.5, label=f'Long cutoff (rank {PORTFOLIO_K})')
    ax.axhline(y=n_coins - PORTFOLIO_K, color='red', linestyle='--', alpha=0.5, label=f'Short cutoff (rank {n_coins-PORTFOLIO_K})')
    ax.axvspan(dd_start, dd_end, alpha=0.2, color='red', label='Max DD period')
    ax.set_ylabel('Rank (1=best prediction)')
    ax.set_title('ZEC & DASH Prediction Ranks')
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: NAV比較 (通常 vs 確率ベース)
    ax = axes[3]
    baseline_nav = (1 + baseline_result.daily_returns).cumprod()
    ax.plot(baseline_nav.index, baseline_nav.values, 'b-', linewidth=1, label='Baseline (k=5 fixed)')

    if best_config:
        best_ret, best_ns = run_backtest_dynamic_with_prob_filter(
            period_ranks, period_probs, sp3_data.test_returns,
            cost_bps=0, hold_threshold=10,
            prob_threshold=best_config[0], min_shorts=best_config[1],
        )
        best_nav = (1 + best_ret).cumprod()
        ax.plot(best_nav.index, best_nav.values, 'r-', linewidth=1,
                label=f'Prob-based (thr={best_config[0]}, min_s={best_config[1]})')
    ax.axvspan(dd_start, dd_end, alpha=0.2, color='red', label='Max DD period')
    ax.set_ylabel('NAV')
    ax.set_title('SP3 NAV Comparison: Baseline vs Probability-based')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "probability_analysis.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  グラフ保存: {fig_path}")

    print("\n" + "=" * 60)
    print("  分析完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
