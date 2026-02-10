"""ボラティリティフィルタ + ストップロス 併用シミュレーション.

個別銘柄フィルタ: 直近N本の4hリターン標準偏差が
全銘柄中央値の X倍を超える銘柄をランキングから除外する。

SL=10%との併用効果を SP1-3 全期間で検証。
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
from data.preprocessor import prepare_study_period, prepare_all_study_periods
from models.lstm_model import load_ensemble
from backtest.engine import calculate_metrics

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_4h_dynamic")


def ensemble_predict_ranks(models, coin_samples):
    coin_ids = list(coin_samples.keys())
    if not coin_ids:
        return {}
    X_batch = np.array([coin_samples[cid] for cid in coin_ids])
    all_ranks = []
    for model in models:
        probs = model.predict(X_batch, verbose=0).flatten()
        order = np.argsort(-probs)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        all_ranks.append(ranks)
    avg_ranks = np.mean(all_ranks, axis=0)
    return {cid: rank for cid, rank in zip(coin_ids, avg_ranks)}


def compute_coin_volatilities(returns_df, date, lookback=6*3):
    """直近lookback本(デフォルト=3日=18本)のリターン標準偏差を計算."""
    idx = returns_df.index.get_loc(date)
    if idx < lookback:
        return {}
    window = returns_df.iloc[idx - lookback:idx]
    vols = window.std()
    return vols.dropna().to_dict()


def run_backtest_with_volfilter_and_sl(
    daily_ranks, returns_df, price_df,
    k=PORTFOLIO_K, hold_threshold=10,
    stop_loss_pct=None,
    vol_multiplier=None,
    vol_lookback=18,
):
    """ボラティリティフィルタ + ストップロス付きバックテスト.

    Args:
        vol_multiplier: 中央値の何倍でフィルタ (None=フィルタなし)
        vol_lookback: ボラ計算の期間 (4h足本数)
        stop_loss_pct: SL水準 (None=SLなし)
    """
    dates = sorted(daily_ranks.keys())

    portfolio_returns = []
    portfolio_dates = []
    current_longs = []
    current_shorts = []
    long_cum_returns = {}
    short_cum_returns = {}
    n_filtered_total = 0
    sl_events = 0

    for date in dates:
        ranks = daily_ranks[date]
        n_coins = len(ranks)
        if n_coins < 2 * k:
            continue

        # --- ボラティリティフィルタ ---
        filtered_ranks = dict(ranks)
        if vol_multiplier is not None:
            vols = compute_coin_volatilities(returns_df, date, vol_lookback)
            if vols:
                vol_values = list(vols.values())
                vol_median = np.median(vol_values)
                threshold = vol_median * vol_multiplier
                excluded = set()
                for coin in list(filtered_ranks.keys()):
                    if coin in vols and vols[coin] > threshold:
                        excluded.add(coin)
                        del filtered_ranks[coin]
                n_filtered_total += len(excluded)

                if len(filtered_ranks) < 2 * k:
                    filtered_ranks = dict(ranks)

        sorted_coins = sorted(filtered_ranks.keys(), key=lambda c: filtered_ranks[c])
        ideal_longs = sorted_coins[:k]
        ideal_shorts = sorted_coins[-k:]

        # --- ストップロス判定 ---
        stopped_longs = []
        stopped_shorts = []
        if stop_loss_pct is not None and current_longs:
            prev_idx = dates.index(date) - 1
            if prev_idx >= 0:
                prev_date = dates[prev_idx]
                if prev_date in returns_df.index:
                    for c in current_longs:
                        if c in returns_df.columns:
                            r = returns_df.loc[prev_date, c]
                            if not np.isnan(r):
                                long_cum_returns[c] = (1 + long_cum_returns.get(c, 0)) * (1 + r) - 1
                                if long_cum_returns[c] < -stop_loss_pct:
                                    stopped_longs.append(c)
                                    sl_events += 1
                    for c in current_shorts:
                        if c in returns_df.columns:
                            r = returns_df.loc[prev_date, c]
                            if not np.isnan(r):
                                short_cum_returns[c] = (1 + short_cum_returns.get(c, 0)) * (1 + r) - 1
                                if short_cum_returns[c] > stop_loss_pct:
                                    stopped_shorts.append(c)
                                    sl_events += 1

        # --- 動的リバランス ---
        if not current_longs:
            new_longs = ideal_longs
            new_shorts = ideal_shorts
        else:
            top_set = set(sorted_coins[:hold_threshold])
            bottom_set = set(sorted_coins[-hold_threshold:])

            active_longs = [c for c in current_longs if c not in stopped_longs]
            active_shorts = [c for c in current_shorts if c not in stopped_shorts]

            kept_longs = [c for c in active_longs if c in top_set and c in filtered_ranks]
            needed = k - len(kept_longs)
            excluded_l = set(kept_longs) | set(stopped_longs)
            cands = [c for c in ideal_longs if c not in excluded_l]
            new_longs = kept_longs + cands[:needed]

            kept_shorts = [c for c in active_shorts if c in bottom_set and c in filtered_ranks]
            needed_s = k - len(kept_shorts)
            excluded_s = set(kept_shorts) | set(stopped_shorts)
            cands_s = [c for c in ideal_shorts if c not in excluded_s]
            new_shorts = kept_shorts + cands_s[:needed_s]

        # 累積リターン更新
        new_lcum = {}
        for c in new_longs:
            new_lcum[c] = long_cum_returns.get(c, 0) if (c in current_longs and c not in stopped_longs) else 0
        long_cum_returns = new_lcum

        new_scum = {}
        for c in new_shorts:
            new_scum[c] = short_cum_returns.get(c, 0) if (c in current_shorts and c not in stopped_shorts) else 0
        short_cum_returns = new_scum

        current_longs = new_longs
        current_shorts = new_shorts

        # リターン計算
        long_rets = [returns_df.loc[date, c] for c in new_longs
                     if c in returns_df.columns and not np.isnan(returns_df.loc[date, c])]
        short_rets = [returns_df.loc[date, c] for c in new_shorts
                      if c in returns_df.columns and not np.isnan(returns_df.loc[date, c])]

        if not long_rets or not short_rets:
            continue

        port_return = np.mean(long_rets) - np.mean(short_rets)
        portfolio_returns.append(port_return)
        portfolio_dates.append(date)

        # 当期間の累積更新
        for c in new_longs:
            if c in returns_df.columns:
                r = returns_df.loc[date, c]
                if not np.isnan(r):
                    long_cum_returns[c] = (1 + long_cum_returns.get(c, 0)) * (1 + r) - 1
        for c in new_shorts:
            if c in returns_df.columns:
                r = returns_df.loc[date, c]
                if not np.isnan(r):
                    short_cum_returns[c] = (1 + short_cum_returns.get(c, 0)) * (1 + r) - 1

    daily_ret = pd.Series(portfolio_returns, index=portfolio_dates, name="return")
    avg_filtered = n_filtered_total / len(dates) if dates else 0
    return daily_ret, sl_events, avg_filtered


def main():
    print("=" * 70)
    print("  ボラティリティフィルタ + ストップロス 併用シミュレーション")
    print("=" * 70)

    # データ読込
    print("\n[1] データ読込...")
    price_df = collect_4h_data()

    print("\n[2] 全SP前処理...")
    sp_data_list = prepare_all_study_periods(price_df)

    print("\n[3] 各SPの予測...")
    model_path = os.path.join(DATA_DIR, "models", "4h")
    all_period_ranks = []

    for i, (sp_data, sp_config) in enumerate(zip(sp_data_list, STUDY_PERIODS)):
        print(f"  SP{sp_config['id']} 予測中...")
        models, config = load_ensemble(model_path)
        sp_ranks = {}
        for date in sp_data.test_dates:
            if date not in sp_data.test_samples_by_day:
                continue
            coin_samples = sp_data.test_samples_by_day[date]
            if len(coin_samples) < 2 * PORTFOLIO_K:
                continue
            ranks = ensemble_predict_ranks(models, coin_samples)
            sp_ranks[date] = ranks
        all_period_ranks.append(sp_ranks)
        import tensorflow as tf
        for m in models:
            del m
        tf.keras.backend.clear_session()
        print(f"    {len(sp_ranks)} 期間完了")

    # --- シミュレーション ---
    print("\n[4] シミュレーション...")

    # テスト条件
    configs = [
        {"label": "ベースライン", "sl": None, "vol": None},
        {"label": "SL=10%のみ", "sl": 0.10, "vol": None},
        {"label": "Vol×2.0のみ", "sl": None, "vol": 2.0},
        {"label": "Vol×2.5のみ", "sl": None, "vol": 2.5},
        {"label": "Vol×3.0のみ", "sl": None, "vol": 3.0},
        {"label": "SL10%+Vol×2.0", "sl": 0.10, "vol": 2.0},
        {"label": "SL10%+Vol×2.5", "sl": 0.10, "vol": 2.5},
        {"label": "SL10%+Vol×3.0", "sl": 0.10, "vol": 3.0},
    ]

    # SP別結果格納
    results = {c["label"]: {"sp_results": []} for c in configs}

    for cfg in configs:
        for i, (sp_data, sp_config) in enumerate(zip(sp_data_list, STUDY_PERIODS)):
            ret_s, sl_ev, avg_filt = run_backtest_with_volfilter_and_sl(
                all_period_ranks[i], sp_data.test_returns, price_df,
                stop_loss_pct=cfg["sl"], vol_multiplier=cfg["vol"],
            )
            m = calculate_metrics(ret_s, periods_per_year=PERIODS_PER_YEAR_4H)
            results[cfg["label"]]["sp_results"].append({
                "sp_id": sp_config["id"],
                "returns": ret_s, "metrics": m,
                "sl_events": sl_ev, "avg_filtered": avg_filt,
            })

    # --- 結果表示 ---
    print(f"\n{'='*90}")
    print(f"  {'設定':<18s} {'SP':>3s} {'Sharpe':>8s} {'総リターン':>10s} {'最大DD':>8s} "
          f"{'SLイベント':>10s} {'平均除外数':>10s}")
    print(f"  {'─'*85}")

    for cfg in configs:
        label = cfg["label"]
        for sp_res in results[label]["sp_results"]:
            m = sp_res["metrics"]
            print(f"  {label:<18s} SP{sp_res['sp_id']:>1d} {m.sharpe_ratio:>8.2f} "
                  f"{m.total_return:>9.1%} {m.max_drawdown:>7.1%} "
                  f"{sp_res['sl_events']:>10d} {sp_res['avg_filtered']:>9.1f}")

        # 全SP結合
        all_rets = pd.concat([r["returns"] for r in results[label]["sp_results"]]).sort_index()
        overall = calculate_metrics(all_rets, periods_per_year=PERIODS_PER_YEAR_4H)
        total_sl = sum(r["sl_events"] for r in results[label]["sp_results"])
        avg_f = np.mean([r["avg_filtered"] for r in results[label]["sp_results"]])
        print(f"  {label:<18s} {'全体':>3s} {overall.sharpe_ratio:>8.2f} "
              f"{overall.total_return:>9.1%} {overall.max_drawdown:>7.1%} "
              f"{total_sl:>10d} {avg_f:>9.1f}")
        results[label]["overall"] = overall
        results[label]["overall_returns"] = all_rets
        print()

    # --- グラフ ---
    print("\n[5] グラフ出力...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # SP3 NAV比較
    ax = axes[0]
    plot_configs = [
        ("ベースライン", "gray", "--"),
        ("SL=10%のみ", "blue", "-"),
        ("SL10%+Vol×2.0", "red", "-"),
        ("SL10%+Vol×2.5", "green", "-"),
        ("SL10%+Vol×3.0", "orange", "-"),
    ]
    for label, color, ls in plot_configs:
        sp3_ret = results[label]["sp_results"][2]["returns"]
        nav = (1 + sp3_ret).cumprod()
        ax.plot(nav.index, nav.values, color=color, linestyle=ls, linewidth=1.2, label=label)
    ax.set_ylabel("NAV")
    ax.set_title("SP3: Stop-Loss + Volatility Filter Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # 全期間NAV比較
    ax = axes[1]
    for label, color, ls in plot_configs:
        all_ret = results[label]["overall_returns"]
        nav = (1 + all_ret).cumprod()
        ax.plot(nav.index, nav.values, color=color, linestyle=ls, linewidth=1.2, label=label)
    ax.set_ylabel("NAV")
    ax.set_title("SP1-3 Combined: Stop-Loss + Volatility Filter Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "volfilter_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  グラフ保存: {fig_path}")

    print("\n" + "=" * 70)
    print("  分析完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
