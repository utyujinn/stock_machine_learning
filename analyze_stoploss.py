"""ストップロス戦略シミュレーション.

動的リバランス戦略にストップロスを追加した場合の効果を検証する。
- ショートポジション: エントリーからX%逆行（値上がり）で強制決済
- ロングポジション: エントリーからX%逆行（値下がり）で強制決済
- 決済後は次の候補で補充 or 空きのまま

複数のストップロス水準を比較:
  5%, 10%, 15%, 20%, 30%, 50%, なし(ベースライン)
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
from backtest.engine import calculate_metrics

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_4h_dynamic")


def ensemble_predict_ranks(models, coin_samples):
    """ランクを返す（標準版）."""
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


def run_backtest_with_stoploss(
    daily_ranks, returns_df,
    k=PORTFOLIO_K, hold_threshold=10,
    stop_loss_pct=None,
    replace_stopped=True,
):
    """ストップロス付き動的リバランスバックテスト.

    Args:
        stop_loss_pct: ストップロス水準 (0.1 = 10%). None=ストップロスなし.
        replace_stopped: True=決済後に次の候補で補充, False=空きのまま
    """
    dates = sorted(daily_ranks.keys())

    portfolio_returns = []
    portfolio_dates = []
    long_positions = {}
    short_positions = {}
    stoploss_events = []

    current_longs = []
    current_shorts = []

    # 各ポジションの累積リターンを追跡
    # {coin_id: cumulative_return_since_entry}
    long_cum_returns = {}
    short_cum_returns = {}

    for date in dates:
        ranks = daily_ranks[date]
        n_coins = len(ranks)
        if n_coins < 2 * k:
            continue

        sorted_coins = sorted(ranks.keys(), key=lambda c: ranks[c])
        ideal_longs = sorted_coins[:k]
        ideal_shorts = sorted_coins[-k:]

        # --- Step 1: ストップロス判定 (前期間の値動きを反映) ---
        stopped_longs = []
        stopped_shorts = []

        if stop_loss_pct is not None and current_longs:
            # 前期間のリターンで累積を更新
            prev_date_idx = dates.index(date) - 1
            if prev_date_idx >= 0:
                prev_date = dates[prev_date_idx]
                if prev_date in returns_df.index:
                    for c in current_longs:
                        if c in returns_df.columns:
                            r = returns_df.loc[prev_date, c]
                            if not np.isnan(r):
                                long_cum_returns[c] = (1 + long_cum_returns.get(c, 0)) * (1 + r) - 1
                                # ロング: 値下がりでストップ
                                if long_cum_returns[c] < -stop_loss_pct:
                                    stopped_longs.append(c)
                                    stoploss_events.append({
                                        "date": date, "coin": c, "side": "LONG",
                                        "cum_return": long_cum_returns[c],
                                    })

                    for c in current_shorts:
                        if c in returns_df.columns:
                            r = returns_df.loc[prev_date, c]
                            if not np.isnan(r):
                                short_cum_returns[c] = (1 + short_cum_returns.get(c, 0)) * (1 + r) - 1
                                # ショート: 値上がりでストップ (ショートの損 = 原資産の値上がり)
                                if short_cum_returns[c] > stop_loss_pct:
                                    stopped_shorts.append(c)
                                    stoploss_events.append({
                                        "date": date, "coin": c, "side": "SHORT",
                                        "cum_return": short_cum_returns[c],
                                    })

        # --- Step 2: 動的リバランス ---
        if not current_longs:
            new_longs = ideal_longs
            new_shorts = ideal_shorts
        else:
            top_set = set(sorted_coins[:hold_threshold])
            bottom_set = set(sorted_coins[-hold_threshold:])

            # ストップされた銘柄を現ポジションから除外
            active_longs = [c for c in current_longs if c not in stopped_longs]
            active_shorts = [c for c in current_shorts if c not in stopped_shorts]

            # ロング: キープ判定
            kept_longs = [c for c in active_longs if c in top_set and c in ranks]
            needed_long = k - len(kept_longs)
            # ストップ済み・既にキープ済み銘柄を除いて補充
            excluded = set(kept_longs) | set(stopped_longs)
            candidates_long = [c for c in ideal_longs if c not in excluded]

            if replace_stopped:
                new_longs = kept_longs + candidates_long[:needed_long]
            else:
                # 補充しない場合、ストップ分だけ少なくなる
                normal_needed = k - len([c for c in active_longs if c in top_set and c in ranks])
                normal_candidates = [c for c in ideal_longs if c not in kept_longs]
                new_longs = kept_longs + normal_candidates[:max(0, normal_needed - len(stopped_longs))]

            # ショート: キープ判定
            kept_shorts = [c for c in active_shorts if c in bottom_set and c in ranks]
            needed_short = k - len(kept_shorts)
            excluded_s = set(kept_shorts) | set(stopped_shorts)
            candidates_short = [c for c in ideal_shorts if c not in excluded_s]

            if replace_stopped:
                new_shorts = kept_shorts + candidates_short[:needed_short]
            else:
                normal_needed_s = k - len([c for c in active_shorts if c in bottom_set and c in ranks])
                normal_candidates_s = [c for c in ideal_shorts if c not in kept_shorts]
                new_shorts = kept_shorts + normal_candidates_s[:max(0, normal_needed_s - len(stopped_shorts))]

        # 累積リターンの更新: 新規ポジションは0から、継続はそのまま
        new_long_cum = {}
        for c in new_longs:
            if c in current_longs and c not in stopped_longs:
                new_long_cum[c] = long_cum_returns.get(c, 0)
            else:
                new_long_cum[c] = 0  # 新規エントリー
        long_cum_returns = new_long_cum

        new_short_cum = {}
        for c in new_shorts:
            if c in current_shorts and c not in stopped_shorts:
                new_short_cum[c] = short_cum_returns.get(c, 0)
            else:
                new_short_cum[c] = 0
        short_cum_returns = new_short_cum

        current_longs = new_longs
        current_shorts = new_shorts
        long_positions[date] = list(new_longs)
        short_positions[date] = list(new_shorts)

        # --- Step 3: リターン計算 ---
        long_rets = []
        for coin in new_longs:
            if coin in returns_df.columns:
                r = returns_df.loc[date, coin]
                if not np.isnan(r):
                    long_rets.append(r)

        short_rets = []
        for coin in new_shorts:
            if coin in returns_df.columns:
                r = returns_df.loc[date, coin]
                if not np.isnan(r):
                    short_rets.append(r)

        if not long_rets and not short_rets:
            continue

        # ポジション数に応じた加重
        n_long = len(long_rets)
        n_short = len(short_rets)
        total = n_long + n_short

        if total > 0:
            long_contrib = (np.mean(long_rets) * n_long / total) if n_long > 0 else 0
            short_contrib = (-np.mean(short_rets) * n_short / total) if n_short > 0 else 0
            port_return = long_contrib + short_contrib
        else:
            port_return = 0

        portfolio_returns.append(port_return)
        portfolio_dates.append(date)

        # 累積リターン更新 (当期間分)
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
    return daily_ret, stoploss_events, long_positions, short_positions


def main():
    print("=" * 70)
    print("  ストップロス戦略シミュレーション")
    print("  SP3 Test: 2025-09-13 ~ 2026-02-09")
    print("=" * 70)

    # --- データ ---
    print("\n[1] データ読込...")
    price_df = collect_4h_data()

    print("\n[2] SP3前処理...")
    sp3_config = STUDY_PERIODS[2]
    sp3_data = prepare_study_period(price_df, sp3_config)

    print("\n[3] モデル読込・予測...")
    model_path = os.path.join(DATA_DIR, "models", "4h")
    models, config = load_ensemble(model_path)

    period_ranks = {}
    for date in sp3_data.test_dates:
        if date not in sp3_data.test_samples_by_day:
            continue
        coin_samples = sp3_data.test_samples_by_day[date]
        if len(coin_samples) < 2 * PORTFOLIO_K:
            continue
        ranks = ensemble_predict_ranks(models, coin_samples)
        period_ranks[date] = ranks

    print(f"  {len(period_ranks)} 期間の予測完了")

    # メモリ解放
    import tensorflow as tf
    for m in models:
        del m
    tf.keras.backend.clear_session()

    # --- ストップロスシミュレーション ---
    print("\n[4] ストップロス水準比較...")

    stop_levels = [None, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    results = {}

    print(f"\n  {'SL水準':>8s} {'Sharpe':>8s} {'総リターン':>10s} {'最大DD':>8s} "
          f"{'SLイベント':>10s} {'L発動':>6s} {'S発動':>6s}")
    print(f"  {'─'*70}")

    for sl in stop_levels:
        label = "なし" if sl is None else f"{sl:.0%}"
        ret_series, events, longs, shorts = run_backtest_with_stoploss(
            period_ranks, sp3_data.test_returns,
            stop_loss_pct=sl, replace_stopped=True,
        )
        metrics = calculate_metrics(ret_series, periods_per_year=PERIODS_PER_YEAR_4H)
        n_long_sl = sum(1 for e in events if e["side"] == "LONG")
        n_short_sl = sum(1 for e in events if e["side"] == "SHORT")

        results[label] = {
            "returns": ret_series, "events": events, "metrics": metrics,
            "sl_pct": sl, "longs": longs, "shorts": shorts,
        }

        print(f"  {label:>8s} {metrics.sharpe_ratio:>8.2f} {metrics.total_return:>9.1%} "
              f"{metrics.max_drawdown:>7.1%} {len(events):>10d} {n_long_sl:>6d} {n_short_sl:>6d}")

    # --- ストップロスイベントの詳細 ---
    print("\n\n[5] ストップロスイベント詳細 (10%水準)...")
    if "10%" in results:
        events_10 = results["10%"]["events"]
        if events_10:
            # ショートのストップロスのみ (損失が大きいもの順)
            short_events = [e for e in events_10 if e["side"] == "SHORT"]
            short_events.sort(key=lambda e: e["cum_return"], reverse=True)

            print(f"\n  ショートSL発動 (損失大順、上位20):")
            print(f"  {'日時':<22s} {'銘柄':<12s} {'累積リターン':>12s} {'ショート損益':>12s}")
            print(f"  {'─'*60}")
            for e in short_events[:20]:
                coin = e["coin"].replace("-USD", "")
                # ショートの損益 = -累積リターン
                sl_pnl = -e["cum_return"]
                print(f"  {str(e['date']):<22s} {coin:<12s} {e['cum_return']:>+11.1%} {sl_pnl:>+11.1%}")

            # 銘柄別のSL発動回数
            from collections import Counter
            coin_sl_count = Counter(e["coin"].replace("-USD", "") for e in short_events)
            print(f"\n  ショートSL発動回数 (銘柄別):")
            for coin, cnt in coin_sl_count.most_common(15):
                print(f"    {coin:<10s} {cnt:>3d}回")

            # ロング側
            long_events = [e for e in events_10 if e["side"] == "LONG"]
            if long_events:
                long_events.sort(key=lambda e: e["cum_return"])
                print(f"\n  ロングSL発動 (損失大順、上位10):")
                for e in long_events[:10]:
                    coin = e["coin"].replace("-USD", "")
                    print(f"  {str(e['date']):<22s} {coin:<12s} {e['cum_return']:>+11.1%}")

    # --- SP1-3全期間での効果 ---
    print("\n\n[6] SP1-3全期間での効果 (10%ストップロス)...")

    from data.preprocessor import prepare_all_study_periods

    sp_data_list = prepare_all_study_periods(price_df)

    # SP1, SP2の予測 (SP3は既に完了)
    all_period_ranks = []
    for i, (sp_data, sp_config) in enumerate(zip(sp_data_list, STUDY_PERIODS)):
        if i == 2:
            all_period_ranks.append(period_ranks)
            continue

        print(f"  SP{sp_config['id']} の予測中...")
        models_sp, config_sp = load_ensemble(model_path)
        sp_ranks = {}
        for date in sp_data.test_dates:
            if date not in sp_data.test_samples_by_day:
                continue
            coin_samples = sp_data.test_samples_by_day[date]
            if len(coin_samples) < 2 * PORTFOLIO_K:
                continue
            ranks = ensemble_predict_ranks(models_sp, coin_samples)
            sp_ranks[date] = ranks
        all_period_ranks.append(sp_ranks)
        for m in models_sp:
            del m
        tf.keras.backend.clear_session()

    print(f"\n  {'SP':>4s} {'SL水準':>8s} {'Sharpe':>8s} {'総リターン':>10s} {'最大DD':>8s} {'SLイベント':>10s}")
    print(f"  {'─'*55}")

    for sl in [None, 0.10]:
        label = "なし" if sl is None else f"{sl:.0%}"
        all_rets = []
        total_events = 0
        for i, (sp_data, sp_config) in enumerate(zip(sp_data_list, STUDY_PERIODS)):
            sp_id = sp_config["id"]
            ret_s, events_s, _, _ = run_backtest_with_stoploss(
                all_period_ranks[i], sp_data.test_returns,
                stop_loss_pct=sl, replace_stopped=True,
            )
            m = calculate_metrics(ret_s, periods_per_year=PERIODS_PER_YEAR_4H)
            print(f"  SP{sp_id:>2d} {label:>8s} {m.sharpe_ratio:>8.2f} {m.total_return:>9.1%} "
                  f"{m.max_drawdown:>7.1%} {len(events_s):>10d}")
            all_rets.append(ret_s)
            total_events += len(events_s)

        combined = pd.concat(all_rets).sort_index()
        overall = calculate_metrics(combined, periods_per_year=PERIODS_PER_YEAR_4H)
        print(f"  {'全体':>4s} {label:>8s} {overall.sharpe_ratio:>8.2f} {overall.total_return:>9.1%} "
              f"{overall.max_drawdown:>7.1%} {total_events:>10d}")
        print()

    # --- グラフ ---
    print("\n[7] グラフ出力...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Panel 1: NAV比較
    ax = axes[0]
    colors = {"なし": "blue", "5%": "red", "10%": "green", "15%": "orange",
              "20%": "purple", "30%": "brown", "50%": "gray"}
    for label, data in results.items():
        nav = (1 + data["returns"]).cumprod()
        ax.plot(nav.index, nav.values, linewidth=1.2, label=f"SL={label}",
                color=colors.get(label, "black"))
    ax.set_ylabel("NAV")
    ax.set_title("SP3: Stop-Loss Level Comparison (0bps, Dynamic Rebalance)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Panel 2: ドローダウン比較 (ベースラインvs10%)
    ax = axes[1]
    for label in ["なし", "10%"]:
        if label in results:
            nav = (1 + results[label]["returns"]).cumprod()
            dd = (nav - nav.cummax()) / nav.cummax()
            ax.fill_between(dd.index, dd.values, 0, alpha=0.3,
                           label=f"SL={label}", color=colors[label])
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown: Baseline vs 10% Stop-Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "stoploss_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  グラフ保存: {fig_path}")

    print("\n" + "=" * 70)
    print("  分析完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
