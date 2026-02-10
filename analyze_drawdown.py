"""SP3ドローダウン分析スクリプト.

SP3 (2025-09-13 ~ 2026-02-09) の大幅ドローダウンの原因を特定する。
- ドローダウン期間の特定
- 保有ポジションの損益分解
- BTC価格・ボラティリティとの相関
- 市場環境の分析
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
from models.lstm_model import ensemble_predict_ranks, load_ensemble
from backtest.engine import run_backtest_dynamic, calculate_metrics

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_4h_dynamic")


def analyze_drawdown():
    print("=" * 60)
    print("  SP3 ドローダウン分析")
    print("  Test: 2025-09-13 ~ 2026-02-09")
    print("=" * 60)

    # --- 1. データ読込 ---
    print("\n[1] 4hデータ読込...")
    price_df = collect_4h_data()
    print(f"  {price_df.shape[1]} 銘柄 × {price_df.shape[0]} 本")

    # --- 2. SP3前処理 ---
    print("\n[2] SP3データ前処理...")
    sp3_config = STUDY_PERIODS[2]
    sp3_data = prepare_study_period(price_df, sp3_config)

    # --- 3. モデル読込・予測 ---
    print("\n[3] モデル読込・予測...")
    model_path = os.path.join(DATA_DIR, "models", "4h")
    models, config = load_ensemble(model_path)
    print(f"  SP{config.get('sp_id', 3)} (units={config['best_units']}, ensemble×{len(models)})")

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

    # --- 4. バックテスト ---
    print("\n[4] 0bps動的リバランス バックテスト...")
    result = run_backtest_dynamic(
        period_ranks, sp3_data.test_returns,
        cost_bps=0, hold_threshold=10,
    )
    returns = result.daily_returns
    nav = (1 + returns).cumprod()

    # --- 5. ドローダウン分析 ---
    print("\n[5] ドローダウン分析...")

    # ドローダウン系列
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max

    # 最大ドローダウン区間
    dd_end = drawdown.idxmin()
    dd_start_candidates = nav.loc[:dd_end]
    dd_start = dd_start_candidates.idxmax()

    print(f"\n  最大ドローダウン: {drawdown.min():.1%}")
    print(f"  開始: {dd_start} (NAV: {nav.loc[dd_start]:.4f})")
    print(f"  底: {dd_end} (NAV: {nav.loc[dd_end]:.4f})")

    # ドローダウン中の日次リターン
    dd_returns = returns.loc[dd_start:dd_end]
    print(f"  期間: {len(dd_returns)} 期間 ({(dd_end - dd_start).days} 日)")
    print(f"  期間中平均リターン: {dd_returns.mean():.5f}")
    print(f"  期間中標準偏差: {dd_returns.std():.5f}")

    # ワースト5日
    print(f"\n  ワースト10期間:")
    worst = returns.nsmallest(10)
    for ts, ret in worst.items():
        # その時点のポジションを表示
        longs = result.long_positions.get(ts, [])
        shorts = result.short_positions.get(ts, [])
        print(f"    {ts} → {ret:+.4f} ({ret:.1%})")
        print(f"      L: {', '.join(longs[:5])}")
        print(f"      S: {', '.join(shorts[:5])}")

    # --- 6. BTC価格との相関 ---
    print("\n[6] BTC価格・ボラティリティとの相関...")

    btc_col = None
    for col in price_df.columns:
        if 'BTC' in col.upper():
            btc_col = col
            break

    if btc_col:
        btc_price = price_df[btc_col].loc[returns.index[0]:returns.index[-1]]
        btc_returns = btc_price.pct_change().dropna()

        # 共通インデックスで揃える
        common = returns.index.intersection(btc_returns.index)
        strat_r = returns.loc[common]
        btc_r = btc_returns.loc[common]

        corr = strat_r.corr(btc_r)
        print(f"  BTC-戦略リターン相関: {corr:.4f}")

        # BTC上昇/下降時の戦略パフォーマンス
        btc_up = btc_r > 0
        btc_down = btc_r <= 0
        print(f"  BTC上昇時の戦略平均: {strat_r[btc_up].mean():.5f} ({btc_up.sum()} 期間)")
        print(f"  BTC下落時の戦略平均: {strat_r[btc_down].mean():.5f} ({btc_down.sum()} 期間)")

        # BTC 20期間ボラティリティ
        btc_vol20 = btc_r.rolling(20).std()

        # 高ボラ/低ボラ時の戦略パフォーマンス
        vol_median = btc_vol20.median()
        high_vol = btc_vol20 > vol_median
        common_vol = high_vol.index.intersection(strat_r.index)
        print(f"  高ボラ時の戦略平均: {strat_r.loc[common_vol][high_vol.loc[common_vol]].mean():.5f}")
        print(f"  低ボラ時の戦略平均: {strat_r.loc[common_vol][~high_vol.loc[common_vol]].mean():.5f}")

        # ドローダウン期間中のBTC
        btc_dd = btc_price.loc[dd_start:dd_end]
        if len(btc_dd) > 1:
            btc_dd_return = (btc_dd.iloc[-1] / btc_dd.iloc[0]) - 1
            print(f"\n  ドローダウン期間中のBTC:")
            print(f"    開始: ${btc_dd.iloc[0]:,.0f} → 底: ${btc_dd.iloc[-1]:,.0f}")
            print(f"    BTC変動: {btc_dd_return:+.1%}")

    # --- 7. ポジション分析 ---
    print("\n[7] ドローダウン期間中のポジション頻度...")

    dd_dates = [d for d in result.long_positions.keys()
                if dd_start <= d <= dd_end]

    from collections import Counter
    long_freq = Counter()
    short_freq = Counter()
    for d in dd_dates:
        for c in result.long_positions.get(d, []):
            long_freq[c] += 1
        for c in result.short_positions.get(d, []):
            short_freq[c] += 1

    print(f"\n  ドローダウン中のLONG頻度 Top10:")
    for coin, cnt in long_freq.most_common(10):
        # コインの実際のリターンを確認
        base = coin.replace("-USD", "")
        col_match = [c for c in sp3_data.test_returns.columns if base in c]
        if col_match:
            coin_ret = sp3_data.test_returns[col_match[0]].loc[dd_start:dd_end]
            cum_ret = (1 + coin_ret).prod() - 1
            print(f"    {coin:<15s} {cnt:>3d}回 (実リターン: {cum_ret:+.1%})")
        else:
            print(f"    {coin:<15s} {cnt:>3d}回")

    print(f"\n  ドローダウン中のSHORT頻度 Top10:")
    for coin, cnt in short_freq.most_common(10):
        base = coin.replace("-USD", "")
        col_match = [c for c in sp3_data.test_returns.columns if base in c]
        if col_match:
            coin_ret = sp3_data.test_returns[col_match[0]].loc[dd_start:dd_end]
            cum_ret = (1 + coin_ret).prod() - 1
            # ショートなのでリターンは逆
            print(f"    {coin:<15s} {cnt:>3d}回 (実リターン: {cum_ret:+.1%}, Short損益: {-cum_ret:+.1%})")
        else:
            print(f"    {coin:<15s} {cnt:>3d}回")

    # --- 8. 月次パフォーマンス ---
    print("\n[8] 月次パフォーマンス分解...")
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    for month, ret in monthly.items():
        # その月のドローダウン
        month_nav = nav.loc[nav.index.month == month.month]
        if len(month_nav) > 0:
            month_dd = ((month_nav - month_nav.cummax()) / month_nav.cummax()).min()
        else:
            month_dd = 0
        print(f"    {month.strftime('%Y-%m')}: {ret:+.1%} (月内最大DD: {month_dd:.1%})")

    # --- 9. グラフ出力 ---
    print("\n[9] グラフ出力...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)

    # Panel 1: NAV + ドローダウン区間
    ax = axes[0]
    ax.plot(nav.index, nav.values, 'b-', linewidth=1, label='Strategy NAV')
    ax.axvspan(dd_start, dd_end, alpha=0.2, color='red', label=f'Max DD: {drawdown.min():.1%}')
    ax.set_ylabel('NAV')
    ax.set_title('SP3 Strategy NAV (0bps Dynamic Rebalance)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: ドローダウン
    ax = axes[1]
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color='red')
    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown from Peak')
    ax.grid(True, alpha=0.3)

    # Panel 3: BTC価格
    if btc_col:
        ax = axes[2]
        btc_test = price_df[btc_col].loc[returns.index[0]:returns.index[-1]]
        ax.plot(btc_test.index, btc_test.values, 'orange', linewidth=1)
        ax.axvspan(dd_start, dd_end, alpha=0.2, color='red')
        ax.set_ylabel('BTC Price ($)')
        ax.set_title('BTC Price')
        ax.grid(True, alpha=0.3)

        # Panel 4: BTC 20期間ボラティリティ
        ax = axes[3]
        btc_ret_full = btc_test.pct_change()
        btc_vol = btc_ret_full.rolling(120).std() * np.sqrt(PERIODS_PER_YEAR_4H)  # 年率化
        ax.plot(btc_vol.index, btc_vol.values, 'purple', linewidth=1)
        ax.axvspan(dd_start, dd_end, alpha=0.2, color='red')
        ax.set_ylabel('Annualized Volatility')
        ax.set_title('BTC Realized Volatility (20-day = 120 4h-periods)')
        ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "sp3_drawdown_analysis.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  グラフ保存: {fig_path}")

    # --- 10. 20日ローリングシャープ ---
    print("\n[10] ローリングシャープレシオ (120期間 = 20日)...")
    rolling_mean = returns.rolling(120).mean()
    rolling_std = returns.rolling(120).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(PERIODS_PER_YEAR_4H)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, 'b-', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvspan(dd_start, dd_end, alpha=0.2, color='red', label='Max DD period')
    ax.set_ylabel('Rolling Sharpe (annualized)')
    ax.set_title('20-day Rolling Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    fig_path2 = os.path.join(OUTPUT_DIR, "sp3_rolling_sharpe.png")
    plt.savefig(fig_path2, dpi=150)
    plt.close()
    print(f"  グラフ保存: {fig_path2}")

    print("\n" + "=" * 60)
    print("  分析完了")
    print("=" * 60)


if __name__ == "__main__":
    analyze_drawdown()
