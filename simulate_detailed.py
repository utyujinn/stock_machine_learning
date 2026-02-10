"""詳細取引ログ付きシミュレーション.

4h足ごとに以下を出力:
- 保有ポジション (LONG/SHORT)
- 入替銘柄 (NEW/KEEP/CLOSE)
- 各銘柄の実際のリターン
- ポートフォリオリターン & 累計NAV
- 予測ランク・確率

直近N日間のみ or 全期間を選択可能。
"""

import argparse
import csv
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PORTFOLIO_K, STUDY_PERIODS, PERIODS_PER_YEAR_4H, DATA_DIR
from data.collector_4h import collect_4h_data
from data.preprocessor import prepare_study_period
from models.lstm_model import load_ensemble

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_4h_dynamic")


def ensemble_predict_full(models, coin_samples):
    """ランクと確率の両方を返す."""
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
    return (
        {cid: rank for cid, rank in zip(coin_ids, avg_ranks)},
        {cid: prob for cid, prob in zip(coin_ids, avg_probs)},
    )


def simulate(price_df, sp_data, models, last_n_days=None):
    """詳細ログ付きバックテスト.

    Args:
        last_n_days: 直近N日分のみ表示 (None=全期間)

    Returns:
        list[dict] - 全期間のログレコード
    """
    k = PORTFOLIO_K
    hold_threshold = 10

    # 予測
    period_ranks = {}
    period_probs = {}
    for date in sp_data.test_dates:
        if date not in sp_data.test_samples_by_day:
            continue
        coin_samples = sp_data.test_samples_by_day[date]
        if len(coin_samples) < 2 * k:
            continue
        ranks, probs = ensemble_predict_full(models, coin_samples)
        period_ranks[date] = ranks
        period_probs[date] = probs

    dates = sorted(period_ranks.keys())
    if last_n_days:
        periods_to_show = last_n_days * 6
        dates_to_show = set(dates[-periods_to_show:])
    else:
        dates_to_show = set(dates)

    current_longs = []
    current_shorts = []
    nav = 1.0
    records = []

    for date in dates:
        ranks = period_ranks[date]
        probs = period_probs[date]
        n_coins = len(ranks)

        sorted_coins = sorted(ranks.keys(), key=lambda c: ranks[c])
        ideal_longs = sorted_coins[:k]
        ideal_shorts = sorted_coins[-k:]

        if not current_longs:
            new_longs = ideal_longs
            new_shorts = ideal_shorts
            changes_long = {c: "NEW" for c in new_longs}
            changes_short = {c: "NEW" for c in new_shorts}
        else:
            top_set = set(sorted_coins[:hold_threshold])
            bottom_set = set(sorted_coins[-hold_threshold:])

            kept_longs = [c for c in current_longs if c in top_set and c in ranks]
            needed_long = k - len(kept_longs)
            candidates_long = [c for c in ideal_longs if c not in kept_longs]
            new_longs = kept_longs + candidates_long[:needed_long]

            kept_shorts = [c for c in current_shorts if c in bottom_set and c in ranks]
            needed_short = k - len(kept_shorts)
            candidates_short = [c for c in ideal_shorts if c not in kept_shorts]
            new_shorts = kept_shorts + candidates_short[:needed_short]

            changes_long = {}
            for c in new_longs:
                changes_long[c] = "KEEP" if c in current_longs else "NEW"
            changes_short = {}
            for c in new_shorts:
                changes_short[c] = "KEEP" if c in current_shorts else "NEW"

        # 各銘柄のリターン取得
        ret_row = sp_data.test_returns.loc[date] if date in sp_data.test_returns.index else pd.Series()

        long_rets = {}
        for c in new_longs:
            if c in ret_row.index and not np.isnan(ret_row[c]):
                long_rets[c] = ret_row[c]
            else:
                long_rets[c] = 0.0

        short_rets = {}
        for c in new_shorts:
            if c in ret_row.index and not np.isnan(ret_row[c]):
                short_rets[c] = ret_row[c]
            else:
                short_rets[c] = 0.0

        # ポートフォリオリターン
        avg_long = np.mean(list(long_rets.values())) if long_rets else 0.0
        avg_short = np.mean(list(short_rets.values())) if short_rets else 0.0
        port_ret = avg_long - avg_short
        nav *= (1 + port_ret)

        # クローズされた銘柄
        closed_longs = [c for c in current_longs if c not in new_longs] if current_longs else []
        closed_shorts = [c for c in current_shorts if c not in new_shorts] if current_shorts else []

        # レコード作成 (表示対象期間のみ)
        if date in dates_to_show:
            rec = {
                "datetime": date,
                "nav": nav,
                "port_return": port_ret,
                "avg_long_ret": avg_long,
                "avg_short_ret": avg_short,
                "avg_prob": np.mean(list(probs.values())),
                "n_coins": n_coins,
                "n_swaps": sum(1 for v in changes_long.values() if v == "NEW")
                         + sum(1 for v in changes_short.values() if v == "NEW"),
            }

            # LONG詳細
            for i, c in enumerate(new_longs):
                base = c.replace("-USD", "")
                rec[f"L{i+1}"] = base
                rec[f"L{i+1}_rank"] = ranks[c]
                rec[f"L{i+1}_prob"] = probs[c]
                rec[f"L{i+1}_ret"] = long_rets[c]
                rec[f"L{i+1}_action"] = changes_long[c]

            # SHORT詳細
            for i, c in enumerate(new_shorts):
                base = c.replace("-USD", "")
                rec[f"S{i+1}"] = base
                rec[f"S{i+1}_rank"] = ranks[c]
                rec[f"S{i+1}_prob"] = probs[c]
                rec[f"S{i+1}_ret"] = short_rets[c]
                rec[f"S{i+1}_action"] = changes_short[c]

            # クローズ
            rec["closed_longs"] = ",".join(c.replace("-USD", "") for c in closed_longs)
            rec["closed_shorts"] = ",".join(c.replace("-USD", "") for c in closed_shorts)

            records.append(rec)

        current_longs = new_longs
        current_shorts = new_shorts

    return records


def print_records(records, verbose=True):
    """レコードを人間が読める形式で出力."""
    k = PORTFOLIO_K

    print(f"\n{'='*100}")
    print(f"  期間: {records[0]['datetime']} ~ {records[-1]['datetime']}")
    print(f"  {len(records)} 期間 (4hごと)")
    print(f"{'='*100}")

    for rec in records:
        dt = rec["datetime"]
        ret = rec["port_return"]
        nav = rec["nav"]
        n_swaps = rec["n_swaps"]

        # ヘッダー
        if verbose or n_swaps > 0 or abs(ret) > 0.01:
            print(f"\n{'─'*100}")
            print(f"  {dt}  |  NAV: {nav:.4f}  |  リターン: {ret:+.4f} ({ret:+.2%})  |  "
                  f"入替: {n_swaps}銘柄  |  平均確率: {rec['avg_prob']:.4f}")

            # LONG
            long_parts = []
            for i in range(1, k + 1):
                name = rec.get(f"L{i}", "?")
                action = rec.get(f"L{i}_action", "?")
                r = rec.get(f"L{i}_ret", 0)
                rank = rec.get(f"L{i}_rank", 0)
                prob = rec.get(f"L{i}_prob", 0)
                marker = "*" if action == "NEW" else " "
                long_parts.append(f"{marker}{name:<6s} rank={rank:>5.1f} prob={prob:.3f} ret={r:+.3f}")

            # SHORT
            short_parts = []
            for i in range(1, k + 1):
                name = rec.get(f"S{i}", "?")
                action = rec.get(f"S{i}_action", "?")
                r = rec.get(f"S{i}_ret", 0)
                rank = rec.get(f"S{i}_rank", 0)
                prob = rec.get(f"S{i}_prob", 0)
                marker = "*" if action == "NEW" else " "
                short_parts.append(f"{marker}{name:<6s} rank={rank:>5.1f} prob={prob:.3f} ret={r:+.3f}")

            print(f"    LONG:  {' | '.join(long_parts)}")
            print(f"    SHORT: {' | '.join(short_parts)}")

            if rec["closed_longs"] or rec["closed_shorts"]:
                closed = []
                if rec["closed_longs"]:
                    closed.append(f"L決済: {rec['closed_longs']}")
                if rec["closed_shorts"]:
                    closed.append(f"S決済: {rec['closed_shorts']}")
                print(f"    決済: {' | '.join(closed)}")

    # サマリー
    total_ret = records[-1]["nav"] - 1
    swaps = [r["n_swaps"] for r in records]
    rets = [r["port_return"] for r in records]
    win_rate = np.mean([r > 0 for r in rets])
    print(f"\n{'='*100}")
    print(f"  サマリー")
    print(f"{'='*100}")
    print(f"  総リターン: {total_ret:+.1%}")
    print(f"  最終NAV: {records[-1]['nav']:.4f}")
    print(f"  勝率: {win_rate:.1%} ({sum(r > 0 for r in rets)}/{len(rets)})")
    print(f"  平均入替数: {np.mean(swaps):.1f}/10 銘柄/期間")
    print(f"  最大利益: {max(rets):+.2%}")
    print(f"  最大損失: {min(rets):+.2%}")
    print(f"  平均リターン: {np.mean(rets):+.5f}")


def save_csv(records, output_path):
    """レコードをCSVに保存."""
    if not records:
        return

    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  CSV保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="詳細取引ログ付きシミュレーション")
    parser.add_argument("--days", type=int, default=None,
                       help="直近N日分のみ表示 (省略=全期間)")
    parser.add_argument("--quiet", action="store_true",
                       help="入替がない期間の詳細を省略")
    parser.add_argument("--csv", action="store_true",
                       help="CSV出力")
    args = parser.parse_args()

    print("=" * 60)
    print("  詳細シミュレーション (4h動的リバランス)")
    if args.days:
        print(f"  表示: 直近 {args.days} 日間")
    else:
        print("  表示: SP3 全期間")
    print("=" * 60)

    # データ
    print("\n[1] データ読込...")
    price_df = collect_4h_data()
    print(f"  {price_df.shape[1]} 銘柄 × {price_df.shape[0]} 本")

    print("\n[2] SP3前処理...")
    sp3_config = STUDY_PERIODS[2]
    sp3_data = prepare_study_period(price_df, sp3_config)

    print("\n[3] モデル読込...")
    model_path = os.path.join(DATA_DIR, "models", "4h")
    models, config = load_ensemble(model_path)

    print("\n[4] シミュレーション実行...")
    records = simulate(price_df, sp3_data, models, last_n_days=args.days)

    # メモリ解放
    import tensorflow as tf
    for m in models:
        del m
    tf.keras.backend.clear_session()

    # 出力
    print_records(records, verbose=not args.quiet)

    if args.csv:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, "simulation_detail.csv")
        save_csv(records, csv_path)

    print("\n完了!")


if __name__ == "__main__":
    main()
