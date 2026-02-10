"""バックテストエンジン・ポートフォリオ管理・評価指標.

論文の方法論 (Section 3.7, 3.8):
- k=5 ロング・ショートポートフォリオ (上位5銘柄ロング、下位5銘柄ショート)
- ドル中立 (均等配分)
- 毎日リバランス
- 取引コスト: 片道15bps
- 評価: シャープレシオ、ソルティノレシオ、最大ドローダウン等
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import PORTFOLIO_K, TRANSACTION_COST_BPS


@dataclass
class BacktestResult:
    """バックテスト結果."""
    sp_id: int
    daily_returns: pd.Series
    nav: pd.Series
    long_positions: dict[pd.Timestamp, list[str]]
    short_positions: dict[pd.Timestamp, list[str]]
    daily_accuracy: pd.Series


@dataclass
class PerformanceMetrics:
    """パフォーマンス評価指標."""
    mean_daily_return: float
    return_std: float
    downside_risk: float
    var_1pct: float
    var_5pct: float
    cvar_1pct: float
    cvar_5pct: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    accuracy: float
    total_return: float


def run_backtest(
    daily_ranks: dict[pd.Timestamp, dict[str, float]],
    returns_df: pd.DataFrame,
    k: int = PORTFOLIO_K,
    cost_bps: int = TRANSACTION_COST_BPS,
) -> BacktestResult:
    """ロング・ショートポートフォリオのバックテストを実行.

    Args:
        daily_ranks: {日付: {coin_id: average_rank}} の辞書
        returns_df: 日次リターン DataFrame (日付 × 銘柄)
        k: ポートフォリオの片側銘柄数
        cost_bps: 片道取引コスト (basis points)
    """
    cost_rate = cost_bps / 10000.0
    dates = sorted(daily_ranks.keys())

    portfolio_returns = []
    portfolio_dates = []
    long_positions = {}
    short_positions = {}
    accuracy_list = []

    prev_longs = set()
    prev_shorts = set()

    for date in dates:
        ranks = daily_ranks[date]
        if len(ranks) < 2 * k:
            continue

        # ランクでソート (ランク値が小さい = 予測が良い = ロング候補)
        sorted_coins = sorted(ranks.keys(), key=lambda c: ranks[c])
        longs = sorted_coins[:k]
        shorts = sorted_coins[-k:]

        long_positions[date] = longs
        short_positions[date] = shorts

        # 当日のリターンを使ってポートフォリオリターンを計算
        # モデルは前日までの特徴量で当日のリターンを予測するため、
        # 予測対象である当日のリターンを使用する
        # ロング側の平均リターン
        long_returns = []
        for coin in longs:
            if coin in returns_df.columns:
                r = returns_df.loc[date, coin]
                if not np.isnan(r):
                    long_returns.append(r)

        # ショート側の平均リターン
        short_returns = []
        for coin in shorts:
            if coin in returns_df.columns:
                r = returns_df.loc[date, coin]
                if not np.isnan(r):
                    short_returns.append(r)

        if not long_returns or not short_returns:
            continue

        # ポートフォリオリターン = ロング平均 - ショート平均
        port_return = np.mean(long_returns) - np.mean(short_returns)

        # 取引コスト: ポジション変更のたびに片道コスト
        # 全ポジションを毎日入れ替えるので、2k銘柄 × 片道コスト × 2 (買い+売り)
        # 論文: "incur the assumed transaction costs of 15 bps of the transaction volume"
        # 各ポジションの構築と解消で片道コストが発生
        turnover_cost = 2 * cost_rate  # 全ポジション解消+構築 = 往復
        port_return -= turnover_cost

        portfolio_returns.append(port_return)
        portfolio_dates.append(date)

        # 精度: ロング銘柄が中央値以上、ショート銘柄が中央値以下かを確認
        all_returns_today = returns_df.loc[date].dropna()
        if len(all_returns_today) > 0:
            median_return = all_returns_today.median()
            correct = 0
            total = 0
            for coin in longs:
                if coin in all_returns_today.index:
                    if all_returns_today[coin] >= median_return:
                        correct += 1
                    total += 1
            for coin in shorts:
                if coin in all_returns_today.index:
                    if all_returns_today[coin] < median_return:
                        correct += 1
                    total += 1
            if total > 0:
                accuracy_list.append(correct / total)

        prev_longs = set(longs)
        prev_shorts = set(shorts)

    # NAV計算
    daily_ret_series = pd.Series(portfolio_returns, index=portfolio_dates, name="return")
    nav = (1 + daily_ret_series).cumprod()
    accuracy_series = pd.Series(accuracy_list, index=portfolio_dates[:len(accuracy_list)])

    return BacktestResult(
        sp_id=0,
        daily_returns=daily_ret_series,
        nav=nav,
        long_positions=long_positions,
        short_positions=short_positions,
        daily_accuracy=accuracy_series,
    )


def run_backtest_dynamic(
    daily_ranks: dict[pd.Timestamp, dict[str, float]],
    returns_df: pd.DataFrame,
    k: int = PORTFOLIO_K,
    cost_bps: int = 0,
    hold_threshold: int = 10,
) -> BacktestResult:
    """動的リバランス付きロング・ショートバックテスト.

    ポジション入替はランクが大幅に変動した時のみ行い、
    ターンオーバーを抑制する。

    Args:
        daily_ranks: {日付: {coin_id: average_rank}} の辞書
        returns_df: リターン DataFrame
        k: ポートフォリオの片側銘柄数
        cost_bps: 片道取引コスト (basis points), 入替分のみ適用
        hold_threshold: 現ポジションをホールドするランク閾値
            ロング銘柄が上位 hold_threshold 位以内なら継続保有
            ショート銘柄が下位 hold_threshold 位以内なら継続保有
    """
    cost_rate = cost_bps / 10000.0
    dates = sorted(daily_ranks.keys())

    portfolio_returns = []
    portfolio_dates = []
    long_positions = {}
    short_positions = {}
    accuracy_list = []
    turnover_counts = []

    current_longs = []
    current_shorts = []

    for date in dates:
        ranks = daily_ranks[date]
        n_coins = len(ranks)
        if n_coins < 2 * k:
            continue

        sorted_coins = sorted(ranks.keys(), key=lambda c: ranks[c])
        ideal_longs = sorted_coins[:k]
        ideal_shorts = sorted_coins[-k:]

        if not current_longs:
            # 初回: そのまま全ポジション構築
            new_longs = ideal_longs
            new_shorts = ideal_shorts
            n_swaps = 2 * k
        else:
            # 動的リバランス: ランクが閾値内なら保持
            top_set = set(sorted_coins[:hold_threshold])
            bottom_set = set(sorted_coins[-hold_threshold:])

            # 現ロングのうち、まだ上位 hold_threshold に残っているものは保持
            kept_longs = [c for c in current_longs if c in top_set and c in ranks]
            # 足りない分を ideal_longs から補充
            needed_long = k - len(kept_longs)
            candidates_long = [c for c in ideal_longs if c not in kept_longs]
            new_longs = kept_longs + candidates_long[:needed_long]

            # 現ショートのうち、まだ下位 hold_threshold に残っているものは保持
            kept_shorts = [c for c in current_shorts if c in bottom_set and c in ranks]
            needed_short = k - len(kept_shorts)
            candidates_short = [c for c in ideal_shorts if c not in kept_shorts]
            new_shorts = kept_shorts + candidates_short[:needed_short]

            # ターンオーバー計算
            swapped_longs = len(set(new_longs) - set(current_longs))
            swapped_shorts = len(set(new_shorts) - set(current_shorts))
            n_swaps = swapped_longs + swapped_shorts

        current_longs = new_longs
        current_shorts = new_shorts
        long_positions[date] = new_longs
        short_positions[date] = new_shorts
        turnover_counts.append(n_swaps)

        # リターン計算
        long_returns = []
        for coin in new_longs:
            if coin in returns_df.columns:
                r = returns_df.loc[date, coin]
                if not np.isnan(r):
                    long_returns.append(r)

        short_returns = []
        for coin in new_shorts:
            if coin in returns_df.columns:
                r = returns_df.loc[date, coin]
                if not np.isnan(r):
                    short_returns.append(r)

        if not long_returns or not short_returns:
            continue

        port_return = np.mean(long_returns) - np.mean(short_returns)

        # 取引コスト: 入れ替えた銘柄分のみ
        # n_swaps 銘柄を入替: 各銘柄の解消(片道) + 構築(片道) = 往復
        # ポートフォリオ全体に対する割合: n_swaps / (2 * k)
        if cost_rate > 0 and n_swaps > 0:
            turnover_fraction = n_swaps / (2 * k)
            turnover_cost = 2 * cost_rate * turnover_fraction
            port_return -= turnover_cost

        portfolio_returns.append(port_return)
        portfolio_dates.append(date)

        # 精度
        all_returns_today = returns_df.loc[date].dropna()
        if len(all_returns_today) > 0:
            median_return = all_returns_today.median()
            correct = 0
            total = 0
            for coin in new_longs:
                if coin in all_returns_today.index:
                    if all_returns_today[coin] >= median_return:
                        correct += 1
                    total += 1
            for coin in new_shorts:
                if coin in all_returns_today.index:
                    if all_returns_today[coin] < median_return:
                        correct += 1
                    total += 1
            if total > 0:
                accuracy_list.append(correct / total)

    daily_ret_series = pd.Series(portfolio_returns, index=portfolio_dates, name="return")
    nav = (1 + daily_ret_series).cumprod()
    accuracy_series = pd.Series(accuracy_list, index=portfolio_dates[:len(accuracy_list)])

    avg_turnover = np.mean(turnover_counts) if turnover_counts else 0
    print(f"    平均ターンオーバー: {avg_turnover:.1f}/{2*k} 銘柄/期間")

    return BacktestResult(
        sp_id=0,
        daily_returns=daily_ret_series,
        nav=nav,
        long_positions=long_positions,
        short_positions=short_positions,
        daily_accuracy=accuracy_series,
    )


def calculate_metrics(daily_returns: pd.Series, periods_per_year: int = 365) -> PerformanceMetrics:
    """パフォーマンス評価指標を計算.

    論文 Table 6 に対応する指標を計算。
    Args:
        daily_returns: 期間リターンのSeries
        periods_per_year: 年率化係数 (日足=365, 4h足=2190)
    """
    r = daily_returns.values
    n = len(r)

    if n == 0:
        return PerformanceMetrics(
            mean_daily_return=0, return_std=0, downside_risk=0,
            var_1pct=0, var_5pct=0, cvar_1pct=0, cvar_5pct=0,
            annualized_volatility=0, sharpe_ratio=0, sortino_ratio=0,
            max_drawdown=0, accuracy=0, total_return=0,
        )

    mean_ret = np.mean(r)
    std_ret = np.std(r, ddof=1)

    # ダウンサイドリスク (負のリターンのみの標準偏差)
    downside = r[r < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0

    # VaR / CVaR
    var_1 = np.percentile(r, 1)
    var_5 = np.percentile(r, 5)
    cvar_1 = np.mean(r[r <= var_1]) if np.any(r <= var_1) else var_1
    cvar_5 = np.mean(r[r <= var_5]) if np.any(r <= var_5) else var_5

    # 年率化
    ann_vol = std_ret * np.sqrt(periods_per_year)
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0
    sortino = (mean_ret / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    # 最大ドローダウン
    cum_returns = (1 + daily_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    # 総リターン
    total_ret = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0.0

    return PerformanceMetrics(
        mean_daily_return=mean_ret,
        return_std=std_ret,
        downside_risk=np.mean(r < 0) if n > 0 else 0,
        var_1pct=var_1,
        var_5pct=var_5,
        cvar_1pct=cvar_1,
        cvar_5pct=cvar_5,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        accuracy=0.0,
        total_return=total_ret,
    )


def print_metrics(metrics: PerformanceMetrics, label: str = "LSTM"):
    """パフォーマンス指標を表形式で表示."""
    print(f"\n{'='*50}")
    print(f"  パフォーマンス評価: {label}")
    print(f"{'='*50}")
    print(f"  日次平均リターン:     {metrics.mean_daily_return:.5f}")
    print(f"  日次標準偏差:         {metrics.return_std:.5f}")
    print(f"  VaR 1%:              {metrics.var_1pct:.5f}")
    print(f"  VaR 5%:              {metrics.var_5pct:.5f}")
    print(f"  CVaR 1%:             {metrics.cvar_1pct:.5f}")
    print(f"  CVaR 5%:             {metrics.cvar_5pct:.5f}")
    print(f"  年率ボラティリティ:   {metrics.annualized_volatility:.5f}")
    print(f"  シャープレシオ:       {metrics.sharpe_ratio:.4f}")
    print(f"  ソルティノレシオ:     {metrics.sortino_ratio:.4f}")
    print(f"  最大ドローダウン:     {metrics.max_drawdown:.4f}")
    print(f"  精度:                 {metrics.accuracy:.4f}")
    print(f"  総リターン:           {metrics.total_return:.4f}")
    print(f"{'='*50}")
