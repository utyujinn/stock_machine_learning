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

        # 翌日のリターンを使ってポートフォリオリターンを計算
        # 論文: "positions are opened at the end of day t ... closed at the end of day t+1"
        date_idx = returns_df.index.get_loc(date)
        if date_idx + 1 >= len(returns_df):
            continue
        next_date = returns_df.index[date_idx + 1]

        # ロング側の平均リターン
        long_returns = []
        for coin in longs:
            if coin in returns_df.columns:
                r = returns_df.loc[next_date, coin]
                if not np.isnan(r):
                    long_returns.append(r)

        # ショート側の平均リターン
        short_returns = []
        for coin in shorts:
            if coin in returns_df.columns:
                r = returns_df.loc[next_date, coin]
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
        portfolio_dates.append(next_date)

        # 精度: ロング銘柄が中央値以上、ショート銘柄が中央値以下かを確認
        all_returns_today = returns_df.loc[next_date].dropna()
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


def calculate_metrics(daily_returns: pd.Series) -> PerformanceMetrics:
    """パフォーマンス評価指標を計算.

    論文 Table 6 に対応する指標を計算。
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

    # 年率化 (暗号資産は365日取引)
    ann_vol = std_ret * np.sqrt(365)
    sharpe = (mean_ret / std_ret) * np.sqrt(365) if std_ret > 0 else 0.0
    sortino = (mean_ret / downside_std) * np.sqrt(365) if downside_std > 0 else 0.0

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
