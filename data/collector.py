"""CoinGecko API からの暗号資産ヒストリカルデータ収集.

論文の方法論:
- 時価総額上位100銘柄 (各Study Period の訓練開始日基準)
- ステーブルコイン・問題銘柄を除外
- 日次終値 (UTC 00:00) を取得
"""

import json
import os
import time
from datetime import datetime

import pandas as pd
import requests

from config import (
    API_RATE_LIMIT_SLEEP,
    COINGECKO_BASE_URL,
    DATA_DIR,
    EXCLUDED_COINS,
    STUDY_PERIODS,
    TOP_K_COINS,
)


def _ensure_cache_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _unix_ms(date_str: str) -> int:
    """'YYYY-MM-DD' → UNIX timestamp (秒)."""
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())


def fetch_top_coins_by_market_cap(date_str: str | None = None) -> list[str]:
    """CoinGecko から時価総額上位銘柄の ID リストを取得.

    CoinGecko の無料 API は過去時点の時価総額ランキングをサポートしないため、
    キャッシュ済みスナップショットを利用するか、現在のランキングを取得する。
    """
    cache_path = os.path.join(DATA_DIR, "top_coins.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    _ensure_cache_dir()
    all_coins = []
    # CoinGecko は 1ページ250件まで
    for page in range(1, 8):  # 最大1750件
        resp = requests.get(
            f"{COINGECKO_BASE_URL}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": page,
                "sparkline": "false",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_coins.extend(data)
        print(f"  時価総額ランキング取得: page {page}, 累計 {len(all_coins)} 件")
        time.sleep(API_RATE_LIMIT_SLEEP)

    # 除外銘柄をフィルタし、上位100を取得
    filtered = [
        c["id"] for c in all_coins
        if c["id"] not in EXCLUDED_COINS
    ][:TOP_K_COINS]

    with open(cache_path, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"  Top {len(filtered)} 銘柄を選定・キャッシュ完了")
    return filtered


def fetch_coin_price_history(coin_id: str, start_date: str, end_date: str) -> pd.Series:
    """特定コインの日次終値を取得.

    Returns:
        pd.Series: index=日付(datetime), values=USD終値
    """
    cache_path = os.path.join(DATA_DIR, f"prices_{coin_id}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df["price"]

    from_ts = _unix_ms(start_date)
    to_ts = _unix_ms(end_date)

    resp = requests.get(
        f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart/range",
        params={
            "vs_currency": "usd",
            "from": from_ts,
            "to": to_ts,
        },
        timeout=30,
    )
    resp.raise_for_status()
    prices_raw = resp.json().get("prices", [])

    if not prices_raw:
        return pd.Series(dtype=float, name="price")

    df = pd.DataFrame(prices_raw, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    # 同一日に複数データがある場合は最後の値 (UTC 00:00 に最も近い) を使用
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.set_index("date").sort_index()

    _ensure_cache_dir()
    df[["price"]].to_csv(cache_path)

    return df["price"]


def collect_all_data() -> pd.DataFrame:
    """全銘柄の日次終値を一つの DataFrame に統合.

    Returns:
        pd.DataFrame: index=日付, columns=coin_id, values=USD終値
    """
    combined_cache = os.path.join(DATA_DIR, "all_prices.csv")
    if os.path.exists(combined_cache):
        print("キャッシュからデータ読み込み中...")
        df = pd.read_csv(combined_cache, index_col=0, parse_dates=True)
        print(f"  {len(df.columns)} 銘柄 × {len(df)} 日のデータ読み込み完了")
        return df

    # 全Study Periodをカバーする日付範囲を算出
    # SEQUENCE_LENGTH (90日) のバッファを含めて最も早い日から最も遅い日まで
    earliest = min(sp["train"][0] for sp in STUDY_PERIODS)
    latest = max(sp["test"][1] for sp in STUDY_PERIODS)

    # 90日前からデータが必要
    from datetime import timedelta
    earliest_dt = datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=120)
    earliest_with_buffer = earliest_dt.strftime("%Y-%m-%d")

    print(f"データ取得範囲: {earliest_with_buffer} ~ {latest}")

    coin_ids = fetch_top_coins_by_market_cap()
    print(f"\n{len(coin_ids)} 銘柄の日次価格データを取得開始...")

    all_series = {}
    for i, coin_id in enumerate(coin_ids):
        print(f"  [{i+1}/{len(coin_ids)}] {coin_id}")
        try:
            series = fetch_coin_price_history(coin_id, earliest_with_buffer, latest)
            if len(series) > 0:
                all_series[coin_id] = series
        except Exception as e:
            print(f"    エラー: {e}")
        time.sleep(API_RATE_LIMIT_SLEEP)

    price_df = pd.DataFrame(all_series)
    price_df = price_df.sort_index()

    _ensure_cache_dir()
    price_df.to_csv(combined_cache)
    print(f"\n{len(price_df.columns)} 銘柄 × {len(price_df)} 日のデータ収集・キャッシュ完了")

    return price_df


if __name__ == "__main__":
    df = collect_all_data()
    print(f"\nデータ形状: {df.shape}")
    print(f"期間: {df.index.min()} ~ {df.index.max()}")
    print(f"銘柄数: {len(df.columns)}")
