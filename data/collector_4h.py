"""暗号資産4時間足ヒストリカルデータ収集 (Binance API).

Binance公開APIを使用して4hローソク足を取得する。
認証不要のパブリックエンドポイントを使用。
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from config import DATA_DIR, STUDY_PERIODS, SEQUENCE_LENGTH

# yfinance形式 (BTC-USD) → Binance形式 (BTCUSDT) への変換
# Binanceに存在しない銘柄は取得時にスキップされる
BINANCE_TICKERS = [
    # --- 大型 (Top 20) ---
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "SHIBUSDT",
    "TRXUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "UNIUSDT",
    "NEARUSDT", "ATOMUSDT", "XLMUSDT", "ETCUSDT", "HBARUSDT",
    # --- 中型 (21-60) ---
    "ICPUSDT", "FILUSDT", "VETUSDT", "ALGOUSDT", "AAVEUSDT",
    "MKRUSDT", "GRTUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
    "THETAUSDT", "XTZUSDT", "EOSUSDT", "CHZUSDT", "COMPUSDT",
    "SNXUSDT", "CRVUSDT", "ENJUSDT", "BATUSDT", "ZECUSDT",
    "DASHUSDT", "NEOUSDT", "1INCHUSDT", "KAVAUSDT", "CELOUSDT",
    "ZILUSDT", "ONEUSDT", "ANKRUSDT", "ZRXUSDT", "KSMUSDT",
    "LRCUSDT", "SKLUSDT", "STORJUSDT", "GALAUSDT", "IMXUSDT",
    "LDOUSDT", "OPUSDT", "QNTUSDT", "FTMUSDT", "FLOWUSDT",
    # --- 2023-2026年の主要新規銘柄 ---
    "SUIUSDT", "APTUSDT", "ARBUSDT", "SEIUSDT", "TIAUSDT",
    "INJUSDT", "STXUSDT", "RENDERUSDT", "FETUSDT", "PEPEUSDT",
    "FLOKIUSDT", "WIFUSDT", "BONKUSDT", "JUPUSDT", "PYTHUSDT",
    "WLDUSDT", "ONDOUSDT", "TAOUSDT", "KASUSDT", "PENDLEUSDT",
    # --- その他中型 ---
    "ARUSDT", "EGLDUSDT", "RUNEUSDT", "CROUSDT",
    "MINAUSDT", "APEUSDT",
    "DGBUSDT", "COTIUSDT", "KNCUSDT", "BANDUSDT", "RVNUSDT",
    "ICXUSDT", "ONTUSDT", "XDCUSDT", "HOTUSDT", "YFIUSDT",
    "SUSHIUSDT", "BNTUSDT", "DENTUSDT", "CKBUSDT", "NKNUSDT",
]

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"


def _ticker_to_column(binance_ticker: str) -> str:
    """Binanceティッカーを表示用カラム名に変換.

    BTCUSDT → BTC-USD (yfinance形式と統一)
    """
    symbol = binance_ticker.replace("USDT", "")
    return f"{symbol}-USD"


def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Binance APIからローソク足データをページネーション付きで取得."""
    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }

        for attempt in range(3):
            try:
                resp = requests.get(BINANCE_API_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    # レート制限
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 2:
                    return all_klines
                time.sleep(2)

        data = resp.json()
        if not data:
            break

        all_klines.extend(data)
        # 次のリクエストの開始時刻 = 最後のローソク足のクローズ時刻 + 1
        current_start = data[-1][6] + 1

        # API制限を考慮して少し待機
        time.sleep(0.1)

    return all_klines


def collect_4h_data() -> pd.DataFrame:
    """全銘柄の4時間足終値をBinance APIから取得.

    Returns:
        pd.DataFrame: index=datetime (4h間隔), columns=ティッカー, values=USD終値
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "all_prices_4h.csv")

    if os.path.exists(cache_path):
        print("4hキャッシュからデータ読み込み中...")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"  {len(df.columns)} 銘柄 × {len(df)} 本のデータ読み込み完了")
        return df

    # 全Study Periodをカバーする日付範囲を算出
    earliest = min(sp["train"][0] for sp in STUDY_PERIODS)
    latest = max(sp["test"][1] for sp in STUDY_PERIODS)

    # ルックバックバッファを追加 (90本 × 4h = 15日 + 余裕)
    earliest_dt = datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=30)
    end_dt = datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)

    start_ms = int(earliest_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"データ取得範囲: {earliest_dt.strftime('%Y-%m-%d')} ~ {latest}")
    print(f"{len(BINANCE_TICKERS)} 銘柄の4h足データを Binance API から取得中...")

    all_data = {}
    failed = []

    for i, ticker in enumerate(BINANCE_TICKERS):
        col_name = _ticker_to_column(ticker)
        klines = _fetch_klines(ticker, "4h", start_ms, end_ms)

        if len(klines) < 100:
            failed.append(ticker)
            continue

        # ローソク足データをパース
        # [openTime, open, high, low, close, volume, closeTime, ...]
        timestamps = [datetime.utcfromtimestamp(k[0] / 1000) for k in klines]
        closes = [float(k[4]) for k in klines]

        all_data[col_name] = pd.Series(closes, index=timestamps)

        progress = (i + 1) / len(BINANCE_TICKERS) * 100
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{progress:5.1f}%] {i + 1}/{len(BINANCE_TICKERS)} 銘柄取得完了")

    if failed:
        print(f"\n{len(failed)} 銘柄が取得失敗: {failed[:10]}{'...' if len(failed) > 10 else ''}")

    price_df = pd.DataFrame(all_data)
    price_df.index.name = "datetime"
    price_df = price_df.sort_index()

    # 重複インデックスを除去
    price_df = price_df[~price_df.index.duplicated(keep="first")]

    price_df.to_csv(cache_path)
    print(f"\n{len(price_df.columns)} 銘柄 × {len(price_df)} 本の4hデータ収集・キャッシュ完了")

    return price_df


if __name__ == "__main__":
    df = collect_4h_data()
    print(f"\nデータ形状: {df.shape}")
    print(f"期間: {df.index.min()} ~ {df.index.max()}")
    print(f"銘柄数: {len(df.columns)}")
    print(f"1日あたりの平均本数: {len(df) / ((df.index.max() - df.index.min()).days + 1):.1f}")
