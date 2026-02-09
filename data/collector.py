"""暗号資産ヒストリカルデータ収集 (yfinance).

論文の方法論:
- 時価総額上位100銘柄 (各Study Period の訓練開始日基準)
- ステーブルコイン・問題銘柄を除外
- 日次終値を取得

yfinance を使用し一括ダウンロードで高速取得する。
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config import DATA_DIR, STUDY_PERIODS

# 論文の対象に近い時価総額上位の暗号資産 (yfinance ティッカー)
# ステーブルコイン・問題銘柄は除外済み
# yfinance では "{SYMBOL}-USD" 形式
CRYPTO_TICKERS = [
    # --- 大型 (Top 20) ---
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "SOL-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "SHIB-USD",
    "TRX-USD", "LINK-USD", "LTC-USD", "BCH-USD", "UNI-USD",
    "NEAR-USD", "ATOM-USD", "XLM-USD", "ETC-USD", "HBAR-USD",
    # --- 中型 (21-60) ---
    "ICP-USD", "FIL-USD", "VET-USD", "ALGO-USD", "AAVE-USD",
    "MKR-USD", "GRT-USD", "SAND-USD", "MANA-USD", "AXS-USD",
    "THETA-USD", "XTZ-USD", "EOS-USD", "CHZ-USD", "COMP-USD",
    "SNX-USD", "CRV-USD", "ENJ-USD", "BAT-USD", "ZEC-USD",
    "DASH-USD", "NEO-USD", "1INCH-USD", "KAVA-USD", "CELO-USD",
    "ZIL-USD", "ONE-USD", "ANKR-USD", "ZRX-USD", "KSM-USD",
    "LRC-USD", "SKL-USD", "STORJ-USD", "GALA-USD", "IMX-USD",
    "LDO-USD", "OP-USD", "QNT-USD", "FTM-USD", "FLOW-USD",
    # --- 2023-2026年の主要新規銘柄 ---
    "SUI-USD", "APT-USD", "ARB-USD", "SEI-USD", "TIA-USD",
    "INJ-USD", "STX-USD", "RENDER-USD", "FET-USD", "PEPE-USD",
    "FLOKI-USD", "WIF-USD", "BONK-USD", "JUP-USD", "PYTH-USD",
    "WLD-USD", "ONDO-USD", "TAO-USD", "KAS-USD", "PENDLE-USD",
    # --- その他中型 ---
    "AR-USD", "EGLD-USD", "RUNE-USD", "CRO-USD", "LEO-USD",
    "OKB-USD", "MINA-USD", "APE-USD", "NEXO-USD", "SC-USD",
    "DGB-USD", "COTI-USD", "KNC-USD", "BAND-USD", "RVN-USD",
    "ICX-USD", "ONT-USD", "XDC-USD", "HOT-USD", "YFI-USD",
    "SUSHI-USD", "BNT-USD", "DENT-USD", "CKB-USD", "NKN-USD",
]


def _ensure_cache_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def collect_all_data() -> pd.DataFrame:
    """全銘柄の日次終値を yfinance から一括ダウンロード.

    Returns:
        pd.DataFrame: index=日付, columns=ティッカー, values=USD終値
    """
    _ensure_cache_dir()
    combined_cache = os.path.join(DATA_DIR, "all_prices.csv")
    if os.path.exists(combined_cache):
        print("キャッシュからデータ読み込み中...")
        df = pd.read_csv(combined_cache, index_col=0, parse_dates=True)
        print(f"  {len(df.columns)} 銘柄 × {len(df)} 日のデータ読み込み完了")
        return df

    # 全Study Periodをカバーする日付範囲を算出
    earliest = min(sp["train"][0] for sp in STUDY_PERIODS)
    latest = max(sp["test"][1] for sp in STUDY_PERIODS)

    # 90日のルックバックバッファを追加
    earliest_dt = datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=120)
    start_date = earliest_dt.strftime("%Y-%m-%d")
    # yfinance の end は exclusive なので1日追加
    end_dt = datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)
    end_date = end_dt.strftime("%Y-%m-%d")

    print(f"データ取得範囲: {start_date} ~ {latest}")
    print(f"{len(CRYPTO_TICKERS)} 銘柄の日次価格データを yfinance から一括取得中...")

    # yfinance で一括ダウンロード
    raw = yf.download(
        CRYPTO_TICKERS,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=True,
    )

    # 終値を抽出
    if isinstance(raw.columns, pd.MultiIndex):
        price_df = raw["Close"]
    else:
        # 1銘柄のみの場合
        price_df = raw[["Close"]].rename(columns={"Close": CRYPTO_TICKERS[0]})

    # データが全くない銘柄を除外
    valid_cols = price_df.columns[price_df.notna().sum() > 100]
    price_df = price_df[valid_cols]

    price_df.to_csv(combined_cache)
    print(f"\n{len(price_df.columns)} 銘柄 × {len(price_df)} 日のデータ収集・キャッシュ完了")

    return price_df


if __name__ == "__main__":
    df = collect_all_data()
    print(f"\nデータ形状: {df.shape}")
    print(f"期間: {df.index.min()} ~ {df.index.max()}")
    print(f"銘柄数: {len(df.columns)}")
