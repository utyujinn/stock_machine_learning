"""動的ティッカー選定 — Bitget出来高 + Binance上場で絞り込み.

再訓練時に呼び出し、流動性の高い銘柄のみを選定する。
"""

import ccxt
import requests

from config import DYNAMIC_TICKER_MIN_VOLUME_USD

# Bitget → Binance シンボル変換 (リブランド対応)
# Bitget が新名採用済み、Binance が旧名のままのケースのみ記載
_REBRAND_MAP = {
    "S": "FTM",       # Sonic → Fantom
    "SKY": "MKR",     # Sky → Maker
}

# 除外するステーブルコイン (ティッカーベース)
_STABLECOINS = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "HUSD", "FRAX", "MIM",
    "UST", "USDN", "SUSD", "MUSD", "FDUSD", "PYUSD", "USDE", "USDD",
    "CUSD", "GUSD", "PAX", "USDP", "EUROC", "EURT", "AEUR",
}


def get_dynamic_tickers(min_volume_usd: float | None = None) -> list[str]:
    """Bitget USDT-FUTURES の24h出来高でフィルタし、Binance上場を確認.

    手順:
      1. Bitget USDT-FUTURES の全銘柄 + 24h出来高を取得
      2. 出来高 > min_volume_usd でフィルタ
      3. ステーブルコイン除外
      4. Binance SPOT の上場確認
      5. リブランドマップで名称を変換

    Returns:
        Binance形式のティッカーリスト (例: ["BTCUSDT", "ETHUSDT", ...])
    """
    if min_volume_usd is None:
        min_volume_usd = DYNAMIC_TICKER_MIN_VOLUME_USD

    # 1. Bitget USDT-FUTURES のティッカー + 24h出来高を取得
    print("  Bitget USDT-FUTURES ティッカー取得中...")
    exchange = ccxt.bitget({"options": {"defaultType": "swap"}})
    exchange.load_markets()
    tickers = exchange.fetch_tickers()

    # 2. 出来高でフィルタ + ステーブル除外
    candidates = {}  # bitget_base -> quoteVolume
    for symbol, ticker in tickers.items():
        if not symbol.endswith("/USDT:USDT"):
            continue
        quote_vol = float(ticker.get("quoteVolume", 0) or 0)
        if quote_vol < min_volume_usd:
            continue
        base = symbol.split("/")[0]
        # ステーブルコインチェック (リブランド後の名前も確認)
        binance_name = _REBRAND_MAP.get(base, base)
        if base in _STABLECOINS or binance_name in _STABLECOINS:
            continue
        candidates[base] = quote_vol

    print(f"  Bitget: {len(candidates)} 銘柄が出来高 > ${min_volume_usd / 1e6:.0f}M")

    # 3. Binance SPOT の上場銘柄を取得
    print("  Binance 上場確認中...")
    resp = requests.get(
        "https://api.binance.com/api/v3/exchangeInfo",
        timeout=30,
    )
    resp.raise_for_status()
    binance_symbols = {
        s["symbol"]
        for s in resp.json()["symbols"]
        if s["status"] == "TRADING"
    }

    # 4. Binance に上場している銘柄のみを残す
    result = []
    not_on_binance = []
    for bitget_base, vol in candidates.items():
        # リブランドマップで変換して確認
        binance_base = _REBRAND_MAP.get(bitget_base, bitget_base)
        binance_ticker = f"{binance_base}USDT"
        if binance_ticker in binance_symbols:
            result.append(binance_ticker)
        elif bitget_base != binance_base:
            # マップ変換がヒットしなかった場合、元の名前でも試す
            orig_ticker = f"{bitget_base}USDT"
            if orig_ticker in binance_symbols:
                result.append(orig_ticker)
            else:
                not_on_binance.append(bitget_base)
        else:
            not_on_binance.append(bitget_base)

    if not_on_binance:
        n = len(not_on_binance)
        preview = ", ".join(sorted(not_on_binance)[:10])
        suffix = "..." if n > 10 else ""
        print(f"  Binance未上場 ({n}銘柄): {preview}{suffix}")

    result.sort()
    print(f"  最終: {len(result)} 銘柄を選定")
    return result
