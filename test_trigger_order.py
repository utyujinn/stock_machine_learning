"""Bitgetトリガー注文APIの動作確認 (v2)."""

import os
import sys

from dotenv import load_dotenv
import ccxt

load_dotenv()

exchange = ccxt.bitget({
    "apiKey": os.getenv("BITGET_API_KEY"),
    "secret": os.getenv("BITGET_API_SECRET"),
    "password": os.getenv("BITGET_API_PASSWORD"),
    "options": {"defaultType": "swap"},
})
exchange.load_markets()

try:
    exchange.set_position_mode(True)
except Exception:
    pass

symbol = "BTC/USDT:USDT"
ticker = exchange.fetch_ticker(symbol)
price = ticker["last"]
print(f"BTC現在価格: ${price:.2f}")

market = exchange.markets[symbol]
min_amount = market["limits"]["amount"]["min"]
print(f"最小注文量: {min_amount} contracts")
print(f"価格精度: {market['precision']['price']}")

# 価格を取引所精度に合わせる
trigger_price = price * 1.10
trigger_price_str = exchange.price_to_precision(symbol, trigger_price)
print(f"トリガー価格: ${trigger_price_str} (現在+10%)")

# --- Method 1: triggerPrice (精度修正版) ---
print("\n--- Method 1: triggerPrice (精度修正) ---")
try:
    order = exchange.create_order(
        symbol, "market", "buy", min_amount, None,
        params={
            "triggerPrice": trigger_price_str,
            "triggerType": "mark_price",
            "tradeSide": "close",
            "holdSide": "short",
        },
    )
    print(f"成功! order_id={order.get('id')}")
    # キャンセル
    try:
        exchange.cancel_order(order["id"], symbol, params={"stop": True})
        print("  → キャンセル成功 (stop=True)")
    except Exception as e:
        print(f"  → stop=Trueキャンセル失敗: {e}")
        try:
            exchange.cancel_order(order["id"], symbol)
            print("  → 通常キャンセル成功")
        except Exception as e2:
            print(f"  → 通常キャンセルも失敗: {e2}")
except Exception as e:
    print(f"失敗: {e}")

# --- Method 1b: tradeSideなし (ポジションなし状態) ---
print("\n--- Method 1b: tradeSide省略 ---")
try:
    order = exchange.create_order(
        symbol, "market", "buy", min_amount, None,
        params={
            "triggerPrice": trigger_price_str,
            "triggerType": "mark_price",
        },
    )
    print(f"成功! order_id={order.get('id')}")
    try:
        exchange.cancel_order(order["id"], symbol, params={"stop": True})
        print("  → キャンセル成功")
    except Exception as e:
        print(f"  → キャンセル失敗: {e}")
except Exception as e:
    print(f"失敗: {e}")

# --- 別のシンボル (DOGE) でもテスト ---
print("\n--- DOGE テスト ---")
symbol2 = "DOGE/USDT:USDT"
ticker2 = exchange.fetch_ticker(symbol2)
price2 = ticker2["last"]
market2 = exchange.markets[symbol2]
min_amount2 = market2["limits"]["amount"]["min"]
trigger2 = exchange.price_to_precision(symbol2, price2 * 1.10)
print(f"DOGE: ${price2:.5f}, min={min_amount2}, trigger=${trigger2}")

try:
    order = exchange.create_order(
        symbol2, "market", "buy", min_amount2, None,
        params={
            "triggerPrice": trigger2,
            "triggerType": "mark_price",
            "tradeSide": "close",
            "holdSide": "short",
        },
    )
    print(f"成功! order_id={order.get('id')}")
    exchange.cancel_order(order["id"], symbol2, params={"stop": True})
    print("  → キャンセル成功")
except Exception as e:
    print(f"失敗: {e}")

# --- トリガー注文一覧 ---
print("\n--- 残っているトリガー注文 ---")
for sym in [symbol, symbol2]:
    try:
        orders = exchange.fetch_open_orders(sym, params={"stop": True})
        if orders:
            print(f"  {sym}: {len(orders)} 件")
            for o in orders:
                print(f"    {o['id']} {o['side']} trigger={o.get('triggerPrice', '?')}")
                # クリーンアップ
                try:
                    exchange.cancel_order(o["id"], sym, params={"stop": True})
                    print(f"    → キャンセル")
                except Exception:
                    pass
        else:
            print(f"  {sym}: なし")
    except Exception as e:
        print(f"  {sym}: 取得失敗: {e}")

print("\n完了")
