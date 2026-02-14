"""LSTM Auto Trader - Bitget Futures 自動売買Bot.

advisor.py の推論パイプラインを再利用し、
Bitget Futures (USDT-M永久先物) で自動ポジション管理を行う。

使い方:
  uv run python trader.py                      # 1回実行 (dry-run)
  uv run python trader.py --live               # 1回実行 (実注文)
  uv run python trader.py --live --loop        # 4h毎ループ実行
  uv run python trader.py --status             # ポジション状況確認
  uv run python trader.py --close-all          # 全ポジション決済 (dry-run)
  uv run python trader.py --close-all --live   # 全ポジション決済 (実行)
"""

import argparse
import json
import math
import os
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 日本標準時 (UTC+9)
JST = timezone(timedelta(hours=9))

import ccxt
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR,
    PORTFOLIO_K,
    RETRAIN_INTERVAL_DAYS,
    RETRAIN_UTC_HOUR,
    TRADE_INTERVAL_HOURS,
    TRADE_LEVERAGE,
    TRADE_LIMIT_OFFSET_PCT,
    TRADE_LIMIT_TIMEOUT_SEC,
    TRADE_STOP_LOSS_PCT,
    TRADE_TOTAL_CAPITAL_USDT,
    VOL_FILTER_MULTIPLIER,
)

from advisor import (
    apply_volatility_filter,
    build_current_features,
    dynamic_rebalance,
    fetch_recent_4h,
    load_positions,
    retrain_models,
    save_positions,
)
from models.lstm_model import ensemble_predict_ranks, load_ensemble

# --- 定数 ---
MODELS_DIR = os.path.join(DATA_DIR, "models")
TRADE_LOG_PATH = os.path.join(DATA_DIR, "trade_log.jsonl")
POSITIONS_PATH = os.path.join(DATA_DIR, "positions_4h.json")

# グレースフル停止用フラグ
_shutdown = False


def _should_retrain() -> bool:
    """モデルの再訓練が必要か判定 (trained_at が RETRAIN_INTERVAL_DAYS 日以上前)."""
    config_path = os.path.join(MODELS_DIR, "4h", "config.json")
    if not os.path.exists(config_path):
        return True
    try:
        with open(config_path) as f:
            config = json.load(f)
        trained_at = datetime.fromisoformat(config["trained_at"])
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - trained_at
        return age.days >= RETRAIN_INTERVAL_DAYS
    except Exception:
        return True


def _handle_signal(signum, frame):
    global _shutdown
    if _shutdown:
        # 2回目のCtrl+C → 即時停止
        log("強制停止")
        sys.exit(1)
    _shutdown = True
    log("Ctrl+C 受信 - 現在のサイクル完了後に安全停止します (ポジションは保持)")
    log("もう一度 Ctrl+C で強制停止")


def log(msg: str):
    """タイムスタンプ付きログ出力 (JST)."""
    ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def log_trade(trade: dict):
    """トレードログをJSONLファイルに追記."""
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    trade["logged_at"] = datetime.now(timezone.utc).isoformat()
    with open(TRADE_LOG_PATH, "a") as f:
        f.write(json.dumps(trade, ensure_ascii=False) + "\n")


# ============================================================
# 取引所接続
# ============================================================

def init_exchange(dry_run: bool = True) -> ccxt.bitget:
    """Bitget Futures クライアントを初期化."""
    load_dotenv()
    api_key = os.getenv("BITGET_API_KEY")
    api_secret = os.getenv("BITGET_API_SECRET")
    api_password = os.getenv("BITGET_API_PASSWORD")

    if not api_key or not api_secret or not api_password:
        log("エラー: .env に BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSWORD を設定してください")
        sys.exit(1)

    exchange = ccxt.bitget({
        "apiKey": api_key,
        "secret": api_secret,
        "password": api_password,
        "options": {"defaultType": "swap"},
    })

    if dry_run:
        exchange.options["createOrder"] = "disabled"

    exchange.load_markets()

    # 双方向ポジションモード (hedge) に設定
    # buy=ロング開設, sell=ショート開設, reduceOnly=決済 として使える
    try:
        exchange.set_position_mode(True)  # True = hedge mode
    except Exception:
        pass  # 既にhedgeモードの場合はエラーになるが無視

    log(f"Bitget Futures 接続完了 (マーケット数: {len(exchange.markets)})")
    return exchange


# Binance → Bitget Futures シンボル名のマッピング (リブランド・デノミ対応)
# 元の名前を先に試し、なければマッピング名を試す
_SYMBOL_MAP = {
    "FTM": "S",           # Fantom → Sonic
    "MKR": "SKY",         # Maker → Sky
}

# 逆マッピング (Bitget → Binance)
_SYMBOL_MAP_REV = {v: k for k, v in _SYMBOL_MAP.items()}

def setup_isolated_margin(exchange, symbol, leverage):
    """銘柄を分離マージンに変更し、レバレッジを設定する"""
    try:
        # 1. 現在のマージンモードとレバレッジを確認（無駄なリクエストを避けるため）
        # Bitgetのポジションモード/マージン設定は銘柄ごとに保持される
        params = {"symbol": symbol}
        # 実際には、設定済みの場合はエラーを投げる取引所が多いため、try-exceptで囲むのが安全
        
        log(f"  設定確認中: {symbol}")
        
        # 2. 分離マージン (isolated) に設定
        # 注意: ポジションがある状態では変更できないため、エラーハンドリングが必要
        try:
            exchange.set_margin_mode('isolated', symbol)
            log(f"    -> 分離マージンに変更成功")
        except Exception as e:
            if "already" in str(e).lower() or "not allowed" in str(e).lower():
                log(f"    -> 分離マージン設定済み、または変更不可（既存ポジ有）")
            else:
                log(f"    -> 分離マージン設定スキップ: {e}")

        # 3. レバレッジを設定
        try:
            exchange.set_leverage(leverage, symbol)
            log(f"    -> レバレッジを {leverage}x に設定完了")
        except Exception as e:
            log(f"    -> レバレッジ設定スキップ: {e}")

    except Exception as e:
        log(f"  {symbol} の設定中にエラーが発生しました: {e}")

def coin_to_exchange_symbol(coin_id: str, exchange: ccxt.bitget) -> str | None:
    """advisor銘柄名 → Bitget Futuresシンボルに変換.

    BTC-USD → BTC/USDT:USDT
    FTM-USD → S/USDT:USDT (元の名前が無い場合のみマッピング適用)

    Returns:
        シンボル文字列、または未対応の場合None
    """
    base = coin_id.replace("-USD", "")
    # 元の名前を先に試す
    symbol = f"{base}/USDT:USDT"
    if symbol in exchange.markets:
        return symbol
    # マッピング名を試す
    mapped = _SYMBOL_MAP.get(base)
    if mapped:
        symbol = f"{mapped}/USDT:USDT"
        if symbol in exchange.markets:
            return symbol
    return None


def exchange_symbol_to_coin(symbol: str) -> str:
    """Bitget Futuresシンボル → advisor銘柄名に変換.

    BTC/USDT:USDT → BTC-USD
    S/USDT:USDT → FTM-USD (逆マッピング適用)
    """
    base = symbol.split("/")[0]
    base = _SYMBOL_MAP_REV.get(base, base)
    return f"{base}-USD"


# ============================================================
# ポジション管理
# ============================================================

def fetch_current_positions(exchange: ccxt.bitget) -> dict[str, dict]:
    """Bitget Futuresの現在のオープンポジションを取得.

    Returns:
        {symbol: {"side": "long"|"short", "size": float, "notional": float, "unrealizedPnl": float, "entryPrice": float}}
    """
    positions = exchange.fetch_positions()
    result = {}
    for pos in positions:
        contracts = float(pos.get("contracts", 0) or 0)
        if contracts == 0:
            continue
        symbol = pos["symbol"]
        side = pos["side"]  # "long" or "short"
        notional = float(pos.get("notional", 0) or 0)
        pnl = float(pos.get("unrealizedPnl", 0) or 0)
        entry = float(pos.get("entryPrice", 0) or 0)
        result[symbol] = {
            "side": side,
            "size": contracts,
            "notional": abs(notional),
            "unrealizedPnl": pnl,
            "entryPrice": entry,
        }
    return result


def check_stop_losses(
    exchange: ccxt.bitget,
    positions: dict[str, dict],
    dry_run: bool,
) -> list[str]:
    """ストップロス判定 & 決済.

    各ポジションのエントリー価格と現在価格を比較し、
    TRADE_STOP_LOSS_PCT (10%) を超えて逆行しているポジションを決済する。

    - LONG: 現在価格がエントリーから10%以上下落 → 決済
    - SHORT: 現在価格がエントリーから10%以上上昇 → 決済

    Returns:
        決済されたcoin_idのリスト (例: ["ZEC-USD", "DASH-USD"])
    """
    if TRADE_STOP_LOSS_PCT <= 0:
        return []

    stopped_coins = []

    for symbol, pos in positions.items():
        entry_price = pos.get("entryPrice", 0)
        if entry_price <= 0:
            continue

        # 現在価格を取得
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker.get("last", 0)
        except Exception:
            continue
        if current_price <= 0:
            continue

        side = pos["side"]
        price_change = (current_price - entry_price) / entry_price

        if side == "long" and price_change < -TRADE_STOP_LOSS_PCT:
            # ロング: 値下がりでストップ
            coin = exchange_symbol_to_coin(symbol)
            log(f"  STOP LOSS: {coin} LONG エントリー${entry_price:.4f} → 現在${current_price:.4f} ({price_change:+.1%})")
            result = close_position(exchange, symbol, "long", dry_run)
            if result:
                result["coin"] = coin
                result["reason"] = "stop_loss"
                log_trade(result)
            stopped_coins.append(coin)

        elif side == "short" and price_change > TRADE_STOP_LOSS_PCT:
            # ショート: 値上がりでストップ
            coin = exchange_symbol_to_coin(symbol)
            log(f"  STOP LOSS: {coin} SHORT エントリー${entry_price:.4f} → 現在${current_price:.4f} ({price_change:+.1%})")
            result = close_position(exchange, symbol, "short", dry_run)
            if result:
                result["coin"] = coin
                result["reason"] = "stop_loss"
                log_trade(result)
            stopped_coins.append(coin)

    return stopped_coins


def place_stop_loss(
    exchange: ccxt.bitget,
    symbol: str,
    side: str,
    entry_price: float,
    contracts: float,
    dry_run: bool,
) -> dict | None:
    """ポジションに対するストップロストリガー注文を取引所に配置.

    LONG: 価格がエントリーの-10%に下落 → 売りで決済
    SHORT: 価格がエントリーの+10%に上昇 → 買いで決済

    Args:
        side: "long" or "short" (ポジションの方向)
        entry_price: エントリー価格
        contracts: コントラクト数
    """
    if TRADE_STOP_LOSS_PCT <= 0 or entry_price <= 0:
        return None

    if side == "long":
        trigger_price = entry_price * (1 - TRADE_STOP_LOSS_PCT)
        order_side = "sell"
    else:
        trigger_price = entry_price * (1 + TRADE_STOP_LOSS_PCT)
        order_side = "buy"

    trigger_price_str = exchange.price_to_precision(symbol, trigger_price)

    if dry_run:
        log(f"  [DRY-RUN] SLトリガー設定: {symbol} {side.upper()} → "
            f"${trigger_price_str} ({TRADE_STOP_LOSS_PCT:.0%}逆行で決済)")
        return {"symbol": symbol, "side": side, "triggerPrice": trigger_price_str, "dry_run": True}

    try:
        order = exchange.create_order(
            symbol, "market", order_side, contracts, None,
            params={
                "triggerPrice": trigger_price_str,
                "triggerType": "mark_price",
                "tradeSide": "close",
                "holdSide": side,
            },
        )
        order_id = order.get("id", "")
        log(f"  SLトリガー設定: {symbol} {side.upper()} → "
            f"${trigger_price_str} (order_id={order_id})")
        return {"symbol": symbol, "side": side, "triggerPrice": trigger_price_str, "order_id": order_id}
    except Exception as e:
        log(f"  SLトリガー設定失敗: {symbol} → {e}")
        return {"symbol": symbol, "error": str(e)}


def cancel_trigger_orders(
    exchange: ccxt.bitget,
    symbol: str,
    dry_run: bool,
) -> int:
    """シンボルの全トリガー注文をキャンセル.

    Returns:
        キャンセルした注文数
    """
    if dry_run:
        return 0

    try:
        orders = exchange.fetch_open_orders(symbol, params={"stop": True})
    except Exception:
        return 0

    cancelled = 0
    for order in orders:
        try:
            exchange.cancel_order(order["id"], symbol, params={"stop": True})
            cancelled += 1
        except Exception:
            pass

    if cancelled > 0:
        log(f"  SLトリガーキャンセル: {symbol} ({cancelled}件)")
    return cancelled


def ensure_stop_losses(
    exchange: ccxt.bitget,
    positions: dict[str, dict],
    dry_run: bool,
):
    """全ポジションにストップロストリガー注文があることを確認.

    不足している場合は新規作成する。
    """
    if TRADE_STOP_LOSS_PCT <= 0:
        return

    for symbol, pos in positions.items():
        entry_price = pos.get("entryPrice", 0)
        contracts = pos.get("size", 0)
        side = pos["side"]

        if entry_price <= 0 or contracts <= 0:
            continue

        # 既存のトリガー注文を確認
        try:
            orders = exchange.fetch_open_orders(symbol, params={"stop": True})
            has_sl = len(orders) > 0
        except Exception:
            has_sl = False

        if not has_sl:
            coin = exchange_symbol_to_coin(symbol)
            log(f"  SLトリガー不足検出: {coin} {side.upper()} → 再設定")
            place_stop_loss(exchange, symbol, side, entry_price, contracts, dry_run)


def cancel_all_open_orders(exchange: ccxt.bitget, dry_run: bool) -> int:
    """全シンボルの未約定注文をキャンセル (起動時クリーンアップ).

    クラッシュ後に残った指値注文やトリガー注文を一掃する。
    """
    if dry_run:
        return 0

    cancelled = 0
    try:
        orders = exchange.fetch_open_orders(params={"productType": "USDT-FUTURES"})
        for order in orders:
            try:
                exchange.cancel_order(order["id"], order["symbol"])
                cancelled += 1
            except Exception:
                pass
    except Exception:
        pass

    # トリガー注文 (SL等) もクリーンアップ
    try:
        trigger_orders = exchange.fetch_open_orders(
            params={"productType": "USDT-FUTURES", "stop": True},
        )
        for order in trigger_orders:
            try:
                exchange.cancel_order(order["id"], order["symbol"], params={"stop": True})
                cancelled += 1
            except Exception:
                pass
    except Exception:
        pass

    return cancelled


def calculate_position_usdt(capital: float, k: int, leverage: int) -> float:
    """1ポジションあたりのUSDT額を計算.

    10ポジション (LONG 5 + SHORT 5) に均等配分。
    """
    return (capital / (2 * k)) * leverage


def _wait_for_fill(
    exchange: ccxt.bitget,
    order_id: str,
    symbol: str,
    timeout: int = TRADE_LIMIT_TIMEOUT_SEC,
) -> dict | None:
    """指値注文の約定を待つ。タイムアウト時はキャンセル.

    Returns:
        約定済みorder or None (キャンセル済み)
    """
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(2)
        try:
            order = exchange.fetch_order(order_id, symbol)
            status = order.get("status", "")
            if status == "closed":
                return order
            if status == "canceled":
                return None
        except Exception:
            pass

    # タイムアウト → キャンセル
    try:
        exchange.cancel_order(order_id, symbol)
    except Exception:
        pass
    return None


def _calc_limit_price(current_price: float, is_buy: bool, offset_pct: float) -> float:
    """指値価格を計算.

    Args:
        current_price: 現在価格
        is_buy: 買い注文ならTrue
        offset_pct: オフセット% (正=有利方向, 負=不利方向もあり得る)
    """
    offset = offset_pct / 100
    if is_buy:
        return current_price * (1 - offset)   # 安く買いたい → 下に指値
    else:
        return current_price * (1 + offset)    # 高く売りたい → 上に指値


# 指値リトライのオフセットスケジュール (% 単位)
# 常に現在価格以上に有利な位置のみ。回を重ねるごとに有利幅を縮小。
_LIMIT_OFFSETS = [
    TRADE_LIMIT_OFFSET_PCT,           # 1回目: 0.05% 有利
    TRADE_LIMIT_OFFSET_PCT,           # 2回目: 0.05% 有利 (最新価格で再指値)
    TRADE_LIMIT_OFFSET_PCT / 2,       # 3回目: 0.025% 有利
    0.0,                               # 4回目: 現在価格ちょうど
]


def _execute_limit_with_retry(
    exchange: ccxt.bitget,
    symbol: str,
    order_side: str,
    amount: float,
    initial_price: float,
    is_buy: bool,
    params: dict | None = None,
    dry_run: bool = False,
    label: str = "",
) -> dict | None:
    """指値注文をリトライしながら約定させる (成行は使わない).

    Args:
        exchange: 取引所クライアント
        symbol: シンボル (例: BTC/USDT:USDT)
        order_side: "buy" or "sell"
        amount: 注文数量 (コントラクト単位)
        initial_price: 初回の参考価格
        is_buy: 買い注文ならTrue
        params: 追加パラメータ (reduceOnly等)
        dry_run: True=注文送信しない
        label: ログ用ラベル

    Returns:
        {"price": float, "order_id": str, "attempt": int, "type": str} or {"error": str}
    """
    if params is None:
        params = {}

    for attempt, offset in enumerate(_LIMIT_OFFSETS):
        # 最新価格取得 (2回目以降)
        if attempt > 0:
            try:
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker.get("last", initial_price)
                if current_price > 0:
                    initial_price = current_price
            except Exception:
                pass

        limit_price = _calc_limit_price(initial_price, is_buy, offset)

        if offset > 0:
            desc = f"{offset}%有利"
        elif offset == 0:
            desc = "現在価格"
        else:
            desc = f"{abs(offset)}%不利(確実約定)"

        if dry_run:
            log(f"  [DRY-RUN] {label} → 指値${limit_price:.4f} ({desc})")
            return {"price": limit_price, "type": "limit", "attempt": 1, "dry_run": True}

        log(f"    [{attempt+1}/{len(_LIMIT_OFFSETS)}] 指値 ${limit_price:.4f} ({desc})")

        try:
            order = exchange.create_order(
                symbol, "limit", order_side, amount, limit_price, params=params,
            )
        except Exception as e:
            err_str = str(e)
            log(f"    注文作成失敗: {e}")
            # リトライしても無意味なエラー → 即座にリターン
            if "45110" in err_str or "minimum amount" in err_str.lower():
                return {"error": f"最小注文額未満: {symbol}"}
            if "insufficient" in err_str.lower():
                return {"error": f"残高不足: {symbol}"}
            if "40774" in err_str or "unilateral" in err_str.lower():
                return {"error": f"ポジションモードエラー: {symbol}"}
            if "22002" in err_str:
                return {"error": f"ポジションなし: {symbol}", "code": "22002"}
            # レート制限 → 少し待ってリトライ
            if "busy" in err_str.lower() or "frequent" in err_str.lower():
                time.sleep(3)
            else:
                time.sleep(2)
            continue

        filled = _wait_for_fill(exchange, order["id"], symbol)

        if filled:
            fill_price = filled.get("average", limit_price)
            log(f"    約定 @ ${fill_price:.4f} (試行{attempt+1})")
            return {
                "price": fill_price,
                "order_id": filled.get("id"),
                "attempt": attempt + 1,
                "type": "limit",
            }

        log(f"    未約定 → キャンセル")

    return {"error": f"全{len(_LIMIT_OFFSETS)}回の指値が未約定"}


def close_position(
    exchange: ccxt.bitget,
    symbol: str,
    side: str,
    dry_run: bool,
) -> dict | None:
    """既存ポジションを指値で決済 (全回失敗時はフラッシュクローズ)."""
    try:
        # ポジションサイズとマージンモードを取得
        positions = exchange.fetch_positions([symbol])
        pos_size = 0
        pos_margin_mode = None
        for p in positions:
            if p["symbol"] == symbol and p.get("side") == side and float(p.get("contracts", 0) or 0) > 0:
                pos_size = float(p["contracts"])
                pos_margin_mode = p.get("marginMode")  # "cross" or "isolated"
                break

        if pos_size == 0:
            log(f"  {symbol} ポジションなし - スキップ")
            return None

        # ポジションのマージンモードに合わせる (cross/isolated 不一致を防止)
        if pos_margin_mode:
            try:
                exchange.set_margin_mode(pos_margin_mode, symbol)
            except Exception:
                pass

        order_side = "sell" if side == "long" else "buy"
        is_buy = order_side == "buy"

        # 現在価格
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker.get("last", 0)
        except Exception:
            current_price = 0

        label = f"CLOSE {side.upper()} {symbol} ({pos_size} contracts)"

        if dry_run:
            if current_price > 0:
                limit_price = _calc_limit_price(current_price, is_buy, TRADE_LIMIT_OFFSET_PCT)
                log(f"  [DRY-RUN] {label} → 指値${limit_price:.4f} (現在${current_price:.4f})")
            else:
                log(f"  [DRY-RUN] {label}")
            return {"action": "close", "symbol": symbol, "side": side, "size": pos_size, "dry_run": True}

        if current_price <= 0:
            log(f"  {label} → 価格不明、フラッシュクローズ")
            result = exchange.close_position(symbol, side=side)
            return {
                "action": "close", "symbol": symbol, "side": side,
                "size": pos_size, "order_id": result.get("id", ""),
                "type": "flash_close",
            }

        log(f"  {label} (現在${current_price:.4f})")
        result = _execute_limit_with_retry(
            exchange, symbol, order_side, pos_size, current_price,
            is_buy=is_buy,
            params={"tradeSide": "close", "holdSide": side},
            label=label,
        )

        if result and "error" not in result:
            result["action"] = "close"
            result["symbol"] = symbol
            result["side"] = side
            result["size"] = pos_size
            return result

        # 22002 (ポジションなし) → 既に決済済みとして成功扱い
        if result and result.get("code") == "22002":
            log(f"    ポジション既に決済済み → スキップ")
            return {
                "action": "close", "symbol": symbol, "side": side,
                "size": pos_size, "type": "already_closed",
            }

        # 指値全回失敗 → ポジション再確認してからフラッシュクローズ
        log(f"    指値全回失敗 → ポジション再確認中...")
        try:
            positions = exchange.fetch_positions([symbol])
            still_open = any(
                p["symbol"] == symbol and p.get("side") == side
                and float(p.get("contracts", 0) or 0) > 0
                for p in positions
            )
        except Exception:
            still_open = True  # 確認できない場合は安全側でフォールバック

        if not still_open:
            log(f"    ポジション既に決済済み → スキップ")
            return {
                "action": "close", "symbol": symbol, "side": side,
                "size": pos_size, "type": "already_closed",
            }

        log(f"    フラッシュクローズにフォールバック")
        try:
            fb = exchange.close_position(symbol, side=side)
            return {
                "action": "close", "symbol": symbol, "side": side,
                "size": pos_size, "order_id": fb.get("id", ""),
                "type": "flash_close_fallback",
            }
        except Exception as e2:
            log(f"    フラッシュクローズも失敗: {e2}")
            return {"action": "close", "symbol": symbol, "error": str(e2)}

    except Exception as e:
        log(f"  CLOSE {symbol} 失敗: {e}")
        return {"action": "close", "symbol": symbol, "error": str(e)}


BITGET_MIN_NOTIONAL_USDT = 5.0  # Bitgetの最小注文額 (ハードリミット)


def _get_min_order_usdt(exchange: ccxt.bitget, symbol: str, price: float) -> float:
    """シンボルの最小注文額 (USDT) を取得."""
    m = exchange.markets.get(symbol, {})
    cs = m.get("contractSize", 1) or 1
    min_amt = (m.get("limits", {}).get("amount", {}).get("min", 0)) or 0
    min_from_contracts = min_amt * cs * price
    return max(min_from_contracts, BITGET_MIN_NOTIONAL_USDT)


def _base_to_contracts(exchange: ccxt.bitget, symbol: str, base_amount: float) -> float:
    """ベース通貨量をコントラクト数に変換.

    例: BTC contractSize=0.001 → 0.005 BTC = 5 contracts
    """
    market = exchange.markets.get(symbol, {})
    contract_size = float(market.get("contractSize", 1) or 1)
    return base_amount / contract_size


def _round_up_contracts(exchange: ccxt.bitget, symbol: str, amount: float) -> float:
    """コントラクト数を有効精度で切り上げ (最小注文額を確保)."""
    market = exchange.markets.get(symbol, {})
    precision = market.get("precision", {}).get("amount")
    if precision is None:
        return amount
    if exchange.precisionMode == ccxt.TICK_SIZE:
        step = float(precision)
        if step <= 0:
            return amount
        return math.ceil(amount / step) * step
    else:
        factor = 10 ** int(precision)
        return math.ceil(amount * factor) / factor


def open_position(
    exchange: ccxt.bitget,
    symbol: str,
    side: str,
    usdt_amount: float,
    price: float,
    dry_run: bool,
) -> dict | None:
    """新規ポジションを開設 (指値リトライ、成行は使わない)."""
    try:
        # マージンモード・レバレッジ設定
        if not dry_run:
            setup_isolated_margin(exchange, symbol, TRADE_LEVERAGE)
        else:
            log(f"  [DRY-RUN] {symbol} を ISOLATED モード / {TRADE_LEVERAGE}x レバレッジに設定想定")

        order_side = "buy" if side == "long" else "sell"
        is_buy = order_side == "buy"

        # 最新価格を取得
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker.get("last", price)
            if current_price > 0:
                price = current_price
        except Exception:
            pass

        # 最小注文額チェック
        min_usdt = _get_min_order_usdt(exchange, symbol, price)
        if usdt_amount < min_usdt:
            log(f"  {symbol} → 注文額 {usdt_amount:.1f} USDT < 最小 {min_usdt:.1f} USDT - スキップ")
            return {"action": "open", "symbol": symbol, "side": side, "error": f"最小注文額不足 ({min_usdt:.1f} USDT)"}

        # 数量計算 (コントラクト単位に変換、切り上げで最小注文額を確保)
        base_amount = usdt_amount / price
        amount = _base_to_contracts(exchange, symbol, base_amount)
        amount = _round_up_contracts(exchange, symbol, amount)
        label = f"NEW {side.upper()} {symbol} {usdt_amount:.1f}USDT"

        if dry_run:
            limit_price = _calc_limit_price(price, is_buy, TRADE_LIMIT_OFFSET_PCT)
            log(f"  [DRY-RUN] {label} → 指値${limit_price:.4f} (現在${price:.4f}, {TRADE_LIMIT_OFFSET_PCT}%有利)")
            return {
                "action": "open", "symbol": symbol, "side": side,
                "usdt": usdt_amount, "price": limit_price, "amount": amount,
                "type": "limit", "dry_run": True,
            }

        log(f"  {label} (現在${price:.4f})")
        result = _execute_limit_with_retry(
            exchange, symbol, order_side, amount, price,
            is_buy=is_buy, params={"tradeSide": "open"}, label=label,
        )

        if result and "error" not in result:
            result["action"] = "open"
            result["symbol"] = symbol
            result["side"] = side
            result["usdt"] = usdt_amount
            result["amount"] = amount
            return result

        # 指値全回失敗 → 成行フォールバック
        log(f"    指値全回失敗 → 成行にフォールバック")
        try:
            order = exchange.create_order(
                symbol, "market", order_side, amount,
                params={"tradeSide": "open"},
            )
            fill_price = order.get("average") or order.get("price") or price
            log(f"    成行約定 @ ${fill_price:.4f}")
            return {
                "action": "open", "symbol": symbol, "side": side,
                "usdt": usdt_amount, "price": fill_price, "amount": amount,
                "order_id": order.get("id", ""),
                "type": "market_fallback",
            }
        except Exception as e2:
            log(f"    成行も失敗: {e2}")
            return {"action": "open", "symbol": symbol, "side": side, "error": str(e2)}

    except Exception as e:
        log(f"  NEW {side.upper()} {symbol} 失敗: {e}")
        return {"action": "open", "symbol": symbol, "side": side, "error": str(e)}


# ============================================================
# 発注指示書
# ============================================================

def print_order_sheet(
    exchange: ccxt.bitget,
    new_longs: list[str],
    new_shorts: list[str],
    changes: dict[str, str],
    prices: dict[str, float],
):
    """Bitget発注指示書を出力.

    銘柄・サイド・現在価格・SLトリガー価格を一覧表示する。
    """
    position_usdt = calculate_position_usdt(
        TRADE_TOTAL_CAPITAL_USDT, PORTFOLIO_K, TRADE_LEVERAGE,
    )

    log("")
    log("=" * 66)
    log("  Bitget 発注指示書")
    log("=" * 66)
    log(f"  {'アクション':<14s} {'銘柄':<22s} {'サイド':<7s} {'価格':>12s} {'金額':>8s}")
    log(f"  {'─' * 62}")

    # NEW + KEEP をまとめて表示
    for coin in new_longs + new_shorts:
        action = changes.get(coin, "")
        side = "LONG" if coin in new_longs else "SHORT"
        price = prices.get(coin, 0)

        # Bitgetシンボル
        symbol = coin_to_exchange_symbol(coin, exchange)
        symbol_str = symbol if symbol else f"{coin} (未対応)"

        # 価格フォーマット
        if symbol and price > 0:
            price_str = f"${exchange.price_to_precision(symbol, price)}"
        elif price > 0:
            price_str = f"${price:.4f}"
        else:
            price_str = "---"

        if action.startswith("NEW"):
            action_label = "NEW"
            usdt_str = f"{position_usdt:.0f}"
        else:
            action_label = "KEEP"
            usdt_str = "---"

        log(f"  {action_label:<14s} {symbol_str:<22s} {side:<7s} {price_str:>12s} {usdt_str:>8s}")

    # CLOSE
    close_items = [(c, a) for c, a in changes.items() if a.startswith("CLOSE")]
    if close_items:
        log(f"  {'─' * 62}")
        for coin, action in close_items:
            side = "LONG" if action == "CLOSE_LONG" else "SHORT"
            symbol = coin_to_exchange_symbol(coin, exchange)
            symbol_str = symbol if symbol else f"{coin} (未対応)"
            log(f"  {'CLOSE':<14s} {symbol_str:<22s} {side:<7s}")

    log(f"  {'─' * 62}")
    n_new = sum(1 for a in changes.values() if a.startswith("NEW"))
    n_keep = sum(1 for a in changes.values() if a.startswith("KEEP"))
    n_close = sum(1 for a in changes.values() if a.startswith("CLOSE"))
    log(f"  NEW: {n_new} | KEEP: {n_keep} | CLOSE: {n_close} | 1pos: {position_usdt:.0f} USDT")
    log("=" * 66)
    log("")


# ============================================================
# ポジション同期
# ============================================================

def _exec_close(exchange, coin, action, dry_run):
    """1銘柄のクローズ処理 (並列実行用)."""
    symbol = coin_to_exchange_symbol(coin, exchange)
    if not symbol:
        log(f"  {coin} → Bitget未対応 - スキップ")
        return None
    cancel_trigger_orders(exchange, symbol, dry_run)
    side = "long" if action == "CLOSE_LONG" else "short"
    result = close_position(exchange, symbol, side, dry_run)
    if result:
        result["coin"] = coin
    return result


def _exec_open(exchange, coin, action, position_usdt, price, dry_run):
    """1銘柄のオープン処理 (並列実行用)."""
    symbol = coin_to_exchange_symbol(coin, exchange)
    if not symbol:
        log(f"  {coin} → Bitget未対応 - スキップ")
        return None
    if price <= 0:
        log(f"  {coin} → 価格不明 - スキップ")
        return None
    side = "long" if action == "NEW_LONG" else "short"
    result = open_position(exchange, symbol, side, position_usdt, price, dry_run)
    if result:
        result["coin"] = coin
        if "error" not in result and TRADE_STOP_LOSS_PCT > 0:
            fill_price = result.get("price", price)
            contracts = result.get("amount", 0)
            if contracts > 0 and fill_price > 0:
                place_stop_loss(exchange, symbol, side, fill_price, contracts, dry_run)
    return result


def sync_positions(
    exchange: ccxt.bitget,
    new_longs: list[str],
    new_shorts: list[str],
    changes: dict[str, str],
    prices: dict[str, float],
    dry_run: bool,
) -> list[dict]:
    """推奨ポジションと実際のポジションを同期 (並列実行).

    1. CLOSEアクション → 全銘柄同時に決済
    2. NEWアクション → 全銘柄同時に新規ポジション
    3. KEEPアクション → 何もしない
    """
    trades = []
    position_usdt = calculate_position_usdt(
        TRADE_TOTAL_CAPITAL_USDT, PORTFOLIO_K, TRADE_LEVERAGE,
    )

    log(f"1ポジション: {position_usdt:.1f} USDT (資金{TRADE_TOTAL_CAPITAL_USDT} / {2*PORTFOLIO_K}ポジション × {TRADE_LEVERAGE}xレバ)")

    # Step 1: クローズ (全銘柄同時)
    close_items = [(c, a) for c, a in changes.items() if a.startswith("CLOSE")]
    if close_items:
        log(f"CLOSE {len(close_items)}銘柄を同時決済中...")
        with ThreadPoolExecutor(max_workers=len(close_items)) as pool:
            futures = {
                pool.submit(_exec_close, exchange, coin, action, dry_run): coin
                for coin, action in close_items
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        trades.append(result)
                        log_trade(result)
                except Exception as e:
                    log(f"  {futures[future]} クローズ例外: {e}")

    # Step 2: 新規ポジション (全銘柄同時)
    new_items = [(c, a) for c, a in changes.items() if a.startswith("NEW")]
    if new_items:
        log(f"NEW {len(new_items)}銘柄を同時発注中...")
        with ThreadPoolExecutor(max_workers=len(new_items)) as pool:
            futures = {
                pool.submit(
                    _exec_open, exchange, coin, action,
                    position_usdt, prices.get(coin, 0), dry_run,
                ): coin
                for coin, action in new_items
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        trades.append(result)
                        log_trade(result)
                except Exception as e:
                    log(f"  {futures[future]} オープン例外: {e}")

    # Step 3: KEEP → ログのみ
    for coin, action in changes.items():
        if action.startswith("KEEP"):
            symbol = coin_to_exchange_symbol(coin, exchange)
            side = "LONG" if "LONG" in action else "SHORT"
            log(f"  KEEP {side} {symbol or coin}")

    return trades


# ============================================================
# ステータス表示
# ============================================================

def cmd_status(exchange: ccxt.bitget):
    """現在のポジション状況を表示."""
    log("===== ポジション状況 =====")

    # Bitget残高
    try:
        balance = exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        log(f"USDT残高: {usdt.get('free', 0):.2f} (利用可能) / {usdt.get('total', 0):.2f} (合計)")
    except Exception as e:
        log(f"残高取得失敗: {e}")

    # Bitgetの実ポジション
    log("")
    log("── Bitget オープンポジション ──")
    try:
        positions = fetch_current_positions(exchange)
        if positions:
            total_pnl = 0
            for sym, pos in sorted(positions.items()):
                pnl = pos["unrealizedPnl"]
                total_pnl += pnl
                pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
                log(f"  {sym:<25s} {pos['side'].upper():<6s} {pos['notional']:>8.2f} USDT  PnL: {pnl_str} USDT  (参入: ${pos['entryPrice']:.4f})")
            log(f"  {'─'*60}")
            pnl_str = f"+{total_pnl:.2f}" if total_pnl >= 0 else f"{total_pnl:.2f}"
            log(f"  合計: {len(positions)} ポジション / 未実現損益: {pnl_str} USDT")
        else:
            log("  ポジションなし")
    except Exception as e:
        log(f"  取得失敗: {e}")

    # 保存済みポジション (advisor/traderの内部状態)
    log("")
    log("── 保存済みポジション (内部状態) ──")
    prev = load_positions(POSITIONS_PATH)
    if prev:
        ts = prev.get("timestamp", "不明")
        log(f"  最終更新: {ts}")
        log(f"  LONG:  {', '.join(prev.get('longs', []))}")
        log(f"  SHORT: {', '.join(prev.get('shorts', []))}")
    else:
        log("  なし (初回実行前)")

    log("=" * 50)


# ============================================================
# 全ポジション決済
# ============================================================

def cmd_close_all(exchange: ccxt.bitget, dry_run: bool):
    """Bitget上の全ポジションを決済し、内部状態もクリア."""
    mode = "DRY-RUN" if dry_run else "LIVE"
    log(f"===== 全ポジション決済 ({mode}) =====")

    positions = fetch_current_positions(exchange)
    if not positions:
        log("オープンポジションなし - 何もしません")
        # 内部状態もクリア
        if os.path.exists(POSITIONS_PATH):
            os.remove(POSITIONS_PATH)
            log(f"内部ポジションファイル削除: {POSITIONS_PATH}")
        return

    log(f"{len(positions)} ポジションを決済します:")
    n_success = 0
    n_fail = 0

    for sym, pos in positions.items():
        # SLトリガー注文をキャンセルしてから決済
        cancel_trigger_orders(exchange, sym, dry_run)
        result = close_position(exchange, sym, pos["side"], dry_run)
        if result:
            log_trade(result)
            if "error" in result:
                n_fail += 1
            else:
                n_success += 1

    log(f"決済完了: {n_success} 成功, {n_fail} 失敗")

    # 内部状態クリア
    if not dry_run and n_fail == 0:
        if os.path.exists(POSITIONS_PATH):
            os.remove(POSITIONS_PATH)
            log(f"内部ポジションファイル削除: {POSITIONS_PATH}")

    # 決済後の残高
    if not dry_run:
        try:
            balance = exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            log(f"残高: {usdt.get('free', 0):.2f} USDT (利用可能) / {usdt.get('total', 0):.2f} USDT (合計)")
        except Exception:
            pass


# ============================================================
# メインサイクル
# ============================================================

def run_once(dry_run: bool, exchange: ccxt.bitget) -> bool:
    """1回の推論→執行サイクルを実行.

    Returns:
        True=成功, False=失敗
    """
    mode = "DRY-RUN" if dry_run else "LIVE"
    log(f"===== LSTM Auto Trader ({mode}) =====")
    log(f"資金: {TRADE_TOTAL_CAPITAL_USDT} USDT | レバレッジ: {TRADE_LEVERAGE}x | k={PORTFOLIO_K}")

    # --- 0. 未約定注文クリーンアップ (クラッシュ復帰対策) ---
    n_cancelled = cancel_all_open_orders(exchange, dry_run)
    if n_cancelled > 0:
        log(f"未約定注文クリーンアップ: {n_cancelled}件キャンセル")

    # --- A. モデル読込 ---
    model_path = os.path.join(MODELS_DIR, "4h")
    if not os.path.exists(os.path.join(model_path, "config.json")):
        log("エラー: モデルが見つかりません。先に advisor.py --retrain を実行してください")
        return False

    log("モデル読込中...")
    models, config = load_ensemble(model_path)
    train_mean = config["train_mean"]
    train_std = config["train_std"]
    saved_tickers = config.get("tickers_binance")
    log(f"モデル: SP{config.get('sp_id', 3)} (units={config['best_units']}, ensemble×{len(models)})")
    if saved_tickers:
        log(f"銘柄リスト: {len(saved_tickers)} 銘柄 (モデル保存時)")

    # --- B. 最新データ取得 ---
    log("最新4hデータ取得中...")
    price_df = fetch_recent_4h(tickers=saved_tickers)

    # --- C. 特徴量生成 ---
    log("特徴量生成中...")
    latest_date, coin_samples, current_prices = build_current_features(
        price_df, train_mean, train_std,
    )
    log(f"{len(coin_samples)} 銘柄の特徴量を生成 (時点: {latest_date})")

    if len(coin_samples) < 2 * PORTFOLIO_K:
        log(f"エラー: 有効銘柄不足 ({len(coin_samples)} < {2*PORTFOLIO_K})")
        return False

    # --- D. 予測 ---
    log("ランク予測中...")
    ranks = ensemble_predict_ranks(models, coin_samples)
    log(f"{len(ranks)} 銘柄の予測完了")

    # --- D2. ボラティリティフィルタ ---
    ranks, excluded = apply_volatility_filter(price_df, ranks)
    if excluded:
        log(f"ボラフィルタ: {len(excluded)}銘柄除外 (Vol>{VOL_FILTER_MULTIPLIER}×中央値): {', '.join(excluded)}")
    else:
        log("ボラフィルタ: 除外なし")

    # メモリ解放
    import tensorflow as tf
    for m in models:
        del m
    tf.keras.backend.clear_session()

    # --- E. 取引所の実ポジションを取得 → 動的リバランス ---
    log("取引所ポジション確認中...")
    try:
        actual_positions = fetch_current_positions(exchange)
    except Exception as e:
        log(f"ポジション取得失敗: {e}")
        actual_positions = {}

    # 取引所の実ポジションをcoin IDに変換
    actual_longs = []
    actual_shorts = []
    for sym, pos in actual_positions.items():
        coin = exchange_symbol_to_coin(sym)
        if pos["side"] == "long":
            actual_longs.append(coin)
        elif pos["side"] == "short":
            actual_shorts.append(coin)

    if actual_longs or actual_shorts:
        log(f"取引所ポジション: LONG={actual_longs}, SHORT={actual_shorts}")

    # --- E2. ストップロスチェック ---
    stopped_coins = []
    if actual_positions and TRADE_STOP_LOSS_PCT > 0:
        log(f"ストップロスチェック (閾値: {TRADE_STOP_LOSS_PCT:.0%})...")
        stopped_coins = check_stop_losses(exchange, actual_positions, dry_run)
        if stopped_coins:
            log(f"ストップロス決済: {stopped_coins}")
            # 決済済みの銘柄をポジションリストから除外
            actual_longs = [c for c in actual_longs if c not in stopped_coins]
            actual_shorts = [c for c in actual_shorts if c not in stopped_coins]
        else:
            log("ストップロス該当なし")

    if actual_longs or actual_shorts:
        prev_longs = actual_longs
        prev_shorts = actual_shorts
    else:
        log("取引所にポジションなし → 全銘柄を新規選定")
        prev_longs = None
        prev_shorts = None

    new_longs, new_shorts, changes = dynamic_rebalance(ranks, prev_longs, prev_shorts)

    # サマリー
    n_new = sum(1 for a in changes.values() if a.startswith("NEW"))
    n_keep = sum(1 for a in changes.values() if a.startswith("KEEP"))
    n_close = sum(1 for a in changes.values() if a.startswith("CLOSE"))
    log(f"リバランス: NEW={n_new}, KEEP={n_keep}, CLOSE={n_close}")

    # --- 発注指示書 ---
    print_order_sheet(exchange, new_longs, new_shorts, changes, current_prices)

    # --- F. ポジション同期 ---
    if n_new > 0 or n_close > 0:
        log("ポジション同期中...")
        trades = sync_positions(
            exchange, new_longs, new_shorts, changes, current_prices, dry_run,
        )
        n_executed = sum(1 for t in trades if "error" not in t)
        n_failed = sum(1 for t in trades if "error" in t)
        log(f"注文完了: {n_executed} 成功, {n_failed} 失敗")
    else:
        log("ポジション変更なし")

    # --- G. SLトリガー確認 (フォールバック) ---
    # KEEP中のポジションにSLトリガーが無い場合に再設定
    if not dry_run and TRADE_STOP_LOSS_PCT > 0:
        try:
            log("SLトリガー確認中...")
            current_positions = fetch_current_positions(exchange)
            ensure_stop_losses(exchange, current_positions, dry_run)
        except Exception as e:
            log(f"SLトリガー確認失敗: {e}")

    # --- H. 残高確認 ---
    if not dry_run:
        try:
            balance = exchange.fetch_balance()
            usdt_free = balance.get("USDT", {}).get("free", 0)
            usdt_total = balance.get("USDT", {}).get("total", 0)
            log(f"残高: {usdt_free:.2f} USDT (利用可能) / {usdt_total:.2f} USDT (合計)")
        except Exception as e:
            log(f"残高取得失敗: {e}")

    # --- H. ポジション保存 (LIVEモードのみ、取引所の実ポジションを確認して保存) ---
    if dry_run:
        log("DRY-RUNのためポジション状態は保存しません")
    else:
        # 取引所の実ポジションを再取得して正確な状態を保存
        try:
            final_positions = fetch_current_positions(exchange)
            final_longs = []
            final_shorts = []
            for sym, pos in final_positions.items():
                coin = exchange_symbol_to_coin(sym)
                if pos["side"] == "long":
                    final_longs.append(coin)
                elif pos["side"] == "short":
                    final_shorts.append(coin)
        except Exception:
            # 取得失敗時は推奨を保存 (次回は取引所から再取得される)
            final_longs = new_longs
            final_shorts = new_shorts

        position_data = {
            "timestamp": latest_date.isoformat(),
            "timeframe": "4h",
            "longs": final_longs,
            "shorts": final_shorts,
            "recommended_longs": new_longs,
            "recommended_shorts": new_shorts,
            "all_ranks": {k: float(v) for k, v in ranks.items()},
        }
        save_positions(POSITIONS_PATH, position_data)
        log(f"ポジション保存: LONG={final_longs}, SHORT={final_shorts}")

    return True


# ============================================================
# インタラクティブコンソール
# ============================================================

def _print_help():
    """待機中に使えるコマンド一覧を表示."""
    print("  コマンド: [s]tatus  [c]lose-all  [q]uit  [h]elp")
    print("  > ", end="", flush=True)


def _stdin_reader(q: queue.Queue):
    """バックグラウンドスレッドでstdinを読み取る (Windows/Linux両対応)."""
    try:
        while True:
            line = sys.stdin.readline()
            if not line:  # EOF
                q.put(None)
                break
            q.put(line.strip().lower())
    except (EOFError, OSError):
        q.put(None)


def _interactive_wait(
    exchange: ccxt.bitget,
    dry_run: bool,
    next_run: datetime,
) -> str | None:
    """次のサイクルまでインタラクティブに待機.

    Returns:
        "quit" → 安全停止
        "close_all" → 全決済して停止
        None → 次のサイクルへ
    """
    global _shutdown

    # stdin読み取りスレッドを起動
    input_q: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(target=_stdin_reader, args=(input_q,), daemon=True)
    reader.start()

    while not _shutdown:
        now = datetime.now(timezone.utc)
        if now >= next_run:
            print()
            return None

        remaining = (next_run - now).total_seconds()

        # キュー確認 (1秒ごと)
        try:
            line = input_q.get(timeout=min(1, max(0.1, remaining)))
        except queue.Empty:
            continue

        if line is None:
            return "quit"

        if not line:
            mins = remaining / 60
            log(f"次回まで {mins:.0f}分")
            print("  > ", end="", flush=True)
            continue

        if line in ("s", "status"):
            cmd_status(exchange)
            print("  > ", end="", flush=True)

        elif line in ("c", "close", "close-all"):
            if dry_run:
                log("DRY-RUNモードのため決済シミュレーションのみ実行します")
                cmd_close_all(exchange, dry_run)
                print("  > ", end="", flush=True)
            else:
                print("  全ポジションを決済して停止しますか? [y/N]: ", end="", flush=True)
                try:
                    confirm = input_q.get(timeout=30)
                except queue.Empty:
                    confirm = ""
                if confirm in ("y", "yes"):
                    return "close_all"
                else:
                    log("キャンセルしました")
                    print("  > ", end="", flush=True)

        elif line in ("q", "quit", "exit"):
            print()
            log("安全停止 (ポジションは保持)")
            return "quit"

        elif line in ("h", "help"):
            print()
            print("  s, status     ポジション・残高・PnLを表示")
            print("  c, close-all  全ポジション決済して停止")
            print("  q, quit       ポジション保持したまま停止")
            print("  h, help       このヘルプ")
            print("  Enter         次回実行までの残り時間を表示")
            print()
            print("  > ", end="", flush=True)

        else:
            log(f"不明なコマンド: {line} (h でヘルプ)")
            print("  > ", end="", flush=True)

    print()
    return "quit"


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LSTM Auto Trader (Bitget Futures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
運用コマンド:
  trader.py --status              ポジション状況確認
  trader.py --close-all           全決済 dry-run
  trader.py --close-all --live    全決済 実行
  trader.py                       1回実行 dry-run
  trader.py --live                1回実行 本番
  trader.py --live --loop         4h自動ループ

Ctrl+C で安全停止 (ポジション保持)
Ctrl+C ×2 で強制停止
""",
    )
    parser.add_argument("--live", action="store_true", help="本番モード (実注文)")
    parser.add_argument("--loop", action="store_true", help="4h毎にループ実行")
    parser.add_argument("--status", action="store_true", help="ポジション状況確認")
    parser.add_argument("--close-all", action="store_true", help="全ポジション決済")
    args = parser.parse_args()

    global _shutdown
    dry_run = not args.live

    # シグナルハンドラ設定
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --status は読取専用なのでdry_runを強制
    if args.status:
        exchange = init_exchange(dry_run=True)
        cmd_status(exchange)
        return

    # --close-all
    if args.close_all:
        exchange = init_exchange(dry_run)
        cmd_close_all(exchange, dry_run)
        return

    # 取引所接続
    exchange = init_exchange(dry_run)

    if args.loop:
        log(f"ループモード開始 ({TRADE_INTERVAL_HOURS}h間隔)")
        if dry_run:
            log("DRY-RUNモード: 注文は送信されません")
        else:
            log("LIVEモード: 実際の注文が送信されます!")

        while not _shutdown:
            # --- 定期再訓練チェック ---
            now_utc = datetime.now(timezone.utc)
            if now_utc.hour == RETRAIN_UTC_HOUR and _should_retrain():
                log(f"定期再訓練を開始 (前回から{RETRAIN_INTERVAL_DAYS}日以上経過)...")
                try:
                    retrain_models("4h")
                    log("再訓練完了 → 新モデルで推論を開始")
                except Exception as e:
                    log(f"再訓練エラー (既存モデルで続行): {e}")

            try:
                run_once(dry_run, exchange)
            except Exception as e:
                log(f"サイクルエラー: {e}")

            if _shutdown:
                break

            # 次の4h足に合わせた実行時刻を計算
            now = datetime.now(timezone.utc)
            next_4h_hour = ((now.hour // TRADE_INTERVAL_HOURS) + 1) * TRADE_INTERVAL_HOURS
            if next_4h_hour >= 24:
                next_run = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
            else:
                next_run = now.replace(hour=next_4h_hour, minute=0, second=5, microsecond=0)

            if next_run <= now:
                next_run += timedelta(hours=TRADE_INTERVAL_HOURS)

            next_run_jst = next_run.astimezone(JST)
            log(f"次回実行: {next_run_jst.strftime('%H:%M JST')}")
            log("")
            _print_help()

            # インタラクティブ待機ループ
            action = _interactive_wait(exchange, dry_run, next_run)

            if action == "quit":
                _shutdown = True
            elif action == "close_all":
                log("全ポジション決済...")
                cmd_close_all(exchange, dry_run)
                _shutdown = True

        log("")
        log("安全停止完了 (ポジションは保持されています)")
        log("再開: uv run python trader.py --live --loop")
        log("全決済: uv run python trader.py --close-all --live")
    else:
        # 1回実行
        if dry_run:
            log("DRY-RUNモード: 注文は送信されません (--live で本番実行)")
        run_once(dry_run, exchange)


if __name__ == "__main__":
    main()
