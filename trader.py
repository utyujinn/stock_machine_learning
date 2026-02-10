"""LSTM Auto Trader - MEXC Futures 自動売買Bot.

advisor.py の推論パイプラインを再利用し、
MEXC Futures (USDT-M永久先物) で自動ポジション管理を行う。

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
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(line_buffering=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import ccxt
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR,
    PORTFOLIO_K,
    TRADE_INTERVAL_HOURS,
    TRADE_LEVERAGE,
    TRADE_LIMIT_OFFSET_PCT,
    TRADE_LIMIT_TIMEOUT_SEC,
    TRADE_TOTAL_CAPITAL_USDT,
)

from advisor import (
    build_current_features,
    dynamic_rebalance,
    fetch_recent_4h,
    load_positions,
    save_positions,
)
from models.lstm_model import ensemble_predict_ranks, load_ensemble

# --- 定数 ---
MODELS_DIR = os.path.join(DATA_DIR, "models")
TRADE_LOG_PATH = os.path.join(DATA_DIR, "trade_log.jsonl")
POSITIONS_PATH = os.path.join(DATA_DIR, "positions_4h.json")

# グレースフル停止用フラグ
_shutdown = False


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
    """タイムスタンプ付きログ出力."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
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

def init_exchange(dry_run: bool = True) -> ccxt.mexc:
    """MEXC Futures クライアントを初期化."""
    load_dotenv()
    api_key = os.getenv("MEXC_API_KEY")
    api_secret = os.getenv("MEXC_API_SECRET")

    if not api_key or not api_secret:
        log("エラー: .env に MEXC_API_KEY / MEXC_API_SECRET を設定してください")
        sys.exit(1)

    exchange = ccxt.mexc({
        "apiKey": api_key,
        "secret": api_secret,
        "options": {"defaultType": "swap"},
    })

    if dry_run:
        exchange.options["createOrder"] = "disabled"

    exchange.load_markets()
    log(f"MEXC Futures 接続完了 (マーケット数: {len(exchange.markets)})")
    return exchange


# Binance → MEXC Futures シンボル名のマッピング (リブランド・デノミ対応)
_SYMBOL_MAP = {
    "FIL": "FILECOIN",    # Filecoin
    "FTM": "S",           # Fantom → Sonic
    "MKR": "SKY",         # Maker → Sky
    "BONK": "1000BONK",   # 1000倍デノミ
}

# 逆マッピング (MEXC → Binance)
_SYMBOL_MAP_REV = {v: k for k, v in _SYMBOL_MAP.items()}


def coin_to_mexc_symbol(coin_id: str) -> str | None:
    """advisor銘柄名 → MEXC Futuresシンボルに変換.

    BTC-USD → BTC/USDT:USDT
    FIL-USD → FILECOIN/USDT:USDT (マッピング適用)
    """
    base = coin_id.replace("-USD", "")
    base = _SYMBOL_MAP.get(base, base)
    symbol = f"{base}/USDT:USDT"
    return symbol


def mexc_symbol_to_coin(symbol: str) -> str:
    """MEXC Futuresシンボル → advisor銘柄名に変換.

    BTC/USDT:USDT → BTC-USD
    FILECOIN/USDT:USDT → FIL-USD (逆マッピング適用)
    """
    base = symbol.split("/")[0]
    base = _SYMBOL_MAP_REV.get(base, base)
    return f"{base}-USD"


def check_symbol_available(exchange: ccxt.mexc, symbol: str) -> bool:
    """シンボルがMEXC Futuresに存在するか確認."""
    return symbol in exchange.markets


# ============================================================
# ポジション管理
# ============================================================

def fetch_current_positions(exchange: ccxt.mexc) -> dict[str, dict]:
    """MEXC Futuresの現在のオープンポジションを取得.

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


def calculate_position_usdt(capital: float, k: int, leverage: int) -> float:
    """1ポジションあたりのUSDT額を計算.

    10ポジション (LONG 5 + SHORT 5) に均等配分。
    """
    return (capital / (2 * k)) * leverage


def _wait_for_fill(
    exchange: ccxt.mexc,
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
    0.0,                               # 4回目: 現在価格ちょうど (maker手数料0%)
]


def _execute_limit_with_retry(
    exchange: ccxt.mexc,
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
        amount: 注文数量
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
            log(f"    注文作成失敗: {e}")
            continue

        filled = _wait_for_fill(exchange, order["id"], symbol)

        if filled:
            fill_price = filled.get("average", limit_price)
            log(f"    約定 @ ${fill_price:.4f} ✓ (試行{attempt+1})")
            return {
                "price": fill_price,
                "order_id": filled.get("id"),
                "attempt": attempt + 1,
                "type": "limit",
            }

        log(f"    未約定 → キャンセル")

    return {"error": f"全{len(_LIMIT_OFFSETS)}回の指値が未約定"}


def close_position(
    exchange: ccxt.mexc,
    symbol: str,
    side: str,
    dry_run: bool,
) -> dict | None:
    """既存ポジションを決済 (指値リトライ、成行は使わない)."""
    try:
        if dry_run:
            log(f"  [DRY-RUN] CLOSE {side.upper()} {symbol}")
            return {"action": "close", "symbol": symbol, "side": side, "dry_run": True}

        close_side = "sell" if side == "long" else "buy"
        is_buy = close_side == "buy"
        params = {"reduceOnly": True}

        # 現在のポジションサイズを取得
        positions = exchange.fetch_positions([symbol])
        pos_size = 0
        for p in positions:
            if p["symbol"] == symbol and float(p.get("contracts", 0) or 0) > 0:
                pos_size = float(p["contracts"])
                break

        if pos_size == 0:
            log(f"  {symbol} ポジションなし - スキップ")
            return None

        # 現在価格取得
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker.get("last", 0)
        if current_price <= 0:
            return {"action": "close", "symbol": symbol, "error": "価格取得失敗"}

        log(f"  CLOSE {side.upper()} {symbol} (現在${current_price:.4f})")
        result = _execute_limit_with_retry(
            exchange, symbol, close_side, pos_size, current_price,
            is_buy=is_buy, params=params, label=f"CLOSE {side.upper()} {symbol}",
        )
        result["action"] = "close"
        result["symbol"] = symbol
        result["side"] = side
        result["size"] = pos_size
        return result

    except Exception as e:
        log(f"  CLOSE {symbol} 失敗: {e}")
        return {"action": "close", "symbol": symbol, "error": str(e)}


def _get_min_order_usdt(exchange: ccxt.mexc, symbol: str, price: float) -> float:
    """シンボルの最小注文額 (USDT) を取得."""
    m = exchange.markets.get(symbol, {})
    cs = m.get("contractSize", 1) or 1
    min_amt = (m.get("limits", {}).get("amount", {}).get("min", 0)) or 0
    return min_amt * cs * price


def open_position(
    exchange: ccxt.mexc,
    symbol: str,
    side: str,
    usdt_amount: float,
    price: float,
    dry_run: bool,
) -> dict | None:
    """新規ポジションを開設 (指値リトライ、成行は使わない)."""
    try:
        # レバレッジ設定
        try:
            exchange.set_leverage(TRADE_LEVERAGE, symbol)
        except Exception:
            pass

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

        amount = usdt_amount / price
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
            is_buy=is_buy, label=label,
        )
        result["action"] = "open"
        result["symbol"] = symbol
        result["side"] = side
        result["usdt"] = usdt_amount
        result["amount"] = amount
        return result

    except Exception as e:
        log(f"  NEW {side.upper()} {symbol} 失敗: {e}")
        return {"action": "open", "symbol": symbol, "side": side, "error": str(e)}


# ============================================================
# ポジション同期
# ============================================================

def sync_positions(
    exchange: ccxt.mexc,
    new_longs: list[str],
    new_shorts: list[str],
    changes: dict[str, str],
    prices: dict[str, float],
    dry_run: bool,
) -> list[dict]:
    """推奨ポジションと実際のポジションを同期.

    1. CLOSEアクション → 決済
    2. NEWアクション → 新規ポジション
    3. KEEPアクション → 何もしない
    """
    trades = []
    position_usdt = calculate_position_usdt(
        TRADE_TOTAL_CAPITAL_USDT, PORTFOLIO_K, TRADE_LEVERAGE,
    )

    log(f"1ポジション: {position_usdt:.1f} USDT (資金{TRADE_TOTAL_CAPITAL_USDT} / {2*PORTFOLIO_K}ポジション × {TRADE_LEVERAGE}xレバ)")

    # Step 1: クローズ
    for coin, action in changes.items():
        if not action.startswith("CLOSE"):
            continue
        symbol = coin_to_mexc_symbol(coin)
        if not symbol or not check_symbol_available(exchange, symbol):
            log(f"  {coin} → MEXC未対応 - スキップ")
            continue

        side = "long" if action == "CLOSE_LONG" else "short"
        result = close_position(exchange, symbol, side, dry_run)
        if result:
            result["coin"] = coin
            trades.append(result)
            log_trade(result)

    # Step 2: 新規ポジション
    for coin, action in changes.items():
        if not action.startswith("NEW"):
            continue
        symbol = coin_to_mexc_symbol(coin)
        if not symbol or not check_symbol_available(exchange, symbol):
            log(f"  {coin} → MEXC未対応 - スキップ")
            continue

        price = prices.get(coin, 0)
        if price <= 0:
            log(f"  {coin} → 価格不明 - スキップ")
            continue

        side = "long" if action == "NEW_LONG" else "short"
        result = open_position(exchange, symbol, side, position_usdt, price, dry_run)
        if result:
            result["coin"] = coin
            trades.append(result)
            log_trade(result)

    # Step 3: KEEP → ログのみ
    for coin, action in changes.items():
        if action.startswith("KEEP"):
            symbol = coin_to_mexc_symbol(coin)
            side = "LONG" if "LONG" in action else "SHORT"
            log(f"  KEEP {side} {symbol or coin}")

    return trades


# ============================================================
# ステータス表示
# ============================================================

def cmd_status(exchange: ccxt.mexc):
    """現在のポジション状況を表示."""
    log("===== ポジション状況 =====")

    # MEXC残高
    try:
        balance = exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        log(f"USDT残高: {usdt.get('free', 0):.2f} (利用可能) / {usdt.get('total', 0):.2f} (合計)")
    except Exception as e:
        log(f"残高取得失敗: {e}")

    # MEXCの実ポジション
    log("")
    log("── MEXC オープンポジション ──")
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

def cmd_close_all(exchange: ccxt.mexc, dry_run: bool):
    """MEXC上の全ポジションを決済し、内部状態もクリア."""
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

def run_once(dry_run: bool, exchange: ccxt.mexc) -> bool:
    """1回の推論→執行サイクルを実行.

    Returns:
        True=成功, False=失敗
    """
    mode = "DRY-RUN" if dry_run else "LIVE"
    log(f"===== LSTM Auto Trader ({mode}) =====")
    log(f"資金: {TRADE_TOTAL_CAPITAL_USDT} USDT | レバレッジ: {TRADE_LEVERAGE}x | k={PORTFOLIO_K}")

    # --- A. モデル読込 ---
    model_path = os.path.join(MODELS_DIR, "4h")
    if not os.path.exists(os.path.join(model_path, "config.json")):
        log("エラー: モデルが見つかりません。先に advisor.py --retrain を実行してください")
        return False

    log("モデル読込中...")
    models, config = load_ensemble(model_path)
    train_mean = config["train_mean"]
    train_std = config["train_std"]
    log(f"モデル: SP{config.get('sp_id', 3)} (units={config['best_units']}, ensemble×{len(models)})")

    # --- B. 最新データ取得 ---
    log("最新4hデータ取得中...")
    price_df = fetch_recent_4h()

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

    # メモリ解放
    import tensorflow as tf
    for m in models:
        del m
    tf.keras.backend.clear_session()

    # --- E. 動的リバランス ---
    prev = load_positions(POSITIONS_PATH)
    prev_longs = prev["longs"] if prev else None
    prev_shorts = prev["shorts"] if prev else None

    if prev:
        log(f"前回ポジション: {prev.get('timestamp', '不明')}")
    else:
        log("初回実行 (前回ポジションなし)")

    new_longs, new_shorts, changes = dynamic_rebalance(ranks, prev_longs, prev_shorts)

    # サマリー
    n_new = sum(1 for a in changes.values() if a.startswith("NEW"))
    n_keep = sum(1 for a in changes.values() if a.startswith("KEEP"))
    n_close = sum(1 for a in changes.values() if a.startswith("CLOSE"))
    log(f"リバランス: NEW={n_new}, KEEP={n_keep}, CLOSE={n_close}")

    # ロング・ショート表示
    log(f"LONG: {', '.join(new_longs)}")
    log(f"SHORT: {', '.join(new_shorts)}")

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

    # --- G. 残高確認 ---
    if not dry_run:
        try:
            balance = exchange.fetch_balance()
            usdt_free = balance.get("USDT", {}).get("free", 0)
            usdt_total = balance.get("USDT", {}).get("total", 0)
            log(f"残高: {usdt_free:.2f} USDT (利用可能) / {usdt_total:.2f} USDT (合計)")
        except Exception as e:
            log(f"残高取得失敗: {e}")

    # --- H. ポジション保存 (LIVEモードのみ) ---
    if dry_run:
        log("DRY-RUNのためポジション状態は保存しません")
    else:
        position_data = {
            "timestamp": latest_date.isoformat(),
            "timeframe": "4h",
            "longs": new_longs,
            "shorts": new_shorts,
            "all_ranks": {k: float(v) for k, v in ranks.items()},
        }
        save_positions(POSITIONS_PATH, position_data)
        log(f"ポジション保存: {POSITIONS_PATH}")

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
    exchange: ccxt.mexc,
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
        description="LSTM Auto Trader (MEXC Futures)",
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
            log("⚠ DRY-RUNモード: 注文は送信されません")
        else:
            log("⚠ LIVEモード: 実際の注文が送信されます!")

        while not _shutdown:
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
                next_run = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
            else:
                next_run = now.replace(hour=next_4h_hour, minute=5, second=0, microsecond=0)

            if next_run <= now:
                next_run += timedelta(hours=TRADE_INTERVAL_HOURS)

            log(f"次回実行: {next_run.strftime('%H:%M UTC')}")
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
            log("⚠ DRY-RUNモード: 注文は送信されません (--live で本番実行)")
        run_once(dry_run, exchange)


if __name__ == "__main__":
    main()
