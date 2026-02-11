"""論文 Jaquart et al. (2022) の全パラメータ定義."""

import os

# --- パス ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "cache")

# --- 銘柄選定 ---
TOP_K_COINS = 100

# ステーブルコイン・データ問題で除外する銘柄 (論文 Table B.1)
EXCLUDED_COINS = {
    # ステーブルコイン
    "tether", "usd-coin", "binance-usd", "dai", "terrausd",
    "true-usd", "husd", "sai", "frax", "magic-internet-money",
    "musd", "susd", "usdn",
    # データ問題
    "terra-luna", "kncl", "compound-dai", "ln", "solve",
    "veri", "vee", "jasmy", "msol", "maid",
}

# --- 時系列パラメータ ---
SEQUENCE_LENGTH = 90   # RNN入力の時系列長 (日)
LOOKBACK_BUFFER = 90   # Study Period開始前に必要なデータ日数

# --- Study Period 分割 ---
TRAIN_DAYS = 500
VAL_DAYS = 150
TEST_DAYS = 150

STUDY_PERIODS = [
    {
        "id": 1,
        "train": ("2023-02-06", "2024-06-19"),
        "val":   ("2024-06-20", "2024-11-16"),
        "test":  ("2024-11-17", "2025-04-15"),
    },
    {
        "id": 2,
        "train": ("2023-07-06", "2024-11-16"),
        "val":   ("2024-11-17", "2025-04-15"),
        "test":  ("2025-04-16", "2025-09-12"),
    },
    {
        "id": 3,
        "train": ("2023-12-03", "2025-04-15"),
        "val":   ("2025-04-16", "2025-09-12"),
        "test":  ("2025-09-13", "2026-02-09"),
    },
]

# --- モデルパラメータ ---
LSTM_UNITS_GRID = [5, 10, 15, 20]
ENSEMBLE_SIZE = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.002
MAX_EPOCHS = 25
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA = 1e-4
DROPOUT_RATE = 0.1

# --- ポートフォリオ ---
PORTFOLIO_K = 5
TRANSACTION_COST_BPS = 15  # 片道 (half-turn) basis points

# --- 4時間足実験 ---
CANDLE_INTERVAL = "4h"
PERIODS_PER_YEAR_DAILY = 365          # 日足: 暗号資産は365日取引
PERIODS_PER_YEAR_4H = 365 * 6        # 4h足: 1日6本 × 365日 = 2190

# --- 自動トレード ---
TRADE_TOTAL_CAPITAL_USDT = 40    # 総資金 (USDT)
TRADE_LEVERAGE = 2                  # レバレッジ倍率
TRADE_INTERVAL_HOURS = 4            # 実行間隔
TRADE_LIMIT_OFFSET_PCT = 0.05      # 指値オフセット (%) - 現在価格から何%有利な位置に指値
TRADE_LIMIT_TIMEOUT_SEC = 30        # 指値タイムアウト (秒) - 未約定なら成行にフォールバック
TRADE_STOP_LOSS_PCT = 0             # ストップロス無効 (ボラフィルタで代替)

# --- ボラティリティフィルタ ---
VOL_FILTER_MULTIPLIER = 2.0         # 中央値の何倍でフィルタ (None=無効)
VOL_FILTER_LOOKBACK = 18            # ボラ計算期間 (4h足本数, 18=3日分)
