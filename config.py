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
        "train": ("2018-07-16", "2019-11-27"),
        "val":   ("2019-11-28", "2020-04-25"),
        "test":  ("2020-04-26", "2020-09-22"),
    },
    {
        "id": 2,
        "train": ("2018-12-13", "2020-04-25"),
        "val":   ("2020-04-26", "2020-09-22"),
        "test":  ("2020-09-23", "2021-02-19"),
    },
    {
        "id": 3,
        "train": ("2019-05-12", "2020-09-22"),
        "val":   ("2020-09-23", "2021-02-19"),
        "test":  ("2021-02-20", "2021-07-19"),
    },
    {
        "id": 4,
        "train": ("2019-10-09", "2021-02-19"),
        "val":   ("2021-02-20", "2021-07-19"),
        "test":  ("2021-07-20", "2021-12-16"),
    },
    {
        "id": 5,
        "train": ("2020-03-07", "2021-07-19"),
        "val":   ("2021-07-20", "2021-12-16"),
        "test":  ("2021-12-17", "2022-05-15"),
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
