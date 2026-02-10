# LSTM Crypto Trading Strategy

Jaquart et al. (2022) "[Machine Learning for Cryptocurrency Market Prediction and Trading](https://doi.org/10.1016/j.jbef.2022.100723)" の LSTM ベース統計的裁定取引戦略の再現実装。

## セットアップ

```bash
# Python 3.13+ / uv 必須
uv sync

# MEXC 自動トレードを使う場合
cp .env.example .env
# .env に MEXC_API_KEY / MEXC_API_SECRET を設定
```

### 依存ライブラリ

| ライブラリ | 用途 |
| --- | --- |
| tensorflow | LSTM モデル |
| numpy / pandas / scikit-learn | データ処理 |
| yfinance | 価格データ取得 (日足・4h足) |
| ccxt | MEXC Futures API |
| python-dotenv | APIキー管理 |
| matplotlib | グラフ出力 |

## 戦略概要

1. 時価総額上位100暗号資産の過去90期間のリターン系列を特徴量に使用
2. LSTM アンサンブル (10モデル) でクロスセクショナルランク予測
3. 上位5銘柄をロング、下位5銘柄をショート (マーケットニュートラル)
4. 動的リバランス: ランクが大きく変動した銘柄のみ入替 (回転率 ~60% 削減)

## 使い方

### バックテスト

```bash
# 日足バックテスト (3 Study Period)
uv run python main.py

# 4時間足バックテスト
uv run python main_4h.py

# 4時間足 動的リバランス実験 (0bps / 5bps / 15bps)
uv run python main_4h_dynamic.py
```

結果は `output/`, `output_4h/`, `output_4h_dynamic/` に出力される。

### 手動トレード推奨 (advisor.py)

```bash
# 4h足で売買推奨を表示 (保存済みモデル使用)
uv run python advisor.py

# 日足で推奨
uv run python advisor.py --timeframe daily

# モデル再訓練してから推奨
uv run python advisor.py --retrain
```

### 自動トレード (trader.py)

MEXC Futures (USDT-M 永久先物) で自動売買を行う。

```bash
# dry-run (注文は送信されない)
uv run python trader.py

# 本番1回実行
uv run python trader.py --live

# 4h毎の自動ループ
uv run python trader.py --live --loop

# ポジション状況確認
uv run python trader.py --status

# 全ポジション決済
uv run python trader.py --close-all --live
```

#### ループ中のインタラクティブコマンド

`--loop` 実行中は待機時間にコマンド入力が可能:

| コマンド | 動作 |
| --- | --- |
| `s` / `status` | 残高・ポジション・PnL 表示 |
| `c` / `close-all` | 全ポジション決済して停止 |
| `q` / `quit` | ポジション保持したまま停止 |
| `Enter` | 次回実行までの残り時間 |
| `Ctrl+C` | 安全停止 (2回で強制停止) |

#### トレード設定 (config.py)

```python
TRADE_TOTAL_CAPITAL_USDT = 10   # 総資金 (USDT)
TRADE_LEVERAGE = 1               # レバレッジ倍率
TRADE_INTERVAL_HOURS = 4         # 実行間隔
TRADE_LIMIT_OFFSET_PCT = 0.05   # 指値オフセット (%)
TRADE_LIMIT_TIMEOUT_SEC = 30    # 指値タイムアウト (秒)
```

- 指値注文のみ使用 (maker手数料 0%)
- 約定しない場合はオフセットを縮小しながら最大4回リトライ
- 最小注文額に満たない銘柄は自動スキップ

## プロジェクト構成

```text
.
├── config.py                 # 全パラメータ定義
├── main.py                   # 日足バックテスト
├── main_4h.py                # 4h足バックテスト
├── main_4h_dynamic.py        # 4h動的リバランス実験
├── advisor.py                # 手動トレード推奨
├── trader.py                 # MEXC自動トレードBot
├── data/
│   ├── collector.py          # 日足データ取得 (yfinance)
│   ├── collector_4h.py       # 4h足データ取得 (Binance API)
│   └── preprocessor.py       # 特徴量・ターゲット生成
├── models/
│   └── lstm_model.py         # LSTMアンサンブル (訓練/予測/保存/読込)
├── backtest/
│   └── engine.py             # バックテストエンジン
├── cache/                    # データキャッシュ・学習済みモデル
│   ├── models/4h/            # 4h足モデル
│   ├── models/daily/         # 日足モデル
│   └── positions_4h.json     # 現在ポジション状態
└── output*/                  # バックテスト結果
```

## バックテスト結果

| 設定 | Sharpe | 精度 | 総リターン |
| --- | --- | --- | --- |
| 日足 | 3.57 | 57.7% | +13,044% |
| 4h 動的リバランス (0bps) | 2.73 | - | +820% |
| 4h 動的リバランス (5bps) | 1.59 | - | +213% |
| 4h フルターンオーバー (15bps) | -5.48 | - | - |

## 参考文献

Jaquart, P., Dann, D., & Weinhardt, C. (2022). Machine learning for cryptocurrency market prediction and trading. *Journal of Behavioral and Experimental Finance*, 36, 100723.
