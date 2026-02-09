"""LSTMアンサンブルモデル.

論文の方法論 (Section 3.6.2):
- 単一LSTMレイヤー (ユニット数: {5, 10, 15, 20} から検証セットで選択)
- ドロップアウト 0.1
- Dense(5, ReLU) → Dense(1, sigmoid)
- Adam(lr=0.002), binary_crossentropy
- Early stopping: patience=4, min_delta=1e-4, monitor=val_loss
- 10個のアンサンブル (異なる乱数シード)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    ENSEMBLE_SIZE,
    LEARNING_RATE,
    LSTM_UNITS_GRID,
    MAX_EPOCHS,
    SEQUENCE_LENGTH,
)


def build_lstm_model(n_units: int) -> keras.Model:
    """論文のLSTMアーキテクチャを構築."""
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQUENCE_LENGTH, 1)),
        keras.layers.LSTM(n_units, dropout=DROPOUT_RATE),
        keras.layers.Dense(5, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def select_best_units(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> int:
    """検証セットの分類精度でベストなLSTMユニット数を選択."""
    best_units = LSTM_UNITS_GRID[0]
    best_acc = 0.0

    for n_units in LSTM_UNITS_GRID:
        print(f"    ユニット数 {n_units} を検証中...")
        tf.random.set_seed(42)
        np.random.seed(42)

        model = build_lstm_model(n_units)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            restore_best_weights=True,
        )
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0,
        )
        _, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"      検証精度: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_units = n_units

        del model
        keras.backend.clear_session()

    print(f"    → 選択されたユニット数: {best_units} (検証精度: {best_acc:.4f})")
    return best_units


def train_ensemble(
    n_units: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> list[keras.Model]:
    """異なるシードで ENSEMBLE_SIZE 個のモデルを訓練."""
    models = []
    for i in range(ENSEMBLE_SIZE):
        seed = i * 42 + 7
        print(f"    アンサンブルモデル {i+1}/{ENSEMBLE_SIZE} (seed={seed})")
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = build_lstm_model(n_units)
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            restore_best_weights=True,
        )
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0,
        )
        _, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"      検証精度: {val_acc:.4f}")
        models.append(model)

    return models


def ensemble_predict_ranks(
    models: list[keras.Model],
    coin_samples: dict[str, np.ndarray],
) -> dict[str, float]:
    """アンサンブルモデルで銘柄ランクを予測.

    論文の方法論:
    1. 各モデルの予測確率で銘柄をランク付け (降順)
    2. 10モデルのランクを平均
    3. 平均ランクが最終ランキング

    Args:
        models: アンサンブルの各モデル
        coin_samples: {coin_id: X_sample (shape=(90,1))} の辞書

    Returns:
        {coin_id: average_rank} の辞書 (ランクが高いほど「良い」予測)
    """
    coin_ids = list(coin_samples.keys())
    if not coin_ids:
        return {}

    # 全銘柄のサンプルをバッチにまとめる
    X_batch = np.array([coin_samples[cid] for cid in coin_ids])

    all_ranks = []
    for model in models:
        # 予測確率
        probs = model.predict(X_batch, verbose=0).flatten()
        # 降順ランク (確率が高い = ランク1 = 良い)
        order = np.argsort(-probs)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        all_ranks.append(ranks)

    # ランクを平均
    avg_ranks = np.mean(all_ranks, axis=0)

    return {cid: rank for cid, rank in zip(coin_ids, avg_ranks)}
