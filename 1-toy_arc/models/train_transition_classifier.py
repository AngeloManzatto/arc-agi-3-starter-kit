"""
Created on Sun Mar 22 07:19:20 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.transition_classifier import TransitionOutcomeClassifier


TARGET_NAMES = ["blocked", "moved", "hud_changed", "success"]

###############################################################################
# Dataset
###############################################################################

def make_dataset(
    npz_path: str,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    seed: int = 123,
):
    data = np.load(npz_path)

    g = data["g"].astype(np.float32)
    action = data["action"].astype(np.int32)

    y = np.stack(
        [
            data["blocked"].astype(np.float32),
            data["moved"].astype(np.float32),
            data["hud_changed"].astype(np.float32),
            data["success"].astype(np.float32),
        ],
        axis=1,
    )

    n = len(g)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_val = int(n * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = tf.data.Dataset.from_tensor_slices(
        ((g[train_idx], action[train_idx]), y[train_idx])
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        ((g[val_idx], action[val_idx]), y[val_idx])
    )

    train_ds = train_ds.shuffle(min(len(train_idx), 10000), seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

###############################################################################
# Execute Training
###############################################################################

def main():
    data_path = "artifacts/deltas/trigger_door_deltas.npz"
    out_dir = Path("artifacts/transition_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = make_dataset(data_path)

    model = TransitionOutcomeClassifier()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks,
        verbose=1,
    )

    model.save_weights(str(out_dir / "last.weights.h5"))
    print("Saved classifier artifacts to:", out_dir)


if __name__ == "__main__":
    main()