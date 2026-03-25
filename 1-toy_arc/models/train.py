"""
Created on Sat Mar 21 08:23:37 2026

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

from models.world_model import WorldModel

###############################################################################
# Dataset creation
###############################################################################

def make_dataset(
    npz_path: str,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    seed: int = 123,
):
    
    # Load replay
    data = np.load(npz_path)
    
    # Extract data from replay
    obs      = data["obs"].astype(np.float32)
    action   = data["action"].astype(np.int32)
    next_obs = data["next_obs"].astype(np.float32)
    
    # Shuffle replay indices for train
    n = len(obs)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    
    # Select and shuffle indices for validation
    n_val = int(n * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    # Create train and validation datasets from sliced tensors
    train_ds = tf.data.Dataset.from_tensor_slices(
        ((obs[train_idx], action[train_idx]), next_obs[train_idx])
    )
    
    val_ds = tf.data.Dataset.from_tensor_slices(
        ((obs[val_idx], action[val_idx]), next_obs[val_idx])
    )

    # Dataset optimization loading 
    train_ds = train_ds.shuffle(min(len(train_idx), 10000), seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

###############################################################################
# Training model pipeline
###############################################################################

def main():
    data_path = "data/trigger_door_random.npz"
    out_dir = Path("artifacts/world_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = make_dataset(data_path)

    model = WorldModel()

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

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks,
        verbose=1,
    )

    model.save_weights(str(out_dir / "last.weights.h5"))

    print("Training complete.")
    print("Best weights saved to:", out_dir / "best.weights.h5")
    print("Last weights saved to:", out_dir / "last.weights.h5")

if __name__ == "__main__":
    main()