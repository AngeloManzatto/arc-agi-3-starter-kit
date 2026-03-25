"""
Created on Sat Mar 21 08:20:22 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

###############################################################################
# libraries
###############################################################################

class WorldModel(keras.Model):
    def __init__(
        self,
        obs_shape=(5, 5, 10),
        num_actions: int = 4,
        action_dim: int = 8,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.obs_shape_ = obs_shape
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Observation encoder
        self.obs_encoder = keras.Sequential(
            [
                keras.layers.Input(shape=obs_shape),
                keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
                keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(hidden_dim, activation="relu"),
                keras.layers.Dense(latent_dim, activation=None),
            ],
            name="obs_encoder",
        )

        # Action embedding
        self.action_embedding = keras.layers.Embedding(
            input_dim=num_actions,
            output_dim=action_dim,
            name="action_embedding",
        )

        # Fusion trunk
        self.fusion_mlp = keras.Sequential(
            [
                keras.layers.Input(shape=(latent_dim + action_dim,)),
                keras.layers.Dense(hidden_dim, activation="relu"),
                keras.layers.Dense(hidden_dim, activation="relu"),
            ],
            name="fusion_mlp",
        )

        # Reconstruction head
        flat_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]
        self.next_obs_head = keras.Sequential(
            [
                keras.layers.Input(shape=(hidden_dim,)),
                keras.layers.Dense(hidden_dim, activation="relu"),
                keras.layers.Dense(flat_dim, activation="sigmoid"),
                keras.layers.Reshape(obs_shape),
            ],
            name="next_obs_head",
        )

    def encode_obs(self, obs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.obs_encoder(obs, training=training)

    def call(self, inputs, training: bool = False):
        obs, action = inputs

        g = self.obs_encoder(obs, training=training)  # [B, latent_dim]
        a = self.action_embedding(action)             # [B, action_dim]

        x = tf.concat([g, a], axis=-1)
        x = self.fusion_mlp(x, training=training)

        next_obs_pred = self.next_obs_head(x, training=training)
        return next_obs_pred