"""
Created on Sun Mar 22 07:11:37 2026

@author: Angelo Antonio Manzatto
"""
###############################################################################
# libraries
###############################################################################

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

###############################################################################
# Transition Outcome Classifier
###############################################################################

class TransitionOutcomeClassifier(keras.Model):
    def __init__(
        self,
        latent_dim: int = 64,
        num_actions: int = 4,
        action_dim: int = 8,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.action_embedding = keras.layers.Embedding(
            input_dim=num_actions,
            output_dim=action_dim,
            name="action_embedding",
        )

        self.backbone = keras.Sequential(
            [
                keras.layers.Input(shape=(latent_dim + action_dim,)),
                keras.layers.Dense(hidden_dim, activation="relu"),
                keras.layers.Dense(hidden_dim, activation="relu"),
                keras.layers.Dense(4, activation="sigmoid"),
            ],
            name="transition_outcome_head",
        )

    def call(self, inputs, training: bool = False):
        g, action = inputs
        a = self.action_embedding(action)
        x = tf.concat([g, a], axis=-1)
        y = self.backbone(x, training=training)
        return y