"""
Created on Sun Mar 22 07:23:09 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report

from models.transition_classifier import TransitionOutcomeClassifier

###############################################################################
# Constants
###############################################################################

TARGET_NAMES = ["blocked", "moved", "hud_changed", "success"]

###############################################################################
# Execute evaluation
###############################################################################

def main():
    data = np.load("artifacts/deltas/trigger_door_deltas.npz")

    g = data["g"].astype(np.float32)
    action = data["action"].astype(np.int32)

    y_true = np.stack(
        [
            data["blocked"].astype(np.int32),
            data["moved"].astype(np.int32),
            data["hud_changed"].astype(np.int32),
            data["success"].astype(np.int32),
        ],
        axis=1,
    )

    model = TransitionOutcomeClassifier()
    model.build([(None, g.shape[1]), (None,)])
    model.load_weights("artifacts/transition_classifier/best.weights.h5")

    y_prob = model.predict((g, action), batch_size=512, verbose=0)
    y_pred = (y_prob >= 0.5).astype(np.int32)

    for i, name in enumerate(TARGET_NAMES):
        print("=" * 70)
        print(name)
        print(classification_report(y_true[:, i], y_pred[:, i], digits=4))


if __name__ == "__main__":
    main()