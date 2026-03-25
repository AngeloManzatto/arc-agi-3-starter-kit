"""
Created on Sat Mar 21 08:09:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np

###############################################################################
# Replay Analysis
###############################################################################

def main(path: str = "data/trigger_door_random.npz") -> None:
    data = np.load(path)

    print("obs shape:", data["obs"].shape)
    print("next_obs shape:", data["next_obs"].shape)
    print("action shape:", data["action"].shape)

    print("blocked rate:", data["blocked"].mean())
    print("moved rate:", data["moved"].mean())
    print("hud_changed rate:", data["hud_changed"].mean())
    print("success rate per transition:", data["success"].mean())

if __name__ == "__main__":
    main()