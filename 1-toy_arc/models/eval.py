"""
Created on Sat Mar 21 08:30:45 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from models.world_model import WorldModel

###############################################################################
# Constants
###############################################################################

CHANNEL_NAMES = [
    "wall",
    "agent",
    "trigger",
    "exit",
    "hud_0",
    "hud_1",
    "hud_2",
    "target_0",
    "target_1",
    "target_2",
]

###############################################################################
# Visual inspection
###############################################################################

def decode_obs(obs: np.ndarray) -> dict:
    """
    Convert [5,5,10] tensor into simple symbolic maps for visualization.
    """
    h, w, _ = obs.shape
    obs_bin = (obs > 0.5).astype(np.int32)

    wall = obs_bin[:, :, 0]
    agent = obs_bin[:, :, 1]
    trigger = obs_bin[:, :, 2]
    exit_ = obs_bin[:, :, 3]

    hud_state = int(np.argmax(obs_bin[0, 0, 4:7]))
    target_state = int(np.argmax(obs_bin[0, 0, 7:10]))

    grid = np.zeros((h, w), dtype=np.int32)
    # 0 empty
    # 1 wall
    # 2 trigger
    # 3 exit
    # 4 agent
    # 5 agent on trigger
    # 6 agent on exit

    grid[wall == 1] = 1
    grid[trigger == 1] = 2
    grid[exit_ == 1] = 3
    grid[agent == 1] = 4

    # overlay agent special cases
    both_trigger = (agent == 1) & (trigger == 1)
    both_exit = (agent == 1) & (exit_ == 1)
    grid[both_trigger] = 5
    grid[both_exit] = 6

    return {
        "grid": grid,
        "hud_state": hud_state,
        "target_state": target_state,
    }

def draw_symbolic(ax, obs: np.ndarray, title: str) -> None:
    decoded = decode_obs(obs)
    grid = decoded["grid"]
    hud_state = decoded["hud_state"]
    target_state = decoded["target_state"]

    ax.imshow(grid, interpolation="nearest")
    ax.set_title(f"{title}\nHUD={hud_state}  TARGET={target_state}")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)


def plot_sample(obs, next_obs_true, next_obs_pred, action, sample_idx: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    draw_symbolic(axes[0], obs, f"Current obs\naction={action}")
    draw_symbolic(axes[1], next_obs_true, "True next obs")
    draw_symbolic(axes[2], next_obs_pred, "Predicted next obs")

    fig.suptitle(f"Sample {sample_idx}", fontsize=14)
    fig.tight_layout()
    plt.show()

###############################################################################
# Summary
###############################################################################

def summarize_obs(obs: np.ndarray) -> dict:
    obs_bin = (obs > 0.5).astype(np.int32)
    summary = {}

    for i, name in enumerate(CHANNEL_NAMES):
        summary[name] = int(obs_bin[:, :, i].sum())

    return summary

###############################################################################
# Eval pipeline
###############################################################################

def main(
    weights_path: str = "artifacts/world_model/best.weights.h5",
    data_path: str = "data/trigger_door_random.npz",
    n_samples: int = 8,
    seed: int = 123,
) -> None:
    rng = np.random.default_rng(seed)

    data = np.load(data_path)
    n = len(data["obs"])
    idx = rng.choice(n, size=min(n_samples, n), replace=False)

    obs = data["obs"][idx].astype(np.float32)
    action = data["action"][idx].astype(np.int32)
    next_obs_true = data["next_obs"][idx].astype(np.float32)

    model = WorldModel()
    model.build([(None, 5, 5, 10), (None,)])
    model.load_weights(weights_path)

    next_obs_pred = model.predict((obs, action), verbose=0)

    for i in range(len(obs)):
        plot_sample(
            obs=obs[i],
            next_obs_true=next_obs_true[i],
            next_obs_pred=next_obs_pred[i],
            action=int(action[i]),
            sample_idx=int(idx[i]),
        )
        
if __name__ == "__main__":
    main()