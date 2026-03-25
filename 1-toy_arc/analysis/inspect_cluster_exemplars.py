"""
Created on Sat Mar 21 11:29:39 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Decode obs into grid
###############################################################################


def decode_obs(obs: np.ndarray) -> dict:
    obs_bin = (obs > 0.5).astype(np.int32)

    wall = obs_bin[:, :, 0]
    agent = obs_bin[:, :, 1]
    trigger = obs_bin[:, :, 2]
    exit_ = obs_bin[:, :, 3]

    hud_state = int(np.argmax(obs_bin[0, 0, 4:7]))
    target_state = int(np.argmax(obs_bin[0, 0, 7:10]))

    grid = np.zeros(obs.shape[:2], dtype=np.int32)
    grid[wall == 1] = 1
    grid[trigger == 1] = 2
    grid[exit_ == 1] = 3
    grid[agent == 1] = 4

    both_trigger = (agent == 1) & (trigger == 1)
    both_exit = (agent == 1) & (exit_ == 1)
    grid[both_trigger] = 5
    grid[both_exit] = 6

    return {
        "grid": grid,
        "hud_state": hud_state,
        "target_state": target_state,
    }

###############################################################################
# Draw obs grid symbols
###############################################################################

def draw_symbolic(ax, obs: np.ndarray, title: str) -> None:
    d = decode_obs(obs)
    ax.imshow(d["grid"], interpolation="nearest")
    ax.set_title(f"{title}\nHUD={d['hud_state']} TARGET={d['target_state']}")
    ax.set_xticks(range(obs.shape[1]))
    ax.set_yticks(range(obs.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

###############################################################################
# Inspect clusters
###############################################################################

def main(path: str = "artifacts/clusters/cluster_exemplars.npz") -> None:
    data = np.load(path)

    obs = data["obs"]
    next_obs = data["next_obs"]
    action = data["action"]
    blocked = data["blocked"]
    moved = data["moved"]
    hud_changed = data["hud_changed"]
    success = data["success"]

    n = len(obs)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        title_left = (
            f"Cluster {i} | action={int(action[i])}\n"
            f"blocked={bool(blocked[i])} moved={bool(moved[i])}"
        )
        title_right = (
            f"next_obs\nhud_changed={bool(hud_changed[i])} success={bool(success[i])}"
        )

        draw_symbolic(axes[i, 0], obs[i], title_left)
        draw_symbolic(axes[i, 1], next_obs[i], title_right)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()