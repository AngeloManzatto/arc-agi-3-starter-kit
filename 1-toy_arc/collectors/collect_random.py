"""
Created on Sat Mar 21 08:06:14 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

from pathlib import Path
import numpy as np

from envs.trigger_door_mini import TriggerDoorMini

###############################################################################
# Random Episode Collector Policy
###############################################################################

def collect_random_dataset(
    n_episodes: int = 1000,
    max_steps: int = 25,
    seed: int = 123,
    save_path: str = "data/trigger_door_random.npz",
) -> None:
    rng = np.random.default_rng(seed)
    env = TriggerDoorMini(max_steps=max_steps, seed=seed)

    obs_list = []
    action_list = []
    next_obs_list = []
    reward_list = []
    done_list = []

    moved_list = []
    blocked_list = []
    hud_changed_list = []
    success_list = []
    episode_id_list = []
    step_id_list = []

    for episode_id in range(n_episodes):
        obs = env.reset()
        done = False
        step_id = 0

        while not done:
            action = int(rng.integers(0, 4))
            result = env.step(action)

            obs_list.append(obs)
            action_list.append(action)
            next_obs_list.append(result.obs)
            reward_list.append(result.reward)
            done_list.append(result.done)

            moved_list.append(result.info["moved"])
            blocked_list.append(result.info["blocked"])
            hud_changed_list.append(result.info["hud_changed"])
            success_list.append(result.info["success"])
            episode_id_list.append(episode_id)
            step_id_list.append(step_id)

            obs = result.obs
            done = result.done
            step_id += 1

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        obs=np.asarray(obs_list, dtype=np.float32),
        action=np.asarray(action_list, dtype=np.int64),
        next_obs=np.asarray(next_obs_list, dtype=np.float32),
        reward=np.asarray(reward_list, dtype=np.float32),
        done=np.asarray(done_list, dtype=np.bool_),
        moved=np.asarray(moved_list, dtype=np.bool_),
        blocked=np.asarray(blocked_list, dtype=np.bool_),
        hud_changed=np.asarray(hud_changed_list, dtype=np.bool_),
        success=np.asarray(success_list, dtype=np.bool_),
        episode_id=np.asarray(episode_id_list, dtype=np.int64),
        step_id=np.asarray(step_id_list, dtype=np.int64),
    )

    print(f"Saved dataset to: {save_path}")
    print(f"Transitions: {len(action_list)}")
    print(f"Episodes: {n_episodes}")
    print(f"Success transitions: {np.sum(success_list)}")
    print(f"Blocked transitions: {np.sum(blocked_list)}")
    print(f"HUD-changed transitions: {np.sum(hud_changed_list)}")


if __name__ == "__main__":
    collect_random_dataset()
