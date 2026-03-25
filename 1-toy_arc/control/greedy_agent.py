"""
Created on Sat Mar 21 15:43:42 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

import numpy as np

from envs.trigger_door_mini import TriggerDoorMini
from models.world_model import WorldModel

###############################################################################
# Convert Observation to State
###############################################################################
def decode_obs_state(obs: np.ndarray) -> dict:
    """
    Robust decoder for predicted observations.
    Works on soft outputs from the world model.
    """
    agent_map = obs[:, :, 1]
    trigger_map = obs[:, :, 2]
    exit_map = obs[:, :, 3]

    agent_pos = tuple(np.unravel_index(np.argmax(agent_map), agent_map.shape))
    trigger_pos = tuple(np.unravel_index(np.argmax(trigger_map), trigger_map.shape))
    exit_pos = tuple(np.unravel_index(np.argmax(exit_map), exit_map.shape))

    hud_state = int(np.argmax(obs[0, 0, 4:7]))
    target_state = int(np.argmax(obs[0, 0, 7:10]))

    return {
        "agent_pos": agent_pos,
        "trigger_pos": trigger_pos,
        "exit_pos": exit_pos,
        "hud_state": hud_state,
        "target_state": target_state,
    }

###############################################################################
# L1 distance
###############################################################################

def l1(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

###############################################################################
# Calculate reward
###############################################################################

def score(obs: np.ndarray) -> float:
    d = decode_obs_state(obs)

    # success
    if d["agent_pos"] == d["exit_pos"] and d["hud_state"] == d["target_state"]:
        return 100.0

    # need trigger
    if d["hud_state"] != d["target_state"]:
        return -l1(d["agent_pos"], d["trigger_pos"])

    # go to exit
    return -l1(d["agent_pos"], d["exit_pos"])

###############################################################################
# Agent Policy
###############################################################################

def select_action(obs, model):
    obs_batch = np.repeat(obs[None, ...], 4, axis=0).astype(np.float32)
    actions = np.arange(4, dtype=np.int32)

    next_obs_pred = model.predict((obs_batch, actions), verbose=0)

    scores = [score(next_obs_pred[a]) for a in range(4)]
    best_action = int(np.argmax(scores))

    return best_action, scores


def run_episode(env, model, render=False):
    obs = env.reset()
    total_reward = 0

    for step in range(env.max_steps):
        action, scores = select_action(obs, model)

        result = env.step(action)
        obs = result.obs
        total_reward += result.reward

        if render:
            print(f"step={step} action={action} scores={scores}")
            print(env.render_ascii())
            print("-" * 30)

        if result.done:
            break

    return total_reward


def evaluate(n_episodes=100):
    env = TriggerDoorMini(seed=42)

    model = WorldModel()
    model.build([(None, 5, 5, 10), (None,)])
    model.load_weights("artifacts/world_model/best.weights.h5")

    rewards = []

    for _ in range(n_episodes):
        r = run_episode(env, model)
        rewards.append(r)

    rewards = np.array(rewards)

    print("Success rate:", np.mean(rewards > 0))
    print("Avg reward:", rewards.mean())


if __name__ == "__main__":
    evaluate()