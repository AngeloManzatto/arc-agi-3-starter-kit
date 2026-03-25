"""
Created on Sat Mar 21 16:04:36 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

import time
import itertools
import numpy as np

from envs.trigger_door_mini import TriggerDoorMini
from models.world_model import WorldModel


def decode_obs_state(obs: np.ndarray) -> dict:
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


def l1(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def score(obs: np.ndarray) -> float:
    d = decode_obs_state(obs)

    if d["agent_pos"] == d["exit_pos"] and d["hud_state"] == d["target_state"]:
        return 100.0

    if d["hud_state"] != d["target_state"]:
        return -float(l1(d["agent_pos"], d["trigger_pos"]))

    return -float(l1(d["agent_pos"], d["exit_pos"]))


def predict_one_step(model, obs: np.ndarray, action: int) -> np.ndarray:
    obs_batch = obs[None, ...].astype(np.float32)
    action_batch = np.asarray([action], dtype=np.int32)
    pred = model.predict((obs_batch, action_batch), verbose=0)[0]
    return pred


def rollout_sequence(model, obs: np.ndarray, action_seq) -> np.ndarray:
    cur = obs
    for a in action_seq:
        cur = predict_one_step(model, cur, int(a))
    return cur


def select_action_rollout(model, obs: np.ndarray, horizon: int = 3):
    best_seq = None
    best_score = -1e18

    for seq in itertools.product(range(4), repeat=horizon):
        pred_final = rollout_sequence(model, obs, seq)
        s = score(pred_final)

        if s > best_score:
            best_score = s
            best_seq = seq

    return int(best_seq[0]), float(best_score), tuple(best_seq)


def run_episode(env, model, horizon=3, render=False):
    obs = env.reset()
    total_reward = 0.0

    for step in range(env.max_steps):
        action, best_score, best_seq = select_action_rollout(model, obs, horizon=horizon)

        result = env.step(action)
        obs = result.obs
        total_reward += result.reward

        if render:
            print(f"step={step} action={action} best_seq={best_seq} best_score={best_score:.3f}")
            print(env.render_ascii())
            print("-" * 40)

        if result.done:
            break

    return total_reward


def evaluate(n_episodes=100, horizon=3):
    env = TriggerDoorMini(seed=42)

    model = WorldModel()
    model.build([(None, 5, 5, 10), (None,)])
    model.load_weights("artifacts/world_model/best.weights.h5")

    rewards = []
    
    t0 = time.time()
    for i in range(n_episodes):
        r = run_episode(env, model, horizon=horizon, render=False)
        rewards.append(r)
        
        dt = time.time() - t0
        
        print(f"Processed episode: {i} with reward:{r} in {dt}")

    rewards = np.asarray(rewards, dtype=np.float32)

    print("Horizon:", horizon)
    print("Success rate:", float(np.mean(rewards > 0)))
    print("Avg reward:", float(np.mean(rewards)))


if __name__ == "__main__":
    evaluate(n_episodes=100, horizon=3)