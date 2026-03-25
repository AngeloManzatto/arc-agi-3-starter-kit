"""
Created on Sun Mar 22 07:28:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

import time
import numpy as np

from envs.trigger_door_mini import TriggerDoorMini
from models.world_model import WorldModel
from models.transition_classifier import TransitionOutcomeClassifier


def decode_obs_state(obs: np.ndarray) -> dict:
    obs_bin = (obs > 0.5).astype(np.int32)

    agent_pos = tuple(np.argwhere(obs_bin[:, :, 1] == 1)[0])
    trigger_pos = tuple(np.argwhere(obs_bin[:, :, 2] == 1)[0])
    exit_pos = tuple(np.argwhere(obs_bin[:, :, 3] == 1)[0])

    hud_state = int(np.argmax(obs[0, 0, 4:7]))
    target_state = int(np.argmax(obs[0, 0, 7:10]))

    return {
        "agent_pos": agent_pos,
        "trigger_pos": trigger_pos,
        "exit_pos": exit_pos,
        "hud_state": hud_state,
        "target_state": target_state,
    }


def l1(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def compute_action_scores(obs: np.ndarray, g: np.ndarray, clf: TransitionOutcomeClassifier):
    d = decode_obs_state(obs)

    g_batch = np.repeat(g[None, :], 4, axis=0).astype(np.float32)
    action_batch = np.arange(4, dtype=np.int32)

    y_prob = clf.predict((g_batch, action_batch), verbose=0)

    rows = []
    for a in range(4):
        p_blocked = float(y_prob[a, 0])
        p_moved = float(y_prob[a, 1])
        p_hud_changed = float(y_prob[a, 2])
        p_success = float(y_prob[a, 3])

        # Approximate next position under the chosen action for geometric scoring
        ar, ac = d["agent_pos"]
        if a == 0:
            cand = (max(ar - 1, 0), ac)
        elif a == 1:
            cand = (min(ar + 1, 4), ac)
        elif a == 2:
            cand = (ar, max(ac - 1, 0))
        else:
            cand = (ar, min(ac + 1, 4))

        # If blocked, the agent likely stays
        next_pos_est = cand if p_blocked < 0.5 else d["agent_pos"]

        if d["hud_state"] != d["target_state"]:
            dist_term = -float(l1(next_pos_est, d["trigger_pos"]))
            score = (
                20.0 * p_success
                + 6.0 * p_hud_changed
                + 2.0 * p_moved
                - 4.0 * p_blocked
                + dist_term
            )
        else:
            dist_term = -float(l1(next_pos_est, d["exit_pos"]))
            score = (
                20.0 * p_success
                + 2.0 * p_moved
                - 4.0 * p_blocked
                + dist_term
            )

        rows.append({
            "action": a,
            "score": score,
            "p_blocked": p_blocked,
            "p_moved": p_moved,
            "p_hud_changed": p_hud_changed,
            "p_success": p_success,
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def select_action(obs, wm, clf):
    g = wm.encode_obs(obs[None, ...].astype(np.float32), training=False).numpy()[0]
    rows = compute_action_scores(obs, g, clf)
    return rows[0]["action"], rows


def run_episode(env, wm, clf, render=False):
    obs = env.reset()
    total_reward = 0.0

    for step in range(env.max_steps):
        action, rows = select_action(obs, wm, clf)
        result = env.step(action)
        obs = result.obs
        total_reward += result.reward

        if render:
            print(f"step={step} action={action}")
            for row in rows:
                print(row)
            print(env.render_ascii())
            print("-" * 50)

        if result.done:
            break

    return total_reward


def evaluate(n_episodes=100, render=False):
    env = TriggerDoorMini(seed=42)

    wm = WorldModel()
    wm.build([(None, 5, 5, 10), (None,)])
    wm.load_weights("artifacts/world_model/best.weights.h5")

    clf = TransitionOutcomeClassifier()
    clf.build([(None, 64), (None,)])
    clf.load_weights("artifacts/transition_classifier/best.weights.h5")

    rewards = []
    t0 = time.time()
    for i in range(n_episodes):
        r = run_episode(env, wm, clf, render=render and i < 3)
        rewards.append(r)
        
        dt = time.time() - t0
        
        print(f"Processed episode: {i} with reward:{r} in {dt}")

    rewards = np.asarray(rewards, dtype=np.float32)
    print("Success rate:", float(np.mean(rewards > 0)))
    print("Avg reward:", float(np.mean(rewards)))


if __name__ == "__main__":
    evaluate(n_episodes=100, render=False)