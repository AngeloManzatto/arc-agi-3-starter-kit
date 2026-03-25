"""
Created on Sun Mar 22 09:26:11 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

from collections import deque, Counter
import numpy as np

from envs.trigger_door_mini import TriggerDoorMini
from models.world_model import WorldModel
from models.transition_classifier import TransitionOutcomeClassifier


def decode_obs_state(obs: np.ndarray) -> dict:
    """
    Decode REAL environment observation, so exact binary decoding is okay.
    """
    obs_bin = (obs > 0.5).astype(np.int32)

    agent_pos = tuple(np.argwhere(obs_bin[:, :, 1] == 1)[0])
    trigger_pos = tuple(np.argwhere(obs_bin[:, :, 2] == 1)[0])
    exit_pos = tuple(np.argwhere(obs_bin[:, :, 3] == 1)[0])

    hud_state = int(np.argmax(obs[0, 0, 4:7]))
    target_state = int(np.argmax(obs[0, 0, 7:10]))

    wall_mask = obs_bin[:, :, 0].copy()

    return {
        "agent_pos": agent_pos,
        "trigger_pos": trigger_pos,
        "exit_pos": exit_pos,
        "hud_state": hud_state,
        "target_state": target_state,
        "wall_mask": wall_mask,
        "grid_h": obs.shape[0],
        "grid_w": obs.shape[1],
    }


def l1(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def next_pos_estimate(agent_pos, action: int, grid_h: int, grid_w: int):
    r, c = agent_pos

    if action == 0:
        return (max(r - 1, 0), c)
    if action == 1:
        return (min(r + 1, grid_h - 1), c)
    if action == 2:
        return (r, max(c - 1, 0))
    if action == 3:
        return (r, min(c + 1, grid_w - 1))

    raise ValueError(f"Invalid action: {action}")


def estimate_next_coarse_state(
    state: dict,
    action: int,
    p_blocked: float,
    p_hud_changed: float,
) -> tuple:
    """
    Estimate only the coarse next state signature.
    """
    cand = next_pos_estimate(
        state["agent_pos"], action, state["grid_h"], state["grid_w"]
    )

    cr, cc = cand
    blocked_by_wall = bool(state["wall_mask"][cr, cc] == 1)

    if blocked_by_wall or p_blocked >= 0.5:
        next_pos = state["agent_pos"]
    else:
        next_pos = cand

    next_hud = state["hud_state"]
    if next_pos == state["trigger_pos"] and p_hud_changed >= 0.35:
        next_hud = (next_hud + 1) % 3

    on_trigger = int(next_pos == state["trigger_pos"])
    on_exit = int(next_pos == state["exit_pos"])

    return (
        next_pos,
        next_hud,
        state["target_state"],
        on_trigger,
        on_exit,
    )


def compute_action_scores(
    obs: np.ndarray,
    g: np.ndarray,
    clf: TransitionOutcomeClassifier,
    memory_counter: Counter,
    memory_window: deque,
    novelty_bonus: float = 1.0,
    revisit_penalty_scale: float = 1.5,
):
    state = decode_obs_state(obs)

    g_batch = np.repeat(g[None, :], 4, axis=0).astype(np.float32)
    action_batch = np.arange(4, dtype=np.int32)

    y_prob = clf.predict((g_batch, action_batch), verbose=0)

    rows = []
    for a in range(4):
        p_blocked = float(y_prob[a, 0])
        p_moved = float(y_prob[a, 1])
        p_hud_changed = float(y_prob[a, 2])
        p_success = float(y_prob[a, 3])

        coarse_next = estimate_next_coarse_state(
            state=state,
            action=a,
            p_blocked=p_blocked,
            p_hud_changed=p_hud_changed,
        )

        revisit_count = memory_counter[coarse_next]

        next_pos = coarse_next[0]

        if state["hud_state"] != state["target_state"]:
            dist_term = -float(l1(next_pos, state["trigger_pos"]))
            base_score = (
                20.0 * p_success
                + 6.0 * p_hud_changed
                + 2.0 * p_moved
                - 4.0 * p_blocked
                + dist_term
            )
        else:
            dist_term = -float(l1(next_pos, state["exit_pos"]))
            base_score = (
                20.0 * p_success
                + 2.0 * p_moved
                - 4.0 * p_blocked
                + dist_term
            )

        novelty_term = novelty_bonus if revisit_count == 0 else 0.0
        memory_penalty = revisit_penalty_scale * revisit_count

        score = base_score + novelty_term - memory_penalty

        rows.append({
            "action": a,
            "score": float(score),
            "base_score": float(base_score),
            "novelty_term": float(novelty_term),
            "memory_penalty": float(memory_penalty),
            "revisit_count": int(revisit_count),
            "coarse_next": coarse_next,
            "p_blocked": p_blocked,
            "p_moved": p_moved,
            "p_hud_changed": p_hud_changed,
            "p_success": p_success,
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def select_action(obs, wm, clf, memory_counter: Counter, memory_window: deque):
    g = wm.encode_obs(obs[None, ...].astype(np.float32), training=False).numpy()[0]
    rows = compute_action_scores(
        obs=obs,
        g=g,
        clf=clf,
        memory_counter=memory_counter,
        memory_window=memory_window,
    )
    return rows[0]["action"], rows


def update_memory(
    obs: np.ndarray,
    memory_counter: Counter,
    memory_window: deque,
    max_memory_len: int = 12,
):
    state = decode_obs_state(obs)
    signature = (
        state["agent_pos"],
        state["hud_state"],
        state["target_state"],
        int(state["agent_pos"] == state["trigger_pos"]),
        int(state["agent_pos"] == state["exit_pos"]),
    )

    memory_window.append(signature)
    memory_counter[signature] += 1

    while len(memory_window) > max_memory_len:
        old = memory_window.popleft()
        memory_counter[old] -= 1
        if memory_counter[old] <= 0:
            del memory_counter[old]


def run_episode(env, wm, clf, render=False):
    obs = env.reset()
    total_reward = 0.0

    memory_window = deque()
    memory_counter = Counter()

    update_memory(obs, memory_counter, memory_window)

    for step in range(env.max_steps):
        action, rows = select_action(obs, wm, clf, memory_counter, memory_window)

        result = env.step(action)
        obs = result.obs
        total_reward += result.reward

        update_memory(obs, memory_counter, memory_window)

        if render:
            print(f"step={step} action={action}")
            for row in rows:
                print(row)
            print(env.render_ascii())
            print("-" * 60)

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
    for i in range(n_episodes):
        r = run_episode(env, wm, clf, render=render and i < 3)
        rewards.append(r)
        print(f"Processed episode: {i} with reward:{r}")

    rewards = np.asarray(rewards, dtype=np.float32)
    print("Success rate:", float(np.mean(rewards > 0)))
    print("Avg reward:", float(np.mean(rewards)))


if __name__ == "__main__":
    evaluate(n_episodes=100, render=False)