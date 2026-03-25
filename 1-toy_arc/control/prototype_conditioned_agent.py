"""
Created on Sun Mar 22 08:38:18 2026

@author: Angelo Antonio Manzatto
"""

from __future__ import annotations

import numpy as np

from envs.trigger_door_mini import TriggerDoorMini
from models.world_model import WorldModel
from models.transition_classifier import TransitionOutcomeClassifier


FAMILY_BLOCKED = "blocked"
FAMILY_TRIGGER = "trigger"
FAMILY_MOVE = "move"
FAMILY_SUCCESS = "success"


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


def infer_family(
    p_blocked: float,
    p_moved: float,
    p_hud_changed: float,
    p_success: float,
    success_thr: float = 0.50,
    trigger_thr: float = 0.35,
    blocked_thr: float = 0.50,
) -> str:
    if p_success >= success_thr:
        return FAMILY_SUCCESS
    if p_hud_changed >= trigger_thr:
        return FAMILY_TRIGGER
    if p_blocked >= blocked_thr:
        return FAMILY_BLOCKED
    return FAMILY_MOVE


def desired_family(obs: np.ndarray) -> str:
    d = decode_obs_state(obs)

    if d["hud_state"] != d["target_state"]:
        return FAMILY_TRIGGER
    return FAMILY_MOVE


def score_action_for_family(
    obs: np.ndarray,
    action: int,
    p_blocked: float,
    p_moved: float,
    p_hud_changed: float,
    p_success: float,
    family: str,
) -> float:
    d = decode_obs_state(obs)

    cand_pos = next_pos_estimate(
        d["agent_pos"], action, d["grid_h"], d["grid_w"]
    )

    next_pos = d["agent_pos"] if p_blocked >= 0.5 else cand_pos

    if family == FAMILY_SUCCESS:
        return 100.0 * p_success - 5.0 * p_blocked

    if family == FAMILY_TRIGGER:
        dist_term = -float(l1(next_pos, d["trigger_pos"]))
        return (
            25.0 * p_success
            + 8.0 * p_hud_changed
            + 2.0 * p_moved
            - 5.0 * p_blocked
            + dist_term
        )

    if family == FAMILY_MOVE:
        dist_term = -float(l1(next_pos, d["exit_pos"]))
        return (
            25.0 * p_success
            + 3.0 * p_moved
            - 5.0 * p_blocked
            + dist_term
        )

    if family == FAMILY_BLOCKED:
        return -10.0 * p_blocked

    raise ValueError(f"Unknown family: {family}")


def compute_action_rows(
    obs: np.ndarray,
    g: np.ndarray,
    clf: TransitionOutcomeClassifier,
):
    g_batch = np.repeat(g[None, :], 4, axis=0).astype(np.float32)
    action_batch = np.arange(4, dtype=np.int32)

    y_prob = clf.predict((g_batch, action_batch), verbose=0)

    rows = []
    for action in range(4):
        p_blocked = float(y_prob[action, 0])
        p_moved = float(y_prob[action, 1])
        p_hud_changed = float(y_prob[action, 2])
        p_success = float(y_prob[action, 3])

        family = infer_family(
            p_blocked=p_blocked,
            p_moved=p_moved,
            p_hud_changed=p_hud_changed,
            p_success=p_success,
        )

        rows.append({
            "action": action,
            "family": family,
            "p_blocked": p_blocked,
            "p_moved": p_moved,
            "p_hud_changed": p_hud_changed,
            "p_success": p_success,
        })

    return rows


def select_action(
    obs: np.ndarray,
    wm: WorldModel,
    clf: TransitionOutcomeClassifier,
    recent_hashes: set[str] | None = None,
):
    g = wm.encode_obs(obs[None, ...].astype(np.float32), training=False).numpy()[0]
    rows = compute_action_rows(obs, g, clf)

    # Immediate success override
    success_rows = [r for r in rows if r["family"] == FAMILY_SUCCESS]
    if success_rows:
        success_rows = sorted(success_rows, key=lambda r: r["p_success"], reverse=True)
        return success_rows[0]["action"], success_rows, FAMILY_SUCCESS

    target_family = desired_family(obs)

    scored_rows = []
    for row in rows:
        score = score_action_for_family(
            obs=obs,
            action=row["action"],
            p_blocked=row["p_blocked"],
            p_moved=row["p_moved"],
            p_hud_changed=row["p_hud_changed"],
            p_success=row["p_success"],
            family=target_family,
        )

        # Small family preference bonus
        if row["family"] == target_family:
            score += 1.5

        # Mild repeated-state penalty from estimated next position
        if recent_hashes is not None:
            d = decode_obs_state(obs)
            cand_pos = next_pos_estimate(
                d["agent_pos"], row["action"], d["grid_h"], d["grid_w"]
            )
            next_pos = d["agent_pos"] if row["p_blocked"] >= 0.5 else cand_pos
            coarse_hash = f"{next_pos}_{d['hud_state']}_{d['target_state']}"
            if coarse_hash in recent_hashes:
                score -= 2.0

        row = dict(row)
        row["score"] = float(score)
        scored_rows.append(row)

    scored_rows.sort(key=lambda r: r["score"], reverse=True)
    return scored_rows[0]["action"], scored_rows, target_family


def run_episode(env, wm, clf, render: bool = False):
    obs = env.reset()
    total_reward = 0.0
    recent_hashes = set()

    for step in range(env.max_steps):
        d = decode_obs_state(obs)
        current_hash = f"{d['agent_pos']}_{d['hud_state']}_{d['target_state']}"
        recent_hashes.add(current_hash)

        action, rows, target_family = select_action(
            obs=obs,
            wm=wm,
            clf=clf,
            recent_hashes=recent_hashes,
        )

        result = env.step(action)
        obs = result.obs
        total_reward += result.reward

        if render:
            print(f"step={step} target_family={target_family} action={action}")
            for row in rows:
                print(row)
            print(env.render_ascii())
            print("-" * 60)

        if result.done:
            break

    return total_reward


def evaluate(n_episodes: int = 100, render: bool = False):
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