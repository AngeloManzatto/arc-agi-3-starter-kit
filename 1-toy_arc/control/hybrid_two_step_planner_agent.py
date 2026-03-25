#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 09:06:18 2026

@author: root
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


def encode_state_to_obs(state: dict) -> np.ndarray:
    h, w = state["grid_h"], state["grid_w"]
    obs = np.zeros((h, w, 10), dtype=np.float32)

    obs[:, :, 0] = state["wall_mask"]

    ar, ac = state["agent_pos"]
    tr, tc = state["trigger_pos"]
    er, ec = state["exit_pos"]

    obs[ar, ac, 1] = 1.0
    obs[tr, tc, 2] = 1.0
    obs[er, ec, 3] = 1.0

    obs[:, :, 4 + state["hud_state"]] = 1.0
    obs[:, :, 7 + state["target_state"]] = 1.0

    return obs


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


def desired_family_from_state(state: dict) -> str:
    if state["hud_state"] != state["target_state"]:
        return FAMILY_TRIGGER
    return FAMILY_MOVE


def predict_action_probs(
    obs: np.ndarray,
    wm: WorldModel,
    clf: TransitionOutcomeClassifier,
):
    g = wm.encode_obs(obs[None, ...].astype(np.float32), training=False).numpy()[0]
    g_batch = np.repeat(g[None, :], 4, axis=0).astype(np.float32)
    action_batch = np.arange(4, dtype=np.int32)
    y_prob = clf.predict((g_batch, action_batch), verbose=0)
    return y_prob


def coarse_transition(state: dict, action: int, probs: np.ndarray) -> dict:
    """
    Cheap approximate transition:
    - if blocked likely, stay
    - else move
    - if moved onto trigger and hud_change likely, increment hud
    """
    p_blocked, p_moved, p_hud_changed, p_success = [float(x) for x in probs]

    next_state = {
        "agent_pos": state["agent_pos"],
        "trigger_pos": state["trigger_pos"],
        "exit_pos": state["exit_pos"],
        "hud_state": state["hud_state"],
        "target_state": state["target_state"],
        "wall_mask": state["wall_mask"].copy(),
        "grid_h": state["grid_h"],
        "grid_w": state["grid_w"],
    }

    cand = next_pos_estimate(
        state["agent_pos"], action, state["grid_h"], state["grid_w"]
    )

    cr, cc = cand
    blocked_by_wall = bool(state["wall_mask"][cr, cc] == 1)

    if blocked_by_wall or p_blocked >= 0.5:
        next_pos = state["agent_pos"]
    else:
        next_pos = cand

    next_state["agent_pos"] = next_pos

    if next_pos == state["trigger_pos"] and p_hud_changed >= 0.35:
        next_state["hud_state"] = (state["hud_state"] + 1) % 3

    return next_state


def score_action_semantics(
    state: dict,
    action: int,
    probs: np.ndarray,
    family_bonus_scale: float = 1.5,
) -> tuple[float, str]:
    p_blocked, p_moved, p_hud_changed, p_success = [float(x) for x in probs]

    family = infer_family(p_blocked, p_moved, p_hud_changed, p_success)
    desired = desired_family_from_state(state)

    cand = next_pos_estimate(
        state["agent_pos"], action, state["grid_h"], state["grid_w"]
    )
    next_pos = state["agent_pos"] if p_blocked >= 0.5 else cand

    if p_success >= 0.5:
        score = 100.0 * p_success - 5.0 * p_blocked
        return score, family

    if desired == FAMILY_TRIGGER:
        dist_term = -float(l1(next_pos, state["trigger_pos"]))
        score = (
            25.0 * p_success
            + 8.0 * p_hud_changed
            + 2.0 * p_moved
            - 5.0 * p_blocked
            + dist_term
        )
    else:
        dist_term = -float(l1(next_pos, state["exit_pos"]))
        score = (
            25.0 * p_success
            + 3.0 * p_moved
            - 5.0 * p_blocked
            + dist_term
        )

    if family == desired:
        score += family_bonus_scale

    return float(score), family


def select_action_hybrid_two_step(
    obs: np.ndarray,
    wm: WorldModel,
    clf: TransitionOutcomeClassifier,
    gamma_second: float = 0.7,
):
    state0 = decode_obs_state(obs)
    y_prob_first = predict_action_probs(obs, wm, clf)

    best = None
    rows = []

    for a1 in range(4):
        probs1 = y_prob_first[a1]
        score1, fam1 = score_action_semantics(state0, a1, probs1)

        state1 = coarse_transition(state0, a1, probs1)
        obs1 = encode_state_to_obs(state1)

        y_prob_second = predict_action_probs(obs1, wm, clf)

        best_second_score = -1e18
        best_a2 = None
        best_fam2 = None

        for a2 in range(4):
            probs2 = y_prob_second[a2]
            score2, fam2 = score_action_semantics(state1, a2, probs2)

            # Small prototype-sequence coherence bonus
            seq_bonus = 0.0
            desired1 = desired_family_from_state(state0)
            desired2 = desired_family_from_state(state1)

            if fam1 == desired1:
                seq_bonus += 0.5
            if fam2 == desired2:
                seq_bonus += 0.5

            total2 = score2 + seq_bonus
            if total2 > best_second_score:
                best_second_score = total2
                best_a2 = a2
                best_fam2 = fam2

        total = score1 + gamma_second * best_second_score

        row = {
            "a1": a1,
            "fam1": fam1,
            "score1": float(score1),
            "best_a2": int(best_a2),
            "best_fam2": best_fam2,
            "best_second_score": float(best_second_score),
            "total_score": float(total),
            "p_blocked_1": float(probs1[0]),
            "p_moved_1": float(probs1[1]),
            "p_hud_changed_1": float(probs1[2]),
            "p_success_1": float(probs1[3]),
        }
        rows.append(row)

        if best is None or total > best["total_score"]:
            best = row

    rows.sort(key=lambda r: r["total_score"], reverse=True)
    return int(best["a1"]), rows


def run_episode(env, wm, clf, render: bool = False):
    obs = env.reset()
    total_reward = 0.0

    for step in range(env.max_steps):
        action, rows = select_action_hybrid_two_step(obs, wm, clf)
        result = env.step(action)
        obs = result.obs
        total_reward += result.reward

        if render:
            print(f"step={step} action={action}")
            for row in rows:
                print(row)
            print(env.render_ascii())
            print("-" * 70)

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