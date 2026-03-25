"""
Created on Sun Mar 22 10:44:39 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

import math
import numpy as np


FAMILY_BLOCKED = "blocked"
FAMILY_TRIGGER = "trigger"
FAMILY_MOVE = "move"
FAMILY_SUCCESS = "success"


def decode_obs_state(obs: np.ndarray) -> dict:
    """
    Decode REAL environment observation into a privileged coarse state.
    This is okay for Track A.
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


def encode_state_to_obs(state: dict) -> np.ndarray:
    """
    Convert coarse privileged state back into an observation tensor.
    Useful for classifier queries during search.
    """
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


def state_to_node_key(state: dict) -> tuple:
    """
    Richer semantic node key for Track A graph / MCTS.
    """
    return (
        state["agent_pos"][0],
        state["agent_pos"][1],
        state["hud_state"],
        state["target_state"],
        int(state["agent_pos"] == state["trigger_pos"]),
        int(state["agent_pos"] == state["exit_pos"]),
    )


def l1(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def next_pos_estimate(agent_pos, action: int, grid_h: int, grid_w: int):
    r, c = agent_pos

    if action == 0:  # up
        return (max(r - 1, 0), c)
    if action == 1:  # down
        return (min(r + 1, grid_h - 1), c)
    if action == 2:  # left
        return (r, max(c - 1, 0))
    if action == 3:  # right
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


def coarse_transition(
    state: dict,
    action: int,
    p_blocked: float,
    p_hud_changed: float,
    trigger_thr: float = 0.35,
) -> dict:
    """
    Cheap symbolic transition model used for graph search.
    """
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
        state["agent_pos"],
        action,
        state["grid_h"],
        state["grid_w"],
    )

    cr, cc = cand
    blocked_by_wall = bool(state["wall_mask"][cr, cc] == 1)

    if blocked_by_wall or p_blocked >= 0.5:
        next_pos = state["agent_pos"]
    else:
        next_pos = cand

    next_state["agent_pos"] = next_pos

    if next_pos == state["trigger_pos"] and p_hud_changed >= trigger_thr:
        next_state["hud_state"] = (state["hud_state"] + 1) % 3

    return next_state


def is_success_state(state: dict) -> bool:
    return (
        state["agent_pos"] == state["exit_pos"]
        and state["hud_state"] == state["target_state"]
    )


def heuristic_value(state: dict, revisit_count: int = 0) -> float:
    """
    Privileged heuristic value for Track A.
    Later, in Track B, we replace this with learned value.
    """
    if is_success_state(state):
        return 100.0

    if state["hud_state"] != state["target_state"]:
        dist = l1(state["agent_pos"], state["trigger_pos"])
        val = -float(dist)
    else:
        dist = l1(state["agent_pos"], state["exit_pos"])
        val = -float(dist)

    val -= 1.5 * revisit_count
    return val


def action_score_for_prior(
    state: dict,
    action: int,
    p_blocked: float,
    p_moved: float,
    p_hud_changed: float,
    p_success: float,
) -> float:
    """
    Semantic prior score before softmax.
    """
    cand = next_pos_estimate(
        state["agent_pos"],
        action,
        state["grid_h"],
        state["grid_w"],
    )
    next_pos = state["agent_pos"] if p_blocked >= 0.5 else cand

    if state["hud_state"] != state["target_state"]:
        dist_term = -float(l1(next_pos, state["trigger_pos"]))
        score = (
            20.0 * p_success
            + 6.0 * p_hud_changed
            + 2.0 * p_moved
            - 4.0 * p_blocked
            + dist_term
        )
    else:
        dist_term = -float(l1(next_pos, state["exit_pos"]))
        score = (
            20.0 * p_success
            + 2.0 * p_moved
            - 4.0 * p_blocked
            + dist_term
        )

    desired = desired_family_from_state(state)
    fam = infer_family(p_blocked, p_moved, p_hud_changed, p_success)
    if fam == desired:
        score += 1.0

    return float(score)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x.astype(np.float64) / max(temperature, 1e-8)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def probs_to_priors(state: dict, y_prob: np.ndarray, temperature: float = 1.0):
    """
    Convert classifier outputs [4,4] into action priors.
    Returns:
      priors: dict[action] -> float
      infos: dict[action] -> metadata dict
    """
    scores = []
    infos = {}

    for action in range(4):
        p_blocked = float(y_prob[action, 0])
        p_moved = float(y_prob[action, 1])
        p_hud_changed = float(y_prob[action, 2])
        p_success = float(y_prob[action, 3])

        fam = infer_family(
            p_blocked=p_blocked,
            p_moved=p_moved,
            p_hud_changed=p_hud_changed,
            p_success=p_success,
        )

        score = action_score_for_prior(
            state=state,
            action=action,
            p_blocked=p_blocked,
            p_moved=p_moved,
            p_hud_changed=p_hud_changed,
            p_success=p_success,
        )

        scores.append(score)
        infos[action] = {
            "family": fam,
            "p_blocked": p_blocked,
            "p_moved": p_moved,
            "p_hud_changed": p_hud_changed,
            "p_success": p_success,
            "prior_score": score,
        }

    priors_arr = softmax(np.asarray(scores, dtype=np.float32), temperature=temperature)
    priors = {a: float(priors_arr[a]) for a in range(4)}

    return priors, infos