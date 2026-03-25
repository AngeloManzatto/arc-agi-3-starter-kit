#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:47:17 2026

@author: Angelo Antonio Manzatto
"""
###############################################################################
# libraries
###############################################################################


from __future__ import annotations

import numpy as np

from envs.trigger_door_mini import TriggerDoorMini
from models.world_model import WorldModel
from models.transition_classifier import TransitionOutcomeClassifier

from planning.semantic_utils import (
    decode_obs_state,
    encode_state_to_obs,
    state_to_node_key,
    probs_to_priors,
)
from planning.semantic_graph import SemanticGraph
from planning.mcts_semantic import SemanticMCTS


def main():
    env = TriggerDoorMini(seed=42)
    obs = env.reset()

    wm = WorldModel()
    wm.build([(None, 5, 5, 10), (None,)])
    wm.load_weights("artifacts/world_model/best.weights.h5")

    clf = TransitionOutcomeClassifier()
    clf.build([(None, 64), (None,)])
    clf.load_weights("artifacts/transition_classifier/best.weights.h5")

    state = decode_obs_state(obs)
    obs2 = encode_state_to_obs(state)
    node_key = state_to_node_key(state)

    print("Node key:", node_key)
    print("Obs reconstruction equal:", np.allclose(obs, obs2))

    g = wm.encode_obs(obs[None, ...].astype(np.float32), training=False).numpy()[0]
    g_batch = np.repeat(g[None, :], 4, axis=0).astype(np.float32)
    action_batch = np.arange(4, dtype=np.int32)
    y_prob = clf.predict((g_batch, action_batch), verbose=0)

    priors, infos = probs_to_priors(state, y_prob, temperature=1.0)

    print("Priors:")
    for a in range(4):
        print(a, priors[a], infos[a])

    graph = SemanticGraph()
    mcts = SemanticMCTS(cpuct=1.5, num_actions=4)

    families = {a: infos[a]["family"] for a in range(4)}
    mcts.expand(node_key, priors=priors, families=families)

    print("\nRoot action summary after expansion:")
    for row in mcts.root_action_summary(node_key):
        print(row)

    # Fake one transition update into graph
    graph.update_transition(
        node_key=node_key,
        action=0,
        family=infos[0]["family"],
        next_node_key=node_key,
        p_blocked=infos[0]["p_blocked"],
        p_moved=infos[0]["p_moved"],
        p_hud_changed=infos[0]["p_hud_changed"],
        p_success=infos[0]["p_success"],
    )

    print("\nGraph summary:")
    for row in graph.summary():
        print(row)


if __name__ == "__main__":
    main()