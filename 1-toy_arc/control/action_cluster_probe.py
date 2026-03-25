"""
Created on Sat Mar 21 11:32:45 2026

@author: Angelo Antonio Manzatto
"""
###############################################################################
# libraries
###############################################################################

from __future__ import annotations

import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from envs.trigger_door_mini import TriggerDoorMini
from models.world_model import WorldModel


def fit_cluster_model(
    delta_path: str = "artifacts/deltas/trigger_door_deltas.npz",
    n_clusters: int = 8,
    seed: int = 123,
):
    data = np.load(delta_path)
    delta = data["delta"].astype(np.float32)

    scaler = StandardScaler()
    delta_scaled = scaler.fit_transform(delta)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=20,
    )
    kmeans.fit(delta_scaled)

    return scaler, kmeans


def decode_obs_state(obs: np.ndarray) -> dict:
    obs_bin = (obs > 0.5).astype(np.int32)

    agent_pos = tuple(np.argwhere(obs_bin[:, :, 1] == 1)[0])
    trigger_pos = tuple(np.argwhere(obs_bin[:, :, 2] == 1)[0])
    exit_pos = tuple(np.argwhere(obs_bin[:, :, 3] == 1)[0])

    hud_state = int(np.argmax(obs_bin[0, 0, 4:7]))
    target_state = int(np.argmax(obs_bin[0, 0, 7:10]))

    return {
        "agent_pos": agent_pos,
        "trigger_pos": trigger_pos,
        "exit_pos": exit_pos,
        "hud_state": hud_state,
        "target_state": target_state,
    }


def l1_dist(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def score_predicted_obs(obs_pred: np.ndarray) -> float:
    """
    Very simple task-aware score for the toy.
    """
    d = decode_obs_state(obs_pred)

    if d["agent_pos"] == d["exit_pos"] and d["hud_state"] == d["target_state"]:
        return 100.0

    if d["hud_state"] != d["target_state"]:
        return -float(l1_dist(d["agent_pos"], d["trigger_pos"]))

    return -float(l1_dist(d["agent_pos"], d["exit_pos"]))


def probe_actions(
    obs: np.ndarray,
    model: WorldModel,
    scaler: StandardScaler,
    kmeans: KMeans,
):
    obs_batch = np.repeat(obs[None, ...], 4, axis=0).astype(np.float32)
    action_batch = np.arange(4, dtype=np.int32)

    next_obs_pred = model.predict((obs_batch, action_batch), verbose=0)

    g = model.encode_obs(obs_batch, training=False).numpy()
    g_next = model.encode_obs(next_obs_pred.astype(np.float32), training=False).numpy()
    delta_pred = g_next - g

    delta_scaled = scaler.transform(delta_pred)
    cluster_ids = kmeans.predict(delta_scaled)

    rows = []
    for action in range(4):
        score = score_predicted_obs(next_obs_pred[action])
        rows.append({
            "action": action,
            "cluster": int(cluster_ids[action]),
            "score": float(score),
            "pred_obs": next_obs_pred[action],
        })

    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    return rows


def main():
    env = TriggerDoorMini(seed=42)
    obs = env.reset()

    print(env.render_ascii())
    print()

    model = WorldModel()
    model.build([(None, 5, 5, 10), (None,)])
    model.load_weights("artifacts/world_model/best.weights.h5")

    scaler, kmeans = fit_cluster_model()

    rows = probe_actions(obs, model, scaler, kmeans)

    for row in rows:
        print(
            f"action={row['action']} | "
            f"cluster={row['cluster']} | "
            f"score={row['score']:.3f}"
        )


if __name__ == "__main__":
    main()