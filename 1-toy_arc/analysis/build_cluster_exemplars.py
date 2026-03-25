"""
Created on Sat Mar 21 11:19:34 2026

@author: Angelo Antonio Manzatto
"""
###############################################################################
# libraries
###############################################################################

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

###############################################################################
# libraries
###############################################################################
def main(
    delta_path: str = "artifacts/deltas/trigger_door_deltas.npz",
    out_dir: str = "artifacts/clusters",
    n_clusters: int = 8,
    seed: int = 123,
) -> None:
    
    # Output directory for clusters
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load deltas (g_next - g)
    data = np.load(delta_path)
    delta = data["delta"].astype(np.float32)
    
    # Normalize deltas
    scaler = StandardScaler()
    delta_scaled = scaler.fit_transform(delta)
    
    # Cluster deltas
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=20,
    )
    
    labels = kmeans.fit_predict(delta_scaled)
    centers = kmeans.cluster_centers_
    
    # Collect cluster metadata
    exemplar_indices = []
    rows = []

    for cluster_id in range(n_clusters):
        idx = np.where(labels == cluster_id)[0]
        cluster_points = delta_scaled[idx]
        center = centers[cluster_id]

        dists = np.sum((cluster_points - center[None, :]) ** 2, axis=1)
        best_local = int(np.argmin(dists))
        best_idx = int(idx[best_local])

        exemplar_indices.append(best_idx)

        rows.append({
            "cluster": cluster_id,
            "dataset_index": best_idx,
            "action": int(data["action"][best_idx]),
            "reward": float(data["reward"][best_idx]),
            "done": bool(data["done"][best_idx]),
            "moved": bool(data["moved"][best_idx]),
            "blocked": bool(data["blocked"][best_idx]),
            "hud_changed": bool(data["hud_changed"][best_idx]),
            "success": bool(data["success"][best_idx]),
            "episode_id": int(data["episode_id"][best_idx]),
            "step_id": int(data["step_id"][best_idx]),
        })

    exemplar_indices = np.asarray(exemplar_indices, dtype=np.int32)
    summary_df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)

    summary_csv = out_dir / "cluster_exemplars.csv"
    summary_df.to_csv(summary_csv, index=False)

    np.savez_compressed(
        out_dir / "cluster_exemplars.npz",
        exemplar_indices=exemplar_indices,
        labels=labels.astype(np.int32),
        obs=data["obs"][exemplar_indices].astype(np.float32),
        next_obs=data["next_obs"][exemplar_indices].astype(np.float32),
        action=data["action"][exemplar_indices].astype(np.int32),
        reward=data["reward"][exemplar_indices].astype(np.float32),
        done=data["done"][exemplar_indices].astype(np.bool_),
        moved=data["moved"][exemplar_indices].astype(np.bool_),
        blocked=data["blocked"][exemplar_indices].astype(np.bool_),
        hud_changed=data["hud_changed"][exemplar_indices].astype(np.bool_),
        success=data["success"][exemplar_indices].astype(np.bool_),
        episode_id=data["episode_id"][exemplar_indices].astype(np.int32),
        step_id=data["step_id"][exemplar_indices].astype(np.int32),
    )

    print(summary_df.to_string(index=False))
    print()
    print(f"Saved: {summary_csv}")
    print(f"Saved: {out_dir / 'cluster_exemplars.npz'}")


if __name__ == "__main__":
    main()