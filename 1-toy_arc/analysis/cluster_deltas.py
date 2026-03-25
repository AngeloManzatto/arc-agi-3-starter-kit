"""
Created on Sat Mar 21 10:54:30 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


###############################################################################
# Build dataframe from results
###############################################################################

def build_dataframe(data: np.lib.npyio.NpzFile, labels: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "cluster": labels.astype(int),
        "action": data["action"].astype(int),
        "reward": data["reward"].astype(float),
        "done": data["done"].astype(bool),
        "moved": data["moved"].astype(bool),
        "blocked": data["blocked"].astype(bool),
        "hud_changed": data["hud_changed"].astype(bool),
        "success": data["success"].astype(bool),
        "episode_id": data["episode_id"].astype(int),
        "step_id": data["step_id"].astype(int),
    })
    return df

###############################################################################
# Summarize clusters data
###############################################################################
def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cluster_id, g in df.groupby("cluster"):
        n = len(g)

        action_counts = g["action"].value_counts(normalize=True).sort_index()
        dominant_action = int(action_counts.idxmax())
        dominant_action_frac = float(action_counts.max())

        rows.append({
            "cluster": int(cluster_id),
            "n": int(n),
            "frac": float(n / len(df)),
            "moved_rate": float(g["moved"].mean()),
            "blocked_rate": float(g["blocked"].mean()),
            "hud_changed_rate": float(g["hud_changed"].mean()),
            "success_rate": float(g["success"].mean()),
            "done_rate": float(g["done"].mean()),
            "avg_reward": float(g["reward"].mean()),
            "dominant_action": dominant_action,
            "dominant_action_frac": dominant_action_frac,
        })

    out = pd.DataFrame(rows).sort_values(["cluster"]).reset_index(drop=True)
    return out

###############################################################################
# Action Vs Cluster table
###############################################################################
def build_action_table(df: pd.DataFrame) -> pd.DataFrame:
    tab = pd.crosstab(df["cluster"], df["action"], normalize="index")
    return tab

###############################################################################
# Save summary data
###############################################################################
def save_json_summary(summary_df: pd.DataFrame, save_path: Path) -> None:
    records = summary_df.to_dict(orient="records")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

###############################################################################
# Execute cluster delta analysis
###############################################################################

def main(
    delta_path: str = "artifacts/deltas/trigger_door_deltas.npz",
    out_dir: str = "artifacts/clusters",
    n_clusters: int = 8,
    seed: int = 123,
) -> None:
    
    # Create output dir for analysis
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load deltas (g_next - g)
    data = np.load(delta_path)
    delta = data["delta"].astype(np.float32)

    scaler = StandardScaler()
    delta_scaled = scaler.fit_transform(delta)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=20,
    )
    
    labels = kmeans.fit_predict(delta_scaled)

    df = build_dataframe(data, labels)
    summary_df = summarize_clusters(df)
    action_table = build_action_table(df)

    summary_csv = out_dir / "cluster_summary.csv"
    action_csv = out_dir / "cluster_action_distribution.csv"
    labels_npz = out_dir / "cluster_labels.npz"
    summary_json = out_dir / "cluster_summary.json"

    summary_df.to_csv(summary_csv, index=False)
    action_table.to_csv(action_csv)
    save_json_summary(summary_df, summary_json)

    np.savez_compressed(
        labels_npz,
        labels=labels.astype(np.int32),
    )

    print("Saved:")
    print(" -", summary_csv)
    print(" -", action_csv)
    print(" -", summary_json)
    print(" -", labels_npz)
    print()
    print(summary_df.to_string(index=False))
    print()
    print("Action distribution by cluster:")
    print(action_table)


if __name__ == "__main__":
    main()