"""
Created on Sat Mar 21 10:49:21 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

from pathlib import Path
import numpy as np

from models.world_model import WorldModel

###############################################################################
# Get latents from trained model on batch of observarionts
###############################################################################
def batched_encode(model, obs: np.ndarray, batch_size: int = 512) -> np.ndarray:
    outputs = []

    n = len(obs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = obs[start:end]
        g = model.encode_obs(batch, training=False)
        outputs.append(g.numpy())

    return np.concatenate(outputs, axis=0)

###############################################################################
# Execute delta extractions
###############################################################################
def main(
    data_path: str = "data/trigger_door_random.npz",
    weights_path: str = "artifacts/world_model/best.weights.h5",
    save_path: str = "artifacts/deltas/trigger_door_deltas.npz",
    batch_size: int = 512,
) -> None:
    
    # Create saving dir dor analysis
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load stored dataset from played game
    data = np.load(data_path)
    
    # Query obs and next obs
    obs = data["obs"].astype(np.float32)
    next_obs = data["next_obs"].astype(np.float32)

    # Load trained model
    model = WorldModel()
    model.build([(None, 5, 5, 10), (None,)])
    model.load_weights(weights_path)
    
    # Extract latent from this obs
    print("Encoding obs...")
    g = batched_encode(model, obs, batch_size=batch_size)
    
    # Extract latent from next obs
    print("Encoding next_obs...")
    g_next = batched_encode(model, next_obs, batch_size=batch_size)
    
    # Calculate difference from curr and next latents
    delta = g_next - g

    np.savez_compressed(
        save_path,
        g=g.astype(np.float32),
        g_next=g_next.astype(np.float32),
        delta=delta.astype(np.float32),
        obs=data["obs"].astype(np.float32),
        next_obs=data["next_obs"].astype(np.float32),
        action=data["action"].astype(np.int32),
        reward=data["reward"].astype(np.float32),
        done=data["done"].astype(np.bool_),
        moved=data["moved"].astype(np.bool_),
        blocked=data["blocked"].astype(np.bool_),
        hud_changed=data["hud_changed"].astype(np.bool_),
        success=data["success"].astype(np.bool_),
        episode_id=data["episode_id"].astype(np.int32),
        step_id=data["step_id"].astype(np.int32),
    )

    print(f"Saved delta dataset to: {save_path}")
    print("g shape:", g.shape)
    print("g_next shape:", g_next.shape)
    print("delta shape:", delta.shape)


if __name__ == "__main__":
    main()