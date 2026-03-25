"""
Created on Sat Mar 21 07:59:22 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

import numpy as np
from envs.trigger_door_mini import TriggerDoorMini

###############################################################################
# Run env
###############################################################################

def main() -> None:
    env = TriggerDoorMini(seed=42)
    obs = env.reset()

    print("Initial observation shape:", obs.shape)
    print(env.render_ascii())
    print("-" * 40)

    rng = np.random.default_rng(123)

    done = False
    total_reward = 0.0
    while not done:
        action = int(rng.integers(0, 4))
        result = env.step(action)

        total_reward += result.reward

        print(f"action={action}")
        print(result.info)
        print(env.render_ascii())
        print("-" * 40)

        done = result.done

    print("Episode finished.")
    print("Total reward:", total_reward)

if __name__ == "__main__":
    main()