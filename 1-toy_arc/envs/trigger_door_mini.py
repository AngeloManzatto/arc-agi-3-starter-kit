"""
Created on Sat Mar 21 07:54:02 2026

@author: Angelo Antonio Manzatto
"""
###############################################################################
# libraries
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

###############################################################################
# Globals
###############################################################################

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_TO_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

###############################################################################
# libraries
###############################################################################

@dataclass
class Frame:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict

###############################################################################
# Env
###############################################################################

class TriggerDoorMini:
    """
    Minimal toy environment:
    - 5x5 grid
    - move agent
    - trigger cycles hud_state in {0,1,2}
    - exit only succeeds if hud_state == target_state
    """

    def __init__(
        self,
        grid_size: int = 5,
        n_states: int = 3,
        max_steps: int = 25,
        seed: Optional[int] = None,
    ) -> None:
        self.grid_size = grid_size
        self.n_states = n_states
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.agent_pos: tuple[int, int] = (0, 0)
        self.trigger_pos: tuple[int, int] = (0, 0)
        self.exit_pos: tuple[int, int] = (0, 0)
        self.walls: set[tuple[int, int]] = set()
        self.hud_state: int = 0
        self.target_state: int = 0
        self.step_count: int = 0

        # A few simple fixed layouts to keep early debugging easy
        self.layouts = [
            {(1, 1), (1, 2), (3, 3)},
            {(1, 3), (2, 3), (3, 1)},
            {(2, 1), (2, 2), (2, 3)},
            {(1, 1), (3, 1), (3, 2)},
        ]

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.walls = set(self.layouts[self.rng.integers(0, len(self.layouts))])

        free_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.walls
        ]

        # Sample distinct positions
        chosen = self.rng.choice(len(free_cells), size=3, replace=False)
        self.agent_pos = free_cells[int(chosen[0])]
        self.trigger_pos = free_cells[int(chosen[1])]
        self.exit_pos = free_cells[int(chosen[2])]

        self.hud_state = int(self.rng.integers(0, self.n_states))
        self.target_state = int(self.rng.integers(0, self.n_states))

        return self._get_obs()

    def step(self, action: int) -> Frame:
        assert action in ACTION_TO_DELTA, f"Invalid action: {action}"

        self.step_count += 1

        old_agent_pos = self.agent_pos
        old_hud_state = self.hud_state

        dr, dc = ACTION_TO_DELTA[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc

        blocked = False
        success = False

        if not self._in_bounds(nr, nc) or (nr, nc) in self.walls:
            blocked = True
        else:
            self.agent_pos = (nr, nc)

        # Trigger mechanic
        if self.agent_pos == self.trigger_pos and self.agent_pos != old_agent_pos:
            self.hud_state = (self.hud_state + 1) % self.n_states

        # Exit mechanic
        if self.agent_pos == self.exit_pos and self.hud_state == self.target_state:
            success = True

        done = success or (self.step_count >= self.max_steps)
        reward = 1.0 if success else 0.0

        info = {
            "moved": self.agent_pos != old_agent_pos,
            "blocked": blocked,
            "hud_changed": self.hud_state != old_hud_state,
            "success": success,
        }

        return Frame(
            obs=self._get_obs(),
            reward=reward,
            done=done,
            info=info,
        )

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _get_obs(self) -> np.ndarray:
        """
        Returns observation of shape [H, W, 10].
        Channels:
          0 wall
          1 agent
          2 trigger
          3 exit
          4-6 hud one-hot
          7-9 target one-hot
        """
        obs = np.zeros((self.grid_size, self.grid_size, 10), dtype=np.float32)

        for r, c in self.walls:
            obs[r, c, 0] = 1.0

        ar, ac = self.agent_pos
        tr, tc = self.trigger_pos
        er, ec = self.exit_pos

        obs[ar, ac, 1] = 1.0
        obs[tr, tc, 2] = 1.0
        obs[er, ec, 3] = 1.0

        obs[:, :, 4 + self.hud_state] = 1.0
        obs[:, :, 7 + self.target_state] = 1.0

        return obs

    def render_ascii(self) -> str:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for r, c in self.walls:
            grid[r][c] = "#"

        tr, tc = self.trigger_pos
        er, ec = self.exit_pos
        ar, ac = self.agent_pos

        grid[tr][tc] = "T"
        grid[er][ec] = "E"
        grid[ar][ac] = "A"

        lines = [" ".join(row) for row in grid]
        lines.append(f"hud_state={self.hud_state} target_state={self.target_state}")
        return "\n".join(lines)