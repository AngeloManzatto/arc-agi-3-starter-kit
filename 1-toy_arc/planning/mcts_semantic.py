"""
Created on Sun Mar 22 10:46:31 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class ActionStats:
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    P: float = 0.0
    family: str | None = None


@dataclass
class TreeNode:
    key: tuple
    expanded: bool = False
    total_N: int = 0
    actions: dict = field(default_factory=dict)  # action -> ActionStats


class SemanticMCTS:
    def __init__(self, cpuct: float = 1.5, num_actions: int = 4):
        self.cpuct = cpuct
        self.num_actions = num_actions
        self.tree: dict[tuple, TreeNode] = {}

    def get_node(self, key: tuple) -> TreeNode:
        if key not in self.tree:
            self.tree[key] = TreeNode(key=key)
        return self.tree[key]

    def is_expanded(self, key: tuple) -> bool:
        return self.get_node(key).expanded

    def expand(self, key: tuple, priors: dict[int, float], families: dict[int, str]) -> None:
        node = self.get_node(key)
        if node.expanded:
            return

        for action in range(self.num_actions):
            node.actions[action] = ActionStats(
                N=0,
                W=0.0,
                Q=0.0,
                P=float(priors.get(action, 0.0)),
                family=families.get(action),
            )

        node.expanded = True

    def select_action(self, key: tuple) -> int:
        node = self.get_node(key)
        if not node.expanded:
            raise ValueError(f"Node {key} not expanded before selection.")

        best_action = None
        best_score = -1e18

        sqrt_N = math.sqrt(max(1, node.total_N))

        for action, stats in node.actions.items():
            u = self.cpuct * stats.P * sqrt_N / (1 + stats.N)
            score = stats.Q + u

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def backprop(self, path: list[tuple[tuple, int]], value: float) -> None:
        """
        path entries are (node_key, action)
        """
        for node_key, action in reversed(path):
            node = self.get_node(node_key)
            stats = node.actions[action]

            stats.N += 1
            stats.W += value
            stats.Q = stats.W / stats.N
            node.total_N += 1

    def root_action_summary(self, root_key: tuple) -> list[dict]:
        node = self.get_node(root_key)
        rows = []

        for action, stats in node.actions.items():
            rows.append({
                "action": action,
                "N": stats.N,
                "Q": stats.Q,
                "P": stats.P,
                "family": stats.family,
            })

        rows.sort(key=lambda x: (-x["N"], -x["Q"]))
        return rows