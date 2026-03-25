"""
Created on Sun Mar 22 10:45:27 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class EdgeStats:
    visits: int = 0
    family_counts: Counter = field(default_factory=Counter)
    next_node_counts: Counter = field(default_factory=Counter)

    avg_p_blocked: float = 0.0
    avg_p_moved: float = 0.0
    avg_p_hud_changed: float = 0.0
    avg_p_success: float = 0.0

    def update_probs(
        self,
        p_blocked: float,
        p_moved: float,
        p_hud_changed: float,
        p_success: float,
    ) -> None:
        self.visits += 1
        alpha = 1.0 / self.visits

        self.avg_p_blocked = (1.0 - alpha) * self.avg_p_blocked + alpha * p_blocked
        self.avg_p_moved = (1.0 - alpha) * self.avg_p_moved + alpha * p_moved
        self.avg_p_hud_changed = (1.0 - alpha) * self.avg_p_hud_changed + alpha * p_hud_changed
        self.avg_p_success = (1.0 - alpha) * self.avg_p_success + alpha * p_success

    def dominant_family(self) -> str | None:
        if not self.family_counts:
            return None
        return self.family_counts.most_common(1)[0][0]


@dataclass
class GraphNode:
    key: tuple
    visits: int = 0
    edges: dict = field(default_factory=dict)  # action -> EdgeStats


class SemanticGraph:
    def __init__(self):
        self.nodes: dict[tuple, GraphNode] = {}

    def get_node(self, key: tuple) -> GraphNode:
        if key not in self.nodes:
            self.nodes[key] = GraphNode(key=key)
        return self.nodes[key]

    def update_transition(
        self,
        node_key: tuple,
        action: int,
        family: str,
        next_node_key: tuple,
        p_blocked: float,
        p_moved: float,
        p_hud_changed: float,
        p_success: float,
    ) -> None:
        node = self.get_node(node_key)

        if action not in node.edges:
            node.edges[action] = EdgeStats()

        edge = node.edges[action]
        edge.family_counts[family] += 1
        edge.next_node_counts[next_node_key] += 1
        edge.update_probs(
            p_blocked=p_blocked,
            p_moved=p_moved,
            p_hud_changed=p_hud_changed,
            p_success=p_success,
        )

        node.visits += 1

    def summary(self) -> list[dict]:
        rows = []

        for node_key, node in self.nodes.items():
            for action, edge in node.edges.items():
                rows.append({
                    "node_key": node_key,
                    "action": action,
                    "node_visits": node.visits,
                    "edge_visits": edge.visits,
                    "dominant_family": edge.dominant_family(),
                    "n_next_nodes": len(edge.next_node_counts),
                    "avg_p_blocked": edge.avg_p_blocked,
                    "avg_p_moved": edge.avg_p_moved,
                    "avg_p_hud_changed": edge.avg_p_hud_changed,
                    "avg_p_success": edge.avg_p_success,
                })

        return rows