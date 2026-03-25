from __future__ import annotations

from collections import Counter
import numpy as np

from envs.trigger_door_mini import TriggerDoorMini
from models.world_model import WorldModel
from models.transition_classifier import TransitionOutcomeClassifier

from planning.semantic_utils import (
    decode_obs_state,
    encode_state_to_obs,
    state_to_node_key,
    infer_family,
    probs_to_priors,
    coarse_transition,
    heuristic_value,
    is_success_state,
)
from planning.semantic_graph import SemanticGraph
from planning.mcts_semantic import SemanticMCTS


class SemanticMCTSAgent:
    def __init__(
        self,
        wm: WorldModel,
        clf: TransitionOutcomeClassifier,
        cpuct: float = 1.5,
        num_actions: int = 4,
        n_simulations: int = 64,
        max_depth: int = 6,
        prior_temperature: float = 1.0,
        gamma: float = 0.95,
        revisit_penalty_scale: float = 2.0,
        novelty_bonus: float = 0.75,
        step_revisit_penalty: float = 0.5,
    ):
        self.wm = wm
        self.clf = clf
        self.cpuct = cpuct
        self.num_actions = num_actions
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.prior_temperature = prior_temperature
        self.gamma = gamma

        self.revisit_penalty_scale = revisit_penalty_scale
        self.novelty_bonus = novelty_bonus
        self.step_revisit_penalty = step_revisit_penalty

        self.graph = SemanticGraph()
        self.mcts = SemanticMCTS(cpuct=cpuct, num_actions=num_actions)

    def predict_action_probs(self, state: dict):
        obs = encode_state_to_obs(state)
        g = self.wm.encode_obs(obs[None, ...].astype(np.float32), training=False).numpy()[0]
        g_batch = np.repeat(g[None, :], self.num_actions, axis=0).astype(np.float32)
        action_batch = np.arange(self.num_actions, dtype=np.int32)
        y_prob = self.clf.predict((g_batch, action_batch), verbose=0)
        return y_prob

    def expand_node(self, state: dict):
        node_key = state_to_node_key(state)

        if self.mcts.is_expanded(node_key):
            return

        y_prob = self.predict_action_probs(state)
        priors, infos = probs_to_priors(
            state=state,
            y_prob=y_prob,
            temperature=self.prior_temperature,
        )

        families = {a: infos[a]["family"] for a in range(self.num_actions)}
        self.mcts.expand(node_key, priors=priors, families=families)

    def simulate(self, root_state: dict):
        """
        One MCTS simulation with short-term memory:
        - visited_counter tracks node revisits within the simulation
        - revisit penalties discourage loops
        - novelty bonus rewards first-time node visits in the simulation
        """
        sim_state = {
            "agent_pos": root_state["agent_pos"],
            "trigger_pos": root_state["trigger_pos"],
            "exit_pos": root_state["exit_pos"],
            "hud_state": root_state["hud_state"],
            "target_state": root_state["target_state"],
            "wall_mask": root_state["wall_mask"].copy(),
            "grid_h": root_state["grid_h"],
            "grid_w": root_state["grid_w"],
        }

        path = []
        visited_counter = Counter()
        path_value_adjustment = 0.0

        for depth in range(self.max_depth):
            node_key = state_to_node_key(sim_state)

            # Count current node visit inside this simulation
            revisit_count_before = visited_counter[node_key]
            visited_counter[node_key] += 1

            # Add novelty / revisit shaping
            if revisit_count_before == 0:
                path_value_adjustment += (self.gamma ** depth) * self.novelty_bonus
            else:
                path_value_adjustment -= (self.gamma ** depth) * self.step_revisit_penalty * revisit_count_before

            # Expand unseen node and evaluate leaf
            if not self.mcts.is_expanded(node_key):
                self.expand_node(sim_state)
                leaf_value = heuristic_value(
                    sim_state,
                    revisit_count=max(0, visited_counter[node_key] - 1),
                )
                leaf_value -= self.revisit_penalty_scale * max(0, visited_counter[node_key] - 1)
                total_value = leaf_value + path_value_adjustment
                self.mcts.backprop(path, total_value)
                return

            # Terminal success
            if is_success_state(sim_state):
                total_value = (self.gamma ** depth) * 100.0 + path_value_adjustment
                self.mcts.backprop(path, total_value)
                return

            action = self.mcts.select_action(node_key)

            # Semantic predictions for selected action
            y_prob = self.predict_action_probs(sim_state)
            probs = y_prob[action]

            p_blocked = float(probs[0])
            p_moved = float(probs[1])
            p_hud_changed = float(probs[2])
            p_success = float(probs[3])

            family = infer_family(
                p_blocked=p_blocked,
                p_moved=p_moved,
                p_hud_changed=p_hud_changed,
                p_success=p_success,
            )

            next_state = coarse_transition(
                state=sim_state,
                action=action,
                p_blocked=p_blocked,
                p_hud_changed=p_hud_changed,
            )
            next_node_key = state_to_node_key(next_state)

            # Update semantic graph
            self.graph.update_transition(
                node_key=node_key,
                action=action,
                family=family,
                next_node_key=next_node_key,
                p_blocked=p_blocked,
                p_moved=p_moved,
                p_hud_changed=p_hud_changed,
                p_success=p_success,
            )

            path.append((node_key, action))
            sim_state = next_state

            if is_success_state(sim_state):
                total_value = (self.gamma ** (depth + 1)) * 100.0 + path_value_adjustment
                self.mcts.backprop(path, total_value)
                return

        # Depth-limit leaf
        final_key = state_to_node_key(sim_state)
        final_revisit_count = max(0, visited_counter[final_key] - 1)

        leaf_value = (self.gamma ** self.max_depth) * heuristic_value(
            sim_state,
            revisit_count=final_revisit_count,
        )
        leaf_value -= self.revisit_penalty_scale * final_revisit_count

        total_value = leaf_value + path_value_adjustment
        self.mcts.backprop(path, total_value)

    def select_action(self, obs: np.ndarray, debug: bool = False):
        root_state = decode_obs_state(obs)
        root_key = state_to_node_key(root_state)

        self.expand_node(root_state)

        for _ in range(self.n_simulations):
            self.simulate(root_state)

        root_summary = self.mcts.root_action_summary(root_key)

        if debug:
            print("Root key:", root_key)
            print("Root action summary:")
            for row in root_summary:
                print(row)

        best_row = sorted(root_summary, key=lambda x: (-x["N"], -x["Q"]))[0]
        return int(best_row["action"]), root_summary

    def reset_search_tree(self):
        self.mcts = SemanticMCTS(cpuct=self.cpuct, num_actions=self.num_actions)


def run_episode(
    env: TriggerDoorMini,
    agent: SemanticMCTSAgent,
    render: bool = False,
    reset_tree_each_step: bool = True,
):
    obs = env.reset()
    total_reward = 0.0
    steps = 0

    for step in range(env.max_steps):
        if reset_tree_each_step:
            agent.reset_search_tree()

        action, root_summary = agent.select_action(obs, debug=render)

        result = env.step(action)
        obs = result.obs
        total_reward += result.reward
        steps += 1

        if render:
            print(f"step={step} chosen_action={action}")
            print(env.render_ascii())
            print("-" * 80)

        if result.done:
            break

    return {
        "reward": total_reward,
        "steps": steps,
        "success": bool(total_reward > 0),
    }


def evaluate(
    n_episodes: int = 100,
    render: bool = False,
    n_simulations: int = 64,
    max_depth: int = 6,
    reset_tree_each_step: bool = True,
):
    env = TriggerDoorMini(seed=42)

    wm = WorldModel()
    wm.build([(None, 5, 5, 10), (None,)])
    wm.load_weights("artifacts/world_model/best.weights.h5")

    clf = TransitionOutcomeClassifier()
    clf.build([(None, 64), (None,)])
    clf.load_weights("artifacts/transition_classifier/best.weights.h5")

    agent = SemanticMCTSAgent(
        wm=wm,
        clf=clf,
        cpuct=1.5,
        num_actions=4,
        n_simulations=n_simulations,
        max_depth=max_depth,
        prior_temperature=1.0,
        gamma=0.95,
        revisit_penalty_scale=2.0,
        novelty_bonus=0.75,
        step_revisit_penalty=0.5,
    )

    results = []
    for i in range(n_episodes):
        out = run_episode(
            env=env,
            agent=agent,
            render=render and i < 2,
            reset_tree_each_step=reset_tree_each_step,
        )
        results.append(out)
        print(f"Processed episode: {i} | reward={out['reward']} | steps={out['steps']}")

    rewards = np.asarray([r["reward"] for r in results], dtype=np.float32)
    successes = np.asarray([r["success"] for r in results], dtype=np.float32)
    steps = np.asarray([r["steps"] for r in results], dtype=np.float32)

    print("\n=== Memory-Aware Semantic MCTS Results ===")
    print("Episodes:", n_episodes)
    print("n_simulations:", n_simulations)
    print("max_depth:", max_depth)
    print("Success rate:", float(np.mean(successes)))
    print("Avg reward:", float(np.mean(rewards)))
    print("Avg steps:", float(np.mean(steps)))
    print("Graph nodes:", len(agent.graph.nodes))

    graph_rows = agent.graph.summary()
    print("Graph edges tracked:", len(graph_rows))
    if graph_rows:
        print("\nSample graph rows:")
        for row in graph_rows[:10]:
            print(row)


if __name__ == "__main__":
    evaluate(
        n_episodes=100,
        render=False,
        n_simulations=64,
        max_depth=6,
        reset_tree_each_step=True,
    )