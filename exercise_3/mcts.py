"""
Step 3 — MCTS Algorithm
========================
Standard Monte Carlo Tree Search adapted for RAP:

  - Uses the "prior" variant (prior=True), which means the selection phase
    itself expands nodes as it walks down the tree — no separate simulation
    step is needed.

  - UCT formula (Upper Confidence Bound for Trees):
        UCT(node) = M[node] + w_exp * sqrt( log(N[parent]) / N[node] )
    where M[node] is the best cumulative reward seen through this node.

  - Aggregation settings used for Blocksworld (matching the original paper):
        aggr_reward = 'mean'   (divide cumulative reward by path length)
        aggr_child  = 'max'    (use best seen reward for UCT, not average)
"""

import math
from collections import defaultdict

from .mcts_node import BlocksworldMCTSNode


class MCTS:
    """
    Parameters
    ----------
    w_exp        : float  — exploration weight in UCT (paper uses 1.0)
    discount     : float  — reward discount per step (1.0 = no discount, it gives more importance to rewards in shorter paths)
    aggr_reward  : str    — 'mean' or 'sum' — how to accumulate path reward
    aggr_child   : str    — 'max' or 'mean' — which Q-value to use in UCT
    """

    def __init__(
        self,
        w_exp: float = 1.0,
        discount: float = 1.0,
        aggr_reward: str = "mean",
        aggr_child: str = "max",
    ):
        self.w_exp       = w_exp
        self.discount    = discount
        self.aggr_reward = aggr_reward
        self.aggr_child  = aggr_child

        # Per-node statistics (keyed by node object identity)
        self.Q: dict[BlocksworldMCTSNode, float] = defaultdict(float)
        self.N: dict[BlocksworldMCTSNode, int]   = defaultdict(int)
        self.M: dict[BlocksworldMCTSNode, float] = defaultdict(lambda: -math.inf)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def rollout(self, root: BlocksworldMCTSNode):
        """Run one full MCTS rollout from the root."""
        path = await self._select_prior(root)
        self._back_propagate(path)

    # ------------------------------------------------------------------
    # Phase 1 — Selection (with expansion built in, "prior" mode)
    # ------------------------------------------------------------------

    async def _select_prior(self, node: BlocksworldMCTSNode) -> list[BlocksworldMCTSNode]:
        """
        Walk down the tree using UCT, expanding each node we visit.
        Stop when we reach a terminal node or a node with no children.
        Returns the path (list of nodes from root to selected leaf).
        """
        path = [node]

        while not node.is_terminal:
            # Expand this node (no-op if already expanded)
            await self._expand(node)

            children = await node.find_children()
            if not children:
                # Dead end — no valid actions from here
                break

            # Pick the child with the best UCT score
            node = self._uct_select(node)
            path.append(node)

        # Expand the terminal/leaf too (marks it as visited, computes its r1)
        await self._expand(node)

        return path

    # ------------------------------------------------------------------
    # Phase 2 — Expansion
    # ------------------------------------------------------------------

    async def _expand(self, node: BlocksworldMCTSNode):
        """
        Trigger child generation if not done yet.
        find_children() handles the LLM calls internally.
        """
        await node.find_children()

    # ------------------------------------------------------------------
    # Phase 3 — Back-propagation
    # ------------------------------------------------------------------

    def _back_propagate(self, path: list[BlocksworldMCTSNode]):
        """
        Walk the path in reverse, accumulating reward and updating Q, N, M.

        With aggr_reward='mean', we divide the cumulative reward by the
        number of steps, so deeper nodes are not unfairly penalised.
        """
        cumulative_reward = 0.0
        coeff = 1  # tracks sum of discount^k coefficients for 'mean'

        for node in reversed(path):
            cumulative_reward = cumulative_reward * self.discount + node.reward
            coeff             = coeff             * self.discount + 1

            c_reward = cumulative_reward / coeff if self.aggr_reward == "mean" else cumulative_reward

            # Q accumulates (we divide by N when computing mean in UCT)
            if node not in self.N:
                self.Q[node] = c_reward # if it's the first time we see this node then we just set the cumulative reward
            else:
                self.Q[node] += c_reward

            self.N[node] += 1
            self.M[node] = max(self.M[node], c_reward)

    # ------------------------------------------------------------------
    # UCT helpers
    # ------------------------------------------------------------------

    def _uct_score(self, node: BlocksworldMCTSNode, log_n_parent: float) -> float:
        """
        UCT score for a single node.
        Unvisited nodes (N=0) get a bonus equal to their prior reward.
        """
        if self.N[node] == 0:
            # Unvisited: use the node's prior reward + exploration bonus
            return node.reward + self.w_exp * math.sqrt(log_n_parent)

        if self.aggr_child == "max":
            exploitation = self.M[node]
        else:
            exploitation = self.Q[node] / self.N[node]

        exploration = self.w_exp * math.sqrt(log_n_parent / self.N[node])
        return exploitation + exploration

    def _uct_select(self, node: BlocksworldMCTSNode) -> BlocksworldMCTSNode:
        """Pick the child of `node` with the highest UCT score."""
        # log of parent's visit count (use 1 as floor to avoid log(0))
        log_n = math.log(max(self.N[node], 1))
        return max(node._children, key=lambda child: self._uct_score(child, log_n))

    # ------------------------------------------------------------------
    # Extract best solution after all rollouts
    # ------------------------------------------------------------------

    def best_terminal(
        self, node: BlocksworldMCTSNode, cumsum: float = 0.0, cnt: int = 0
    ) -> tuple[BlocksworldMCTSNode, float]:
        """
        Recursively find the terminal node with the highest mean reward
        across the path from root to that node.
        """
        if node.is_terminal:
            if node.visited:
                return node, (cumsum + node.reward) / (cnt + 1)
            return node, -math.inf

        children = node._children
        if not children:
            return node, -math.inf

        return max(
            (self.best_terminal(child, cumsum + node.reward, cnt + 1) for child in children),
            key=lambda x: x[1],
        )
