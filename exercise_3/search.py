"""
Step 4 — Blocksworld RAP Search
=================================
This function wires together the world model, the MCTS node, and the MCTS
algorithm into one end-to-end search.

Usage:
    best_node, best_reward, all_trajs = await blocksworld_rap_search(
        initial_state = "I have that, the orange block is clear, ...",
        goal          = "My goal is to have that the orange block is on top of the blue block.",
        prompts       = <dict loaded from my_mcts_prompts_update.json>,
        world_model   = <BlocksworldWorldModel instance>,
        rollouts      = 10,
        max_depth     = 4,
        r_alpha       = 0.5,
    )
"""

import re
from typing import Optional

from .mcts import MCTS
from .mcts_node import BlocksworldMCTSNode
from .world_model import BlocksworldWorldModel


async def blocksworld_rap_search(
    initial_state: str,
    goal: str,
    prompts: dict,
    world_model: BlocksworldWorldModel,
    rollouts: int = 10,
    max_depth: int = 4,
    r_alpha: float = 0.5,
    r1_default: float = 0.5,
    w_exp: float = 1.0,
) -> tuple[Optional[BlocksworldMCTSNode], float, list[str]]:
    """
    Run RAP-MCTS on a single Blocksworld problem.

    Parameters
    ----------
    initial_state : str  — plain-text state, e.g.
                           "I have that, the orange block is clear, the hand is empty, ..."
    goal          : str  — plain-text goal, e.g.
                           "My goal is to have that the orange block is on top of the blue block."
    prompts       : dict — loaded from my_mcts_prompts_update.json
    world_model   : BlocksworldWorldModel
    rollouts      : int  — number of MCTS iterations
    max_depth     : int  — maximum plan length
    r_alpha       : float — reward weighting (paper uses 0.5)
    r1_default    : float — initial r1 before the node is evaluated
    w_exp         : float — UCT exploration constant

    Returns
    -------
    (best_node, best_reward, trajectories)
      best_node    — the MCTS node with the best mean path reward
      best_reward  — its reward value
      trajectories — list of best-so-far prompt strings, one per rollout
    """

    # ------------------------------------------------------------------
    # Build the root node
    # The root prompt starts with the goal and the initial state, matching
    # the format used by the prompts and the apply_change / generate_all_actions
    # utilities.
    # ------------------------------------------------------------------
    root_prompt = (
        prompts["goal_prefix"]
        + goal.strip() + "\n"
        + prompts["state_prefix"].format(0)
        + " " + initial_state.strip() + "\n"
    )

    root = BlocksworldMCTSNode(
        prompt      = root_prompt,
        world_model = world_model,
        prompts     = prompts,
        goal        = goal,
        depth       = 0,
        r_alpha     = r_alpha,
        max_depth   = max_depth,
        r1_default  = r1_default,
        parent      = None,
        r0          = 0.0,
    )

    mcts = MCTS(
        w_exp       = w_exp,
        discount    = 1.0,
        aggr_reward = "mean",
        aggr_child  = "max",
    )

    # ------------------------------------------------------------------
    # Run rollouts
    # ------------------------------------------------------------------
    trajectories: list[str] = []

    for i in range(rollouts):
        await mcts.rollout(root)

        best_node, best_reward = mcts.best_terminal(root)
        trajectories.append(best_node.prompt)

        # Extract the action sequence for a readable progress log
        actions = re.findall(r"\[ACTION \d+\](.*)", best_node.prompt)
        plan_str = " → ".join(a.strip().rstrip(".") for a in actions) or "(none yet)"
        print(f"  Rollout {i+1:2d}/{rollouts}  reward={best_reward:.3f}  plan: {plan_str}")

    return best_node, best_reward, trajectories


# ------------------------------------------------------------------
# Helper: extract the action sequence from a node's prompt
# ------------------------------------------------------------------

def extract_plan(prompt: str) -> list[str]:
    """Return the list of action strings found in the node prompt."""
    return [a.strip().rstrip(".") for a in re.findall(r"\[ACTION \d+\](.*)", prompt)]
