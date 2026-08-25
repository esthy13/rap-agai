"""
Step 2 — Blocksworld MCTS Node
================================
Each node represents one point in the search tree — a partial plan.

The node's `prompt` is the full text history built up so far, e.g.:

    [GOAL] My goal is to have the orange block on top of the blue block.
    [STATE 0] I have that, the orange block is clear, ...
    [ACTION 1] Pick up the orange block.
    [STATE 1] I have that, the orange block is in the hand, ...
    [ACTION 2] Stack the orange block on top of the blue block.
    ← depth 2, [STATE 2] not yet filled — added by _calculate_reward

When a node is expanded the steps are:
  1. _calculate_reward()  — LLM predicts [STATE depth], computes r1
  2. gen_fn()             — pure Python enumerates valid actions, async LLM scores them (r0)
  3. one child per action  — child prompt = current prompt + [ACTION depth+1] <action>
"""

import random
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure the RAP package directory is on the path regardless of how this
# module is imported (e.g. via exercise_3.__init__ or directly).
_RAP_DIR = Path(__file__).parent / "RAP"
if str(_RAP_DIR) not in sys.path:
    sys.path.insert(0, str(_RAP_DIR))

from rap.utils.blocksworld import apply_change, generate_all_actions  # noqa: E402

from .world_model import BlocksworldWorldModel  # noqa: E402


class BlocksworldMCTSNode:
    """
    One node in the MCTS tree for Blocksworld.

    Parameters
    ----------
    prompt      : str   — full text history up to (but not including) the
                         state that results from the last action
    world_model : BlocksworldWorldModel
    prompts     : dict  — the RAP prompt templates
    goal        : str   — goal description (used for scoring)
    depth       : int   — how many actions deep we are (root = 0)
    r_alpha     : float — weight balancing r0 and r1  (paper uses 0.5)
    max_depth   : int   — stop expanding beyond this depth
    r1_default  : float — r1 value before the node has been evaluated
    parent      : BlocksworldMCTSNode | None
    r0          : float — action-prior score assigned by the parent
    """

    def __init__(
        self,
        prompt: str,
        world_model: BlocksworldWorldModel,
        prompts: dict,
        goal: str,
        depth: int,
        r_alpha: float,
        max_depth: int,
        r1_default: float = 1.0,
        parent: Optional["BlocksworldMCTSNode"] = None,
        r0: float = 0.0,
    ):
        self.prompt      = prompt
        self.world_model = world_model
        self.prompts     = prompts
        self.goal        = goal
        self.depth       = depth
        self._r_alpha    = r_alpha
        self.max_depth   = max_depth
        self._r1         = r1_default  # updated by _calculate_reward
        self._r0         = r0          # set by the parent when creating this node
        self.parent      = parent

        self._visited  = False
        self._children: list["BlocksworldMCTSNode"] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def visited(self) -> bool:
        return self._visited

    @property
    def reward(self) -> float:
        """Combined reward: r0^alpha * r1^(1-alpha). Negative if either < 0."""
        if self._r0 < 0 or self._r1 < 0:
            return min(self._r0, self._r1)
        return self._r0 ** self._r_alpha * self._r1 ** (1 - self._r_alpha)

    @property
    def is_terminal(self) -> bool:
        """
        Terminal when:
          - r1 > 50  →  all goal conditions are satisfied  (success!)
          - depth >= max_depth  →  plan too long
          - reward < -1  →  the LLM predicted an impossible state
        """
        if self._r1 > 50:
            return True
        if self.depth >= self.max_depth:
            return True
        if self.reward < -1:
            return True
        return False

    # ------------------------------------------------------------------
    # Step 1 of expansion: world model predicts the new state (r1)
    # ------------------------------------------------------------------

    async def _calculate_reward(self):
        """
        Ask the LLM to simulate the state that results from the last action,
        then count how many goal conditions are met → r1.

        Depth 0 is the root (no action taken yet), so we skip it.
        """
        if self.depth == 0:
            self._r1 = 1.0
            return

        # Extract the state and action from the current prompt
        state_key  = self.prompts["state_prefix"].format(self.depth - 1)
        action_key = self.prompts["action_prefix"].format(self.depth)

        last_state  = re.search(rf".*{re.escape(state_key)}(.*)",  self.prompt, re.DOTALL)[1]
        last_action = re.search(rf".*{re.escape(action_key)}(.*)", self.prompt, re.DOTALL)[1]

        # Ask the LLM for the [CHANGE] description
        change_text = await self.world_model.predict_next_state(last_state.strip(), last_action.strip())

        # Extract just what comes after "[CHANGE]" if the model echoed the tag
        if "[CHANGE]" in change_text:
            change_text = change_text.split("[CHANGE]")[-1]

        # Apply the change to the previous state text to get the new state text
        prev_state_text = self.prompt.split(state_key)[-1].split(action_key)[0]
        new_state = apply_change(change_text, prev_state_text)

        # Append [STATE depth] new_state to the prompt
        self.prompt = (
            self.prompt
            + self.prompts["state_prefix"].format(self.depth)
            + " " + new_state + "\n"
        )

        # r1: count how many goal conditions are met in the new state
        goals    = re.findall(r"the [a-z]{0,10} block is on top of the [a-z]{0,10} block", self.goal)
        meetings = [g in new_state for g in goals]

        if goals and all(meetings):
            self._r1 = 100.0   # terminal success
        elif goals:
            self._r1 = sum(meetings) / len(meetings) + 0.5
        else:
            self._r1 = 0.5     # fallback if regex found nothing

    # ------------------------------------------------------------------
    # Step 2 of expansion: enumerate actions and score them (r0)
    # ------------------------------------------------------------------

    async def _get_children(self) -> list["BlocksworldMCTSNode"]:
        """Expand this node: compute r1, then generate all child nodes."""
        self._visited = True
        await self._calculate_reward()

        if self.is_terminal:
            return self._children

        # Extract the current state text (after the last [STATE depth] tag)
        state_key  = self.prompts["state_prefix"].format(self.depth)
        current_state = self.prompt.split(state_key)[-1].strip()

        # Pure-Python enumeration of valid actions — no LLM needed here
        raw_actions = generate_all_actions(current_state)

        # Build the next-action prompt for each action
        action_key = self.prompts["action_prefix"].format(self.depth + 1)
        child_prompts = [
            self.prompt + action_key + " " + action.capitalize() + ".\n"
            for action in raw_actions
        ]

        # Score each action: ask the LLM "is this a step toward the goal?"
        r0_scores = []
        for action, child_prompt in zip(raw_actions, child_prompts):
            # We need: state0 (before action), state1 (after action, not yet known)
            # We approximate by asking about the action itself relative to the goal.
            # For a cheaper approximation we reuse the confidence prompt with the
            # current state as both state0 and a description of the intended action.
            score = await self.world_model.score_action(
                state0=current_state,
                state1=action,   # placeholder — we pass the action text as "state1"
                goal=self.goal,
            )
            r0_scores.append(score)

        # Create one child node per action
        for child_prompt, r0 in zip(child_prompts, r0_scores):
            child = BlocksworldMCTSNode(
                prompt=child_prompt,
                world_model=self.world_model,
                prompts=self.prompts,
                goal=self.goal,
                depth=self.depth + 1,
                r_alpha=self._r_alpha,
                max_depth=self.max_depth,
                r1_default=self._r1,   # inherit parent's r1 as default
                parent=self,
                r0=r0,
            )
            self._children.append(child)

        return self._children

    # ------------------------------------------------------------------
    # Public interface used by MCTS
    # ------------------------------------------------------------------

    async def find_children(self) -> list["BlocksworldMCTSNode"]:
        """Return children, expanding this node if not yet done."""
        if not self._children and not self._visited:
            await self._get_children()
        return self._children

    async def find_one_child(self) -> "BlocksworldMCTSNode":
        """Return a random child (used during simulation/rollout)."""
        children = await self.find_children()
        return random.choice(children) if children else None

    # ------------------------------------------------------------------
    # Pretty-print helper (useful for debugging)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        plan = re.findall(r"\[ACTION \d+\](.*)", self.prompt)
        plan_str = " → ".join(a.strip() for a in plan) or "(root)"
        return f"Node(depth={self.depth}, r0={self._r0:.2f}, r1={self._r1:.2f}, plan={plan_str})"
