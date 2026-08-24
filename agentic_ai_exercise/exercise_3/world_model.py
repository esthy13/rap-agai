"""
Step 1 — Blocksworld World Model
=================================
This module wraps AgentScope's OpenAIChatModel to give us the two LLM
capabilities that RAP needs for Blocksworld:

  1. predict_next_state(state, action)
        → asks the LLM what the world looks like after an action is taken
          (uses the few-shot world_update_* prompts from the RAP data)

  2. score_action(state0, state1, goal)
        → asks the LLM whether the new state is closer to the goal
          (replaces the log-likelihood scoring from the original LLaMA code)
          returns a float in [0, 1]
"""

from agentscope.model import OpenAIChatModel


class BlocksworldWorldModel:
    """
    Wraps an AgentScope chat model with the two LLM calls RAP needs.

    Parameters
    ----------
    prompts : dict
        The full prompt dictionary loaded from my_mcts_prompts_update.json.
    model_name : str
        The model identifier string (e.g. QWEN3_VL_4B_Instruct).
    api_key : str
    api_base : str
    """

    def __init__(self, prompts: dict, model_name: str, api_key: str, api_base: str):
        self.prompts = prompts

        # AgentScope model — async, so every method here is also async
        self.model = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            client_kwargs={"base_url": api_base},
            stream=False,
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    async def _call(self, system: str, user: str, max_tokens: int = 200) -> str:
        """Single async LLM call. Returns the raw text of the response."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        response = await self.model(messages, max_tokens=max_tokens, temperature=0.0)
        # AgentScope 1.0.20 returns ChatResponse with content blocks
        text = "".join(
            block["text"]
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        return text.strip()

    # ------------------------------------------------------------------
    # LLM call 1 — World update (predict next state)
    # ------------------------------------------------------------------

    async def predict_next_state(self, last_state: str, action: str) -> str:
        """
        Given the current state description and the action just taken,
        ask the LLM to predict what CHANGES happen, then apply them.

        Returns the new state as a plain text string (same format as the
        input state).
        """
        # Pick the right few-shot prompt based on action type
        if "Pick" in action:
            prompt_template = self.prompts["world_update_pickup"]
        elif "Unstack" in action:
            prompt_template = self.prompts["world_update_unstack"]
        elif "Put" in action:
            prompt_template = self.prompts["world_update_putdown"]
        else:  # Stack
            prompt_template = self.prompts["world_update_stack"]

        # The template has two {} placeholders: state and action
        filled_prompt = prompt_template.format(last_state, action)

        system = (
            "You are playing a block-stacking game. "
            "Complete the scenario exactly as shown in the examples above. "
            "Output only the [CHANGE] line, nothing else."
        )

        change_text = await self._call(system, filled_prompt, max_tokens=150)
        return change_text

    # ------------------------------------------------------------------
    # LLM call 2 — Action scoring (replaces log-likelihood)
    # ------------------------------------------------------------------

    async def score_action(self, state0: str, state1: str, goal: str) -> float:
        """
        Ask the LLM whether state1 is closer to the goal than state0.
        Returns 0.9 for "Yes" and 0.1 for "No".

        This replaces the log-likelihood scoring from the original LLaMA code.
        """
        filled_prompt = self.prompts["confidence"].format(state0, state1, goal)

        system = (
            "You are evaluating a block-stacking plan. "
            "Answer the final question with only 'Yes' or 'No'."
        )

        answer = await self._call(system, filled_prompt, max_tokens=10)

        # Parse the Yes/No answer into a probability
        if answer.lower().startswith("yes"):
            return 0.9
        else:
            return 0.1
