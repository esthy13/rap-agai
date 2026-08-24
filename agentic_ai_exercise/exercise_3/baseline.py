"""
Step 6 — Baseline: Direct LLM Planning (no MCTS)
==================================================
Given a Blocksworld problem, ask the LLM directly for a plan using the
same baseline_action few-shot prompt from my_mcts_prompts_update.json.
This is the comparison point for RAP.
"""

import re

from agentscope.model import OpenAIChatModel

from agentic_ai_exercise import QWEN3_VL_4B_Instruct


async def baseline_plan(
    initial_state: str,
    goal: str,
    prompts: dict,
    model: OpenAIChatModel,
) -> list[str]:
    """
    Ask the LLM to produce a plan directly, with no search.

    Uses the same baseline_action few-shot prompt as the original RAP repo,
    so the comparison is fair (same context, same model, different strategy).

    Parameters
    ----------
    initial_state : str  — "I have that, the red block is clear, ..."
    goal          : str  — "My goal is to have that the blue block is on top of ..."
    prompts       : dict — loaded from my_mcts_prompts_update.json
    model         : OpenAIChatModel

    Returns
    -------
    list[str] — the list of natural-language action strings extracted from the
                LLM response (empty list if parsing fails)
    """
    # Build the prompt exactly as in the original run_blocksworld.py
    # The baseline_action prompt already contains 4 complete plan examples.
    user_prompt = (
        prompts["baseline_action"]
        + "\n[STATEMENT]\n"
        + f"As initial conditions I have that, {initial_state.strip().lstrip('I have that,').strip()}.\n"
        + f"{goal.strip()}\n\n"
        + "My plan is as follows:\n\n[PLAN]\n"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a block-stacking planner. "
                "Given a statement and a goal, output only the plan actions, "
                "one per line, ending with [PLAN END]. Do not explain."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    response = await model(messages, max_tokens=200, temperature=0.0)
    # AgentScope 1.0.20 returns ChatResponse with content blocks
    raw = "".join(
        block["text"]
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    )

    # Extract actions — everything before [PLAN END]
    plan_text = raw.split("[PLAN END]")[0].strip()

    # Parse individual action lines
    actions = _parse_actions(plan_text)
    return actions


def _parse_actions(plan_text: str) -> list[str]:
    """
    Extract valid Blocksworld action strings from the model's raw output.
    Handles both the natural-language format and PDDL-style format.
    """
    actions = []
    patterns = [
        r"(?i)(pick up the \w+ block)",
        r"(?i)(unstack the \w+ block from on top of the \w+ block)",
        r"(?i)(put down the \w+ block)",
        r"(?i)(stack the \w+ block on top of the \w+ block)",
    ]
    for line in plan_text.splitlines():
        line = line.strip().rstrip(".")
        if not line:
            continue
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                # Capitalise first letter for consistency
                action = m.group(1)
                actions.append(action[0].upper() + action[1:])
                break
    return actions
