"""
Step 5 — Blocksworld Utilities
================================
Three responsibilities:

  1. parse_pddl(filepath, encoded_objects)
        Reads a PDDL instance file and returns (initial_state_text, goal_text)
        in the exact format used by the RAP prompts — no tarski dependency.

  2. execute_plan(initial_state_text, actions)
        Deterministically simulates a sequence of actions on a state using
        the real Blocksworld rules (no LLM). Returns the final state text.

  3. evaluate_plan(initial_state_text, goal_text, actions)
        Returns True if executing `actions` from `initial_state` reaches `goal`.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------
# 1. PDDL Parser
# ------------------------------------------------------------------

# Letter → colour mapping from bw_config.yaml
ENCODED_OBJECTS: dict[str, str] = {
    "a": "red block",     "b": "blue block",   "c": "orange block",
    "d": "yellow block",  "e": "white block",  "f": "magenta block",
    "g": "black block",   "h": "cyan block",   "i": "green block",
    "j": "violet block",  "k": "silver block", "l": "gold block",
}


def parse_pddl(filepath: str | Path) -> tuple[str, str]:
    """
    Parse a Blocksworld PDDL instance file.

    Returns
    -------
    (initial_state_text, goal_text)
        Both strings are in the RAP prompt format, e.g.:
            initial_state_text = "the red block is clear, the hand is empty, ..."
            goal_text          = "the blue block is on top of the red block"
    """
    text = Path(filepath).read_text()

    # ---- objects -------------------------------------------------------
    obj_match = re.search(r":objects\s+(.*?)\s*\)", text, re.DOTALL)
    raw_objs  = obj_match.group(1).split() if obj_match else []
    # map each letter to its colour name
    obj_map   = {o: ENCODED_OBJECTS[o] for o in raw_objs if o in ENCODED_OBJECTS}

    # ---- init ----------------------------------------------------------
    init_match = re.search(r":init\s+(.*?)\)\s*\(:goal", text, re.DOTALL)
    init_text  = init_match.group(1) if init_match else ""
    init_preds = _parse_predicates(init_text, obj_map)

    # ---- goal ----------------------------------------------------------
    # Extract the raw goal section, then parse all atoms from it.
    # We locate the (:goal ...) block by finding the matching closing paren.
    goal_section_match = re.search(r":goal\s*\(and(.*)", text, re.DOTALL)
    if goal_section_match:
        goal_section = goal_section_match.group(1)
    else:
        # Single-atom goal (no 'and')
        goal_section_match = re.search(r":goal\s*(\(.*)", text, re.DOTALL)
        goal_section = goal_section_match.group(1) if goal_section_match else ""
    goal_preds = _parse_predicates(goal_section, obj_map)

    # ---- format to natural language ------------------------------------
    init_nl = _predicates_to_nl(init_preds)
    goal_nl = _predicates_to_nl(goal_preds)

    return init_nl, goal_nl


def _parse_predicates(text: str, obj_map: dict[str, str]) -> list[str]:
    """Extract (predicate-name arg1 arg2 ...) atoms from PDDL text."""
    results = []
    for m in re.finditer(r"\((\w[\w-]*)((?:\s+\w+)*)\)", text):
        pred = m.group(1).strip()
        args = [obj_map.get(a.strip(), a.strip()) for a in m.group(2).split() if a.strip()]
        results.append((pred, args))
    return results


def _predicates_to_nl(predicates: list[tuple[str, list[str]]]) -> str:
    """Convert structured predicates to a comma-separated natural language string."""
    templates = {
        "ontable":    "the {} is on the table",
        "clear":      "the {} is clear",
        "handempty":  "the hand is empty",
        "on":         "the {} is on top of the {}",
    }
    parts = []
    for pred, args in predicates:
        tmpl = templates.get(pred)
        if tmpl:
            parts.append(tmpl.format(*args))
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


# ------------------------------------------------------------------
# 2. Deterministic Blocksworld State Machine
# ------------------------------------------------------------------

@dataclass
class BlocksState:
    """Structured state for deterministic plan execution."""
    on_table: set = field(default_factory=set)    # colours on the table
    on:       dict = field(default_factory=dict)  # colour → colour (above → below)
    clear:    set = field(default_factory=set)    # colours with nothing on top
    holding:  Optional[str] = None               # colour currently held


def state_from_text(text: str) -> BlocksState:
    """Parse a natural-language state string into a BlocksState."""
    s = BlocksState()

    for m in re.finditer(r"the (\w+) block is on the table", text):
        s.on_table.add(m.group(1))
    for m in re.finditer(r"the (\w+) block is clear", text):
        s.clear.add(m.group(1))
    for m in re.finditer(r"the (\w+) block is on top of the (\w+) block", text):
        s.on[m.group(1)] = m.group(2)
    m = re.search(r"hand is holding the (\w+) block", text)
    if m:
        s.holding = m.group(1)

    return s


def state_to_text(s: BlocksState) -> str:
    """Convert a BlocksState back to natural-language format."""
    parts = []
    if s.holding:
        parts.append(f"the {s.holding} block is in the hand")
        parts.append(f"the hand is holding the {s.holding} block")
    else:
        parts.append("the hand is empty")
    for c in sorted(s.clear):
        parts.append(f"the {c} block is clear")
    for above, below in sorted(s.on.items()):
        parts.append(f"the {above} block is on top of the {below} block")
    for c in sorted(s.on_table):
        parts.append(f"the {c} block is on the table")

    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "I have that, " + ", ".join(parts[:-1]) + f" and {parts[-1]}."


def apply_action(s: BlocksState, action: str) -> Optional[BlocksState]:
    """
    Apply one Blocksworld action to a state.
    Returns the new state, or None if the action is invalid.
    """
    import copy
    ns = copy.deepcopy(s)
    action = action.strip().rstrip(".")

    # Pick up the X block
    m = re.match(r"(?i)pick up the (\w+) block", action)
    if m:
        c = m.group(1).lower()
        if ns.holding or c not in ns.on_table or c not in ns.clear:
            return None
        ns.on_table.discard(c)
        ns.clear.discard(c)
        ns.holding = c
        return ns

    # Unstack the X block from on top of the Y block
    m = re.match(r"(?i)unstack the (\w+) block from on top of the (\w+) block", action)
    if m:
        c, base = m.group(1).lower(), m.group(2).lower()
        if ns.holding or ns.on.get(c) != base or c not in ns.clear:
            return None
        del ns.on[c]
        ns.clear.discard(c)
        ns.clear.add(base)
        ns.holding = c
        return ns

    # Put down the X block
    m = re.match(r"(?i)put down the (\w+) block", action)
    if m:
        c = m.group(1).lower()
        if ns.holding != c:
            return None
        ns.holding = None
        ns.on_table.add(c)
        ns.clear.add(c)
        return ns

    # Stack the X block on top of the Y block
    m = re.match(r"(?i)stack the (\w+) block on top of the (\w+) block", action)
    if m:
        c, base = m.group(1).lower(), m.group(2).lower()
        if ns.holding != c or base not in ns.clear:
            return None
        ns.holding = None
        ns.on[c] = base
        ns.clear.discard(base)
        ns.clear.add(c)
        return ns

    return None   # unknown action format


def execute_plan(initial_state_text: str, actions: list[str]) -> tuple[str, bool]:
    """
    Simulate a plan step by step using the real Blocksworld rules.

    Returns
    -------
    (final_state_text, valid)
        valid = False if any action was illegal (invalid plan).
    """
    # Strip the "I have that," prefix if present
    clean = re.sub(r"^I have that,?\s*", "", initial_state_text.strip(), flags=re.IGNORECASE)
    state = state_from_text(clean)

    for action in actions:
        new_state = apply_action(state, action)
        if new_state is None:
            return state_to_text(state), False
        state = new_state

    return state_to_text(state), True


# ------------------------------------------------------------------
# 3. Evaluation
# ------------------------------------------------------------------

def goal_met(final_state_text: str, goal_text: str) -> bool:
    """Check whether all goal conditions appear in the final state."""
    goals = re.findall(r"the \w+ block is on top of the \w+ block", goal_text)
    final = final_state_text.lower()
    return all(g in final for g in goals)


def evaluate_plan(
    initial_state_text: str,
    goal_text: str,
    actions: list[str],
) -> dict:
    """
    Full evaluation of a plan.

    Returns a dict with:
      valid        — True if every action was legal
      goal_reached — True if the final state satisfies the goal
      steps        — number of actions executed
      final_state  — natural-language description of the final state
    """
    final_state, valid = execute_plan(initial_state_text, actions)
    reached = goal_met(final_state, goal_text)
    return {
        "valid":        valid,
        "goal_reached": reached,
        "steps":        len(actions),
        "final_state":  final_state,
    }


# ------------------------------------------------------------------
# Ground-truth plan helpers (from step_N.json format)
# ------------------------------------------------------------------

def pddl_plan_to_actions(pddl_plan: str) -> list[str]:
    """
    Convert a PDDL plan string like "(pick-up yellow)\n(stack yellow orange)\n"
    to a list of natural-language actions.
    """
    action_map = {
        "pick-up":  lambda args: f"Pick up the {args[0]} block",
        "put-down": lambda args: f"Put down the {args[0]} block",
        "stack":    lambda args: f"Stack the {args[0]} block on top of the {args[1]} block",
        "unstack":  lambda args: f"Unstack the {args[0]} block from on top of the {args[1]} block",
    }
    actions = []
    for line in pddl_plan.strip().splitlines():
        line = line.strip().strip("()")
        if not line:
            continue
        parts = line.split()
        verb, args = parts[0], parts[1:]
        if verb in action_map:
            actions.append(action_map[verb](args))
    return actions
