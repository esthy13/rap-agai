"""
Step 7 — Blocksworld Experiment Runner
========================================
Reproduces the Blocksworld experiments from the RAP paper and compares
with a direct-LLM baseline (no MCTS).

Usage (from the agentic-ai-exercise root with the ama venv active):
    python -m agentic_ai_exercise.exercise_3.run

Options (via CLI flags or editing the CONFIG block below):
    --steps   2|4|6   dataset to use   (default: 2)
    --n       N       number of test instances  (default: all)
    --rollouts R      MCTS rollouts per problem (default: 10)
    --depth   D       max plan depth            (default: 4)
    --no-rap          skip RAP; evaluate baseline only
    --no-baseline     skip baseline; evaluate RAP only
"""

import asyncio
import json
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ── make the RAP utilities importable ──────────────────────────────
RAP_DIR = Path(__file__).parent / "RAP"
sys.path.insert(0, str(RAP_DIR))

from agentic_ai_exercise import ENV_PATH, QWEN3_VL_4B_Instruct
from agentic_ai_exercise.exercise_3.world_model import BlocksworldWorldModel
from agentic_ai_exercise.exercise_3.search import blocksworld_rap_search, extract_plan
from agentic_ai_exercise.exercise_3.baseline import baseline_plan
from agentic_ai_exercise.exercise_3.utils import (
    parse_pddl,
    evaluate_plan,
    pddl_plan_to_actions,
)

# ── Load API credentials ────────────────────────────────────────────
load_dotenv(ENV_PATH)
API_KEY  = os.environ["LLM_API_KEY"]
API_BASE = os.environ["LLM_BASE_URL"]

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "RAP" / "data" / "blocksworld"
PROMPTS_FILE = DATA_DIR / "my_mcts_prompts_update.json"

# ── Formatting helpers ───────────────────────────────────────────────

def fmt_state(raw: str) -> str:
    """
    Ensure the initial state has the 'I have that, ' prefix stripped so it
    can be embedded in 'As initial conditions I have that, {state}.'
    """
    raw = raw.strip()
    if raw.lower().startswith("i have that,"):
        raw = raw[len("i have that,"):].strip()
    raw = raw.rstrip(".")
    return raw


def fmt_goal(raw: str) -> str:
    """
    Ensure the goal has no leading 'My goal is to have that' prefix —
    used in 'My goal is to have that {goal}.'
    """
    raw = raw.strip().rstrip(".")
    for prefix in (
        "my goal is to have that",
        "my goal is to have",
        "my goal is that",
    ):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix):].strip()
    return raw


# ── Dataset loader ───────────────────────────────────────────────────

def load_dataset(steps: int, n: int | None = None) -> list[dict]:
    """
    Load up to `n` instances from step_{steps}.json.

    Each returned dict has:
        pddl_path      : Path   — absolute path to the PDDL file
        gt_pddl        : str    — ground-truth plan in PDDL format
        gt_actions     : list   — GT actions in natural language
        initial_state  : str    — NL initial state (stripped)
        goal           : str    — NL goal (stripped)
    """
    dataset_file = DATA_DIR.parent / f"step_{steps}.json"

    if not dataset_file.exists():
        dataset_file = DATA_DIR / f"step_{steps}.json"

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: step_{steps}.json")

    raw = json.loads(dataset_file.read_text())
    if n is not None:
        raw = raw[:n]

    instances = []
    for pddl_rel, gt_pddl, _ in raw:
        pddl_path = RAP_DIR / pddl_rel
        if not pddl_path.exists():
            print(f"  [WARN] PDDL not found, skipping: {pddl_path}")
            continue

        init_nl, goal_nl = parse_pddl(pddl_path)
        instances.append({
            "pddl_path":     pddl_path,
            "gt_pddl":       gt_pddl,
            "gt_actions":    pddl_plan_to_actions(gt_pddl),
            "initial_state": init_nl,
            "goal":          goal_nl,
        })

    return instances


# ── Single-problem runner ────────────────────────────────────────────

async def run_one(
    inst: dict,
    prompts: dict,
    world_model: BlocksworldWorldModel,
    rollouts: int,
    max_depth: int,
    run_rap: bool,
    run_base: bool,
) -> dict:
    """Run RAP and/or baseline on one problem, return result dict."""
    init_stripped = fmt_state(inst["initial_state"])
    goal_stripped = fmt_goal(inst["goal"])

    # --- Rebuild state / goal in the format the rest of the code expects ---
    # For RAP search we pass the full formatted strings
    rap_init  = f"I have that, {init_stripped}."
    rap_goal  = f"My goal is to have that {goal_stripped}."

    result = {
        "pddl":         str(inst["pddl_path"].name),
        "gt_actions":   inst["gt_actions"],
        "rap_actions":  None,
        "rap_correct":  None,
        "base_actions": None,
        "base_correct": None,
    }

    # ---- RAP ----
    if run_rap:
        print(f"\n  [RAP] {inst['pddl_path'].name}")
        try:
            best_node, best_reward, _ = await blocksworld_rap_search(
                initial_state = rap_init,
                goal          = rap_goal,
                prompts       = prompts,
                world_model   = world_model,
                rollouts      = rollouts,
                max_depth     = max_depth,
            )
            rap_actions = extract_plan(best_node.prompt)
        except Exception as e:
            print(f"  [RAP ERROR] {e}")
            rap_actions = []

        rap_eval = evaluate_plan(rap_init, inst["goal"], rap_actions)
        result["rap_actions"] = rap_actions
        result["rap_correct"] = rap_eval["goal_reached"]
        status = "✓" if rap_eval["goal_reached"] else "✗"
        print(f"  [RAP] {status} actions={rap_actions}")

    # ---- Baseline ----
    if run_base:
        print(f"\n  [Baseline] {inst['pddl_path'].name}")
        try:
            base_actions = await baseline_plan(
                initial_state = rap_init,
                goal          = rap_goal,
                prompts       = prompts,
                model         = world_model.model,
            )
        except Exception as e:
            print(f"  [Baseline ERROR] {e}")
            base_actions = []

        base_eval = evaluate_plan(rap_init, inst["goal"], base_actions)
        result["base_actions"] = base_actions
        result["base_correct"] = base_eval["goal_reached"]
        status = "✓" if base_eval["goal_reached"] else "✗"
        print(f"  [Baseline] {status} actions={base_actions}")

    return result


# ── Results saving ──────────────────────────────────────────────────

def save_results(
    results: list[dict],
    output_path: Path,
    config: dict,
    run_rap: bool,
    run_base: bool,
):
    """
    Write results to a JSON file.
    Called after every problem so partial results survive a crash.
    """
    n = len(results)
    summary: dict = {"n": n}
    if run_rap:
        rap_correct = sum(1 for r in results if r["rap_correct"])
        summary["rap_correct"] = rap_correct
        summary["rap_accuracy"] = round(rap_correct / n, 4) if n else 0
    if run_base:
        base_correct = sum(1 for r in results if r["base_correct"])
        summary["base_correct"] = base_correct
        summary["base_accuracy"] = round(base_correct / n, 4) if n else 0

    output = {
        "config":  config,
        "summary": summary,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))


# ── Results table ────────────────────────────────────────────────────

def print_results(results: list[dict], steps: int, run_rap: bool, run_base: bool):
    n  = len(results)
    sep = "─" * 72

    print(f"\n{sep}")
    print(f"  Blocksworld Experiment  |  steps={steps}  |  n={n} problems")
    print(sep)

    if run_rap:
        rap_correct = sum(1 for r in results if r["rap_correct"])
        print(f"  RAP-MCTS accuracy :  {rap_correct}/{n}  ({100*rap_correct/n:.1f}%)")

    if run_base:
        base_correct = sum(1 for r in results if r["base_correct"])
        print(f"  Baseline accuracy :  {base_correct}/{n}  ({100*base_correct/n:.1f}%)")

    if run_rap and run_base:
        both = sum(
            1 for r in results
            if r["rap_correct"] and not r["base_correct"]
        )
        print(f"  RAP wins over baseline: {both} problem(s)")

    print(sep)

    # Per-instance table
    header = f"  {'Instance':<30}  {'GT':^5}"
    if run_rap:  header += f"  {'RAP':^5}"
    if run_base: header += f"  {'Base':^5}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        row = f"  {r['pddl']:<30}  {len(r['gt_actions']):^5}"
        if run_rap:
            sym = "✓" if r["rap_correct"] else "✗"
            row += f"  {sym:^5}"
        if run_base:
            sym = "✓" if r["base_correct"] else "✗"
            row += f"  {sym:^5}"
        print(row)

    print(sep)


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="RAP Blocksworld experiment")
    parser.add_argument("--steps",       type=int,  default=2,    choices=[2, 4, 6])
    parser.add_argument("--n",           type=int,  default=None, help="number of instances")
    parser.add_argument("--rollouts",    type=int,  default=10)
    parser.add_argument("--depth",       type=int,  default=4)
    parser.add_argument("--no-rap",      action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON (default: results/step<N>_<timestamp>.json)",
    )
    args = parser.parse_args()

    run_rap  = not args.no_rap
    run_base = not args.no_baseline

    # ── Output path ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(__file__).parent / "results"
        n_tag = f"_n{args.n}" if args.n else ""
        output_path = results_dir / f"step{args.steps}{n_tag}_{timestamp}.json"

    config = {
        "steps":     args.steps,
        "n":         args.n,
        "rollouts":  args.rollouts,
        "depth":     args.depth,
        "run_rap":   run_rap,
        "run_base":  run_base,
        "model":     QWEN3_VL_4B_Instruct,
        "timestamp": timestamp,
    }

    # ── Load shared resources ──────────────────────────────────────
    prompts = json.loads(PROMPTS_FILE.read_text())

    world_model = BlocksworldWorldModel(
        prompts    = prompts,
        model_name = QWEN3_VL_4B_Instruct,
        api_key    = API_KEY,
        api_base   = API_BASE,
    )

    print(f"\nLoading step_{args.steps} dataset …")
    instances = load_dataset(args.steps, args.n)
    print(f"  {len(instances)} instances loaded.")

    # ── Run experiments ────────────────────────────────────────────
    results = []
    for idx, inst in enumerate(instances, 1):
        print(f"\n{'='*60}")
        print(f"  Problem {idx}/{len(instances)}:  {inst['pddl_path'].name}")
        print(f"  Init : {inst['initial_state'][:80]}...")
        print(f"  Goal : {inst['goal']}")
        print(f"  GT   : {inst['gt_actions']}")

        res = await run_one(
            inst        = inst,
            prompts     = prompts,
            world_model = world_model,
            rollouts    = args.rollouts,
            max_depth   = args.depth,
            run_rap     = run_rap,
            run_base    = run_base,
        )
        results.append(res)

        # Save after every problem — survives interruption
        save_results(results, output_path, config, run_rap, run_base)
        print(f"  [saved → {output_path}]")

    # ── Print and save final summary ───────────────────────────────
    print_results(results, args.steps, run_rap, run_base)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
