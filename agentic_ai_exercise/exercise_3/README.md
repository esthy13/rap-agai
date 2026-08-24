# Exercise 3 — Quick Start

## 1. Prerequisites

- Python 3.11+ (the venv was built with Python 3.14)
- [`uv`](https://github.com/astral-sh/uv) (recommended) **or** standard `pip`
- API access: a key and base URL for an OpenAI-compatible endpoint

---

## 2. Environment Setup

### Create and activate the virtual environment

```bash
cd agentic-ai-exercise/agentic_ai_exercise/exercise_3

python -m venv rap_venv
source rap_venv/bin/activate        # macOS / Linux
# rap_venv\Scripts\activate         # Windows
```

### Install dependencies

```bash
# with uv (faster)
uv pip install -r requirements.txt

# or with pip
pip install -r requirements.txt
```

### Make the project importable from within the venv

```bash
# create a .pth file so `agentic_ai_exercise` can be imported
echo "/path/to/agentic-ai-exercise" \
  > rap_venv/lib/python3.*/site-packages/agentic_ai_exercise.pth
```

Replace `/path/to/agentic-ai-exercise` with the absolute path to the repo root
(the folder that contains `pyproject.toml`).

---

## 3. API Credentials

Create a `.env` file in the **repo root** (`agentic-ai-exercise/.env`):

```
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://your-endpoint-base-url
```

These are loaded automatically by `run.py` via `python-dotenv`.

---

## 4. Running the Experiment

All commands below assume:
- the `rap_venv` is active
- you are in the repo root (`agentic-ai-exercise/`)

### Full experiment (RAP + baseline, 2-step problems, 5 problems)

```bash
python -m agentic_ai_exercise.exercise_3.run --steps 2 --n 5
```

### RAP only (no baseline)

```bash
python -m agentic_ai_exercise.exercise_3.run --steps 2 --n 5 --no-baseline
```

### Baseline only (no RAP / no MCTS)

```bash
python -m agentic_ai_exercise.exercise_3.run --steps 2 --n 5 --no-rap
```

### All difficulty levels used in the paper

```bash
python -m agentic_ai_exercise.exercise_3.run --steps 2 --n 10
python -m agentic_ai_exercise.exercise_3.run --steps 4 --n 10
python -m agentic_ai_exercise.exercise_3.run --steps 6 --n 10
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `2` | Problem difficulty: `2`, `4`, or `6` solution steps |
| `--n` | `5` | Number of problems to evaluate |
| `--rollouts` | `10` | MCTS rollouts per problem |
| `--depth` | `4` | Maximum plan depth in MCTS |
| `--no-rap` | off | Skip RAP / MCTS evaluation |
| `--no-baseline` | off | Skip direct-LLM baseline evaluation |
| `--output` | auto | Path for the JSON results file |

---

## 5. Results

Results are saved to `exercise_3/results/` as JSON files named:

```
step{N}_n{M}_{YYYYMMDD_HHMMSS}.json
```

The file is written **incrementally** after every problem, so a partial run is
never lost.

Example result entry:

```json
{
  "problem_id": "step_2_1",
  "rap": {
    "plan": ["Pick up the red block", "Stack the red block on top of the blue block"],
    "goal_reached": true,
    "valid": true,
    "steps": 2,
    "reward": 0.948
  },
  "baseline": {
    "plan": ["Pick up the red block", "Put down the red block"],
    "goal_reached": false,
    "valid": true,
    "steps": 2
  }
}
```

A summary section at the top of each file shows aggregate success rates for
RAP and the baseline side by side.


### Note
The pipeline is failing due to the fact that, in order to use some of the original code, we had to copy the original RAP repository into ours.
