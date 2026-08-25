# RAP for Agentic AI

An experimental implementation of **Reasoning via Planning (RAP)** for language-model agents. The project combines an LLM-based world model with Monte Carlo Tree Search (MCTS) to solve Blocksworld planning tasks, then evaluates the search-based agent against a direct-LLM baseline.

Developed as part of the Agentic AI Seminar (Summer Semester 2026) at Bielefeld University, this repository showcases practical work in agentic systems, search algorithms, prompt engineering, and reproducible evaluation.

## Highlights

- Implements a RAP-style planning pipeline for Blocksworld.
- Uses an LLM as both a state-transition model and action evaluator.
- Applies MCTS to explore and rank candidate plans.
- Compares search-augmented reasoning with a direct prompting baseline.
- Evaluates 2-, 4-, and 6-step planning problems and stores structured JSON results.
- Includes reusable AgentScope chat and embedding examples.

## Project structure

```text
.
├── example.py                # Minimal AgentScope chat agent
├── embedding_example.py      # Embedding and chat example
├── exercise_3/
│   ├── run.py                # Experiment CLI
│   ├── config.py             # Shared paths and model identifiers
│   ├── world_model.py        # LLM-based transition and scoring model
│   ├── mcts.py               # Monte Carlo Tree Search
│   ├── mcts_node.py          # Blocksworld search node
│   ├── search.py             # RAP search orchestration
│   ├── baseline.py           # Direct-LLM comparison
│   ├── utils.py              # PDDL parsing and plan evaluation
│   ├── RAP/                  # RAP reference code and benchmark data
│   └── results/              # Example experiment outputs
└── tests/                    # Automated tests
```

## Setup

Requirements: Python 3.13+, [`uv`](https://docs.astral.sh/uv/), and access to an OpenAI-compatible model endpoint.

```bash
git clone https://github.com/esthy13/rap-agai.git
cd rap-agai
uv pip install -e '.[dev]'
```

Create a `.env` file in the repository root:

```dotenv
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-endpoint.example/v1
```

Never commit credentials; `.env` is ignored by Git.

## Run the experiments

Run RAP and the direct-LLM baseline on five 2-step tasks:

```bash
uv run python -m exercise_3.run --steps 2 --n 5
```

Useful variants:

```bash
# RAP only
uv run python -m exercise_3.run --steps 4 --n 10 --no-baseline

# Direct-LLM baseline only
uv run python -m exercise_3.run --steps 6 --n 10 --no-rap

# Tune the search budget
uv run python -m exercise_3.run --steps 4 --n 10 --rollouts 20 --depth 6
```

Results are written incrementally to `exercise_3/results/`, preserving partial progress if a run is interrupted. See [`exercise_3/README.md`](exercise_3/README.md) for the full CLI reference and result format.

## Quality checks

```bash
uv run ruff check .
uv run pytest
```

GitLab CI runs linting, tests, and generated API documentation.

## Technical focus

This project demonstrates experience with Python, asynchronous LLM APIs, AgentScope, MCTS, PDDL planning, experiment design, baseline comparison, structured evaluation, and CI/CD.

## Attribution

The approach and bundled reference implementation build on *Reasoning with Language Model is Planning with World Model*. Third-party RAP code retains its original license in [`exercise_3/RAP/LICENSE`](exercise_3/RAP/LICENSE).

## License

The repository is released under the terms in [`LICENSE`](LICENSE).
