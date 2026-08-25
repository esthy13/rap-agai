# Exercise 3 — RAP: Reasoning via Planning with AgentScope

**Paper:** *Reasoning with Language Model is Planning with World Model*  
Hao et al., 2023 — [arXiv:2305.14992](https://arxiv.org/abs/2305.14992)

**Task:** Blocksworld — arrange coloured blocks into target stacks using a
sequence of pick/unstack/put-down/stack actions.

---

> **Presentation guide** — three sections below map directly to your three
> 2-3 minute blocks:
> - [Part 1 — Formal Approach](#part-1--the-formal-approach-23-min) — what the paper proposes
> - [Part 2 — Our Implementation](#part-2--our-implementation-23-min) — how we built it, file by file
> - [Part 3 — Evaluation](#part-3--evaluation-23-min) — how we measure correctness and compare

---

## Part 1 — The Formal Approach (2–3 min)

### The Problem: Why Normal LLMs Fail at Planning

When you ask a standard LLM "how do I stack these blocks?", it generates the
answer left-to-right in one shot. It has no ability to:
- Look ahead to see if a sequence of actions will actually reach the goal
- Backtrack when it realises it went down a dead end
- Explore multiple possible plans and compare them

RAP solves this by turning LLM reasoning into a **tree search problem**.

### The Core Idea: LLM as Both Reasoner and World Model

RAP gives the LLM **two jobs at once**:

| Role | Task | In Blocksworld |
|------|------|----------------|
| **Reasoner** | Proposes which action to try next | "Pick up the red block" |
| **World Model** | Predicts what the world looks like after that action, and scores how close we are to the goal | Predicts the new block configuration, then says "yes, this is closer to the goal" |

This is unusual — most planning systems need a *separate* hand-coded simulator.
RAP uses the LLM itself as the simulator via few-shot prompting.

### The Search Algorithm: Monte Carlo Tree Search (MCTS)

Instead of committing greedily to the first action that looks good, RAP builds
a **search tree** where:
- Each **node** = one specific state in the planning process (a partial plan)
- Each **edge** = one action taken
- Each node has a **reward** that measures how promising this partial plan is

MCTS runs in repeated **rollouts**. Each rollout does three things:

```
1. SELECT   — Walk down the tree from the root, choosing the most
              promising branch at each step using the UCT formula.

2. EXPAND   — At the selected node, ask the LLM:
              (a) "What does the world look like after this action?" → r1
              (b) "Is this action a step toward the goal?" → r0
              Create one child node per valid action.

3. BACKPROPAGATE — Walk back up the path just taken, updating the
                   running statistics (visit count, total reward) at
                   every node we passed through.
```

After all rollouts are done, the algorithm picks the path through the tree
that has the highest average reward — this is the predicted plan.

### The Reward Formula

Each node's reward combines two signals:

$$r(s, a) = r_0(s, a)^{\,\alpha} \cdot r_1(s, a)^{\,1-\alpha}$$

| Symbol | Name | Meaning | Value range |
|--------|------|---------|-------------|
| $r_0$ | Action score | How promising is this specific action from this state? | 0–1 |
| $r_1$ | State score | How close is the resulting state to the goal? | 0–100 |
| $\alpha$ | Balance weight | Set to 0.5 in the paper — equal weight to both | 0–1 |

- **r1 = 100** → all goal conditions are satisfied → **terminal success**
- **r1 = 0.5 + fraction** → some goal conditions met (e.g. 0.5 + 1/2 = 1.0 if half the goals are met)
- **r1 = 1.0** → root node (no action taken yet), neutral starting point

### Why MCTS and Not Greedy Search?

A greedy planner always picks the action with the highest immediate reward,
which can trap it in a local optimum. MCTS balances **exploitation** (follow
the best known path) and **exploration** (try branches that haven't been
visited much) using the **UCT formula**:

$$\text{UCT}(v) = \underbrace{M(v)}_{\text{best reward seen}} + \; w_{\exp} \cdot \underbrace{\sqrt{\frac{\ln N(\text{parent})}{N(v)}}}_{\text{exploration bonus}}$$

- $M(v)$ = best cumulative reward seen through node $v$ so far
- $N(v)$ = how many times node $v$ has been visited
- $w_{\exp}$ = exploration weight (set to 1.0)

The exploration term gets **larger** when a node has been visited rarely
(small $N(v)$), which pushes MCTS to occasionally explore less-visited
branches rather than always following the greedy path.

---

## Part 2 — Our Implementation (2–3 min)

### Overview: 7 Files

```
exercise_3/
│
├── world_model.py   ← The two LLM calls RAP needs (predict state + score action)
├── mcts_node.py     ← One node in the search tree, with its expansion logic
├── mcts.py          ← The MCTS algorithm (select → expand → backpropagate)
├── search.py        ← Ties everything together for one Blocksworld problem
│
├── utils.py         ← Data loading: PDDL parser + deterministic plan simulator
├── baseline.py      ← Direct LLM planner (no search) for comparison
└── run.py           ← Full experiment: loads data, runs both methods, prints results
```

### What We Reused from the Original Repository

The original paper's codebase is in `RAP/`. We reused these parts unchanged:

| File | What it provides | Used in |
|------|-----------------|---------|
| `RAP/data/blocksworld/my_mcts_prompts_update.json` | All few-shot prompts for LLM calls | `world_model.py`, `baseline.py` |
| `RAP/rap/utils/blocksworld.py` → `generate_all_actions(state)` | Pure-Python function that lists every valid action from a text state | `mcts_node.py` |
| `RAP/rap/utils/blocksworld.py` → `apply_change(change, state)` | Pure-Python function that applies a `[CHANGE]` description to a text state | `mcts_node.py` |
| `RAP/data/blocksworld/step_2/4/6.json` | Dataset: list of (PDDL path, ground-truth plan, difficulty) | `run.py` |
| `RAP/gpt-plan-benchmark/.../instance-N.pddl` | The actual Blocksworld problem instances | `utils.py` |
| `RAP/data/blocksworld/bw_config.yaml` | Mapping from PDDL object letters to colour names | `utils.py` |

### What We Built / Changed

#### `world_model.py` — The Two LLM Calls

This file wraps AgentScope's `OpenAIChatModel` to provide the two capabilities
RAP needs. Every method is `async` because AgentScope 1.0.20's model calls are
asynchronous.

**Call 1 — `predict_next_state(last_state, action) → change_text`**

Given a state description like:
```
"I have that, the red block is clear, the hand is empty,
 the blue block is on top of the red block, the red block is on the table."
```
and an action like `"Pick up the red block"`, the LLM is asked to predict
what physically changes. There are four few-shot prompt templates in the JSON
file, one per action type (pickup/unstack/putdown/stack). The LLM outputs a
`[CHANGE]` line like:
```
[CHANGE] the red block is no longer clear and the red block is no longer on the
table and the red block is now in the hand and the hand is no longer empty and
the hand is now holding the red block
```
This change text is then fed to `apply_change()` (from the original RAP code)
which applies it to the previous state string to produce the new state string.

**Call 2 — `score_action(state0, state1, goal) → float`**

In the original paper, the `r0` score for each candidate action was computed
using **token log-likelihoods** from LLaMA. Concretely, for every valid action
`a` at a given node, the original code constructed a prompt:

```
<few-shot baseline prompt with worked-plan examples>
<action history so far>
<candidate next action a>
```

and called `get_ll(baseline_prompt, completion)`, which returns
$\log P(\text{action } a \mid \text{context})$ — the probability that LLaMA
would naturally generate that exact action text given the context. The scores
for all candidate actions at a node were then **softmax-normalised** so they
formed a probability distribution summing to 1.

The intuition is: *"given a few-shot prompt full of good plans, which action
does the model most want to write next?"* — an implicit quality signal that
requires no explicit question. Crucially, it requires **direct access to token
logits**, which is only available when running LLaMA locally (not via any API).

We do not have logit access. Instead, we use the `confidence` few-shot prompt,
which asks the LLM an explicit Yes/No question: *"Is this state closer to the
goal than before?"* and map **Yes → 0.9**, **No → 0.1**.

| | Original (LLaMA) | Ours |
|---|---|---|
| **Signal** | Implicit: $\log P(\text{action} \mid \text{good-plan context})$ | Explicit: "Is this progress?" |
| **Output** | Soft distribution over all actions (softmax) | Binary: 0.9 or 0.1 |
| **Requires** | Direct access to token logits | Any chat API |

This is coarser — we lose the relative ranking between actions at the same
node — but it achieves the same purpose: steering MCTS toward actions the
model believes lead to the goal.

---

#### `mcts_node.py` — One Node in the Search Tree

Each `BlocksworldMCTSNode` represents one point in the plan:

```
[GOAL] My goal is to have that the blue block is on top of the red block.
[STATE 0] I have that, the red block is clear, the hand is empty, ...
[ACTION 1] Pick up the red block.
[STATE 1] I have that, the hand is holding the red block, ...   ← added by LLM
[ACTION 2] Stack the red block on top of the orange block.
           ↑ not yet expanded — the [STATE 2] line will be added by the LLM
             when this node is expanded during MCTS
```

The node stores:
- `prompt` — the full text history up to this point (grows as the plan deepens)
- `depth` — how many actions have been taken (root = 0)
- `_r0` — the action score assigned by the parent when this node was created
- `_r1` — the state score computed when *this* node is expanded
- `parent` — reference to the parent node

**Expansion logic (`_get_children`):**

When MCTS visits a node for the first time, it:
1. Calls `_calculate_reward()`:
   - Asks the LLM to predict the new state after the last action (→ `change_text`)
   - Calls `apply_change(change_text, prev_state)` to produce the new state text
   - Appends `[STATE depth] new_state` to the node's prompt
   - Counts how many goal conditions are met → sets `_r1`
2. Calls `generate_all_actions(current_state)`:
   - This is pure Python — no LLM. It reads the state text and lists every
     valid action (e.g. you can only pick up a block if your hand is empty and
     the block is on the table and clear)
3. For each valid action, calls `score_action()` (LLM) → gets `r0`
4. Creates one child node per action with the corresponding `r0`

**Terminal conditions:**
- `r1 > 50` → all goal conditions satisfied → **success**
- `depth >= max_depth` → plan too long → stop
- `reward < -1` → LLM predicted an impossible state → discard branch

---

#### `mcts.py` — The MCTS Algorithm

The `MCTS` class keeps three running statistics for every node it has visited:

| Variable | Meaning |
|----------|---------|
| `Q[node]` | Cumulative reward accumulated through this node across all rollouts |
| `N[node]` | How many times this node has been visited |
| `M[node]` | Best single reward seen through this node (used in UCT) |

**One rollout** (`rollout(root)`) does two phases:

1. **`_select_prior(root)`** — walks the tree from root to a leaf:
   - At each node, expands it (if not already done) → LLM calls happen here
   - Picks the child with the highest UCT score
   - Repeats until reaching a terminal node or dead end
   - Returns the full path (list of nodes) traversed

2. **`_back_propagate(path)`** — walks the path in reverse:
   - Accumulates reward going upward: `cumulative_reward = reward[node] + discount * cumulative_reward`
   - Updates `Q[node] += cumulative_reward`, `N[node] += 1`, `M[node] = max(M[node], cumulative_reward)`
   - With `aggr_reward='mean'` (used for Blocksworld), divides by path length so deeper nodes are not penalised

After all rollouts, **`best_terminal(root)`** recursively walks the tree and
returns the terminal node (depth ≥ max_depth or r1 > 50) with the highest
mean path reward.

---

#### `search.py` — End-to-End Search for One Problem

`blocksworld_rap_search(initial_state, goal, prompts, world_model, rollouts=10, max_depth=4)`

1. Builds the **root node** prompt:
   ```
   [GOAL] My goal is to have that the blue block is on top of the red block.
   [STATE 0] I have that, the red block is clear, ...
   ```
2. Creates a `BlocksworldMCTSNode` (root) and an `MCTS` instance
3. Runs `rollouts` iterations — after each one, logs the best plan found so far
4. Returns `(best_node, best_reward, trajectories)`

The helper `extract_plan(prompt)` reads all `[ACTION N] ...` lines from a
node's prompt to get the action sequence.

---

#### `utils.py` — Data Loading and Evaluation

**PDDL Parser (`parse_pddl`)**

Each problem is stored as a `.pddl` file like:
```pddl
(:objects a b c d )
(:init (handempty) (ontable a) (on b a) (on c b) (ontable d) (clear c) (clear d))
(:goal (and (on b a) (on d c)))
```

The letters are **not** colour names — they map via `bw_config.yaml`:
`a→red, b→blue, c→orange, d→yellow`. Our regex parser reads the file,
substitutes the colours, and converts PDDL predicates to natural language:

| PDDL predicate | Natural language |
|----------------|-----------------|
| `(ontable a)` | `"the red block is on the table"` |
| `(on b a)` | `"the blue block is on top of the red block"` |
| `(clear c)` | `"the orange block is clear"` |
| `(handempty)` | `"the hand is empty"` |

The original code used the `tarski` formal methods library for this. We
replaced it with a simple regex parser — no extra dependency.

**Deterministic Plan Simulator (`execute_plan`, `apply_action`)**

To evaluate a plan we need to know if it actually works. We cannot ask the
LLM — the LLM might hallucinate a valid-sounding but incorrect plan. Instead,
`utils.py` implements the **real Blocksworld rules** in pure Python:

```python
BlocksState:
    on_table: set   # which blocks are on the table
    on: dict        # above → below (block A is on top of block B)
    clear: set      # blocks with nothing on top
    holding: str    # which block the hand is holding (None if empty)
```

`apply_action(state, action)` applies one action using the four rules:
- **Pick up X** — only valid if hand is empty, X is on the table, X is clear
- **Unstack X from Y** — only valid if hand is empty, X is on Y, X is clear
- **Put down X** — only valid if hand is holding X
- **Stack X on Y** — only valid if hand is holding X, Y is clear

If the preconditions fail, it returns `None` (illegal action). This lets us
detect plans that are syntactically correct but physically impossible.

---

#### `baseline.py` — Direct LLM Planning (No Search)

The baseline uses the `baseline_action` few-shot prompt from the JSON file.
This prompt contains four complete worked examples of Blocksworld problems and
their solutions. The LLM is shown the problem and asked to output the plan
directly — no search, no rollouts, one single LLM call.

This is essentially how GPT-3/4 is typically used for planning tasks. It is
the comparison point that shows whether the extra complexity of RAP-MCTS
actually improves things.

---

#### `run.py` — The Experiment

The run script orchestrates everything:

1. Loads the API key and base URL from `.env`
2. Loads all prompts from `my_mcts_prompts_update.json`
3. Loads the dataset (`step_2/4/6.json`), parses each PDDL file
4. For each problem, runs RAP and/or baseline, then evaluates both
5. Prints a per-instance table and overall accuracy numbers

```bash
# Run from exercise_3/ with rap_venv active
python -m exercise_3.run --steps 2 --n 3
```

---

### Dependency Choices and Workarounds

| Original | Our replacement | Why |
|----------|----------------|-----|
| `tarski` PDDL library | Regex parser in `utils.py` | Heavy dependency, only needed for parsing |
| VAL plan validator (external binary) | Python state machine in `utils.py` | No binary installation needed |
| LLaMA log-likelihoods for r0 | Yes/No LLM prompt → 0.9/0.1 | API models don't expose log-probabilities |
| Synchronous code | Fully `async/await` | AgentScope 1.0.20 model calls are async |
| `torch.distributed.barrier()` in error paths | `RAP/torch.py` no-op stub | We don't need PyTorch; stub avoids huge install |

---

## Part 3 — Evaluation (2–3 min)

### What "Correct" Means

A plan is considered **successful** if and only if **both** of the following
hold when we simulate it using our deterministic state machine:

1. **Every action is legally applicable** — no precondition is violated at any
   step (e.g. you cannot pick up a block if your hand is already holding one)
2. **The final state satisfies all goal conditions** — every `(on X Y)` in the
   PDDL goal appears in the final state

This is checked by `evaluate_plan(initial_state, goal, actions)` in `utils.py`.
It returns a dict with `valid`, `goal_reached`, `steps`, and `final_state`.

### The Dataset

The dataset is split by **ground-truth plan length**:

| Split | Instances | GT plan length | Difficulty |
|-------|-----------|----------------|------------|
| step_2 | 30 | 2 actions | Easy |
| step_4 | 57 | 4 actions | Medium |
| step_6 | 114 | 6 actions | Hard |

Each instance is a `.pddl` file with 4 blocks (red, blue, orange, yellow).
The entry in `step_N.json` looks like:
```json
["gpt-plan-benchmark/.../instance-5.pddl", "(pick-up yellow)\n(stack yellow orange)\n", 2]
```
— the PDDL file path, the ground-truth plan in PDDL format, and the step count.

### Two Methods Compared

| Method | How it works | LLM calls per problem |
|--------|-------------|----------------------|
| **Baseline** | One-shot: show the problem + 4 examples → ask for a plan | 1 |
| **RAP-MCTS** | 10 rollouts × (state prediction + action scoring per valid action) | ~30–60 |

The baseline is the "naive" approach — what you'd get by just prompting the LLM.
RAP-MCTS is the paper's contribution — structured search with a world model.

### What Results to Expect

The paper (with LLaMA-33B) reports approximately:

| Split | Baseline (GPT-4 in paper) | RAP-MCTS |
|-------|--------------------------|----------|
| step_2 | ~60% | ~80% |
| step_4 | ~20% | ~60% |
| step_6 | ~5% | ~35% |

The key takeaway: **the gap between baseline and RAP widens as plan length
increases**. RAP's structured search becomes more valuable precisely when
greedy one-shot generation is most likely to fail.

With our smaller model (Qwen3-VL-4B vs LLaMA-33B) and coarser r0 scoring
(Yes/No vs log-probability), we expect lower absolute numbers, but the
same qualitative pattern should hold.

### Running the Experiments

```bash
# Activate the environment (from the project root)
source exercise_3/rap_venv/bin/activate
cd exercise_3

# Quick sanity check: 3 problems, baseline only (3 LLM calls)
python -m exercise_3.run --steps 2 --n 3 --no-rap

# Step-2 full run: 30 problems, both methods (~20-40 min)
python -m exercise_3.run --steps 2

# All three difficulty levels (run sequentially)
for s in 2 4 6; do
  python -m exercise_3.run --steps $s
done
```

**All CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--steps 2\|4\|6` | `2` | Which dataset split to use |
| `--n N` | all | Limit to the first N instances |
| `--rollouts R` | `10` | MCTS rollouts per problem (fewer = faster, less accurate) |
| `--depth D` | `4` | Maximum plan depth before the search stops |
| `--no-rap` | off | Skip RAP, run baseline only |
| `--no-baseline` | off | Skip baseline, run RAP only |

### Sample Output

```
============================================================
  Problem 1/3:  instance-5.pddl
  Init : the hand is empty, the red block is on the table, ...
  Goal : the blue block is on top of the red block and the yellow block is on top of the orange block
  GT   : ['Pick up the yellow block', 'Stack the yellow block on top of the orange block']

  [RAP] instance-5.pddl
  Rollout  1/10  reward=0.612  plan: Pick up the yellow block → Stack the yellow block on top of the orange block
  Rollout  2/10  reward=0.734  plan: Pick up the yellow block → Stack the yellow block on top of the orange block
  ...
  [RAP] ✓ actions=['Pick up the yellow block', 'Stack the yellow block on top of the orange block']

  [Baseline] instance-5.pddl
  [Baseline] ✓ actions=['Pick up the yellow block', 'Stack the yellow block on top of the orange block']

────────────────────────────────────────────────────────────────────────
  Blocksworld Experiment  |  steps=2  |  n=3 problems
────────────────────────────────────────────────────────────────────────
  RAP-MCTS accuracy :  2/3  (66.7%)
  Baseline accuracy :  1/3  (33.3%)
  RAP wins over baseline: 1 problem(s)
────────────────────────────────────────────────────────────────────────
  Instance                         GT     RAP   Base
  instance-5.pddl                   2      ✓      ✓
  instance-21.pddl                  2      ✓      ✗
  instance-31.pddl                  2      ✗      ✗
────────────────────────────────────────────────────────────────────────
```
