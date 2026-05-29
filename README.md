# Project-Newcomb

Proof-of-concept infrabayesian reinforcement learning (IBRL) agent that
converges to optimal policies on Newcomb-like problems and other
decision-theoretically complex environments.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

    git clone <repo-url>
    cd Project-Newcomb
    uv sync

## Structure

    ibrl/
        agents/          — agent implementations (subclass base)
        environments/    — environment implementations (subclass base)
        simulators/      — simulate(agent, env, **kwargs) -> results
        analysis/        — metrics, plotting, comparison tools

    experiments/
        agent_comparison_sweep/    — make-driven sweep of agents x environments
        coin_toss_toy/             — coin-tossing infra-Bayes toy
        decision_theory_exercises/ — Newcomb and coin-toss IB exercises
        ib_architecture/           — infra-Bayesian agent architecture proposal
        ib_exploration/            — KU comparison and IB planning notes
        nim_rl/                    — Q-learning on Nim
        example/         — working intro simulation
        <your-name>/     — your personal experiment space


### Source code

**ibrl/** is the shared library. Every change requires a reviewed PR with at least one approval (approve or request changes, not just comments).

Anyone can contribute new agents and environments! To do so, please:
1. Create a new file under either agents or environments (depending on which you are creating)
2. Look at existing classes and follow the naming patterns. EG, File name should be the class name. Name should be descriptive.
    - For example, if you are building an infrabayesian agent, you might call your new class InfraBayesianAgent(BaseAgent) (inherits from BaseAgent) and would name the file infrabayesian.py
3. Be sure to inherit from a prior instance, eg BaseAgent for agents or BaseEnvironment for environments. This will ensure you follow the proper protocol in your class definition.
4. Look at the base classes to determine which functions you need to define and their required function signatures. If you are confused, it helps to look at other inheriting instances and see how they do it. For example, InfraBayesianAgent required only a get_probabilities method be defined, though you may want to overwrite other methods (like update) or add additional helper methods in your class.
5. When you are confident your contribution works (a great way to do this is by also contributing unit tests for your new code) and you want to share your code with others, you can open a PR

### Experiments

**experiments/** is the exploration zone. Create a folder with your git handle. Each experiment gets its own subfolder with a README covering: what and why, design decisions, chat logs if vibe coded, results and interpretation, ideas for shared architecture evolution. PRs here get lighter review.

For reproduction instructions for **Infra-Bayesian Reinforcement Learning Agents
Outperform Classical RL For Worst-Case Robustness**, start with
[experiments/README.md](experiments/README.md).

Some example work flows you can copy for your experiments:
- Using a main.py file in experiments/fllor/main.py
- Using a jupyter notebook in experiments/alaro/example.ipynb

`ibrl/` is the shared library. `experiments/` contains standalone
explorations, each with its own README describing what was tried,
results, and interpretation.

## Running scripts

Use `uv run` to execute scripts without manually activating the virtual
environment:

    uv run python experiments/<experiment-name>/script.py

To launch a Jupyter notebook:

    uv run jupyter lab

## Imports

`uv sync` installs `ibrl/` as a local package. From any script or notebook
in `experiments/`, import normally:

```python
from ibrl.agents.base import BaseAgent
from ibrl.environments.newcomb import NewcombEnv
from ibrl.simulators.basic import simulate
```

## Tests

```bash
uv sync --extra test
uv run pytest                              # all tests
uv run pytest tests/test_smoke.py -v       # smoke tests only
uv run pytest --cov=ibrl --cov-report=term-missing
```

Or push a commit and check GitHub Actions:
https://github.com/SPAR-S26-IBRL-Stream/Project-Newcomb/actions
