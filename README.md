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
