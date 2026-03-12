# Project-Newcomb

Proof-of-concept infrabayesian reinforcement learning (IBRL) agent that converges to optimal policies on Newcomb-like problems and other decision-theoretically complex environments. SPAR Spring 2026 project.

Full design spec: [Architecture Planning Document](https://docs.google.com/document/d/1WkUK5Mc_OsfeaE6wUbuEhdzCXzDbq6pAapHp1LOLCMk)

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

    git clone https://github.com/Lorxus/Project-Newcomb.git
    cd Project-Newcomb
    uv sync

To add a dependency: `uv add <package-name>`. This updates `pyproject.toml` and `uv.lock`. Commit both.

## Structure

    ibrl/
        agents/          — agent implementations (subclass base)
        environments/    — environment implementations (subclass base)
        simulators/      — simulate(agent, env, **kwargs) -> results
        analysis/        — metrics, plotting, comparison tools

    experiments/
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

Some example work flows you can copy for your experiments:
- Using a main.py file in experiments/fllor/main.py
- Using a jupyter notebook in experiments/alaro/example.ipynb


## Running scripts

Use `uv run` to execute scripts without manually activating the virtual environment:

    uv run python experiments/yourname/script.py

To launch a Jupyter notebook:

    uv run jupyter lab

`uv run` ensures the script uses the project's environment and has access to `ibrl/` and all dependencies.

## Imports

`uv sync` installs `ibrl/` as a local package. From any script or notebook in `experiments/`, import normally:

```python
# sample — these modules don't exist yet
from ibrl.agents.base import BaseAgent
from ibrl.environments.newcomb import NewcombEnv
from ibrl.simulators.basic import simulate
```

If something useful lives in someone else's experiment, promote it to `ibrl/` through a reviewed PR. Experiments should never import from other experiments.

## Workflow

    git checkout -b yourname/short-description
    # make changes
    git push origin yourname/short-description

Then go to the repo on GitHub. You'll see a banner to create a pull request from your branch. Click it, set the base branch to `main`, fill out the PR template (it loads automatically) and request a review. Once approved, squash merge.

All PRs squash merge into main. Squash merging collapses all commits from a branch into a single commit on main, so the history stays clean regardless of how messy your branch was while iterating. Keep branches short-lived.

## Commit messages

Start with the area:

    environments: add parfit hitchhiker
    agents: implement thompson sampling
    experiments: newcomb sweep with varying predictor accuracy
    infra: add smoke test CI

Use infra: for non-code changes like README, CI, pyproject.toml and repo config.

## Rules

- Every new agent subclasses `ibrl.agents.base`
- Every new environment subclasses `ibrl.environments.base`
- Pin a seed for every experiment
- Save your config alongside your results
- If you vibe code, review the output for correctness and adherence to spec
- Don't commit large data files (results/ is gitignored)