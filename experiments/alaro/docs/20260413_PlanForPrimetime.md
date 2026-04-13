# Making `ibrl` a Shareable Library

## Overview

The repo is in a good structural state — clear abstractions, tests, CI, and a working agent. The main gap is that everything is currently oriented toward *contributors* (people building inside the repo) rather than *users* (people who want to run the IB agent in their own experiments). The items below are split by phase: what to do **now**, before the new IB architecture lands, and what to do **after** it stabilizes.

---

## Phase 1: Do Now (architecture-independent)

### 1. Audit and loudly flag all incomplete code

Every place where something is unimplemented, stubbed, or numerically unstable should raise `NotImplementedError` or `warnings.warn` with a clear message — not just a `# TODO` comment that an external user will miss.

Specific locations:

| Location | Issue | Action |
|---|---|---|
| `ibrl/infrabayesian/beliefs.py` — `GaussianBelief.compute_outcome_probability` | Not implemented | Raise `NotImplementedError("GaussianBelief.compute_outcome_probability is not yet implemented")` |
| `ibrl/agents/infrabayesian.py` — `_solve_game` | Heuristic (diagonal only); proper π^T V π optimization is missing | Add docstring warning; raise `NotImplementedError` if `V` is not symmetric diagonal-dominated, OR document clearly that this is a diagonal approximation |
| `ibrl/infrabayesian/infradistribution.py` — `g ∈ (0,1)` | Causes numerical blowup over many steps | Add a `warnings.warn` at construction time if `0 < g < 1`, pointing to the known issue |
| `ExperimentalAgent1`, `ExperimentalAgent2`, `ExperimentalAgent3` | Exploratory, not production-ready | Add docstring: `"Experimental — API and behavior may change without notice"` |

### 2. Separate the public API from internals

Right now everything is importable. Define what's public:

- **Public**: `InfraBayesianAgent`, `BayesianAgent`, `BernoulliBayesianAgent`, `EXP3Agent`, `QLearningAgent`, all environments, `simulate()`, and the belief/infradistribution layer (`BaseBelief`, `AMeasure`, `Infradistribution`) since researchers will want to write custom beliefs
- **Internal/experimental**: `ExperimentalAgent1-3`, `construct_agent()`/`construct_environment()` string-based factories (too brittle for external use)

Move experimental agents out of `__all__` in `ibrl/agents/__init__.py` and note them as internal.

### 3. Rewrite the README for external readers

The current README is excellent for contributors but assumes you're already on the team. Add a new top section for external users before the contributor workflow. It should cover:

- **What is infrabayesian RL?** — one paragraph, link to Kosoy's papers
- **What does this library do?** — implements an IB agent, lets you run it against standard bandits and Newcomb-like environments, compare against classical agents
- **Why is the environment interface different from gymnasium?** — brief explanation that Newcomb-like problems require the agent's *policy* (not just its *action*) to be observable by the environment, so `step(probabilities, action)` is intentional
- **Quickstart** — 10-line example: create an agent, create an environment, run a simulation, plot cumulative reward
- Keep all the contributor content below, clearly separated

### 4. Add a `CITATION.cff` and theory references

Add a `CITATION.cff` file so people citing the library have a standard format. More importantly, add a `THEORY.md` (or a section in the README) with:

- Vanessa Kosoy's infrabayesian papers (with links/DOIs)
- What `g`, `AMeasure`, `Infradistribution` correspond to in the theory
- A note on how the Newcomb environments map to the decision-theoretic problems in the literature

This is the thing that most distinguishes this library — without the theory pointers, external users won't know what they're looking at.

### 5. Add a `LICENSE` file

The repo has no license. Without one, the legal default is "all rights reserved," meaning external users technically cannot use or redistribute it. Add `MIT` or `Apache-2.0` — MIT is the norm for academic research libraries.

### 6. Clean up `pyproject.toml` packaging metadata

Add the fields that matter for external distribution (even before PyPI):

```toml
[project]
description = "Infrabayesian reinforcement learning agent for Newcomb-like problems"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "SPAR Spring 2026 IBRL Team" }]

[project.urls]
"Source" = "https://github.com/SPAR-S26-IBRL-Stream/Project-Newcomb"
"Theory" = "<link to Kosoy papers>"
```

Also: `jupyter` and `matplotlib` should not be core dependencies for someone who just wants to import the agent. Move them to an optional `[project.optional-dependencies]` group called `experiments` or `viz`.

### 7. A minimal working example (notebook or script)

Create `examples/quickstart.py` or a notebook at the repo root showing the simplest possible end-to-end usage:

```python
from ibrl.agents.infrabayesian import InfraBayesianAgent
from ibrl.infrabayesian.beliefs import BernoulliBelief
from ibrl.environments.bernoulli_bandit import BernoulliBanditEnvironment
from ibrl.simulators.simulator import simulate

agent = InfraBayesianAgent(num_actions=2, beliefs=[BernoulliBelief(num_actions=2)])
env = BernoulliBanditEnvironment(num_actions=2)
results = simulate(env, agent, {"num_steps": 500, "num_runs": 10})
```

This is the thing someone reads first when evaluating whether to use a library.

---

## Phase 2: After the New Architecture Stabilizes

The `fllor/ib-agent-architecture` branch introduces a significantly cleaner design: it eliminates `BaseBelief`, removes the `g` parameter problem, and handles mixed infradistributions properly with history-based probability computation. The items below should wait until that lands.

### 8. Document the new architecture's design decisions

Once the new architecture merges, write short design documentation (inline docstrings or a `DESIGN.md`) explaining:

- Why the mixture-over-hypotheses approach is used
- What `scale`, `offset`, `coefficients`, and `history` represent in terms of the theory
- Why `compute_probabilities` uses `log_probabilities` and why `1e-300` is the floor

### 9. Regret bound experiments

This is the main scientific contribution. Create a structured experiment (in `experiments/` or promoted to `ibrl/analysis/`) that shows:

- IB agent vs `BernoulliBayesianAgent` vs `EXP3` on standard stochastic MAB — cumulative regret plot
- IB agent vs other agents on Newcomb, Damascus, Coordination — show IB converges to optimal, others don't
- Vary `num_hypotheses` to show robustness to that parameter

These become the plots in the README and the thing people share when pointing others to the library.

### 10. Proper API documentation

Use `mkdocs` with `mkdocstrings` (lighter-weight than Sphinx, renders on GitHub Pages automatically). The docstrings already exist in many places — this mostly means rendering them. Priority order:

- `BaseAgent` and `BaseEnvironment` — the contract for implementing new agents/envs
- `InfraBayesianAgent` — the main contribution
- `simulate()` — the main entry point
- The infradistribution layer — for researchers extending the theory

### 11. PyPI publishing

Once the architecture is stable and the public API is clean, publish to PyPI so people can `pip install ibrl`. This requires:

- Bumping the version to `0.1.0` properly (currently set but never published)
- Setting up a GitHub Actions workflow for publishing on tagged releases
- Checking that the package name `ibrl` is available on PyPI

### 12. Address `num_actions=2` restriction

The Newcomb-like environments are currently restricted to 2 actions. If the goal is demonstrating general IB regret bounds, this needs to be generalized or clearly marked as a known scope limitation with a GitHub issue tracking it.

---

## What Does Not Need Changing

- The `step(probabilities, action)` interface — it's correct and intentional; just needs documentation
- The `simulate()` return format — it's clean and comprehensive
- The overall `ibrl/agents/` + `ibrl/environments/` + `ibrl/simulators/` structure
- The test suite structure — it's well-organized and thorough
- The `experiments/<name>/` contributor workflow

---

## Priority Order

1. LICENSE (legal blocker for external use)
2. `NotImplementedError`/warnings on all incomplete code
3. README external-facing intro + quickstart example
4. Theory references / `CITATION.cff`
5. `pyproject.toml` cleanup
6. *(wait for new IB architecture to land)*
7. Regret bound experiments
8. API docs
9. PyPI
