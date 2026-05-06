# `experiments/testing/` — IB vs baselines on a heavy-tailed bandit

Headline empirical experiment for the `testing` branch. Designed to satisfy
three NeurIPS-bar criteria simultaneously, addressing the structural gaps in
the existing `main`-branch experiments:

1. **Multi-episode learning.** The agent's belief persists across episodes
   (50 episodes × 200 steps by default). `simulate_multi_episode` resets only
   the per-episode RNG between episodes — bandit arm parameters and the
   agent's belief carry forward.
2. **Genuine non-realizability.** The environment is a Cauchy-contaminated
   Gaussian bandit (`HeavyTailedBanditEnvironment`). The Normal-Normal
   likelihood family that `BayesianAgent` and `ThompsonSamplingAgent` assume
   cannot represent Cauchy contamination — no choice of prior fixes the
   misspecification, because the misspecification is in the likelihood family
   itself.
3. **Non-trivial baselines.** Both Thompson sampling and UCB1 rely on
   sub-Gaussian tail concentration in their regret guarantees. Cauchy
   contamination violates that assumption, so their predicted underperformance
   is theoretically motivated, not circumstantial.

## Reproduce

```bash
uv sync --extra dev
uv run python experiments/testing/main.py     # ~minutes for default config
uv run python experiments/testing/plot_results.py
```

Outputs land in `experiments/testing/outputs/`:

- `results.npz` — per-agent reward and optimal arrays of shape
  `(num_seeds, num_episodes, steps_per_episode)`.
- `regret_vs_total_steps.png` — cumulative regret across the flattened
  multi-episode trajectory; mean + 95% bootstrap CI band per agent.
- `per_episode_regret.png` — mean within-episode regret per episode;
  surfaces how each agent's per-episode performance evolves as belief
  accumulates.

The `outputs/` directory is gitignored (`experiments/**/outputs/` in
`.gitignore`); only `.gitkeep` is tracked.

## Configuration

`main.py` accepts CLI flags with the defaults below:

| flag                   | default | meaning                                         |
| ---------------------- | ------- | ----------------------------------------------- |
| `--num-seeds`          | 30      | independent seeds per agent                     |
| `--num-episodes`       | 50      | episodes per seed                               |
| `--steps-per-episode`  | 200     | steps within an episode                         |
| `--num-actions`        | 10      | bandit arms                                     |
| `--output`             | `outputs/results.npz` | where to save raw results       |

## Agents compared

| name              | class                           | role in the experiment              |
| ----------------- | ------------------------------- | ----------------------------------- |
| `bayesian-greedy` | `BayesianAgent` (epsilon=0.0)   | misspecified-likelihood baseline    |
| `q-learning`      | `QLearningAgent` (epsilon=0.1)  | classical RL reference              |
| `thompson`        | `ThompsonSamplingAgent`         | posterior-sampling baseline         |
| `ucb1`            | `UCB1Agent`                     | concentration-bound baseline        |
| `infrabayesian`   | `InfraBayesianAgent` (1 belief) | non-KU IB; matches Bayesian when realizable |

The single-belief `infrabayesian` configuration short-circuits to a plain
Bayesian update on its `GaussianBelief` (see comment in `infrabayesian.py`).
A full KU configuration with multiple beliefs spanning Normal and
heavy-tailed assumptions is a natural follow-up.

## Caveat

Cauchy-tailed reward distributions make the empirical mean unstable — that's
the point of the experiment. As a consequence individual seeds can swing
widely. The bootstrap-CI plot is the right summary; raw means are not.

## What lives elsewhere

- Multi-episode simulator: `ibrl/simulators/simulator.py`
  (`simulate_multi_episode`)
- Reset-semantics split: `ibrl/agents/base.py`, `ibrl/environments/base.py`
  (`reset_belief`, `reset_episode`)
- Heavy-tailed env: `ibrl/environments/heavy_tailed_bandit.py`
- Baselines: `ibrl/agents/{thompson,ucb}.py`
- Metrics + plotting: `ibrl/analysis/`
- IB agent + beliefs (vendored from `alaro/coin-learning-clean`):
  `ibrl/agents/infrabayesian.py`, `ibrl/infrabayesian/`
