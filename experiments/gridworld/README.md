# `experiments/gridworld/` — robust gridworld with interval-uncertain transitions

Headline experiment for the dynamic-consistency thesis: classical robust DP
plans against the worst-case kernel once and never updates. IB plans against
the same worst-case kernel but updates the polytope from observations and
re-plans every episode while preserving the lower-expectation guarantees that
were valid ex-ante. The hypothesis is that IB beats one-shot Robust DP on
multi-episode regret, beats Bayesian-greedy under non-realizability, and
matches or beats Thompson sampling.

## Result: NULL

Across 30 seeds × 20 episodes × 100 steps, on a 5×5 gridworld with
interval-uncertain transitions and an adversary that picks the worst-case
intended-direction probability from `[p_nominal − ε, p_nominal + ε]`, IB DP
**ties Robust DP within 95% bootstrap CIs** and **ties Bayesian DP within CIs**
at all three ε ∈ {0.0, 0.05, 0.10}, under both adversary modes (`static`,
`per_episode_visit`). Final cumulative regret across 30 seeds:

| ε | adversary | robust-dp | bayesian-dp | thompson-dp | ib-dp |
|---|---|---:|---:|---:|---:|
| 0.05 | static | −0.41 ± 0.39 | +0.15 ± 0.75 | +7.33 ± 1.92 | −0.41 ± 0.42 |
| 0.10 | static | −0.62 ± 0.46 | −0.08 ± 0.85 | +7.74 ± 1.51 | −0.53 ± 0.58 |
| 0.05 | per_episode_visit | −0.27 ± 0.42 | +0.22 ± 0.75 | +6.89 ± 1.92 | −0.35 ± 0.43 |
| 0.10 | per_episode_visit | −0.62 ± 0.46 | +0.13 ± 0.92 | +7.97 ± 1.49 | −0.37 ± 0.60 |

(Negative cum-regret is normal: the agent's discounted return frequently
exceeds the env's worst-case-kernel V*, since individual episodes don't
always realize the adversary's static commitment.)

The Thompson DP baseline is decisively worse (PSRL on a sparse Dirichlet
posterior is high-variance), but Thompson is not the right comparison
target — it doesn't represent "what a competent classical baseline does."
The right comparison is to the structurally-similar Robust DP and to the
posterior-mean Bayesian DP, and IB ties both.

## Mechanism: why the dynamic-consistency story doesn't materialize here

The IB agent and the Robust DP agent both plan with `lower_expectation_value_iteration`
on the same defensible initial polytope `[p_nominal − 2ε, p_nominal + 2ε]`. IB
shrinks that polytope across episodes (verified via test_mdp_agents.py — width
is monotone non-increasing), but the *policy* selected against the shrunken
polytope is the same as the Robust DP policy on the wider polytope at every
state of the 5×5 gridworld. The path-to-reward and trap-avoidance choices are
robust to interval kernel uncertainty at this scale, so the worst-case argmax
is constant under polytope shrinkage. Without an environment in which the
optimal policy is *sensitive* to the polytope width (i.e. where shrinking
the worst-case set genuinely opens up better actions), there is no online
benefit to the IB update rule over a single up-front robust plan.

This is the same family of issue we saw with the bandit experiments: the
worst-case rule needs a setting where the resolved-likelihood pessimism
collapses meaningfully into a different policy. Gridworlds with stationary
kernels and continuous transitions don't provide it.

## Reproduce

```bash
uv sync --extra dev

# Day-3 run (static adversary, the headline pilot per the plan):
uv run python experiments/gridworld/main.py
uv run python experiments/gridworld/plot_results.py

# Day-4 fallback (per-episode-visit adversary; results were also null):
uv run python experiments/gridworld/main.py \
    --adversary-mode per_episode_visit \
    --output experiments/gridworld/outputs/results_dynamic.npz
uv run python experiments/gridworld/plot_results.py \
    --input experiments/gridworld/outputs/results_dynamic.npz
```

Each run is ~30s on Apple Silicon. CLI flags: `--num-seeds`, `--num-episodes`,
`--max-steps`, `--epsilons`, `--adversary-mode`.

## Outputs

`outputs/` contains the raw `.npz` results and rendered PNGs. The directory
is gitignored except `.gitkeep`; only the figures are committed for paper
inclusion.

| file | content |
|---|---|
| `results.npz` | static-adversary headline run, full per-step rewards |
| `results_dynamic.npz` | per-episode-visit adversary run |
| `regret_per_episode__eps{0,0.05,0.10}__{static,per_episode_visit}.png` | per-episode regret with 95% bootstrap CIs |
| `cum_regret__eps{0,0.05,0.10}__{static,per_episode_visit}.png` | cumulative regret across episodes |
| `final_regret_by_epsilon__{static,per_episode_visit}.png` | final cum regret as a function of ε |

## Configuration

| flag                  | default              | meaning |
|-----------------------|----------------------|---------|
| `--num-seeds`         | 30                   | independent seeds per (agent × ε) |
| `--num-episodes`      | 20                   | episodes per seed |
| `--max-steps`         | 100                  | step budget per episode |
| `--epsilons`          | `0.0 0.05 0.10`      | polytope half-widths |
| `--adversary-mode`    | `static`             | `static` or `per_episode_visit` |
| `--output`            | `outputs/results.npz`| where to save |

Gridworld defaults (in `ibrl/mdp/gridworld.py`): 5×5 grid, start (0,0),
reward (4,4) +1, trap (2,2) −1, step cost −0.01, p_nominal = 0.8, gamma = 0.95.
Both `RobustDPAgent` and `IBDPAgent` are constructed with the same
`initial_polytope = widen(env.kernel_polytope(), factor=2.0)` so that the only
difference between them is online polytope updating.

## Where the code lives

- MDP layer (new): `ibrl/mdp/{base,simulator,gridworld,interval_belief,value_iteration,agents}.py`
- Reused without modification: `ibrl/analysis/{metrics,plotting}.py` (bootstrap CI,
  regret plotting), `tests/conftest.py`, `.github/workflows/ci.yml`.
- Bandit-layer experiments and infrastructure are in `experiments/testing/`
  (heavy-tailed bandit, mechanism finding) and `ibrl/agents/`,
  `ibrl/environments/`, `ibrl/simulators/`, `ibrl/infrabayesian/`.
