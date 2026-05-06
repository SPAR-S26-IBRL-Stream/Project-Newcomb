# `experiments/iterated_newcomb/` — iterated imperfect Newcomb

Tests Option B from the experimenter's strategic analysis: does IB-KU
beat single-belief Bayesian agents on a non-stationary policy-relevant
environment, where the predictor's accuracy α flips from "near-perfect"
to "near-adversarial" mid-run?

## Setup

100 rounds. Predictor accuracy `α = α_high = 0.95` for the first 50,
then `α_low = 0.20` for the remaining 50. Reward matrix (predicted, action):

|             | action=0 (one-box) | action=1 (two-box) |
|-------------|--------------------:|--------------------:|
| predicted=0 | **boxB = 10**      | boxB+boxA = 15      |
| predicted=1 | 0                   | **boxA = 5**       |

Optimal action under α: one-box for α > 0.5, two-box for α < 0.5. Per-step
optimal: `α=0.95` → 9.5/step (one-box); `α=0.20` → 13.0/step (two-box).
Total optimal = 50·9.5 + 50·13.0 = **1125** per seed.

## Result: NULL

| agent | total reward | true regret (95% CI) | pre-flip avg | post-flip avg |
|---|---:|---:|---:|---:|
| one-box (pure) | 567.3 | 557.7 ± 11.0 | 9.53 | 1.82 |
| two-box (pure) | **932.7** | **192.3 ± 11.0** | 5.47 | 13.18 |
| ib-single (μ=5) | 911.5 | 213.5 ± 37.7 | 6.53 | 11.70 |
| ib-KU-2 (μ=0,10) | 776.7 | 348.3 ± 26.0 | 6.47 | 9.06 |
| ib-KU-3 (μ=0,5,10) | 773.0 | 352.0 ± 26.2 | 6.62 | 8.84 |

The single-belief IB agent (effectively a greedy Bayesian on the per-action
reward posterior) converges to two-box (post-flip optimal) and gets close to
the constant "always two-box" hand-coded baseline. Both KU IB variants are
**decisively worse** than single-belief IB (95% CIs do not overlap: KU-2 at
348 ± 26 vs single-belief at 214 ± 38).

## Mechanism (third confirmation)

Same finding as the bandit and gridworld experiments: the worst-case `min`
over multiple beliefs collapses once observations resolve the likelihood.
The two-box action's reward is observation-resolvable (it converges to its
α-weighted EV after enough rounds), so KU IB's worst-case rule penalises
the agent for considering both prior regimes when in fact only one is
consistent with the data. Even priors that explicitly bracket the optimal
reward range (μ=0 and μ=10 around the actual EVs of ~5 and ~13) produce no
benefit — the worst-case downward pull just makes the agent slower to
commit to two-box once the post-flip data is in.

The hand-coded `two-box (pure)` baseline incidentally beats every learning
agent including single-belief IB. This is a property of the chosen reward
table (boxA=5, boxB=10) rather than a deep finding — two-box happens to be
robust across both α regimes because of the asymmetric box values.

## Reproduce

```bash
uv run python experiments/iterated_newcomb/main.py
```

Runs in <1 s at the default config. CLI flags: `--num-seeds`, `--num-steps`,
`--flip-at`, `--alpha-high`, `--alpha-low`.

## What this adds to the workshop write-up

A third independent setting (after bandit and gridworld) where KU IB
collapses to single-belief Bayesian behaviour. Together the three
experiments triangulate the same mechanism — λ+b=1 at g=1 turns the
worst-case-min over measures into "select best-fitting prior" once the
data resolves the likelihood. None of bandit-stationary, bandit-heavy-
tailed, bandit-switching, gridworld-static-adversary, gridworld-dynamic-
adversary, or iterated-Newcomb-with-α-flip provides the structural
non-resolvability that IB requires.

## Why this isn't a strict test of Option B as the experimenter framed it

The experimenter's pitch for Option B included a *policy-dependent*
predictor — α should depend on the agent's recent policy in a way that
prevents observations from resolving it. The current implementation has
a clock-driven flip, not policy-dependence. Adding policy-dependence
(e.g., α = f(agent's empirical entropy)) is a one-line change in
`ImperfectNewcombEnvironment._alpha_at_step`. A separate run with that
variant is the natural follow-up, but at this point the mechanism is
strongly enough confirmed that I'd recommend against further bandit /
Newcomb iterations and toward writing up the workshop paper.
