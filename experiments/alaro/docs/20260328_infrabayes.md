# InfraBayesian Agent: Implementation Reference

**Date**: 2026-03-28
**Branch**: `alaro/coin-learning-ku`
**Source of truth for**: the infrabayesian agent implementation in `ibrl/`

This document walks through the infrabayesian (IB) agent's architecture, the
math behind it, and how that math maps to the codebase. It is scoped to what
is currently implemented, using BernoulliBelief as the concrete example
throughout.

---

## 1. What This Agent Does

The `InfraBayesianAgent` is a bandit agent that supports **Knightian
uncertainty (KU)** — ambiguity over models without a prior. It generalizes
standard Bayesian learning.

The agent is initialized with a list of beliefs. Each belief is wrapped in
an `AMeasure` (with scale λ and offset b), and the set of AMeasures forms
an `Infradistribution`. There is a single code path regardless of how many
beliefs are provided:

- **Planning**: the infradistribution returns the element-wise **min** over
  all measures' reward models (worst-case reasoning).
- **Updating**: Definition 11 from *Basic Inframeasure Theory* updates each
  measure's belief (Bayesian conditioning), scale λ, and offset b.

With a single belief, the math naturally reduces to standard Bayesian
learning — λ stays 1, b stays 0, and the min over one element is that
element. This is verified by an equivalence test against
`BernoulliBayesianAgent`. With multiple beliefs, the agent exhibits genuine
Knightian uncertainty: measures diverge in scale and offset based on how
well they predicted each observation.

---

## 2. Key Concepts

### A-measures

An a-measure α = (λμ, b) is a scaled probability measure μ with scale λ ≥ 0
and offset b ≥ 0. It evaluates a function f as:

$$\alpha(f) = \lambda \cdot \mu(f) + b$$

In code (`a_measure.py`):

```python
class AMeasure:
    def __init__(self, belief: BaseBelief, log_scale: float = 0.0, offset: float = 0.0):
        self.belief = belief         # μ — normalized distribution (sufficient statistics)
        self.log_scale = log_scale   # log(λ) — scale in log space for numerical stability
        self.offset = offset         # b
    def expected_reward_model(self, context=None):
        """α(f) = λ · μ(f) + b — named 'expected_reward_model' because the
        belief returns posterior mean rewards per arm, and this method
        applies the a-measure's scale and offset to that estimate."""
        scale = np.exp(self.log_scale)
        return scale * self.belief.expected_reward_model(context) + self.offset
```

With a single belief, λ=1 and b=0 throughout, making this a standard
probability measure.

### Infradistributions

An infradistribution is a **set** of a-measures. It evaluates a function f by
taking the **infimum** (worst case) over all its a-measures:

$$E_H(f) = \min_k \left[\lambda_k \cdot \mu_k(f) + b_k\right]$$

With a single a-measure, the min is a no-op and this reduces to ordinary
expectation. With multiple a-measures, it gives worst-case reasoning.

### Beliefs

A belief is the agent's epistemic model — its posterior over hypotheses,
encoded via sufficient statistics. Beliefs are **independent of
environments**: the agent chooses what to believe; the environment is
indifferent.

`BernoulliBelief` tracks Beta(α, β) per arm:
- `update(action, outcome)`: α[a] += r, β[a] += (1-r)
- `expected_reward_model()`: returns α / (α + β) — posterior mean per arm
- `observation_probability(action, outcome)`: p^r · (1-p)^(1-r) where
  p = α[a] / (α[a] + β[a])

**Misspecified observations**: `BernoulliBelief.update()` raises `ValueError`
if reward ∉ [0, 1]. The agent does not attempt to handle observations outside
its belief's domain — this is by design (fail-fast). Use a utility mapping
(§6) to transform arbitrary rewards into [0, 1] before they reach the belief.

### Model/Planning Separation

The agent has two distinct phases with no circularity:

1. **MODEL**: The belief provides a reward estimate. The infradistribution
   takes the worst-case over all measures.
2. **PLAN**: The agent converts the reward model into a policy via
   `build_greedy_policy(values)`.

The belief does not need the policy. The policy does not feed back into the
belief. One-directional, like dynamic programming.

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                            Simulator                                   │
│                         (simulate loop)                                │
│                                                                        │
│  ┌──────────────────────┐         ┌──────────────────────────────────┐ │
│  │   BaseEnvironment    │         │    InfraBayesianAgent            │ │
│  │                      │         │    (BaseGreedyAgent)             │ │
│  │  step(π, action)     │─ Outcome ─>                                │ │
│  │   -> Outcome         │         │  ┌─ get_probabilities() ───────┐ │ │
│  │                      │         │  │ MODEL: infradist            │ │ │
│  │                      │         │  │   .expected_reward_model()  │ │ │
│  └──────��───────────────┘         │  │ PLAN: build_greedy_policy() │ │ │
│                                   │  └─────────────────────────────┘ │ │
│                                   │                                  │ │
│                                   │  ┌─ update(π, action, outcome) ┐ │ │
│                                   │  │ (optional utility mapping)  │ │ │
│                                   │  │ infradist.update(...)       │ │ │
│                                   │  └────────────────────────────┘ │ │
│                                   │                                  │ │
│                                   │  ┌────────────────────────────┐  │ │
│                                   │  │     Infradistribution      │  │ │
│                                   │  │                            │  │ │
│                                   │  │  1..N AMeasures            │  │ │
│                                   │  │                            │  │ │
│                                   │  │  expected_reward_model()   │  │ │
│                                   │  │    -> min over measures    │  │ │
│                                   │  │                            │  │ │
│                                   │  │  update()                  │  │ │
│                                   │  │    -> Definition 11        │  │ │
│                                   │  │                            │  │ │
│                                   │  │  ┌──────────────────────┐ │  │ │
��                                   │  │  │   AMeasure           │ ��  │ │
│                                   │  │  │   (log_scale, offset,│ │  │ │
│                                   │  │  │    belief)           │ │  │ │
│                                   │  │  │                      │ │  │ │
│                                   │  │  │  expected_reward_     │ │  │ │
│                                   │  │  │    model():          │ │  │ │
│                                   │  │  │    λ * belief_model  �� │  │ │
│                                   │  │  │     + offset         │ │  │ │
│                                   │  │  │                      │ │  │ │
│                                   │  │  │  ┌────────────────┐ │ │  │ │
│                                   │  │  │  │ BernoulliBelief│ │ │  │ │
│                                   │  │  │  │ Beta(α,β)/arm  │ │ │  │ │
│                                   │  │  │  │ model: (K,)    │ │  │ │
│                                   │  │  │  └────────────────┘ │ │  │ │
│                                   │  │  └──────────────────────┘ │  │ │
│                                   │  └────���───────────────────────┘  │ │
│                                   └────���─────────────────────────────┘ │
└───────────────────────────────────��────────────────────────────────────┘
```

### Data flow (one step)

```
Simulator                                    Agent
─────────                                    ─────
1. agent.get_probabilities()
                                     MODEL: infradist.expected_reward_model()
                                            -> for each AMeasure:
                                                 λ * belief.expected_reward_model() + b
                                            -> min over all measures
                                     PLAN:  build_greedy_policy(values)
                                     returns π

2. sample action ~ π
3. env.step(π, action) -> Outcome
4. agent.update(π, action, outcome)
                                     (utility mapping if provided)
                                     infradist.update() — Definition 11
                                       (see §4 below)
```

### File layout

| File | Contents |
|------|----------|
| `ibrl/infrabayesian/beliefs.py` | `BaseBelief`, `BernoulliBelief`, `GaussianBelief`, `NewcombLikeBelief` |
| `ibrl/infrabayesian/a_measure.py` | `AMeasure` — wraps belief with (λ, b) |
| `ibrl/infrabayesian/infradistribution.py` | `Infradistribution` — set of AMeasures, Definition 11 update |
| `ibrl/agents/infrabayesian.py` | `InfraBayesianAgent` — model/planning separation |
| `ibrl/agents/bernoulli_bayesian.py` | `BernoulliBayesianAgent` — Bayesian baseline for comparison |
| `tests/test_infrabayesian_beliefs.py` | Belief unit tests, single-belief equivalence tests |
| `tests/test_knightian_uncertainty.py` | Definition 11 math tests, validation, cohomogeneity |

---

## 4. The Update Rule (Definition 11)

This section explains the infradistribution's update step, using the `BernoulliBelief` as a concrete example. Each term is
annotated with the corresponding code in `infradistribution.py`.

### Setup

We have K a-measures. Each a-measure k has:
- A `BernoulliBelief` with parameters (α_k, β_k) per arm
- Scale factor λ_k (`exp(log_scale)`)
- Offset b_k (`offset`)

We observe: pulled arm `a`, got reward `r ∈ {0, 1}`.

The counterfactual value function `g` is a constant (default g=1; see §5).

### Step 1: Snapshot pre-update state

Before updating any beliefs, capture each measure's observation probability.
This is critical because `observation_probability()` uses the current
(prior) parameters, not the posterior.

For BernoulliBelief, the observation probability under measure k is:

$$P_k(\text{obs}) = p_k^r \cdot (1 - p_k)^{1-r}$$

where $p_k = \alpha_k[a] / (\alpha_k[a] + \beta_k[a])$ is measure k's
predicted probability of reward=1 on arm a.

**Code**: `_snapshot_measures()` → creates `_MeasureSnapshot` for each
measure, storing `obs_prob`, `scale` (λ), and `offset` (b). The dataclass
computes `not_obs_prob = 1 - obs_prob` internally.

### Step 2: Compute normalization

The infradistribution's "probability" of the observation:

$$P^g_H(L) = E_H(1 \star_L g) - E_H(0 \star_L g)$$

where the **glue operator** $f \star_L g = L \cdot f + (1-L) \cdot g$ splices
f on the observed branch and g on the non-observed branch.

Expanding with constant g:

$$E_H(1 \star_L g) = \min_k \left[\lambda_k \cdot P_k(\text{obs}) + g \cdot \lambda_k \cdot (1 - P_k(\text{obs})) + b_k\right]$$

$$E_H(0 \star_L g) = \min_k \left[g \cdot \lambda_k \cdot (1 - P_k(\text{obs})) + b_k\right]$$

The first term is each measure's value of "1 on observed, g on non-observed."
The second term is each measure's counterfactual value — what it assigns to
the non-observed branch alone.

**Note on history**: The agent does not store the full observation history.
Instead, the history is fully encoded in each measure's sufficient statistics
(α, β for BernoulliBelief) and a-measure parameters (λ, b). Each update step
conditions on the new observation, and the sufficient statistics accumulate all
past observations — this is the standard property of Bayesian sufficient
statistics. The a-measure's λ and b track cumulative scale and counterfactual
surplus from all prior updates.

**Code**:
- `_full_observation_value(snap)` → λ · obs_prob + g · λ · not_obs_prob + b
- `_counterfactual_value(snap)` → g · λ · not_obs_prob + b
- `_observation_probability(snapshots)` → min(full) - min(counterfactual)

### Step 3: Update each a-measure

For each measure k, three things happen:

**(a) Bayesian belief update** — standard conditioning on the observation:

$$\alpha_k[a] \mathrel{+}= r, \quad \beta_k[a] \mathrel{+}= (1 - r)$$

**(b) Scale update** — rescale λ by how well this measure predicted the
observation, normalized by the infradistribution's probability:

$$\lambda_k^{\text{new}} = \frac{\lambda_k \cdot P_k(\text{obs})}{P^g_H(L)}$$

Measures that predicted the observation well (high $P_k(\text{obs})$) get
upweighted. Measures that didn't predict it get downweighted.

**(c) Offset update** — absorb the counterfactual surplus:

$$b_k^{\text{new}} = \frac{\text{cfval}_k - \min_j(\text{cfval}_j)}{P^g_H(L)}$$

where $\text{cfval}_k = c \cdot \lambda_k \cdot (1 - P_k(\text{obs})) + b_k$.

Measures that assign more value to the non-observed branch (relative to the
worst-case measure) accumulate a larger offset. The offset tracks how much
"counterfactual surplus" this measure carries — value it assigned to things
that didn't happen, above the minimum.

**Code**: `_apply_ku_update()` ties all three steps together.

### Sanity check: single belief reduces to Bayesian

With K=1, λ=1, b=0, the update naturally simplifies:
- Counterfactual values: only one measure, so surplus is 0 → b stays 0
- $P^g_H(L) = P_1(\text{obs})$ → λ_new = P(obs)/P(obs) = 1
- Only the belief updates. Pure Bayesian conditioning. ✓

No special-casing needed — the single code path handles this automatically.

---

## 5. The Role of g (Counterfactual Value)

`g` controls how much value the agent assigns to outcomes that **didn't
happen**. It is stored as `self.g` on `Infradistribution`.

### Why g exists

In standard Bayesian conditioning, you observe evidence and discard
everything inconsistent with it. But infradistributions are not probability
distributions — the infimum makes expected values nonlinear. When you
condition (restrict to the observation), you throw away parts of the
measure that contributed to the infimum. Without compensation, this causes
**dynamic inconsistency**: the agent's preferences shift after observing
evidence, even though no new information was gained about the relevant
future.

The offset b in a-measures tracks counterfactual value — what the agent
committed to on the other branch — preventing this inconsistency.

### g=0 vs g=1

| g | Behavior | Character |
|---|----------|-----------|
| 0 | Zero utility to non-observed outcomes. Offsets stay 0. KU structure preserved only in λ. | KU collapses — no benefit over independent Bayesian |
| 1 | Full utility to non-observed outcomes. Offsets become non-zero. Measures genuinely diverge. | Non-trivial KU. **Preserves cohomogeneity** (λ+b=1). |

**g=1 is recommended** based on the IB literature (Diffractor: "somewhat
more sensible behavior"). It conditions on distributions most likely to have
generated the observation and preserves the structural invariant λ_k + b_k = 1
for all measures at every step.

### g is not a free hyperparameter

In the full theory, g is determined by the problem structure — specifically,
it falls out of decomposing the agent's utility function relative to the
history partition. The "correct" g depends on the agent's utility function
and counterfactual policy. Constant g=1 is a well-motivated simplification
that exercises the full KU machinery. The interface accepts g as a parameter
for future experimentation.

### Key references

- [Basic Inframeasure Theory](https://www.alignmentforum.org/posts/YAa4qcMyoucRS2Ykr/) — Definitions 9-11
- [Belief Functions and Decision Theory](https://www.greaterwrong.com/posts/e8qFDMzs2u9xf5ie6/) — Dynamic consistency proof (Theorem 5)
- [Less Basic Inframeasure Theory](https://www.greaterwrong.com/posts/idP5E5XhJGh9T5Yq9/) — g=0 vs g=1 comparison

---

## 6. Utility Mapping

IB theory requires all functions to map to [0, 1]. Environments can produce
arbitrary rewards. The agent applies an optional **utility mapping** that
transforms raw rewards before passing them to the infradistribution.

```python
agent = InfraBayesianAgent(
    beliefs=[...],
    utility=lambda r: np.clip(r / 15.0, 0.0, 1.0),  # for Newcomb rewards in [0, 15]
)
```

- Default: `None` (no mapping — suitable when rewards are already in [0, 1],
  e.g., Bernoulli bandits)
- The base agent still sees raw rewards (for regret tracking etc.)
- Only the infradistribution sees mapped rewards
- The mapping must produce values in [0, 1]; a `ValueError` is raised otherwise

With a single belief, the utility mapping doesn't affect behavior (λ=1, b=0,
only relative ordering matters). With multiple beliefs, it matters because
the offset arithmetic assumes [0, 1] bounds.

---

## 7. Concrete Example: BernoulliBelief KU

### Setup

Two BernoulliBelief measures with different priors for a 2-arm bandit:
- Measure A: Beta(2, 4) per arm → prior mean p_A = 1/3 (pessimistic)
- Measure B: Beta(4, 2) per arm → prior mean p_B = 2/3 (optimistic)

Both start with λ=1, b=0. Using g=1.

### Observe: arm 0, reward 1

**Snapshot** (before update):
- P_A(obs) = p_A = 1/3, P_A(not obs) = 2/3
- P_B(obs) = p_B = 2/3, P_B(not obs) = 1/3

**Counterfactual values** (g=1):
- cfval_A = 1 · 1 · 2/3 + 0 = 2/3
- cfval_B = 1 · 1 · 1/3 + 0 = 1/3
- worst-case: min(2/3, 1/3) = 1/3

**Full observation values**:
- full_A = 1 · 1/3 + 1 · 1 · 2/3 + 0 = 1
- full_B = 1 · 2/3 + 1 · 1 · 1/3 + 0 = 1
- worst-case: min(1, 1) = 1

**Normalization**: P^g_H(L) = 1 - 1/3 = 2/3

**Updated scales**:
- λ_A = 1 · (1/3) / (2/3) = 1/2
- λ_B = 1 · (2/3) / (2/3) = 1

**Updated offsets**:
- b_A = (2/3 - 1/3) / (2/3) = 1/2
- b_B = (1/3 - 1/3) / (2/3) = 0

**Check cohomogeneity**: λ_A + b_A = 1/2 + 1/2 = 1 ✓, λ_B + b_B = 1 + 0 = 1 ✓

Measure A (pessimistic) got downweighted in scale but gained offset.
Measure B (optimistic) kept full scale and zero offset — it predicted the
observation better.

---

## 8. Test Coverage

### `test_infrabayesian_beliefs.py` (17 tests)

- **Belief unit tests**: BernoulliBelief conjugate update, NewcombLikeBelief
  matrix filling, copy independence
- **AMeasure**: pass-through with λ=1/b=0, scale+offset applied correctly
- **Infradistribution**: single-measure pass-through, update propagation
- **Equivalence**: IB agent with a single BernoulliBelief produces identical
  rewards/actions to `BernoulliBayesianAgent` on the same bernoulli bandit
- **Simulator integration**: agent learns on bernoulli bandit, runs on Newcomb

### `test_knightian_uncertainty.py` (26 tests)

- **Observation probability**: BernoulliBelief P(obs) for reward=0 and
  reward=1, boundary cases (p near 0 or 1)
- **Single belief**: Definition 11 with one measure leaves λ=1, b=0
- **Multiple beliefs**: hand-computed λ and b values after one update with
  g=1.0 (the worked example in §7 above)
- **g=0 degeneracy**: offsets stay 0, KU collapses
- **g non-trivial** (g=0.5): intermediate behavior between g=0 and g=1
- **Cohomogeneity**: λ+b=1 preserved over 20 multi-step updates with g=1
- **Validation**: ValueError for invalid g, empty measures, out-of-range
  probabilities
- **Agent integration**: `beliefs=[single]` works, `beliefs=[multiple]`
  runs without error

---

## 9. Open Questions

1. **How to choose initial beliefs?** The current demos use hand-picked Beta
   priors (e.g., pessimistic Beta(1,3) + optimistic Beta(3,1)). What's a
   principled default set of priors for a given problem?

2. **Refining g beyond a constant**: g=1 is well-motivated (see §5), but
   the full theory says g should be derived from the agent's utility function
   and counterfactual policy. For cumulative-reward bandits, what does this
   reduce to?

3. **Does the vertex set need to grow?** The paper mentions that downward
   closure may introduce new extremal points. Currently we only track the
   initial vertices (an approximation). Monitor whether this causes issues.

4. **GaussianBelief KU support**: `observation_probability()` is not yet
   implemented for `GaussianBelief` — it needs an assumed noise variance.
   See the docstring on that method for notes.

5. **Construction string for multiple beliefs**: How should multiple beliefs
   with different priors be specified via `construction.py`? E.g.,
   `"infrabayesian:beliefs=bernoulli:3"` to create 3 vertices?

6. **Fuzzy observations**: All current environments have crisp observation
   indicators (L ∈ {0, 1}). The math already handles fuzzy L (L ∈ [0, 1]),
   which would matter for noisy sensors, continuous outcomes with soft
   likelihoods, or aggregated batch observations. Supporting this could
   demonstrate IB's theoretical advantages over standard Bayesian conditioning.
