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

### Beliefs

A belief is the agent's epistemic model — its posterior over hypotheses,
encoded via sufficient statistics. Beliefs are **independent of
environments**: the agent chooses what to believe; the environment is
indifferent.

The abstract interface (`beliefs.py`):

```python
class BaseBelief(ABC):
    @abstractmethod
    def update(self, action: int, outcome: Outcome):
        """Incorporate one observation into the sufficient statistics."""

    @abstractmethod
    def predict_rewards(self) -> NDArray[np.float64]:
        """Current estimate of the reward structure.
        Shape (num_actions,) for bandits, (num_env_actions, num_actions) for games."""

    @abstractmethod
    def compute_outcome_probability(self, action: int, outcome: Outcome) -> float:
        """P(this observation) under the current belief.
        Must be called BEFORE update() — uses prior parameters, not posterior."""

    @abstractmethod
    def copy(self) -> "BaseBelief":
        """Return an independent copy."""
```

`BernoulliBelief` is the concrete implementation used throughout. It tracks
a Beta(α, β) distribution per arm:

```python
class BernoulliBelief(BaseBelief):
    def __init__(self, num_actions: int):
        # Beta(1,1) uniform prior
        self.alpha = np.ones(num_actions)   
        self.beta = np.ones(num_actions)    

    def update(self, action: int, outcome: Outcome):
        self.alpha[action] += outcome.reward          # α += r
        self.beta[action] += 1.0 - outcome.reward     # β += (1 - r)

    def predict_rewards(self) -> NDArray[np.float64]:
        return self.alpha / (self.alpha + self.beta)   # posterior mean per arm

    def compute_outcome_probability(self, action: int, outcome: Outcome) -> float:
        r = outcome.reward
        p = self.alpha[action] / (self.alpha[action] + self.beta[action])
        return p ** r * (1 - p) ** (1 - r)             # Bernoulli likelihood
```

**Misspecified observations**: `BernoulliBelief.update()` raises `ValueError`
if reward ∉ [0, 1]. The agent does not attempt to handle observations outside
its belief's domain — this is by design (fail-fast). For environments with
rewards outside [0, 1], a reward-to-value mapping is needed before the
observation reaches the belief (see open question #8).

### A-measures

An a-measure α = (λμ, b) is a scaled probability measure μ with scale λ ≥ 0
and offset b ≥ 0. It evaluates a function f as:

$$\alpha(f) = \lambda \cdot \mu(f) + b$$

In code (`a_measure.py`):

```python
class AMeasure:
    def __init__(self, belief: BaseBelief, scale: float = 1.0, offset: float = 0.0):
        self.belief = belief
        self.scale = scale    # λ
        self.offset = offset  # b

    def update(self, action: int, outcome: Outcome):
        self.belief.update(action, outcome)

    def evaluate(self) -> NDArray[np.float64]:
        """α(f) = λ · μ(f) + b"""
        return self.scale * self.belief.predict_rewards() + self.offset
```

With a single belief, λ=1 and b=0 throughout, making `evaluate()` a pure
pass-through to `belief.predict_rewards()`.

### Infradistributions

An infradistribution is a **set** of a-measures. It evaluates a function f by
taking the **infimum** (worst case) over all its a-measures:

$$E_H(f) = \min_k \left[\lambda_k \cdot \mu_k(f) + b_k\right]$$

```python
class Infradistribution:
    def __init__(self, measures: list[AMeasure], g: float = 1.0):
        self.measures = measures
        self.g = g

    def evaluate(self) -> NDArray[np.float64]:
        # min across measures (axis=0), producing shape (num_actions,)
        models = [m.evaluate() for m in self.measures]
        return np.min(models, axis=0)
```

With a single a-measure, the min is a no-op and this reduces to ordinary
expectation. With multiple a-measures, it gives worst-case reasoning.

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

### Agent lifecycle

On `reset()`, the agent copies each belief template and wraps it in an
AMeasure, then bundles them into an Infradistribution:

```python
class InfraBayesianAgent(BaseGreedyAgent):
    def __init__(self, *args, beliefs: list[BaseBelief],
                 g: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._belief_templates = beliefs
        self._g = g

    def reset(self):
        super().reset()
        measures = [AMeasure(b.copy()) for b in self._belief_templates]
        self.infradist = Infradistribution(measures, g=self._g)
```

### Data flow (one step)

**Planning** — `get_probabilities()`:

```python
def get_probabilities(self) -> NDArray[np.float64]:
    # MODEL: evaluate the reward function under worst-case measure
    reward_model = self.infradist.evaluate()

    # PLAN: convert reward structure into a policy
    if reward_model.ndim == 1:
        values = reward_model                    # bandit: (num_actions,)
    elif reward_model.ndim == 2:
        values = self._solve_game(reward_model)  # game: use diagonal heuristic
    return self.build_greedy_policy(values)
```

**Updating** — `update()`:

```python
def update(self, probabilities, action, outcome):
    super().update(probabilities, action, outcome)
    self.infradist.update(action, outcome)
```

The simulator calls these in a loop:

```
1. π = agent.get_probabilities()        # MODEL + PLAN
2. action ~ π                            # sample
3. outcome = env.step(π, action)         # environment responds
4. agent.update(π, action, outcome)      # MODEL update (Definition 11)
```

---

## 4. The Update Rule (Definition 11)

This section walks through `Infradistribution.update()`, which implements
Definition 11 from *Basic Inframeasure Theory*. The full method is shown
at the end of this section; the subsections below explain each piece.

### Setup

We have K a-measures. Each a-measure k has:
- A `BernoulliBelief` with parameters (α_k, β_k) per arm
- Scale factor λ_k (`scale`)
- Offset b_k (`offset`)

We observe: pulled arm `a`, got reward `r ∈ {0, 1}`.

The counterfactual value function `g` is a constant (default g=1; see §5).

### Step 1: Snapshot pre-update state

Before updating any beliefs, capture each measure's state. This is critical
because `compute_outcome_probability()` uses the current (prior) parameters,
not the posterior.

For BernoulliBelief, the observation probability under measure k is:

$$P_k(\text{obs}) = p_k^r \cdot (1 - p_k)^{1-r}$$

where $p_k = \alpha_k / (\alpha_k + \beta_k)$ is measure k's predicted
probability of reward=1 on the pulled arm.

```python
@dataclass
class _MeasureSnapshot:
    obs_prob: float       # μ_k(L)     — P(observation) under this belief
    scale: float          # λ_k
    offset: float         # b_k        — current offset
    not_obs_prob: float = field(init=False)  # 1 - obs_prob

    def __post_init__(self):
        self.not_obs_prob = 1.0 - self.obs_prob

def _snapshot_measures(self, action, outcome):
    snapshots = []
    for m in self.measures:
        snapshots.append(_MeasureSnapshot(
            obs_prob=m.belief.compute_outcome_probability(action, outcome),
            scale=m.scale,
            offset=m.offset,
        ))
    return snapshots
```

### Step 2: Compute normalization

The infradistribution's "probability" of the observation (How much value does actually observing L add, beyond what I'd get from the counterfactual path alone?):

$$P^g_H(L) = E_H(1 \star_L g) - E_H(0 \star_L g)$$

where the **glue operator** $f \star_L g = L \cdot f + (1-L) \cdot g$ splices
f on the observed branch and g on the non-observed branch.

Expanding with constant g:

$$E_H(1 \star_L g) = \min_k \left[\lambda_k \cdot P_k(\text{obs}) + g \cdot \lambda_k \cdot (1 - P_k(\text{obs})) + b_k\right]$$

$$E_H(0 \star_L g) = \min_k \left[g \cdot \lambda_k \cdot (1 - P_k(\text{obs})) + b_k\right]$$

The first term is each measure's value of "1 on observed, g on non-observed."
The second term is each measure's counterfactual value — what it assigns to
the non-observed branch alone.

```python
def _compute_counterfactual_value(self, snap: _MeasureSnapshot) -> float:
    """α_k((1-L) · g) = g · λ · P(not obs) + b"""
    return self.g * snap.scale * snap.not_obs_prob + snap.offset

def _compute_full_value(self, snap: _MeasureSnapshot) -> float:
    """α_k(1 ★_L g) = λ · P(obs) + g · λ · P(not obs) + b"""
    return (snap.scale * snap.obs_prob
            + self.g * snap.scale * snap.not_obs_prob
            + snap.offset)

def _compute_normalization(self, snapshots) -> float:
    """P^g_H(L) = min_k[full_value_k] - min_k[counterfactual_value_k]"""
    worst_case_full = min(self._compute_full_value(s) for s in snapshots)
    worst_case_counterfactual = min(
        self._compute_counterfactual_value(s) for s in snapshots
    )
    prob = worst_case_full - worst_case_counterfactual
    if prob <= 0:
        raise ValueError(
            f"P^g_H(L) must be > 0 (observation has zero probability "
            f"under worst-case measure), got {prob}")
    return prob
```

**Note on history**: The agent does not store the full observation history.
Instead, the history is fully encoded in each measure's sufficient statistics
(α, β for BernoulliBelief) and a-measure parameters (λ, b). Each update step
conditions on the new observation, and the sufficient statistics accumulate all
past observations — this is the standard property of Bayesian sufficient
statistics. The a-measure's λ and b track cumulative scale and counterfactual
surplus from all prior updates.

### Step 3: Update each a-measure

For each measure k, three things happen:

**(a) Bayesian belief update** — standard conditioning on the observation:

$$\alpha_k \mathrel{+}= r, \quad \beta_k \mathrel{+}= (1 - r)$$

**(b) Scale update** — rescale λ by how well this measure predicted the
observation, normalized by the infradistribution's probability:

$$\lambda_k^{\text{new}} = \frac{\lambda_k \cdot P_k(\text{obs})}{P^g_H(L)}$$

Measures that predicted the observation well (high $P_k(\text{obs})$) get
upweighted. Measures that didn't predict it get downweighted.

**(c) Offset update** — absorb the counterfactual surplus:

$$b_k^{\text{new}} = \frac{\text{cfval}_k - \min_j(\text{cfval}_j)}{P^g_H(L)}$$

where $\text{cfval}_k = g \cdot \lambda_k \cdot (1 - P_k(\text{obs})) + b_k$.

Measures that assign more value to the non-observed branch (relative to the
worst-case measure) accumulate a larger offset. The offset tracks how much
"counterfactual surplus" this measure carries — value it assigned to things
that didn't happen, above the minimum.

### Putting it all together

The complete `update()` method — snapshot, normalize, then update each measure
in a single loop:

```python
def update(self, action: int, outcome: Outcome):
    snapshots = self._snapshot_measures(action, outcome)
    normalization = self._compute_normalization(snapshots)

    worst_case_counterfactual = min(
        self._compute_counterfactual_value(s) for s in snapshots
    )

    for snap, m in zip(snapshots, self.measures):
        # (a) Bayesian update — belief conditions on observation
        m.belief.update(action, outcome)

        # (b) Scale update — rescale by P_k(obs), normalize
        m.scale = snap.scale * snap.obs_prob / normalization

        # (c) Offset update — absorb counterfactual surplus, normalize
        counterfactual_surplus = (
            self._compute_counterfactual_value(snap) - worst_case_counterfactual
        )
        m.offset = max(0.0, counterfactual_surplus) / normalization
```

### Sanity check: single belief reduces to Bayesian

With K=1, λ=1, b=0, the update naturally simplifies:
- Counterfactual values: only one measure, so surplus is 0 → b stays 0
- $P^g_H(L) = P_1(\text{obs})$ → λ_new = P(obs)/P(obs) = 1
- Only the belief updates. Pure Bayesian conditioning.

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

## 6. Concrete Example: BernoulliBelief KU

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

**Check cohomogeneity**: λ_A + b_A = 1/2 + 1/2 = 1, λ_B + b_B = 1 + 0 = 1

Measure A (pessimistic) got downweighted in scale but gained offset.
Measure B (optimistic) kept full scale and zero offset — it predicted the
observation better.

---

## 7. Test Coverage

### `test_infrabayesian_beliefs.py` (17 tests)

- **Belief unit tests**: BernoulliBelief conjugate update, NewcombLikeBelief
  matrix filling, copy independence
- **AMeasure**: pass-through with λ=1/b=0, scale+offset applied correctly
- **Infradistribution**: single-measure pass-through, update propagation
- **Equivalence**: IB agent with a single BernoulliBelief produces identical
  rewards/actions to `BernoulliBayesianAgent` on the same bernoulli bandit
- **Simulator integration**: agent learns on bernoulli bandit, runs on Newcomb

### `test_knightian_uncertainty.py` (27 tests)

- **Outcome probability**: BernoulliBelief P(obs) for reward=0 and
  reward=1, boundary cases (p near 0 or 1)
- **Single belief**: Definition 11 with one measure leaves λ=1, b=0
- **Multiple beliefs**: hand-computed λ and b values after one update with
  g=1.0 (the worked example in §6 above)
- **g=0 degeneracy**: offsets stay 0, KU collapses
- **g non-trivial** (g=0.5): intermediate behavior between g=0 and g=1
- **Cohomogeneity**: λ+b=1 preserved over 20 multi-step updates with g=1;
  100-step long-run test with asymmetric priors confirms drift stays < 1e-3
- **Validation**: ValueError for invalid g, empty measures, out-of-range
  probabilities
- **Agent integration**: `beliefs=[single]` works, `beliefs=[multiple]`
  runs without error

---

## 8. Open Questions

1. **How to choose initial beliefs?** The current demos use hand-picked Beta
   priors (e.g., pessimistic Beta(1,3) + optimistic Beta(3,1)). What's a
   principled default set of priors for a given problem?

2. **Refining g beyond a constant**: g=1 is well-motivated (see §5), but
   the full theory says g should be derived from the agent's utility function
   and counterfactual policy. For cumulative-reward bandits, what does this
   reduce to?

3. **Fixed vertex set vs. true infradistribution update**: Definition 11 is
   defined on the infradistribution (a closed convex set of a-measures), not
   on individual measures. Our implementation tracks a fixed list of vertices
   and updates each one independently using the global normalization constants.
   This works because the formula maps each (m, b) pair independently given
   E_H, so updating vertices and taking the lower envelope gives the correct
   updated infradistribution — *provided the vertex set is complete*. The
   concern is that conditioning can introduce new extreme points via downward
   closure. If that happens, our fixed set is an outer approximation: the min
   over our vertices is ≥ the true min, making the agent less pessimistic than
   it should be. Whether the vertex set needs to grow in practice, and how to
   detect or correct this, is unresolved.

4. **GaussianBelief KU support**: `compute_outcome_probability()` is not yet
   implemented for `GaussianBelief` — it needs an assumed noise variance.
   See the docstring on that method for notes.

5. **Construction string for multiple beliefs**: How should multiple beliefs
   with different priors be specified via `construction.py`? E.g.,
   `"infrabayesian:beliefs=bernoulli:3"` to create 3 vertices?

6. **Out-of-support observations and belief misspecification**: There is a
   meaningful distinction between *unlikely* data (reward=1 when p=0.01 —
   handled fine, the belief just updates aggressively) and *out-of-support*
   data (reward=0.5 under a Bernoulli belief — the belief literally cannot
   represent this). Currently the code raises `ValueError` and crashes, but
   this is not a principled answer. Observing out-of-support data is evidence
   that the belief class itself is wrong. Should the agent discard beliefs
   that can't accommodate the observation? Fall back to a more flexible belief
   class? How does this interact with KU — if one measure's belief is
   misspecified but another's isn't, should the misspecified one be removed
   from the infradistribution?

7. **Utility mapping and [0,1] bounds**: IB theory requires functions in
   [0,1], but environments like Newcomb produce rewards in [0,15]. A utility
   mapping parameter was previously implemented in the agent but removed
   because the abstraction was leaky — the belief didn't know whether it was
   being fed rewards or utilities, and the agent was rebuilding Outcome objects
   with mapped values stuffed into the reward field. The mapping needs to be
   reintroduced with a cleaner design. Options include: making it the caller's
   responsibility (map before passing to the agent), putting it in the belief,
   or using a wrapper/decorator pattern. Related: `predict_rewards()` is
   misleading if the belief has been trained on mapped utilities.

8. **Explore-exploit tradeoffs**: IB theory prescribes how to update and plan
   under ambiguity, but is silent on exploration. Worst-case planning over
   multiple beliefs is *pessimistic* (avoid arms that could be bad), which is
   the opposite of UCB-style *optimism* (try arms that could be good). Does
   KU provide implicit exploration, or does it lead to underexploration by
   settling on "safe" arms too quickly? Should we combine KU with explicit
   exploration strategies (UCB, Thompson sampling) on top of the IB reward
   model?

9. **0<g<1 causes cohomogeneity blowup in individual a-measures**: Definition
    11 uses a global normalization (P^g_H(L) = min over all measures) rather
    than per-measure normalization. With g=1 all measures have the same full
    value (λ+b=1 → α(1★_L 1)=1), so the global normalization works perfectly.
    With g<1, full values differ across measures, and non-worst-case measures
    end up with λ+b > 1 after each update — compounding exponentially. The
    infradistribution's lower envelope (the min) remains valid, but individual
    measures blow up numerically. Note: even with g=1, floating point error
    seeds the same instability if not corrected — the code enforces m.scale =
    1 - m.offset as a projection step for g=1. For g<1 no such projection is
    known. Options: (a) per-measure normalization (deviates from Definition 11),
    (b) periodic clamping/projection (not theoretically justified), (c) pruning
    irrelevant measures whose floor exceeds others' max, (d) restrict to g=1.

10. **Should updates account for action selection probability?** The
    `probabilities` argument is passed to `agent.update()` but ignored by
    almost every agent (including IB). In contextual bandits, inverse
    propensity weighting (IPW) corrects for the fact that rarely-chosen arms
    produce observations that should be upweighted: r_hat = r / P(a_chosen).
    Currently, Definition 11's P_k(obs) captures the *outcome* likelihood
    under each belief, but not the *action selection* likelihood. For on-policy
    Bayesian updates on the chosen arm this doesn't matter (we only update
    the arm we pulled). But for off-policy learning or cross-arm inference,
    ignoring the selection probability could introduce bias.
