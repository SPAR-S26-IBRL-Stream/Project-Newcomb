# Refactoring Plan: InfraBayesian Learner

**Date**: 2026-03-24 (updated 2026-03-25)
**Based on**: `coin-learning.ipynb` exploration of IB updating on coin flips
**Goal**: Incrementally refactor the notebook's IB implementation into production
source code, fix known bugs, generalize to arbitrary structured hypotheses, and
introduce Knightian uncertainty.

---

## 1. Motivation and Background

### 1.1 What this plan is about

The notebook `coin-learning.ipynb` implements infrabayesian (IB) updating on a
simple coin-flip problem. This plan refactors that work into production source
code, fixing bugs and generalizing the design to work with all environments in
the codebase.

### 1.2 Key IB concepts (for implementer context)

**A-measures**: An a-measure α = (λμ, b) is a scaled probability measure μ
with scale factor λ ≥ 0 and offset b. It acts on functions f via
α(f) = λ·μ(f) + b. In the non-KU case (single a-measure), this is equivalent
to a standard probability measure (λ=1, b=0).

**Infradistributions**: A set of a-measures. The expected value of a function
under an infradistribution is the *infimum* over all a-measures in the set.
With a single a-measure, this reduces to ordinary expectation. With multiple
a-measures (KU), it gives worst-case reasoning.

**Non-KU vs KU**: In the non-KU case (single a-measure = mixture over
hypotheses), IB updating is mathematically identical to Bayesian updating.
The IB framework adds value only in the KU case (multiple a-measures =
Knightian uncertainty over which model is correct, with no prior over models).

**Beliefs vs hypotheses**: A *hypothesis* is a single possible state of the
world (e.g., "the coin has p=0.3"). A *belief* is the agent's full epistemic
state — its posterior over hypotheses (e.g., "Beta(5, 12) over p"). The belief
encodes what the agent thinks after processing all observations.

### 1.3 What the notebook implements

- `AMeasure`: a-measures (λμ, b) over an explicit finite history space
- `Infradistribution`: sets of a-measures with expected values as infimum
- Classical Bayesian updating (single a-measure, non-KU)
- Knightian uncertainty (multiple a-measures, KU — partially explored)

### 1.4 Problems identified in code review

**Bugs**:
- **Offset double-counting**: `measure.offset += expect_m - expect0` should
  be `= expect_m - expect0`. The `+=` double-counts b because `expect_m`
  already includes the offset. Invisible when b=0 (all non-KU cases), but
  corrupts multi-step KU updates.
- **Invalid normalization**: With degenerate reward functions (e.g., f=0), the
  update denominator (global infimum of event probability) can be much smaller
  than individual a-measures' event probabilities, producing a-measures with
  α(1) > 1. This is a theoretical limitation of Definition 11, not a code bug.

**Limitations**:
- Enumerates all 2^n histories explicitly — infeasible beyond ~10 steps.
- KU case is exploratory, not integrated into the codebase.

### 1.5 Why history enumeration is unnecessary

For any hypothesis θ where the per-step likelihood P(observation | θ, context)
is computable, the a-measure can be updated sequentially without materializing
the history space. The requirement is not stationarity — it is that each
hypothesis defines a generative model with computable per-step likelihoods.

This covers all environments in the codebase:
- **Stationary bandits**: θ = reward probs per arm. Bernoulli likelihood.
- **Newcomb-like games**: θ = reward matrix M. Deterministic: δ(r = M[e,a]).
- **Switching environments**: θ = (probs_before, probs_after, switch_time).
  Likelihood depends on step number.

Settings where no structured hypothesis class exists (fully adversarial
arbitrary sequences) are outside both IB's practical scope and the scope
of this parameterization.

### 1.6 Why hypothesis enumeration is also unnecessary

Going further, none of the existing environments require enumerating a
discrete grid of hypotheses. Analytical sufficient statistics exist for all:

| Environment | Sufficient statistic | Enumeration? |
|---|---|---|
| Bandit | Beta(α, β) per arm | None |
| All Newcomb-like | Observed cells of reward matrix | None |
| Switching | Per-switch-time Beta stats | Switch times only (1D, O(T)) |

The design uses these analytical forms via "belief" objects rather than
hypothesis grids.

---

## 2. Key Design Principles

These principles emerged from design discussion and should guide
implementation decisions.

### 2.1 Beliefs are independent of environments

A belief is the agent's model of the world, which may be *misspecified*
relative to the actual environment. The agent doesn't know what world it's in.

- A `BanditBelief` can run against `SwitchingAdversaryEnvironment` (wrong
  model — assumes stationarity, but slowly adapts after switch)
- A `SwitchingBelief` can run against `BanditEnvironment` (over-parameterized
  — wastes capacity on non-existent switch points, but converges)

Therefore:
- Beliefs live in `ibrl/infrabayesian/beliefs.py`, independent of environments
- The agent is initialized with a **belief**, not an environment
- Environments know nothing about beliefs; they just produce Outcomes
- No `create_belief()` on environments — the correspondence is guidance, not code

### 2.2 Separate model from planning (DP principle)

The belief provides the agent's **model** of the reward structure. The agent's
**planning** step converts that model into a policy. These are never circular.

The belief exposes a **reward model** — its estimate of what reward each
(env_action, action) pair yields:

```python
class BaseBelief(ABC):
    def expected_reward_model(self, context=None):
        """
        Return the agent's current estimate of the reward structure.

        Returns:
            For bandit-like beliefs: NDArray of shape (num_actions,)
                values[a] = E[reward | arm=a]
            For game-like beliefs: NDArray of shape (num_env_actions, num_actions)
                values[e, a] = E[reward | env_action=e, arm=a]
        """
```

The agent then **plans** over this model to produce a policy:
- **1D reward vector** (bandits): feed directly to `build_greedy_policy(values)`.
  Trivial — just pick the arm with highest expected reward (with exploration).
- **2D reward matrix** (Newcomb-like): the predictor copies the agent's policy,
  so expected reward is π^T V π (a quadratic form in π). This is a well-defined
  optimization problem. `BaseNewcombLikeEnvironment.get_optimal_reward()` already
  solves this exact problem. The agent solves the same game in its policy builder.

This eliminates the circularity that existed when `expected_reward(arm)` required
a policy as input. The belief doesn't need to know the policy. The policy doesn't
feed back into the belief. One-directional dependency, like DP.

### 2.3 IB plumbing validated by equivalence testing

In the non-KU case, the AMeasure/Infradistribution wrapping is a no-op: λ=1,
b=0, single measure. The agent should produce **identical** results to using
the belief directly. Commit 3 tests validate this equivalence, proving the
pipes work before commit 4 adds KU logic that actually uses them.

### 2.4 Only InfraBayesianAgent is affected

Existing agents (`BayesianAgent`, `UCBAgent`, `IUCBAgent`, `EXP3Agent`, etc.)
are completely unaffected. They never touch AMeasure, Belief, or
Infradistribution. The `BaseAgent` and `BaseEnvironment` interfaces are
unchanged (except the Bernoulli fix to `SwitchingAdversaryEnvironment`).

---

## 3. Overview of Commits

| # | Summary | Key change |
|---|---------|------------|
| 1 | Extract notebook to source code | Direct port of `AMeasure`, `Infradistribution`, helpers |
| 2 | Fix accuracy bugs | Offset assignment, normalization guard |
| 3 | Belief-based a-measures | Independent beliefs, model/planning split, agent with IB plumbing |
| 4 | Knightian uncertainty | Multi-a-measure infradistributions, reward-dependent updates |

Each commit is independently testable and reviewable.

---

## 4. Commit 1: Extract Notebook to Source Code

### Goal
Faithful port of the notebook's classes and functions into `ibrl/`, preserving
the existing behavior exactly (including the bugs). Tests reproduce the
notebook's numerical outputs.

### New files

| File | Contents |
|------|----------|
| `ibrl/infrabayesian/a_measure.py` | `AMeasure` class — direct port from notebook |
| `ibrl/infrabayesian/infradistribution.py` | `Infradistribution` class — direct port |
| `ibrl/infrabayesian/helpers.py` | `match()`, `glue()`, reward functions, `Coin` enum |
| `ibrl/infrabayesian/__init__.py` | Public exports |
| `tests/test_infrabayesian.py` | Tests reproducing notebook outputs |

### Design notes

- Place IB primitives in `ibrl/infrabayesian/` (a new subpackage), not in
  `ibrl/agents/`. These are mathematical objects, not agents.
- `AMeasure` keeps the explicit measure vector over history space X, the scale λ,
  and offset b — exactly as in the notebook.
- `Infradistribution` keeps a list of `AMeasure` objects, computes expected values
  as the min, and implements `update()` and `probability()`.
- The global `X` (history space) is passed as a parameter rather than being a
  module-level variable, so the classes can work with different history spaces.
- Helper functions (`match`, `glue`, reward functions) go in `helpers.py`.

### Tests

1. **Classical (non-KU) coin flip**: 3 hypotheses (p=0.3, 0.5, 0.7), uniform
   prior, observe H. Assert P(H₂|H₁) = 0.553 (to 3 decimal places).
2. **KU coin flip**: 3 vertex a-measures, update with each reward function
   (zero, one, arbitrary). Assert updated scales and probabilities match
   notebook output exactly.
3. **Reward function independence (non-KU)**: Verify that for a single-a-measure
   infradistribution, probability and update are independent of reward function.

---

## 5. Commit 2: Fix Accuracy Bugs

### Goal
Fix the two known bugs. Tests demonstrate fixes without changing non-KU behavior.

### Bug 1: Offset double-counting in `Infradistribution.update()`

**Location**: The line `measure.offset += expect_m - expect0`

**Problem**: `expect_m = α(f·1_Eᶜ) = λ·μ(f·1_Eᶜ) + b` already includes b.
After `chop()`, offset is still b. So `+= expect_m - expect0` gives
`2b + λ·μ(f·1_Eᶜ) - expect0` instead of `λ·μ(f·1_Eᶜ) + b - expect0`.

**Fix**: `measure.offset = expect_m - expect0` (assignment, not `+=`).

**Impact**: No change when b=0 (non-KU). Fixes multi-step KU updates.

### Bug 2: Invalid a-measures after update (α(1) > 1)

**Problem**: Degenerate reward functions make the normalizing denominator
artificially small, producing a-measures with α(1) > 1.

**Fix**: Add `AMeasure.is_valid()` checking α(1) ≤ 1+ε. Add post-update
warning in `Infradistribution.update()`. Document that the update is only
well-defined for non-degenerate reward functions. This is a theoretical
limitation, not a code bug — the fix is validation and documentation.

### Tests

1. **Offset regression**: Infradistribution with nonzero offsets, verify
   updated evaluation matches the formula.
2. **Normalization**: KU update with reward_one → α(1) ≈ 1 ✓.
   KU update with reward_zero → warning fires ✓.

---

## 6. Commit 3: Belief-Based A-Measures (No History Enumeration)

### Goal
Replace explicit history vectors with belief-based a-measures using analytical
sufficient statistics. Cleanly separate the agent's **model** (belief) from
its **planning** (policy computation). Agent pipes everything through
AMeasure/Infradistribution, with tests proving equivalence to direct beliefs.

### 6.1 Belief interface

```python
class BaseBelief(ABC):
    """
    Agent's epistemic model of an environment. Encapsulates prior
    assumptions, sufficient statistics, and update rule.

    Independent of any specific environment — the agent chooses what
    to believe, and the environment is indifferent to that choice.
    """
    @abstractmethod
    def update(self, action, outcome, context=None):
        """Incorporate one observation into the sufficient statistics."""
        pass

    @abstractmethod
    def expected_reward_model(self, context=None):
        """
        The agent's current estimate of the reward structure.

        Returns:
            NDArray — either shape (num_actions,) for bandit-like beliefs,
            or shape (num_env_actions, num_actions) for game-like beliefs.

        The agent's planning step decides what to do with this model.
        The belief does NOT need a policy to compute this. No circularity.
        """
        pass

    @abstractmethod
    def copy(self):
        """Return an independent copy (needed for KU in commit 4)."""
        pass
```

### 6.2 Belief implementations

#### `BanditBelief`

```python
class BanditBelief(BaseBelief):
    """
    Belief for i.i.d. Bernoulli rewards per arm.
    Well-specified for: BanditEnvironment
    Misspecified but ok: SwitchingAdversaryEnvironment (slowly adapts)
    """
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.alpha = np.ones(num_actions)   # Beta prior α=1 (uniform)
        self.beta = np.ones(num_actions)    # Beta prior β=1

    def update(self, action, outcome, context=None):
        self.alpha[action] += outcome.reward        # reward is 0.0 or 1.0
        self.beta[action] += 1.0 - outcome.reward

    def expected_reward_model(self, context=None):
        # Returns shape (num_actions,) — a reward vector, not a matrix
        return self.alpha / (self.alpha + self.beta)

    def copy(self):
        c = BanditBelief.__new__(BanditBelief)
        c.num_actions = self.num_actions
        c.alpha = self.alpha.copy()
        c.beta = self.beta.copy()
        return c
```

No enumeration. O(1) per arm per step. Two floats per arm.

#### `NewcombLikeBelief`

```python
class NewcombLikeBelief(BaseBelief):
    """
    Belief for deterministic reward matrices.
    Well-specified for: Newcomb, Damascus, AsymDamascus, Coordination, PDbandit
    """
    def __init__(self, num_actions, prior_mean=0.5):
        self.num_actions = num_actions
        self.prior_mean = prior_mean
        self.observed = np.full((num_actions, num_actions), np.nan)

    def update(self, action, outcome, context=None):
        self.observed[outcome.env_action, action] = outcome.reward

    def expected_reward_model(self, context=None):
        # Returns shape (num_env_actions, num_actions) — a reward MATRIX
        # The agent's planning step handles the game theory (π^T V π)
        model = self.observed.copy()
        model[np.isnan(model)] = self.prior_mean
        return model

    def copy(self):
        c = NewcombLikeBelief.__new__(NewcombLikeBelief)
        c.num_actions = self.num_actions
        c.prior_mean = self.prior_mean
        c.observed = self.observed.copy()
        return c
```

No enumeration. O(1) per update. Returns K×K matrix for the planner.

#### `SwitchingBelief`

Note: this commit also updates `SwitchingAdversaryEnvironment` to use
Bernoulli rewards (matching `BanditEnvironment`) instead of Gaussian.
The Gaussian was an artifact. The environment change is:
`_resolve` returns `float(self.random.random() < self.values[action])`
instead of `self.random.normal(self.values[action], 0.1)`.

```python
class SwitchingBelief(BaseBelief):
    """
    Belief for Bernoulli rewards with a single unknown switch time.
    Well-specified for: SwitchingAdversaryEnvironment
    Over-parameterized but ok: BanditEnvironment (concentrates on "never switches")
    """
    def __init__(self, num_actions, max_steps):
        self.num_actions = num_actions
        self.max_steps = max_steps
        T, K = max_steps, num_actions

        self.log_weights = np.zeros(T)          # uniform prior over switch times
        self.alpha_before = np.ones((T, K))     # Beta stats before switch
        self.beta_before  = np.ones((T, K))
        self.alpha_after  = np.ones((T, K))     # Beta stats after switch
        self.beta_after   = np.ones((T, K))

    def update(self, action, outcome, context=None):
        step = context['step']
        r = outcome.reward  # 0.0 or 1.0

        for t in range(self.max_steps):
            if step < t:
                p = self.alpha_before[t, action] / (
                    self.alpha_before[t, action] + self.beta_before[t, action])
                self.log_weights[t] += r * np.log(p) + (1 - r) * np.log(1 - p)
                self.alpha_before[t, action] += r
                self.beta_before[t, action] += 1.0 - r
            else:
                p = self.alpha_after[t, action] / (
                    self.alpha_after[t, action] + self.beta_after[t, action])
                self.log_weights[t] += r * np.log(p) + (1 - r) * np.log(1 - p)
                self.alpha_after[t, action] += r
                self.beta_after[t, action] += 1.0 - r

    def expected_reward_model(self, context=None):
        # Returns shape (num_actions,) — weighted average over switch hypotheses
        step = context['step']
        log_w = self.log_weights - self.log_weights.max()
        weights = np.exp(log_w)
        weights /= weights.sum()

        model = np.zeros(self.num_actions)
        for t in range(self.max_steps):
            for a in range(self.num_actions):
                if step < t:
                    p = self.alpha_before[t, a] / (
                        self.alpha_before[t, a] + self.beta_before[t, a])
                else:
                    p = self.alpha_after[t, a] / (
                        self.alpha_after[t, a] + self.beta_after[t, a])
                model[a] += weights[t] * p
        return model

    def copy(self):
        c = SwitchingBelief.__new__(SwitchingBelief)
        c.num_actions = self.num_actions
        c.max_steps = self.max_steps
        c.log_weights = self.log_weights.copy()
        c.alpha_before = self.alpha_before.copy()
        c.beta_before = self.beta_before.copy()
        c.alpha_after = self.alpha_after.copy()
        c.beta_after = self.beta_after.copy()
        return c
```

Enumerates over T switch times only. O(T×K) per step.

### 6.3 `BeliefAMeasure`

Wraps a belief with the (λ, b) structure needed for IB. In non-KU mode,
λ=1 and b=0, making this a pure pass-through.

```python
class BeliefAMeasure:
    def __init__(self, belief, log_scale=0.0, offset=0.0):
        self.belief = belief
        self.log_scale = log_scale  # log(λ)
        self.offset = offset        # b

    def update(self, action, outcome, context=None):
        self.belief.update(action, outcome, context)

    def expected_reward_model(self, context=None):
        """λ * belief.expected_reward_model() + b"""
        scale = np.exp(self.log_scale)
        return scale * self.belief.expected_reward_model(context) + self.offset
```

### 6.4 `Infradistribution` (belief-based)

```python
class Infradistribution:
    def __init__(self, measures):
        self.measures = measures  # list of BeliefAMeasure

    def update(self, action, outcome, context=None):
        for m in self.measures:
            m.update(action, outcome, context)

    def expected_reward_model(self, context=None):
        """
        Non-KU (1 measure): returns that measure's model.
        KU (N measures): returns element-wise min over all models.
        """
        models = [m.expected_reward_model(context) for m in self.measures]
        return np.min(models, axis=0)
```

### 6.5 `InfraBayesianAgent`

The agent has two distinct responsibilities:
1. **Model update** (epistemics): pass observations to the infradistribution
2. **Planning** (decision-making): convert the reward model into a policy

These are never circular. The model does not depend on the policy.

```python
class InfraBayesianAgent(BaseGreedyAgent):
    """
    Agent using infrabayesian inference.

    Initialized with a BELIEF (epistemic model), not an environment.
    Wraps the belief in AMeasure/Infradistribution.

    get_probabilities() has two phases:
      1. MODEL: ask infradist for the expected reward structure
      2. PLAN: solve for the best policy given that structure

    update() has one phase:
      1. MODEL: pass observation to infradist to update beliefs

    Only InfraBayesianAgent uses AMeasure/Infradistribution/Belief.
    Other agents are unaffected.
    """
    def __init__(self, *args, belief=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._belief_template = belief

    def reset(self):
        super().reset()
        belief = self._belief_template.copy()
        self.infradist = Infradistribution([
            BeliefAMeasure(belief)  # single measure, λ=1, b=0
        ])

    def update(self, probabilities, action, outcome):
        """MODEL phase: update beliefs with observation."""
        super().update(probabilities, action, outcome)
        context = {'step': self.step, 'policy': probabilities}
        self.infradist.update(action, outcome, context)

    def get_probabilities(self):
        """MODEL then PLAN: get reward structure, solve for policy."""
        context = {'step': self.step}

        # MODEL: get the reward structure from the infradistribution
        reward_model = self.infradist.expected_reward_model(context)

        # PLAN: convert reward structure into a policy
        if reward_model.ndim == 1:
            # Bandit-like: reward_model is shape (num_actions,)
            # Each entry is the expected reward for that arm
            values = reward_model
        elif reward_model.ndim == 2:
            # Game-like: reward_model is shape (num_env_actions, num_actions)
            # This is a matrix game where the predictor copies our policy
            # Expected reward = π^T V π, need to optimize over π
            values = self._solve_game(reward_model)
        else:
            raise ValueError(f"Unexpected reward model shape: {reward_model.shape}")

        return self.build_greedy_policy(values)

    def _solve_game(self, V):
        """
        Solve the Newcomb-like game: find values to feed to greedy policy.

        The predictor samples from our policy π, so expected reward for
        arm a is Σ_e π(e) * V[e,a]. For a 2-action game, the optimal π
        can be found analytically (same logic as
        BaseNewcombLikeEnvironment.get_optimal_reward).

        For now, return the diagonal of V as a heuristic (expected reward
        when predictor correctly predicts our action). This is a
        simplification; proper game solving can be added later.
        """
        return np.diag(V)

    def dump_state(self):
        """Show current reward model (for debugging)."""
        context = {'step': self.step}
        model = self.infradist.expected_reward_model(context)
        return dump_array(model) if model.ndim == 1 else dump_array(np.diag(model))
```

**Notes on `_solve_game`**: The diagonal heuristic (V[a,a] = reward when
predictor correctly predicts action a) is a starting point. The proper
solution is to find π maximizing π^T V π, which for 2×2 is analytical:

```python
(a,b),(c,d) = V
# Optimal is max of: always-0 (reward=a), always-1 (reward=d),
# or mixed if a+d-b-c < 0
```

This matches `BaseNewcombLikeEnvironment.get_optimal_reward()`. Can be
refined later; the architecture supports it cleanly because the model
and planning are separated.

**Notes on `build_greedy_policy`**: Inherited from `BaseGreedyAgent`. Takes
a "values" array (one value per arm) and produces a policy using either
epsilon-greedy (mostly pick the best arm, occasionally explore uniformly)
or softmax (probability proportional to exponentiated values). The
exploration parameters (epsilon, temperature, decay) are configurable.
This is the same policy builder used by `BayesianAgent`.

**Notes on `dump_state`**: A debugging method called by the simulator when
`verbose > 0`. Shows the agent's internal state in a single line. For the
IB agent, this is the posterior expected reward per arm.

### 6.6 Registration in `construction.py`

Since the agent takes a belief (not an environment), the construction
string maps belief names to classes:

```python
# Usage: "infrabayesian:belief=bandit" or "infrabayesian:belief=newcomb"
# construction.py maps these to BanditBelief(num_actions), etc.
```

### 6.7 `SwitchingAdversaryEnvironment`: Bernoulli update

Changes `_resolve` from Gaussian to Bernoulli:

```python
# Before: return self.random.normal(self.values[action], 0.1)
# After:
return float(self.random.random() < self.values[action])
```

Standalone fix. Other agents see 0/1 rewards instead of noisy Gaussians.

### 6.8 Tests

1. **Bandit: IB pipe vs direct belief**
   - Create `BanditBelief` directly, feed 50 fixed (action, reward) pairs
   - Create `InfraBayesianAgent` with same belief type, feed same observations
   - Assert `belief.expected_reward_model()` == `agent.infradist.expected_reward_model()`
     at every step, to machine precision

2. **Newcomb-like: IB pipe vs direct belief**
   - Same structure with `NewcombLikeBelief`
   - Feed fixed (env_action, action, reward) triples
   - Assert identical reward matrices at every step

3. **Switching: IB pipe vs direct belief**
   - Same structure with `SwitchingBelief`
   - Feed observations that switch mid-sequence
   - Assert identical reward vectors

4. **Misspecified belief**
   - `BanditBelief` against `SwitchingAdversaryEnvironment`: agent learns,
     slowly adapts after switch
   - `SwitchingBelief` against `BanditEnvironment`: posterior concentrates
     on "never switches"

5. **Equivalence with explicit a-measures (small problem)**
   - 2-flip coin from commit 1 (explicit history AMeasure) vs hand-built
     BanditBelief (3 hypotheses, p=0.3/0.5/0.7)
   - Assert identical posterior P(H₂|H₁) = 0.553

6. **Simulator integration (bandit)**
   - `simulate(BanditEnvironment, InfraBayesianAgent)` for 200 steps
   - Average reward increases over time

7. **Simulator integration (Newcomb)**
   - `simulate(NewcombEnvironment, InfraBayesianAgent)` for 200 steps
   - Learning signal present

---

## 7. Commit 4: Knightian Uncertainty

### Goal
Support infradistributions with multiple a-measures, representing genuine
ambiguity over hypotheses. Expected value becomes an infimum, and the update
depends on the reward function.

### Changes to `Infradistribution`

- Constructor accepts multiple `BeliefAMeasure` objects (credal set vertices),
  each with its own independent belief copy and (λ, b)
- `expected_reward_model()` already returns element-wise min — with multiple
  measures this gives the pessimistic (worst-case) model
- `update()` applies the full Definition 11 update: modifies λ and b per
  measure using the global infimum in the denominator. Each belief still
  updates its own sufficient statistics.

### KU-specific design questions to resolve during implementation

1. **How to specify the credal set?**
   - Explicit vertex list (as in notebook)
   - Group hypotheses: KU between groups, Bayesian within
   - Full ambiguity: every hypothesis is its own vertex

2. **Reward function selection**: The KU update depends on reward function.
   For the agent, this should be the actual task reward. Must thread through
   update method.

3. **Vertex tracking**: Update is affine in α, so vertices map to vertices.
   Downward closure may introduce new extremal points. Track only explicit
   vertices for now (approximation).

### Agent changes

- `InfraBayesianAgent` gains `knightian=True/False` flag
- When `knightian=True`, `reset()` creates multiple `BeliefAMeasure` objects,
  each wrapping an independent `belief.copy()`
- Planning step receives pessimistic reward model (element-wise min) and
  solves for policy accordingly — may need maximin rather than greedy

### Tests

1. **KU coin flip**: Reproduce notebook's KU results with belief-based
   a-measures (3 hypotheses, various reward functions)
2. **KU vs non-KU**: Same belief type, same observations. KU agent is
   more conservative (lower expected rewards)
3. **KU on bandit**: KU agent learns (more slowly than non-KU, but converges)

---

## 8. File Plan (all commits)

### New files

| File | Commit | Contents |
|------|--------|----------|
| `ibrl/infrabayesian/__init__.py` | 1 | Public exports |
| `ibrl/infrabayesian/a_measure.py` | 1 | `AMeasure` (explicit history); `BeliefAMeasure` added in commit 3 |
| `ibrl/infrabayesian/infradistribution.py` | 1 | `Infradistribution` class |
| `ibrl/infrabayesian/helpers.py` | 1 | `match()`, `glue()`, reward functions |
| `ibrl/infrabayesian/beliefs.py` | 3 | `BaseBelief`, `BanditBelief`, `NewcombLikeBelief`, `SwitchingBelief` |
| `tests/test_infrabayesian.py` | 1 | Tests, expanded each commit |
| `ibrl/agents/infrabayesian.py` | 3 | `InfraBayesianAgent(BaseGreedyAgent)` |

### Modified files

| File | Commit | Change |
|------|--------|--------|
| `ibrl/infrabayesian/a_measure.py` | 2 | Add `is_valid()` |
| `ibrl/infrabayesian/infradistribution.py` | 2 | Fix offset bug, add normalization warning |
| `ibrl/infrabayesian/a_measure.py` | 3 | Add `BeliefAMeasure` |
| `ibrl/infrabayesian/infradistribution.py` | 3 | Add belief-based `update()` and `expected_reward_model()` |
| `ibrl/environments/switching.py` | 3 | Change `_resolve` from Gaussian to Bernoulli |
| `ibrl/agents/__init__.py` | 3 | Add `InfraBayesianAgent` |
| `ibrl/utils/construction.py` | 3 | Register `"infrabayesian"` with belief kwarg |
| `ibrl/infrabayesian/infradistribution.py` | 4 | KU update with reward function dependency |
| `ibrl/agents/infrabayesian.py` | 4 | Add `knightian` flag, multi-measure reset |

---

## 9. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Simulator                                       │
│                           (simulate loop)                                    │
│                                                                              │
│  ┌────────────────────────┐           ┌───────────────────────────────────┐  │
│  │    BaseEnvironment     │           │     InfraBayesianAgent            │  │
│  │    (knows nothing      │           │     (BaseGreedyAgent)             │  │
│  │     about beliefs)     │           │                                   │  │
│  │                        │           │  Initialized with a BELIEF,       │  │
│  │  step(π, action)       │──────────>│  not an environment.              │  │
│  │    -> Outcome          │  Outcome  │                                   │  │
│  │                        │           │                                   │  │
│  └──┬─────────────────────┘           │  ┌─ update(π, action, outcome) ─┐ │  │
│     │                                 │  │ MODEL: pass obs to infradist │ │  │
│     │ Environment types:              │  └──────────────────────────────┘ │  │
│     │  BanditEnvironment              │                                   │  │
│     │  Newcomb, Damascus, etc.        │  ┌─ get_probabilities() ────────┐ │  │
│     │  SwitchingAdversary             │  │ MODEL: get reward structure  │ │  │
│     │    (Bernoulli after fix)        │  │ PLAN:  solve for policy     │ │  │
│     │                                 │  │   1D model -> greedy        │ │  │
│     │                                 │  │   2D model -> solve game    │ │  │
│     │                                 │  └──────────────────────────────┘ │  │
│     │                                 │                                   │  │
│     │                                 │  ┌─────────────────────────────┐  │  │
│     │                                 │  │     Infradistribution      │  │  │
│     │                                 │  │  Non-KU: 1 BeliefAMeasure  │  │  │
│     │                                 │  │  KU:     N BeliefAMeasures │  │  │
│     │                                 │  │                             │  │  │
│     │                                 │  │  update(action, out, ctx)   │  │  │
│     │                                 │  │  expected_reward_model(ctx) │  │  │
│     │                                 │  │    non-KU: single model     │  │  │
│     │                                 │  │    KU: element-wise min     │  │  │
│     │                                 │  │                             │  │  │
│     │                                 │  │  ┌───────────────────────┐ │  │  │
│     │                                 │  │  │  BeliefAMeasure       │ │  │  │
│     │                                 │  │  │  (λ, b, belief)       │ │  │  │
│     │                                 │  │  │  λ*model + b          │ │  │  │
│     │                                 │  │  │                       │ │  │  │
│     │                                 │  │  │  ┌─────────────────┐ │ │  │  │
│     │                                 │  │  │  │   BaseBelief    │ │ │  │  │
│     │                                 │  │  │  │  (sufficient    │ │ │  │  │
│     │                                 │  │  │  │   statistics)   │ │ │  │  │
│     │                                 │  │  │  └─────────────────┘ │ │  │  │
│     │                                 │  │  └───────────────────────┘ │  │  │
│     │                                 │  └─────────────────────────────┘  │  │
│     │                                 └───────────────────────────────────┘  │
│     │                                                                        │
│     │  Belief types (independent of environments):                           │
│     │                                                                        │
│     │  BanditBelief          NewcombLikeBelief     SwitchingBelief           │
│     │    Beta(α,β) per arm     Observed cells        Per-switch-time         │
│     │    model: (K,) vector    model: (K,K) matrix   Beta stats              │
│     │    O(1) per update       O(1) per update       model: (K,) vector      │
│     │                                                 O(T×K) per update      │
│     │  Well-specified for:   Well-specified for:    Well-specified for:       │
│     │    Bandit                 all Newcomb-like      Switching               │
│     │  Misspecified ok:      Misspecified ok:       Over-param. ok:           │
│     │    Switching              (N/A)                 Bandit                  │
│     │                                                                        │
│     │  Other agents (BayesianAgent, UCBAgent, IUCBAgent, EXP3Agent, etc.)    │
│     │  are COMPLETELY UNAFFECTED. They never touch any of the above objects.  │
│     │                                                                        │
└──────────────────────────────────────────────────────────────────────────────┘

Data flow (one step, non-KU):

    Simulator                                         Agent
    ─────────                                         ─────
    1. calls agent.get_probabilities()
                                              MODEL: infradist.expected_reward_model()
                                                     -> BeliefAMeasure.expected_reward_model()
                                                        -> belief.expected_reward_model()
                                                           returns reward vector/matrix
                                              PLAN:  if 1D: build_greedy_policy(values)
                                                     if 2D: _solve_game(V) then greedy
                                              returns π

    2. samples action ~ π
    3. calls env.step(π, action) -> Outcome
    4. calls agent.update(π, action, outcome)
                                              MODEL: infradist.update(action, outcome, ctx)
                                                     -> BeliefAMeasure.update(...)
                                                        -> belief.update(action, outcome, ctx)
                                                           updates sufficient stats
```
