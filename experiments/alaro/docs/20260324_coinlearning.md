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

## 7. Phase 4: Knightian Uncertainty

**Status**: Planning (as of 2026-03-26). Phases 1-3 complete. Phase 1 code
deleted from production (preserved in git history). Production code uses
belief-based architecture exclusively: BernoulliBelief, GaussianBelief,
NewcombLikeBelief.

**Note on stale names in this doc**: Earlier sections reference `BanditBelief`
(now `BernoulliBelief`), `SwitchingBelief` (removed from PR, saved here for
later), and Phase 1 classes (`AMeasure`, `Infradistribution`, `helpers.py` —
deleted from production).

---

### 7.1 What is Knightian Uncertainty and why does it matter?

In standard Bayesian learning, the agent has a **prior distribution** over
hypotheses and updates it with Bayes' rule. Even when there are many
hypotheses, the agent always has a single combined model — a weighted average
of all hypotheses.

**Knightian uncertainty (KU)** is when the agent doesn't even have a prior.
It has a *set* of plausible models and refuses to assign relative
probabilities to them. Instead of "I think model A is 60% likely and model B
is 40% likely," the agent says "I know the truth is either A or B, but I
won't bet on which."

**Why this matters for decision-making**: A Bayesian agent that averages
over models can be exploited — an adversary who knows the agent's prior can
construct scenarios where the average is misleading. A KU agent reasons about
the **worst case** across its model set, making it robust to adversarial or
ambiguous environments.

**How it shows up in our code**: Currently, the `BeliefInfradistribution`
wraps a single `BeliefAMeasure` (one belief with λ=1, b=0). This is the
non-KU case — pure Bayesian learning. In the KU case, we'd have **multiple**
`BeliefAMeasure` objects, each wrapping its own independent belief with its
own (λ, b). The expected reward model becomes the **element-wise minimum**
across all measures — pessimistic/robust reasoning.

The tricky part: how to **update** these multiple measures when we see new
evidence. That's Definition 11.

---

### 7.2 The IB Update Rule (Definition 11) — Intuitive Explanation

#### The setup

We have an **infradistribution** H — a set of a-measures.

**Paper notation**: Each a-measure is a pair `(m, b)` where:
- `m` = λ · μ is a **scaled** probability measure (scale factor λ times a
  normalized distribution μ over outcomes)
- `b ≥ 0` is an offset

**Our code**: `BeliefAMeasure` stores these as three separate pieces:
- `belief` = the normalized distribution μ (via sufficient statistics,
  e.g., Beta(α, β) for BernoulliBelief)
- `log_scale` = log(λ), the scale factor (in log space for numerical
  stability)
- `offset` = b

The paper's `m` = `exp(log_scale) * belief` in our code. They're the same
object, just factored differently. The paper bundles λ and μ together; our
code keeps them separate because the belief updates μ independently.

An a-measure evaluates a **function f** as:

$$\alpha(f) = m(f) + b = \lambda \cdot \mu(f) + b$$

**What is f?** In the paper, f is any function from the outcome space to
[0,1] — it maps each possible outcome to a value. You can think of f as
"the thing you want to take the expected value of." The a-measure tells
you: "under my model of the world, the expected value of f is α(f)."

In our bandit setting, the f we care about most is the **reward function**:
"how much reward does each outcome give?" When we call
`BeliefAMeasure.expected_reward_model()`, we're evaluating α on this
specific f, getting E[reward | arm=a] for each arm.

Definition 11 also evaluates the a-measure on two trivial functions:
- f = 0 (the function that always returns 0) → α(0) = b
- f = 1 (the function that always returns 1) → α(1) = λ + b

These are used to compute normalization terms. The fact that Definition 11
only needs f=0, f=1, and f=reward is what makes the belief-based approach
tractable — we never need to evaluate arbitrary functions over the full
history space.

In code: `exp(log_scale) * belief.expected_reward_model() + offset`
(this is exactly what `BeliefAMeasure.expected_reward_model()` returns).

The **infradistribution** evaluates `f` by taking the **worst case**
across all its a-measures:

$$E_H(f) = \min_{(m,b) \in H} [m(f) + b] = \min_k [\lambda_k \cdot \mu_k(f) + b_k]$$

In code: `BeliefInfradistribution.expected_reward_model()` computes
`np.min(models, axis=0)` where each model comes from one `BeliefAMeasure`.

This is the key pessimism: the agent always asks "what's the least I can
expect?"

#### What happens when we see evidence?

We observe event `L` (e.g., "pulled arm 0, got reward 1"). We need to
update H to a new infradistribution H|L that:
1. Conditions on L (only consider worlds consistent with what we saw)
2. Preserves the KU structure (keeps multiple measures, adjusts their
   relative strengths)

#### The observation indicator L

L is an indicator function over the outcome space: L(x) = 1 if outcome x
is consistent with what we observed, L(x) = 0 otherwise.

**Is L crisp (0 or 1) or fuzzy?** In our bandit setting, **L is crisp**.
The observation is definite: "I pulled arm a and got reward r." That's a
hard fact — either an outcome matches it or it doesn't. The agent's mixed
strategy determines *which* arm gets pulled, but once the agent has acted
and observed the result, the observation is sharp.

(Fuzzy L would arise in settings with partial or noisy observations, e.g.,
"I observed something that's 70% likely to be reward=1." We don't have
that case.)

For BernoulliBelief: pulling arm a and getting reward r means:
- L = 1 for the outcome "arm a gives reward r"
- L = 0 for the outcome "arm a gives reward (1-r)"
- Outcomes for other arms are unaffected (we only update arm a's belief)

#### The glue operator ★

Given two functions `f` and `g` and an indicator `L`:

$$f \star_L g = L \cdot f + (1 - L) \cdot g$$

In words: "use `f` on the part of the world where L happened, and `g` on
the part where L didn't happen." This splices two functions together along
the boundary of the observation.

Since L is crisp in our case:
- On outcomes matching the observation: `f ★_L g = f`
- On outcomes NOT matching the observation: `f ★_L g = g`

**How the glue works in the belief-based approach**: We never construct
a glued function object (the deleted Phase 1 code did this via
`helpers.glue()`). Instead, we compute the **expectation of the glued
function** directly from belief parameters. For any function f and
constant g = c:

$$\mu(f \star_L c) = \mu(L) \cdot E_\mu[f|L] + \mu(1-L) \cdot c$$

In other words: the probability-weighted value on the observed branch,
plus the constant c times the probability of the non-observed branch.

Definition 11 only ever needs these specific glued expectations:
- `μ(0 ★_L c) = 0 · μ(L) + c · μ(1-L) = c · (1 - obs_prob)`
- `μ(1 ★_L c) = 1 · μ(L) + c · μ(1-L) = obs_prob + c · (1 - obs_prob)`

Both are simple functions of `obs_prob` (the observation probability from
the belief) and the constant c. No enumeration, no function objects — just
arithmetic on belief parameters. This is why the glue operator disappears
in the belief-based approach: its work is absorbed into the formulas.

#### The role of g (important — see §7.3 for full discussion)

`g` is a function that assigns value to **things that didn't happen**.
When you observe "arm 0, reward 1", the function `g` says how much you
"care about" the counterfactual outcomes (arm 0 reward 0, etc.).

In code: stored as `self.g` on `BeliefInfradistribution` (a constant,
default 1.0). See §7.3 for why g=1 is recommended.

#### The update formula

**Definition 11** says: to update infradistribution H after observing L
with counterfactual-value function g, transform each a-measure `(m, b)`
= `(λμ, b)` as:

$$(m, b) \mapsto \frac{1}{P^g_H(L)} \left( m \cdot L, \quad b + m((1-L) \cdot g) - E_H((1-L) \cdot g) \right)$$

Or equivalently, decomposing m = λμ:

$$(\lambda, \mu, b) \mapsto \frac{1}{P^g_H(L)} \left( \lambda \cdot \mu|_L, \quad b + \lambda \cdot \mu((1-L) \cdot g) - E_H((1-L) \cdot g) \right)$$

Let's break this down piece by piece.

**Term 1: `m · L` = restrict and rescale the measure.**

This does two things:
- **μ|_L**: Condition the normalized distribution μ on the observation L.
  This is standard Bayesian conditioning. In code: `belief.update(action,
  outcome)` — e.g., for BernoulliBelief, `α[a] += r`, `β[a] += (1-r)`.
- **λ rescales**: The new λ becomes `λ · μ(L)` — the old scale times the
  probability of the observation under this measure's distribution.
  `μ(L)` is what we need `belief.observation_probability()` for (§7.5.1).
  After normalization by P^g_H(L), the final scale is
  `λ · μ(L) / P^g_H(L)`.

In code (proposed): `m.log_scale = log(λ * obs_prob / prob)`.

**Term 2: `m((1-L) · g)` = `λ · μ((1-L) · g)`** — This measure's
"counterfactual value."

How much value does *this specific measure* assign to the things that
didn't happen, weighted by g? This is the integral of g over the
non-observed outcomes, under the scaled measure m = λμ.

Note: this is the measure integral **without** offset b. It uses λ (the
scale) but not b.

For constant g = c: this simplifies to `c · λ · μ(1-L)` =
`c · λ · (1 - obs_prob)`. Easily computable from belief parameters.

In code (proposed): `c * lambda_k * (1 - obs_prob_k)`.

**Term 3: `E_H((1-L) · g)`** — The worst-case counterfactual value.

Same type of quantity as Term 2, but evaluated as an **infradistribution**
— taking the infimum over ALL a-measures in H. Since this is E_H, it uses
the full a-measure evaluation (measure integral + offset):

$$E_H((1-L) \cdot g) = \min_k [\lambda_k \cdot \mu_k((1-L) \cdot g) + b_k]$$

For constant g = c: `min_k [c · λ_k · (1 - obs_prob_k) + b_k]`.

In code (proposed): `global_cfval = min(cfvals)` where
`cfvals[k] = c * lambdas[k] * not_obs_probs[k] + offsets[k]`.

**The offset update: `b_new = (b + Term2 - Term3) / P^g_H(L)`**

The new offset absorbs the difference between this measure's counterfactual
value and the global worst case, then normalizes.

- `Term2 - Term3` ≥ 0 always (since Term3 is the min over all measures,
  and Term2 is one specific measure's value — but note Term2 doesn't
  include b while Term3 does, so this isn't quite the min of Term2).
  Actually: `b + Term2` is this measure's full counterfactual evaluation,
  and Term3 is the min of all measures' full evaluations. So
  `b + Term2 - Term3` ≥ 0. ✓

- Measures that assign MORE value to the non-observed branch get a LARGER
  offset. Intuitively: they're "richer" — they had more value outside the
  observation, and that surplus gets converted into offset.

In code (proposed):
`m.offset = (offsets[k] + c * lambdas[k] * not_obs_probs[k] - global_cfval) / prob`

**The normalization: `P^g_H(L)`** — The "probability" of the observation.

Defined as:

$$P^g_H(L) = E_H(1 \star_L g) - E_H(0 \star_L g)$$

Expanding:
- `1 ★_L g = L + (1-L)·g` → "1 on observed, g on non-observed"
- `0 ★_L g = (1-L)·g` → "0 on observed, g on non-observed"
- Difference = the infradistribution's "probability" of L, adjusted for g

For constant g = c:
- `E_H(1 ★_L g) = min_k [λ_k · obs_prob_k + c · λ_k · (1-obs_prob_k) + b_k]`
- `E_H(0 ★_L g) = min_k [c · λ_k · (1-obs_prob_k) + b_k]` (= Term 3)
- `P^g_H(L) = min_k[...full...] - min_k[...counterfactual...]`

In code (proposed): `prob = global_full - global_cfval`.

This normalizes the updated infradistribution so it remains valid (i.e.,
evaluates the constant function 1 at value 1).

---

### 7.3 What is g? (Counterfactual value / off-history utility)

This is the most confusing part of the theory. After reviewing the full
infrabayesian literature (Diffractor & Vanessa Kosoy's sequence on
LessWrong/AlignmentForum), here is our understanding.

#### g is the off-history utility function

The notebook confusingly calls g `reward_function`. The paper calls it
the function specifying "how things go outside of L." What it actually
is: **g is the component of the agent's utility function that applies to
outcomes outside the observed event.**

When you observe event L (e.g., "arm 0 gave reward 1"), the universe of
outcomes splits in two:
- **Inside L**: what actually happened (arm 0, reward 1)
- **Outside L**: what didn't happen (arm 0, reward 0)

The glue operator `f ★_L g` says: evaluate `f` on the observed branch,
evaluate `g` on the non-observed branch. The function g encodes **how much
utility the agent assigns to the counterfactual branch**.

#### g is NOT a free hyperparameter

This is a key finding from deeper in the sequence. In "Belief Functions
and Decision Theory" (post 3 of the infrabayesian sequence), the full
update takes the form:

$$(Θ | π_{\text{not-h}}, g, h)$$

where:
- `h` = the observed history
- `π_not-h` = the agent's policy for everything *except* what happens
  after history h (the counterfactual policy)
- `g` = the utility function for outcomes after h

**g is determined by the problem structure** — specifically, it falls out
of decomposing the agent's full utility function U relative to the history
partition. The agent does not get to freely pick g.

As Vanessa Kosoy explained (AXRP Episode 5): *"The update rule actually
depends on your utility function and your policy in the counterfactuals.
The way you should update if a certain event happens depends on what you
would do if this event did not happen."*

#### Why g exists: dynamic consistency

In standard Bayesian conditioning, you observe evidence and throw away
everything inconsistent with it. You don't need g because you simply don't
care about what didn't happen.

But infradistributions are **not** probability distributions. The infimum
operation makes expected values nonlinear. When you condition (restrict to
L), you're throwing away parts of the measure that were contributing to the
infimum. Without compensation, this causes **dynamic inconsistency**: the
agent's preferences change after observing evidence, even though no new
information was gained about the relevant future.

The classic example is counterfactual mugging: before a coin flip, you'd
agree to a deal covering both outcomes. But after seeing heads, standard
conditioning makes you "forget" your commitment to the tails branch. The
offset term `b` in a-measures tracks the counterfactual utility — what
you committed to on the other branch — preventing this inconsistency.

The theory proves (Theorem 5 in "Belief Functions and Decision Theory")
that this update rule IS dynamically consistent: if the agent would prefer
policy π* over π after updating on h, then from the original perspective,
switching to π* post-h is also preferred.

#### g=0 vs g=1: the two canonical choices

"Less Basic Inframeasure Theory" compares these directly:

- **g = 0**: Assigns zero utility to non-observed outcomes. The adversary
  (Murphy) selects distributions with the *lowest* probability of the
  observed event. Closest to naive element-wise Bayesian conditioning.

- **g = 1**: Assigns unit utility to non-observed outcomes. The adversary
  selects distributions with the *highest* probability of the observed
  event. Diffractor describes this as **"somewhat more sensible behavior"**
  because it conditions on distributions most likely to have generated
  the observation — more aligned with standard epistemic rationality.
  Additionally, **g=1 preserves cohomogeneity** (the structural property
  where all minimal points satisfy λ + b = 1), which supports better
  sequential/dynamic properties.

The notebook demonstrates both (plus an arbitrary g):

| g used for update | P(HH) after seeing H | Character |
|---|---|---|
| g = 0 (zero) | 0.300 | KU collapsed — offsets stay 0, probs flat |
| g = 1 (one) | 0.643 | Non-trivial KU, large offsets, sensible |
| g = arbitrary | 0.647 | Non-trivial KU, intermediate offsets |

With g=0, the KU update degenerates — the measures' scales (λ) differ but
their offsets are all zero, so the probabilities are flat. The KU structure
is preserved in λ but doesn't affect predictions.

With g=1, the offsets become non-zero and the measures genuinely diverge in
their predictions. This is where KU adds value.

#### Is g always a constant? Does it change over time?

In the full theory, **no, g is not always a constant**. The "correct" g
is derived from the agent's utility function and counterfactual policy,
which can change as the agent learns. At step t, g should technically be:
"what utility would the agent get on the counterfactual branch, given its
current policy and remaining horizon?"

For example, in a discounted-reward bandit:
- Early in learning, the agent is uncertain → g might be ~0.5 (uncertain
  about counterfactual value)
- Late in learning, the agent has converged → g might be close to the
  actual arm probability

However, **g = 1 (constant) is a well-motivated simplification**:
- The literature explicitly recommends g=1 over other constant choices
- It's the "optimistic about counterfactuals" choice, which Diffractor
  calls "more sensible"
- It preserves cohomogeneity (λ + b = 1), a structural invariant
- It's simple to implement and reason about

**Our implementation will NOT require g to be constant forever.** The
interface accepts g as a parameter on `BeliefInfradistribution`. For
Phase 4, we use g=1 (constant). A future refinement could make g a
callable `g(step, belief_state) -> float` that adapts over time, or
derive it from the agent's value estimates. But constant g=1 exercises
all the KU machinery and is the right starting point.

#### What g should be for bandits — summary

| Option | Pros | Cons |
|---|---|---|
| g = 0 | Simplest | KU collapses, no benefit over Bayesian |
| g = 1 (recommended) | Literature-backed, preserves cohomogeneity, non-trivial KU | Not "correct" in full theory |
| g = adaptive | Most theoretically correct | Complex, unclear how to derive for beliefs |

**Practical recommendation for Phase 4**: Start with **g = 1** as the
default. Make g a parameter so we can later experiment with adaptive g,
but g=1 is a well-motivated starting point that exercises the full KU
machinery.

#### Key references

- [Basic Inframeasure Theory](https://www.alignmentforum.org/posts/YAa4qcMyoucRS2Ykr/) — Definitions 9-11
- [Belief Functions and Decision Theory](https://www.greaterwrong.com/posts/e8qFDMzs2u9xf5ie6/) — Full update rule, dynamic consistency proof (Theorem 5), g determined by problem structure
- [Less Basic Inframeasure Theory](https://www.greaterwrong.com/posts/idP5E5XhJGh9T5Yq9/) — g=0 vs g=1 comparison, g=1 recommended
- [AXRP Episode 5](https://axrp.net/episode/2021/03/10/episode-5-infra-bayesianism-vanessa-kosoy.html) — Kosoy on update depending on utility and counterfactual policy
- [Elementary Infra-Bayesianism (Kirchner)](https://universalprior.substack.com/p/elementary-infra-bayesianism) — Accessible g=0 vs g=1 examples

---

### 7.4 The math for BernoulliBelief (concrete formulas)

Here's the good news: **the KU update IS computable from belief parameters
for BernoulliBelief**. No history enumeration needed.

#### Setup

We have K a-measures. Each a-measure k consists of:
- A `BernoulliBelief` with parameters `(α_k[a], β_k[a])` for each arm `a`
- A scale factor `λ_k`
- An offset `b_k`

At each step, the agent pulls arm `a` and observes reward `r ∈ {0, 1}`.

#### Why beliefs are sufficient

At each step, the observation is a **single binary outcome** on one arm.
The "outcome space" for the observation is just {0, 1} for that arm.
The belief's sufficient statistics (α, β) fully determine the probability
of any single-step outcome. So all the integrals in Definition 11 reduce
to simple functions of (α, β, λ, b).

This is the key insight that makes belief-based KU tractable: we update
**one step at a time**, and each step's outcome space is tiny (binary for
Bernoulli). We never need to enumerate multi-step histories.

#### Step-by-step formulas

**Observation**: pulled arm `a`, got reward `r ∈ {0, 1}`.

**Choose**: g = constant `c` (recommend c=1 per §7.3; we'll generalize later).

For each a-measure k, let:

$$p_k = \frac{\alpha_k[a]}{\alpha_k[a] + \beta_k[a]}$$

This is measure k's predicted probability of reward=1 on arm a.

**Probability of the observation under measure k:**

$$P_k(\text{obs}) = p_k^r \cdot (1 - p_k)^{1-r}$$

(If r=1, this is p_k. If r=0, this is 1 - p_k.)

**Counterfactual value for measure k** (Term 2 from §7.2):

$$\text{cfval}_k = c \cdot \lambda_k \cdot P_k(\text{not obs}) + b_k$$

where `P_k(not obs) = 1 - P_k(obs)`. This is `m_k((1-L)·g) + b_k` —
the full a-measure evaluation (including offset) of g on the non-observed
outcome.

Wait — let me be more careful. `m_k((1-L)·g)` is the measure integral
*without* offset. So:

$$m_k((1-L) \cdot g) = c \cdot \lambda_k \cdot P_k(\text{not obs})$$

**Global worst-case counterfactual value** (Term 3 from §7.2):

$$E_H((1-L) \cdot g) = \min_k \left[ c \cdot \lambda_k \cdot P_k(\text{not obs}) + b_k \right]$$

Note: this IS `min_k [m_k((1-L)·g) + b_k]` — the infimum uses the full
a-measure evaluation (measure integral + offset).

**Normalization term:**

$$E_H(1 \star_L g) = \min_k \left[ \lambda_k \cdot P_k(\text{obs}) + c \cdot \lambda_k \cdot P_k(\text{not obs}) + b_k \right]$$

$$P^g_H(L) = E_H(1 \star_L g) - E_H(0 \star_L g)$$

where `E_H(0 ★_L g) = E_H((1-L)·g)` (same as global worst-case above).

**The update for each a-measure k:**

1. **Belief update** (standard Bayesian):
   - `α_k[a] += r`
   - `β_k[a] += (1 - r)`
   - (Other arms unchanged)

2. **Scale update**:
   $$\lambda_k^{\text{new}} = \frac{\lambda_k \cdot P_k(\text{obs})}{P^g_H(L)}$$

3. **Offset update**:
   $$b_k^{\text{new}} = \frac{b_k + m_k((1-L) \cdot g) - E_H((1-L) \cdot g)}{P^g_H(L)}$$

   Substituting:
   $$b_k^{\text{new}} = \frac{b_k + c \cdot \lambda_k \cdot P_k(\text{not obs}) - E_H((1-L) \cdot g)}{P^g_H(L)}$$

#### Sanity check: non-KU case

With a single a-measure (K=1), λ=1, b=0, and g=c:

- `E_H((1-L)·g)` = `c · 1 · P_1(not obs) + 0` = `c · P_1(not obs)`
- `m_1((1-L)·g)` = `c · 1 · P_1(not obs)` = same
- Offset: `(0 + c·P(not obs) - c·P(not obs)) / P^g_H(L)` = **0** ✓
- Scale: `P(obs) / P^g_H(L)`

And `P^g_H(L) = min[P(obs) + c·P(not obs)] - min[c·P(not obs)]`
= `P(obs) + c·P(not obs) - c·P(not obs)` = `P(obs)`

So scale = `P(obs) / P(obs)` = **1** ✓

Non-KU: offset stays 0, scale stays 1, only the belief updates. Pure
Bayesian conditioning. Exactly what we already have. ✓

#### Sanity check: what g=0 does

With g=0 (c=0):
- All counterfactual values are just b_k (the offsets themselves)
- `E_H((1-L)·0) = min_k [b_k]`
- Offset: `(b_k + 0 - min_k[b_k]) / P^0_H(L)`
- If initially all b_k = 0: offset stays 0, and we get independent
  Bayesian conditioning per measure. KU is preserved only in λ.

This matches the notebook: g=0 gives "KU collapsed" behavior where
probabilities are flat (0.300 for all reward functions).

---

### 7.5 Implementation Plan for BernoulliBelief KU

#### 7.5.1 New method on BaseBelief: `observation_probability()`

The KU update needs to know **how likely this observation was** under each
belief, computed BEFORE the belief updates (since updating changes the
parameters). This is a new abstract method on `BaseBelief`:

```python
class BaseBelief(ABC):
    # ... existing methods (update, expected_reward_model, copy) ...

    @abstractmethod
    def observation_probability(self, action, outcome) -> float:
        """P(this observation) under the current belief.

        Must be called BEFORE update(), since update() changes the
        sufficient statistics that this method reads.

        Returns a probability in [0, 1].
        """
        pass
```

Implementations:

```python
# BernoulliBelief: P(reward=r | arm=a) = p^r * (1-p)^(1-r)
def observation_probability(self, action, outcome):
    p = self.alpha[action] / (self.alpha[action] + self.beta[action])
    r = outcome.reward
    if not (0.0 <= r <= 1.0):
        raise ValueError(
            f"BernoulliBelief expects reward in [0, 1], got {r}")
    prob = p ** r * (1 - p) ** (1 - r)
    assert 0.0 <= prob <= 1.0  # guaranteed by math, but verify
    return prob

# GaussianBelief: normal likelihood (may need numerical care)
def observation_probability(self, action, outcome):
    # P(r | μ, σ) — density of observed reward under current estimate
    # Details TBD; needs assumed noise variance
    # Note: for continuous distributions this is a density, not a
    # probability, so it can exceed 1. May need to cap or normalize.
    ...

# NewcombLikeBelief: deterministic — P = 1 if matches observed cell, else 0
def observation_probability(self, action, outcome):
    expected = self.observed[outcome.env_action, action]
    if np.isnan(expected):
        return 1.0  # unobserved cell — any outcome is "expected"
    return 1.0 if outcome.reward == expected else 0.0
```

Note: `BeliefAMeasure` does NOT need its own `observation_probability()`
method. The infradistribution's `_snapshot_measures()` calls
`m.belief.observation_probability()` directly and combines it with
`exp(m.log_scale)` as needed (see §7.5.2).

#### 7.5.2 Changes to BeliefInfradistribution

The `update()` method gains the KU logic. For a single measure (non-KU),
behavior is unchanged. For multiple measures (KU), it applies Definition 11.

The update is broken into private methods with descriptive names so the
code reads as a narrative of Definition 11:

```python
from dataclasses import dataclass, field

@dataclass
class _MeasureSnapshot:
    """Pre-update snapshot of one a-measure's state.

    Captured BEFORE beliefs update, because the KU adjustment
    needs the prior probabilities, not the posterior ones.
    """
    obs_prob: float       # μ_k(L)     — P(observation) under this belief
    scale: float          # λ_k        — exp(log_scale)
    offset: float         # b_k        — current offset
    not_obs_prob: float = field(init=False)  # μ_k(1-L) — derived

    def __post_init__(self):
        if not (0.0 <= self.obs_prob <= 1.0):
            raise ValueError(
                f"obs_prob must be in [0, 1], got {self.obs_prob}")
        if self.scale <= 0:
            raise ValueError(
                f"scale (λ) must be > 0, got {self.scale}")
        if self.offset < 0:
            raise ValueError(
                f"offset (b) must be ≥ 0, got {self.offset}")
        self.not_obs_prob = 1.0 - self.obs_prob


class BeliefInfradistribution:
    def __init__(self, measures, g=1.0):
        if len(measures) == 0:
            raise ValueError("Must provide at least one measure")
        if not (0.0 <= g <= 1.0):
            raise ValueError(
                f"g must be in [0, 1], got {g}")
        self.measures = measures
        self.g = g  # counterfactual value constant (default 1.0, see §7.3)

    # ── Public interface ──────────────────────────────────────────────

    def update(self, action, outcome, context=None):
        if len(self.measures) == 1:
            # Optimization: skip KU arithmetic when there's only one measure.
            # The full KU path produces identical results (see §7.4 sanity
            # check), but this avoids unnecessary computation.
            self.measures[0].update(action, outcome, context)
            return

        # KU update (Definition 11, §7.2)
        snapshots = self._snapshot_measures(action, outcome)
        normalization = self._observation_probability(snapshots)
        self._apply_ku_update(snapshots, normalization, action, outcome, context)

    # ── Private: Definition 11 steps ──────────────────────────────────

    def _snapshot_measures(self, action, outcome):
        """Capture each measure's state BEFORE updating beliefs.

        We need the prior observation probabilities (before Bayesian update)
        to compute the KU adjustment terms.
        """
        snapshots = []
        for m in self.measures:
            snapshots.append(_MeasureSnapshot(
                obs_prob=m.belief.observation_probability(action, outcome),
                scale=np.exp(m.log_scale),
                offset=m.offset,
            ))
        return snapshots

    def _counterfactual_value(self, snap):
        """α_k((1-L) · g) — How much value does a-measure k assign to the
        non-observed outcome, weighted by g?

        This is the full a-measure evaluation: λ_k · μ_k((1-L)·g) + b_k.
        For constant g = c: c · λ_k · P_k(not obs) + b_k.

        Corresponds to Term 2 + offset in §7.2.
        """
        return self.g * snap.scale * snap.not_obs_prob + snap.offset

    def _full_observation_value(self, snap):
        """α_k(1 ★_L g) — The a-measure's value of "1 on observed branch,
        g on non-observed branch."

        = λ_k · P_k(obs) + c · λ_k · P_k(not obs) + b_k

        Used to compute P^g_H(L) (the normalization denominator).
        """
        return (snap.scale * snap.obs_prob
                + self.g * snap.scale * snap.not_obs_prob
                + snap.offset)

    def _observation_probability(self, snapshots):
        """P^g_H(L) — The infradistribution's "probability" of the
        observation, adjusted for g.

        = E_H(1 ★_L g) - E_H(0 ★_L g)
        = min_k[full_value_k] - min_k[counterfactual_value_k]

        This is the normalization factor for the entire update.
        """
        worst_case_full = min(self._full_observation_value(s) for s in snapshots)
        worst_case_counterfactual = min(self._counterfactual_value(s) for s in snapshots)
        prob = worst_case_full - worst_case_counterfactual
        if prob <= 0:
            raise ValueError(
                f"P^g_H(L) must be > 0 (observation has zero probability "
                f"under worst-case measure), got {prob}")
        return prob

    def _apply_ku_update(self, snapshots, normalization, action, outcome, context):
        """Apply Definition 11 to each a-measure.

        For each measure k:
          1. Bayesian update of belief (μ_k conditions on observation)
          2. Scale update: λ_new = λ · P_k(obs) / P^g_H(L)
          3. Offset update: absorb counterfactual value surplus, normalize
        """
        worst_case_counterfactual = min(
            self._counterfactual_value(s) for s in snapshots
        )

        for snap, m in zip(snapshots, self.measures):
            # (1) Bayesian update — belief conditions on observation
            m.belief.update(action, outcome, context)

            # (2) Scale update — rescale by P_k(obs), normalize
            #     λ_new = λ · P_k(obs) / P^g_H(L)
            m.log_scale = np.log(snap.scale * snap.obs_prob / normalization)

            # (3) Offset update — absorb counterfactual surplus, normalize
            #     b_new = (b + λ·μ((1-L)·g) - E_H((1-L)·g)) / P^g_H(L)
            #
            #     The numerator is: this measure's counterfactual value
            #     minus the global worst-case counterfactual value.
            #     Always ≥ 0 (since worst_case is the min).
            counterfactual_surplus = (
                self._counterfactual_value(snap) - worst_case_counterfactual
            )
            if counterfactual_surplus < -1e-12:
                raise ValueError(
                    f"counterfactual_surplus must be ≥ 0, got "
                    f"{counterfactual_surplus}")
            m.offset = max(0.0, counterfactual_surplus) / normalization
```

Note on the method structure:
- `_snapshot_measures`: captures pre-update state (must happen before
  beliefs update, since observation_probability uses prior parameters)
- `_counterfactual_value`: per-measure Term 2+b (§7.2) — "how much does
  this measure value the non-observed branch?"
- `_full_observation_value`: per-measure "1 on observed, g on non-observed"
- `_observation_probability`: the global normalization P^g_H(L)
- `_apply_ku_update`: ties it all together, updates each measure's
  (belief, λ, b)

#### 7.5.3 Reward-to-utility mapping on the agent

The IB theory requires all functions (f, g) to map to [0, 1]. This means
rewards must be bounded in [0, 1] before entering the IB pipeline. But
environments can produce arbitrary rewards (negative, > 1, unbounded
Gaussians, etc.).

**Solution**: The `InfraBayesianAgent` applies a **utility mapping** that
transforms raw environment rewards to [0, 1] before passing them to the
infradistribution. This is the right place for this because:
- The belief models the world in utility-space, not raw-reward-space
- The a-measure's (λ, b) arithmetic assumes [0, 1] bounds
- The agent owns the "how I value rewards" decision, separate from the
  belief's "how I model the world" logic
- This gives g a concrete meaning: g = 1 means "I assign maximum utility
  to the counterfactual branch"

```python
class InfraBayesianAgent(BaseGreedyAgent):
    def __init__(self, *args, utility=None, **kwargs):
        # ...
        # utility: callable mapping raw reward -> [0, 1]
        # Default: clamp to [0, 1] (identity for Bernoulli rewards)
        self._utility = utility or (lambda r: np.clip(r, 0.0, 1.0))

    def update(self, probabilities, action, outcome):
        # Map reward to utility BEFORE passing to infradistribution
        mapped_reward = self._utility(outcome.reward)
        if not (0.0 <= mapped_reward <= 1.0):
            raise ValueError(
                f"Utility mapping must produce values in [0, 1], "
                f"got {mapped_reward} from reward {outcome.reward}")
        mapped_outcome = Outcome(
            reward=mapped_reward,
            env_action=outcome.env_action,
        )
        context = {'step': self.step, 'policy': probabilities}
        super().update(probabilities, action, outcome)  # base agent sees raw reward
        self.infradist.update(action, mapped_outcome, context)  # IB sees utility
```

Built-in utility mappings:
- `"clamp"`: `lambda r: np.clip(r, 0, 1)` — clamp (default; identity
  for Bernoulli)
- `"sigmoid"`: `lambda r: 1 / (1 + np.exp(-r))` — for unbounded rewards
- Affine rescale (`lambda r: (r - r_min) / (r_max - r_min)`) can be
  passed directly when reward bounds are known; not registered as a named
  type since it requires parameters.

**Registration in `construction.py`**: Named utility functions are
registered alongside belief types, using the same pattern:

```python
utility_types = {
    "clamp":   lambda: lambda r: np.clip(r, 0.0, 1.0),
    "sigmoid": lambda: lambda r: 1.0 / (1.0 + np.exp(-r)),
}

# In the infrabayesian block of construct_agent():
if name == "infrabayesian" and "utility" in arguments:
    utility_name = arguments.pop("utility")
    if isinstance(utility_name, float):
        utility_name = str(int(utility_name))
    if utility_name not in utility_types:
        raise RuntimeError("Invalid utility type: " + str(utility_name))
    arguments["utility"] = utility_types[utility_name]()

# Usage: "infrabayesian:belief=bernoulli,utility=sigmoid"
# Default (no utility kwarg): clamp
```

Note: the `expected_reward_model()` will now return values in utility space
[0, 1], not raw reward space. The planning step (`build_greedy_policy`)
only cares about relative ordering, so this is fine — the best arm in
utility space is the best arm in reward space (for monotone mappings).

Note: for the non-KU case, the utility mapping is irrelevant — λ=1, b=0,
and the belief's Bayesian update produces the same relative ordering
regardless of affine transforms. But for KU, the mapping matters because
the offset arithmetic assumes [0, 1] bounds.

#### 7.5.4 Changes to InfraBayesianAgent

The agent accepts `beliefs` — a list of belief templates. A single-element
list gives non-KU behavior (equivalent to standard Bayesian update);
multiple elements give KU. No separate code paths.

```python
class InfraBayesianAgent(BaseGreedyAgent):
    def __init__(self, *args, beliefs, g=1.0, utility=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(beliefs, list) or len(beliefs) == 0:
            raise ValueError(
                "beliefs must be a non-empty list of BaseBelief")
        self._belief_templates = beliefs  # list of BaseBelief
        self._g = g
        self._utility = utility or (lambda r: np.clip(r, 0.0, 1.0))

    def reset(self):
        super().reset()
        measures = [BeliefAMeasure(b.copy()) for b in self._belief_templates]
        self.infradist = BeliefInfradistribution(measures, g=self._g)
```

The `BeliefInfradistribution.update()` already handles both cases
gracefully: with one measure, the KU offset/scale adjustments are no-ops
(see §7.4 sanity check), so the behavior is identical to the current
non-KU code path. We still keep the `len(self.measures) == 1` early
return in `update()` as a **performance optimization** (skips the
snapshot/normalization arithmetic), not as a separate code path.

#### 7.5.5 What "multiple beliefs" means for bandits

In the notebook's coin flip, the three a-measures correspond to three
hypotheses: p=0.3, p=0.5, p=0.7. Each hypothesis defines a probability
distribution over outcomes, and KU means "I don't know which is right."

For BernoulliBelief, "multiple beliefs" means multiple **priors**:
- Belief A: "I think arm 0 has low success rate" → Beta(1, 3) for arm 0
- Belief B: "I think arm 0 has medium rate" → Beta(2, 2)
- Belief C: "I think arm 0 has high rate" → Beta(3, 1)

Each belief updates independently via Bayesian conditioning. The KU
structure is in the (λ, b) adjustments, which change how much weight
each belief gets in the pessimistic evaluation.

This is analogous to the notebook's setup, but using sufficient statistics
instead of explicit history vectors.

#### 7.5.6 Tests

1. **Non-KU unchanged**: Single-measure KU update produces identical
   results to current non-KU update. Verify λ=1, b=0 throughout.

2. **KU offset adjustment**: Two BernoulliBelief measures with different
   priors. After update, verify offsets match hand-computed values from
   §7.4 formulas.

3. **KU scale adjustment**: Same setup. Verify λ values match formulas.

4. **g=0 degeneracy**: With g=0, verify offsets remain 0 and probabilities
   are flat (matching notebook's "Zero" row).

5. **g=constant non-trivial**: With g=0.5, verify offsets become non-zero
   and measures diverge meaningfully.

#### 7.5.7 Open questions

1. **How to choose initial beliefs for KU?** The notebook uses three
   point hypotheses (p=0.3, 0.5, 0.7). For BernoulliBelief, we'd use
   different Beta priors. What's a good default set of priors?

2. **Refining g beyond constant**: g=1 is a good starting point (see §7.3),
   but the full theory says g should be derived from the agent's utility
   function. For cumulative-reward bandits, what does this reduce to?

3. **Does the vertex set need to grow?** The paper mentions that downward
   closure may introduce new extremal points. For now, track only the
   initial vertices (an approximation). Monitor whether this causes issues.

4. **Generalizing beyond BernoulliBelief**: The update formulas above
   assume we can compute `observation_probability()` from the belief.
   GaussianBelief can do this (normal likelihood). NewcombLikeBelief has
   deterministic observations (probability 0 or 1). Need to verify the
   formulas are numerically stable for these cases.

5. **Integration with `construction.py` for KU**: How should KU be
   specified in the construction string? E.g.,
   `"infrabayesian:belief=bernoulli,ku=3"` to create 3 KU vertices?
   Need to decide how to specify multiple beliefs with different priors.

6. **Fuzzy observations (future)**: In all our current environments, the
   observation indicator L is crisp (0 or 1) — the agent always observes a
   definite (action, reward) pair. Stochastic rewards and partial feedback
   (not observing other arms) do NOT make L fuzzy; the reward is random
   before you observe it, but the observation itself is definite.

   Fuzzy L would matter for settings where the **observation itself is
   uncertain** — where you're not sure what you actually saw. Examples:
   - **Noisy sensor**: pull an arm, sensor reports "probably reward 1, but
     20% chance of misread" → L(reward=1) = 0.8, L(reward=0) = 0.2
   - **Continuous outcomes with soft likelihood**: reward is a real-valued
     measurement and L encodes "how consistent is this reading with each
     hypothesis?" as a soft weight rather than a hard match
   - **Aggregated data**: observe a batch statistic ("3 of 5 pulls gave
     reward 1") rather than individual outcomes

   Supporting fuzzy L could be important for (a) demonstrating IB's
   theoretical advantages in settings where standard Bayesian conditioning
   breaks down, and (b) expanding to more real-world use cases where
   observations genuinely are noisy. The math in §7.2 and §7.4 already
   handles fuzzy L in principle (the glue operator is defined for
   L ∈ [0,1], not just {0,1}), but our `observation_probability()`
   interface currently assumes crisp observations.

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
