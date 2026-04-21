# Refactor Plan: Generic `WorldModel` for `Infradistribution`

**Branch**: `alaro/beliefs` (off `fllor/ib-agent-architecture`)
**Date**: 2026-04-20 (supersedes 2026-04-18 draft)

---

## Motivation

The current `Infradistribution` has two hardcoded assumptions that limit generality:

1. **Count-based history**: `self.history = np.zeros(num_outcomes)` is the sufficient statistic for IID Bernoulli outcomes only. It won't generalize to Gaussian bandits, Newcomb, or any model where observations aren't exchangeable.

2. **Likelihood computation in `AMeasure`**: The posterior predictive computation is baked into `AMeasure._compute_probabilities`. This is model-family logic — a Gaussian or Newcomb model would compute it differently. It belongs in `WorldModel`, not in the a-measure, which should be pure data.

A third issue not in the original draft: `AMeasure`'s hypothesis parameters are named `log_probabilities` and `coefficients` — these are categorical-specific names. For Gaussian or Newcomb models, hypothesis parameters have different structure (kernel parameters, reward matrices). Making `AMeasure` generic over its params type `P` keeps the interface honest.

The fix is a `WorldModel` abstraction that owns the belief state type, update logic, and likelihood/expected reward computation for a given model family. `AMeasure[P]` is reduced to pure data: `(params: P, scale, offset)`.

---

### Clarifying three distinct arrays

A common source of confusion: AMeasure parameters, the WorldModel belief state, and the agent's prior over hypotheses all mean completely different things:

| Array | Where | What it means |
|---|---|---|
| `AMeasure.params` | AMeasure | Hypothesis parameters: defines the probability measure μ for this world |
| `Infradistribution.belief_state` | Infradistribution | Evidence: sufficient statistic of observations so far |
| `prior` | Agent / `Infradistribution.mix` | Prior over hypotheses: weight assigned to each hypothesis |

These interact only in `WorldModel.compute_likelihood(belief_state, outcome, params, ...)` — which combines evidence with hypothesis parameters to produce a likelihood scalar.

### Note on reward bounds

The `0` and `1` in `_glue(0, ...)` and `_glue(1, ...)` are the extremal values required by the theory's normalization conditions (E_H(**0**) = 0, E_H(**1**) = 1) — not reward min/max. Definition 11 requires the reward function to live in [0, 1]. If an environment produces rewards outside that range, the agent normalizes before passing to the infradistribution. The `0` and `1` stay as-is.

---

## Proposed Design

### `AMeasure`

`AMeasure` is a plain container: hypothesis params plus the scale λ and offset b from the (λμ, b) triplet. It has no knowledge of what params means — that belongs entirely to `WorldModel`.

```python
# ibrl/infrabayesian/a_measure.py

import numpy as np


class AMeasure:
    """
    An a-measure (λμ, b): scale λ > 0, probability measure μ encoded as
    hypothesis params (opaque — WorldModel defines the structure), offset b ≥ 0.
    Stateless with respect to observations — belief state lives in Infradistribution.
    """
    def __init__(self, params, scale: float = 1.0, offset: float = 0.0):
        assert offset >= 0
        assert scale > 0
        self.params = params
        self.scale = np.float64(scale)
        self.offset = np.float64(offset)

    def __itruediv__(self, other: float):
        self.scale /= other
        self.offset /= other
        return self

    def reset(self):
        self.scale = np.float64(1)
        self.offset = np.float64(0)   # was np.float64(1) — bug fix

    def evaluate_action(self, world_model, belief_state, reward_function,
                       action: int, policy=None) -> float:
        raw = world_model.compute_expected_reward(
            belief_state, reward_function, self.params, action=action, policy=policy
        )
        return self.scale * raw + self.offset

    def evaluate_policy(self, world_model, belief_state,
                                  reward_function, policy) -> float:
        """V(π) = Σ_a π(a) × E[reward | a, π] for this a-measure.
        reward_function: shape (num_actions, num_outcomes).
        """
        raw = sum(
            policy[a] * world_model.compute_expected_reward(
                belief_state, reward_function[a], self.params, action=a, policy=policy
            )
            for a in range(len(policy))
        )
        return self.scale * raw + self.offset
```

`action` and `policy` default to `None` and are ignored by world models that don't need them (Bernoulli). They are used by `NewcombWorldModel` and, in the future, Fund 4 state-space models.

Params are constructed via `WorldModel.make_params(...)`, not via factories on `AMeasure`. This keeps all model-family knowledge in `WorldModel`.

---

### `WorldModel[S]`

`WorldModel` owns everything model-family-specific: the belief state type, how to update it, how to compute likelihoods and expected rewards, and how to construct hypothesis params. `AMeasure` treats params as opaque.

```python
# ibrl/infrabayesian/world_model.py

from abc import ABC, abstractmethod
import numpy as np
from ..outcome import Outcome

class WorldModel(ABC):
    """
    Defines the belief state type, update logic, likelihood computation,
    and hypothesis param construction for a model family. Stateless —
    belief state is owned by Infradistribution.

    Belief state and params are opaque to callers — structure is defined
    by each subclass and documented there.
    """

    @abstractmethod
    def make_params(self, *args, **kwargs):
        """Construct hypothesis params for one a-measure."""
        pass

    @abstractmethod
    def mix_params(self, params_list: list, coefficients: np.ndarray):
        """
        Combine params from multiple a-measures into one mixed a-measure.
        Called by Infradistribution.mix. Each element of params_list is the
        params of one input a-measure; coefficients are the mixing weights.
        """
        pass

    @abstractmethod
    def event_index(self, outcome: Outcome) -> int:
        """
        Extract the discrete event index from an outcome.
        Used by the gluing operator and belief state update.
        """
        pass

    @abstractmethod
    def initial_state(self):
        """Return the initial belief state (no observations)."""
        pass

    @abstractmethod
    def update_state(self, state, outcome: Outcome, action: int):
        """
        Return new belief state after observing outcome under agent action.
        Does not mutate state.
        """
        pass

    @abstractmethod
    def is_initial(self, state) -> bool:
        """True if state is the initial (no observations) state."""
        pass

    @abstractmethod
    def compute_likelihood(self, belief_state, outcome: Outcome, params,
                           action: int,
                           policy: np.ndarray | None = None) -> float:
        """
        P(outcome | belief_state, params, action) under this hypothesis.
        Returns a scalar in [0, 1].
        """
        pass

    @abstractmethod
    def compute_expected_reward(self, belief_state, reward_function: np.ndarray,
                                params, action: int,
                                policy: np.ndarray | None = None) -> float:
        """
        E[reward_function(outcome) | belief_state, params, action].
        Returns a scalar.
        """
        pass
```

---

### `MultiBernoulliWorldModel`

Multi-arm Bernoulli bandit. Params is a list of K per-arm hypothesis grids — each arm has its own mixture of components. `compute_likelihood` and `compute_expected_reward` index into `params[action]`, so only the taken arm's components are used. Arms are fully independent: observing arm 0 only updates arm 0's posterior predictive.

```python
class MultiBernoulliWorldModel(WorldModel):
    """
    World model for a multi-arm Bernoulli bandit.
    Belief state: integer array of shape (num_arms, num_outcomes),
                  state[a, i] = times outcome i was observed on arm a.
    Hypothesis params: list of num_arms per-arm params, params[a] = (log_probs, coefficients)
                       encoding the reward distribution for arm a.
    Construct params via make_params(probabilities_list) where probabilities_list[a]
    is the probability vector for arm a.
    """

    def __init__(self, num_arms: int, num_outcomes: int = 2):
        self.num_arms = num_arms
        self.num_outcomes = num_outcomes

    def make_params(self, arm_hypotheses: list):
        """
        arm_hypotheses[a]: list of probability vectors for arm a, each shape (num_outcomes,).
        A single vector means a point hypothesis for that arm.
        Multiple vectors mean a mixture prior (uniform weights) — use this for
        independent per-arm inference without enumerating joint hypotheses.
        """
        assert len(arm_hypotheses) == self.num_arms
        result = []
        for hyps in arm_hypotheses:
            if isinstance(hyps, np.ndarray):
                hyps = [hyps]  # single point hypothesis
            probs = np.stack(hyps)  # shape (num_components, num_outcomes)
            assert np.allclose(probs.sum(axis=1), 1)
            log_probs = np.log(np.maximum(probs, 1e-300))
            coefficients = np.ones(len(hyps)) / len(hyps)
            result.append((log_probs, coefficients))
        return result

    def mix_params(self, params_list: list, coefficients: np.ndarray):
        """Mix per-arm params independently across arms."""
        mixed = []
        for arm in range(self.num_arms):
            arm_params = [p[arm] for p in params_list]
            log_probs = np.exp(np.concatenate([p[0] for p in arm_params], axis=0))
            coefs = np.concatenate([p[1] * c for p, c in zip(arm_params, coefficients)])
            coefs /= coefs.sum()
            mixed.append((np.log(np.maximum(log_probs, 1e-300)), coefs))
        return mixed

    def event_index(self, outcome: Outcome) -> int:
        return int(round(outcome.reward * (self.num_outcomes - 1)))

    def initial_state(self) -> np.ndarray:
        return np.zeros((self.num_arms, self.num_outcomes), dtype=np.int64)

    def update_state(self, state: np.ndarray, outcome: Outcome,
                     action: int) -> np.ndarray:
        new_state = state.copy()
        new_state[action, self.event_index(outcome)] += 1
        return new_state

    def is_initial(self, state: np.ndarray) -> bool:
        return (state == 0).all()

    def compute_likelihood(self, belief_state, outcome, params,
                           action: int, policy=None) -> float:
        probs = self._predictive(belief_state[action], params[action])
        return float(probs[self.event_index(outcome)])

    def compute_expected_reward(self, belief_state, reward_function,
                                params, action: int, policy=None) -> float:
        probs = self._predictive(belief_state[action], params[action])
        return float(probs @ reward_function)

    def _predictive(self, arm_counts: np.ndarray, arm_params) -> np.ndarray:
        """Posterior predictive P(next outcome | arm history) for mixture of categoricals."""
        log_probs, coefficients = arm_params
        lp = np.expand_dims((log_probs * arm_counts).sum(axis=1), axis=1)
        lp -= lp.max()  # numerical stability before exp
        probs = coefficients @ np.exp(lp + log_probs)
        return probs / probs.sum()
```

---

### `NewcombWorldModel`

Policy-dependent environment: the predictor's action is correlated with the agent's policy, not just with observable history. Requires both `action` and `policy` to compute likelihoods correctly.

```python
class NewcombWorldModel(WorldModel):
    """
    World model for environments where the predictor conditions on the agent's policy.

    Event = env_action (what the predictor did). reward_function[e] is the reward
    when env_action=e, for the specific agent action being evaluated — the caller
    constructs this. This means the reward structure is fully captured by
    reward_function; no belief state is needed.

    Hypothesis params: np.ndarray of shape (num_actions, num_actions),
                       predictor_matrix[e, a] = P(env_action=e | agent plays a).

    Assumption: P(env_action | policy) is independent of the reward realization.
    This holds for all standard Newcomb-like problems where the predictor's
    strategy does not correlate with the reward at stake.
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    @property
    def num_events(self) -> int:
        return self.num_actions

    def make_params(self, predictor_matrix: np.ndarray):
        assert predictor_matrix.shape == (self.num_actions, self.num_actions)
        return predictor_matrix

    def mix_params(self, params_list: list, coefficients: np.ndarray):
        return sum(c * p for p, c in zip(params_list, coefficients))

    def event_index(self, outcome: Outcome) -> int:
        return outcome.env_action

    def initial_state(self):
        return None

    def update_state(self, state, outcome: Outcome, action: int):
        return None  # stateless — hypothesis is entirely about predictor strategy

    def is_initial(self, state) -> bool:
        return state is None

    def compute_likelihood(self, belief_state, outcome, params,
                           action: int, policy=None) -> float:
        assert policy is not None
        return float((params @ policy)[outcome.env_action])

    def compute_expected_reward(self, belief_state, reward_function,
                                params, action: int, policy=None) -> float:
        assert policy is not None
        return float((params @ policy) @ reward_function)
```

---

### Changes to `ibrl/infrabayesian/infradistribution.py`

**1. Accept `world_model` in `__init__`, replace `history` with `belief_state`**

```diff
-def __init__(self, measures: list[AMeasure]):
+def __init__(self, measures: list[AMeasure], world_model: WorldModel = None):
     assert isinstance(measures, list) and len(measures) > 0
     self.measures = measures
-    self.history = np.zeros(self.measures[0].num_outcomes, dtype=np.int64)
+    self.world_model = world_model  # required — no default, caller must supply
+    self.belief_state = self.world_model.initial_state()
```

**2. Thread `world_model`, `action`, and `policy` through all calls**

```diff
-def evaluate(self, reward_function: np.ndarray) -> float:
+def evaluate_action(self, reward_function: np.ndarray,
+                   action: int,
+                   policy: np.ndarray | None = None) -> float:
-    return min(measure.evaluate(self.history, reward_function)
+    return min(measure.evaluate_action(self.world_model, self.belief_state,
+                                      reward_function[action], action=action, policy=policy)
               for measure in self.measures)
```

**3. Update signature and body**

`update` always normalizes by Q(a_t | π) — the expected value conditioned on the specific action taken. Using V(π) here would be wrong: the observed event is the joint (action=a_t, outcome=o_t), so the normalizer P^g_H(L) must be the probability of that joint event, which is action-specific. V(π) = Σ_a π(a)×P(o_t | a, π) conflates it with all other actions' probabilities. `policy` is passed through transparently so world models that need it (Newcomb) can compute P(env | π); world models that don't (Bernoulli) ignore it.

```diff
-def update(self, reward_function: np.ndarray, event: int) -> None:
+def update(self, reward_function: np.ndarray, outcome: Outcome,
+           action: int,
+           policy: np.ndarray | None = None) -> None:
+    event = self.world_model.event_index(outcome)
-    glued0 = self._glue(0, event, reward_function)
-    glued1 = self._glue(1, event, reward_function)
+    glued0 = self._glue(0, event, reward_function[action])
+    glued1 = self._glue(1, event, reward_function[action])
-    expect0 = self.evaluate_action(glued0)
-    expect1 = self.evaluate_action(glued1)
+    expect0  = self.evaluate_action(glued0, action=action, policy=policy)
+    expect1  = self.evaluate_action(glued1, action=action, policy=policy)
     prob = expect1 - expect0
-    expect_m = [measure.evaluate_action(self.history, glued0) for measure in self.measures]
+    expect_m = [measure.evaluate_action(self.world_model, self.belief_state, glued0,
+                                       action=action, policy=policy)
+                for measure in self.measures]

     for measure in self.measures:
-        measure.scale *= measure.compute_probabilities(self.history)[event]
-    self.history[event] += 1
+        measure.scale *= self.world_model.compute_likelihood(
+            self.belief_state, outcome, measure.params, action=action, policy=policy)
+    self.belief_state = self.world_model.update_state(self.belief_state, outcome,
+                                                      action=action)

     for i, measure in enumerate(self.measures):
         measure.offset = expect_m[i]

     for measure in self.measures:
         measure.offset -= expect0
         measure /= prob
```

`action` is now a required positional argument (not `int | None`). The agent always has a concrete action at update time.

**4. Add `evaluate_policy`**

```python
def evaluate_policy(self, reward_function: np.ndarray,
                               policy: np.ndarray) -> float:
    """V(π) = E_H[reward | policy=π] = inf over a-measures of policy-weighted expected reward.
    Used by the shared-infradistribution agent in place of per-arm evaluate_action.
    Concave in π (infimum of linear functions), so maximisation over the simplex
    is a convex optimisation problem.
    """
    return min(
        measure.evaluate_policy(
            self.world_model, self.belief_state, reward_function, policy
        )
        for measure in self.measures
    )
```

**5. Update `mix` and `mixKU`**

`mix` now delegates mixed-param construction to `world_model.mix_params` rather than accessing params internals directly. This keeps all model-family knowledge in `WorldModel`.

```diff
 for component in components:
-    assert component.history.sum() == 0
+    assert component.world_model.is_initial(component.belief_state), \
+        "Can only mix unused infradistributions"
     for measure in component.measures:
         assert np.isclose(measure.scale, 1)
         assert np.isclose(measure.offset, 0)
-        assert measure.num_outcomes == components[0].measures[0].num_outcomes
+assert all(
+    isinstance(c.world_model, type(components[0].world_model))
+    for c in components
+), "All components must share the same WorldModel type"

 new_measures = []
 for measures in itertools.product(*(c.measures for c in components)):
+    params_list = [m.params for m in measures]
+    coef_weights = coefficients  # one per component
+    mixed_params = components[0].world_model.mix_params(params_list, coef_weights)
-    mix_probabilities = np.exp(np.concatenate([m.log_probabilities for m in measures]))
-    mix_coefficients = np.concatenate([m.coefficients * coefficients[i] for i, m in enumerate(measures)])
-    mix_coefficients /= mix_coefficients.sum()
-    new_measures.append(AMeasure.mixed(mix_probabilities, mix_coefficients))
+    new_measures.append(AMeasure(mixed_params))

-return cls(new_measures)
+return cls(new_measures, world_model=components[0].world_model)
```

---

### Changes to `ibrl/agents/infrabayesian.py`

**Constructor: explicit hypotheses and prior**

```diff
 def __init__(self, *args,
-        num_hypotheses: int = 5,
+        hypotheses: list[Infradistribution],
+        prior: np.ndarray | None = None,
+        reward_function: np.ndarray | None = None,
         **kwargs):
     super().__init__(*args, **kwargs)
-    self.num_hypotheses = num_hypotheses
-    self.hypotheses = np.stack([
-        np.linspace(1., 0., self.num_hypotheses),
-        np.linspace(0., 1., self.num_hypotheses)
-    ], axis=-1)
-    self.reward_function = np.array([0., 1.])
+    assert len(hypotheses) > 0
+    assert all(isinstance(h.world_model, type(hypotheses[0].world_model))
+               for h in hypotheses), "All hypotheses must share the same WorldModel type"
+    self.hypotheses = hypotheses
+    self.prior = prior if prior is not None else np.ones(len(hypotheses)) / len(hypotheses)
+    self.reward_function = reward_function if reward_function is not None \
+                           else np.tile([0., 1.], (self.num_actions, 1))
```

**`reset()` — one shared infradistribution**

```diff
 def reset(self):
     super().reset()
-    self.dists = []
-    for _ in range(self.num_actions):
-        infradistributions = [
-            Infradistribution([AMeasure.pure(self.hypotheses[i])])
-                for i in range(self.num_hypotheses)
-        ]
-        coefficients = np.ones(self.num_hypotheses) / self.num_hypotheses
-        self.dists.append(Infradistribution.mix(infradistributions, coefficients))
+    self.dist = Infradistribution.mix(self.hypotheses, self.prior)
```

A single `self.dist` replaces the per-arm `self.dists` list. Evidence from any arm is accumulated into one infradistribution, so observations under one action inform predictions for all others. This is the theoretically faithful implementation of Fund 3 for policy-dependent environments — the belief is over the causal law Λ: Π → □(A×O)^ω, which is shared.

**`update` — updates shared dist on every step**

```diff
 def update(self, probabilities: NDArray, action: int, outcome: Outcome) -> None:
     super().update(probabilities, action, outcome)
-    observation = int(outcome.reward > 0.5)
-    self.dists[action].update(self.reward_function, observation)
+    self.dist.update(self.reward_function, outcome,
+                     action=action, policy=probabilities)
```

Every step updates `self.dist` regardless of which arm was taken. `probabilities` (the agent's mixed strategy) is passed as `policy` — policy-dependent world models use it to compute P(env_action | policy).

**`get_policy` — optimise V(π) over the simplex**

```diff
-def get_probabilities(self) -> np.ndarray:
-    return self.build_greedy_policy(self._expected_rewards())
+def get_probabilities(self) -> np.ndarray:
+    return self._optimise_policy()
+
+def _optimise_policy(self) -> np.ndarray:
+    from scipy.optimize import linprog
+    k = self.num_actions
+    K = len(self.dist.measures)
+
+    # V(π) = min_k α_k(π) is piecewise-linear and concave.
+    # Maximise V(π) over the simplex via LP:
+    #   max t  s.t.  α_m(π) >= t  for all m,  π ∈ Δ_k
+    #
+    # Decision variables: x = [π_0, ..., π_{k-1}, t]
+    # linprog minimises, so minimise -t.
+
+    c = np.zeros(k + 1)
+    c[-1] = -1.0  # minimise -t
+
+    # Inequality: -α_m(π) + t <= 0  for each measure m
+    # α_m(π) = Σ_a π(a) · v_m(a)  where v_m = measure.evaluate_action()
+    A_ub = np.zeros((K, k + 1))
+    for m_idx, m in enumerate(self.dist.measures):
+        v = m.evaluate_action(self.dist.world_model, self.dist.belief_state,
+                             self.reward_function)
+        A_ub[m_idx, :k] = -v  # placeholder — see note
+        A_ub[m_idx, -1] = 1.0
+    b_ub = np.zeros(K)
+
+    # Equality: Σ π(a) = 1
+    A_eq = np.zeros((1, k + 1))
+    A_eq[0, :k] = 1.0
+    b_eq = np.array([1.0])
+
+    bounds = [(0, 1)] * k + [(None, None)]
+    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
+                     bounds=bounds)
+    return result.x[:k]
```

`V(π) = dist.evaluate_policy(reward_function, π)` is **concave in π** — it is an infimum of linear functions. Maximising a concave function over the simplex is a convex optimisation problem with a guaranteed global optimum.

Because `V(π)` is piecewise-linear (not smooth), gradient-based solvers like SLSQP can struggle at the kinks where the argmin measure switches — numerical gradients are unreliable there. The LP formulation avoids this: kinks are just constraint boundaries, handled natively by the simplex or interior-point method.

The three qualitatively different cases:

| Environment | V(π) shape | Optimal policy |
|---|---|---|
| Standard MAB (independent arms) | Linear | Deterministic — LP finds the argmax vertex |
| Newcomb | Linear (slope favours one-boxing) | Deterministic one-boxing |
| Death in Damascus | Quadratic (V = 2p(1-p)), interior maximum | Uniform mix — requires SLSQP, not LP |

For MABs, the LP is equivalent to argmax — no behavioural change. For NDPs, it correctly finds mixed policies that per-arm argmax cannot represent.

Note: for `NewcombWorldModel` with a 1D reward_function, `α_m(π) = (P_m @ π) @ rf` is linear in π, so the LP is valid. With a 2D reward_function (action-dependent payoffs, e.g. Death in Damascus), `α_m(π) = Σ_a π[a] × (P_m @ π) @ rf[a]` is quadratic in π — the LP constraint matrix is no longer constant and the LP formulation is invalid. In that case, maximise `dist.evaluate_policy(reward_function, π)` directly via SLSQP. The objective is always concave (infimum of concave functions), so SLSQP finds the global optimum. This is Fix 2 — to be addressed separately.

**`dump_state`**

```diff
-state = "["+",".join(dump_array(dist.history, "%d") for dist in self.dists)+"]"
+state = str(self.dist.belief_state)
```

**Convenience constructor preserving the Bernoulli grid default**

```python
@classmethod
def bernoulli_grid(cls, num_actions: int, num_hypotheses: int = 5,
                   prior: np.ndarray | None = None, **kwargs) -> "InfraBayesianAgent":
    """Convenience: uniform grid of Bernoulli hypotheses, matching old default behaviour."""
    wm = MultiBernoulliWorldModel(num_arms=num_actions)
    grid = [np.array([1 - p, p]) for p in np.linspace(0., 1., num_hypotheses)]
    # Each arm gets the same N-point hypothesis grid independently.
    # A single a-measure with N components per arm gives correct independent
    # per-arm Bayesian inference — no joint enumeration needed.
    params = wm.make_params([grid] * num_actions)
    hypotheses = [Infradistribution([AMeasure(params)], world_model=wm)]
    return cls(num_actions=num_actions, hypotheses=hypotheses,
               prior=np.array([1.0]), **kwargs)
```

For `BernoulliWorldModel`, arms are independent so V(π) is linear and the optimiser returns the argmax arm — identical behaviour to the previous per-arm design. `bernoulli_grid` therefore also preserves equivalence with `DiscreteBayesianAgent`.

**Newcomb / Death in Damascus agent construction**

```python
# Newcomb: prior over two predictor strategy hypotheses
# Reward: one-box (action=1) gets 1 if predictor opens box B (env=1), else 0.
wm = NewcombWorldModel(num_actions=2)
agent = InfraBayesianAgent(
    num_actions=2,
    hypotheses=[
        Infradistribution([AMeasure(wm.make_params(np.eye(2)))],             world_model=wm),
        Infradistribution([AMeasure(wm.make_params(np.full((2,2), 0.5)))],   world_model=wm),
    ],
    prior=np.array([0.5, 0.5]),
    # reward_function defaults to np.tile([0., 1.], (2, 1)) — same rf for both actions.
    # Correct for Newcomb where reward depends only on env_action.
)
# Agent will learn the predictor's strategy and converge to one-boxing policy.

# Death in Damascus: action-dependent reward — you live only if you go where Death isn't.
# stay=0, flee=1; env_action=0 means Death stays in Damascus, env_action=1 means Death flees.
# Reward: 1 if agent and Death are in different cities, 0 if same.
agent_did = InfraBayesianAgent(
    num_actions=2,
    hypotheses=[Infradistribution([AMeasure(wm.make_params(np.eye(2)))], world_model=wm)],
    prior=np.array([1.0]),
    reward_function=np.array([
        [0., 1.],   # stay (action=0): die if Death stays (env=0), live if Death flees (env=1)
        [1., 0.],   # flee (action=1): live if Death stays (env=0), die if Death flees (env=1)
    ]),
)
# V(π) = 2·π[0]·π[1], maximised at π=(0.5, 0.5). Agent will converge to uniform mixing.
```

---

## Fund 4 Extension Path

Fund 3 (this plan) represents a hypothesis as a distribution over trajectories, using sufficient statistics as belief state. Fund 4 (supra-POMDP) represents the same hypothesis as a factored state-space model: initial state infradistribution Θ₀ ∈ □S, transition infrakernel T: S×A → □S, observation model B: S → O. Fund 4 is a factored representation of Fund 3 — they are not competing theories.

The `WorldModel[S]` interface supports both:

| | Fund 3 (this plan) | Fund 4 (future) |
|---|---|---|
| `belief_state` | sufficient statistic of trajectory | distribution over latent states |
| `update_state` | extend sufficient statistic | Bayesian filter: predict via T, update via B |
| `params` | trajectory distribution params | kernel params (T, B, Θ₀) |
| `compute_likelihood` | posterior predictive from statistic | marginalise over latent states |

A Fund 4 world model requires no changes to `AMeasure`, `Infradistribution`, or the agent:

```python
class SupraPOMDPWorldModel(WorldModel):
    """Fund 4: belief state is a distribution over latent states.
    Params: tuple (transition, observation, initial) where
        transition: shape (num_states, num_actions, num_states) — T[s, a, s']
        observation: shape (num_states, num_obs) — B[s, o]
        initial: shape (num_states,) — Θ₀
    """
    def make_params(self, transition, observation, initial):
        return (transition, observation, initial)

    def update_state(self, belief, outcome, action=None):
        transition, observation, _ = belief_params  # params not needed here; belief carries state
        belief_pred = belief @ transition[:, action, :]
        obs = self.event_index(outcome)
        belief_post = observation[:, obs] * belief_pred
        return belief_post / belief_post.sum()
    ...
```

One current limitation: the per-action infradistribution design assumes latent state is independent across actions. For environments where the predictor builds a persistent model of the agent across rounds (across actions), a single shared infradistribution would be needed. This is a future architectural change.

---

## Fixed Vertex Set Is Preserved Under Definition 11

A concern from `20260328_infrabayes_issues.md` (Issue 2): does the fixed vertex set
lose extreme points after conditioning? The imprecise probability literature warns
that credal set conditioning can create new extreme points, because the map
`P → P(·|E) = P(·∩E)/P(E)` divides by `P(E)` which varies per distribution —
a nonlinear map that distorts the convex hull.

**This does not apply to inframeasure conditioning.** Definition 11's update is
affine on a-measures, so it preserves convex structure exactly.

*Proof sketch.* Take a convex combination `α = w₁α₁ + w₂α₂` (with `w₁+w₂ = 1`)
from the infradistribution's convex hull. Under Definition 11, each a-measure
`(m, b)` maps to:

```
cond(α)(f) = (m(L·f) + b + m((1-L)·g) - E_H((1-L)·g)) / P^g_H(L)
```

The terms `E_H((1-L)·g)` and `P^g_H(L)` are global constants (the same for every
a-measure). The remaining terms `m(L·f)`, `b`, `m((1-L)·g)` are all linear in
`(m, b)`. Substituting `m_α = w₁m₁ + w₂m₂`, `b_α = w₁b₁ + w₂b₂`:

```
cond(α)(f) = [w₁(m₁(L·f) + b₁ + m₁((1-L)g)) + w₂(m₂(L·f) + b₂ + m₂((1-L)g)) - C] / P
           = w₁·cond(α₁)(f) + w₂·cond(α₂)(f) + (w₁ + w₂ - 1)·C/P
           = w₁·cond(α₁)(f) + w₂·cond(α₂)(f)
```

So the conditioned version of any convex combination equals the convex combination
of conditioned vertices. The convex hull maps to the convex hull; upper closure
commutes with affine maps; no new extreme points are created.

The key difference from credal set conditioning: in Definition 11, the normalization
`P^g_H(L)` is a single global scalar, not a per-measure `P_k(E)`. The per-measure
likelihood `μ_k(obs)` does vary across measures, but it only appears as a
multiplicative factor in the scale update `λ_k *= μ_k(obs)/P` — this changes each
vertex differently but doesn't affect the linearity of the map on convex
combinations, because the map on (m, b) pairs is still affine with a shared
constant offset.

**Consequence**: `evaluate_policy(π) = min_k α_k(π)` is exact — no
missing extreme points, no need for vertex recomputation or support function
representations. The concern in Issue 2 of the issues doc is a non-issue for
Definition 11.

---

## Scalability and Representational Constraints

### Exponential growth under KU mixing

`Infradistribution.mix` takes the Cartesian product of a-measures across all input components:

```python
for measures in itertools.product(*(component.measures for component in components)):
```

Mixing K infradistributions with n₁, n₂, …, nK a-measures produces **n₁ × n₂ × … × nK a-measures** in the result. Every subsequent `update` and `evaluate_action` call iterates over all of them.

**For pure hypotheses** (each input has 1 a-measure): product is 1^K = 1. You always get a single mixed a-measure with K internal probability components. Bayesian updating happens implicitly inside `_predictive` via the `c_j × P(history | j)` weights. Cost per update is O(K × num_outcomes) — no exponential.

**For KU hypotheses** (each input has M a-measures from `mixKU`): product is M^K. With K=5 hypotheses each with M=2 KU worlds: 32 a-measures; K=10: 1024. This growth is a **mathematical necessity, not an implementation artifact**. Under KU, the infra-expected value requires taking the minimum over all a-measures; when you combine that with a classical prior over K such hypotheses, the worst-case must account for all K-tuples of extremal points — one from each hypothesis. There is no representation that avoids this without approximation.

Practical consequence: keep the number of KU worlds per hypothesis small (2–3 is typical). The number of hypotheses in the classical prior can be large without issue, as long as each has 1 a-measure (no KU).

**For `MultiBernoulliWorldModel` with `bernoulli_grid`**: independent per-arm inference requires no joint enumeration. Each arm carries its own N-component hypothesis grid inside a single a-measure. `_predictive(belief_state[a], params[a])` uses only arm a's counts and arm a's components — other arms never appear. Cost per update is O(N × num_outcomes) regardless of K.

### Discrete outcome assumption

The entire current framework — `BernoulliWorldModel`, `_glue`, `event_index`, `reward_function` as `np.ndarray` — assumes a **finite discrete outcome space**. Concretely:

- `event_index(outcome) -> int` must return a finite integer index
- `reward_function` is a vector indexed by that integer
- `_glue(value, event, reward_function)` sets `reward_function[event] = value` — requires integer indexing
- `_predictive` returns a probability vector over the finite outcome set
- `compute_expected_reward` is a dot product `probs @ reward_function`

For a **Gaussian bandit**, the outcome is a continuous real-valued reward. This breaks every one of these:
- No finite event index — `event_index` cannot discretize without information loss
- `reward_function` would need to be a callable `f: R → [0, 1]`, not a vector
- `compute_expected_reward` becomes an integral `∫ f(r) p(r | belief, params) dr`
- `_glue` would need to produce a new function with a point replaced, which for continuous outcomes means the gluing operation `f ★^x g` (gluing at a specific observation x) acts pointwise: `(f ★^x g)(y) = f(y)` for y ≠ x, value at x — but measure-zero points don't affect integrals

This is not fixed by writing a `GaussianWorldModel` subclass alone. It requires changing `Infradistribution._glue` and the `update` method to work with function-valued reward functions rather than vectors. The `WorldModel[S, P]` interface would need `compute_expected_reward` to accept a callable. This is a deeper architectural change than what this plan covers.

The `GaussianWorldModel` stub in the plan is only tractable if rewards are discretized before passing to the infradistribution — which is what the current agent already does (`int(outcome.reward > 0.5)`). A true continuous-outcome infrabayesian agent requires extending the framework beyond discrete O.

---

## Behavioral Anchors

These tests verify the observable behavior of `Infradistribution` under the new API. They are written for the post-refactor interface and should all pass after step 7 of the order of work.

### `tests/test_ib_world_model.py`

```python
import numpy as np
import pytest
from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.infrabayesian.world_model import MultiBernoulliWorldModel, NewcombWorldModel
from ibrl.outcome import Outcome

NUM_ARMS = 2
ARM = 0  # default arm for single-arm tests

def make_dist(num_hypotheses=5):
    """Uniform prior over linspace Bernoulli hypotheses (single arm)."""
    wm = MultiBernoulliWorldModel(num_arms=NUM_ARMS)
    hypotheses = [
        Infradistribution([AMeasure(wm.make_params(np.array([1 - p, p])))], world_model=wm)
        for p in np.linspace(0., 1., num_hypotheses)
    ]
    return Infradistribution.mix(hypotheses, np.ones(num_hypotheses) / num_hypotheses)


def obs(reward: float) -> Outcome:
    return Outcome(reward=reward)


REWARD = np.array([0., 1.])


# ── Normalization conditions ────────────────────────────────────────────────

def test_e_h_zero_is_zero():
    """E_H([0, 0]) == 0 at initialization."""
    dist = make_dist()
    assert abs(dist.evaluate_action(np.zeros(2), action=ARM)) < 1e-9


def test_e_h_zero_stays_zero_after_updates():
    """E_H([0, 0]) == 0 is preserved after updates."""
    dist = make_dist()
    for r in [1., 0., 1., 1., 0.]:
        dist.update(REWARD, obs(r), action=ARM)
    assert abs(dist.evaluate_action(np.zeros(2), action=ARM)) < 1e-9


def test_evaluate_action_in_unit_interval():
    dist = make_dist()
    assert 0. <= dist.evaluate_action(REWARD, action=ARM) <= 1.


# ── Learning direction ──────────────────────────────────────────────────────

def test_update_increases_evaluate_action_after_rewards():
    dist = make_dist()
    ev_before = dist.evaluate_action(REWARD, action=ARM)
    for _ in range(20):
        dist.update(REWARD, obs(1.), action=ARM)
    assert dist.evaluate_action(REWARD, action=ARM) > ev_before


def test_update_decreases_evaluate_action_after_no_rewards():
    dist = make_dist()
    for _ in range(5):
        dist.update(REWARD, obs(1.), action=ARM)
    ev_before = dist.evaluate_action(REWARD, action=ARM)
    for _ in range(20):
        dist.update(REWARD, obs(0.), action=ARM)
    assert dist.evaluate_action(REWARD, action=ARM) < ev_before


# ── Scale and offset invariants ─────────────────────────────────────────────

def test_scale_and_offset_nonnegative():
    dist = make_dist()
    for r in [1., 0., 1., 1., 0., 0., 1.]:
        dist.update(REWARD, obs(r), action=ARM)
    for m in dist.measures:
        assert m.scale >= 0.
        assert m.offset >= 0.


def test_pure_measure_evaluate_action():
    wm = MultiBernoulliWorldModel(num_arms=NUM_ARMS)
    dist = Infradistribution([AMeasure(wm.make_params(np.array([0.3, 0.7])))], world_model=wm)
    assert abs(dist.evaluate_action(REWARD, action=ARM) - 0.7) < 1e-6


def test_mix_pessimistic():
    """Classical mixture: expected value is the more pessimistic hypothesis."""
    wm = MultiBernoulliWorldModel(num_arms=NUM_ARMS)
    d_low  = Infradistribution([AMeasure(wm.make_params(np.array([0.8, 0.2])))], world_model=wm)
    d_high = Infradistribution([AMeasure(wm.make_params(np.array([0.2, 0.8])))], world_model=wm)
    dist = Infradistribution.mix([d_low, d_high], np.array([0.5, 0.5]))
    assert abs(dist.evaluate_action(REWARD, action=ARM) - 0.2) < 1e-6


# ── Policy-dependent (Newcomb) ──────────────────────────────────────────────

def newcomb_obs(reward: float, env_action: int) -> Outcome:
    return Outcome(reward=reward, env_action=env_action)


def test_newcomb_perfect_predictor_one_boxing():
    """Perfect predictor: one-boxing policy has higher expected reward.
    reward_function[e] = reward when env_action=e for this agent action.
    For one-boxing (action=1): env=0 → 0, env=1 → 1.
    """
    wm = NewcombWorldModel(num_actions=2)
    perfect = wm.make_params(np.eye(2))

    one_box_policy = np.array([0., 1.])  # always action 1
    two_box_policy = np.array([1., 0.])  # always action 0

    # Under perfect predictor: P(env=e | policy) = policy[e]
    # one-box: P(env=1)=1.0, reward=1 → EV=1
    # two-box: P(env=0)=1.0, reward=0 → EV=0
    ev_one_box = wm.compute_expected_reward(None, REWARD, perfect,
                                            action=1, policy=one_box_policy)
    ev_two_box = wm.compute_expected_reward(None, REWARD, perfect,
                                            action=0, policy=two_box_policy)
    assert ev_one_box > ev_two_box


def test_newcomb_random_predictor_independence():
    """Random predictor: expected reward is independent of policy."""
    wm = NewcombWorldModel(num_actions=2)
    rand = wm.make_params(np.full((2, 2), 0.5))

    ev1 = wm.compute_expected_reward(None, REWARD, rand,
                                     action=1, policy=np.array([0., 1.]))
    ev2 = wm.compute_expected_reward(None, REWARD, rand,
                                     action=1, policy=np.array([0.5, 0.5]))
    assert abs(ev1 - ev2) < 1e-9


def test_newcomb_event_index_is_env_action():
    """event_index returns env_action, not binarized reward."""
    wm = NewcombWorldModel(num_actions=2)
    assert wm.event_index(newcomb_obs(0.9, env_action=0)) == 0
    assert wm.event_index(newcomb_obs(0.8, env_action=1)) == 1


# ── Policy optimisation (shared dist) ──────────────────────────────────────

def make_newcomb_agent():
    """IB agent with prior over perfect vs random predictor."""
    wm = NewcombWorldModel(num_actions=2)
    return InfraBayesianAgent(
        num_actions=2,
        hypotheses=[
            Infradistribution([AMeasure(wm.make_params(np.eye(2)))],             world_model=wm),
            Infradistribution([AMeasure(wm.make_params(np.full((2,2), 0.5)))],   world_model=wm),
        ],
        prior=np.array([0.5, 0.5]),
    )


def test_newcomb_policy_optimisation_favours_one_boxing():
    """After many one-boxing rounds under perfect predictor, agent converges to one-boxing."""
    from ibrl.agents.infrabayesian import InfraBayesianAgent
    agent = make_newcomb_agent()
    agent.reset()

    # Simulate: agent one-boxes, predictor mirrors, reward = 1
    for _ in range(30):
        policy = agent.get_probabilities()
        agent.update(policy, action=1, outcome=Outcome(reward=1., env_action=1))

    final_policy = agent.get_probabilities()
    assert final_policy[1] > 0.9  # should have converged to one-boxing


def test_death_in_damascus_converges_to_uniform():
    """Death in Damascus: V(π) is quadratic with the correct action-dependent reward,
    and the optimal policy is uniform mix.

    With perfect predictor P=eye(2) and reward_function=[[0,1],[1,0]]:
        E[r | stay,   π] = (eye @ π) @ [0, 1] = π[1]
        E[r | flee,   π] = (eye @ π) @ [1, 0] = π[0]
        V(π) = π[0]·π[1] + π[1]·π[0] = 2·π[0]·π[1]
    Maximised at π = (0.5, 0.5), V = 0.5.
    """
    from ibrl.agents.infrabayesian import InfraBayesianAgent
    wm = NewcombWorldModel(num_actions=2)
    agent = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[Infradistribution([AMeasure(wm.make_params(np.eye(2)))], world_model=wm)],
        prior=np.array([1.0]),
        reward_function=np.array([
            [0., 1.],   # stay: die if Death stays (env=0), live if Death flees (env=1)
            [1., 0.],   # flee: live if Death stays (env=0), die if Death flees (env=1)
        ]),
    )
    agent.reset()
    policy = agent.get_probabilities()
    # With a fully known symmetric predictor, the policy optimiser should immediately
    # find the interior optimum without any learning steps needed.
    assert abs(policy[0] - 0.5) < 0.05
    assert abs(policy[1] - 0.5) < 0.05
```

---

## What This Unlocks

After this refactor, adding a new model family requires only a new `WorldModel` subclass — no changes to `AMeasure`, `Infradistribution`, or the agent:

- **Non-binary discrete outcomes**: `BernoulliWorldModel(num_outcomes=3)` works immediately
- **Gaussian bandit**: `GaussianWorldModel` with Welford belief state; `compute_likelihood` uses Gaussian posterior predictive; `GaussianParams(prior_mean, prior_precision, noise_precision)`
- **Newcomb / game environments**: `NewcombWorldModel` (implemented above); shared infradistribution correctly converges to one-boxing
- **Death in Damascus and NDPs with mixed optimal policies**: shared infradistribution + `_optimise_policy` finds interior optima that per-arm argmax cannot represent; standard RL agents fail these problems
- **Supra-POMDP (Fund 4)**: `SupraPOMDPWorldModel` with distribution-over-states belief state; `compute_likelihood` marginalises over latent states; no other changes

---

## Order of Work

1. Write `tests/test_ib_world_model.py` (the anchor tests above) — they will fail until step 7
2. Add `world_model.py`: `WorldModel[S]` (abstract), `MultiBernoulliWorldModel`, `NewcombWorldModel`
3. Refactor `AMeasure`: replace `(log_probabilities, coefficients)` with `params: P`; update `pure`/`mixed` factories; fix `reset()` bug (`offset = 0`); add `evaluate_policy`; update `compute_likelihood`/`evaluate` to delegate to passed-in `world_model`
4. Refactor `Infradistribution`: rename `history` → `belief_state`; add `world_model`; add `evaluate_policy`; update `update` to dispatch on policy presence; thread `action`/`policy` through `mix`/`mixKU`; extract `event_index` from `world_model`
5. Update `InfraBayesianAgent`: replace `num_hypotheses` with explicit `hypotheses`/`prior`; replace per-arm `self.dists` with shared `self.dist`; replace `_expected_rewards` + argmax with `_optimise_policy`; add `bernoulli_grid` classmethod
6. Confirm all anchor tests pass
7. Confirm `DiscreteBayesianAgent` equivalence still holds via `bernoulli_grid` (V(π) linear for MABs → optimiser returns argmax vertex)
8. Confirm Newcomb tests pass end-to-end on `NewcombEnvironment`
9. Confirm Death in Damascus test: `_optimise_policy` finds uniform mix at prior without learning steps
