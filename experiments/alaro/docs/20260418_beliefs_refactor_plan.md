# Refactor Plan: Generic `WorldModel` for `Infradistribution`

**Branch**: `alaro/beliefs` (off `fllor/ib-agent-architecture`)  
**Date**: 2026-04-18

---

## Motivation

The current `Infradistribution` has two hardcoded assumptions that limit generality:

1. **Count-based history**: `self.history = np.zeros(num_outcomes)` is the sufficient statistic for IID Bernoulli outcomes only. It won't generalize to Gaussian bandits, Newcomb, or any model where observations aren't exchangeable.

2. **Likelihood computation in `AMeasure`**: The posterior predictive computation (how to evaluate P(outcome | belief_state) under a hypothesis) is baked into `AMeasure._compute_probabilities`. This is model-family logic — a Gaussian or Newcomb model would compute it differently. It belongs in `WorldModel`, not in the a-measure, which should be pure data.

The fix is a `WorldModel` abstraction that owns the belief state, the update logic, and the likelihood/expected reward computation for a given model family. `AMeasure` is reduced to pure data: `(log_probabilities, coefficients, scale, offset)`. The change is entirely internal to `ibrl/infrabayesian/` — the public agent interface stays unchanged.

### Clarifying three distinct arrays

A common source of confusion: AMeasure parameters, the WorldModel belief state, and the agent's prior over hypotheses are all outcome-shaped arrays that mean completely different things:

| Array | Where | What it means |
|---|---|---|
| `AMeasure.log_probabilities` | AMeasure | Hypothesis parameters: "my model says outcome k occurs with probability p[k]" |
| `Infradistribution.belief_state` | Infradistribution | Evidence: "outcome k has been observed this many times" |
| `prior` | Agent / `Infradistribution.mix` | Prior over hypotheses: "I assign weight w[j] to hypothesis j" |

These interact only in `WorldModel.compute_likelihood(belief_state, outcome, log_probabilities, coefficients)` — which combines the evidence (belief_state) with the hypothesis parameters (from AMeasure) to compute a likelihood. They never otherwise touch.

### Note on reward bounds

The `0` and `1` in `_glue(0, ...)` and `_glue(1, ...)` are the extremal values required by the theory's normalization conditions (E_H(**0**) = 0, E_H(**1**) = 1) — not reward min/max. Definition 11 requires the reward function to live in [0, 1]. If an environment produces rewards outside that range, the agent normalizes before passing to the infradistribution. The `0` and `1` stay as-is.

---

## Proposed Design

### New file: `ibrl/infrabayesian/world_model.py`

`WorldModel` owns three things:
- Belief state lifecycle: `initial_state`, `update_state`, `is_initial`
- Likelihood computation: `compute_likelihood` — P(outcome | belief_state, hypothesis_params)
- Expected reward computation: `compute_expected_reward` — E[f | belief_state, hypothesis_params]

The contract is uniform: all methods take `outcome: Outcome` and read whichever fields are relevant to the model family.

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import numpy as np
from ..outcome import Outcome

S = TypeVar('S')


class WorldModel(ABC, Generic[S]):
    """
    Defines the belief state type, update logic, and likelihood computation
    for a parametric model family. Stateless itself — belief state is owned
    by Infradistribution.

    AMeasure stores hypothesis parameters (log_probabilities, coefficients).
    WorldModel knows how to use those parameters together with the belief state
    to compute likelihoods and expected rewards.
    """

    @abstractmethod
    def initial_state(self) -> S:
        """Return the initial belief state (sufficient statistic for this model)."""
        pass

    @abstractmethod
    def update_state(self, state: S, outcome: Outcome) -> S:
        """Return new belief state after observing outcome. Does not mutate state."""
        pass

    @abstractmethod
    def is_initial(self, state: S) -> bool:
        """True if state is the initial (no observations) state. Used by mix/mixKU."""
        pass

    @abstractmethod
    def compute_likelihood(self,
                           belief_state: S,
                           outcome: Outcome,
                           log_probabilities: np.ndarray,
                           coefficients: np.ndarray) -> float:
        """
        P(outcome | belief_state) under the hypothesis parameterised by
        (log_probabilities, coefficients). Returns a scalar.
        """
        pass

    @abstractmethod
    def compute_expected_reward(self,
                                belief_state: S,
                                reward_function: np.ndarray,
                                log_probabilities: np.ndarray,
                                coefficients: np.ndarray) -> float:
        """
        E[reward_function(outcome) | belief_state] under this hypothesis.
        Returns a scalar.
        """
        pass


class BernoulliWorldModel(WorldModel[np.ndarray]):
    """
    World model for IID discrete outcomes.
    Belief state: integer count array, state[i] = times outcome i was observed.
    Sufficient statistic for a mixture of categorical (Bernoulli) hypotheses.

    Reads from Outcome: outcome.outcome (discrete event index)
    """
    def __init__(self, num_outcomes: int = 2):
        self.num_outcomes = num_outcomes

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.num_outcomes, dtype=np.int64)

    def update_state(self, state: np.ndarray, outcome: Outcome) -> np.ndarray:
        new_state = state.copy()
        new_state[outcome.outcome] += 1
        return new_state

    def is_initial(self, state: np.ndarray) -> bool:
        return (state == 0).all()

    def compute_likelihood(self, belief_state, outcome, log_probabilities, coefficients):
        probs = self._predictive(belief_state, log_probabilities, coefficients)
        return float(probs[outcome.outcome])

    def compute_expected_reward(self, belief_state, reward_function,
                                log_probabilities, coefficients):
        probs = self._predictive(belief_state, log_probabilities, coefficients)
        return float(probs @ reward_function)

    def _predictive(self, belief_state, log_probabilities, coefficients) -> np.ndarray:
        """Posterior predictive: P(next outcome | history) for mixture of categoricals."""
        log_probs = np.expand_dims(
            (log_probabilities * belief_state).sum(axis=1), axis=1
        )
        probs = coefficients @ np.exp(log_probs + log_probabilities)
        return probs / probs.sum()
```

Future world models — showing the uniform `Outcome`-based contract:

```python
class GaussianWorldModel(WorldModel[dict]):
    """
    Belief state: Welford online statistics {'n', 'mean', 'M2'}.
    Reads from Outcome: outcome.reward (continuous scalar)
    """
    def initial_state(self):
        return {'n': 0, 'mean': 0., 'M2': 0.}

    def update_state(self, state, outcome: Outcome):
        n = state['n'] + 1
        delta = outcome.reward - state['mean']
        mean = state['mean'] + delta / n
        M2 = state['M2'] + delta * (outcome.reward - mean)
        return {'n': n, 'mean': mean, 'M2': M2}

    def is_initial(self, state):
        return state['n'] == 0

    def compute_likelihood(self, belief_state, outcome, log_probabilities, coefficients):
        # Gaussian posterior predictive given Welford stats + hypothesis params
        ...

    def compute_expected_reward(self, belief_state, reward_function,
                                log_probabilities, coefficients):
        # Integral of reward_function over posterior predictive density
        ...


class NewcombWorldModel(WorldModel):
    """
    Belief state: reward matrix, state[env_action, agent_action] = observed reward.
    Reads from Outcome: outcome.env_action, outcome.outcome, outcome.reward
    """
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def initial_state(self):
        return np.full((self.num_actions, self.num_actions), None, dtype=object)

    def update_state(self, state, outcome: Outcome):
        new_state = state.copy()
        new_state[outcome.env_action, outcome.outcome] = outcome.reward
        return new_state

    def is_initial(self, state):
        return all(v is None for v in state.flat)

    def compute_likelihood(self, belief_state, outcome, log_probabilities, coefficients):
        ...

    def compute_expected_reward(self, belief_state, reward_function,
                                log_probabilities, coefficients):
        ...
```

---

### Changes to `ibrl/infrabayesian/a_measure.py`

`AMeasure` becomes pure data. Remove `compute_probabilities` and `_compute_probabilities` entirely. `expected_value` and `compute_likelihood` delegate to `world_model`, which is passed in at call time (AMeasure does not hold a reference to the world model — that lives in Infradistribution).

```diff
-def compute_probabilities(self, history: np.ndarray) -> np.ndarray:
-    log_probs = np.expand_dims((self.log_probabilities*history).sum(axis=1),axis=1)
-    probs = self.coefficients @ np.exp(log_probs + self.log_probabilities)
-    return probs / probs.sum()
-
-def expected_value(self, history: np.ndarray, reward_function: np.ndarray) -> float:
-    return self.scale * (self.compute_probabilities(history) @ reward_function) + self.offset

+def compute_likelihood(self, world_model, belief_state, outcome: Outcome) -> float:
+    """Delegate to world_model — AMeasure provides its hypothesis parameters."""
+    return world_model.compute_likelihood(
+        belief_state, outcome, self.log_probabilities, self.coefficients
+    )
+
+def expected_value(self, world_model, belief_state, reward_function: np.ndarray) -> float:
+    raw = world_model.compute_expected_reward(
+        belief_state, reward_function, self.log_probabilities, self.coefficients
+    )
+    return self.scale * raw + self.offset
```

---

### Changes to `ibrl/infrabayesian/infradistribution.py`

**1. Accept `world_model` in `__init__`, replace `history` with `belief_state`**

```diff
-def __init__(self, measures: list[AMeasure]):
+def __init__(self, measures: list[AMeasure], world_model: WorldModel = None):
     assert isinstance(measures, list)
     assert len(measures) > 0
     self.measures = measures
-    self.history = np.zeros(self.measures[0].num_outcomes, dtype=np.int64)
+    self.world_model = world_model or BernoulliWorldModel()
+    self.belief_state = self.world_model.initial_state()
```

**2. Pass `world_model` through to all `AMeasure` calls**

`Infradistribution` is the single owner of `world_model` and threads it through every call to `measure.expected_value` and `measure.compute_likelihood`:

```diff
 def expected_value(self, reward_function: np.ndarray) -> float:
-    return min(measure.expected_value(self.history, reward_function)
+    return min(measure.expected_value(self.world_model, self.belief_state, reward_function)
                for measure in self.measures)
```

**3. Update `update` to use `Outcome` throughout**

Signature changes from `(reward_function, event: int)` to `(reward_function, outcome: Outcome)`. The `0` and `1` in gluing are unchanged — they are theory constants.

```diff
-def update(self, reward_function: np.ndarray, event: int) -> None:
+def update(self, reward_function: np.ndarray, outcome: Outcome) -> None:
-    glued0 = self._glue(0, event, reward_function)
-    glued1 = self._glue(1, event, reward_function)
+    glued0 = self._glue(0, outcome.outcome, reward_function)
+    glued1 = self._glue(1, outcome.outcome, reward_function)
     expect0 = self.expected_value(glued0)
     expect1 = self.expected_value(glued1)
     prob = expect1 - expect0
-    expect_m = [measure.expected_value(self.history, glued0) for measure in self.measures]
+    expect_m = [measure.expected_value(self.world_model, self.belief_state, glued0)
+                for measure in self.measures]

     for measure in self.measures:
-        measure.scale *= measure.compute_probabilities(self.history)[event]
-    self.history[event] += 1
+        measure.scale *= measure.compute_likelihood(self.world_model, self.belief_state, outcome)
+    self.belief_state = self.world_model.update_state(self.belief_state, outcome)

     for i, measure in enumerate(self.measures):
         measure.offset += expect_m[i]

     for measure in self.measures:
         measure.offset -= expect0
         measure /= prob
```

**4. Update `mix` and `mixKU`**

Validate freshness via the world model and forward it to the constructed infradistribution. Mixing requires the same observation space (same world model type):

```diff
 for component in components:
-    assert component.history.sum() == 0
+    assert component.world_model.is_initial(component.belief_state), \
+        "Can only mix unused infradistributions"
+assert all(
+    type(c.world_model) == type(components[0].world_model)
+    for c in components
+), "All components must share the same WorldModel type (same observation space)"
 ...
-return cls(new_measures)
+return cls(new_measures, world_model=components[0].world_model)
```

---

### `glue` as a static method

```diff
-def glue(value: float, event: int, reward_function: np.ndarray) -> np.ndarray:
-    reward_function = reward_function.copy()
-    reward_function[event] = value
-    return reward_function

 class Infradistribution:
+    @staticmethod
+    def _glue(value: float, event: int, reward_function: np.ndarray) -> np.ndarray:
+        """Gluing operator: value *^event reward_function (Definition 11)."""
+        reward_function = reward_function.copy()
+        reward_function[event] = value
+        return reward_function
```

---

### Changes to `ibrl/agents/infrabayesian.py`

The current agent constructs hypotheses internally from `num_hypotheses`, generating a uniform grid of Bernoulli priors. The new design makes hypotheses **explicit**: a list of `Infradistribution` objects and an optional prior. All hypotheses must share the same world model type.

```python
# Before — implicit, hardcoded Bernoulli grid
InfraBayesianAgent(num_actions=2, num_hypotheses=5)

# After — explicit hypotheses, optional non-uniform prior
InfraBayesianAgent(
    num_actions=2,
    hypotheses=[
        Infradistribution([AMeasure.pure(np.array([0.9, 0.1]))], world_model=BernoulliWorldModel()),
        Infradistribution([AMeasure.pure(np.array([0.5, 0.5]))], world_model=BernoulliWorldModel()),
        Infradistribution([AMeasure.pure(np.array([0.1, 0.9]))], world_model=BernoulliWorldModel()),
    ],
    prior=np.array([0.2, 0.6, 0.2]),
    reward_function=np.array([0., 1.])
)

# KU between two worlds, mixed with a third under a prior
InfraBayesianAgent(
    num_actions=2,
    hypotheses=[
        Infradistribution.mixKU([
            Infradistribution([AMeasure.pure(np.array([0.2, 0.8]))], world_model=BernoulliWorldModel()),
            Infradistribution([AMeasure.pure(np.array([0.8, 0.2]))], world_model=BernoulliWorldModel()),
        ]),
        Infradistribution([AMeasure.pure(np.array([0.5, 0.5]))], world_model=BernoulliWorldModel()),
    ],
    prior=np.array([0.5, 0.5])
)
```

**New constructor signature:**

```diff
 def __init__(self, *args,
-        num_hypotheses: int = 5,
+        hypotheses: list[Infradistribution],
+        prior: np.ndarray | None = None,
+        reward_function: np.ndarray | None = None,
         **kwargs):
     super().__init__(*args, **kwargs)
+    assert len(hypotheses) > 0
+    assert all(type(h.world_model) == type(hypotheses[0].world_model) for h in hypotheses), \
+        "All hypotheses must share the same WorldModel type"
+    self.hypotheses = hypotheses
+    self.prior = prior if prior is not None else np.ones(len(hypotheses)) / len(hypotheses)
+    self.reward_function = reward_function if reward_function is not None else np.array([0., 1.])
-    self.num_hypotheses = num_hypotheses
-    self.hypotheses = np.stack([...], axis=-1)
-    self.reward_function = np.array([0., 1.])
```

**`reset()` — hypothesis templates are never mutated, so mix directly:**

```diff
 def reset(self):
     super().reset()
-    self.dists = []
-    for _ in range(self.num_actions):
-        infradistributions = [Infradistribution([AMeasure.pure(self.hypotheses[i])])
-                               for i in range(self.num_hypotheses)]
-        coefficients = np.ones(self.num_hypotheses) / self.num_hypotheses
-        self.dists.append(Infradistribution.mix(infradistributions, coefficients))
+    self.dists = [
+        Infradistribution.mix(self.hypotheses, self.prior)
+        for _ in range(self.num_actions)
+    ]
```

The hypothesis `Infradistribution` objects are never modified by `mix` or by learning — only `self.dists` is updated. The `is_initial` check in `mix` guards against accidental mutation.

**`update` passes full `Outcome`:**

```diff
 def update(self, probabilities, action: int, outcome: Outcome) -> None:
     super().update(probabilities, action, outcome)
-    assert outcome.outcome is not None
-    self.dists[action].update(self.reward_function, outcome.outcome)
+    self.dists[action].update(self.reward_function, outcome)
```

**`dump_state`:**

```diff
-state = "["+",".join(dump_array(dist.history, "%d") for dist in self.dists)+"]"
+state = "["+",".join(str(dist.belief_state) for dist in self.dists)+"]"
```

**Convenience constructor preserving the old Bernoulli grid default:**

```python
@classmethod
def bernoulli_grid(cls, num_actions: int, num_hypotheses: int = 5,
                   prior: np.ndarray | None = None, **kwargs) -> InfraBayesianAgent:
    """Convenience: uniform grid of Bernoulli hypotheses, matching old default behaviour."""
    wm = BernoulliWorldModel()
    hypotheses = [
        Infradistribution([AMeasure.pure(np.array([1-p, p]))], world_model=wm)
        for p in np.linspace(0., 1., num_hypotheses)
    ]
    return cls(num_actions=num_actions, hypotheses=hypotheses, prior=prior, **kwargs)
```

This also makes the relationship to `DiscreteBayesianAgent` explicit: it is `InfraBayesianAgent.bernoulli_grid` where each hypothesis has a single AMeasure and no KU.

---

## Behavioral Anchors

Write these tests first against the current branch code to establish what must remain true after the refactor. They check observable behavior only — not internal field names — so they pass before and after without modification.

### `tests/test_ib_world_model.py`

```python
import numpy as np
import pytest
from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.outcome import Outcome


def make_outcome(event: int, reward: float = None) -> Outcome:
    """Helper: construct an Outcome for a discrete event."""
    return Outcome(reward=reward if reward is not None else float(event), outcome=event)


def make_dist(num_hypotheses=5):
    """Helper: uniform prior over linspace Bernoulli hypotheses."""
    hypotheses = np.stack([
        np.linspace(1., 0., num_hypotheses),
        np.linspace(0., 1., num_hypotheses)
    ], axis=-1)
    components = [Infradistribution([AMeasure.pure(hypotheses[i])]) for i in range(num_hypotheses)]
    coefficients = np.ones(num_hypotheses) / num_hypotheses
    return Infradistribution.mix(components, coefficients)


REWARD = np.array([0., 1.])


def test_pure_measure_expected_value():
    """Pure AMeasure with p=[0.3, 0.7] gives expected_value = 0.7."""
    dist = Infradistribution([AMeasure.pure(np.array([0.3, 0.7]))])
    assert abs(dist.expected_value(REWARD) - 0.7) < 1e-6


def test_expected_value_in_unit_interval():
    """expected_value is always in [0, 1] for reward_function in [0, 1]."""
    dist = make_dist()
    assert 0. <= dist.expected_value(REWARD) <= 1.


def test_e_h_zero_is_zero():
    """E_H([0, 0]) == 0 at initialization (normalization condition)."""
    dist = make_dist()
    assert abs(dist.expected_value(np.array([0., 0.]))) < 1e-9


def test_e_h_zero_stays_zero_after_updates():
    """E_H([0, 0]) == 0 is preserved after updates (normalization invariant)."""
    dist = make_dist()
    for event in [1, 0, 1, 1, 0]:
        dist.update(REWARD, make_outcome(event))
    assert abs(dist.expected_value(np.array([0., 0.]))) < 1e-9


def test_update_increases_expected_value_after_rewards():
    """Observing outcome=1 repeatedly shifts expected value upward."""
    dist = make_dist()
    ev_before = dist.expected_value(REWARD)
    for _ in range(20):
        dist.update(REWARD, make_outcome(1))
    ev_after = dist.expected_value(REWARD)
    assert ev_after > ev_before


def test_update_decreases_expected_value_after_no_rewards():
    """Observing outcome=0 repeatedly shifts expected value downward."""
    dist = make_dist()
    for _ in range(5):
        dist.update(REWARD, make_outcome(1))
    ev_before = dist.expected_value(REWARD)
    for _ in range(20):
        dist.update(REWARD, make_outcome(0))
    ev_after = dist.expected_value(REWARD)
    assert ev_after < ev_before


def test_all_scale_and_offset_nonnegative():
    """scale and offset remain non-negative after updates."""
    dist = make_dist()
    for event in [1, 0, 1, 1, 0, 0, 1]:
        dist.update(REWARD, make_outcome(event))
    for m in dist.measures:
        assert m.scale >= 0.
        assert m.offset >= 0.


def test_single_hypothesis_pure_measure():
    """Single pure hypothesis: expected value starts at hypothesis probability."""
    p = np.array([0.4, 0.6])
    dist = Infradistribution([AMeasure.pure(p)])
    assert abs(dist.expected_value(REWARD) - 0.6) < 1e-6
    for _ in range(5):
        dist.update(REWARD, make_outcome(1))
    assert 0. <= dist.expected_value(REWARD) <= 1.


def test_mix_of_two_pure_measures():
    """Classical mixture: worst-case expected value equals the more pessimistic hypothesis."""
    p_low  = np.array([0.8, 0.2])
    p_high = np.array([0.2, 0.8])
    d_low  = Infradistribution([AMeasure.pure(p_low)])
    d_high = Infradistribution([AMeasure.pure(p_high)])
    dist = Infradistribution.mix([d_low, d_high], np.array([0.5, 0.5]))
    ev = dist.expected_value(REWARD)
    assert abs(ev - 0.2) < 1e-6
```

---

## What This Unlocks

After this refactor, adding support for a new model family requires:
1. A new `WorldModel` subclass implementing `initial_state`, `update_state`, `is_initial`, `compute_likelihood`, `compute_expected_reward`
2. No changes to `AMeasure`, `Infradistribution`, or the agent

Concretely:
- **Gaussian bandit**: `GaussianWorldModel` with Welford state; `compute_likelihood` uses Gaussian posterior predictive
- **Non-binary discrete outcomes**: `BernoulliWorldModel(num_outcomes=3)` works immediately
- **Newcomb**: `NewcombWorldModel` with reward matrix state — also requires `compute_likelihood` to accept a policy argument, which is a separate PR

---

## Order of Work

1. Write `tests/test_ib_world_model.py` and confirm all tests pass on the current branch
2. Add `world_model.py` with `WorldModel[S]` (generic), `BernoulliWorldModel` (with `compute_likelihood`, `compute_expected_reward`, `_predictive`)
3. Refactor `AMeasure`: remove `compute_probabilities`/`_compute_probabilities`, update `compute_likelihood` and `expected_value` to delegate to a passed-in `world_model`
4. Refactor `Infradistribution`: rename `history` → `belief_state`, add `world_model`, thread it through all `measure.*` calls, update `update`/`expected_value`/`mix`/`mixKU`
5. Move `glue` to `Infradistribution._glue` static method
6. Update `InfraBayesianAgent`: replace `num_hypotheses` with explicit `hypotheses`/`prior`, add `bernoulli_grid` classmethod, pass `Outcome` to `infradist.update`
7. Confirm all anchor tests still pass
8. Confirm `DiscreteBayesianAgent` equivalence still holds via `bernoulli_grid`
