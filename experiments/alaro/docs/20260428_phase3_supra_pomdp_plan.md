# Phase 3 Implementation Plan: `SupraPOMDPWorldModel` (Point-Valued T)

**Date**: 2026-04-28
**Branch base**: `fllor/ib-features` (Phases 1+2 complete)
**Target**: minimal viable infra-Bayesian RL agent for stateful environments
**Architecture decision rationale**: `20260427_phase3_supra_pomdp_research.md`
**Phase 3 sketch this supersedes**: `20260418_beliefs_refactor_plan.md` §"Phase 3"

---

## 0. Scope and design commitments

This plan adds one concrete `WorldModel` subclass — `SupraPOMDPWorldModel` —
that lets the existing `Infradistribution` and `InfraBayesianAgent` machinery
operate on stateful (multi-step, latent-state) environments. After this
phase, the agent is capable of solving arbitrary causal/pseudocausal
hypotheses representable as point-valued POMDPs, including but not limited
to: multi-step Newcomblike problems, robust gridworlds, partially-observed
MDPs with model uncertainty.

Throughout this document, S denotes the finite latent state space and |S| its cardinality; A denotes the 
finite action space the agent chooses from; O denotes the finite observation space the agent receives at
each step. A supra-POMDP hypothesis is specified by four objects: an initial-state distribution Θ₀ ∈ ΔS
giving Θ₀[s] = P(s₀ = s); a transition kernel T : S × A → ΔS with T[s, a, s'] = P(s_{t+1} = s' | s_t = s,
a_t = a); an observation kernel B : S → ΔO with B[s, o] = P(o_t = o | s_t = s) (drawn from the
post-transition state, per the textbook POMDP convention); and a reward function R : S × A × S → ℝ with
R[s, a, s'] the scalar reward for transitioning from s to s' under action a. The agent maintains a belief
state b ∈ ΔS — a probability distribution over latent states — which is updated each step via the standard
Bayesian filter. A policy π ∈ ΔA is a (here, stochastic) action distribution. γ ∈ [0, 1) is the
infinite-horizon discount factor used in value iteration. When multiple hypotheses are combined into a
credal mixture, each component k carries its own (T_k, B_k, Θ₀_k, R_k) and a prior weight w_k; we write b_k
for the per-component belief and reserve H for the Infradistribution-level credal set comprising all
components.

**Design commitments** (settled in conversation 2026-04-28):

1. **T is policy-independent**: `T : S × A → ΔS`. Policy enters only through
   the agent's action selection, not through the kernel itself.
2. **T is stationary**: one shared transition matrix across timesteps. No
   `[H]` time-indexing.
3. **Observation model `B : S → ΔO`** (POMDP textbook convention,
   "Convention 1"): observation is drawn from the post-transition state.
4. **Reward function `R : S × A × S → ℝ`** as a fourth element of `params`,
   following the classical RL convention. The agent-level
   `reward_function` (used by `_glue` for normalisation) stays unchanged.
5. **Planning is infinite-horizon value iteration** with discount γ, run
   inside `compute_expected_reward`. The agent is no longer myopic.
6. **Belief state is a dense numpy vector** of shape `(|S|,)`.
7. **Existing world models stay unchanged**: `MultiBernoulliWorldModel`
   and `NewcombWorldModel` are untouched.
8. **Strict scope**: causal/pseudocausal hypotheses only. Acausal (perfect
   Transparent Newcomb) and per-step adversarial (Appel-Kosoy RMDP) regimes
   are explicitly out of scope; documented as known limitations.

---

## 1. The supra-POMDP, in one paragraph

A supra-POMDP factors a single hypothesis through a latent state space `S`
into four components: `Θ₀ ∈ ΔS` (initial-state distribution),
`T : S × A → ΔS` (transition kernel), `B : S → ΔO` (observation kernel),
and `R : S × A × S → ℝ` (reward function). The agent maintains a *belief*
— a distribution over `S` — and updates it via the standard POMDP Bayesian
filter on each (action, observation) pair: predict via `T`, condition via
`B`. The IB part comes from having a *credal mixture* of such POMDPs in
`Infradistribution.mix`; the agent acts to maximise the worst-case expected
return over the mixture. Each individual POMDP inside the mixture is
ordinary (point-valued); the Knightian uncertainty lives across hypotheses.

**Classical RL parallel**: this is an ordinary POMDP, with belief filter
identical to Kaelbling-Littman-Cassandra (1998). What differs is (a) the
agent maintains *multiple* such POMDP hypotheses with a credal min over
them at planning time, and (b) the Bellman backup in `compute_expected_reward`
is wrapped inside an a-measure for IB-style update consistency.

---

## 2. File-by-file plan

### 2.1 Move `world_model.py` into the existing `world_models/` package

The package `ibrl/infrabayesian/world_models/` already exists and already
contains `bernoulli_world_model.py` (defining `MultiBernoulliWorldModel`)
and `newcomb_world_model.py` (defining `NewcombWorldModel`). The only
structural change needed is to move the abstract base class out of
`ibrl/infrabayesian/world_model.py` into the package as `base.py`, and to
add the new `supra_pomdp_world_model.py` alongside the existing concrete
classes.

```diff
- ibrl/infrabayesian/world_model.py                           # contains abstract WorldModel
+ ibrl/infrabayesian/world_models/base.py                     # abstract WorldModel (moved from above)
+ ibrl/infrabayesian/world_models/supra_pomdp_world_model.py  # NEW (defines SupraPOMDPWorldModel)
  ibrl/infrabayesian/world_models/__init__.py                 # update imports
  ibrl/infrabayesian/world_models/bernoulli_world_model.py    # update import of WorldModel
  ibrl/infrabayesian/world_models/newcomb_world_model.py      # update import of WorldModel
```

Filename convention follows the existing two concrete classes
(`*_world_model.py`).

`__init__.py` re-exports everything for backwards compatibility:

```python
from .base import WorldModel
from .bernoulli_world_model import MultiBernoulliWorldModel
from .newcomb_world_model import NewcombWorldModel
from .supra_pomdp_world_model import SupraPOMDPWorldModel

__all__ = ["WorldModel", "MultiBernoulliWorldModel",
           "NewcombWorldModel", "SupraPOMDPWorldModel"]
```

**Backwards-compat check**: the two existing concrete classes import
`WorldModel` from `..world_model` (the file we're moving). Update those
imports to `from .base import WorldModel`. Run unit tests to confirm
nothing else in the codebase depended on the old
`ibrl.infrabayesian.world_model` import path; if any test or external
caller still uses it, leave a one-line shim
`ibrl/infrabayesian/world_model.py` that does
`from .world_models.base import WorldModel`. Otherwise delete it.

### 2.2 New: `ibrl/infrabayesian/world_models/supra_pomdp.py`

The substantive new code. Below is the full pseudocode for the new class,
broken into sections with plain-English commentary and classical-RL
parallels. Diff syntax uses `+` because every line is new.

#### 2.2.1 The class signature and `__init__`

```python
+class SupraPOMDPWorldModel(WorldModel):
+    """
+    World model for a stateful POMDP.
+
+    Belief state: np.ndarray of shape (|S|,) — distribution over latent
+        states. Initial belief state is Θ₀ from params.
+
+    Hypothesis params:
+        Single hypothesis: tuple (T, B, theta_0, R) where
+            T:       shape (|S|, |A|, |S|) — T[s, a, s'] = P(s' | s, a)
+            B:       shape (|S|, |O|)      — B[s, o]    = P(o | s)
+            theta_0: shape (|S|,)          — Θ₀[s]      = P(s_0 = s)
+            R:       shape (|S|, |A|, |S|) — R[s, a, s'] = scalar reward
+
+        Mixed params (after Infradistribution.mix): list of (params, prior_weight)
+            pairs, one per input hypothesis. Weights sum to 1.
+
+    Outcomes carry the realised observation in `Outcome.observation`
+    (added to the Outcome dataclass — see §2.3).
+
+    Construct params via make_params(T, B, theta_0, R).
+    """
+
+    def __init__(self, num_states: int, num_actions: int, num_obs: int,
+                 discount: float = 0.95, value_iter_tol: float = 1e-6,
+                 value_iter_max: int = 1000):
+        self.num_states = num_states
+        self.num_actions = num_actions
+        self.num_obs = num_obs
+        self.discount = discount
+        self.value_iter_tol = value_iter_tol
+        self.value_iter_max = value_iter_max
```

**Plain English**: We store the dimensions of the state, action, and
observation spaces, plus three planning hyperparameters: a discount factor
γ (so we can do infinite-horizon value iteration with bounded value
function), a tolerance for the value-iteration fixed-point check, and a
hard iteration cap so we never loop forever. We also keep a value-function
cache (introduced below).

**Classical RL parallel**: γ is the standard discount in
$V^\pi(s) = \mathbb{E}[\sum_t γ^t R_t]$. `value_iter_tol` and
`value_iter_max` are the Bellman-residual stopping criterion used in any
textbook value-iteration loop (Sutton & Barto §4.4).

**On caching value iteration**: in classical RL you solve value iteration
*once* per known MDP and reuse `V` across every step until the model or
policy changes. The naïve version of `compute_expected_reward` re-runs
value iteration on every call — that's once per (action, hypothesis
component) on every agent decision, ~1000× more compute than necessary
for `|S|=16, |A|=4, K=16`. We cache: `V_k(π)` only depends on
`(T_k, R_k, π)`, none of which are touched by observations. We invalidate
the cache only when the policy changes.

The IB-ness of the agent is *not* affected by caching. Each component's
value iteration is ordinary single-MDP value iteration on `(T_k, R_k, π)`
— no IB-specific modification. The IB part lives in (a) the
posterior-credal-weighting in `_posterior_weights`, and (b) the worst-case
min in `Infradistribution.evaluate_action` — both *outside* the value
function. So the planning algorithm is literally the textbook one;
caching it doesn't sacrifice any IB structure.

```diff
+    def __init__(self, num_states: int, num_actions: int, num_obs: int,
+                 discount: float = 0.95, value_iter_tol: float = 1e-6,
+                 value_iter_max: int = 1000):
+        # ... fields above ...
+        # Cache: maps (id(component_params), hash(policy_bytes)) → V vector.
+        # Invalidated when params identity or policy bytes change.
+        self._v_cache: dict = {}
```

#### 2.2.2 `make_params` and `event_index`

```python
+    def make_params(self, T: np.ndarray, B: np.ndarray,
+                    theta_0: np.ndarray, R: np.ndarray):
+        """Validate shapes and stochasticity, then bundle into a tuple."""
+        assert T.shape == (self.num_states, self.num_actions, self.num_states)
+        assert B.shape == (self.num_states, self.num_obs)
+        assert theta_0.shape == (self.num_states,)
+        assert R.shape == (self.num_states, self.num_actions, self.num_states)
+        assert np.allclose(T.sum(axis=2), 1), "T rows must sum to 1"
+        assert np.allclose(B.sum(axis=1), 1), "B rows must sum to 1"
+        assert np.isclose(theta_0.sum(), 1), "theta_0 must sum to 1"
+        return (T, B, theta_0, R)
+
+    def event_index(self, outcome: Outcome) -> int:
+        """Map an outcome to its discrete observation index."""
+        return outcome.observation
```

**Plain English**: `make_params` is the constructor for a single
hypothesis's parameters: it validates that the kernels are proper
stochastic matrices and bundles them. `event_index` extracts the discrete
observation from an Outcome, used by `_glue` (the operator that builds
extremal reward functions) and by the belief filter.

**Classical RL parallel**: `make_params` is roughly "specify a POMDP".
`event_index` is the trivial identity in standard POMDPs where the
observation index is already an integer.

#### 2.2.3 `mix_params`

```python
+    def mix_params(self, params_list: list, coefficients: np.ndarray):
+        """
+        Mix point-valued POMDP hypotheses into a credal mixture.
+
+        Returns a list of (params, prior_weight) tuples — one per input.
+        Flattens nested mixtures (an input that is itself a mixture is
+        unpacked). The prior weights become the initial credal weights
+        and are folded into the belief filter via _posterior_weights.
+        """
+        components = []
+        for p, c in zip(params_list, coefficients):
+            if isinstance(p, list):  # already a mixture, flatten
+                components.extend((cp, w * float(c)) for cp, w in p)
+            else:
+                components.append((p, float(c)))
+        total = sum(w for _, w in components)
+        return [(cp, w / total) for cp, w in components]
```

**Plain English**: When `Infradistribution.mix` combines multiple
hypotheses, we need to fuse their `params` into a single object that
tracks all of them as a *list of point POMDPs with prior weights*. We do
not collapse the POMDPs into one averaged kernel — that would lose the
credal structure. Each component keeps its full (T, B, Θ₀, R) intact, and
its prior weight rides alongside.

**Classical RL parallel**: Bayesian model averaging would average kernels
weighted by posterior to get a single effective POMDP, then plan in it.
We deliberately do *not* do this: we keep every POMDP separate and let
the credal min do the worst-case work. This matches the
`NewcombWorldModel`'s mix_params shape exactly.

#### 2.2.4 `initial_state` and `is_initial`

```python
+    def initial_state(self):
+        """
+        Belief state at t=0 is None — we lazily initialise on first
+        update to a per-hypothesis belief vector. Reasoning: for a
+        mixture, the belief is *per-component* (each component POMDP
+        has its own belief), and we don't know how many components
+        until params is bound to the Infradistribution.
+        """
+        return None
+
+    def is_initial(self, state) -> bool:
+        return state is None
```

**Plain English**: We can't construct the actual initial belief state
until we know how many hypotheses there are, because the belief state for
a *mixed* hypothesis is a list of one belief vector per component. We
return `None` and lazily expand on the first `update_state`.

**Classical RL parallel**: A standard POMDP would just return Θ₀ here.
The lazy init is purely a side effect of supporting credal mixtures: we
need belief state per component, not per hypothesis.

#### 2.2.5 The Bayesian filter — `update_state`

```python
+    def update_state(self, state, outcome: Outcome, action: int,
+                     params=None, policy: np.ndarray | None = None):
+        """
+        Standard POMDP Bayesian filter, applied per-component, with
+        per-component log-marginal-likelihood tracked alongside.
+
+        Per-component update for one POMDP (T, B, theta_0, R):
+            belief_pred[s']  = Σ_s belief[s] · T[s, action, s']
+            belief_post[s']  ∝ B[s', outcome.observation] · belief_pred[s']
+            log_marginal    += log(Σ_{s'} B[s', obs] · belief_pred[s'])
+            renormalise belief_post.
+
+        State shape: list[(belief_vec, log_marginal_scalar)], one tuple
+        per hypothesis component. The log-marginal is the running
+        log P(o_1, ..., o_t | component k); it feeds _posterior_weights
+        to compute credal weights over components.
+
+        Components are filtered independently. The credal min over
+        components happens at the Infradistribution layer, not here.
+        """
+        components = params if isinstance(params, list) else [(params, 1.0)]
+        if state is None:
+            state = [(self._initial_belief(c[0]), 0.0) for c in components]
+        new_state = []
+        obs = outcome.observation
+        for (belief, log_m), (component_params, _w) in zip(state, components):
+            T, B, _theta_0, _R = component_params
+            belief_pred = belief @ T[:, action, :]
+            belief_post = B[:, obs] * belief_pred
+            total = belief_post.sum()
+            if total <= 0:
+                # zero-probability observation under this component:
+                # keep belief unchanged and push log_marginal toward -inf
+                # so the credal weight handles the elimination.
+                new_state.append((belief.copy(), log_m + np.log(1e-300)))
+            else:
+                new_state.append((belief_post / total, log_m + np.log(total)))
+        return new_state
+
+    def _initial_belief(self, component_params):
+        _T, _B, theta_0, _R = component_params
+        return theta_0.copy()
```

**Plain English**: This is the core POMDP filtering equation, applied
once per hypothesis component. For each component we (i) propagate the
belief forward through the action — multiplying by T to get the predicted
distribution over the new state — then (ii) condition on the observation
by multiplying pointwise by `B[:, obs]` and renormalising. Components are
filtered independently; the IB credal weighting across components is
handled separately, in the likelihood and posterior-weight computation,
to keep the per-component belief filter mathematically clean.

**Classical RL parallel**: Exact Kaelbling-Littman-Cassandra belief update,
$b'(s') = \eta \cdot O(o | s') \cdot \sum_s T(s' | s, a) \cdot b(s)$,
identical line-for-line. The only deviation is the per-component loop,
which a classical POMDP wouldn't have because there is one model.

#### 2.2.6 The credal weight tracker — `_posterior_weights`

```python
+    def _posterior_weights(self, state, params) -> np.ndarray:
+        """
+        Posterior credal weights over hypothesis components:
+            weights[k] ∝ prior_weight[k] · P(history | component k).
+
+        Reads the per-component log-marginal-likelihood that
+        update_state has been accumulating, combines it with the prior
+        weights from params, and renormalises in log-space for stability.
+        """
+        log_marginals = np.array([lm for (_b, lm) in state])
+        prior_log_w = np.log([w for _, w in params])
+        log_w = prior_log_w + log_marginals
+        log_w -= log_w.max()
+        w = np.exp(log_w)
+        return w / w.sum()
```

**Plain English**: For each component POMDP we track the cumulative
log-likelihood `log P(o₁, …, oₜ | component k)` — the running
"how well does this hypothesis explain the data so far". Combining
this with the prior weight and renormalising gives the posterior weight
over hypothesis components. This is the credal-set bookkeeping that
turns a classical Bayesian model average into an IB credal mixture: we
keep all components alive (none are pruned), but their weights
reallocate based on the data.

**Classical RL parallel**: Standard Bayesian model selection tracks
exactly this same log-marginal-likelihood per model. The difference: a
Bayesian agent computes a posterior and *averages* using it; we compute
the same posterior and *credal-min* over it. Same intermediate quantity,
different downstream operation.

#### 2.2.7 The likelihood — `compute_likelihood`

```python
+    def compute_likelihood(self, belief_state, outcome, params,
+                           action: int, policy=None) -> float:
+        """
+        P(observation | belief, action) marginalised over latent state and
+        averaged over hypothesis components by current posterior weights.
+
+            P(o | b, a) = Σ_k weights[k] · Σ_{s,s'} b_k[s] · T_k[s,a,s'] · B_k[s', o]
+
+        Used by Infradistribution.update to rescale a-measure scales.
+        """
+        components = params if isinstance(params, list) else [(params, 1.0)]
+        if belief_state is None:
+            beliefs = [(self._initial_belief(p), 0.0) for (p, _w) in components]
+        else:
+            beliefs = belief_state
+        weights = self._posterior_weights(beliefs, components)
+        obs = outcome.observation
+        total = 0.0
+        for (belief, _lm), (component_params, _w), weight in zip(
+            beliefs, components, weights
+        ):
+            T, B, _, _ = component_params
+            belief_pred = belief @ T[:, action, :]
+            total += weight * float((belief_pred @ B[:, obs]))
+        return total
```

**Plain English**: To compute the likelihood of seeing observation `o`
after taking action `a` from the current belief, we (i) push the belief
forward through T to get the predicted state distribution, (ii) take the
inner product with `B[:, o]` to get the marginal probability of the
observation, (iii) average across hypothesis components weighted by the
current credal posterior. This scalar is what `Infradistribution.update`
multiplies into each a-measure's λ, driving the IB rescaling.

**Classical RL parallel**: A classical POMDP doing model selection would
compute exactly this same quantity per model and use it as the
multiplicative weight in a recursive Bayesian update. The piece that's
new is the marginalisation over the credal set, but it's a sum, so the
math is identical.

#### 2.2.8 The expected reward — `compute_expected_reward`

This is the substantive RL part. Pseudocode:

```python
+    def compute_expected_reward(self, belief_state, reward_function: np.ndarray,
+                                params, action: int, policy=None) -> float:
+        """
+        Worst-case expected return at the current belief state for a
+        single (action, policy) decision, evaluated by:
+
+        1. For each hypothesis component, run value iteration on the
+           component's POMDP-belief MDP under the current policy π.
+           This gives V_k(b) for each component k and current belief b.
+        2. Take the action: Q_k(b, a) = Σ_{s,s',o} b[s] · T_k[s,a,s'] ·
+           B_k[s', o] · (R_k[s, a, s'] + γ · V_k(b'(b, a, o))).
+        3. Average across components by posterior credal weights.
+
+        The credal min over components happens at the
+        Infradistribution layer (min over a-measures); each call here
+        returns one a-measure's expected reward, *not* the worst-case.
+        """
+        components = params if isinstance(params, list) else [(params, 1.0)]
+        if belief_state is None:
+            beliefs = [(self._initial_belief(p), 0.0) for (p, _w) in components]
+        else:
+            beliefs = belief_state
+        weights = self._posterior_weights(beliefs, components)
+
+        total = 0.0
+        policy_key = None if policy is None else policy.tobytes()
+        for (belief, _lm), (component_params, _w), credal_weight in zip(
+            beliefs, components, weights
+        ):
+            T, B, _theta_0, R = component_params
+            # one-step Q-value at this belief, action under policy π for
+            # subsequent steps. We approximate the belief-state MDP's
+            # value function V(b) by collapsing to the underlying
+            # state-MDP value V(s), then projecting V_b = b @ V_s.
+            cache_key = None if policy_key is None else (id(component_params), policy_key)
+            V_s = self._value_iteration(T, R, policy, cache_key=cache_key)
+            # Q(b, a) = Σ_{s} b[s] · Σ_{s'} T[s,a,s'] · (R[s,a,s'] + γ V[s'])
+            R_sa = (T[:, action, :] * R[:, action, :]).sum(axis=1)  # shape (|S|,)
+            EV = T[:, action, :] @ V_s                              # shape (|S|,)
+            Q_sa = R_sa + self.discount * EV                        # shape (|S|,)
+            total += credal_weight * float(belief @ Q_sa)
+        return total
+
+    def _value_iteration(self, T: np.ndarray, R: np.ndarray,
+                          policy: np.ndarray,
+                          cache_key=None) -> np.ndarray:
+        """
+        Standard infinite-horizon value iteration for an MDP under a
+        fixed stochastic policy π.
+
+        V[s] = Σ_a π[a] · Σ_{s'} T[s, a, s'] · (R[s, a, s'] + γ · V[s'])
+
+        Iterates until ||V_new - V||_∞ < tol or value_iter_max iterations.
+        Returns a vector of shape (|S|,).
+
+        Cache: if cache_key is not None, return the cached V if present.
+        cache_key is the (id(component_params), policy.tobytes()) tuple
+        constructed by the caller in compute_expected_reward.
+        """
+        if cache_key is not None and cache_key in self._v_cache:
+            return self._v_cache[cache_key]
+        if policy is None:
+            policy = np.ones(self.num_actions) / self.num_actions
+        # Pre-compute pieces that don't depend on V — both shape (|S|, |A|).
+        R_pi = (T * R).sum(axis=2)        # E_{s'}[R | s, a]
+        V = np.zeros(self.num_states)
+        for _ in range(self.value_iter_max):
+            EV_pi = T @ V                     # shape (|S|, |A|)
+            Q = R_pi + self.discount * EV_pi  # shape (|S|, |A|)
+            V_new = Q @ policy                # shape (|S|,)
+            if np.max(np.abs(V_new - V)) < self.value_iter_tol:
+                V = V_new
+                break
+            V = V_new
+        if cache_key is not None:
+            self._v_cache[cache_key] = V
+        return V
```

**Plain English (the planning section)**: The agent isn't just asking
"what reward do I get on the next step"; it's asking "what's the value of
this state, taking action a now and following π afterwards, summed over
all future steps with discount γ". Computing that requires solving the
Bellman fixed point, which we do with value iteration: a simple loop that
applies the Bellman operator to V until convergence. We do this *per
hypothesis component*, then take a posterior-weighted average — the
credal min over components happens one level up, in
`Infradistribution.evaluate_action`.

**A note on the approximation**: Strictly, the value of a *belief* in a
POMDP is not the belief-weighted average of state values
(`V(b) = b @ V_s`). The true belief-state MDP has a richer value function
because future actions can be informed by future observations, which the
state-MDP V can't model. We use the state-MDP approximation as the MVP
because it's exact for fully-observable cases (B = identity) and a known
upper bound for POMDPs in general. Documented as a known approximation in
the docstring.

**Classical RL parallel**: This is *exactly* tabular value iteration as
in Sutton & Barto §4.4, modified only by the wrapper that runs it once
per hypothesis component. The Bellman backup line is verbatim from any
RL textbook. What's new here is *not* the planning algorithm — it's that
we run it inside an a-measure and let the IB credal min produce
worst-case-over-hypotheses guarantees.

#### 2.2.9 Note on baselines for the eventual experiments

Settled in conversation 2026-04-28: the experiments report
(`20260427_ib_experiments_report.md`) will compare the IB agent against
**two** classical baselines, not one. The supra-POMDP world model is
designed so both share as much code as possible with IB:

1. **Bayesian-model-averaging agent** — "BayesianRLAgent". Same
   hypothesis class, same `SupraPOMDPWorldModel`, same belief filter,
   same per-hypothesis value iteration (so the V cache and code path
   above are reused verbatim). The only divergence from IB is at the
   evaluation step: where IB takes the **min over a-measures** in
   `Infradistribution.evaluate_action`, the Bayesian baseline takes the
   **posterior-weighted average over hypothesis components** of each
   component's expected reward. Concretely this is a thin
   `BayesianAgent` class that holds the same `Infradistribution`
   internals but defines its own `evaluate_action` and `get_probabilities`
   methods. This isolates IB's *distinctive* contribution (worst-case
   credal min) from everything else.
2. **Tabular Q-learning with ε-greedy** — "QLearningAgent". The textbook
   model-free reference. No hypothesis class, no belief filter, no value
   iteration: a Q-table over (observation, action) updated by TD bootstrap.
   Standard ε-greedy exploration. This is the baseline a reviewer will
   expect to see, and it shows the cost of being model-free.

**Exploration**: IB's pessimism *is* the exploration policy — the credal
min discourages actions whose worst-case is bad, naturally producing
information-seeking behaviour in well-designed environments. The Bayesian
baseline above is greedy on its posterior expected value (no separate
exploration noise, mirroring IB's lack of one). Q-learning uses ε-greedy
because that's its convention. We deliberately do *not* add Thompson
sampling / PSRL as a third baseline: scope creep, and "Bayesian averaging"
already captures the model-based-Bayesian-RL design space the academic
community will expect to see.

**Implementation note**: the Bayesian baseline is a separate agent class
(~1–2 days of work) but reuses `SupraPOMDPWorldModel`. It is **not** part of this plan's required
deliverables — it's listed here so the supra-POMDP design accommodates
it without future refactor. The plan for `BayesianRLAgent` and
`QLearningAgent` belongs in a separate doc; both are tracked as
follow-ups in the experiments report's roadmap.

### 2.3 Rename `Outcome.env_action` → `Outcome.observation`

The existing `Outcome.env_action` field is the discrete signal the agent
receives from the environment each step (currently: the predictor's
prediction, in Newcomb-family environments). This is *exactly* what
POMDP convention calls an observation. The name `env_action` is a
hangover from when Newcomb was the only multi-channel environment; it
now misleads. We rename it to `observation`.

**Audit of existing uses** (found via `grep -rn env_action`, 2026-04-28):

- `ibrl/outcome.py` — field definition (1 site).
- `ibrl/environments/base.py`, `bandit.py`, `bernoulli_bandit.py`,
  `switching.py`, `base_newcomb_like.py` — environment internals
  (5–6 sites).
- `ibrl/environments/base.py` — produces `Outcome(reward=..., env_action=...)`
  (1 site).
- `ibrl/agents/base.py` — docstring only (1 site).
- `tests/test_infrabayesian_beliefs.py`, `test_knightian_uncertainty.py`,
  `test_agents.py` — test fixtures (~14 sites).
- `infrabayesian/world_models/bernoulli_world_model.py`,
  `newcomb_world_model.py` — **do not read `env_action`**. Both compute
  `event_index` from `outcome.reward` and `action`. The IB stack does
  not currently consume `env_action` at all.

**Migration**: hard rename, no compat alias.

```diff
 # ibrl/outcome.py
 @dataclass
 class Outcome:
     reward: float
-    env_action: int | None = None
+    observation: int | None = None
```

Update every other reference (~28 sites total) in the same commit. The
`_resolve(self, env_action, action)` *parameter* names inside environment
classes can be renamed too (`_resolve(self, observation, action)`) — that
is the same concept under a clearer name. After the rename, run the full
unit-test suite to confirm no callers we missed.

**Plain English**: One field, two consistent names across the codebase.
`observation` is the discrete signal index the agent receives each step.
For environments that don't have a separate observation channel beyond
reward (Bernoulli bandits), this stays None — and `event_index` continues
to derive its value from `reward` and `action`, exactly as today.

**Classical RL parallel**: An RL trajectory is a sequence of (state,
action, reward, observation) tuples (or for fully-observable MDPs, the
observation is just the state). The new `observation` field aligns the
codebase with this convention. The supra-POMDP world model is the first
world model that actually reads this field — for Bernoulli and Newcomb
world models, the observation field is set by the environment but unused
by the IB stack.

### 2.4 No changes required to `AMeasure`, `Infradistribution`, `InfraBayesianAgent`

Phase 1 made `params` and `belief_state` opaque to these classes by
design. The pseudocode above respects that:

- `AMeasure.evaluate_action` already calls `world_model.compute_expected_reward`
  with opaque `params` — works unchanged.
- `Infradistribution.update` already calls `world_model.update_state` and
  `world_model.compute_likelihood` with opaque arguments — works unchanged.
- `Infradistribution.mix` already calls `world_model.mix_params` and
  `world_model.is_initial` — works unchanged.
- `InfraBayesianAgent.get_probabilities` calls `evaluate_action` per
  action; the LP optimiser doesn't care about the world model.

The agent supports supra-POMDP hypotheses by being constructed with them:

```python
wm = SupraPOMDPWorldModel(num_states=16, num_actions=4, num_obs=16,
                          discount=0.95)
hypotheses = [
    Infradistribution([AMeasure(wm.make_params(T_k, B_k, theta_0, R_k))],
                       world_model=wm)
    for k in range(num_hypotheses)
]
agent = InfraBayesianAgent(num_actions=4, hypotheses=hypotheses,
                           prior=np.ones(num_hypotheses) / num_hypotheses,
                           reward_function=...)
```

The `reward_function` argument to `InfraBayesianAgent` retains its
existing role (used by `_glue` for normalisation in
`Infradistribution.update`). It must be consistent with the supra-POMDP's
`R` in the sense that `reward_function[a]` is a vector indexed by
observation that bounds the achievable rewards in [0, 1] — needed for
the IB normalisation theorem. In practice, set `reward_function = np.ones(...)`
when the supra-POMDP carries the real reward in `R`.

---

## 3. Tier 1 — `SupraPOMDPWorldModel` unit tests

File: `tests/test_supra_pomdp_world_model.py`. Each test exercises one
piece of the world model in isolation, with a hand-crafted POMDP.

### 3.1 Param construction and validation

```python
def test_make_params_validates_shapes():
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    T = np.array([[[0.9, 0.1], [0.5, 0.5]],
                   [[0.5, 0.5], [0.1, 0.9]]])
    B = np.eye(2)
    theta_0 = np.array([1.0, 0.0])
    R = np.zeros((2, 2, 2))
    wm.make_params(T, B, theta_0, R)  # must not raise

def test_make_params_rejects_nonstochastic_T():
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    bad_T = np.ones((2, 2, 2))  # rows sum to 2, not 1
    with pytest.raises(AssertionError):
        wm.make_params(bad_T, np.eye(2), np.array([1., 0.]), np.zeros((2,2,2)))

def test_make_params_rejects_nonstochastic_B():
    # similar, mutate B to have row sums != 1
def test_make_params_rejects_nonstochastic_theta_0():
    # similar
```

### 3.2 Initial state

```python
def test_initial_state_is_none():
    wm = SupraPOMDPWorldModel(num_states=3, num_actions=2, num_obs=2)
    assert wm.initial_state() is None

def test_is_initial_recognises_none():
    wm = SupraPOMDPWorldModel(num_states=3, num_actions=2, num_obs=2)
    assert wm.is_initial(None)
    assert not wm.is_initial([(np.array([1., 0., 0.]), 0.0)])
```

### 3.3 Belief filter — `update_state`

These are the most important tests in Tier 1.

```python
def test_update_state_lazy_init_uses_theta_0():
    """First update_state call from None initial state uses theta_0
    as the prior belief."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # identity transitions
    B = np.eye(2)
    theta_0 = np.array([0.7, 0.3])
    R = np.zeros((2, 1, 2))
    params = wm.make_params(T, B, theta_0, R)
    state = wm.update_state(None, Outcome(reward=0., observation=0),
                              action=0, params=params)
    # observed obs=0 under identity B from (0.7, 0.3) prior:
    # belief_post[0] ∝ 1 · 0.7 = 0.7, belief_post[1] ∝ 0 · 0.3 = 0
    belief, log_m = state[0]
    np.testing.assert_allclose(belief, [1.0, 0.0])
    # log_marginal records the 0.7 likelihood of the observation
    assert np.isclose(log_m, np.log(0.7))

def test_update_state_pomdp_filter_textbook_example():
    """Standard tiger-problem-like setup: two states, listening
    observation noisy. Verify against hand-computed posterior."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    # listen action: state stays the same
    T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    # listen observation: 85% accurate
    B = np.array([[0.85, 0.15], [0.15, 0.85]])
    theta_0 = np.array([0.5, 0.5])
    R = np.zeros((2, 1, 2))
    params = wm.make_params(T, B, theta_0, R)

    # observe obs=0 (suggesting state=0)
    state = wm.update_state(None, Outcome(reward=0., observation=0),
                              action=0, params=params)
    belief, _ = state[0]
    np.testing.assert_allclose(belief, [0.85, 0.15])

    # observe obs=0 again
    state = wm.update_state(state, Outcome(reward=0., observation=0),
                              action=0, params=params)
    belief, _ = state[0]
    expected = np.array([0.85**2, 0.15**2])
    expected /= expected.sum()
    np.testing.assert_allclose(belief, expected)

def test_update_state_handles_zero_likelihood_observation():
    """If an observation has zero probability under a component,
    belief stays unchanged and log_marginal records -inf-ish."""
    # construct env where obs=0 has 0 probability from current belief,
    # verify no NaN, log_marginal goes to log(1e-300).

def test_update_state_does_not_mutate_input_state():
    """update_state must be pure."""
    # build a state, update, verify original state is bit-identical.
```

### 3.4 Mixed params and credal weights

```python
def test_mix_params_preserves_components():
    """Two single-component params mixed with weights (0.6, 0.4)
    produce a 2-component mixture."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    p1 = wm.make_params(...identity_T..., np.eye(2), np.array([1.,0.]),
                         np.zeros((2,2,2)))
    p2 = wm.make_params(...swap_T..., np.eye(2), np.array([0.,1.]),
                         np.zeros((2,2,2)))
    mixed = wm.mix_params([p1, p2], np.array([0.6, 0.4]))
    assert len(mixed) == 2
    assert np.isclose(mixed[0][1], 0.6)
    assert np.isclose(mixed[1][1], 0.4)

def test_mix_params_flattens_nested_mixtures():
    """Mixing a 2-component mixture with a 1-component mixture
    yields a 3-component mixture with weights renormalised."""
    # build mixture m1 with 2 components weights (0.5, 0.5)
    # mix(m1, p3) with coefficients (0.7, 0.3)
    # expected result: 3 components with weights (0.35, 0.35, 0.3)

def test_posterior_weights_at_initial_state_equal_prior():
    """Before any observations, posterior weights = prior weights."""
    # construct mixture, lazily init beliefs, verify weights == prior

def test_posterior_weights_shift_toward_truth_under_data():
    """Two competing components, one matches truth. After many
    observations from the true component, posterior weight on it → 1."""
    # construct two POMDPs that disagree on B; sample observations from
    # one; run update_state repeatedly; verify weight on truth → 1.
```

### 3.5 Likelihood

```python
def test_compute_likelihood_at_initial_state():
    """At initial state, likelihood = Σ_s θ_0[s] · Σ_s' T[s,a,s'] · B[s',o]."""
    # hand-compute and verify

def test_compute_likelihood_marginalises_over_credal_mixture():
    """Two-component mixture: likelihood = Σ_k w_k · L_k."""
    # verify against component-wise computation

def test_compute_likelihood_in_unit_interval():
    """For any valid params, likelihood ∈ [0, 1]."""
    # property test with random valid params
```

### 3.6 Value iteration

```python
def test_value_iteration_terminal_absorbing_state():
    """Two-state MDP with one absorbing terminal: V should be
    R_terminal / (1 - γ) at the terminal, ≤ that elsewhere."""

def test_value_iteration_uniform_policy_known_solution():
    """2-state, 2-action MDP with hand-computed V under uniform π."""

def test_value_iteration_converges_within_max_iter():
    """For γ < 1, value iteration converges in O(log(tol) / log(γ)) iters."""

def test_value_iteration_optimal_policy_higher_than_uniform():
    """Sanity: V(optimal_pi) ≥ V(uniform_pi) component-wise."""
```

### 3.7 Expected reward

```python
def test_compute_expected_reward_fully_observable_recovers_state_value():
    """When B = identity, V(b) = b @ V(s) is exact, not an approximation.
    Verify expected_reward matches the underlying state-MDP's V."""

def test_compute_expected_reward_at_initial_belief():
    """E[reward | initial belief, action a, policy π] equals
    Σ_s θ_0[s] · Q^π(s, a)."""

def test_compute_expected_reward_credal_average():
    """Mixed hypothesis: expected_reward = Σ_k posterior_weight[k] · E_k."""
```

---

## 4. Tier 2 — Integration with `Infradistribution`

File: `tests/test_supra_pomdp_integration.py`. These mirror the
`test_ib_world_model.py` Bernoulli anchor tests but built around supra-POMDP.

### 4.1 Normalisation conditions

```python
def test_e_h_zero_is_zero_supra_pomdp():
    """E_H(reward_function = 0) == 0 at initialisation."""
    dist = make_supra_pomdp_dist()
    rf = np.zeros(num_obs)
    assert abs(dist.evaluate_action(rf, action=0)) < 1e-9

def test_e_h_zero_stays_zero_after_updates():
    """After arbitrary observations, E_H(0) = 0 still holds."""
```

### 4.2 Mixing

```python
def test_mix_pessimistic_supra_pomdp():
    """Two POMDP hypotheses with different reward distributions —
    mixed evaluation equals the pessimistic component, not the average."""
    # build two POMDPs: one rewards-optimistic, one rewards-pessimistic
    # verify Infradistribution.mix produces the pessimistic value at
    # initial belief
```

### 4.3 Update rule with credal reweighting

```python
def test_update_concentrates_on_correct_hypothesis():
    """Two-hypothesis mixture, one matches truth. After repeated
    updates from the true env, posterior credal weight on truth → 1
    and the agent's evaluate_action approaches the true POMDP's V*."""

def test_update_does_not_change_behaviour_at_zero_likelihood_step():
    """If observation has 0 probability under one component, that
    component's scale → 0 but the rest of the mixture stays valid."""
```

### 4.4 Behavioural anchors analogous to Bernoulli

```python
def test_supra_pomdp_evaluate_action_in_unit_interval():
    """For reward_function in [0, 1] and any belief, evaluate_action ∈ [0, 1]."""

def test_supra_pomdp_higher_reward_after_favourable_observations():
    """In a POMDP where state 0 is high-reward, after observing
    observation 0 (which suggests state 0), evaluate_action increases."""
```

---

## 5. Tier 3 — End-to-end agent tests

File: `tests/test_supra_pomdp_agent.py`. Small fully-controlled
environments where we know the optimal policy.

### 5.1 Sanity: degenerate POMDP recovers MAB behaviour

```python
def test_supra_pomdp_with_single_state_matches_bandit():
    """A supra-POMDP with |S|=1 and B = identity should produce
    the same agent behaviour as MultiBernoulliWorldModel on the same
    underlying Bernoulli arms."""
    # construct equivalent agents, run for 100 steps, verify identical
    # action selections.
```

### 5.2 Fully-observable gridworld with known trap

```python
def test_supra_pomdp_gridworld_known_trap_avoids_it():
    """4x4 gridworld, single hypothesis with known trap location.
    Agent must reach goal without entering trap. Verify trap-hit
    rate = 0 over 50 episodes."""
```

### 5.3 Robust gridworld with credal mixture over trap location

```python
def test_supra_pomdp_gridworld_credal_avoids_all_possible_traps():
    """4x4 gridworld, 4-hypothesis mixture (4 candidate trap
    locations). Agent should avoid every cell in the credal envelope
    of trap locations — even though only one is the true trap."""
    # primary metric: trap-hit rate over 50 episodes < (1/4)*classical_baseline_rate
```

### 5.4 POMDP with noisy observation — tiger-problem-like

```python
def test_supra_pomdp_tiger_listens_before_committing():
    """Two doors, treasure behind one, tiger behind the other. A
    'listen' action gives a noisy observation of the tiger's location.
    Optimal policy: listen until belief is concentrated, then open.
    Verify the agent learns to listen first, then open the lower-belief
    door."""
```

### 5.5 Two-step Newcomb-family problem

```python
def test_supra_pomdp_transparent_newcomb_with_epsilon():
    """Transparent Newcomb with ε=0.05, encoded as a 2-state
    supra-POMDP. Verify the agent's terminal policy is one-box-on-full
    (i.e. action probability for one-box conditional on the full-box
    observation is > 0.9 after 200 episodes)."""
```

---

## 6. Open questions and known limitations

1. **Belief-state-MDP value iteration is approximated by state-MDP value
   iteration**, projected via `V(b) = b @ V_s`. Exact for fully-observable
   POMDPs; an upper-bound approximation for partially-observable ones.
   Acceptable for MVP. Improving this would be a separate effort
   (POMDP value iteration is a known hard problem; PBVI etc.).

2. **Value iteration is cached** by `(id(component_params),
   policy.tobytes())` in `self._v_cache` (see §2.2.1 and §2.2.8). The
   cache is invalidated implicitly: a new policy or a new component
   identity produces a new key. For a long-running agent with a
   stable hypothesis class this means value iteration runs essentially
   once per (component, policy) pair, not once per agent step. The
   only case where the cache misbehaves is if a caller mutates a
   `params` tuple in place — discouraged, but worth a unit test.

3. **Reward function lives inside `params` for the supra-POMDP** but
   inside the agent's `reward_function` for the other world models. This
   asymmetry is documented but not eliminated. Could be unified later by
   moving reward into all world models' params.

4. **`Outcome.env_action` has been renamed to `Outcome.observation`** —
   see §2.3. Existing `BernoulliBanditEnvironment` and `NewcombEnvironment`
   are updated as part of the rename. The supra-POMDP world model is the
   first to actually consume this field; for Bernoulli/Newcomb it is
   set-but-unused, exactly as `env_action` was.

5. **Acausal hypotheses (perfect Transparent Newcomb) and per-step
   adversarial transitions are out of scope.** Documented in the class
   docstring with a pointer to `20260427_phase3_supra_pomdp_research.md`.

6. **Baselines for the experiments report (`BayesianRLAgent`,
   `QLearningAgent`) are not part of this plan.** See §2.2.9. The
   supra-POMDP design is built so the Bayesian baseline can reuse it
   verbatim, with only `evaluate_action`/`get_probabilities` swapped to
   posterior-weighted average. Tracked as a follow-up plan.

---

## 7. Order of work

1. Move `world_model.py` → `world_models/base.py` (no behaviour change).
   Update imports in `bernoulli_world_model.py`, `newcomb_world_model.py`,
   and `world_models/__init__.py`. Run unit tests; if any external caller
   used the old import path, leave a one-line shim per §2.1.
2. Rename `Outcome.env_action` → `Outcome.observation` everywhere in the
   codebase (~28 sites; see §2.3). Run unit tests to confirm no caller
   was missed.
3. Implement an empty skeleton of `SupraPOMDPWorldModel` with the
   structure in §2.2.
4. Tier 1 unit tests (§3). All tests for one method before moving to
   the next; the dependency order is param construction → initial state
   → update_state → posterior_weights → likelihood → value_iteration →
   expected_reward.
5. Tier 2 integration tests (§4).
6. Tier 3 agent tests (§5). Start with the degenerate-POMDP sanity check
   (§5.1) before building gridworlds.
7. Commit the tests. Ensure they all fail with empty skeleton.
8. Implement `SupraPOMDPWorldModel` with the structure in §2.2,
   inclusive of the log-marginal tracking in `update_state` (§2.2.5)
   and the V cache (§2.2.1, §2.2.8). Validate that the tests now all
   pass. Explain why if they do not.

---

## 8. Summary table

| Component | New / Changed / Unchanged | Approx. LOC |
|---|---|---|
| `ibrl/infrabayesian/world_models/__init__.py` | Updated re-exports | ~5 |
| `ibrl/infrabayesian/world_models/base.py` | Moved from `world_model.py` | 0 (move only) |
| `ibrl/infrabayesian/world_models/bernoulli_world_model.py` | Update import only | 1 |
| `ibrl/infrabayesian/world_models/newcomb_world_model.py` | Update import only | 1 |
| `ibrl/infrabayesian/world_models/supra_pomdp_world_model.py` | New | ~170 (incl. cache) |
| `ibrl/infrabayesian/world_model.py` | Deleted (or one-line shim) | 0 |
| `ibrl/outcome.py` | Field rename `env_action`→`observation` | 1 |
| `ibrl/environments/*.py` | Field rename in 5–6 files | ~15 |
| `tests/*.py` | Field rename in fixtures (~14 sites) | ~14 |
| `ibrl/infrabayesian/a_measure.py` | Unchanged | 0 |
| `ibrl/infrabayesian/infradistribution.py` | Unchanged | 0 |
| `ibrl/agents/infrabayesian.py` | Unchanged | 0 |
| `tests/test_supra_pomdp_world_model.py` | New (Tier 1) | ~250 |
| `tests/test_supra_pomdp_integration.py` | New (Tier 2) | ~120 |
| `tests/test_supra_pomdp_agent.py` | New (Tier 3) | ~200 |
