# SupraPOMDP Sequel Repair Plan

Date: 2026-05-04

This plan describes the follow-up PR into `ZKMathquant/suprapomdp_sequel`.
The goal is to keep the useful belief-aware policy infrastructure while
restoring the `alaro/suprapomdp` infra-Bayesian contracts that the branch
currently breaks.

## Goals

- Support belief-aware policies for finite SupraPOMDP agents.
- Preserve infra-Bayesian update semantics from `alaro/suprapomdp`.
- Keep Newcomb-style policy-dependent kernels working.
- Reduce test churn by avoiding unnecessary API/signature rewrites.

## Non-Goals

- Do not implement full finite-state controller planning.
- Do not replace the existing generic `InfraBayesianAgent` decision rule.
- Do not convert `SupraPOMDPWorldModelParameters` back into raw tuples/lists.
- Do not make `compute_expected_reward()` perform multi-step planning.

## Core Principle

Keep these operations separate:

```text
IB update:
  one-step expectation over next observations

POMDP planning:
  multi-step value / Q computation

Policy dependence:
  candidate committed BeliefPolicy passed to T(policy), B(policy), theta_0(policy)

Belief-aware control:
  action distribution selected from current belief
```

The infra-Bayesian update rule uses glued reward functions over observations.
Therefore this method must keep one-step semantics:

```python
def compute_expected_reward(
    belief_state,
    reward_function: np.ndarray,
    params: SupraPOMDPWorldModelParameters,
    action: int,
    committed_policy: BeliefPolicy,
) -> float:
    """E[reward_function[next_obs] | belief, action, committed_policy]."""
```

Multi-step planning should remain separate:

```python
def compute_q_values(
    belief_state,
    params: SupraPOMDPWorldModelParameters,
    committed_policy: BeliefPolicy,
) -> np.ndarray:
    """Multi-step Q-values for action choice under the candidate policy."""
```

## Keep From This PR

- `BeliefPolicy` concept: a table mapping discretized beliefs to action
  distributions. Rename later if useful, but keep the representation.
- `BeliefIndexer`, `simplex_grid()`, and `corner_beliefs()`.
- `PolicyOptimizer` greedy and softmax extraction from `Q[belief, action]`.
- Focused tests for policy-table behavior and belief discretization.
- Tiger-style tests that demonstrate different beliefs imply different actions,
  after correcting the Tiger dynamics/reward setup.

## Discard Or Rework

- Discard the rewrite of generic `InfraBayesianAgent.get_probabilities()`.
  Restore the base branch policy-space `_expected_rewards()` path for non-POMDP
  and flat-policy agents.
- Discard `compute_expected_reward()` as value iteration. Restore the one-step
  observation expectation from `alaro/suprapomdp`.
- Discard the raw tuple/list parameter representation. Keep
  `SupraPOMDPWorldModelParameters` from `alaro/suprapomdp`.
- Rework `compute_q_values_belief_indexed()` so it never resolves
  policy-dependent kernels against an implicit uniform policy.
- Do not carry over the PR's `StatefulPolicy.to_flat_policy()` method; do not
  support implicit averaging of belief policies into flat action vectors.
- Remove committed coverage artifacts such as `codecov_recent.md`.

## Proposed API Shape

Use one policy representation for SupraPOMDP control:

```python
@dataclass(frozen=True)
class BeliefPolicy:
    belief_points: np.ndarray
    policy_table: np.ndarray

    def action_dist(self, belief: np.ndarray) -> np.ndarray: ...
    def cache_key(self) -> bytes: ...
```

Flat policies are the one-belief special case:

```text
policy_table.shape == (1, num_actions)
belief_points.shape == (1, num_states)
action_dist(any_belief) -> policy_table[0]
```

Fully observable state policies are also a special case: represent each state
as a one-hot belief and use the same table/indexer machinery.

SupraPOMDP callable kernels receive the full `BeliefPolicy` object:

```python
def theta_0(policy: BeliefPolicy) -> np.ndarray: ...
def T(policy: BeliefPolicy) -> np.ndarray: ...
def B(policy: BeliefPolicy) -> np.ndarray: ...
```

This makes Newcomb-style commitment semantics explicit: the predictor/world can
inspect the whole policy table and the belief points that give each row its
meaning. Existing flat Newcomb tests become one-row `BeliefPolicy` tests. Do
not silently average a genuinely belief-dependent policy into a flat vector.

## Agent Plan

Keep `InfraBayesianAgent` close to `alaro/suprapomdp`.

Add or repair SupraPOMDP-specific behavior in `SupraPOMDPAgent`:

```python
class SupraPOMDPAgent(InfraBayesianAgent):
    def _expected_rewards(self) -> np.ndarray:
        """Existing policy-space optimization over constant BeliefPolicy objects."""

    def get_belief_probabilities(self, belief: np.ndarray) -> np.ndarray:
        """Return pi(a | belief) when a belief policy is configured."""

    def _compute_belief_policy(self) -> BeliefPolicy:
        """Build Q[belief, action] and optimize per belief."""
```

Decision rule:

- Always select or compute a `BeliefPolicy`.
- For bandits, one-state problems, and existing Newcomb tests, that policy is
  constant and has one belief row.
- For belief-aware POMDP control, the policy has multiple belief rows.
- At runtime, query the selected policy with the current belief:

```python
return current_policy.action_dist(current_belief)
```

Keep two values distinct at runtime:

```python
committed_policy: BeliefPolicy
action_probs: np.ndarray = committed_policy.action_dist(current_belief)
```

The simulator samples from `action_probs`. The SupraPOMDP world model and
infra-Bayesian update receive `committed_policy`, so callable kernels can
condition on the whole policy table.

## World Model Plan

Restore the `alaro/suprapomdp` world-model contracts:

```python
@dataclass
class SupraPOMDPWorldModelParameters:
    T: list
    B: list
    theta_0: list
    R: list
    weights: np.ndarray
```

Keep:

```python
def update_state(...): ...
def compute_likelihood(...): ...
def compute_expected_reward(...): ...
def compute_q_values(...): ...
```

Add only if needed:

```python
def compute_q_values_for_beliefs(
    belief_points: np.ndarray,
    params: SupraPOMDPWorldModelParameters,
    committed_policy: BeliefPolicy,
) -> np.ndarray:
    """Return Q[belief_idx, action_idx]."""
```

The committed policy must be explicit and must be passed unchanged to callable
`T`, `B`, and `theta_0`. No hidden uniform policy should be used.

## Test Plan

The current PR changes a large part of the test suite mostly because it changed
public signatures and parameter representation:

- `InfraBayesianAgent.__init__` became positional and made `reward_function`
  required.
- `SupraPOMDPWorldModelParameters` was replaced with raw tuple/list params.
- `compute_expected_reward()` semantics changed, requiring tests to be rewritten
  around value iteration.

With this repair plan, much less of the existing test suite should change.
Most `alaro/suprapomdp` tests should remain valid because the public contracts
are restored.

Keep:

- Existing `alaro/suprapomdp` world-model tests for param construction,
  Bayesian filtering, posterior weights, likelihood, one-step
  `compute_expected_reward()`, `compute_q_values()`, and policy-dependent
  kernels.
- New focused tests for `BeliefPolicy`,
  `BeliefIndexer`, simplex/corner grids, and optimizer output.
- New or revised Tiger tests proving belief-aware control chooses different
  actions at different beliefs.
- New regression tests that `compute_expected_reward()` ignores `R` and uses
  the passed `reward_function`.
- New regression tests that Newcomb candidate `BeliefPolicy` objects are passed
  to `theta_0(policy)` and one-boxing wins as a committed policy.

Discard or shrink:

- Large rewrites of `test_supra_pomdp_world_model.py` that only adapt to raw
  tuple/list params.
- Broad smoke-test duplication in `test_supra_pomdp_smoke.py`; keep only import
  and integration cases that cover unique behavior.
- `tests/conftest.py` fixtures that encode broken signatures or duplicate local
  helpers better kept inside the relevant test file.
- The Bernoulli test change that forces `reward_function=np.ones((n, 2))`.
  The default reward function behavior from `alaro/suprapomdp` should continue
  to work.

## Patch Order

1. Restore base branch signatures and `SupraPOMDPWorldModelParameters`.
2. Restore one-step `compute_expected_reward()` and `compute_likelihood()`.
3. Restore generic `InfraBayesianAgent.get_probabilities()`.
4. Reintroduce `BeliefPolicy`, belief discretization, and `PolicyOptimizer`
   with minimal cleanup.
5. Add a SupraPOMDP-specific belief-policy path that does not flatten by
   default.
6. Rework tests by keeping base tests, adding focused belief-policy tests, and
   deleting churn that only served the broken rewrite.
