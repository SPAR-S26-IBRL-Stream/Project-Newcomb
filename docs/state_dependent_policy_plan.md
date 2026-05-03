# Plan: State-Dependent Policies

This note describes what it would take to support state-dependent policies in the current `ibrl` codebase. The immediate target is finite, fully observable MDP-style experiments such as lava gridworlds, side-effect gridworlds, and island navigation.

The goal is not full POMDP policy planning yet. Start with stationary state policies:

```text
policy[state, action] -> probability of action in that state
```

## Current Assumption

Most agents currently expose:

```python
get_probabilities() -> np.ndarray  # shape (num_actions,)
```

This returns one global action distribution. That vector is also passed through the simulator and into policy-dependent world models.

This is enough for:

- bandits;
- one-step reward/spec ambiguity;
- simple Newcomb-like cases;
- one-state POMDPs.

It is not enough for gridworld navigation, because a useful policy must say:

```text
go right from this cell
go up from that cell
avoid lava near that other cell
```

## Desired Minimal Interface

Add a new agent type rather than changing every existing agent:

```python
StatePolicySupraPOMDPAgent(SupraPOMDPAgent)
```

It should support:

```python
get_policy() -> np.ndarray
```

where:

```text
policy.shape == (num_states, num_actions)
```

For simulator compatibility, it should still expose:

```python
get_probabilities() -> np.ndarray
```

which returns the action distribution for the current inferred state:

```python
current_state = infer_current_state(self.dist.belief_state)
return self.current_policy[current_state]
```

For fully observable experiments, `infer_current_state` can use a one-hot belief or `argmax`.

## World Model Changes

`SupraPOMDPWorldModel._value_iteration()` currently evaluates a fixed global action distribution:

```python
V_new = Q @ policy
```

For state-dependent policies, update it to handle:

```text
policy.shape == (num_states, num_actions)
```

Pseudocode:

```python
R_sa = (T * R).sum(axis=2)       # shape (S, A)
V = np.zeros(num_states)

for _ in range(value_iter_max):
    EV = T @ V                   # shape (S, A)
    Q = R_sa + discount * EV     # shape (S, A)

    if policy.ndim == 1:
        V_new = Q @ policy       # old behavior
    else:
        V_new = np.sum(policy * Q, axis=1)

    if max_abs(V_new - V) < tol:
        break
    V = V_new
```

Also add a cleaner value API:

```python
compute_policy_value(belief_state, params, policy) -> float
```

Pseudocode:

```python
def compute_policy_value(belief_state, params, policy):
    beliefs = init_or_current_beliefs(belief_state, params, policy)
    weights = posterior_weights(beliefs, params)
    total = 0.0

    for belief, T_k, R_k, weight in components:
        T_arr = resolve(T_k, policy)
        V = value_iteration(T_arr, R_k, policy)
        total += weight * (belief @ V)

    return total
```

This is simpler than trying to make `compute_q_values()` serve both global-policy and state-policy planning.

## Robust Evaluation Over A-Measures

State-dependent policies should be added together with robust planning over all a-measures.

For each candidate policy:

```python
values = []
for measure in self.dist.measures:
    raw_value = wm.compute_policy_value(
        self.dist.belief_state,
        measure.params,
        policy,
    )
    values.append(measure.scale * raw_value + measure.offset)

policy_value = min(values)
```

Then select the policy with maximal `policy_value`.

This fixes the current issue where `SupraPOMDPAgent` only plans with `self.dist.measures[0].params`.

## Policy Search

Naively enumerating stochastic state policies is expensive.

If there are `S` states, `A` actions, and `d` policy-discretization levels, the number of policies grows roughly like:

```text
num_discrete_action_distributions ^ S
```

For deterministic policies:

```text
A ^ S
```

That is acceptable only for tiny MDPs.

Recommended progression:

1. **Tiny tests:** enumerate deterministic state policies for very small `S`.
2. **Gridworld demos:** compute optimal policies by Bellman backups instead of enumerating all policies.
3. **Later:** support stochastic state policies only when policy-dependence requires them.

For robust MDP-style gridworlds, direct Bellman optimality is better:

```python
V[s] = max_a min_hypothesis Q_hypothesis[s, a]
```

or, for loss-minimization:

```python
V[s] = min_a max_hypothesis Q_hypothesis[s, a]
```

But for a first patch, deterministic enumeration is easier and enough to test the interface.

## Simulator Compatibility

The simulator currently does:

```python
probabilities = agent.get_probabilities()
action = sample_action(agent.random, probabilities)
outcome = env.step(probabilities, action)
agent.update(probabilities, action, outcome)
```

For ordinary gridworlds, this can remain unchanged because the environment only needs the current action distribution and sampled action.

For policy-dependent environments, such as Newcomb predictors, the environment or world model may need the full committed policy. Avoid changing every environment immediately. Instead:

- keep `get_probabilities()` for current-step simulation;
- keep the full policy inside the state-policy agent as `self.current_policy`;
- pass the full policy into `Infradistribution.update()` from that agent subclass;
- only add environment-level full-policy support if an experiment requires it.

## Fully Observable Assumption

This plan assumes observations identify the current state.

For gridworlds:

```text
observation == state_index
```

Then after each update, the belief should be one-hot or nearly one-hot, and the agent can choose:

```python
current_state = np.argmax(belief)
```

This is not full POMDP planning. For true POMDPs, the correct policy object is a belief/history policy:

```text
policy[belief, action]
```

or a finite-state controller. That is a larger project and should not be the first step.

## Tests To Add

Add focused tests before using this in experiments:

1. **One-state compatibility**

   A one-state MDP with a state policy should produce the same action choice as the old global-policy agent.

2. **Two-state policy difference**

   Create an MDP where the optimal action differs by state:

   ```text
   state 0: action 0 is best
   state 1: action 1 is best
   ```

   Assert the learned/chosen policy uses different actions.

3. **Belief-to-state selection**

   With fully observable observations, after observing state `s`, `get_probabilities()` should return `policy[s]`.

4. **Robust a-measure planning**

   With KU over two reward hypotheses, assert policy selection uses worst-case value, not only the first measure.

5. **Tiny gridworld path**

   A 3-cell line or 2x2 grid should require different actions in different states to reach the goal.

## Suggested Patch Order

1. Add `compute_policy_value()` to `SupraPOMDPWorldModel`.
2. Update `_value_iteration()` to support both `(A,)` and `(S,A)` policies.
3. Fix robust evaluation over all a-measures.
4. Add `StatePolicySupraPOMDPAgent`.
5. Add tests for one-state compatibility and two-state state-dependent choice.
6. Add a tiny fully observable gridworld smoke test.

Once this works, the project can support meaningful tabular versions of:

- lava world;
- side-effect gridworld;
- island navigation;
- absent-supervisor gridworld;
- simple reward/specification ambiguity with navigation.
