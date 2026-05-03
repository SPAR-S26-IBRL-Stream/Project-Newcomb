# Trap Bandit Implementation Plan

This plan describes how to implement the experiment in [trap_bandit_experiment.md](./trap_bandit_experiment.md).

Implementation note: the final code promotes a generic `JointBanditWorldModel`
that stores per-arm finite outcome distributions in each component. Trap-specific
logic such as `argmax(p1, p2)` and `p_cat` lives in
`ibrl.infrabayesian.builders.trap_bandit` and
`ibrl.environments.trap_bandit`, not in the generic world model.

The core idea is to use the same infra-Bayesian machinery for both agents:

- **Bayesian agents** use `Infradistribution.mix(...)`.
- **Infra-Bayesian agent** uses Knightian uncertainty over safe-vs-risky families via `Infradistribution.mixKU(...)`.

This keeps hypothesis specification and update mechanics as aligned as possible. The main implementation work is a new joint bandit world model and a cleaner pluggable exploration interface for `InfraBayesianAgent`.

## Experiment Recap

There are two arms.

In a **safe world**:

```text
arm i -> Bernoulli(p_i)
```

In a **risky world**:

```text
trapped_arm = argmax(p1, p2)
trapped_arm -> catastrophe with probability p_cat
               reward 1 with probability p_i
               reward 0 otherwise
other arm   -> Bernoulli(p_i)
```

Data-generating process:

```text
alpha ~ Beta(2, 2)      # probability the sampled world is risky
p1, p2 ~ Beta(2, 2)
p_cat = 1 / 100
```

Correct-prior condition:

```text
agent risky prior uses alpha ~ Beta(2, 2)
```

Misspecified-prior condition:

```text
Bayesian risky prior uses alpha ~ Beta(2, 5)
IB keeps KU over safe vs risky, but uses the same p1/p2 grid
```

For implementation, use the mean of the alpha prior to weight safe vs risky in Bayesian mixtures:

```text
P(risky) = E[alpha]
```

So:

```text
Beta(2,2): E[alpha] = 0.5
Beta(2,5): E[alpha] = 2/7
```

## Outcomes

Use discrete outcomes so the existing `Infradistribution.update()` path can condition on event indices.

```text
outcome 0: zero reward
outcome 1: reward 1
outcome 2: catastrophe reward -1000
```

Reward function:

```python
reward_function = np.array([
    [0.0, 1.0, -1000.0],
    [0.0, 1.0, -1000.0],
])
```

The environment should return:

```python
Outcome(reward=reward_value, observation=outcome_index)
```

The world model should use `Outcome.observation` as the event index, not infer it from `Outcome.reward`.

## New World Model: `JointBanditWorldModel`

Add:

```text
ibrl/infrabayesian/world_models/joint_bandit_world_model.py
```

Export it from:

```text
ibrl/infrabayesian/world_models/__init__.py
ibrl/infrabayesian/__init__.py
```

### Params

Each component is a complete joint world, preserving the relationship between `p1`, `p2`, and `trapped_arm`.

```python
@dataclass(frozen=True)
class JointBanditComponent:
    world_type: str  # "safe" or "risky"
    p1: float
    p2: float
    p_cat: float

@dataclass
class JointBanditWorldModelParameters:
    components: list[JointBanditComponent]
    weights: np.ndarray
```

### Belief State

Track history by action and outcome.

```python
@dataclass
class JointBanditBeliefState:
    history: np.ndarray  # shape (num_arms=2, num_outcomes=3)
```

### Methods

Implement the `WorldModel` interface:

```python
class JointBanditWorldModel(WorldModel):
    def __init__(self, num_arms: int = 2, p_cat: float = 0.01):
        ...

    def make_params(self, components, weights=None):
        ...

    def mix_params(self, params_list, coefficients):
        ...

    def event_index(self, outcome, action):
        return int(outcome.observation)

    def initial_state(self):
        return JointBanditBeliefState(np.zeros((2, 3), dtype=np.int64))

    def update_state(self, state, outcome, action, policy, params=None):
        new = state.history.copy()
        new[action, self.event_index(outcome, action)] += 1
        return JointBanditBeliefState(new)

    def is_initial(self, state):
        return not state.history.any()

    def compute_likelihood(self, belief_state, outcome, params, action, policy):
        probs = self._posterior_predictive(belief_state, params, action)
        return probs[self.event_index(outcome, action)]

    def compute_expected_reward(self, belief_state, reward_function, params, action, policy):
        probs = self._posterior_predictive(belief_state, params, action)
        return float(probs @ reward_function)
```

### Outcome Distribution

```python
def _component_outcome_probs(self, component, action):
    p = component.p1 if action == 0 else component.p2

    if component.world_type == "risky":
        trapped_arm = 0 if component.p1 >= component.p2 else 1
        if action == trapped_arm:
            return np.array([
                1.0 - p - component.p_cat,
                p,
                component.p_cat,
            ])

    return np.array([
        1.0 - p,
        p,
        0.0,
    ])
```

Validate:

```python
assert 0 <= p <= 1
assert 0 <= p_cat <= 1
assert p + p_cat <= 1 for trapped arms
```

### Posterior Predictive

Given history, update component weights by likelihood of all observed arm/outcome counts.

```python
def _posterior_component_weights(self, belief_state, params):
    log_w = np.log(params.weights)
    for k, component in enumerate(params.components):
        for action in range(self.num_arms):
            probs = self._component_outcome_probs(component, action)
            log_w[k] += np.sum(belief_state.history[action] * np.log(np.maximum(probs, 1e-300)))
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()

def _posterior_predictive(self, belief_state, params, action):
    w = self._posterior_component_weights(belief_state, params)
    total = np.zeros(self.num_outcomes)
    for weight, component in zip(w, params.components):
        total += weight * self._component_outcome_probs(component, action)
    return total
```

This makes each `AMeasure` internally Bayesian over its component family.

## Hypothesis Construction

Create helper functions under an experiment module, e.g.:

```text
experiments/alaro/trap_bandit/
  hypotheses.py
```

### Fixed p-grid

Use a fixed grid first:

```python
p_grid = np.linspace(0.05, 0.95, num_grid)
```

Weight grid points by Beta density:

```python
def beta_grid_weights(p_grid, a=2, b=2):
    raw = p_grid ** (a - 1) * (1 - p_grid) ** (b - 1)
    return raw / raw.sum()
```

Joint `p1,p2` weight:

```python
w_p = w[p1_index] * w[p2_index]
```

### Safe Family

```python
safe_components = [
    JointBanditComponent("safe", p1, p2, p_cat)
    for p1 in p_grid
    for p2 in p_grid
]

safe_weights = [
    w_p1 * w_p2
    for p1 in p_grid
    for p2 in p_grid
]

safe_dist = Infradistribution(
    [AMeasure(wm.make_params(safe_components, safe_weights))],
    world_model=wm,
)
```

### Risky Family

```python
risky_components = [
    JointBanditComponent("risky", p1, p2, p_cat)
    for p1 in p_grid
    for p2 in p_grid
]

risky_dist = Infradistribution(
    [AMeasure(wm.make_params(risky_components, safe_weights))],
    world_model=wm,
)
```

### Bayesian Agent Hypothesis

```python
p_risky = alpha_a / (alpha_a + alpha_b)

bayes_hypothesis = Infradistribution.mix(
    [safe_dist, risky_dist],
    np.array([1.0 - p_risky, p_risky]),
)
```

Use:

```python
correct prior:      alpha_a=2, alpha_b=2
misspecified prior: alpha_a=2, alpha_b=5
```

### IB Agent Hypothesis

IB is KU over safe vs risky but classical over `p1,p2` inside each family:

```python
ib_hypothesis = Infradistribution.mixKU([safe_dist, risky_dist])
```

## Exploration Strategy Refactor

The current `BaseGreedyAgent` bundles epsilon/softmax logic directly into the class. For this experiment, refactor toward composable exploration strategies.

Add:

```text
ibrl/agents/exploration.py
```

### Interface

Use an object that can inspect the agent, because Thompson sampling needs posterior/hypothesis state and UCB needs counts.

```python
class ExplorationStrategy:
    def get_probabilities(self, agent, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError
```

### Strategies

```python
class Greedy(ExplorationStrategy):
    def get_probabilities(self, agent, values):
        best = np.isclose(values, values.max())
        return best / best.sum()
```

```python
class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, epsilon):
        self.epsilon = epsilon  # float or schedule object

    def get_probabilities(self, agent, values):
        greedy = Greedy().get_probabilities(agent, values)
        eps = self._epsilon(agent.step)
        uniform = np.ones(agent.num_actions) / agent.num_actions
        return (1 - eps) * greedy + eps * uniform
```

```python
class UniformPrefixThen(ExplorationStrategy):
    def __init__(self, prefix_steps, base_strategy):
        self.prefix_steps = prefix_steps
        self.base_strategy = base_strategy

    def get_probabilities(self, agent, values):
        if agent.step <= self.prefix_steps:
            return np.ones(agent.num_actions) / agent.num_actions
        return self.base_strategy.get_probabilities(agent, values)
```

```python
class UCB(ExplorationStrategy):
    def __init__(self, c=2.0):
        self.c = c

    def get_probabilities(self, agent, values):
        counts = np.maximum(agent.action_counts, 1)
        total = max(agent.step, 2)
        bonus = self.c * np.sqrt(np.log(total) / counts)
        scores = agent.empirical_values + bonus
        best = np.isclose(scores, scores.max())
        return best / best.sum()
```

```python
class ThompsonSampling(ExplorationStrategy):
    def get_probabilities(self, agent, values):
        # Sample a joint component from the Bayesian posterior represented by
        # the agent's current mixed hypothesis, then choose the best arm under
        # that sampled component.
        component = agent.sample_component_from_posterior()
        sampled_values = agent.values_for_component(component)
        best = np.isclose(sampled_values, sampled_values.max())
        return best / best.sum()
```

For Thompson sampling, implement helper methods on the experiment subclass or on `InfraBayesianAgent` only if they can be cleanly generic.

Recommended minimal path:

- Add generic strategies for `Greedy`, `EpsilonGreedy`, `UniformPrefixThen`, `UCB`.
- Add trap-bandit-specific Thompson helper in the experiment folder if generic implementation becomes awkward.

## InfraBayesianAgent Changes

Modify `InfraBayesianAgent` to optionally accept:

```python
exploration_strategy: ExplorationStrategy | None = None
```

Default behavior should preserve current tests:

```python
if exploration_strategy is None:
    use existing exploration_prefix / greedy behavior
else:
    values = self._action_values()
    return exploration_strategy.get_probabilities(self, values)
```

It may be worth splitting current `_expected_rewards()` into:

```python
_action_values() -> np.ndarray
_policy_from_values(values) -> np.ndarray
```

For this bandit experiment, policies are just action distributions, so no state-policy work is needed.

### Agent Statistics for UCB

Track:

```python
self.action_counts: np.ndarray
self.reward_sums: np.ndarray
self.empirical_values: np.ndarray
```

Update them in `InfraBayesianAgent.update()` after calling `super().update(...)`.

Use actual scalar reward, including catastrophes, for empirical UCB:

```python
self.action_counts[action] += 1
self.reward_sums[action] += outcome.reward
self.empirical_values[action] = self.reward_sums[action] / self.action_counts[action]
```

## Trap Bandit Environment

Add:

```text
experiments/alaro/trap_bandit/environment.py
```

or shared:

```text
ibrl/environments/trap_bandit.py
```

For a first experiment, keep it under `experiments/`.

### Environment Params

```python
class TrapBanditEnvironment(BaseEnvironment):
    def __init__(self, p1, p2, risky, p_cat=0.01, catastrophe_reward=-1000.0, seed=..., **kwargs):
        super().__init__(num_actions=2, ...)
        ...
```

### Step Logic

```python
def _resolve(self, observation, action):
    p = self.p1 if action == 0 else self.p2
    trapped_arm = 0 if self.p1 >= self.p2 else 1

    if self.risky and action == trapped_arm:
        u = self.random.random()
        if u < self.p_cat:
            self.last_outcome_index = 2
            return self.catastrophe_reward
        if u < self.p_cat + p:
            self.last_outcome_index = 1
            return 1.0
        self.last_outcome_index = 0
        return 0.0

    reward = float(self.random.random() < p)
    self.last_outcome_index = int(reward)
    return reward
```

Override `step` if needed so `Outcome.observation` carries the discrete index:

```python
def step(self, probabilities, action):
    reward = self._resolve(None, action)
    return Outcome(reward=reward, observation=self.last_outcome_index)
```

### Optimal Reward / Regret

Regret is against the best policy with full knowledge of the true world.

Per-step optimal expected value:

```python
def expected_value(action):
    p = p1 if action == 0 else p2
    if risky and action == trapped_arm:
        return p * 1.0 + p_cat * catastrophe_reward
    return p

optimal_value = max(expected_value(0), expected_value(1))
```

Instant regret:

```text
optimal_value - expected_value(chosen_action)
```

For plots, record cumulative expected regret and realized cumulative regret separately if useful.

## Experiment Runner

Add:

```text
experiments/alaro/trap_bandit/run.py
```

### World Sampling

For each world:

```python
alpha = rng.beta(2, 2)
risky = rng.random() < alpha
p1 = rng.beta(2, 2)
p2 = rng.beta(2, 2)
```

Reject or clip samples where:

```text
max(p1, p2) + p_cat > 1
```

Simpler:

```python
p1 = min(p1, 1.0 - p_cat)
p2 = min(p2, 1.0 - p_cat)
```

### Agents

Construct:

- Bayesian greedy with correct prior.
- Bayesian Thompson with correct prior.
- Bayesian UCB with correct prior.
- IB with correct p-grid and KU over safe/risky.
- Bayesian greedy with misspecified alpha prior.
- Bayesian Thompson with misspecified alpha prior.
- Bayesian UCB with misspecified alpha prior.
- IB in misspecified setting, if applicable, still KU over safe/risky but same `p1,p2` grid.

Potential IB exploration strategy:

```python
UniformPrefixThen(prefix_steps=0, base_strategy=Greedy())
```

or, if explicitly testing early tie behavior:

```python
Greedy() with uniform tie-breaking
```

### Metrics

For each agent/world/step record:

```text
reward
action
outcome_index
is_catastrophe
is_risky_world
trapped_arm
pulled_trapped_arm
optimal_expected_value
chosen_expected_value
instant_expected_regret
cumulative_expected_regret
cumulative_realized_reward
```

Aggregate across worlds:

```text
5th, 50th, 95th percentiles of cumulative regret
5th, 50th, 95th percentiles of probability/frequency of choosing trapped arm
catastrophe counts/rate
```

Separate trapped-arm plots by:

```text
risky worlds
safe worlds
```

In safe worlds there is no actually catastrophic arm, but `argmax(p1,p2)` is still the arm that would be trapped under the risky mechanism. Label this carefully.

## Plotting

Add:

```text
experiments/alaro/trap_bandit/plot.py
```

Generate:

1. Correct-prior cumulative regret percentiles.
2. Correct-prior trapped-arm selection percentiles.
3. Misspecified-prior cumulative regret percentiles.
4. Misspecified-prior trapped-arm selection percentiles.
5. Optional catastrophe-rate bar chart.

Save results/configs:

```text
experiments/alaro/trap_bandit/results/
  config.json
  metrics.npz or metrics.csv
  fig_regret.png
  fig_trapped_arm.png
```

## Tests

Add:

```text
tests/test_joint_bandit_world_model.py
tests/test_trap_bandit_experiment.py
```

### World Model Tests

1. Safe component arm probabilities:

```text
safe, p1=0.7, p2=0.3
arm0 -> [0.3, 0.7, 0]
arm1 -> [0.7, 0.3, 0]
```

2. Risky component traps higher-p arm:

```text
risky, p1=0.7, p2=0.3, p_cat=0.01
arm0 -> [0.29, 0.7, 0.01]
arm1 -> [0.7, 0.3, 0]
```

3. Posterior weights update toward components that predict observed outcomes.

4. `event_index` uses `Outcome.observation`.

5. `mix_params` preserves joint components and weights.

### Agent Tests

1. Bayesian `mix` value equals weighted average of safe/risky values.
2. IB `mixKU` value equals minimum of safe/risky values.
3. Greedy tie-breaking returns uniform over tied best actions.
4. UCB tries unpulled or low-count actions.
5. Thompson sampling samples full joint components, not independent per-arm components.

## Caveats To Document

- Thompson/UCB/greedy are heuristic Bayesian exploration strategies, not Bayes-optimal planning.
- The IB agent's early exploration depends on explicit tie-breaking or prefix strategy; do not claim hard KU uniquely implies uniform exploration.
- Regret is measured against full-knowledge optimal expected value, so conservative IB may have worse median regret but better lower-tail/catastrophe behavior.
- The key comparison is not "IB beats ideal Bayes"; it is robustness under tail risk and prior misspecification.

## Suggested Patch Order

1. Add `JointBanditWorldModel` and tests.
2. Add trap-bandit environment and tests.
3. Add exploration strategy interface with `Greedy`, `EpsilonGreedy`, `UniformPrefixThen`, and `UCB`.
4. Add trap-bandit-specific Thompson sampling.
5. Update `InfraBayesianAgent` to accept an optional exploration strategy while preserving current behavior.
6. Add hypothesis-construction helpers for Bayesian and IB agents.
7. Add experiment runner and metrics.
8. Add plotting.
