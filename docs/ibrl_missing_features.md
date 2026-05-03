# Missing Features for Stronger IB Experiments

This note summarizes features discussed for future infra-Bayesian (IB) experiments that are not fully supported by the current implementation. It is intended as onboarding context for a new contributor.

Current state, roughly:

- `SupraPOMDPWorldModel` supports finite, point-valued POMDP hypotheses.
- `T`, `B`, and `theta_0` may depend on the candidate policy, which is useful for Newcomb-like predictors.
- `Infradistribution` can represent Bayesian mixtures and Knightian uncertainty over a finite set of a-measures.
- `SupraPOMDPAgent` currently plans over global action distributions, not state/observation-dependent policies.

## 1. Robust Planning Over All A-Measures

Current gap:

- `SupraPOMDPAgent._expected_rewards()` uses only `self.dist.measures[0].params`.
- This bypasses the intended worst-case/minimum over all a-measures when planning.

What to add:

- Evaluate each candidate policy under every `AMeasure` in `self.dist.measures`.
- Include each measure's `scale` and `offset` if the value is meant to reflect updated affine-measure state.
- Choose the policy with the best worst-case value.

Supports:

- genuine Knightian uncertainty over finite POMDP hypotheses;
- robust map/layout experiments;
- robust reward/specification ambiguity;
- cleaner IB vs Bayesian comparisons.

Priority: high. This is a near-term correctness issue.

## 2. State-Dependent and Observation-Dependent Policies

Current gap:

- Policies are global action distributions:

  ```text
  pi -> P(action)
  ```

- Gridworlds and transparent Newcomb with post-observation action require conditional policies:

  ```text
  pi(state) -> P(action)
  pi(observation) -> P(action)
  pi(history or belief) -> P(action)
  ```

What to add:

- For fully observable gridworlds, start with stationary state policies:

  ```text
  policy[state, action]
  ```

- For simple transparent Newcomb, support observation policies:

  ```text
  policy[observation, action]
  ```

- Later, support belief-state policies for true POMDP planning.

Supports:

- lava/gridworld navigation;
- side-effect gridworlds;
- island navigation / safe exploration;
- transparent Newcomb where the agent observes full/empty before acting;
- policies that can change after learning or after observations.

Priority: high for gridworld experiments.

## 3. Gridworld / Finite MDP Builders

Current gap:

- There is no shared utility for turning a gridworld layout into `T`, `B`, `theta_0`, and `R`.

What to add:

- A small builder that takes:

  ```text
  grid layout
  start states
  terminal states
  action set
  walls/hazards/goals
  slip probabilities
  observation convention
  reward table
  ```

- Outputs arrays compatible with `SupraPOMDPWorldModel`.

Supports:

- AI Safety Gridworld-inspired experiments;
- lava world;
- side effects;
- absent supervisor;
- island navigation;
- easier experiment reproducibility.

Priority: medium-high after policy support.

## 4. Hidden Performance Metrics

Current gap:

- `Outcome.reward` is the only scalar recorded by `simulate`.
- The AI Safety Gridworlds paper distinguishes observed reward from hidden performance.

What to add:

- Extend `Outcome` or environment state to expose an optional performance metric:

  ```text
  observed_reward: what the agent receives
  performance_reward: what the evaluator cares about
  safety_violation: optional boolean/count
  ```

- Update simulator results to record both reward and performance.

Supports:

- side-effect experiments;
- reward gaming / tomato watering;
- absent supervisor;
- safe exploration metrics;
- honest evaluation where the agent's reward is intentionally misspecified.

Priority: medium.

## 5. Set-Valued Transition Support

Current gap:

- `SupraPOMDPWorldModel` uses point-valued transitions:

  ```text
  T(s, a) -> one distribution over next states
  ```

- True crisp supra-MDPs use set-valued transitions:

  ```text
  T(s, a) -> credal set of possible next-state distributions
  ```

What to add:

- Represent each transition credal set as intervals, vertices of a polytope, or a small optimizer.
- Implement robust Bellman backups:

  ```text
  Q(s,a) = R(s,a) + gamma * worst_case_{p in T(s,a)} E_p[V]
  ```

- Be careful about global consistency: purely local worst-case transitions may be more adversarial than "one hidden layout sampled at episode start."

Supports:

- true supra-MDP experiments;
- coarse-map abstraction;
- robust lava/island navigation without enumerating every hidden layout;
- compact uncertainty over local dynamics.

Priority: medium. Enumerated hidden layouts can be used first.

## 6. Reward / Specification Ambiguity

Current gap:

- Multiple reward functions can be represented as different hypotheses, but there is no dedicated experiment structure or evaluation support.

What to add:

- Utilities for creating hypotheses with shared dynamics but different `R`.
- Clear separation between:

  ```text
  observed reward
  hypothesized true reward
  hidden performance
  ```

- Robust planning over those reward hypotheses.

Supports:

- avoiding side effects;
- tomato watering / observation corruption;
- ambiguous user intent;
- comparing Bayesian priors over reward functions against IB credal uncertainty.

Priority: high for the easiest safety demonstrations.

## 7. Observation Corruption / Delusion Models

Current gap:

- POMDP observations are supported, but there is no reusable pattern for actions that corrupt observations or reward signals.

What to add:

- Model latent true state separately from observed state.
- Allow actions that change observation kernel `B` or enter a sensor-corrupted latent state.
- Add hypotheses such as:

  ```text
  bucket waters tomatoes
  bucket corrupts tomato sensors
  bucket does both partially
  ```

Supports:

- tomato watering;
- wireheading/delusion-box examples;
- reward-gaming experiments.

Priority: medium.

## 8. Learning From Numeric Rewards

Current gap:

- `Infradistribution.update()` updates on discrete `Outcome.observation` through `event_index`.
- Some proposed experiments require learning unknown payouts from scalar rewards, not just discrete observations.

What to add:

- Either discretize observed rewards into event indices, or extend world models to include likelihoods over numeric rewards.
- For transparent Newcomb with unknown payouts, hypotheses should update from both:

  ```text
  observed box state
  received payout
  ```

Supports:

- repeated transparent Newcomb with unknown predictor accuracy and payouts;
- payout-learning bandits;
- reward uncertainty that shrinks with evidence.

Priority: medium for Newcomb learning experiments.

## 9. Conditional Policy-Dependent Predictors

Current gap:

- Policy-dependent `theta_0(policy)` exists, but policies are currently simple action distributions.
- Transparent Newcomb with a visible full/empty box needs the predictor to evaluate a conditional policy.

What to add:

- Define a policy object or array convention for:

  ```text
  policy[observation, action]
  ```

- Define predictor semantics, for example:

  ```text
  P(predicted one-box) depends on pi(one-box | full)
  ```

- Thread this conditional policy through callable `T`, `B`, or `theta_0`.

Supports:

- self-consistent transparent Newcomb;
- repeated Newcomb with unknown predictor accuracy;
- friend/foe policy-prediction experiments.

Priority: medium-high if Newcomb/self-consistency is a near-term goal.

## Suggested Build Order

1. Fix robust planning over all a-measures.
2. Add conditional/state-dependent policy support.
3. Add gridworld/finite-MDP builder utilities.
4. Add hidden performance metrics.
5. Implement reward/specification ambiguity experiments.
6. Implement lava with enumerated hidden layouts.
7. Add observation-corruption utilities for tomato watering.
8. Add conditional policy-dependent predictors for transparent Newcomb.
9. Add set-valued transitions for true supra-MDP/coarse-abstraction experiments.

The first two items are architectural. The rest are experiment-enabling layers.
