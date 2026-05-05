# `ibrl` Architecture and Theory Notes

`ibrl` is the shared library for the Project-Newcomb infra-Bayesian reinforcement learning experiments. The current package is best understood as a small finite interaction framework plus an infra-Bayesian inference core for repeated bandit and Newcomb-like decision problems.

The stable code supports:

- ordinary repeated agent-environment simulation;
- baseline bandit agents and policy-dependent experimental agents;
- Newcomb-like environments where the environment responds to the agent's whole action distribution;
- a finite infra-Bayesian representation using affine measures, Bayesian mixtures, and Knightian uncertainty over finitely many minimal points;
- pluggable world models for Bernoulli bandits and Newcomb-like predictors.

## Package Layout

```text
ibrl/
    agents/             Agent interfaces and implementations.
    environments/       Environment interfaces and toy environments.
    infrabayesian/      Infra-Bayesian representation and world-model API.
    simulators/         Generic agent-environment rollout loop.
    utils/              Construction, sampling, and formatting helpers.
    outcome.py          Shared reward/observation result object.
```

The package intentionally keeps the simulation protocol simple: on each step, an agent returns a distribution over actions, the simulator samples an action from that distribution, the environment returns an `Outcome`, and the agent updates from the policy, action, and outcome. This is enough for bandits and one-step policy-dependent environments, and it gives Newcomb-like environments access to the policy that was committed before the action was sampled.

## Runtime Protocol

The main interaction loop is `simulators/simulator.py::simulate`:

1. `agent.get_probabilities()` returns a probability vector over discrete actions.
2. `sample_action` samples an action using the agent's RNG.
3. `env.step(probabilities, action)` returns an `Outcome(reward, observation)`.
4. `agent.update(probabilities, action, outcome)` updates the agent state.
5. The simulator records policies, sampled actions, rewards, and average reward statistics.

`Outcome` has two fields:

- `reward`: scalar payoff received by the agent;
- `observation`: optional discrete signal, such as a Newcomb predictor's prediction or a Bernoulli outcome index.

There is not yet a separate hidden-performance metric in the stable simulator. Experiments that distinguish observed reward from evaluator performance currently need a custom loop or simulator extension.

## Agents

All agents inherit from `agents/base.py::BaseAgent`. The base interface assumes a finite discrete action set and requires:

- `get_probabilities() -> np.ndarray`;
- `update(probabilities, action, outcome)`;
- `reset()`;
- `dump_state()`.

`BaseGreedyAgent` adds epsilon-greedy and softmax exploration utilities. These exploration mechanisms are useful for baseline learning experiments but are not part of the infra-Bayesian theory itself.

Stable baseline agents include:

- `QLearningAgent`: sample-average or fixed-step reward estimation per action;
- `BayesianAgent`: simple Gaussian-style reward-mean updating with per-action precision;
- `DiscreteBayesianAgent`: Bayesian updating over a discrete grid of Bernoulli reward hypotheses;
- `BernoulliBayesianAgent`: Bernoulli-specific Bayesian baseline;
- `EXP3Agent`: adversarial bandit baseline using exponential weights.

Experimental policy-dependent agents include:

- `ExperimentalAgent1`: samples a deterministic action from a learned policy before committing, exposing only diagonal reward-table entries in Newcomb-like problems;
- `ExperimentalAgent2`: estimates the full prediction/action reward table using peaked stochastic policies;
- `ExperimentalAgent3`: discretizes the continuous policy simplex and treats policies as actions.

`InfraBayesianAgent` is the stable infra-Bayesian agent. It can hold mixtures of classical and knightian uncertain hypotheses. After observing an outcome, the agent applies `Infradistribution.update(...)` to the shared distribution.

## Environments

All environments inherit from `environments/base.py::BaseEnvironment`. The stable environment protocol is one-step-per-interaction rather than full MDP state evolution. This makes the current package strongest for bandits and Newcomb-like repeated decision problems.

Implemented environments include:

- `BanditEnvironment`: Gaussian multi-armed bandit.
- `BernoulliBanditEnvironment`: Bernoulli multi-armed bandit.
- `SwitchingAdversaryEnvironment`: bandit whose best arm changes at a fixed time.
- `BaseNewcombLikeEnvironment`: base class for policy-dependent one-step environments.
- `NewcombEnvironment`: two-action Newcomb's problem.
- `DeathInDamascusEnvironment`: two-city Death in Damascus game.
- `AsymmetricDeathInDamascusEnvironment`: Death in Damascus with asymmetric death payoffs.
- `CoordinationGameEnvironment`: policy-dependent coordination game.
- `PolicyDependentBanditEnvironment`: random reward-table generalization of bandits and Newcomb-like games.
- `MatchEnvironment`: coin-tossing game rewarding action/observation matches.
- `ReverseTailsEnvironment`: coin-tossing game rewarding heads mismatch and indifferent tails.

The Newcomb-like base class represents a predictor by interpolating between a perfect policy predictor and a uniform random predictor. For predictor accuracy `acc`, the prediction distribution is:

```text
prediction = policy * (2 * acc - 1) + uniform * (2 - 2 * acc)
```

This convention makes `acc = 1` a perfect predictor and `acc = 0.5` an uninformative predictor.

## Infra-Bayesian Core

The infra-Bayesian implementation lives in `infrabayesian/`.

### `AMeasure`

`a_measure.py::AMeasure` represents one affine measure-like object:

```text
(lambda * mu, b)
```

where:

- `params` are opaque model-family parameters interpreted by a `WorldModel`;
- `scale` is the multiplicative weight `lambda`;
- `offset` is the affine constant `b`.

Evaluation of a reward function is:

```text
lambda * E_mu[reward] + b
```

This is the part of the code closest to the affine-measure presentation of infra-Bayesian updating. The actual measure semantics are delegated to `WorldModel`.

### `Infradistribution`

`infradistribution.py::Infradistribution` stores a finite list of `AMeasure`s and a shared `WorldModel`. The list is interpreted as extremal/minimal points of a finite convex representation. Evaluation is pessimistic:

```text
E_H(reward | action, policy) = min_m m.evaluate_action(...)
```

The minimum is always taken over the `AMeasure`s present in the `Infradistribution`. When there is only one measure, this minimum is vacuous and evaluation reduces to ordinary expected value under that measure. Classical Bayesian uncertainty is represented by `Infradistribution.mix(...)`, which folds prior-weighted hypotheses into mixed measure parameters. Knightian uncertainty is represented by `Infradistribution.mixKU(...)`, which keeps alternatives as separate measures; in that case the same minimum becomes a substantive worst-case operation. This realizes a finite maximin decision rule whenever more than one measure remains present.

The class has two mixture constructors:

- `Infradistribution.mix(components, coefficients)` creates a Bayesian mixture of infradistributions;
- `Infradistribution.mixKU(components)` creates Knightian uncertainty by taking the union of measures from multiple components.

### Updating

`Infradistribution.update(...)` implements a finite affine-style update inspired by Definition 11 in the Basic Inframeasure Theory sequence:

1. compute glued reward functions for the observed event;
2. compute the event probability from the difference of pessimistic expectations;
3. scale each measure by its event likelihood;
4. update the shared belief state through the `WorldModel`;
5. store off-history value in each measure's offset;
6. renormalize scale and offset by the event probability.

This is more than naive credal-set conditioning because it tracks scale and offset terms. However, it is still a finite, model-specific implementation. It does not claim to cover arbitrary signed affine measures, continuous infradistributions, or all dynamic-consistency cases from the full theory.

### `WorldModel`

`world_model.py::WorldModel` is the abstraction boundary between generic infra-Bayesian machinery and a concrete hypothesis family. It defines:

- how to construct hypothesis parameters;
- how to mix parameter objects;
- how to map an `Outcome` to a discrete event index;
- how to initialize and update the belief state;
- how to compute event likelihoods;
- how to compute expected rewards.

The `Infradistribution` does not inspect model parameters directly. This keeps the affine-measure logic generic while allowing different finite model families.

Stable world models include:

- `MultiBernoulliWorldModel`: repeated multi-arm categorical/Bernoulli outcomes;
- `NewcombWorldModel`: policy-dependent predictor model parameterized by predictor accuracy.

## Implemented Theory

The current stable package implements a finite fragment of infra-Bayesian decision theory:

- **Finite infradistributions via extremal affine measures.** `Infradistribution` stores finitely many `AMeasure`s rather than arbitrary closed convex sets or general functional representations.
- **Worst-case expectation.** Action and policy values are evaluated by the minimum over represented measures.
- **Bayesian mixtures.** `mix` represents ordinary probabilistic uncertainty over hypotheses.
- **Knightian uncertainty.** `mixKU` represents imprecise uncertainty by keeping alternative measures separate and evaluating pessimistically.
- **Affine-style update bookkeeping.** Updates track measure scale and offset, rather than only replacing each distribution by a conditional posterior.
- **Policy-dependent causal structure.** Newcomb-like environments and `NewcombWorldModel` allow outcome distributions to depend on the agent's committed policy, not only the sampled action.

This is enough to model the central contrast used in the experiments: a Bayesian agent averages over a precise prior, while an infra-Bayesian agent can preserve multiple compatible hypotheses and choose the policy with the best worst-case value.

Uncertainty is represented by finite, explicitly listed measures. The stable package does not directly represent continuous credal sets or solve optimization problems over arbitrary convex sets of distributions; such sets must be discretized or manually reduced to finitely many `AMeasure`s.



