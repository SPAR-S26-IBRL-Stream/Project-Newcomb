"""InfraBayesianAgent — agent using infrabayesian inference."""
import itertools
import numpy as np

from . import BaseGreedyAgent
from ..exploration import ExplorationStrategy, Greedy
from ..infrabayesian.a_measure import AMeasure
from ..infrabayesian.infradistribution import Infradistribution


class InfraBayesianAgent(BaseGreedyAgent):
    """
    Agent using infrabayesian inference.

    Maintains a single shared infradistribution (a Bayesian prior over hypotheses),
    updated on every step regardless of which action was taken. Each hypothesis is an
    Infradistribution over a shared WorldModel; the prior is a 1D array of weights.

    Arguments:
        hypotheses:       list of hypotheses
        prior:            distribution over hypotheses; default: uniform
        reward_function:  reward_function[a,o] is reward upon seeing outcome o from action a
        policy_discretisation:  number of mixed policies to consider per action; default: 0 (i.e. only pure policies)
        exploration_strategy:  optional strategy object for Bayesian single-measure exploration
    """
    def __init__(self, *,
            hypotheses : list[Infradistribution],
            prior : np.ndarray | None = None,           # shape (len(hypotheses),)
            reward_function : np.ndarray | None = None, # shape (num_actions, num_outcomes)
            policy_discretisation : int = 0,
            exploration_strategy : ExplorationStrategy | None = None,
            **kwargs):
        base_strategy = exploration_strategy if exploration_strategy is not None else Greedy()
        super().__init__(exploration_strategy=base_strategy, **kwargs)
        assert len(hypotheses) > 0
        assert all(isinstance(h.world_model, type(hypotheses[0].world_model))
                   for h in hypotheses), "All hypotheses must share the same WorldModel type"
        self.hypotheses = hypotheses
        self.prior = prior if prior is not None else np.ones(len(hypotheses)) / len(hypotheses)
        # default: reward_function[a,o] = o with o ∈ {0,1}
        self.reward_function = (reward_function if reward_function is not None
                                else np.linspace(np.zeros(self.num_actions),np.ones(self.num_actions),2).T)
        self.exploration_strategy = exploration_strategy

        # Discretise policy space:
        # Let n be the number of actions and 1/d be the distance between discretised policies
        # For every n-tuple of non-negative integers (a1, a2, ..., an) with Σ_i ai = d, we get
        # one possible policy as [a1/d, a2/d, ..., an/d]
        d = policy_discretisation+1
        self.policies = [
            np.array(x)/d
            for x in itertools.product(*[range(d+1) for _ in range(self.num_actions)])
            if sum(x)==d
        ]

    def reset(self):
        super().reset()
        self.dist = Infradistribution.mix(self.hypotheses, self.prior)

    def update(self, probabilities: np.ndarray, action: int, outcome) -> None:
        super().update(probabilities, action, outcome)
        self.dist.update(self.reward_function, outcome, action, probabilities)

    def get_probabilities(self) -> np.ndarray:
        if self.exploration_strategy is not None:
            if len(self.dist.measures) != 1:
                raise RuntimeError(
                    "Multi-measure infradistributions require "
                    "exploration_strategy=None until IB exploration "
                    "semantics are defined."
                )
            return self.exploration_strategy.get_probabilities(self, self._action_values())

        return self._optimal_policy()

    def dump_state(self) -> str:
        state = str(self.dist.belief_state)
        if self.verbose > 1:
            state += ";" + repr(self.dist)
        return state

    def _expected_rewards(self) -> np.ndarray:
        """Expected reward of each policy."""
        return np.array([
            sum(self._action_values_for_policy(policy) * policy)
            for policy in self.policies
        ])

    def _action_values(self) -> np.ndarray:
        policy = np.ones(self.num_actions) / self.num_actions
        return self._action_values_for_policy(policy)

    def _action_values_for_policy(self, policy: np.ndarray) -> np.ndarray:
        return np.array([
            self.dist.evaluate_action(self.reward_function[a], a, policy)
            for a in range(self.num_actions)
        ])

    def sample_component_from_posterior(self):
        """Sample one complete component from a world model posterior mixture."""
        params = self._component_mixture_params(
            "HypothesisThompsonSampling requires a single Bayesian mixture "
            "measure; KU/multi-measure infradistributions are not supported. "
        )
        world_model = self.dist.world_model
        weights = world_model.get_posterior_component_weights(self.dist.belief_state, params)
        idx = int(self.random.choice(len(params.components), p=weights))
        return params.components[idx]

    def posterior_component_action_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Return component action values and posterior weights for one Bayesian measure."""
        params = self._component_mixture_params(
            "Bayesian posterior component exploration requires a single "
            "Bayesian mixture measure."
        )
        world_model = self.dist.world_model
        weights = world_model.get_posterior_component_weights(self.dist.belief_state, params)
        component_values = np.array([
            world_model.get_component_expected_rewards(component, self.reward_function)
            for component in params.components
        ])
        return component_values, weights

    def values_for_component(self, component) -> np.ndarray:
        self._component_mixture_params(
            "Bayesian posterior component exploration requires a single "
            "Bayesian mixture measure."
        )
        return self.dist.world_model.get_component_expected_rewards(component, self.reward_function)

    def _component_mixture_params(self, multi_measure_message: str):
        if len(self.dist.measures) != 1:
            raise RuntimeError(multi_measure_message)
        params = self.dist.measures[0].params
        if not hasattr(params, "components"):
            raise RuntimeError(
                "Bayesian posterior component exploration requires finite "
                "component-mixture world-model parameters with a components "
                "attribute."
            )
        return params

    def _optimal_policy(self) -> np.ndarray:
        """Policy that maximises expected reward"""
        expected_rewards = self._expected_rewards()
        # Find all optimal policies, sum them up and normalise
        optimal_policies = np.isclose(expected_rewards, expected_rewards.max())
        return (self.policies*np.expand_dims(optimal_policies,axis=1)).sum(axis=0) / sum(optimal_policies)
