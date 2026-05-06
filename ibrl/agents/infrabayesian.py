"""InfraBayesianAgent — agent using infrabayesian inference."""
import itertools
import numpy as np

from . import BaseGreedyAgent
from ..exploration import ExplorationStrategy
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
        exploration_prefix:     parameter to control exploration
                                =0    no exploration
                                >0    forced exploration prefix for given number of steps, then no exploration
                                None  greedy exploration (epsilon or softmax; breaks regret bounds)
    """
    def __init__(self, *,
            hypotheses : list[Infradistribution],
            prior : np.ndarray | None = None,           # shape (len(hypotheses),)
            reward_function : np.ndarray | None = None, # shape (num_actions, num_outcomes)
            policy_discretisation : int = 0,
            exploration_prefix : int | None = 0,
            exploration_strategy : ExplorationStrategy | None = None,
            **kwargs):
        super().__init__(**kwargs)
        assert len(hypotheses) > 0
        assert all(isinstance(h.world_model, type(hypotheses[0].world_model))
                   for h in hypotheses), "All hypotheses must share the same WorldModel type"
        self.hypotheses = hypotheses
        self.prior = prior if prior is not None else np.ones(len(hypotheses)) / len(hypotheses)
        # default: reward_function[a,o] = o with o ∈ {0,1}
        self.reward_function = (reward_function if reward_function is not None
                                else np.linspace(np.zeros(self.num_actions),np.ones(self.num_actions),2).T)
        self.exploration_prefix = exploration_prefix
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
        self.action_counts = np.zeros(self.num_actions, dtype=np.int64)
        self.reward_sums = np.zeros(self.num_actions)
        self.empirical_values = np.zeros(self.num_actions)

    def update(self, probabilities: np.ndarray, action: int, outcome) -> None:
        super().update(probabilities, action, outcome)
        self.action_counts[action] += 1
        self.reward_sums[action] += outcome.reward
        self.empirical_values[action] = self.reward_sums[action] / self.action_counts[action]
        self.dist.update(self.reward_function, outcome, action, probabilities)

    def get_probabilities(self) -> np.ndarray:
        if self.exploration_strategy is not None:
            return self.exploration_strategy.get_probabilities(self, self._action_values())

        # Greedy policy: reproduces classical agent, breaks regret bounds
        if self.exploration_prefix is None:
            return self.build_greedy_policy(self._expected_rewards())

        # Forced exploration prefix: regret bounds asymptotically preserved
        if self.step < self.exploration_prefix:
            return np.ones(self.num_actions) / self.num_actions

        # Use optimal policy: no exploration, almost no learning
        return self._expected_rewards()

    def dump_state(self) -> str:
        state = str(self.dist.belief_state)
        if self.verbose > 1:
            state += ";" + repr(self.dist)
        return state

    def _expected_rewards(self) -> np.ndarray:
        expected_rewards = np.array([
            sum(self._action_values_for_policy(policy) * policy)
            for policy in self.policies
        ])

        # Find all optimal policies, sum them up and normalise
        optimal_policies = np.isclose(expected_rewards, expected_rewards.max())
        return (self.policies*np.expand_dims(optimal_policies,axis=1)).sum(axis=0) / sum(optimal_policies)

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
        measure = self.dist.measures[0]
        world_model = self.dist.world_model
        params = measure.params
        weights = world_model.get_posterior_component_weights(self.dist.belief_state, params)
        idx = int(self.random.choice(len(params.components), p=weights))
        return params.components[idx]

    def values_for_component(self, component) -> np.ndarray:
        return self.dist.world_model.get_component_expected_rewards(component, self.reward_function)
