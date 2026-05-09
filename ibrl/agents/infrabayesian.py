"""InfraBayesianAgent — agent using infrabayesian inference."""
import itertools
import numpy as np

from . import BaseGreedyAgent
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
        # Greedy policy: reproduces classical agent, breaks regret bounds
        if self.exploration_prefix is None:
            return self.build_greedy_policy(self._optimal_policy())

        # Forced exploration prefix: regret bounds asymptotically preserved
        if self.step <= self.exploration_prefix:
            return np.ones(self.num_actions) / self.num_actions

        # Use optimal policy: no exploration, almost no learning
        return self._optimal_policy()

    def dump_state(self) -> str:
        state = str(self.dist.belief_state)
        if self.verbose > 1:
            state += ";" + repr(self.dist)
        return state

    def _expected_rewards(self) -> np.ndarray:
        """Expected reward of each policy"""
        # Iterate over policies and compute expected reward: E[π] = Σ_a E_π[a] π(a)
        return np.array([sum(
            self.dist.evaluate_action(self.reward_function[a], a, policy) * policy[a]
                for a in range(self.num_actions)
            ) for policy in self.policies
        ])

    def _optimal_policy(self) -> np.ndarray:
        """Policy that maximises expected reward"""
        expected_rewards = self._expected_rewards()
        # Find all optimal policies, sum them up and normalise
        optimal_policies = np.isclose(expected_rewards, expected_rewards.max())
        return (self.policies*np.expand_dims(optimal_policies,axis=1)).sum(axis=0) / sum(optimal_policies)
