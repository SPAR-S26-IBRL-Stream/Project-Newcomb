"""InfraBayesianAgent — agent using infrabayesian inference."""
import numpy as np
from numpy.typing import NDArray

from . import BaseGreedyAgent
from ..infrabayesian.a_measure import AMeasure
from ..infrabayesian.infradistribution import Infradistribution


class InfraBayesianAgent(BaseGreedyAgent):
    """
    Agent using infrabayesian inference.

    Maintains a single shared infradistribution (a Bayesian prior over hypotheses),
    updated on every step regardless of which action was taken. Each hypothesis is an
    Infradistribution over a shared WorldModel; the prior is a 1D array of weights.

    reward_function has shape (num_actions, num_outcomes): reward_function[a, o] is
    the reward when action a produces outcome o.
    """
    def __init__(self, *args,
            hypotheses : list[Infradistribution],
            prior : np.ndarray | None = None,
            reward_function : np.ndarray | None = None,
            policy_discretisation : int = 0,
            **kwargs):
        super().__init__(*args, **kwargs)
        assert len(hypotheses) > 0
        assert all(isinstance(h.world_model, type(hypotheses[0].world_model))
                   for h in hypotheses), "All hypotheses must share the same WorldModel type"
        self.hypotheses = hypotheses
        self.prior = prior if prior is not None else np.ones(len(hypotheses)) / len(hypotheses)
        self.reward_function = (reward_function if reward_function is not None
                                else np.tile([0., 1.], (self.num_actions, 1)))

        # TODO should be possible to generalise this to more actions
        assert self.num_actions == 2
        self.policies = [
            np.array([1-p, p])
            for p in np.linspace(0., 1., 2 + policy_discretisation)
        ]

    def reset(self):
        super().reset()
        self.dist = Infradistribution.mix(self.hypotheses, self.prior)

    def update(self, probabilities: NDArray[np.float64], action: int, outcome) -> None:
        super().update(probabilities, action, outcome)
        self.dist.update(self.reward_function, outcome, action=action, policy=probabilities)

    def get_probabilities(self) -> NDArray[np.float64]:
        # TODO handle this better
        if len(self.policies) == 2:
            # For validation against classical agent: return greedy policy
            return self.build_greedy_policy(self._expected_rewards())
        else:
            # For optimal IB behaviour: just return optimal policy (without exploration)
            return self._expected_rewards()

    def dump_state(self) -> str:
        state = str(self.dist.belief_state)
        if self.verbose > 1:
            state += ";" + repr(self.dist)
        return state

    def _expected_rewards(self) -> np.ndarray:
        # Iterate over policies and compute expected reward: E[π] = Σ_a E_π[a] π(a)
        expected_rewards = np.array([sum(
            self.dist.evaluate_action(self.reward_function[a], a, policy) * policy[a]
                for a in range(self.num_actions)
            ) for policy in self.policies
        ])

        # Find all optimal policies, sum them up and normalise
        optimal_policies = (expected_rewards > expected_rewards.max()*0.999999)
        return (self.policies*np.expand_dims(optimal_policies,axis=1)).sum(axis=0) / sum(optimal_policies)
