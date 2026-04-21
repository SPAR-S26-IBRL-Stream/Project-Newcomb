"""InfraBayesianAgent — agent using infrabayesian inference."""
import numpy as np
from numpy.typing import NDArray

from . import BaseGreedyAgent
from ..infrabayesian.a_measure import AMeasure
from ..infrabayesian.infradistribution import Infradistribution
from ..infrabayesian.world_model import MultiBernoulliWorldModel


class InfraBayesianAgent(BaseGreedyAgent):
    """
    Agent using infrabayesian inference with an explicit WorldModel.

    A single shared infradistribution is maintained over all actions. Observations
    under any action update the shared belief, so evidence from arm 0 informs
    predictions for arm 1 (and vice versa) through the shared hypothesis prior.

    Use InfraBayesianAgent.bernoulli_grid() for the default Bernoulli bandit setup
    matching the old num_hypotheses interface.
    """

    def __init__(self, *args,
                 hypotheses: list[Infradistribution],
                 prior: np.ndarray | None = None,
                 reward_function: np.ndarray | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert len(hypotheses) > 0
        assert all(
            isinstance(h.world_model, type(hypotheses[0].world_model))
            for h in hypotheses
        ), "All hypotheses must share the same WorldModel type"
        self.hypotheses = hypotheses
        self.prior = (
            prior if prior is not None
            else np.ones(len(hypotheses)) / len(hypotheses)
        )
        self.reward_function = (
            reward_function if reward_function is not None
            else np.tile([0., 1.], (self.num_actions, 1))
        )

    @classmethod
    def bernoulli_grid(cls, num_actions: int, num_hypotheses: int = 5,
                       prior: np.ndarray | None = None,
                       **kwargs) -> "InfraBayesianAgent":
        """
        Convenience constructor: uniform grid of Bernoulli hypotheses.
        Matches the old default num_hypotheses behaviour and is equivalent to
        DiscreteBayesianAgent (V(π) is linear for MABs so argmax is optimal).
        """
        wm = MultiBernoulliWorldModel(num_arms=num_actions)
        grid = [np.array([1 - p, p]) for p in np.linspace(0., 1., num_hypotheses)]
        # Each arm gets the same N-point hypothesis grid independently.
        # A single a-measure with N components per arm gives correct independent
        # per-arm Bayesian inference — no joint enumeration needed.
        params = wm.make_params([grid] * num_actions)
        hypotheses = [Infradistribution([AMeasure(params)], world_model=wm)]
        return cls(
            num_actions=num_actions,
            hypotheses=hypotheses,
            prior=np.array([1.0]),
            **kwargs,
        )

    def reset(self):
        super().reset()
        self.dist = Infradistribution.mix(self.hypotheses, self.prior)

    def update(self, probabilities: NDArray[np.float64], action: int, outcome) -> None:
        super().update(probabilities, action, outcome)
        self.dist.update(
            self.reward_function, outcome, action=action, policy=probabilities
        )

    def get_probabilities(self) -> NDArray[np.float64]:
        rewards = np.array([
            self.dist.evaluate_action(self.reward_function[a], action=a)
            for a in range(self.num_actions)
        ])
        return self.build_greedy_policy(rewards)

    def dump_state(self) -> str:
        return str(self.dist.belief_state)
