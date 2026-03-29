"""Belief classes — agent's epistemic models, independent of environments."""
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..outcome import Outcome


class BaseBelief(ABC):
    """Agent's epistemic model of an environment.

    Encapsulates prior assumptions, sufficient statistics, and update rule.
    Independent of any specific environment — the agent chooses what to
    believe, and the environment is indifferent to that choice.
    """

    @abstractmethod
    def update(self, action: int, outcome: Outcome):
        """Incorporate one observation into the sufficient statistics."""
        pass

    @abstractmethod
    def expected_reward_model(self) -> NDArray[np.float64]:
        """The agent's current estimate of the reward structure.

        Returns:
            NDArray — shape (num_actions,) for bandit-like beliefs,
            or shape (num_env_actions, num_actions) for game-like beliefs.
        """
        pass

    @abstractmethod
    def observation_probability(self, action: int, outcome: Outcome) -> float:
        """P(this observation) under the current belief.

        Must be called BEFORE update(), since update() changes the
        sufficient statistics that this method reads.

        Returns a probability in [0, 1].
        """
        pass

    @abstractmethod
    def copy(self) -> "BaseBelief":
        """Return an independent copy."""
        pass


class BernoulliBelief(BaseBelief):
    """Belief for i.i.d. Bernoulli rewards per arm.

    Well-specified for: BanditEnvironment
    Misspecified but ok: SwitchingAdversaryEnvironment (slowly adapts)
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.alpha = np.ones(num_actions)   # Beta prior alpha=1 (uniform)
        self.beta = np.ones(num_actions)    # Beta prior beta=1

    def update(self, action: int, outcome: Outcome):
        if (outcome.reward < 0) or (outcome.reward > 1):
            raise ValueError(f"BernoulliBelief expects reward in [0, 1], got {outcome.reward}")
        self.alpha[action] += outcome.reward
        self.beta[action] += 1.0 - outcome.reward

    def expected_reward_model(self) -> NDArray[np.float64]:
        return self.alpha / (self.alpha + self.beta)

    def observation_probability(self, action: int, outcome: Outcome) -> float:
        r = outcome.reward
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"BernoulliBelief expects reward in [0, 1], got {r}")
        p = self.alpha[action] / (self.alpha[action] + self.beta[action])
        return p ** r * (1 - p) ** (1 - r)

    def copy(self) -> "BernoulliBelief":
        c = BernoulliBelief.__new__(BernoulliBelief)
        c.num_actions = self.num_actions
        c.alpha = self.alpha.copy()
        c.beta = self.beta.copy()
        return c


class GaussianBelief(BaseBelief):
    """Belief using Gaussian precision-weighted updates per arm.

    Identical update math to BayesianAgent: tracks a mean and precision
    per arm, updates via precision-weighted averaging. This means
    InfraBayesianAgent(belief=GaussianBelief(n)) should produce identical
    behavior to BayesianAgent on bandit environments.
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.values = np.zeros(num_actions)
        self.precision = np.ones(num_actions) * 0.1

    def update(self, action: int, outcome: Outcome):
        r = outcome.reward
        self.values[action] = (
            self.precision[action] * self.values[action] + r
        ) / (self.precision[action] + 1.0)
        self.precision[action] += 1

    def expected_reward_model(self) -> NDArray[np.float64]:
        return self.values.copy()

    def observation_probability(self, action: int, outcome: Outcome) -> float:
        """Not yet implemented — required for KU but not for non-KU (single belief).

        Needs an assumed observation noise variance (sigma^2) to compute
        P(reward | action) = Normal(reward; values[action], sigma^2).
        Currently GaussianBelief only tracks the mean estimate's precision,
        not observation noise. Simplest path: add a fixed noise_var constructor
        parameter. Could also be estimated from data.
        """
        raise NotImplementedError(
            "GaussianBelief.observation_probability requires assumed "
            "noise variance — see docstring for details"
        )

    def copy(self) -> "GaussianBelief":
        c = GaussianBelief.__new__(GaussianBelief)
        c.num_actions = self.num_actions
        c.values = self.values.copy()
        c.precision = self.precision.copy()
        return c


class NewcombLikeBelief(BaseBelief):
    """Belief for deterministic reward matrices.

    Well-specified for: Newcomb, Damascus, AsymDamascus, Coordination, PDbandit
    """

    def __init__(self, num_actions: int, prior_mean: float = 0.5):
        self.num_actions = num_actions
        self.prior_mean = prior_mean
        self.observed = np.full((num_actions, num_actions), np.nan)

    def update(self, action: int, outcome: Outcome):
        self.observed[outcome.env_action, action] = outcome.reward

    def expected_reward_model(self) -> NDArray[np.float64]:
        model = self.observed.copy()
        model[np.isnan(model)] = self.prior_mean
        return model

    def observation_probability(self, action: int, outcome: Outcome) -> float:
        expected = self.observed[outcome.env_action, action]
        if np.isnan(expected):
            return 1.0  # unobserved cell — any outcome is "expected"
        return 1.0 if outcome.reward == expected else 0.0

    def copy(self) -> "NewcombLikeBelief":
        c = NewcombLikeBelief.__new__(NewcombLikeBelief)
        c.num_actions = self.num_actions
        c.prior_mean = self.prior_mean
        c.observed = self.observed.copy()
        return c


