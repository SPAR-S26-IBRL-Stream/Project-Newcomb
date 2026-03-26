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
    def update(self, action: int, outcome: Outcome, context: dict | None = None):
        """Incorporate one observation into the sufficient statistics."""
        pass

    @abstractmethod
    def expected_reward_model(self, context: dict | None = None) -> NDArray[np.float64]:
        """The agent's current estimate of the reward structure.

        Returns:
            NDArray — shape (num_actions,) for bandit-like beliefs,
            or shape (num_env_actions, num_actions) for game-like beliefs.
        """
        pass

    @abstractmethod
    def copy(self) -> "BaseBelief":
        """Return an independent copy."""
        pass


class BanditBelief(BaseBelief):
    """Belief for i.i.d. Bernoulli rewards per arm.

    Well-specified for: BanditEnvironment
    Misspecified but ok: SwitchingAdversaryEnvironment (slowly adapts)
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.alpha = np.ones(num_actions)   # Beta prior alpha=1 (uniform)
        self.beta = np.ones(num_actions)    # Beta prior beta=1

    def update(self, action: int, outcome: Outcome, context: dict | None = None):
        self.alpha[action] += outcome.reward
        self.beta[action] += 1.0 - outcome.reward

    def expected_reward_model(self, context: dict | None = None) -> NDArray[np.float64]:
        return self.alpha / (self.alpha + self.beta)

    def copy(self) -> "BanditBelief":
        c = BanditBelief.__new__(BanditBelief)
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

    def update(self, action: int, outcome: Outcome, context: dict | None = None):
        r = outcome.reward
        self.values[action] = (
            self.precision[action] * self.values[action] + r
        ) / (self.precision[action] + 1.0)
        self.precision[action] += 1

    def expected_reward_model(self, context: dict | None = None) -> NDArray[np.float64]:
        return self.values.copy()

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

    def update(self, action: int, outcome: Outcome, context: dict | None = None):
        self.observed[outcome.env_action, action] = outcome.reward

    def expected_reward_model(self, context: dict | None = None) -> NDArray[np.float64]:
        model = self.observed.copy()
        model[np.isnan(model)] = self.prior_mean
        return model

    def copy(self) -> "NewcombLikeBelief":
        c = NewcombLikeBelief.__new__(NewcombLikeBelief)
        c.num_actions = self.num_actions
        c.prior_mean = self.prior_mean
        c.observed = self.observed.copy()
        return c


class SwitchingBelief(BaseBelief):
    """Belief for Bernoulli rewards with a single unknown switch time.

    Well-specified for: SwitchingAdversaryEnvironment
    Over-parameterized but ok: BanditEnvironment (concentrates on "never switches")
    """

    def __init__(self, num_actions: int, max_steps: int):
        self.num_actions = num_actions
        self.max_steps = max_steps
        T, K = max_steps, num_actions

        self.log_weights = np.zeros(T)          # uniform prior over switch times
        self.alpha_before = np.ones((T, K))     # Beta stats before switch
        self.beta_before = np.ones((T, K))
        self.alpha_after = np.ones((T, K))      # Beta stats after switch
        self.beta_after = np.ones((T, K))

    def update(self, action: int, outcome: Outcome, context: dict | None = None):
        step = context['step']
        r = outcome.reward

        for t in range(self.max_steps):
            if step < t:
                p = self.alpha_before[t, action] / (
                    self.alpha_before[t, action] + self.beta_before[t, action])
                self.log_weights[t] += r * np.log(p + 1e-300) + (1 - r) * np.log(1 - p + 1e-300)
                self.alpha_before[t, action] += r
                self.beta_before[t, action] += 1.0 - r
            else:
                p = self.alpha_after[t, action] / (
                    self.alpha_after[t, action] + self.beta_after[t, action])
                self.log_weights[t] += r * np.log(p + 1e-300) + (1 - r) * np.log(1 - p + 1e-300)
                self.alpha_after[t, action] += r
                self.beta_after[t, action] += 1.0 - r

    def expected_reward_model(self, context: dict | None = None) -> NDArray[np.float64]:
        step = context['step']
        log_w = self.log_weights - self.log_weights.max()
        weights = np.exp(log_w)
        weights /= weights.sum()

        model = np.zeros(self.num_actions)
        for t in range(self.max_steps):
            for a in range(self.num_actions):
                if step < t:
                    p = self.alpha_before[t, a] / (
                        self.alpha_before[t, a] + self.beta_before[t, a])
                else:
                    p = self.alpha_after[t, a] / (
                        self.alpha_after[t, a] + self.beta_after[t, a])
                model[a] += weights[t] * p
        return model

    def copy(self) -> "SwitchingBelief":
        c = SwitchingBelief.__new__(SwitchingBelief)
        c.num_actions = self.num_actions
        c.max_steps = self.max_steps
        c.log_weights = self.log_weights.copy()
        c.alpha_before = self.alpha_before.copy()
        c.beta_before = self.beta_before.copy()
        c.alpha_after = self.alpha_after.copy()
        c.beta_after = self.beta_after.copy()
        return c
