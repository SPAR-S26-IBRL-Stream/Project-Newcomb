"""SupraPOMDPAgent — IB agent with multi-step planning via compute_q_values."""
from __future__ import annotations

import numpy as np

from .infrabayesian import InfraBayesianAgent


class SupraPOMDPAgent(InfraBayesianAgent):
    """InfraBayesianAgent subclass for supra-POMDP world models.

    Overrides _expected_rewards to use world_model.compute_q_values for
    action selection. The candidate policy from the optimizer's loop is
    threaded directly into compute_q_values so that policy-dependent
    kernels (T(π), B(π), θ₀(π)) see the correct committed policy —
    the mechanism that enforces pseudocausal coupling in Newcomb-family
    environments.

    IB belief updating (Infradistribution.update) continues to use
    compute_expected_reward (one-step E[rf[next_obs]]) for normalization.
    """

    def _expected_rewards(self) -> np.ndarray:
        wm = self.dist.world_model
        params = self.dist.measures[0].params

        expected_rewards = np.array([
            float(np.dot(
                wm.compute_q_values(self.dist.belief_state, params, policy=policy),
                policy,
            ))
            for policy in self.policies
        ])

        optimal = np.isclose(expected_rewards, expected_rewards.max())
        return (
            np.array(self.policies) * np.expand_dims(optimal, axis=1)
        ).sum(axis=0) / optimal.sum()
