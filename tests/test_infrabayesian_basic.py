"""Smoke + behaviour tests for the vendored IB agent and beliefs.

This is a slim subset of the upstream tests/test_infrabayesian_beliefs.py
on origin/alaro/coin-learning-clean. The upstream version pulls in
BernoulliBanditEnvironment, the env.step() interface, and other API
surface that this branch does not vendor. We test the IB agent against
this branch's float-reward simulate() / simulate_multi_episode() and
exercise the non-KU short-circuit and the float-or-Outcome shim.
"""
import numpy as np
import pytest

from ibrl.outcome import Outcome
from ibrl.infrabayesian import (
    BernoulliBelief, GaussianBelief, NewcombLikeBelief, AMeasure, Infradistribution,
)
from ibrl.agents.infrabayesian import InfraBayesianAgent
from ibrl.environments.bandit import BanditEnvironment
from ibrl.simulators import simulate_multi_episode


class TestBeliefs:
    def test_bernoulli_initial_uniform(self):
        b = BernoulliBelief(num_actions=3)
        np.testing.assert_allclose(b.predict_rewards(), [0.5, 0.5, 0.5])

    def test_bernoulli_update_shifts_estimate(self):
        b = BernoulliBelief(num_actions=2)
        for _ in range(10):
            b.update(0, Outcome(reward=1.0))
        for _ in range(10):
            b.update(1, Outcome(reward=0.0))
        m = b.predict_rewards()
        assert m[0] > 0.8 and m[1] < 0.2

    def test_bernoulli_copy_is_independent(self):
        b = BernoulliBelief(num_actions=2)
        b.update(0, Outcome(reward=1.0))
        c = b.copy()
        c.update(0, Outcome(reward=1.0))
        assert b.predict_rewards()[0] != c.predict_rewards()[0]

    def test_gaussian_compute_outcome_probability_in_unit_interval(self):
        b = GaussianBelief(num_actions=2)
        # Density at the mean of N(0, 1) is 1/sqrt(2π) ≈ 0.3989 — under 1.
        assert b.compute_outcome_probability(0, Outcome(reward=0.0)) == pytest.approx(
            1.0 / np.sqrt(2.0 * np.pi), rel=1e-6)
        # Far-tail draw is floored at 1e-12, never zero.
        assert b.compute_outcome_probability(0, Outcome(reward=1e6)) == pytest.approx(1e-12)

    def test_newcomb_like_initial_prior_mean(self):
        b = NewcombLikeBelief(num_actions=2, prior_mean=0.5)
        np.testing.assert_allclose(b.predict_rewards(), 0.5 * np.ones((2, 2)))


class TestAMeasure:
    def test_evaluate_passes_through_at_unit_scale_zero_offset(self):
        b = BernoulliBelief(num_actions=3)
        m = AMeasure(b)
        np.testing.assert_allclose(m.evaluate(), b.predict_rewards())

    def test_evaluate_applies_scale_and_offset(self):
        b = BernoulliBelief(num_actions=2)
        m = AMeasure(b, scale=0.5, offset=0.1)
        np.testing.assert_allclose(m.evaluate(), 0.5 * b.predict_rewards() + 0.1)


class TestInfradistribution:
    def test_non_ku_evaluate_equals_single_measure_model(self):
        b = BernoulliBelief(num_actions=3)
        infra = Infradistribution([AMeasure(b)])
        np.testing.assert_allclose(infra.evaluate(), b.predict_rewards())

    def test_ku_evaluate_takes_min_across_measures(self):
        b1 = BernoulliBelief(num_actions=2)
        b2 = BernoulliBelief(num_actions=2)
        # Tilt b1 high on arm 0, b2 high on arm 1 — element-wise min is the lower of each.
        for _ in range(5):
            b1.update(0, Outcome(reward=1.0))
            b2.update(1, Outcome(reward=1.0))
        infra = Infradistribution([AMeasure(b1), AMeasure(b2)])
        # min should be the worse-case prediction per arm
        m = infra.evaluate()
        assert m[0] == min(b1.predict_rewards()[0], b2.predict_rewards()[0])
        assert m[1] == min(b1.predict_rewards()[1], b2.predict_rewards()[1])


class TestInfraBayesianAgentShim:
    def test_accepts_float_reward(self):
        agent = InfraBayesianAgent(num_actions=3, seed=0, beliefs=[GaussianBelief(3)])
        agent.reset()
        agent.update(np.array([1/3, 1/3, 1/3]), action=0, reward_or_outcome=2.0)
        belief = agent.infradist.measures[0].belief
        assert belief.values[0] != 0.0, "belief should have absorbed the float reward"

    def test_accepts_outcome_object(self):
        agent = InfraBayesianAgent(num_actions=3, seed=0, beliefs=[GaussianBelief(3)])
        agent.reset()
        agent.update(np.array([1/3, 1/3, 1/3]), action=0,
                     reward_or_outcome=Outcome(reward=2.0))
        belief = agent.infradist.measures[0].belief
        assert belief.values[0] != 0.0

    def test_non_ku_matches_bayesian_on_stationary_bandit(self):
        from ibrl.agents.bayesian import BayesianAgent
        env = BanditEnvironment(num_actions=4, seed=42)
        ib = InfraBayesianAgent(num_actions=4, seed=7, beliefs=[GaussianBelief(4)])
        r_ib = simulate_multi_episode(env, ib, num_episodes=2, steps_per_episode=50, seed=1)

        env2 = BanditEnvironment(num_actions=4, seed=42)
        bayes = BayesianAgent(num_actions=4, seed=7)
        r_bayes = simulate_multi_episode(env2, bayes, num_episodes=2, steps_per_episode=50, seed=1)

        # Same seeds → identical reward streams, but action selection may differ
        # because exploration policies differ (IBAgent inherits BaseGreedyAgent).
        # Final mean values should match closely since update math is identical.
        np.testing.assert_allclose(
            ib.infradist.measures[0].belief.values,
            bayes.values, rtol=1e-9, atol=1e-9,
            err_msg="non-KU IB with GaussianBelief should track BayesianAgent's posterior")
