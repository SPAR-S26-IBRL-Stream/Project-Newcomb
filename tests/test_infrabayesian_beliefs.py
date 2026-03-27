"""Tests for belief-based infrabayesian agent (Phase 3)."""
import numpy as np
import pytest

from ibrl.outcome import Outcome
from ibrl.infrabayesian.beliefs import (
    BaseBelief, BernoulliBelief, GaussianBelief, NewcombLikeBelief,
)
from ibrl.infrabayesian.belief_a_measure import BeliefAMeasure
from ibrl.infrabayesian.belief_infradistribution import BeliefInfradistribution
from ibrl.agents.infrabayesian import InfraBayesianAgent
from ibrl.environments.bandit import BanditEnvironment
from ibrl.environments.bernoulli_bandit import BernoulliBanditEnvironment
from ibrl.environments.newcomb import NewcombEnvironment
from ibrl.environments.switching import SwitchingAdversaryEnvironment
from ibrl.simulators.simulator import simulate
from ibrl.utils import sample_action


# ── BernoulliBelief ─────────────────────────────────────────────────────────────

class TestBernoulliBelief:
    def test_initial_model_is_uniform(self):
        b = BernoulliBelief(num_actions=3)
        model = b.expected_reward_model()
        np.testing.assert_allclose(model, [0.5, 0.5, 0.5])

    def test_update_shifts_estimate(self):
        b = BernoulliBelief(num_actions=2)
        # Observe arm 0 succeeding 10 times
        for _ in range(10):
            b.update(action=0, outcome=Outcome(reward=1.0))
        # Observe arm 1 failing 10 times
        for _ in range(10):
            b.update(action=1, outcome=Outcome(reward=0.0))

        model = b.expected_reward_model()
        assert model[0] > 0.8  # arm 0 should be estimated high
        assert model[1] < 0.2  # arm 1 should be estimated low

    def test_model_shape_is_1d(self):
        b = BernoulliBelief(num_actions=4)
        assert b.expected_reward_model().shape == (4,)

    def test_copy_is_independent(self):
        b = BernoulliBelief(num_actions=2)
        b.update(action=0, outcome=Outcome(reward=1.0))
        c = b.copy()
        c.update(action=0, outcome=Outcome(reward=1.0))
        # Original should not be affected
        assert b.expected_reward_model()[0] != c.expected_reward_model()[0]


# ── NewcombLikeBelief ────────────────────────────────────────────────────────

class TestNewcombLikeBelief:
    def test_initial_model_is_prior_mean(self):
        b = NewcombLikeBelief(num_actions=2, prior_mean=0.5)
        model = b.expected_reward_model()
        np.testing.assert_allclose(model, [[0.5, 0.5], [0.5, 0.5]])

    def test_update_records_observation(self):
        b = NewcombLikeBelief(num_actions=2)
        b.update(action=0, outcome=Outcome(reward=1.0, env_action=0))
        model = b.expected_reward_model()
        assert model[0, 0] == 1.0
        # Other cells should still be prior
        assert model[0, 1] == 0.5
        assert model[1, 0] == 0.5

    def test_model_shape_is_2d(self):
        b = NewcombLikeBelief(num_actions=3)
        assert b.expected_reward_model().shape == (3, 3)

    def test_copy_is_independent(self):
        b = NewcombLikeBelief(num_actions=2)
        b.update(action=0, outcome=Outcome(reward=1.0, env_action=0))
        c = b.copy()
        c.update(action=1, outcome=Outcome(reward=0.0, env_action=1))
        assert np.isnan(b.observed[1, 1])  # original unaffected


# ── BeliefAMeasure ───────────────────────────────────────────────────────────

class TestBeliefAMeasure:
    def test_passthrough_with_unit_scale_zero_offset(self):
        """With lambda=1, b=0, BeliefAMeasure is a pure pass-through."""
        belief = BernoulliBelief(num_actions=2)
        belief.update(action=0, outcome=Outcome(reward=1.0))
        bam = BeliefAMeasure(belief)
        np.testing.assert_allclose(
            bam.expected_reward_model(),
            belief.expected_reward_model(),
        )

    def test_scale_and_offset_applied(self):
        belief = BernoulliBelief(num_actions=2)
        bam = BeliefAMeasure(belief, log_scale=np.log(2.0), offset=0.1)
        model = bam.expected_reward_model()
        expected = 2.0 * belief.expected_reward_model() + 0.1
        np.testing.assert_allclose(model, expected)


# ── BeliefInfradistribution ─────────────────────────────────────────────────

class TestBeliefInfradistribution:
    def test_single_measure_passthrough(self):
        """Non-KU: single measure, should match belief directly."""
        belief = BernoulliBelief(num_actions=2)
        belief.update(action=0, outcome=Outcome(reward=1.0))
        bam = BeliefAMeasure(belief)
        infradist = BeliefInfradistribution([bam])
        np.testing.assert_allclose(
            infradist.expected_reward_model(),
            belief.expected_reward_model(),
        )

    def test_update_propagates_to_belief(self):
        belief = BernoulliBelief(num_actions=2)
        bam = BeliefAMeasure(belief)
        infradist = BeliefInfradistribution([bam])
        infradist.update(action=0, outcome=Outcome(reward=1.0))
        # Belief should have been updated
        model = infradist.expected_reward_model()
        assert model[0] > 0.5  # arm 0 should be higher after success


# ── IB pipe vs direct belief equivalence ─────────────────────────────────────

class TestBanditEquivalence:
    """InfraBayesianAgent with BernoulliBelief should produce identical reward
    models to using BernoulliBelief directly, at every step."""

    def test_ib_pipe_matches_direct_belief(self):
        belief = BernoulliBelief(num_actions=3)
        agent = InfraBayesianAgent(num_actions=3, beliefs=[BernoulliBelief(num_actions=3)], epsilon=0.1)
        agent.reset()

        # Feed same observations to both
        rng = np.random.default_rng(42)
        for step in range(1, 51):
            action = rng.integers(3)
            reward = float(rng.random() < 0.7) if action == 0 else float(rng.random() < 0.3)
            outcome = Outcome(reward=reward)

            belief.update(action=action, outcome=outcome)

            probs = agent.get_probabilities()
            agent.update(probs, action, outcome)

            # Compare reward models
            direct_model = belief.expected_reward_model()
            agent_model = agent.infradist.expected_reward_model(context={'step': agent.step})
            np.testing.assert_allclose(
                agent_model, direct_model, atol=1e-12,
                err_msg=f"Mismatch at step {step}",
            )


class TestNewcombLikeEquivalence:
    """InfraBayesianAgent with NewcombLikeBelief should match direct belief."""

    def test_ib_pipe_matches_direct_belief(self):
        belief = NewcombLikeBelief(num_actions=2)
        agent = InfraBayesianAgent(
            num_actions=2, beliefs=[NewcombLikeBelief(num_actions=2)], epsilon=0.1
        )
        agent.reset()

        observations = [
            (0, Outcome(reward=1.0, env_action=0)),
            (1, Outcome(reward=0.1, env_action=0)),
            (0, Outcome(reward=0.0, env_action=1)),
            (1, Outcome(reward=0.1, env_action=1)),
        ]

        for action, outcome in observations:
            belief.update(action=action, outcome=outcome)

            probs = agent.get_probabilities()
            agent.update(probs, action, outcome)

            direct_model = belief.expected_reward_model()
            agent_model = agent.infradist.expected_reward_model()
            np.testing.assert_allclose(agent_model, direct_model, atol=1e-12)


# ── GaussianBelief ↔ BayesianAgent equivalence ───────────────────────────────

class TestGaussianBeliefMatchesBayesianAgent:
    """IB agent with GaussianBelief should produce identical rewards to BayesianAgent
    on the same bandit environment with the same seed and epsilon."""

    def test_identical_rewards_on_bandit(self):
        from ibrl.agents import BayesianAgent

        seed = 42
        n = 3
        env_seed = 99
        opts = {"num_steps": 100, "num_runs": 5}

        env = BanditEnvironment(num_actions=n, seed=env_seed)
        bayesian = BayesianAgent(num_actions=n, seed=seed, epsilon=0.1)
        r_bay = simulate(env, bayesian, opts)

        env = BanditEnvironment(num_actions=n, seed=env_seed)
        ib = InfraBayesianAgent(
            num_actions=n, seed=seed,
            beliefs=[GaussianBelief(num_actions=n)], epsilon=0.1,
        )
        r_ib = simulate(env, ib, opts)

        np.testing.assert_array_equal(
            r_bay["rewards"], r_ib["rewards"],
            err_msg="IB+GaussianBelief should produce identical rewards to BayesianAgent",
        )
        np.testing.assert_array_equal(
            r_bay["actions"], r_ib["actions"],
            err_msg="IB+GaussianBelief should produce identical actions to BayesianAgent",
        )


# ── Simulator integration ───────────────────────────────────────────────────

class TestSimulatorIntegration:
    def test_bandit_agent_learns(self):
        """IB agent on bandit: average reward should increase over time."""
        env = BernoulliBanditEnvironment(num_actions=3, seed=42)
        agent = InfraBayesianAgent(
            num_actions=3, beliefs=[BernoulliBelief(num_actions=3)],
            epsilon=(0.5, 0.5, 0.01), seed=123,
        )
        results = simulate(env, agent, {"num_steps": 200, "num_runs": 5})
        # Compare first 20 steps vs last 20 steps
        early_reward = results["average_reward"][0, :20].mean()
        late_reward = results["average_reward"][0, -20:].mean()
        assert late_reward > early_reward, (
            f"Agent didn't learn: early={early_reward:.3f}, late={late_reward:.3f}"
        )

    def test_newcomb_agent_runs(self):
        """IB agent on Newcomb: should complete without errors."""
        env = NewcombEnvironment(num_actions=2, seed=42)
        agent = InfraBayesianAgent(
            num_actions=2, beliefs=[NewcombLikeBelief(num_actions=2)],
            epsilon=0.1, seed=123,
        )
        results = simulate(env, agent, {"num_steps": 50, "num_runs": 2})
        assert results["rewards"].shape == (2, 50)


# ── SwitchingAdversary Bernoulli fix ─────────────────────────────────────────


