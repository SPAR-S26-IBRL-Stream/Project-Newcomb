"""Tests for simulate_multi_episode and the reset_belief / reset_episode split."""
import numpy as np
import pytest

from ibrl.simulators import simulate_multi_episode
from ibrl.environments.bandit import BanditEnvironment
from ibrl.agents.bayesian import BayesianAgent


def _make_pair(seed_env=42, seed_agent=7, num_actions=4):
    return (
        BanditEnvironment(num_actions=num_actions, seed=seed_env),
        BayesianAgent(num_actions=num_actions, seed=seed_agent),
    )


class TestBeliefPersistsAcrossEpisodes:
    def test_values_change_across_episodes_when_belief_persists(self):
        env, agent = _make_pair()
        simulate_multi_episode(env, agent, num_episodes=2, steps_per_episode=20, seed=1)
        end_ep1_values = agent.values.copy()
        end_ep1_precision = agent.precision.copy()

        env2, agent2 = _make_pair()
        simulate_multi_episode(env2, agent2, num_episodes=4, steps_per_episode=20, seed=1)
        end_ep3_values = agent2.values
        end_ep3_precision = agent2.precision

        assert not np.allclose(end_ep1_values, end_ep3_values), (
            "values should differ between 2-episode and 4-episode runs since belief carries forward")
        assert end_ep3_precision.sum() > end_ep1_precision.sum(), (
            "precision should accumulate as more observations arrive")

    def test_total_precision_grows_linearly_with_total_steps(self):
        env, agent = _make_pair()
        n_eps, t = 5, 30
        simulate_multi_episode(env, agent, num_episodes=n_eps, steps_per_episode=t, seed=1)
        # Each step adds exactly 1 to one arm's precision; initial precision is 0.1 per arm.
        expected_total = agent.num_actions * 0.1 + n_eps * t
        assert agent.precision.sum() == pytest.approx(expected_total)


class TestResetBeliefControl:
    def test_reset_belief_re_zeros_state_each_episode(self):
        env, agent = _make_pair()
        simulate_multi_episode(env, agent, num_episodes=3, steps_per_episode=20,
                               seed=1, reset_agent_belief=True)
        # After last episode, agent has precision from only the most recent episode
        # (initial 0.1 per arm + 20 single-step increments distributed across arms).
        expected_total = agent.num_actions * 0.1 + 20
        assert agent.precision.sum() == pytest.approx(expected_total)

    def test_reset_belief_yields_higher_regret_than_persisting(self):
        env_p, agent_p = _make_pair()
        rp = simulate_multi_episode(env_p, agent_p, num_episodes=8, steps_per_episode=40, seed=1)

        env_r, agent_r = _make_pair()
        rr = simulate_multi_episode(env_r, agent_r, num_episodes=8, steps_per_episode=40, seed=1,
                                    reset_agent_belief=True)

        late = slice(-3, None)
        assert rp["per_episode_regret"][late].mean() < rr["per_episode_regret"][late].mean(), (
            "persisting belief should yield lower regret on late episodes than re-zeroing each episode")


class TestReturnShapes:
    def test_returned_arrays_have_correct_shapes(self):
        env, agent = _make_pair()
        n_eps, t = 3, 15
        r = simulate_multi_episode(env, agent, num_episodes=n_eps, steps_per_episode=t, seed=1,
                                   snapshot_belief=True)
        assert r["rewards"].shape == (n_eps, t)
        assert r["actions"].shape == (n_eps, t)
        assert r["probabilities"].shape == (n_eps, t, agent.num_actions)
        assert r["optimal_per_episode"].shape == (n_eps,)
        assert r["regret_per_step"].shape == (n_eps, t)
        assert r["cum_regret_flat"].shape == (n_eps * t,)
        assert r["per_episode_regret"].shape == (n_eps,)
        assert len(r["belief_snapshots"]) == n_eps
