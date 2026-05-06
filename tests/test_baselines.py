"""Sanity checks for ThompsonSamplingAgent and UCB1Agent."""
import numpy as np
import pytest

from ibrl.simulators import simulate_multi_episode
from ibrl.environments.bandit import BanditEnvironment
from ibrl.agents.thompson import ThompsonSamplingAgent
from ibrl.agents.ucb import UCB1Agent
from ibrl.agents.bayesian import BayesianAgent
from ibrl.agents.q_learning import QLearningAgent


def _avg_cum_regret(agent_factory, env_seeds, n_steps=2000, num_actions=5):
    finals = []
    for seed in env_seeds:
        env = BanditEnvironment(num_actions=num_actions, seed=seed)
        agent = agent_factory(num_actions=num_actions, seed=seed)
        r = simulate_multi_episode(env, agent, num_episodes=1, steps_per_episode=n_steps, seed=seed)
        finals.append(r["cum_regret_flat"][-1])
    return float(np.mean(finals)), float(np.std(finals))


SEEDS = list(range(10))


class TestUCBBeatsEpsGreedy:
    def test_ucb_lower_regret_than_eps_greedy_qlearning(self):
        ucb_mean, _ = _avg_cum_regret(lambda **kw: UCB1Agent(**kw), SEEDS)
        ql_mean, _ = _avg_cum_regret(
            lambda **kw: QLearningAgent(epsilon=0.1, **kw), SEEDS)
        assert ucb_mean < ql_mean, (
            f"UCB1 cum regret {ucb_mean:.2f} should be lower than "
            f"epsilon-greedy QLearning cum regret {ql_mean:.2f}")


class TestThompsonBeatsGreedyBayesian:
    def test_thompson_lower_regret_than_greedy_bayesian(self):
        ts_mean, _ = _avg_cum_regret(lambda **kw: ThompsonSamplingAgent(**kw), SEEDS)
        # BayesianAgent is greedy and won't explore enough; Thompson posterior-samples.
        bayes_mean, _ = _avg_cum_regret(
            lambda **kw: BayesianAgent(epsilon=0.0, **kw), SEEDS)
        assert ts_mean < bayes_mean, (
            f"Thompson cum regret {ts_mean:.2f} should be lower than "
            f"greedy BayesianAgent cum regret {bayes_mean:.2f}")


class TestUCBExploresAllArms:
    def test_ucb_pulls_each_arm_at_least_once_in_first_k_steps(self):
        num_actions = 6
        env = BanditEnvironment(num_actions=num_actions, seed=0)
        agent = UCB1Agent(num_actions=num_actions, seed=0)
        r = simulate_multi_episode(env, agent, num_episodes=1, steps_per_episode=30, seed=0)
        first_k_actions = r["actions"][0, :num_actions]
        assert set(first_k_actions.tolist()) == set(range(num_actions)), (
            f"UCB1 should pull every arm in first {num_actions} steps; got {first_k_actions}")
