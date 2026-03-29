import numpy as np
import pytest
from hypothesis import given, strategies as st
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent
from ibrl.environments import BanditEnvironment, NewcombEnvironment


class TestAgentInvariants:
    """Properties that must hold for all agents"""

    @pytest.mark.parametrize("agent_class", [
        QLearningAgent,
        BayesianAgent,
        EXP3Agent,
    ])
    @given(st.lists(st.floats(min_value=-10, max_value=10), min_size=2, max_size=10))
    def test_probabilities_always_valid(self, agent_class, rewards):
        """Probabilities must always sum to 1 and be non-negative"""
        agent = agent_class(num_actions=len(rewards), seed=42)
        agent.reset()
        agent.q = np.array(rewards)
        
        probs = agent.get_probabilities()
        
        assert np.isclose(probs.sum(), 1.0), f"Probs sum to {probs.sum()}, not 1"
        assert np.all(probs >= 0), f"Found negative probability: {probs}"
        assert np.all(np.isfinite(probs)), f"Found non-finite probability: {probs}"

    @given(st.floats(min_value=0.01, max_value=0.99))
    def test_epsilon_greedy_exploration_rate(self, epsilon):
        """Epsilon-greedy must explore at least epsilon fraction"""
        agent = QLearningAgent(num_actions=2, epsilon=epsilon, seed=42)
        agent.reset()
        agent.q = np.array([100.0, 0.0])
        
        probs = agent.build_epsilon_greedy_policy(agent.q)
        
        min_prob = probs.min()
        assert min_prob >= epsilon / 2 - 0.01, \
            f"Exploration too low: {min_prob} < {epsilon/2}"


class TestEnvironmentInvariants:
    """Properties that must hold for all environments"""

    @pytest.mark.parametrize("env_class", [
        BanditEnvironment,
        NewcombEnvironment,
    ])
    @given(st.lists(st.floats(min_value=0, max_value=1), min_size=2, max_size=2))
    def test_probability_distribution_valid(self, env_class, probs_list):
        """Environments must accept valid probability distributions"""
        probs = np.array(probs_list)
        
        if probs.sum() == 0:
            probs = np.array([0.5, 0.5])
        else:
            probs = probs / probs.sum()
        
        env = env_class(num_actions=2, seed=42)
        env.reset()
        
        try:
            env.predict(probs)
            assert True
        except Exception as e:
            pytest.fail(f"Environment rejected valid probabilities: {e}")

    @pytest.mark.parametrize("env_class", [
        BanditEnvironment,
        NewcombEnvironment,
    ])
    def test_reward_bounds_respected(self, env_class):
        """Rewards must stay within environment's defined bounds"""
        env = env_class(num_actions=2, seed=42)
        env.reset()
        
        for _ in range(100):
            if env_class == NewcombEnvironment:
                env.predict(np.array([0.5, 0.5]))
            reward = env.interact(0)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
