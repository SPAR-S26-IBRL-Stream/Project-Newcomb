import numpy as np
import pytest
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent
from ibrl.environments import BanditEnvironment, NewcombEnvironment


class TestEdgeCases:
    """Test behavior under extreme/degenerate conditions"""

    def test_agent_handles_zero_rewards(self):
        """Agent should not crash when all rewards are zero"""
        env = BanditEnvironment(num_actions=2, seed=42)
        env.reset()
        agent = QLearningAgent(num_actions=2, seed=43)
        agent.reset()
        
        env.rewards = np.array([0.0, 0.0])
        
        for _ in range(50):
            probs = agent.get_probabilities()
            action = np.argmax(probs)
            reward = env.interact(action)
            agent.update(probs, action, reward)
        
        assert np.all(np.isfinite(agent.q))

    def test_agent_handles_extreme_rewards(self):
        """Agent should not crash with very large rewards"""
        env = BanditEnvironment(num_actions=2, seed=42)
        env.reset()
        agent = QLearningAgent(num_actions=2, learning_rate=0.01, seed=43)
        agent.reset()
        
        env.rewards = np.array([1e6, 1e6])
        
        for _ in range(50):
            probs = agent.get_probabilities()
            action = np.argmax(probs)
            reward = env.interact(action)
            agent.update(probs, action, reward)
        
        assert np.all(np.isfinite(agent.q)), "Q-values became non-finite"

    def test_deterministic_policy_is_valid(self):
        """Deterministic policies (prob=1 for one action) should be valid"""
        agent = QLearningAgent(num_actions=2, seed=42)
        agent.reset()
        agent.q = np.array([1e10, 0.0])
        
        probs = agent.build_epsilon_greedy_policy(agent.q)
        
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
        assert np.all(np.isfinite(probs))

    def test_newcomb_with_extreme_probabilities(self):
        """Newcomb should handle edge case probabilities"""
        env = NewcombEnvironment(num_actions=2, seed=42)
        env.reset()
        
        for probs in [[1.0, 0.0], [0.0, 1.0]]:
            env.predict(np.array(probs))
            reward = env.interact(0)
            assert np.isfinite(reward)

    def test_bayesian_handles_extreme_rewards(self):
        """Bayesian agent should handle extreme rewards"""
        agent = BayesianAgent(num_actions=2, seed=42)
        agent.reset()
        
        for _ in range(50):
            probs = agent.get_probabilities()
            agent.update(probs, 0, 1e6)
        
        assert np.all(np.isfinite(agent.values))

    def test_exp3_handles_extreme_rewards(self):
        """EXP3 agent should handle extreme rewards"""
        agent = EXP3Agent(num_actions=2, seed=42)
        agent.reset()
        
        for _ in range(50):
            probs = agent.get_probabilities()
            agent.update(probs, 0, 1e6)
        
        assert np.all(np.isfinite(agent.log_weights))


class TestMetamorphicProperties:
    """Test that transformations preserve expected relationships"""

    def test_reward_scaling_preserves_preference(self):
        """If we scale rewards, agent preference should not flip"""
        agent = QLearningAgent(num_actions=2, seed=42)
        agent.reset()
        
        agent.q = np.array([5.0, 3.0])
        probs1 = agent.get_probabilities()
        
        agent.q = np.array([10.0, 6.0])
        probs2 = agent.get_probabilities()
        
        assert np.argmax(probs1) == np.argmax(probs2), \
            "Scaling changed action preference"

    def test_symmetry_in_equal_rewards(self):
        """Equal Q-values should give equal probabilities"""
        agent = QLearningAgent(num_actions=2, epsilon=0.0, seed=42)
        agent.reset()
        
        agent.q = np.array([5.0, 5.0])
        probs = agent.get_probabilities()
        
        assert np.allclose(probs, [0.5, 0.5]), \
            f"Equal Q-values didn't give uniform policy: {probs}"

    def test_convergence_with_noisy_rewards(self):
        """Learning should work even with noise"""
        env = BanditEnvironment(num_actions=2, seed=42)
        env.reset()
        agent = QLearningAgent(num_actions=2, seed=43)
        agent.reset()
        
        env.rewards = np.array([1.0, 0.0])
        for _ in range(200):
            probs = agent.get_probabilities()
            action = np.argmax(probs)
            reward = env.interact(action) + np.random.randn() * 0.1
            agent.update(probs, action, reward)
        
        assert agent.q[0] > agent.q[1]

    def test_probability_monotonicity(self):
        """If Q-value increases, probability should not decrease"""
        agent = QLearningAgent(num_actions=2, seed=42)
        agent.reset()
        
        agent.q = np.array([1.0, 0.0])
        probs1 = agent.get_probabilities()
        
        agent.q = np.array([2.0, 0.0])
        probs2 = agent.get_probabilities()
        
        assert probs2[0] >= probs1[0], \
            "Increasing Q-value decreased probability"
