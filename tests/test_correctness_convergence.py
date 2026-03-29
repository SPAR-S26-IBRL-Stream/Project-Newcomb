import numpy as np
import pytest
from ibrl.agents import QLearningAgent, BayesianAgent
from ibrl.environments import BanditEnvironment
from ibrl.simulators import simulate


class TestAgentLearning:
    """Test that agents actually learn, not just run"""

    @pytest.mark.parametrize("agent_class,learning_rate", [
        (QLearningAgent, 0.1),
        (BayesianAgent, None),
    ])
    def test_agent_converges_on_simple_bandit(self, agent_class, learning_rate):
        """Agent should converge to optimal arm on simple bandit"""
        env = BanditEnvironment(num_actions=2, seed=42)
        if agent_class == QLearningAgent:
            agent = agent_class(num_actions=2, learning_rate=learning_rate, seed=43)
        else:
            agent = agent_class(num_actions=2, seed=43)
        
        options = {"num_steps": 500, "num_runs": 1}
        results = simulate(env, agent, options)
        
        rewards = results["rewards"][0]
        early_avg = rewards[:50].mean()
        late_avg = rewards[-50:].mean()
        
        assert late_avg > early_avg, \
            f"Agent didn't learn: early={early_avg:.3f}, late={late_avg:.3f}"

    def test_bayesian_agent_reduces_uncertainty(self):
        """Bayesian agent should reduce uncertainty over time"""
        env = BanditEnvironment(num_actions=2, seed=42)
        agent = BayesianAgent(num_actions=2, seed=43)
        agent.reset()
        
        initial_precision = agent.precision.copy()
        probs = agent.get_probabilities()
        
        for _ in range(100):
            agent.update(probs, 0, 1.0)
        
        final_precision = agent.precision
        
        assert final_precision[0] > initial_precision[0], \
            f"Precision didn't increase: {initial_precision[0]} → {final_precision[0]}"

    @pytest.mark.parametrize("agent_class,better_arm_reward", [
        (QLearningAgent, 10.0),
        (BayesianAgent, 10.0),
    ])
    def test_agent_prefers_better_arm(self, agent_class, better_arm_reward):
        """Agent should eventually prefer the arm with higher reward"""
        env = BanditEnvironment(num_actions=2, seed=42)
        env.reset()
        env.rewards = np.array([better_arm_reward, 1.0])
        
        if agent_class == QLearningAgent:
            agent = agent_class(num_actions=2, learning_rate=0.1, seed=43)
        else:
            agent = agent_class(num_actions=2, seed=43)
        agent.reset()
        
        for _ in range(200):
            probs = agent.get_probabilities()
            action = np.argmax(probs)
            reward = env.interact(action)
            agent.update(probs, action, reward)
        
        attr_name = 'q' if hasattr(agent, 'q') else 'values'
        attr = getattr(agent, attr_name)
        assert attr[0] > attr[1], \
            f"Agent didn't learn reward difference: {attr_name}={attr}"


class TestEnvironmentCorrectness:
    """Test that environments compute rewards correctly"""

    @pytest.mark.parametrize("probs,action,expected_reward", [
        ([1.0, 0.0], 0, 10),
        ([1.0, 0.0], 1, 15),
        ([0.0, 1.0], 0, 0),
        ([0.0, 1.0], 1, 5),
    ])
    def test_newcomb_reward_table_consistency(self, probs, action, expected_reward, seed):
        """Newcomb environment should return correct rewards"""
        from ibrl.environments import NewcombEnvironment
        
        env = NewcombEnvironment(num_actions=2, seed=seed)
        env.reset()
        env.predict(np.array(probs))
        reward = env.interact(action)
        
        assert reward == expected_reward, \
            f"Newcomb reward mismatch: probs={probs}, action={action}, " \
            f"expected={expected_reward}, got={reward}"
