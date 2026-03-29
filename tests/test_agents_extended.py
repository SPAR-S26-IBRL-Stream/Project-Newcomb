import numpy as np
import pytest
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent
from ibrl.utils import dump_array


class TestBaseGreedyAgent:
    """Test BaseGreedyAgent functionality through QLearningAgent"""

    @pytest.mark.parametrize("epsilon,should_exploit", [
        pytest.param(0.1, True, id="low_epsilon"),
        pytest.param(0.9, False, id="high_epsilon"),
    ])
    def test_epsilon_greedy_policy(self, epsilon, should_exploit, num_actions, seed):
        agent = QLearningAgent(num_actions=num_actions, epsilon=epsilon, seed=seed)
        agent.reset()
        agent.q = np.array([1.0, 0.0])
        probs = agent.build_epsilon_greedy_policy(agent.q)
        assert np.isclose(probs.sum(), 1.0)
        if should_exploit:
            assert probs[0] > probs[1]
        else:
            assert probs[1] > 0.1

    @pytest.mark.parametrize("temperature,should_exploit", [
        pytest.param(0.1, True, id="cold_temp"),
        pytest.param(100.0, False, id="hot_temp"),
    ])
    def test_softmax_policy(self, temperature, should_exploit, num_actions, seed):
        agent = QLearningAgent(num_actions=num_actions, temperature=temperature, seed=seed)
        agent.reset()
        agent.q = np.array([1.0, 0.0])
        probs = agent.build_softmax_policy(agent.q)
        assert np.isclose(probs.sum(), 1.0)
        if should_exploit:
            assert probs[0] > probs[1]
        else:
            assert np.allclose(probs, [0.5, 0.5], atol=0.1)

    @pytest.mark.parametrize("decay_type,start,decay_const,end", [
        pytest.param(0, 1.0, 0.5, 0.01, id="exponential"),
        pytest.param(1, 1.0, 500, 0.01, id="linear"),
    ])
    def test_parameter_decay(self, decay_type, start, decay_const, end, num_actions, seed):
        agent = QLearningAgent(
            num_actions=num_actions,
            epsilon=(start, decay_const, end),
            decay_type=decay_type,
            seed=seed
        )
        agent.reset()
        eps1 = agent.parse_parameter(agent.epsilon)
        agent.step = 10
        eps2 = agent.parse_parameter(agent.epsilon)
        assert eps2 < eps1
        assert eps2 >= end

    def test_parse_parameter_fixed_value(self):
        """Test parse_parameter with fixed float value"""
        agent = QLearningAgent(num_actions=2, epsilon=0.5, seed=42)
        agent.reset()
        eps = agent.parse_parameter(agent.epsilon)
        assert eps == 0.5


class TestQLearningAgentExtended:
    """Extended tests for QLearningAgent"""

    def test_sample_average_mode(self):
        """Test Q-learning with sample averaging"""
        agent = QLearningAgent(num_actions=2, learning_rate=None, seed=42)
        agent.reset()
        probs = agent.get_probabilities()
        
        agent.update(probs, 0, 1.0)
        assert agent.counts[0] == 1
        assert np.isclose(agent.q[0], 1.0)
        
        agent.update(probs, 0, 3.0)
        assert agent.counts[0] == 2
        assert np.isclose(agent.q[0], 2.0)

    def test_sample_average_multiple_actions(self):
        """Test sample averaging across different actions"""
        agent = QLearningAgent(num_actions=2, learning_rate=None, seed=42)
        agent.reset()
        probs = agent.get_probabilities()
        
        agent.update(probs, 0, 2.0)
        agent.update(probs, 0, 4.0)
        agent.update(probs, 1, 1.0)
        
        assert np.isclose(agent.q[0], 3.0)
        assert np.isclose(agent.q[1], 1.0)
        assert agent.counts[0] == 2
        assert agent.counts[1] == 1

    def test_learning_rate_mode(self):
        """Test Q-learning with fixed learning rate"""
        agent = QLearningAgent(num_actions=2, learning_rate=0.1, seed=42)
        agent.reset()
        probs = agent.get_probabilities()
        initial_q = agent.q[0]
        agent.update(probs, 0, 1.0)
        expected_q = initial_q + 0.1 * (1.0 - initial_q)
        assert np.isclose(agent.q[0], expected_q)


class TestBayesianAgentExtended:
    """Extended tests for BayesianAgent"""

    def test_precision_increases_with_updates(self):
        """Test that precision increases with each update"""
        agent = BayesianAgent(num_actions=2, seed=42)
        agent.reset()
        initial_precision = agent.precision[0]
        probs = agent.get_probabilities()
        
        agent.update(probs, 0, 1.0)
        assert agent.precision[0] > initial_precision
        
        agent.update(probs, 0, 1.0)
        assert agent.precision[0] > initial_precision + 1

    def test_value_converges_to_reward(self):
        """Test that values converge to consistent rewards"""
        agent = BayesianAgent(num_actions=2, seed=42)
        agent.reset()
        probs = agent.get_probabilities()
        
        for _ in range(10):
            agent.update(probs, 0, 5.0)
        
        assert np.isclose(agent.values[0], 5.0, atol=0.5)


class TestEXP3AgentExtended:
    """Extended tests for EXP3Agent"""

    def test_weights_change_with_reward(self):
        """Test that weights change based on rewards"""
        agent = EXP3Agent(num_actions=2, seed=42)
        agent.reset()
        initial_weights = agent.log_weights.copy()
        probs = agent.get_probabilities()
        agent.update(probs, 0, 1.0)
        assert not np.allclose(agent.log_weights, initial_weights)

    def test_probabilities_sum_to_one(self):
        """Test that probabilities always sum to 1"""
        agent = EXP3Agent(num_actions=2, seed=42)
        agent.reset()
        for _ in range(10):
            probs = agent.get_probabilities()
            assert np.isclose(probs.sum(), 1.0)
            agent.update(probs, 0, 1.0)


class TestDebugUtils:
    """Test debug utility functions"""

    @pytest.mark.parametrize("array,expected_contains", [
        pytest.param(np.array([0.5, 0.3, 0.2]), "0.50", id="normal"),
        pytest.param(np.array([1.0]), "1.00", id="single"),
        pytest.param(np.array([-0.5, 0.5]), "-0.50", id="negative"),
    ])
    def test_dump_array(self, array, expected_contains):
        result = dump_array(array)
        assert "[" in result
        assert "]" in result
        assert expected_contains in result
