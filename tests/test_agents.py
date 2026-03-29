import numpy as np
import pytest
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent


class TestAgentInitialization:
    """Parametrized agent initialization tests"""
    
    @pytest.mark.parametrize("agent_class,agent_name", [
        pytest.param(QLearningAgent, "QLearningAgent", id="qlearn"),
        pytest.param(BayesianAgent, "BayesianAgent", id="bayesian"),
        pytest.param(EXP3Agent, "EXP3Agent", id="exp3"),
    ])
    def test_initialization(self, agent_class, agent_name, num_actions, seed):
        agent = agent_class(num_actions=num_actions, seed=seed)
        assert agent.num_actions == num_actions
        assert agent.seed == seed

    @pytest.mark.parametrize("agent_class", [
        pytest.param(QLearningAgent, id="qlearn"),
        pytest.param(BayesianAgent, id="bayesian"),
        pytest.param(EXP3Agent, id="exp3"),
    ])
    def test_reset(self, agent_class, num_actions, seed):
        agent = agent_class(num_actions=num_actions, seed=seed)
        agent.reset()
        assert agent.step == 1
        assert hasattr(agent, 'random')

    @pytest.mark.parametrize("agent_class", [
        pytest.param(QLearningAgent, id="qlearn"),
        pytest.param(BayesianAgent, id="bayesian"),
        pytest.param(EXP3Agent, id="exp3"),
    ])
    def test_get_probabilities(self, agent_class, num_actions, seed):
        agent = agent_class(num_actions=num_actions, seed=seed)
        agent.reset()
        probs = agent.get_probabilities()
        assert probs.shape == (num_actions,)
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)

    @pytest.mark.parametrize("agent_class", [
        pytest.param(QLearningAgent, id="qlearn"),
        pytest.param(BayesianAgent, id="bayesian"),
        pytest.param(EXP3Agent, id="exp3"),
    ])
    def test_update(self, agent_class, num_actions, seed):
        agent = agent_class(num_actions=num_actions, seed=seed)
        agent.reset()
        probs = agent.get_probabilities()
        action = 0
        reward = 1.0
        agent.update(probs, action, reward)
        assert agent.step == 2

    @pytest.mark.parametrize("agent_class", [
        pytest.param(QLearningAgent, id="qlearn"),
        pytest.param(BayesianAgent, id="bayesian"),
        pytest.param(EXP3Agent, id="exp3"),
    ])
    def test_probability_always_normalized(self, agent_class, num_actions, seed):
        """Probabilities must always sum to 1"""
        agent = agent_class(num_actions=num_actions, seed=seed)
        agent.reset()
        for _ in range(10):
            probs = agent.get_probabilities()
            assert np.isclose(probs.sum(), 1.0), f"Probs sum to {probs.sum()}"

    def test_agent_rejects_invalid_num_actions(self):
        """Agent should reject invalid num_actions"""
        with pytest.raises(AssertionError):
            QLearningAgent(num_actions=1, seed=42)

    def test_learning_rate_none(self, num_actions, seed):
        agent = QLearningAgent(num_actions=num_actions, learning_rate=None, seed=seed)
        agent.reset()
        assert agent.learning_rate is None
        assert hasattr(agent, 'counts')

    def test_q_values_remain_finite(self):
        """Q-values should never become NaN or Inf"""
        agent = QLearningAgent(num_actions=2, seed=42)
        agent.reset()
        for _ in range(100):
            probs = agent.get_probabilities()
            agent.update(probs, 0, np.random.randn() * 100)
        assert np.all(np.isfinite(agent.q)), "Q-values became non-finite"

    def test_bayesian_values_remain_finite(self):
        """Bayesian values should never become NaN or Inf"""
        agent = BayesianAgent(num_actions=2, seed=42)
        agent.reset()
        for _ in range(100):
            probs = agent.get_probabilities()
            agent.update(probs, 0, np.random.randn() * 100)
        assert np.all(np.isfinite(agent.values))

    def test_exp3_probabilities_always_valid(self):
        """EXP3 probabilities must always be valid"""
        agent = EXP3Agent(num_actions=2, seed=42)
        agent.reset()
        for _ in range(50):
            probs = agent.get_probabilities()
            assert np.isclose(probs.sum(), 1.0)
            assert np.all(probs >= 0)
            assert np.all(np.isfinite(probs))
            agent.update(probs, 0, 1.0)
