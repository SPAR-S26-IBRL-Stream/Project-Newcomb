import numpy as np
import pytest
from ibrl.simulators import simulate
from ibrl.agents import QLearningAgent
from ibrl.environments import BanditEnvironment


class TestSimulator:
    def test_simulate_complete(self, num_actions, seed):
        """to be without error, produces correct shapes, and valid values."""
        env = BanditEnvironment(num_actions=num_actions, seed=seed)
        agent = QLearningAgent(num_actions=num_actions, seed=seed + 1)
        num_steps, num_runs = 10, 2
        options = {"num_steps": num_steps, "num_runs": num_runs}
        
        results = simulate(env, agent, options)
        
        # Check output structure
        assert "average_reward" in results
        assert "optimal_reward" in results
        assert results["average_reward"].shape == (2, num_steps)
        assert results["probabilities"].shape == (num_runs, num_steps, num_actions)
        assert results["actions"].shape == (num_runs, num_steps)
        assert results["rewards"].shape == (num_runs, num_steps)
        
        # Check values are valid
        assert np.all(np.isfinite(results["rewards"]))
        assert np.all(np.isfinite(results["average_reward"]))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
