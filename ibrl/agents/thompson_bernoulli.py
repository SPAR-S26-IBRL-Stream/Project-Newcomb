import numpy as np
from ibrl.agents import BaseAgent # Assuming the base class is in base_agent.py

from ..outcome import Outcome

class ThompsonSamplingBernoulli(BaseAgent):
    """
    Thompson Sampling agent for Bernoulli Bandits using a Beta-Bernoulli conjugate prior.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alphas = None
        self.betas = None

    def reset(self) -> None:
        """Resets the Beta priors to Alpha=1, Beta=1 (Uniform distribution)."""
        super().reset()
        self.alphas = np.ones(self.num_actions)
        self.betas = np.ones(self.num_actions)

    def get_probabilities(self) -> np.ndarray:
        """
        Performs Thompson Sampling:
        1. Samples a value from the Beta(alpha, beta) distribution for each action.
        2. Identifies the action with the maximum sampled value.
        3. Returns a one-hot probability vector for that action.
        """
        # Sample from Beta distribution for each arm
        samples = self.random.beta(self.alphas, self.betas)
        
        # Select the arm with the highest sample
        best_action = np.argmax(samples)
        
        # Create one-hot distribution
        probs = np.zeros(self.num_actions)
        probs[best_action] = 1.0
        return probs

    def update(self, probabilities: np.ndarray, action: int, outcome: Outcome) -> None:
        """
        Updates the Beta parameters based on the observed reward.
        Assumes outcome.reward is binary (0 or 1).
        """
        super().update(probabilities, action, outcome)
        
        # Bernoulli reward update
        reward = outcome.reward
        if reward == 1:
            self.alphas[action] += 1
        else:
            self.betas[action] += 1

    def dump_state(self) -> str:
        """Returns a string representation of the current Alpha/Beta parameters."""
        state_info = [f"A{i}:({self.alphas[i]:.1f},{self.betas[i]:.1f})" 
                      for i in range(self.num_actions)]
        return f"Step {self.step} | " + " ".join(state_info)
