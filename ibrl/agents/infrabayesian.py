"""InfraBayesianAgent — agent using infrabayesian inference."""
import itertools
import numpy as np

from . import BaseGreedyAgent
from ..infrabayesian.a_measure import AMeasure
from ..infrabayesian.infradistribution import Infradistribution


class InfraBayesianAgent(BaseGreedyAgent):
    """
    Agent using infrabayesian inference.

    Maintains a single shared infradistribution (a Bayesian prior over hypotheses),
    updated on every step regardless of which action was taken. Each hypothesis is an
    Infradistribution over a shared WorldModel; the prior is a 1D array of weights.

    Arguments:
        hypotheses:       list of hypotheses
        prior:            distribution over hypotheses; default: uniform
        reward_function:  reward_function[a,o] is reward upon seeing outcome o from action a
        policy_discretisation:  number of mixed policies to consider per action; default: 0 (i.e. only pure policies)
        exploration_prefix:     parameter to control exploration
                                =0    no exploration
                                >0    forced exploration prefix for given number of steps, then no exploration
                                None  greedy exploration (epsilon or softmax; breaks regret bounds)

    #Recent update:
    Infrabayesian agent that computes belief-dependent policies.
    """
    
    def __init__(self, 
             num_actions: int,
             hypotheses: list,
             prior: np.ndarray,
             reward_function: np.ndarray,
             policy_discretisation: int = 10,
             policy_optimization: str = "greedy",
             softmax_beta: float = 1.0,
             exploration_prefix: int = 0,
             seed: int = 42,
             **kwargs):
    
        super().__init__(num_actions=num_actions, seed=seed, **kwargs)
        self.hypotheses = hypotheses
        self.prior = prior
        self.reward_function = reward_function
        self.policy_discretisation = policy_discretisation
        self.policy_optimization = policy_optimization
        self.softmax_beta = softmax_beta
        self.exploration_prefix = exploration_prefix
        
        # NEW: Cache for belief-dependent policy
        self.current_stateful_policy = None
        self.Q_table = None
        self.belief_indexer = None
    
    def reset(self):
        """Initialize belief-dependent policy via Q-value computation."""
        super().reset()
        # Ensure reward_function is set before mixing
        if self.reward_function is None:
            self.reward_function = np.tile([0., 1.], (self.num_actions, 1))
        self.dist = Infradistribution.mix(self.hypotheses, self.prior)
        self._compute_stateful_policy()
    
    def _compute_stateful_policy(self):
        """Compute belief-dependent policy from hypotheses."""
        from ..utils.belief_discretization import BeliefIndexer, corner_beliefs
        from ..agents.policy_optimizer import PolicyOptimizer
        from ..agents.supra_pomdp_agent import StatefulPolicy
        
        if len(self.hypotheses) == 0:
            raise ValueError("No hypotheses provided")
        
        h = self.hypotheses[0]
        wm = h.world_model
        
        # Only compute stateful policy for SupraPOMDP models
        if not hasattr(wm, 'num_states'):
            # For non-POMDP models, use flat policy
            self.current_stateful_policy = None
            return
        
        belief_state = wm.initial_state()
        if hasattr(h, 'measures') and len(h.measures) > 0:
            params = h.measures[0].params
        else:
            raise ValueError("Cannot extract parameters from hypothesis")
        
        try:
            Q_table, belief_indexer = wm.compute_q_values_belief_indexed(
                belief_state,
                params,
                policy_discretisation=self.policy_discretisation
            )
        except AttributeError:
            belief_space = corner_beliefs(wm.num_states)
            Q_table = np.zeros((belief_space.shape[0], self.num_actions))
            
            for b_idx, belief in enumerate(belief_space):
                for a in range(self.num_actions):
                    Q_table[b_idx, a] = self.reward_function[a, :] @ belief
            
            belief_indexer = BeliefIndexer(belief_space)
        
        optimizer = PolicyOptimizer(Q_table, belief_indexer, self.num_actions)
        
        if self.policy_optimization == "greedy":
            self.current_stateful_policy = optimizer.greedy_policy()
        elif self.policy_optimization == "softmax":
            self.current_stateful_policy = optimizer.softmax_policy(beta=self.softmax_beta)
        elif self.policy_optimization == "lp":
            self.current_stateful_policy = optimizer.linear_program_policy()
        else:
            raise ValueError(f"Unknown policy optimization: {self.policy_optimization}")
        
        self.Q_table = Q_table
        self.belief_indexer = belief_indexer
    
    def get_probabilities(self) -> np.ndarray:
        """Return action probabilities."""
        if self.current_stateful_policy is not None:
            # SupraPOMDP: use stateful policy
            return self.current_stateful_policy.to_flat_policy()
        
        # Bernoulli: evaluate from mixed infradistribution
        uniform_policy = np.ones(self.num_actions) / self.num_actions
        rewards = np.array([
            self.dist.evaluate_action(self.reward_function[a], a, uniform_policy)
            for a in range(self.num_actions)
        ])
        return self.build_greedy_policy(rewards)
    
    def get_stateful_probabilities(self, belief: np.ndarray) -> np.ndarray:
        """
        NEW: Return action probabilities conditioned on a specific belief.
        
        Args:
            belief: Belief state (shape (num_states,))
        
        Returns:
            Action probability distribution (shape (num_actions,))
        """
        if self.current_stateful_policy is None:
            self._compute_stateful_policy()
        
        return self.current_stateful_policy(belief)
    

    def update(self, probabilities: np.ndarray, action: int, outcome) -> None:
        super().update(probabilities, action, outcome)
        self.dist.update(self.reward_function, outcome, action, probabilities)
        # Recompute policy after learning
        self._compute_stateful_policy()


    def dump_state(self) -> str:
        state = str(self.dist.belief_state)
        if self.verbose > 1:
            state += ";" + repr(self.dist)
        return state

    def _expected_rewards(self) -> np.ndarray:
        # Iterate over policies and compute expected reward: E[π] = Σ_a E_π[a] π(a)
        expected_rewards = np.array([sum(
            self.dist.evaluate_action(self.reward_function[a], a, policy) * policy[a]
                for a in range(self.num_actions)
            ) for policy in self.policies
        ])

        # Find all optimal policies, sum them up and normalise
        optimal_policies = np.isclose(expected_rewards, expected_rewards.max())
        return (self.policies*np.expand_dims(optimal_policies,axis=1)).sum(axis=0) / sum(optimal_policies)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
## Code Rationale:

# Integrates belief-dependent policy optimization into existing ib agent
# Maintains backward compatibility via to_flat_policy()
# Adds new get_stateful_probabilities(belief) for explicit belief conditioning
# Policy is computed once at reset (can be extended to online refinement)
