import numpy as np
from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.infrabayesian.world_models.supra_pomdp_world_model import SupraPOMDPWorldModel
from ibrl.agents.infrabayesian import InfraBayesianAgent
from ibrl.outcome import Outcome

def obs(o: int, reward: float = 0.):
    return Outcome(reward=reward, observation=o)

# ============================================================================
# TEST 1: State-Dependent Beliefs (Multi-step Planning)
# ============================================================================
print("=" * 70)
print("TEST 1: State-Dependent Beliefs (Multi-step Planning)")
print("=" * 70)

wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2, discount=0.9)
T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])  # State transitions
B = np.array([[0.85, 0.15], [0.15, 0.85]])  # Noisy observations
theta_0 = np.array([0.5, 0.5])
R = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # Different rewards per state

params = wm.make_params(T, B, theta_0, R)
hypothesis = Infradistribution([AMeasure(params)], world_model=wm)

# Compute expected reward from initial belief
reward_fn = np.array([0., 1.])
policy = np.array([1.0])

# This uses VALUE ITERATION (multi-step planning)
er = wm.compute_expected_reward(hypothesis.belief_state, reward_fn, params, 
                                action=0, policy=policy)
print(f"✓ Multi-step expected reward computed: {er:.4f}")


# ============================================================================
# TEST 2: Policy-Dependent Predictor (Newcomb-like)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Policy-Dependent Predictor (Newcomb-like)")
print("=" * 70)

eps = 0.05
num_actions = 2
num_states = 2
num_obs = 2

wm_newcomb = SupraPOMDPWorldModel(num_states=num_states, num_actions=num_actions,
                                   num_obs=num_obs, discount=0.9)

# CRITICAL: theta_0 is a FUNCTION of policy
# Predictor predicts agent's policy with epsilon error
def theta_0_fn(pi):
    p_one_box = (1 - eps) * pi[0] + eps * 0.5
    return np.array([p_one_box, 1. - p_one_box])

T_newcomb = np.zeros((num_states, num_actions, num_states))
T_newcomb[0, :, 0] = 1.  # State 0 stays
T_newcomb[1, :, 1] = 1.  # State 1 stays

B_newcomb = np.eye(num_states)  # Fully observable
R_newcomb = np.zeros((num_states, num_actions, num_states))
R_newcomb[0, 0, 0] = 1.0   # one-box on predicted_one_box → +1
R_newcomb[1, 1, 1] = 0.1   # two-box on predicted_two_box → +0.1

params_newcomb = wm_newcomb.make_params(T_newcomb, B_newcomb, theta_0_fn, R_newcomb)

# Test with different policies
policy_one_box = np.array([1.0, 0.0])
policy_two_box = np.array([0.0, 1.0])
policy_mixed = np.array([0.5, 0.5])

reward_fn_newcomb = np.array([0., 1.])

er_one_box = wm_newcomb.compute_expected_reward(
    wm_newcomb.initial_state(), reward_fn_newcomb, params_newcomb,
    action=0, policy=policy_one_box
)
er_two_box = wm_newcomb.compute_expected_reward(
    wm_newcomb.initial_state(), reward_fn_newcomb, params_newcomb,
    action=1, policy=policy_two_box
)
er_mixed = wm_newcomb.compute_expected_reward(
    wm_newcomb.initial_state(), reward_fn_newcomb, params_newcomb,
    action=0, policy=policy_mixed
)

print(f"  Expected reward (one-box policy): {er_one_box:.4f}")
print(f"  Expected reward (two-box policy): {er_two_box:.4f}")
print(f"  Expected reward (mixed policy):   {er_mixed:.4f}")
print(f"  → One-boxing is better: {er_one_box > er_two_box}")


## TEST 3: Agent Learns State-Dependent Policy
print("\n" + "="*70)
print("TEST 3: Agent Learns State-Dependent Policy")
print("="*70)

hypothesis_newcomb = Infradistribution([AMeasure(params_newcomb)], world_model=wm_newcomb)
agent = InfraBayesianAgent(
    num_actions=num_actions,
    hypotheses=[hypothesis_newcomb],
    prior=np.array([1.0]),
    reward_function=np.ones((num_actions, num_obs)),
    policy_discretisation=10,
    exploration_prefix=0,
)

agent.reset()
initial_policy = agent.get_probabilities()
print(f"✓ Initial policy: {initial_policy}")

# Simulate: agent commits to one-boxing and gets rewarded
# Use the agent's actual policy for consistency
for step in range(10):
    policy = agent.get_probabilities()
    # Agent takes action 0 (one-box), predictor predicts one-box (matches policy[0])
    # This gives reward 1.0 (boxB filled)
    agent.update(policy, action=0, outcome=obs(0, reward=1.0))

final_policy = agent.get_probabilities()
print(f"✓ Final policy after 10 steps: {final_policy}")
print(f"  → Agent converged to one-boxing: {final_policy[0] > 0.9}")


# ============================================================================
# TEST 4: Belief State is State-Dependent
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: Belief State Tracks Latent States")
print("=" * 70)

wm_belief = SupraPOMDPWorldModel(num_states=3, num_actions=2, num_obs=3)
T_belief = np.random.dirichlet(np.ones(3), size=(3, 2))
B_belief = np.random.dirichlet(np.ones(3), size=3)
theta_0_belief = np.array([0.33, 0.33, 0.34])
R_belief = np.zeros((3, 2, 3))

params_belief = wm_belief.make_params(T_belief, B_belief, theta_0_belief, R_belief)

state = wm_belief.initial_state()
print(f"✓ Initial state: {state.components}")

state = wm_belief.update_state(state, obs(0), action=0, 
                               policy=np.array([0.5, 0.5]), params=params_belief)
belief, log_m = state.components[0]
print(f"✓ After observation 0:")
print(f"  Belief over 3 latent states: {belief}")
print(f"  Log-marginal likelihood: {log_m:.4f}")
print(f"  (Belief is state-dependent, not just action-dependent)")

state = wm_belief.update_state(state, obs(1), action=1,
                               policy=np.array([0.5, 0.5]), params=params_belief)
belief, log_m = state.components[0]
print(f"✓ After observation 1:")
print(f"  Updated belief: {belief}")
print(f"  (Belief updated via Bayesian filter over latent states)")

