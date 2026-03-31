import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(42)

# --- Utilities ---

def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()

def score(p, q):
    """Expected score of policy p against opponent q (win - loss)."""
    win = p[0]*q[2] + p[1]*q[0] + p[2]*q[1]
    loss = p[0]*q[1] + p[1]*q[2] + p[2]*q[0]
    return win - loss

def random_policy():
    return np.random.dirichlet([1, 1, 1])

# Reward matrix: reward[a, opp_a] for our action a vs opponent action opp_a
# Actions: 0=Rock, 1=Paper, 2=Scissors
REWARD = np.array([
    [ 0, -1,  1],   # Rock vs R/P/S
    [ 1,  0, -1],   # Paper vs R/P/S
    [-1,  1,  0],   # Scissors vs R/P/S
])

def exact_Q(opponent):
    """Compute exact Q-values: Q(a) = E[reward | a] = sum_j opponent[j] * REWARD[a,j]."""
    return REWARD @ opponent

# --- Same opponents as ibrl_rps.py (same seed) ---
opponents = [random_policy() for _ in range(5)]
single_opponent = opponents[0]

opp_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']

print("Opponents:")
for i, opp in enumerate(opponents):
    print(f"  {i}: R={opp[0]:.3f} P={opp[1]:.3f} S={opp[2]:.3f}")

# =============================================================================
# Exact MDP Q-iteration: Single Opponent vs IBRL
# =============================================================================
# The MDP has 1 state, 3 actions. Transition probabilities and rewards are
# determined by the opponent's policy. Q(a) = E[r | a] under the current MDP.
#
# Single opponent: fixed MDP, Q-values are computed exactly in one step.
# IBRL: at each iteration, Murphy picks the worst opponent for our current
# policy, changing the MDP. We recompute Q-values under the new MDP and
# update with a Bellman-like step: Q <- (1-alpha)*Q + alpha*Q_new.
# This models an agent that gradually learns the MDP's Q-values, but the
# MDP keeps shifting underneath it.

n_steps = 300
alpha = 0.3  # learning rate for Q-value updates (blending old and new MDP)

# --- Single opponent: exact Q-values, iterated ---
Q = np.zeros(3)
Q_hist_single = np.zeros((n_steps, 3))
policy_hist_single = np.zeros((n_steps, 3))
reward_hist_single = np.zeros(n_steps)

Q_true = exact_Q(single_opponent)

for t in range(n_steps):
    Q_new = exact_Q(single_opponent)  # exact Q under this MDP
    Q = (1 - alpha) * Q + alpha * Q_new
    policy = softmax(Q)
    Q_hist_single[t] = Q.copy()
    policy_hist_single[t] = policy.copy()
    reward_hist_single[t] = score(policy, single_opponent)

print(f"\nSingle opponent MDP:")
print(f"  True Q:  R={Q_true[0]:.4f} P={Q_true[1]:.4f} S={Q_true[2]:.4f}")
print(f"  Final Q: R={Q[0]:.4f} P={Q[1]:.4f} S={Q[2]:.4f}")
print(f"  Final policy: R={policy[0]:.4f} P={policy[1]:.4f} S={policy[2]:.4f}")
print(f"  Final reward: {reward_hist_single[-1]:.4f}")

# --- IBRL: Murphy picks worst opponent, MDP changes each step ---
Q = np.zeros(3)
Q_hist_ibrl = np.zeros((n_steps, 3))
policy_hist_ibrl = np.zeros((n_steps, 3))
reward_hist_ibrl = np.zeros(n_steps)
worst_opp_idxs = []

for t in range(n_steps):
    policy = softmax(Q)
    # Murphy picks worst opponent for our current policy
    scores_vs = [score(policy, q) for q in opponents]
    worst_idx = np.argmin(scores_vs)
    worst_opp_idxs.append(worst_idx)
    # Compute Q under Murphy's chosen MDP
    Q_new = exact_Q(opponents[worst_idx])
    Q = (1 - alpha) * Q + alpha * Q_new
    policy_after = softmax(Q)
    Q_hist_ibrl[t] = Q.copy()
    policy_hist_ibrl[t] = policy_after.copy()
    reward_hist_ibrl[t] = score(policy_after, opponents[worst_idx])

print(f"\nIBRL MDP:")
print(f"  Final Q: R={Q[0]:.4f} P={Q[1]:.4f} S={Q[2]:.4f}")
final_pol = softmax(Q)
print(f"  Final policy: R={final_pol[0]:.4f} P={final_pol[1]:.4f} S={final_pol[2]:.4f}")
print(f"  Worst-case reward: {min(score(final_pol, q) for q in opponents):.4f}")

# --- Plotting: 2x3 ---
fig, axes = plt.subplots(2, 3, figsize=(22, 10))

def draw_simplex(ax):
    tri = plt.Polygon([[0, 0], [1, 0], [0, 1]], fill=False, edgecolor='gray', linestyle='--', linewidth=1)
    ax.add_patch(tri)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('P(Rock)')
    ax.set_ylabel('P(Paper)')
    ax.set_aspect('equal')

# --- Left column: reward over time ---
ax = axes[0, 0]
ax.plot(reward_hist_single, color='tab:blue', linewidth=1.5)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Step')
ax.set_ylabel('Expected Reward')
ax.set_title('Single Opponent — Expected Reward')

ax = axes[1, 0]
ax.plot(reward_hist_ibrl, color='tab:red', linewidth=1.5)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Step')
ax.set_ylabel('Expected Reward')
ax.set_title('IBRL — Expected Reward')
# Background shading
band = 5
for t_start in range(0, n_steps - band, band):
    chunk = worst_opp_idxs[t_start:t_start+band]
    dominant = max(set(chunk), key=chunk.count)
    ax.axvspan(t_start, t_start+band, alpha=0.1, color=opp_colors[dominant])
handles, labels = ax.get_legend_handles_labels()
for i in range(5):
    handles.append(Line2D([0], [0], color=opp_colors[i], linewidth=6, alpha=0.3))
    labels.append(f'Murphy: Opp {i}')
ax.legend(handles, labels, fontsize=7, loc='lower right')

# --- Middle column: Q-values over time ---
ax = axes[0, 1]
for a, (name, color) in enumerate(zip(['Rock', 'Paper', 'Scissors'], ['tab:red', 'tab:blue', 'tab:green'])):
    ax.plot(Q_hist_single[:, a], label=name, color=color, linewidth=1.5)
    ax.axhline(y=Q_true[a], color=color, linestyle='--', alpha=0.4)
ax.set_xlabel('Step')
ax.set_ylabel('Q-value')
ax.set_title('Single Opponent — Q-Values (dashed = true)')
ax.legend(fontsize=8)

ax = axes[1, 1]
for a, (name, color) in enumerate(zip(['Rock', 'Paper', 'Scissors'], ['tab:red', 'tab:blue', 'tab:green'])):
    ax.plot(Q_hist_ibrl[:, a], label=name, color=color, linewidth=1.5)
ax.set_xlabel('Step')
ax.set_ylabel('Q-value')
ax.set_title('IBRL — Q-Values')
ax.legend(fontsize=8)
for t_start in range(0, n_steps - band, band):
    chunk = worst_opp_idxs[t_start:t_start+band]
    dominant = max(set(chunk), key=chunk.count)
    ax.axvspan(t_start, t_start+band, alpha=0.1, color=opp_colors[dominant])

# --- Right column: policy trajectory on simplex ---
ax = axes[0, 2]
draw_simplex(ax)
ps = policy_hist_single
ax.plot(ps[:, 0], ps[:, 1], 'b-', alpha=0.5, linewidth=1.2)
for k in range(0, n_steps - 1, 30):
    ax.annotate('', xy=(ps[k+1, 0], ps[k+1, 1]),
                xytext=(ps[k, 0], ps[k, 1]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.plot(ps[0, 0], ps[0, 1], 'go', markersize=10, zorder=5, label='Start')
ax.plot(ps[-1, 0], ps[-1, 1], 'r*', markersize=14, zorder=5, label='End')
ax.plot(single_opponent[0], single_opponent[1], 'kx', markersize=10, markeredgewidth=2, label='Opponent')
ax.set_title('Single Opponent — Policy Trajectory')
ax.legend(fontsize=8)

ax = axes[1, 2]
draw_simplex(ax)
# Color by worst opponent
for k in range(n_steps - 1):
    ax.plot(policy_hist_ibrl[k:k+2, 0], policy_hist_ibrl[k:k+2, 1],
            color=opp_colors[worst_opp_idxs[k]], alpha=0.5, linewidth=1.2)
# Arrows every 20 steps
for k in range(0, n_steps - 1, 20):
    ax.annotate('', xy=(policy_hist_ibrl[k+1, 0], policy_hist_ibrl[k+1, 1]),
                xytext=(policy_hist_ibrl[k, 0], policy_hist_ibrl[k, 1]),
                arrowprops=dict(arrowstyle='->', color=opp_colors[worst_opp_idxs[k]], lw=1.5))
ax.plot(policy_hist_ibrl[0, 0], policy_hist_ibrl[0, 1], 'go', markersize=10, zorder=5, label='Start')
ax.plot(policy_hist_ibrl[-1, 0], policy_hist_ibrl[-1, 1], 'r*', markersize=14, zorder=5, label='End')
ax.plot(1/3, 1/3, 'm^', markersize=12, zorder=5, label='Uniform (1/3)')
for i, opp in enumerate(opponents):
    ax.plot(opp[0], opp[1], 'x', color=opp_colors[i], markersize=10, markeredgewidth=2)
    ax.annotate(f'Opp {i}', (opp[0]+0.02, opp[1]+0.02), fontsize=8, color=opp_colors[i])
handles, labels = ax.get_legend_handles_labels()
for i in range(5):
    handles.append(Line2D([0], [0], color=opp_colors[i], linewidth=2))
    labels.append(f'vs Opp {i}')
ax.legend(handles, labels, fontsize=6, loc='upper right')
ax.set_title('IBRL — Policy Trajectory')

plt.suptitle('Q-Learning (exact MDP): Single Opponent (stable) vs IBRL (non-stationary MDP)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/home/emanuelr/Documents/IBRLRockPaperScissors/ibrl_qlearning.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved to ibrl_qlearning.png")
