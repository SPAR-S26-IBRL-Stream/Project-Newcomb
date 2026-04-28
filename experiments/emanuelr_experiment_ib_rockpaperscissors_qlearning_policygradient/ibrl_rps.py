import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

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

def score_grad_logits(logits, q):
    """Gradient of score w.r.t. logits (through softmax)."""
    p = softmax(logits)
    ds_dp = np.array([q[2] - q[1], q[0] - q[2], q[1] - q[0]])
    jac = np.diag(p) - np.outer(p, p)
    return jac @ ds_dp

def random_policy():
    x = np.random.dirichlet([1, 1, 1])
    return x

def run_single(init_logits, opponent, lr=0.1, n_steps=300):
    logits = init_logits.copy()
    trajectory = []
    for _ in range(n_steps):
        p = softmax(logits)
        trajectory.append(p[:2].copy())
        grad = score_grad_logits(logits, opponent)
        logits += lr * grad
    return np.array(trajectory), softmax(logits)

def run_ibrl(init_logits, opponents, lr=0.1, n_steps=300):
    logits = init_logits.copy()
    trajectory = []
    worst_idxs = []
    for _ in range(n_steps):
        p = softmax(logits)
        trajectory.append(p[:2].copy())
        scores = [score(p, q) for q in opponents]
        worst_idx = np.argmin(scores)
        worst_idxs.append(worst_idx)
        grad = score_grad_logits(logits, opponents[worst_idx])
        logits += lr * grad
    return np.array(trajectory), softmax(logits), worst_idxs

def worst_convex_opponent(p, opponents):
    """Find convex combination of opponents that minimizes score(p, q).
    Since score(p, q) = sum_i lambda_i * score(p, q_i), this is a linear program."""
    n = len(opponents)
    # Coefficients: minimize c^T lambda where c_i = score(p, opponents[i])
    c = np.array([score(p, q) for q in opponents])
    # Constraints: lambda >= 0, sum lambda = 1
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    bounds = [(0, 1)] * n
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    lam = res.x
    q_mix = sum(lam[i] * opponents[i] for i in range(n))
    return q_mix, lam

def run_convex_ibrl(init_logits, opponents, lr=0.1, n_steps=300):
    logits = init_logits.copy()
    trajectory = []
    lambdas_history = []
    for _ in range(n_steps):
        p = softmax(logits)
        trajectory.append(p[:2].copy())
        q_mix, lam = worst_convex_opponent(p, opponents)
        lambdas_history.append(lam.copy())
        grad = score_grad_logits(logits, q_mix)
        logits += lr * grad
    return np.array(trajectory), softmax(logits), lambdas_history

# --- Generate opponents ---
opponents = [random_policy() for _ in range(5)]
single_opponent = opponents[0]

print("Opponents:")
for i, opp in enumerate(opponents):
    print(f"  {i}: R={opp[0]:.3f} P={opp[1]:.3f} S={opp[2]:.3f}")

# --- Grid search for optimal IBRL policy ---
resolution = 500
best_worst_score = -np.inf
best_p_grid = None

for i in range(resolution + 1):
    for j in range(resolution + 1 - i):
        k = resolution - i - j
        p = np.array([i, j, k], dtype=float) / resolution
        worst = min(score(p, q) for q in opponents)
        if worst > best_worst_score:
            best_worst_score = worst
            best_p_grid = p.copy()

print(f"\nOptimal IBRL (grid search): R={best_p_grid[0]:.4f} P={best_p_grid[1]:.4f} S={best_p_grid[2]:.4f}")
print(f"  Worst-case score: {best_worst_score:.4f}")

# --- Grid search for optimal Convex Hull IBRL policy ---
# Since score(p, q) is linear in q, min over conv hull = min over vertices.
# So theoretically the optimal is the same. Let's verify with the LP solver.
best_worst_convex = -np.inf
best_p_convex = None

for i in range(resolution + 1):
    for j in range(resolution + 1 - i):
        k = resolution - i - j
        p = np.array([i, j, k], dtype=float) / resolution
        _, lam = worst_convex_opponent(p, opponents)
        q_mix = sum(lam[m] * opponents[m] for m in range(len(opponents)))
        worst = score(p, q_mix)
        if worst > best_worst_convex:
            best_worst_convex = worst
            best_p_convex = p.copy()

print(f"\nOptimal Convex Hull IBRL (grid search): R={best_p_convex[0]:.4f} P={best_p_convex[1]:.4f} S={best_p_convex[2]:.4f}")
print(f"  Worst-case score: {best_worst_convex:.4f}")
print(f"  (Theory: should match discrete IBRL since score is linear in q)")

# --- Two different initializations ---
rng = np.random.RandomState(99)
init1 = rng.randn(3) * 0.1   # near uniform
init2 = rng.randn(3) * 2.0   # far from uniform, more extreme start

inits = [init1, init2]
init_labels = ['Init 1 (near uniform)', 'Init 2 (far from uniform)']

# --- Plotting: 2 rows x 3 cols ---
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

def draw_simplex(ax):
    tri = plt.Polygon([[0, 0], [1, 0], [0, 1]], fill=False, edgecolor='gray', linestyle='--', linewidth=1)
    ax.add_patch(tri)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('P(Rock)')
    ax.set_ylabel('P(Paper)')
    ax.set_aspect('equal')

# Color the IBRL trajectory segments by which opponent is worst-case
opp_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']

for row, (init_logits, init_label) in enumerate(zip(inits, init_labels)):
    # Single opponent
    traj_s, final_s = run_single(init_logits, single_opponent)
    print(f"\n{init_label} - Single: R={final_s[0]:.4f} P={final_s[1]:.4f} S={final_s[2]:.4f}, score={score(final_s, single_opponent):.4f}")

    ax = axes[row, 0]
    draw_simplex(ax)
    ax.plot(traj_s[:, 0], traj_s[:, 1], 'b-', alpha=0.5, linewidth=1.2)
    # Arrow every 30 steps
    for k in range(0, len(traj_s) - 1, 30):
        dx = traj_s[k+1, 0] - traj_s[k, 0]
        dy = traj_s[k+1, 1] - traj_s[k, 1]
        ax.annotate('', xy=(traj_s[k+1, 0], traj_s[k+1, 1]),
                    xytext=(traj_s[k, 0], traj_s[k, 1]),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.plot(traj_s[0, 0], traj_s[0, 1], 'go', markersize=10, zorder=5, label='Start')
    ax.plot(traj_s[-1, 0], traj_s[-1, 1], 'r*', markersize=14, zorder=5, label='End')
    ax.plot(single_opponent[0], single_opponent[1], 'kx', markersize=10, markeredgewidth=2, label='Opponent')
    ax.set_title(f'Single Opponent — {init_label}')
    ax.legend(fontsize=8)

    # IBRL
    traj_i, final_i, worst_idxs = run_ibrl(init_logits, opponents)
    print(f"{init_label} - IBRL:   R={final_i[0]:.4f} P={final_i[1]:.4f} S={final_i[2]:.4f}, worst-case={min(score(final_i, q) for q in opponents):.4f}")

    ax = axes[row, 1]
    draw_simplex(ax)
    # Plot trajectory colored by worst-case opponent
    for k in range(len(traj_i) - 1):
        ax.plot(traj_i[k:k+2, 0], traj_i[k:k+2, 1],
                color=opp_colors[worst_idxs[k]], alpha=0.6, linewidth=1.2)
    # Arrows every 30 steps
    for k in range(0, len(traj_i) - 1, 30):
        dx = traj_i[k+1, 0] - traj_i[k, 0]
        dy = traj_i[k+1, 1] - traj_i[k, 1]
        ax.annotate('', xy=(traj_i[k+1, 0], traj_i[k+1, 1]),
                    xytext=(traj_i[k, 0], traj_i[k, 1]),
                    arrowprops=dict(arrowstyle='->', color=opp_colors[worst_idxs[k]], lw=1.5))
    ax.plot(traj_i[0, 0], traj_i[0, 1], 'go', markersize=10, zorder=5, label='Start')
    ax.plot(traj_i[-1, 0], traj_i[-1, 1], 'r*', markersize=14, zorder=5, label='IBRL End')
    ax.plot(best_p_grid[0], best_p_grid[1], 'm^', markersize=14, zorder=5, label='Optimal IBRL (grid)')
    # Plot opponents
    for i, opp in enumerate(opponents):
        ax.plot(opp[0], opp[1], 'x', color=opp_colors[i], markersize=10, markeredgewidth=2)
        ax.annotate(f'Opp {i}', (opp[0]+0.02, opp[1]+0.02), fontsize=8, color=opp_colors[i])
    # Legend entries for opponent colors
    handles, labels = ax.get_legend_handles_labels()
    for i in range(5):
        handles.append(Line2D([0], [0], color=opp_colors[i], linewidth=2))
        labels.append(f'vs Opp {i}')
    ax.legend(handles, labels, fontsize=7, loc='upper right')
    ax.set_title(f'IBRL (5 opponents) — {init_label}')

    # Convex Hull IBRL
    traj_c, final_c, lam_hist = run_convex_ibrl(init_logits, opponents)
    print(f"{init_label} - Convex: R={final_c[0]:.4f} P={final_c[1]:.4f} S={final_c[2]:.4f}, worst-case={score(final_c, worst_convex_opponent(final_c, opponents)[0]):.4f}")

    ax = axes[row, 2]
    draw_simplex(ax)
    # Color trajectory by dominant opponent in the convex mix
    for k in range(len(traj_c) - 1):
        dom_opp = np.argmax(lam_hist[k])
        ax.plot(traj_c[k:k+2, 0], traj_c[k:k+2, 1],
                color=opp_colors[dom_opp], alpha=0.6, linewidth=1.2)
    for k in range(0, len(traj_c) - 1, 30):
        dom_opp = np.argmax(lam_hist[k])
        ax.annotate('', xy=(traj_c[k+1, 0], traj_c[k+1, 1]),
                    xytext=(traj_c[k, 0], traj_c[k, 1]),
                    arrowprops=dict(arrowstyle='->', color=opp_colors[dom_opp], lw=1.5))
    ax.plot(traj_c[0, 0], traj_c[0, 1], 'go', markersize=10, zorder=5, label='Start')
    ax.plot(traj_c[-1, 0], traj_c[-1, 1], 'r*', markersize=14, zorder=5, label='Convex IBRL End')
    ax.plot(best_p_convex[0], best_p_convex[1], 'm^', markersize=14, zorder=5, label='Optimal (grid)')
    # Also show discrete IBRL optimal for comparison
    ax.plot(best_p_grid[0], best_p_grid[1], 'cs', markersize=10, zorder=5, label='Discrete IBRL Opt.', alpha=0.7)
    for i, opp in enumerate(opponents):
        ax.plot(opp[0], opp[1], 'x', color=opp_colors[i], markersize=10, markeredgewidth=2)
        ax.annotate(f'Opp {i}', (opp[0]+0.02, opp[1]+0.02), fontsize=8, color=opp_colors[i])
    # Shade convex hull of opponents
    opp_2d = np.array([o[:2] for o in opponents])
    hull = ConvexHull(opp_2d)
    hull_pts = opp_2d[hull.vertices]
    hull_poly = plt.Polygon(hull_pts, alpha=0.1, color='gray', label='Opp. convex hull')
    ax.add_patch(hull_poly)
    handles, labels = ax.get_legend_handles_labels()
    for i in range(5):
        handles.append(Line2D([0], [0], color=opp_colors[i], linewidth=2))
        labels.append(f'dom. Opp {i}')
    ax.legend(handles, labels, fontsize=6, loc='upper right')
    ax.set_title(f'Convex Hull IBRL — {init_label}')

plt.tight_layout()
plt.savefig('/home/emanuelr/Documents/IBRLRockPaperScissors/ibrl_rps.png', dpi=150)
plt.close()
print("\nPlot saved to ibrl_rps.png")
