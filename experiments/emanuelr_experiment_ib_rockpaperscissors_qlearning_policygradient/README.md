


# IBRL: Q-Learning vs. Policy Gradient in a Non-Realizable RMDP

## What is this experiment?
A minimalist Rock-Paper-Scissors RMDP demonstrating learning dynamics under Infra-Bayesian Knightian uncertainty. 

The agent plays RPS. Instead of a stationary environment, it faces **Murphy**, who adversarially selects the worst-case opponent from a set of 5 fixed hypotheses (or their convex hull) at every step, based on the agent's current policy. We compare two learning algorithms in this non-realizable setting:
1. **Q-Learning** (`ibrl_qlearning.py`): Exact Q-iteration updates.
2. **Policy Gradient** (`ibrl_rps.py`): Gradient ascent on policy logits.

## Why did we run it?
To empirically demonstrate a core theoretical tension in IB/RL alignment: 
In classical, realizable MDPs, Q-learning is mathematically guaranteed to converge to the global optimum. However, when deployed in a non-realizable RMDP where Murphy shifts the environment, those guarantees collapse. 

We hypothesized that Policy Gradient (PG) would succeed where Q-learning fails by directly optimizing the maximin objective. The underlying tension is that while PG empirically solves this IB environment, **PG fundamentally lacks universal convergence guarantees in both general MDPs and RMDPs**, forcing a trade-off between the robust convergence theorems of classical RL and the non-realizable capability of gradient-based methods.

## Results
* **Q-Learning (`ibrl_qlearning.png`)**: Works perfectly against a single stationary opponent. However, against Murphy, it fails catastrophically. The shifting MDP constantly invalidates the agent's action-value estimates, trapping the policy in an infinite limit cycle on the simplex. It never finds the maximin equilibrium.
* **Policy Gradient (`ibrl_rps.png`)**: Successfully solves the environment. By taking small steps against the *current* worst-case hypothesis, it smoothly navigates the interior of the simplex and converges exactly to the minimax equilibrium, both against the discrete set of 5 opponents and their convex hull.

## Interpretation
Q-learning's fails, even though it gets close to the solution it starts oscillating, since it assumes the environment is a classical MDP.

Policy gradient works, since in this environment, where murphy changes the adversary as the policy passes certain bounds, the reward is differentiable almost everywhere with respect to the parameters, and this derivative matches the classical policy gradient when choosing the current adversary.

**The Catch:** While PG solves this IB environment, it is susceptible to local optima in more complex environments. We are left with a gap in the theory: Q-learning has global convergence guarantees but fails in non-realizable RMDPs, whereas PG handles this non-realizability but lacks the strict global convergence guarantees we want to create an elegant IBRL algorithm for RMDPs.

## Comment about convexity
The convex combination experiment (allowing Murphy to pick from the convex hull of the 5 opponents) is theoretically equivalent to the discrete case in this specific setup. 

Because the expected reward is linear with respect to the opponent's policy, the reward against a mixed adversarial strategy $q_{mix} = \sum \lambda_i q_i$ is simply $\sum \lambda_i R(q_i)$. Since $\lambda_i \ge 0$ and $\sum \lambda_i = 1$, this expected reward is strictly bounded by the minimum and maximum of the pure components:

$$
\sum \lambda_i R(q_i) \ge \min_i R(q_i)
$$

Therefore, the minimizer is always achieved at a vertex (i.e., setting $\lambda_i = 1$ for the single opponent yielding the lowest reward). Thus, Murphy gains no additional adversarial power from mixing, and finding the worst-case opponent over the convex hull collapses to simply argmin-ing over the discrete set.  
---

### Usage
```bash
pip install numpy matplotlib scipy
python ibrl_rps.py        # Generates ibrl_rps.png
python ibrl_qlearning.py  # Generates ibrl_qlearning.png
```
