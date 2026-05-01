We propose several experiments to test our infra-bayesian agent. 
1. A standard stochastic multi-armed bandit with K = {2, 10, 100} arms and \Delta = {0.01, 0.1, 0.49} relative performance of the biases of the best vs worst arm. All arms have bias = 0.5 except the best arm, which has bias 0.5 + \Delta. The bandit is run for $n = {1000}$ rounds. 
2.  The same setting as above, but with an adversary who predicts which arm you are least likely to choose and makes that arm the best arm (its bias will be 0.5 + \Delta). All other arms have bias = 0.5.
3. The same setting as (1) but with non-adversarial distribution shift. Every $n/10$ rounds, the best arm is randomly re-chosen.

Note that experiment 2 can be thought of as newcomb-like and closely resembles the game death in damascus when $K=2$.

In each experiment, we compare three agents:
1. A classic bayesian thompson sampling learner starting with uninformative beta priors on each arm.
2. An exp3 learner with $\eta = \sqrt(\log(k)/(nk))$ (chosen to optimize the regret bound)
3. An infra-bayesian learner with knightian uncertainty on each arm: the bias could be anywhere between $[0.5, 0.99]$ with uncertain probability.

Our hypothesis is that the infra-bayesian agent will perform better than the classic learner in experiments (2) and (3) but worse in (1) (ideally, not too much worse). We hope it will do better than Exp3 in all experiments.

[Add plot here]
Figure 1. A 3x3 grid showing results from Experiment 1 (the standard stochastic multi-armed bandit setting) for each combination of $K = {2, 10, 100}$ and $\Delta = {0.01, 0.1, 0.49}$. Dashed blue curve is the classic learner; dash dotted green curve is Exp3; solid red curve is infra-bayesian learner. 

[Add plot here]
Figure 2. A 3x3 grid showing results from Experiment 2 (the adversarial multi-armed bandit setting) for each combination of $K = {2, 10, 100}$ and $\Delta = {0.01, 0.1, 0.49}$. Dashed blue curve is the classic learner; dash dotted green curve is Exp3; solid red curve is infra-bayesian learner. 

[Add plot here]
Figure 3. A 3x3 grid showing results from Experiment 3 (the distribution shift multi-armed bandit setting) for each combination of $K = {2, 10, 100}$ and $\Delta = {0.01, 0.1, 0.49}$. Dashed blue curve is the classic learner; dash dotted green curve is Exp3; solid red curve is infra-bayesian learner. 


