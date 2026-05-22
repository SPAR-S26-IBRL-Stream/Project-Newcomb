# Trap Bandit Experiment

Below we describe a simple experiment to demonstrate how a robust infra-Bayesian learner may be beneficial even in a stateless, stochastic bandit setting. Of course, it is in more complex, stateful settings, e.g. that require self-consistency and are likely to suffer from non-realizability, that infra-Bayesian learners are expected to be most beneficial. Nevertheless, this simple experiment demonstrates learning under Knightian uncertainty in a realizable single state setting.

The details of our experiment are as follows. There are `K=2` possible arms to pull. There is a data-generating probability `p_risky` of being in a risky world, and probability `1 - p_risky` of being in a safe world.

At the beginning of a new run, the world_type = {risky, safe} is sampled from `Bernoulli(p_risky)`. Then, `p_1` and `p_2` are newly assigned such that one arm has reward probability `0.3` and the other has reward probability `0.7`. In the safe world, each arm is Bernoulli and has fixed probability, `p_i={0.3,0.7}`, of yielding reward `1`. In the risky world, the arm with the 0.7 bias is a three-sided die with a small probability `p_catastrophe` of yielding reward `-1000`; with probability `0.7`, it yields reward `1`; otherwise it yields reward `0`. The arm with the lower realized bias is still Bernoulli with reward = `{1,0}`.

```text
For each new run:
    sample world_type ~ Bernoulli(p_risky)
    sample (p1, p2) uniformly from {(0.3, 0.7), (0.7, 0.3)}

    if safe world:
        arm i -> Bernoulli(p_i)

    if risky world:
        trapped_arm = argmax(p1, p2)
        trapped_arm -> reward -1000 with probability p_cat
                        reward 1 with probability p_i
                        reward 0 otherwise
        other arm   -> Bernoulli(p_i)
```
Schema 1. Experiment world design.

We compare classical Bayesian agents and an infra-Bayesian agent using the same joint hypothesis machinery. Bayesian agents always use `Infradistribution.mix(...)`; the infra-Bayesian agent uses Knightian uncertainty over the safe-vs-risky world families via `Infradistribution.mixKU(...)`, while continuing to use classical/Bayesian uncertainty (employing `Infradistribution.mix(...)`) over `p1,p2` within each family.

```
Our results can be reproduced by running:
uv run python -m experiments.alaro.trap_bandit.run \
--num-worlds 200 \
--num-steps 100 \
--num-grid 7 \
--p-cat 0.01 \
--condition-preset mostly_risky \
--p-mode separated \
--p-low 0.3 \
--p-high 0.7 \
--bootstrap-samples 5000 \
--output-dir experiments/alaro/trap_bandit/results_separated_arms_200_pcat001 \
--force

uv run python -m experiments.alaro.trap_bandit.run \
--num-worlds 200 \
--num-steps 100 \
--num-grid 7 \
--p-cat 0.01 \
--condition-preset baseline \
--p-mode separated \
--p-low 0.3 \
--p-high 0.7 \
--bootstrap-samples 5000 \
--output-dir experiments/alaro/trap_bandit/results_baseline_200_pcat001_overpessimistic \
--force
```

We consider three data generating processes: a mostly risky worlds setting, in which the data-generating process has `p_risky=0.99`; a mostly safe worlds setting, in which the data-generating process has `p_risky=0.01`; and a balanced risky/safe worlds setting, in which the data-generating process has `p_risky=0.5`. For each data generating process, we compare Bayesian agents with correctly specified point priors, Bayesian agents with misspecified point priors, and infra-Bayesian learners, which do not specify a prior probability but instead maintain Knightian uncertainty over whether they are in a risky or safe world. The infra-Bayesian agent always shares the same classical `p1,p2` prior as the Bayesian agent but maintains Knightian uncertainty over whether the world is safe or risky.

Because agents begin each run without knowing which arm is high-reward or whether the world is risky, we must choose strategies to balance the explore-exploit tradeoff. For Bayesian agents, we compare three exploration strategies:

- greedy,
- Thompson sampling,
- Bayesian UCB.

For the infra-Bayesian agent, we use greedy action selection over its robust lower values, with uniform tie-breaking.

Regret is measured against the best policy with full knowledge of the true world. We report cumulative expected regret percentiles and trapped-arm pull-rate percentiles.

## Results


Each comparison figure has four subplots. Columns are the two Bayesian safe-vs-risky priors being compared. The first row shows cumulative expected regret; the second row shows the high-reward/trapped-arm pull rate. Solid lines show medians and shaded bands show empirical 5th-95th percentile ranges across all sampled runs at each time step.

### Mostly risky worlds setting

![Mostly-risky prior comparison](results_separated_arms_200_pcat001/mostly_risky_prior_comparison_grid.png)

Figure 2a. Mostly-risky DGP, with `p_risky=0.99`. The left column uses the correctly specified Bayesian prior `p_risky_prior=0.99`; the right column uses the severely misspecified prior `p_risky_prior=0.01`.

When the Bayesian prior is correctly specified, greedy Bayes and the infra-Bayesian agent behave nearly identically. Under severe misspecification, however, the Bayesian agents initially treat the world as mostly safe, pull the high-reward trapped arm much more often, and incur substantially larger regret. This is the main robustness result: IB is insensitive to this particular underestimation of risky worlds because it does not collapse safe-vs-risky uncertainty to a single point prior.

### Mostly safe worlds setting

![Mostly-safe prior comparison](results_separated_arms_200_pcat001/mostly_safe_prior_comparison_grid.png)

Figure 2b. Mostly-safe DGP, with `p_risky=0.01`. The left column uses the correctly specified Bayesian prior `p_risky_prior=0.01`; the right column uses the severely misspecified pessimistic prior `p_risky_prior=0.99`.

This figure shows the cost of the same conservative behavior. In the correctly specified mostly-safe setting, Bayes exploits the higher-reward arm and obtains much lower cumulative regret, while the infra-Bayesian agent remains cautious because the risky-world hypothesis is still live. When Bayes is instead given a severely pessimistic prior, the greedy Bayesian agent behaves like the conservative infra-Bayesian agent.

### Balanced safe/risky worlds setting

![Balanced-risk prior comparison](results_baseline_200_pcat001_overpessimistic/balanced_prior_comparison_grid.png)

Figure 2c. Balanced-risk DGP, with `p_risky=0.5`. The left column uses the correctly specified Bayesian prior `p_risky_prior=0.5`; the right column uses the severely pessimistic prior `p_risky_prior=0.99`.

In the balanced setting, the value maximizing decision is to avoid the risky arm. Thus, both the correctly specified greedy Bayes agent and the misspecified, overly pessimistic greedy Bayes agent behave like the infra-Bayesian agent. These greedy-policy results do not distinguish IB from a sufficiently pessimistic fixed Bayesian `p_risky_prior`. Thompson sampling and Bayesian UCB can differ because their exploration strategies deliberately sample actions that greedy policies avoid. Interestingly, the pessimistic greedy Bayesian agent does not self-correct its behavior like we see in the misspecified mostly risky worlds setting. This is because the agent will never collect information to disprove its hypothesis that it is in a risky world. In choosing to only pull the safe arm, it will never gather information on the risky arm that would help it learn the world it is in is safe.

# Summary

Across these experiments, infra-Bayes behaves conservatively in a way that protects it from not knowing whether the world is risky: when Bayes has a misspecified point prior that strongly underestimates risky worlds, greedy Bayes pulls the high-reward/high-risk arm more often and suffers worse regret, while IB's performance is stable. With a correct or mildly misspecified point prior, greedy Bayes and IB are broadly similar in worlds where it "pays" to pull the guaranteed-safe arm. The tradeoff is clear in the mostly-safe, correctly specified setting (i.e. where it "pays" to pull the risky arm): Bayes exploits the high-reward arm and achieves much lower regret, while IB remains cautious because the risky-world hypothesis is still live.

# Appendix

Final cumulative expected-regret percentiles from `results_separated_arms_200_pcat001`. Brackets show 95% bootstrap CIs from 5000 resamples over worlds. This table reports the mostly-risky and mostly-safe runs from that output directory; the balanced-risk ablation is summarized in Figure 2c.

IB rows are not repeated across the three mostly-risky prior conditions because the IB agent does not use a Bayesian safe-vs-risky point prior. Rows are ordered by agent family: IB, Bayesian UCB, all greedy Bayesian `p_risky_prior` values, all Thompson-sampling Bayesian `p_risky_prior` values, then the mostly-safe results in the same order.

| DGP p_risky | Bayesian p_risky_prior | agent | catastrophe rate | p5, 95% CI | p50, 95% CI | p95, 95% CI |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0.99 | n/a | infra_bayesian | 0.040 | 0.00 [0.00, 0.00] | 9.60 [9.60, 9.60] | 144.00 [96.48, 183.36] |
| 0.99 | 0.99 | bayes_ucb | 0.075 | 9.60 [9.60, 19.20] | 38.40 [38.40, 48.00] | 134.88 [115.20, 172.80] |
| 0.99 | 0.5 | bayes_ucb | 0.635 | 48.00 [18.72, 84.96] | 595.20 [504.00, 758.40] | 950.40 [940.80, 950.40] |
| 0.99 | 0.01 | bayes_ucb | 0.635 | 48.00 [18.72, 66.72] | 585.60 [504.00, 724.80] | 940.80 [931.20, 950.40] |
| 0.99 | 0.99 | bayes_greedy | 0.040 | 0.00 [0.00, 0.00] | 9.60 [9.60, 9.60] | 144.00 [96.48, 183.36] |
| 0.99 | 0.5 | bayes_greedy | 0.040 | 0.00 [0.00, 0.00] | 9.60 [9.60, 9.60] | 144.00 [96.48, 183.36] |
| 0.99 | 0.01 | bayes_greedy | 0.650 | 48.00 [19.20, 76.80] | 609.60 [508.80, 739.20] | 960.00 [950.40, 960.00] |
| 0.99 | 0.99 | bayes_thompson | 0.075 | 9.60 [0.00, 9.60] | 28.80 [28.80, 38.40] | 124.80 [96.00, 154.56] |
| 0.99 | 0.5 | bayes_thompson | 0.465 | 48.00 [19.18, 67.20] | 499.20 [436.80, 518.40] | 614.88 [604.80, 633.60] |
| 0.99 | 0.01 | bayes_thompson | 0.645 | 57.12 [19.20, 76.80] | 595.20 [499.20, 739.20] | 950.40 [940.80, 950.40] |
| 0.01 | n/a | infra_bayesian | 0.000 | 31.58 [30.36, 34.00] | 39.60 [39.60, 39.60] | 40.00 [40.00, 40.00] |
| 0.01 | 0.01 | bayes_ucb | 0.015 | 0.40 [0.40, 0.78] | 2.00 [1.60, 2.00] | 7.24 [5.22, 12.00] |
| 0.01 | 0.01 | bayes_greedy | 0.015 | 0.00 [0.00, 0.00] | 0.40 [0.40, 0.40] | 4.80 [2.84, 6.40] |
| 0.01 | 0.01 | bayes_thompson | 0.015 | 0.38 [0.00, 0.40] | 1.60 [1.20, 1.60] | 6.82 [4.42, 7.60] |
