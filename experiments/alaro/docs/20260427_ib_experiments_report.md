# Empirical Evaluation of an Infra-Bayesian Agent in Newcomblike and Stateful Environments

**Status**: Skeleton report — methodology and predictions populated, results pending.
**Companion to**: `20260418_beliefs_refactor_plan.md` (implementation), `20260427_phase3_supra_pomdp_research.md` (architecture).

---

## Introduction

Bell et al. [1] introduce *Newcomblike Decision Processes* (NDPs) — a generalisation of MDPs in which transitions and rewards depend not only on the realised action but also on the agent's policy. They show that value-based reinforcement learning agents can converge only to *ratifiable* policies in such environments, and that there exist NDPs in which value-based RL cannot converge to any optimal policy at all. The paper concludes that "the most realistic paths to developing learning algorithms for [Newcomblike] real-world scenarios will involve value-based RL", but leaves open the question of what *non-value-based* agents can do in the same problems, and whether the negative results extend to richer classes of environments.

The infra-Bayesian framework of Kosoy and collaborators [2, 3] proposes such a non-value-based alternative. An infra-Bayesian (IB) agent represents its uncertainty about the environment as a *credal set* of hypotheses rather than a single posterior, and acts to maximise the worst-case expected reward over that set. The framework is theoretically attractive — it offers a clean treatment of policy-dependent environments, robustness to model misspecification, and a maximin formulation of updateless decision theory — but published empirical evaluations have been scarce.

This report presents four experiments designed to test whether a working implementation of an infra-Bayesian agent (i) recovers the theoretically predicted behaviour on policy-dependent bandit problems studied by Bell et al., (ii) demonstrates online learning across hypotheses in those settings, (iii) extends correctly to two-stage Newcomblike environments that go beyond the bandit case, and (iv) provides robust performance in stateful environments with model misspecification. Experiments 1–2 use the trajectory-level formulation already supported by our codebase; experiments 3–4 require the supra-POMDP extension of [3] and exercise the agent's latent-state filtering. The progression is intended to follow a single narrative: starting from the simplest setting where Bell et al.'s negative results bite, we trace the IB agent's behaviour outward through increasingly demanding regimes.

We compare throughout against two baselines, both designed to share as much architecture as possible with the IB agent so that any observed performance gap is attributable to a single design decision:

1. **Bayesian-model-averaging agent** ("Bayesian baseline"): maintains the *same* hypothesis class as the IB agent, runs the *same* belief filter (per-component Bayesian filtering over latent states, in experiments 3–4), and runs the *same* per-hypothesis value iteration. The only divergence from the IB agent is at the policy-evaluation step: where IB takes the worst case over a-measures (`min_k`), the Bayesian baseline takes the posterior-weighted average over hypothesis components. Action selection is greedy on the resulting expected value (no separate exploration noise, mirroring IB). This isolates IB's *distinctive* contribution — the credal min — from every other implementation choice.

2. **Value-based RL agent** ("Q-learning baseline"): tabular Q-learning with ε-greedy exploration (ε = 0.1, learning rate α = 0.1), matching the setup studied by Bell et al. Model-free; no hypothesis class, no belief filter, no value iteration. This is the canonical reference any RL reviewer expects to see, and it shows the cost of being model-free relative to either model-based agent.

The Bayesian baseline is the more informative comparison for theoretical claims about IB; the Q-learning baseline anchors performance against the field-standard reference. Throughout the report, *"the Bayesian agent"* refers to baseline (1) and *"value-based RL"* refers to baseline (2).

---

## Experiment 1 — Mixed-Strategy Convergence in Bandit NDPs

### Motivation

Bell et al. show that for certain bandit NDPs — most notably *Death in Damascus* — there exist no ratifiable pure policies, and value-based RL agents either fail to converge or converge to dominated mixed action frequencies. The optimal policy is a uniform mixture over the two actions. This is the simplest empirical setting in which the limitations of value-based RL in Newcomblike environments are visible, and it is the natural starting point for showing that an alternative framework can succeed.

### Methodology

We instantiate Death in Damascus as a bandit NDP with two actions (`stay`, `flee`), a perfect predictor, and reward function $R(a, \pi) = 1 - \mathbb{E}_{e \sim \pi}[\mathbb{1}[a = e]]$. The optimal policy is $\pi^\star = (0.5, 0.5)$, with value $V(\pi^\star) = 0.5$. We run three agents for $T = 1000$ rounds:

- **IB agent** with hypothesis class containing the perfect predictor (a single point hypothesis); uses the LP+SLSQP policy optimiser to maximise $V_H(\pi)$.
- **Bayesian baseline** with the same single hypothesis, posterior-weighted average expected reward (equivalent to IB in this single-hypothesis case, by design — see "Expected result" below).
- **Q-learning baseline** (tabular, $\varepsilon = 0.1$, $\alpha = 0.1$).

Primary outcome: the agent's empirical action frequency over the final 200 rounds, compared to $\pi^\star$.

### What this tests

Whether the IB agent — when given the correct hypothesis — recovers the maximin-optimal mixed policy on a problem that value-based RL is theoretically known to fail.

### Expected result and interpretation

We expect the IB agent's terminal action frequencies to lie within ±0.05 of $(0.5, 0.5)$, reproducing $\pi^\star$. We expect the value-based agent to exhibit cycling action frequencies that do not converge, consistent with Bell et al.'s Theorem 4. The classical Bayesian agent, since this is a single-hypothesis setting, should match the IB agent — confirming that any divergence from the value-based baseline is not an artefact of the IB-specific update rule but of the planning step. **If we observe the predicted result**, this establishes the IB agent as a working alternative on the simplest setting where value-based RL fails. **If the IB agent fails to converge** to $\pi^\star$, this would indicate a bug in the policy optimiser or in `Infradistribution.evaluate_policy`, since the theory unambiguously predicts $\pi^\star$ for this problem.

### Results

> **Figure 1.** Empirical action frequency $\hat\pi_t(\text{stay})$ over $t \in [0, 1000]$ for the three agents. *(plot pending)*

> **Figure 2.** Cumulative reward over $t$ for the three agents, with a horizontal line at $T \cdot V(\pi^\star) = 500$. *(plot pending)*

| Agent | $\hat\pi(\text{stay})$ | $\hat\pi(\text{flee})$ | Mean reward | $\|\hat\pi - \pi^\star\|_\infty$ |
|---|---|---|---|---|
| IB |  |  |  |  |
| Bayesian baseline |  |  |  |  |
| Q-learning baseline |  |  |  |  |

---

## Experiment 2 — Online Learning Across Predictor Hypotheses in Iterated Newcomb

### Motivation

Experiment 1 establishes that the IB agent recovers known optima when the hypothesis is correct, but does not exercise the update rule: the hypothesis class has size one. To demonstrate that the IB agent is performing genuine inference — not just static maximin optimisation — we need a setting where multiple hypotheses are consistent with prior data and the agent must reweight them online. Iterated Newcomb against an unknown predictor type provides exactly this. The experiment also lets us isolate the contribution of IB-style pessimism over the credal set, by comparing to a classical Bayesian agent on the *same* hypothesis class.

### Methodology

We construct an iterated Newcomb environment with two actions (`one-box`, `two-box`) and an unknown predictor drawn from a finite class of three predictor types: (a) **perfect** ($P_{ij} = \mathbb{1}[i = j]$), (b) **biased** (predicts one-box with probability 0.7 regardless of the agent's policy), (c) **uniform** ($P_{ij} = 0.5$). Each agent is given a uniform prior over these three hypotheses. We run $N = 100$ episodes of length $T = 200$ for each agent, sampling the true predictor type from a fixed but agent-blind distribution.

Agents:
- **IB agent** with the three-hypothesis credal mixture; uses `Infradistribution.update` to reweight hypotheses online via the likelihood of the observed `observation`.
- **Bayesian baseline** maintaining a posterior over the same three hypotheses (using the same per-hypothesis log-marginal-likelihood machinery as the IB agent's `_posterior_weights`), choosing the action with highest *posterior-weighted average* expected reward. The only difference from the IB agent is min vs. average over hypotheses.
- **Q-learning baseline** (tabular, $\varepsilon = 0.1$).

Primary outcomes: (i) cumulative reward at $T = 200$, averaged across episodes and stratified by true predictor type; (ii) posterior weight (or scale-ratio for IB) on the true predictor over time.

### What this tests

Two things at once. First, that the IB agent's update rule reallocates weight to the correct hypothesis when likelihoods differ across the credal set — the Phase 2 anchor for "we have an actual learning IB agent". Second, that the IB agent's behaviour differs from a Bayesian agent on the same hypothesis class in the way the theory predicts: more cautious action choice when multiple hypotheses are still plausible, converging to similar performance once the hypothesis is identified.

### Expected result and interpretation

We expect (i) all three agents to identify the true predictor within $\sim$50 rounds when it is the *biased* or *uniform* type, with comparable terminal reward; (ii) the IB agent to commit to one-boxing earlier than the classical Bayesian agent under the *perfect* predictor, because the worst-case-over-hypotheses pessimism upweights perfect-predictor consequences while posterior is still diffuse; (iii) the value-based agent to converge to two-boxing under all three predictors during the early exploration phase, recovering only slowly once Q-values reflect policy-dependence. **If the IB agent's hypothesis weights converge to the truth and its early-round reward strictly exceeds the classical Bayesian agent's under the perfect predictor**, we have direct empirical evidence that IB pessimism provides a tangible advantage over Bayesian model averaging in policy-dependent environments — the cleanest possible demonstration of the framework's distinctive contribution. **If the IB and Bayesian agents perform identically**, the credal-set width in this construction is too narrow to differentiate them; we would re-run with hypotheses spanning a wider range of predictor accuracies.

### Results

> **Figure 3.** Posterior weight (Bayesian) and scale-ratio (IB) on the true predictor hypothesis as a function of round $t$, averaged over episodes, faceted by true predictor type. *(plot pending)*

> **Figure 4.** Cumulative reward over $t$ for the three agents, faceted by true predictor type. *(plot pending)*

| True predictor | Agent | Mean reward at $T = 200$ | Round at which $\hat\pi(\text{one-box}) > 0.9$ | Final hypothesis-weight on truth |
|---|---|---|---|---|
| Perfect | IB |  |  |  |
| Perfect | Bayesian baseline |  |  |  |
| Perfect | Q-learning baseline |  |  |  |
| Biased | IB |  |  |  |
| Biased | Bayesian baseline |  |  |  |
| Biased | Q-learning baseline |  |  |  |
| Uniform | IB |  |  |  |
| Uniform | Bayesian baseline |  |  |  |
| Uniform | Q-learning baseline |  |  |  |

---

## Experiment 3 — Two-Stage Newcomblike Environments (Transparent Newcomb)

### Motivation

Bell et al.'s analysis is restricted to bandit NDPs (single state, length-1 trajectories). Their negative results may or may not extend to multi-stage Newcomblike environments — and a working IB agent with a state-level world model gives us the first opportunity to ask whether the theoretical advantages persist when the environment has internal structure. Transparent Newcomb is the canonical two-stage policy-dependent problem: the predictor commits to filling or leaving the box based on a prediction of the agent's policy *contingent on observation*, then the agent observes the box and acts. The optimal (UDT) policy is "one-box if full, anything if empty". Standard CDT and value-based RL provably cannot reach this policy [3, §"Acausal Hypotheses and Surmeasures"]; UDT can.

This experiment requires the supra-POMDP extension to the world model: the latent state is the predictor's prediction, the observation is the box state, and the reward is determined by both. Its purpose is to establish that the implemented IB-with-supra-POMDP agent reproduces the UDT-optimal behaviour on the simplest two-stage Newcomblike problem in the literature.

### Methodology

We instantiate Transparent Newcomb with $\varepsilon$-error: the predictor predicts correctly with probability $1 - \varepsilon$ and with probability $\varepsilon$ predicts the opposite policy. We use $\varepsilon = 0.05$, which makes the hypothesis pseudocausal [3] and keeps the problem within the range our agent's `WorldModel` interface supports. The environment is encoded as a supra-POMDP with $|S| = 2$ latent states (predicted-one-box, predicted-two-box), observation $o \in \{\text{full}, \text{empty}\}$ deterministic in $s$, and rewards as in standard Transparent Newcomb. The agent class:

- **IB agent (supra-POMDP)**: hypothesis class containing two predictor hypotheses (the $\varepsilon$-imperfect predictor and a uniform-random predictor); uses `SupraPOMDPWorldModel` to filter beliefs over the latent prediction.
- **Bayesian baseline (supra-POMDP)**: identical `SupraPOMDPWorldModel`, identical belief filter, identical hypothesis class — only `evaluate_action` differs (posterior-weighted average over hypothesis components instead of credal min).
- **Q-learning baseline**: tabular, over the observation-conditioned policy space (states = box observations, actions = $\{$one-box, two-box$\}$).

Primary outcome: the agent's policy contingent on observation, after $T = 500$ rounds.

### What this tests

(i) That the supra-POMDP world model correctly maintains and updates a credal belief over latent states, and (ii) that the IB agent acting on those beliefs reproduces UDT-optimal behaviour on a problem where value-based RL is known to fail.

### Expected result and interpretation

We expect the IB agent's terminal policy to be approximately $\pi(\text{one-box} \mid \text{full}) \approx 1$ with $\pi(\cdot \mid \text{empty})$ unconstrained. We expect the value-based RL agent to converge to a CDT-like policy: $\pi(\text{two-box} \mid \cdot) \approx 1$ in both observation states, since two-boxing dominates one-boxing for any fixed posterior over the box state. The classical Bayesian agent's behaviour is the most theoretically interesting: depending on whether the posterior averaging is taken over predictor hypotheses or over (predictor × box) joint states, it may behave like CDT (two-box always) or like the IB agent. We will report both possibilities.

**If the IB agent reproduces UDT-optimality and the value-based agent converges to two-boxing**, we have empirical evidence that the supra-POMDP IB framework operationalises UDT on a non-trivial multi-stage problem — the central theoretical claim of [3]. **If the IB agent fails to one-box on full**, the most likely cause is that the credal-set width is insufficient, and we will re-run with a wider hypothesis class (e.g., including a perfect-predictor hypothesis at the boundary of pseudocausality).

### Results

> **Figure 5.** Action probability $\hat\pi_t(\text{one-box} \mid \text{box state})$ over $t$ for each agent, faceted by box observation. *(plot pending)*

> **Figure 6.** Belief state — IB credal interval and Bayesian posterior — on the $\varepsilon$-imperfect predictor hypothesis, over $t$. *(plot pending)*

| Agent | $\hat\pi(\text{one-box} \mid \text{full})$ | $\hat\pi(\text{one-box} \mid \text{empty})$ | Mean reward | UDT-optimal? |
|---|---|---|---|---|
| IB (supra-POMDP) |  |  |  |  |
| Bayesian baseline (supra-POMDP) |  |  |  |  |
| Q-learning baseline |  |  |  |  |

---

## Experiment 4 — Robust Decision-Making Under Latent-State Misspecification

### Motivation

The previous three experiments target the policy-dependent regime that motivates Bell et al. The fourth experiment turns to the *other* core motivation for infra-Bayesianism: robustness to model misspecification in stateful environments. If IB's distinctive contribution is "worst-case over a credal set", we should be able to show that a credal mixture of stateful-environment hypotheses outperforms a single Bayesian posterior on the *same* hypothesis class when the true environment is at the edge of, or just outside, the class. This is the regime that motivates [4]'s robust MDP work and that has no equivalent in classical RL theory without strong realisability assumptions.

This experiment is also a stress-test of the supra-POMDP world model in a setting unrelated to policy-dependence. Showing that the same machinery handles both Newcomblike and robust-RL settings demonstrates the breadth of the IB framework relative to classical alternatives, each of which addresses one regime but not both.

### Methodology

We construct a tabular gridworld with $|S| = 16$ cells (a 4 × 4 grid), four cardinal actions, deterministic transitions away from a designated trap cell, and stochastic transitions in a 1-cell radius around the trap (representing localised model uncertainty). Reward is $-1$ per step, $-100$ at the trap, $+10$ at the goal. The agent does not know the trap cell. Hypothesis class: 16 hypotheses, one per possible trap location.

The true environment is sampled from one of two regimes per episode:
- **Realisable**: trap is at one of the 16 hypothesised locations. Either agent can in principle identify it.
- **Misspecified**: trap is at the hypothesised location *with probability 0.7* and at a neighbouring cell *with probability 0.3*. No hypothesis exactly matches the true environment.

Agents:
- **IB agent (supra-POMDP)**: 16-hypothesis credal mixture; latent state is current grid cell; planning by maximising worst-case expected return via the supra-POMDP belief filter and infinite-horizon value iteration per hypothesis.
- **Bayesian baseline (supra-POMDP)**: identical `SupraPOMDPWorldModel`, identical 16-hypothesis class, identical belief filter and value iteration. Differs from IB only in the planning step: posterior-weighted average over hypothesis components instead of credal min.
- **Q-learning baseline**: tabular, $\varepsilon = 0.1$. No model.

Primary outcomes: (i) mean and worst-case episode return over $N = 200$ episodes; (ii) trap-hit rate; (iii) posterior or credal weight on the true trap location over time, in the realisable regime.

### What this tests

Whether the IB framework's worst-case-over-hypotheses planning produces meaningfully safer behaviour than Bayesian posterior averaging when (a) the hypothesis class is correct and (b) the hypothesis class is misspecified. Both regimes are run because the theoretically distinctive prediction is that IB's advantage *grows* under misspecification: when the truth lies outside the class, the Bayesian agent has no realisability guarantee, while the IB agent's credal min still bounds its worst-case loss.

### Expected result and interpretation

In the realisable regime, we expect both Bayesian and IB agents to identify the trap within $\sim$10 episodes and to perform comparably thereafter; both should outperform value-based RL on cumulative return and trap-hit rate. In the misspecified regime, we expect the Bayesian agent to occasionally walk into traps it has confidently mislocated (high mean return but heavy left tail), the IB agent to maintain a wider credal envelope and avoid uncertain regions (slightly lower mean return, much smaller left tail), and the value-based agent to be uniformly worst.

**If the IB agent's worst-case return strictly exceeds the Bayesian agent's worst-case return under misspecification**, we have empirical evidence that IB's robustness guarantee is not just theoretical: it produces measurably safer behaviour at a controlled cost in mean reward. This is the strongest possible single-experiment argument for IB in stateful environments. **If IB and Bayesian agents have indistinguishable performance even under misspecification**, the credal set construction is too narrow to express the relevant uncertainty and we would re-run with a larger hypothesis class (e.g., trap location × trap reward magnitude).

### Results

> **Figure 7.** Distribution of episode returns over $N = 200$ episodes, by agent and regime, with worst-case (5th percentile) and mean overlaid. *(plot pending)*

> **Figure 8.** Trap-hit rate over episode index, by agent and regime. *(plot pending)*

> **Figure 9.** Mean–worst-case Pareto frontier for IB at varying credal-set widths, with Bayesian and value-based RL points overlaid. *(plot pending)*

| Regime | Agent | Mean return | 5th-percentile return | Trap-hit rate | Goal-reach rate |
|---|---|---|---|---|---|
| Realisable | IB |  |  |  |  |
| Realisable | Bayesian baseline |  |  |  |  |
| Realisable | Q-learning baseline |  |  |  |  |
| Misspecified | IB |  |  |  |  |
| Misspecified | Bayesian baseline |  |  |  |  |
| Misspecified | Q-learning baseline |  |  |  |  |

---

## Discussion

The four experiments are arranged to walk a reader along an increasing-difficulty path: from a single-hypothesis maximin recovery (Exp. 1), to online inference across a finite hypothesis class in the same bandit-NDP regime (Exp. 2), to two-stage policy-dependent environments requiring latent-state filtering (Exp. 3), to stateful environments with model misspecification (Exp. 4). Each experiment is designed to be informative under any outcome: the predicted results constitute positive evidence for the framework, while specific failure modes point to identifiable diagnostics in the implementation or the experimental construction.

Three threads run through the set. First, all four experiments include a **Q-learning baseline** specifically chosen to match the conditions of Bell et al.'s negative results, so that the IB agent's behaviour is comparable to a known-failing model-free alternative on the same problems. Second, all four include a **Bayesian baseline** that shares the IB agent's hypothesis class, belief filter, and (for experiments 3–4) per-hypothesis value iteration — differing only in the planning step (posterior-weighted average versus credal min). This isolates IB's distinctive contribution from every other architectural decision and makes any observed performance gap between IB and the Bayesian baseline directly attributable to the credal-min rule. Third, experiments 3 and 4 collectively demonstrate that the same supra-POMDP world model handles both the policy-dependence regime (the focus of Bell et al.) and the model-misspecification regime (the focus of Appel and Kosoy [4]), suggesting that IB is not a narrow alternative for one or the other but a unifying framework.

If the predicted results hold, this report would provide — to our knowledge — the first empirical demonstration of a working infra-Bayesian agent on (a) the bandit NDPs of Bell et al., (b) the two-stage Newcomblike environments anticipated by the belief-functions framework of [3], and (c) the robust stateful environments of [4]. Negative or partial results would themselves be informative, and would direct future work toward the specific component (planning, update rule, credal-set construction, or supra-POMDP encoding) responsible for the gap.

---

## References

[1] J. Bell, L. Linsefors, C. Oesterheld, J. Skalse. *Reinforcement Learning in Newcomblike Environments*. NeurIPS 2021. <https://proceedings.neurips.cc/paper_files/paper/2021/file/b9ed18a301c9f3d183938c451fa183df-Paper.pdf>

[2] V. Kosoy, A. Appel. *Infra-Bayesianism Sequence I–III* (Basic Inframeasure Theory). Alignment Forum, 2020–2021. <https://www.alignmentforum.org/posts/YAa4qcMyoucRS2Ykr/basic-inframeasure-theory>

[3] Diffractor, V. Kosoy. *Belief Functions and Decision Theory*. Alignment Forum, 2020. <https://www.greaterwrong.com/posts/e8qFDMzs2u9xf5ie6/belief-functions-and-decision-theory>

[4] A. Appel, V. Kosoy. *Regret Bounds for Robust Online Decision Making*. arXiv:2504.06820, 2025. <https://arxiv.org/pdf/2504.06820>
