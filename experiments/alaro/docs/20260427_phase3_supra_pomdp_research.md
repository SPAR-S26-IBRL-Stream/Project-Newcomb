# Phase 3 Research: Supra-POMDPs and the Set-vs-Point-Valued Kernel Decision

**Date**: 2026-04-27
**Status**: Pre-implementation research, decision pending
**Companion to**: `20260418_beliefs_refactor_plan.md` (Phases 1–2 plan; Phase 3 sketch in §"Phase 3: Supra-POMDP")

This document is intended to be self-contained: a downstream agent should be
able to read only this file and understand (a) the high-level architecture
question, (b) what each external source says, and (c) the tradeoffs. Links
to external sources are preserved for cases where someone needs to revisit
the originals, but the substantive content is summarised inline.

---

## 1. The question

Phases 1 and 2 of `20260418_beliefs_refactor_plan.md` are largely complete on
branch `fllor/ib-features`. They provide:

- A `WorldModel` abstraction with `belief_state` and opaque `params`.
- `MultiBernoulliWorldModel` (count-based sufficient statistic for IID
  Bernoulli arms).
- `NewcombWorldModel` (predictor matrix + accumulated log-likelihood for
  policy-dependent environments).
- An `InfraBayesianAgent` that maintains one shared `Infradistribution`
  across actions and optimises over policies via LP / SLSQP.

These cover **trajectory-level hypotheses** of a very narrow form: length-1
trajectories, IID actions, optionally policy-dependent rewards (Newcomb).
They do not cover environments with internal state — anything where today's
reward distribution depends on yesterday's history in a way that isn't a
sufficient statistic.

Phase 3 is meant to fix this by introducing a **supra-POMDP** world model: a
factored state-space representation of the same trajectory-level hypothesis
via three kernel-valued objects (initial-state distribution, transition
kernel, observation kernel). The open architecture question is:

> **Should the transition kernel `T` (and initial state Θ₀ and observation
> kernel B) be point-valued (one distribution per (s, a)) or set-valued
> (an infrakernel — a convex set of distributions per (s, a))?**

Within-hypothesis Knightian uncertainty either lives at the kernel level
(set-valued T) or at the hypothesis-class level (a credal mixture of
point-T POMDPs maintained by `Infradistribution.mix`).

---

## 2. Glossary and notation

| Symbol | Meaning |
|---|---|
| `A`, `O`, `S` | finite action / observation / state spaces |
| `Π` | policy space |
| `ΔX` | probability distributions on X |
| `□X` | infradistributions on X — convex, downward-closed sets of a-measures on X (Knightian-valued probability) |
| a-measure | pair (m, b) of a non-negative measure m and offset b ≥ 0 |
| `Λ : Π → □((A×O)^ω)` | trajectory-level causal law: for each policy, an infradistribution over infinite (action, observation) streams |
| `Θ₀ ∈ □S` | initial-state infradistribution |
| `T : S × A → □S` | (set-valued) transition infrakernel |
| `B : S → O` (or `S → □O`) | observation kernel |
| `M : A → □O` | Appel-Kosoy "robust model": multivalued action → distribution-set map |
| `σ \|= M` | "selection" σ is consistent with set-valued model M (picks one point distribution from each M(s,a)) |
| `σ ▷◁ π` | semidirect product: trajectory distribution from running selection σ against policy π |
| RMDP | Robust MDP (set-valued transition kernel) |
| NDP | Newcomblike Decision Process (policy-dependent point-valued MDP) |
| RUE / E2D | Robust Universal Estimator / Estimations to Decisions (Appel-Kosoy algorithms) |
| DEC | Decision-Estimation Coefficient |
| Nirvana | "infinite reward" trick used to encode policy selection in Kosoy's belief functions |

---

## 3. Phase 1+2 architecture, in one paragraph

A hypothesis in current code is an `Infradistribution` whose `WorldModel`
subclass answers "given policy π, action a, and a history-summary belief
state, what is P(next outcome) and E[reward]?". This is a *trajectory-level*
representation: the mathematical object is `Λ : Π → □((A×O)^ω)`, with the
sample space being the entire infinite action-observation stream. The
implementation only works because Bernoulli outcomes admit count-based
sufficient statistics and Newcomb is length-1, so we never have to
materialise full trajectories. For any environment with non-trivial
internal state, no compact sufficient statistic exists and `belief_state`
would have to be the full history. That is the wall Phase 3 wants to break
through, by re-encoding hypotheses as supra-POMDPs (initial state +
transition + observation, in inframeasure-theoretic kernels).

---

## 4. Source 1: Kosoy / Diffractor — "Belief Functions and Decision Theory"

**URL**: <https://www.greaterwrong.com/posts/e8qFDMzs2u9xf5ie6/belief-functions-and-decision-theory>
(also on Alignment Forum / LessWrong)

**Authors**: Diffractor (Vanessa Kosoy's collaborator); part of the
Infra-Bayesianism Sequence.

### What it covers

This is the trajectory-level / belief-function part of the IB sequence. It
formalises:

- **Belief function `Θ`**: assigns each "policy stub" `π_pa` to a *non-empty
  set of a-measures over F(π_pa)*. So Θ is set-valued at the trajectory level.
  If Θ satisfies nine listed conditions (nonemptiness, restricted minimals,
  Hausdorff continuity, etc.) it is called a hypothesis.
- **Causal / pseudocausal / acausal / surcausal** taxonomy:
  - *Causal* — may include Nirvana (infinite reward marker), satisfies a
    causality condition that lets Θ be reconstructed as a set of "a-environments".
  - *Pseudocausal* — Nirvana-free but satisfies a weaker version of causality.
    Used for Counterfactual Mugging, Newcomb, Death in Damascus, XOR Blackmail,
    Transparent Newcomb with ε-error.
  - *Acausal* — Nirvana-free, no causality assumption. Needed only for problems
    like *perfect* Transparent Newcomb where you can be locked out of disproving
    a misprediction.
  - *Surcausal* — causal hypothesis over "a-surmeasures" (allows infinitesimal
    probabilities for Nirvana tricks).
- **Hypothesis Translation Theorems**: causal ↔ pseudocausal ↔ acausal ↔
  surcausal are inter-convertible by adding/removing Nirvana.
- **Theorem 4 (Belief Function Bayes)**: mixing hypotheses to make a prior,
  then updating, equals mixing the updated hypotheses weighted by the
  probability they put on the observation. Direct analogue of classical Bayes.
- **Theorem 6 (Maximin UDT)**: A set `S` of policy-selection environments
  with uniform modulus of continuity translates to an acausal hypothesis
  via `Θ?ω(π) := {(m, b) | b=0, ∃e ∈ S: π·e = m}`, then closing under the
  IB conditions. This thing acts as maximin UDT on policy-dependent
  environments and reproduces UDT exactly when |S|=1.

### What it says about supra-POMDPs

Almost nothing directly:

> "there's material on learnability and infra-POMDP's and Game Theory and
> 'we have a theorem prover, what do we do with that' deferred for a later
> post."

But it provides the crucial **bridge**: a single set-valued belief function
Θ is interconvertible with a *set* of point-valued environments
(a-environments). So at the trajectory level, "credal mixture of point
environments" and "single set-valued Θ" are the same object. This is the
basis for the equivalence claim in §7.

### Why it matters for the kernel-level decision

This post tells us the trajectory-level theory tolerates *both* set-valued
and point-environment-mixture representations. It does *not* tell us which
factorisation is preferable at the kernel level — that is exactly what's
deferred to the (yet-unwritten) infra-POMDP post.

---

## 5. Source 2: Appel-Kosoy 2025 — "Regret Bounds for Robust Online Decision Making"

**URL**: <https://arxiv.org/pdf/2504.06820>

**Existing project notes**: `20260404_appel_kosoy_comparison.md`,
`20260404_rue_e2d_plan.md` — those documents focus on potentially
implementing E2D + RUE as an alternative agent. They are *not* about Phase 3
architecture, but they share vocabulary.

### Core formalism

A "model" is `M : A → □O` — a multivalued (set-valued) function from
actions to convex sets of distributions over observations. The hypothesis
class `H` is a set of such models. Nature can adversarially pick any
distribution from `M(a)`, possibly depending on past history (non-oblivious
adversary). The agent has uncertainty at *both* levels:

1. **Within-model**: nature adversarially picks from `M(a)` each step.
2. **Across-model**: which `M ∈ H` is the true model?

This is the paper's core relaxation of standard online learning: instead of
"one true distribution per (a, history)", you get "a *constraint set* per
(a, history) plus uncertainty about which constraint set is the true one".

### Tabular RMDP (their stateful special case)

The paper's tabular RMDP setting is the closest analogue to our Phase 3
goal. From §4.2 and Appendix K:

- **RMDP type**:
  ```
  RMDP := □([0,1] × S)                               # initial reward+state
        × ([H−1] × S × A → □([0,1] × S))             # transition infrakernel
        × (S × A → □[0,1])                           # terminal reward infrakernel
  ```
  All three components are **set-valued** (`□`).

- **Selection of an RMDP** (Definition 7): a function
  `σ : (S×A×[0,1])^≤H × S × A → Δ([0,1]×S)`
  such that `σ(tr<h, s, a) ∈ M(h, s, a)` for all `h, tr<h, s, a`. Selections
  are exactly the point-valued environments compatible with M.

- **Trajectory model**: `M ↦ {σ ▷◁ π | σ |= M}` — the set of trajectory
  distributions you get by letting all selections interact with policy π.
  This is the bridge from kernel-level RMDP to trajectory-level imprecise
  belief.

- **Hypothesis class**: all 1-bounded RMDPs with fixed S, A, H. Infinite
  class — handled via covering numbers (their Definition 6).

- **Halfspace RMDPs** (`H_parhalf`): a tractable subclass where each
  `M(h, s, a)` is a halfspace `{μ ∈ Δ([0,1]×S) | E_μ[f(s)+r] ≥ c}` —
  one linear constraint, three parameters. Estimation tractable here.

### Update mechanism

RUE (Algorithm 2): prediction-market over "bettors" (one per model in H +
uniform bettor + pessimistic bettor). Wealth distribution `ζ` over bettors
updated multiplicatively via Hellinger distance:

```
ζ(B) ← ζ(B) · (√(μ_B(o) / M̂(a)(o)) + D²_H(M̂(a) → M_B(a)))
```

Asymmetric squared Hellinger `D²_H(M̂(a) → M(a)) = max_{μ∈M̂(a)} min_{ν∈M(a)} D²_H(μ, ν)`
is the key metric; it measures whether `M̂` is "almost a subset" of `M`,
which is the right notion when models are set-valued.

E2D (Algorithm 1): solves a constrained min-max LP per step using RUE's
estimate `M̂_t`. Confidence set `H_t` shrinks over time as models with high
cumulative loss are eliminated.

### Regret bounds

Theorem 1 gives `REG ≤ 2T·dec_ε^{f,L}(H) + α(T,δ) + 2Tδ`.

For tabular RMDPs (§4.2): `Õ(√(poly(H,S,A) · T))` regret on all 1-bounded
RMDPs in the episodic RMDP setting. The proof relies critically on the
halfspace structure of `H_parhalf` and the asymmetric Hellinger optimisation.

### What it implies for the kernel-level decision

This is the strongest argument *for* set-valued T: Appel-Kosoy give a
formal regret bound for stateful environments with adversarial per-step
transitions, achieved by exploiting the halfspace structure of the
infrakernel. With point-valued T plus credal mixing, the asymmetric
Hellinger collapses to ordinary Hellinger between point distributions, the
LP loses its halfspace structure, and the regret guarantee does not
transfer.

This does not mean point-valued T can't reach the same regret on the same
problems — it means *Appel-Kosoy's specific machinery* doesn't lift, and
any equivalent guarantee would need a different proof.

---

## 6. Source 3: Bell, Linsefors, Oesterheld, Skalse — NeurIPS 2021

**URL**: <https://proceedings.neurips.cc/paper_files/paper/2021/file/b9ed18a301c9f3d183938c451fa183df-Paper.pdf>

**Title**: "Reinforcement Learning in Newcomblike Environments"

### Important clarification

This is **not** a Kosoy paper. It is by Bell, Linsefors, Oesterheld, Skalse,
and uses no inframeasure machinery at all. The reason it's relevant to our
question is that it occupies the opposite extreme of the
Knightian-uncertainty design space: zero set-valuedness anywhere.

### Formalism

**NDP (Newcomblike Decision Process)**: a tuple `⟨S, A, T, R, γ⟩` where:

- `S`, `A` finite.
- `T : S × A × (S → ΔA) → ΔS` — a *stochastic* transition function that
  takes a policy as additional argument. ("Nondeterministic" in the paper
  means random / distribution-valued, not multivalued. Each (s, a, π) gives
  a single distribution over next states.)
- `R : S × A × S × (S → ΔA) → ΔR` — analogously policy-dependent
  stochastic reward.
- Single true environment, no hypothesis class, no Knightian uncertainty.

A *bandit NDP* has |S|=1 and γ=0; this is the setting most of their results
target. Newcomb's Problem is encoded as a bandit NDP with two actions
where `R(a₁, π) = 0 w.p. π(a₂), 10 w.p. π(a₁)` etc.

### Results

- **Ratifiability** (§2): a value-based RL agent can converge only to a
  policy π such that every action with positive probability under π is
  optimal under the Q-values induced by π. Some NDPs have no optimal
  ratifiable policy.
- **Convergence failures** (§3): there exist NDPs in which value-based RL
  cannot converge to *any* policy.
- **Action frequencies** (§4): even when policies don't converge, action
  frequencies sometimes do; conditions for this are characterised.

### What it implies for the kernel-level decision

NDPs are the simplest possible formalisation of policy-dependent
environments and they get by entirely with point-valued kernels. This is
*existence proof* that point-valued T is enough for Newcomb-family
problems if you're willing to live with policy-dependence in the kernel
type signature. It says nothing about the harder regimes Appel-Kosoy
tackle (per-step adversarial transitions, robust online RL).

It also illustrates a crucial design point: **even with point-valued T,
you may need T to depend on π** to handle Newcomb. Our current
`NewcombWorldModel` already does this via the `policy` argument; a
supra-POMDP would need to decide whether the same policy-dependence
threads into `T : S × A × Π → ΔS` or whether policy-dependence is
factored out elsewhere.

---

## 7. Synthesis: do the three sources point to the same approach?

**No.** They sit at three different points on the spectrum:

| | Within-hypothesis | Across-hypothesis | Kernel-level T |
|---|---|---|---|
| Bell et al. (NDP) | None | None (single env) | Point, but π-dependent |
| Belief Functions (Kosoy) | Set-valued Θ at trajectory level | Yes (mix Θᵢ) | Not addressed (kernel level deferred) |
| Appel-Kosoy 2025 (RMDP) | Set-valued T (infrakernel) | Yes (hypothesis class H) | Set-valued |

But there is a **bridge** that lets us reason about the choice:

> A single set-valued POMDP (RMDP / supra-POMDP) is equivalent to the
> credal set of its point-valued *selections*. Mixing point-valued
> POMDPs in `Infradistribution.mix` already gives us across-hypothesis
> Knightian uncertainty — the same effect at the trajectory level.

Formally, this is what Theorem 6 of the Belief Functions post is doing:
turning a *set* of policy-dependent point environments into a *single*
acausal belief function. So at the trajectory level, "credal mixture of
point environments" and "single set-valued Θ" are formally
inter-translatable.

Where the equivalence is *not* tight is at the kernel level:

- **Finite credal mixture vs. infinite/continuous infrakernel**: a
  halfspace `{μ : E_μ[f(s)+r] ≥ c}` contains an uncountable family of
  distributions. Representing it as a credal mixture of point T's needs
  infinite covering, which is theoretically fine (Appel-Kosoy use covering
  numbers) but not what we'd implement.
- **Oblivious vs. non-oblivious adversary**: a finite mixture of point T's
  has nature pick one model and stick with it across the episode (oblivious).
  A genuine infrakernel has nature pick adversarially each step, with the
  pick possibly depending on history (non-oblivious).

These are real differences but only matter for specific use cases.

---

## 8. The Phase 3 sketch from `20260418_beliefs_refactor_plan.md`

From §"Phase 3: Supra-POMDP", the existing draft (which is the starting
point for this decision):

```python
class SupraPOMDPWorldModel(WorldModel):
    """Fund 4: belief state is a distribution over latent states.
    Params: tuple (transition, observation, initial) where
        transition: shape (num_states, num_actions, num_states) — T[s, a, s']
        observation: shape (num_states, num_obs) — B[s, o]
        initial: shape (num_states,) — Θ₀
    """
    def make_params(self, transition, observation, initial):
        return (transition, observation, initial)

    def update_state(self, belief, outcome, action=None, params=None, policy=None):
        transition, observation, _ = params
        belief_pred = belief @ transition[:, action, :]
        obs = self.event_index(outcome)
        belief_post = observation[:, obs] * belief_pred
        return belief_post / belief_post.sum()
    ...
```

The sketch as written is **point-valued** (T is a numpy array, one
distribution per (s, a)). Set-valued T would change `params` to something
like a list of (T, B, Θ₀) tuples, or a parametric description of a convex
set (e.g. halfspace constraints), and would require redesigning
`update_state` to maintain a credal set rather than a point distribution
over S.

Plan-file mapping:

| | Phases 1–2 | Phase 3 (sketch) |
|---|---|---|
| `belief_state` | sufficient statistic of trajectory | distribution over latent states |
| `update_state` | extend sufficient statistic | Bayesian filter: predict via T, condition via B |
| `params` | trajectory distribution params | kernel params (T, B, Θ₀) |
| `compute_likelihood` | posterior predictive from statistic | marginalise over latent states |

The `WorldModel` interface from Phase 1 was *designed* to support this
without changes to `AMeasure`, `Infradistribution`, or the agent — `params`
and `belief_state` are opaque from the outside.

---

## 9. Pros and cons: set-valued T vs. point-valued T

Both approaches keep the across-hypothesis credal mixture provided by
`Infradistribution.mix`. The question is whether *each individual hypothesis*
also carries Knightian uncertainty inside its kernel.

### Point-valued T (one distribution per (s, a), credal across hypotheses)

**Pros**:

1. **Smaller, simpler implementation.** `T` is a `(|S|, |A|, |S|)` numpy
   array. `update_state` is a one-line Bayesian filter. No infrakernel
   data structure to design.
2. **Belief state is a single distribution over S.** Updates in O(|S|² · |A|)
   per step. Compose well with existing `_predictive`-style code paths.
3. **Hypothesis params are physically interpretable.** Writing down a
   hypothesis means writing down a POMDP — a thing every RL practitioner
   already knows how to specify.
4. **Recovers ordinary POMDPs as the special case |H|=1.** Any classical
   POMDP can be a single-hypothesis IB agent. Useful for sanity-checking
   against known baselines.
5. **Sufficient for all current target problems**: Newcomb,
   Death-in-Damascus, Counterfactual Mugging, Transparent Newcomb (with ε),
   plus standard RL gridworlds and bandits. Each can be encoded as a
   credal mixture of point POMDPs.
6. **Matches Bell et al.'s NDP formalism** for the policy-dependent
   subset, after threading π into T's signature if needed.
7. **Smaller surface for bugs.** No need to handle "nature picks
   adversarially per step inside this hypothesis" — pessimism happens once,
   at the `min_k` over a-measures step that already exists.
8. **Easier to mix with `Infradistribution.mix`.** Combining point-T
   hypotheses is concatenation of components; combining set-T hypotheses
   requires intersecting/joining convex sets at every (s, a).

**Cons**:

1. **No compact representation of halfspace / interval-valued robust
   transitions.** "Next state probability is in [0.4, 0.6]" needs an
   infinite mixture; we'd have to discretise.
2. **No per-step adversarial transitions within a hypothesis.** Nature is
   effectively oblivious within a hypothesis (it picks one POMDP and
   sticks). Across the hypothesis class, the worst-case is taken once via
   the IB min, which is itself a per-step worst-case at the trajectory
   level, but the *kernel* dynamics inside each hypothesis are deterministic
   in the imprecise-probability sense.
3. **Cannot directly reproduce Appel-Kosoy regret bounds.** Their RUE/E2D
   machinery uses asymmetric Hellinger over set-valued M(a); with point M
   this collapses and the LP structure is lost. We'd need a different
   proof to claim equivalent guarantees.
4. **Cannot represent acausal-but-not-pseudocausal hypotheses**. Perfect
   Transparent Newcomb (no ε error) is the canonical example. With finite
   point hypotheses we get pseudocausal at best. (For the ε-error variant,
   point-valued is fine — it's pseudocausal.)
5. **Over time, the credal set size needed to approximate a robust
   environment grows.** If we discover we need halfspace-valued T later,
   migrating from point-mixture to set-valued is a non-trivial refactor —
   but `params` is opaque, so it's contained inside `WorldModel`.

### Set-valued T (infrakernel per hypothesis, credal across hypotheses)

**Pros**:

1. **Full theoretical generality** of Kosoy's IB framework. Any
   hypothesis Θ that satisfies the nine belief-function conditions can be
   represented at the kernel level (modulo causal/acausal taxonomy issues).
2. **Compact representation of halfspace / interval / Lipschitz robust
   models.** A halfspace RMDP is three numbers per (s, a); a credal set
   approximation of the same thing is enormous.
3. **Per-step adversarial transitions.** Nature can adversarially pick a
   transition distribution at every step, possibly depending on history.
   Matches Appel-Kosoy's "non-oblivious adversary" assumption.
4. **Direct port of Appel-Kosoy regret machinery** (RUE, E2D, asymmetric
   Hellinger, halfspace LP). If we ever want their regret bounds, we need
   set-valued models.
5. **Cleaner semantics for "model misspecification" robustness.** "I think
   the true T is somewhere in this convex set" is one set-valued kernel,
   not a discretised mixture.
6. **Compose more naturally with infinite hypothesis classes** via
   covering numbers, the way Appel-Kosoy handle |H| = ∞.

**Cons**:

1. **Belief state is a credal set over S, not a single distribution.**
   Update is non-trivial: maintaining the convex hull of posterior
   distributions through Bayesian filtering can grow vertex count
   per step. Listed as a known issue in `20260328_infrabayes_issues.md`
   §2 ("Fixed Vertex Set After Conditioning"): the credal set of
   reachable posteriors can gain new extreme points after conditioning.
2. **Substantially more code and design surface.** Infrakernel data
   structure, set-valued operations (intersect, join, project, push
   forward through B), and update rule all need to be designed from scratch.
3. **Mixing two set-valued T's in `Infradistribution.mix` is not just
   concatenation.** Need to define what a "mixture of infrakernels" is
   — at minimum a Minkowski-style convex hull at each (s, a), which
   has its own algorithmic cost.
4. **Numerical issues already observed** at trajectory level
   (cohomogeneity blowup; `20260328_infrabayes_issues.md` §1) get worse
   with more imprecision per step.
5. **Excess pessimism risk.** The per-step `min_k` already produces
   "different worst-case adversary at every step" pessimism
   (`20260328_infrabayes_issues.md` §5). Adding within-hypothesis per-step
   adversarial choice compounds this; agents may settle on safe-but-bad
   actions and fail to learn.
6. **Specifying a hypothesis is harder.** "Write down an MDP" is a thing
   every researcher does. "Write down an RMDP" requires committing to a
   specific representation of the imprecise transition (interval, halfspace,
   ball, ...).
7. **For all current target environments** in the project (Bernoulli
   bandits, Newcomb, Death in Damascus, multi-arm versions), point-valued
   T with credal mixing is sufficient. Set-valued kernels are unused
   capacity.
8. **No published literature on supra-POMDP-with-explicit-states yet** in
   the IB sequence — the Belief Functions post defers it. We'd be
   designing the data structure ourselves, with no reference implementation.

### Decision-relevant facts

- **Across-hypothesis pessimism (the IB `min_k`) is preserved either way.**
  We do not lose worst-case reasoning over hypotheses by going point-valued;
  we lose worst-case reasoning over *transitions within a hypothesis*.
- **The plan-file Phase 3 sketch is already point-valued** and the
  `WorldModel` interface was designed to keep `params` opaque. Choosing
  point-valued now is the smallest change; choosing set-valued is a bigger
  redesign.
- **Migration path is contained.** Even if we choose point-valued and
  later regret it, swapping in a set-valued `SupraPOMDPWorldModel` does
  not touch `Infradistribution`, `AMeasure`, or the agent — `params` is
  opaque. The cost of a future migration is bounded to one class.
- **The three sources we read do not converge on a recommendation.**
  Appel-Kosoy argue strongly for set-valued (their entire contribution).
  Bell et al. demonstrate point-valued is sufficient for Newcomb-style
  problems. Belief Functions defers the question. The choice is
  project-dependent, not theory-mandated.

---

## 10. Concrete environment table — what each choice can represent

Read each row as: "to represent this environment as one IB hypothesis (or
as a credal mixture of hypotheses), is point-valued T enough or do we need
set-valued T?"

| Environment | Point-valued T (single hyp.) | Point-valued T (mixture) | Set-valued T (single hyp.) |
|---|---|---|---|
| Standard MAB (single Bernoulli arm) | Yes | Yes (trivial) | Yes |
| Multi-arm Bernoulli (unknown probabilities) | No (uncertainty) | Yes (mixture over `p` grids) | Yes (interval over `p`) |
| Newcomb's problem (perfect predictor) | Yes (one POMDP per predictor strategy) | Yes (mixture over predictors) | Yes |
| Death in Damascus | Yes (single POMDP) | Yes | Yes |
| Counterfactual Mugging | Yes | Yes | Yes |
| Transparent Newcomb (with ε error) — pseudocausal | Yes | Yes | Yes |
| Transparent Newcomb (perfect, no ε) — acausal | No | No (mixture is causal) | Possibly (with surcausal/Nirvana extensions) |
| Standard tabular POMDP | Yes (the classic case) | Yes (mixture if unknown) | Yes |
| Robust tabular RL with halfspace constraints (Appel-Kosoy 1-bounded RMDP) | No | Approximate (discretise halfspace, exponential) | Yes (compact) |
| Non-oblivious adversary within an episode | No | No | Yes |
| Continuous-state robust RL (general) | No | No | Out of scope for both — Phase 1 plan §"Discrete outcome assumption" |

**Reading**: The only rows where set-valued is *strictly* necessary are
perfect Transparent Newcomb (which we don't currently target), halfspace
RMDPs (Appel-Kosoy territory; tracked separately in
`20260404_rue_e2d_plan.md`), and non-oblivious within-episode adversaries
(out of current scope).

---

## 11. Existing project documents and what they cover

These are linked here so a future agent does not have to rediscover them.

- **`20260418_beliefs_refactor_plan.md`** — the Phase 1+2 plan that
  Phases 1–2 of the current branch implement. Contains the Phase 3 sketch
  (point-valued, ~25 lines) that this document is researching.

- **`20260328_infrabayes.md`** — implementation reference for the original
  IB agent (pre-refactor). Useful for context on `BernoulliBelief` and the
  KU machinery that Phase 1 replaced.

- **`20260328_infrabayes_issues.md`** — eleven known issues with the
  original IB implementation. Most relevant to Phase 3:
  - §1 Cohomogeneity numerical blowup (will worsen with more imprecision).
  - §2 Fixed vertex set after conditioning (the credal set's extreme
    points can change after conditioning; we don't track this — directly
    relevant to set-valued T's update rule).
  - §5 Per-step worst-case vs. consistent adversary (excess pessimism;
    set-valued T compounds this).
  - §6 Belief dilation (credal sets can widen after observation).

- **`20260404_appel_kosoy_comparison.md`** — eight-section comparison of
  Appel-Kosoy 2025 vs. our codebase. Concludes their approach is more
  general (set-valued models, principled exploration via DEC, regret
  bounds) but is a *replacement* for our IB stack rather than an extension.

- **`20260404_rue_e2d_plan.md`** — separate plan for implementing E2D+RUE
  in our codebase as an alternative agent type. *Not* about Phase 3; if we
  choose set-valued T for Phase 3 there is potential overlap with this
  plan that should be reconciled before either is implemented.

- **`20260407_stack_ordering_analysis.md`** — orthogonal to this question
  (analysis of stack-ordering issues in the current `Infradistribution`).
  Listed for completeness.

---

## 12. Open questions to resolve before drafting Phase 3 plan

1. **Point-valued or set-valued T?** (the subject of this document)
2. **Does T depend on π?** Bell et al.'s NDPs need `T(s, a, π)`. If
   point-valued, do we thread π into T's signature or factor
   policy-dependence into a separate "policy-aware observation map"?
3. **Stationary or time-indexed kernels?** Appel-Kosoy use
   `[H] × S × A → □(...)` — explicitly non-stationary, finite horizon.
   Do we adopt the same, or assume stationary T?
4. **Initial state: point-valued Θ₀ or infradistribution?** Symmetric to
   the T question. Cheaper to commit consistently.
5. **Observation model B: deterministic, stochastic point, or
   set-valued?** Same axis again. The Phase 3 sketch has B stochastic
   point (`shape (num_states, num_obs)`).
6. **How does `mix_params` compose two `(T, B, Θ₀)` triples?** For
   point-valued, each hypothesis stays distinct (mixture-of-POMDPs); the
   `params` is naturally a list of triples. For set-valued, we need to
   define infrakernel mixing.
7. **One-step lookahead vs. value iteration in
   `compute_expected_reward`?** The current code computes one-step
   expected reward. For supra-POMDPs with horizon > 1, we may need to
   plan over a horizon — possibly via Bellman backups inside
   `compute_expected_reward`.
8. **Belief-state representation:** discrete distribution vector over
   `|S|`, or a particle/mixture representation if `|S|` is large or
   continuous?

---

## 13. Recommendation (provisional, decision still open)

**Provisional lean: point-valued T**, on these grounds:

- Smallest change from the existing Phase 1+2 architecture and the
  plan-file sketch.
- Sufficient for every environment currently on the project roadmap.
- Across-hypothesis pessimism via `Infradistribution.mix` covers most of
  what within-hypothesis Knightian uncertainty would buy.
- Migration to set-valued is contained to one `WorldModel` subclass if
  ever needed, since `params` is opaque.
- Avoids known numerical issues (`infrabayes_issues.md` §1, §5, §6) that
  set-valued kernels would compound.

**Reasons to choose set-valued instead**:

- If the project plans to implement Appel-Kosoy regret bounds in the
  near term, set-valued is required and it's better to commit once.
- If perfect Transparent Newcomb or non-oblivious within-episode
  adversaries are in scope.
- If "robust to misspecification of T" within a single hypothesis is a
  near-term priority.

The recommendation is provisional. The user has not yet decided. The
intent of this document is to support that decision.

---

## 14. Summary table for the impatient

| Question | Point-valued T | Set-valued T |
|---|---|---|
| Code complexity | Low (numpy arrays) | High (infrakernel data structure) |
| Hypothesis specifiability | "It's a POMDP" | "It's an RMDP — pick a representation" |
| Belief state | Distribution over S | Credal set over S |
| Existing target problems covered | All of them | All of them |
| Appel-Kosoy regret bounds | No | Yes (with their machinery) |
| Acausal hypotheses (perfect Transparent Newcomb) | No | Possibly |
| Per-step adversarial transitions | No | Yes |
| Halfspace / interval robust transitions | Discretise | Compact |
| Composes with `Infradistribution.mix` | Concatenation | Convex set mixing |
| Numerical issues (#1, #5, #6 in issues doc) | Same as today | Likely worse |
| Migration cost if we switch later | Low (params is opaque) | n/a |
| Sketch in `20260418_beliefs_refactor_plan.md` | Already drafted | Would need redesign |
