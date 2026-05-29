# Code review — `ib_bandit_1.ipynb`

Status: all items resolved in the current notebook. Kept as a record of what
was changed and why.

## Resolved

1. **Greedy DP / Local search baselines (§9).** Removed (and the dependent §3
   markdown). The comparison is intentionally just **Bayesian (50/50 prior)
   vs IB (Knightian) under Omega**.

2. **§7 Policy Inspection — completely redesigned.** Was an unreadable table
   of 9 states with most rows marked `(unreachable)` and notes describing
   behaviour at unvisited states. Now:
   - A small markdown table summarising the headline difference (Bayesian
     plays Arm 0 first, IB plays Arm 1; both follow stay-on-success / switch-
     on-failure thereafter, because the bimodal prior makes one observation
     almost decisive).
   - The cell prints **only the IB hedges** — i.e. the 5 states (out of 25
     reachable) where IB does *not* play the mirror-of-Bayesian action. By
     symmetry, plain mirroring would only buy `min = 2.78`; these 5 deeper
     hedges are what lifts IB to 2.85.

3. **`extract_policy` revisit handling.** Replaced the silent `if state in
   policy: continue` with the underlying invariant: under any deterministic
   policy from the root, the success and failure subtrees of every state are
   disjoint (success bumps `s_a`, failure does not), so the reachable graph
   is a tree and each state is visited exactly once. The check is therefore
   dead code and has been removed; the docstring explains why it is safe.

4. **Plots — single 2-panel figure with tight axes and clear colours.** The
   per-step panel (data range was ~0.5–0.65 plotted on a 0.3–1.0 axis) and
   the standalone cumulative plot (lines bunched and overlapping) have been
   replaced by a single figure:
   - **Left:** Pareto frontier with `v_r = v_b` diagonal; Bayesian (red
     square) and IB (blue circle) marked with consistent colours; equal
     aspect ratio; tight bounds derived from the data.
   - **Right:** Cumulative expected reward, three lines (grey for Bayes-
     benign, red for Bayes-adversarial, blue for IB-adversarial), distinct
     markers (○/■/▲), tight `ylim` based on data range with an 8% margin.
   - Safe-arm baseline (constant 0.5) removed — it overlapped the data and
     added clutter.

5. **`pareto_frontier` tolerance coherence.** Dedup precision derived from
   `eps` (`decimals = -log10(eps)`) so dedup matches dominance comparison.

6. **"Perfectly unexploitable" diagonal.** Softened in §6 markdown and plot
   legend — the deterministic-policy frontier need not touch the diagonal
   exactly; in this setup the IB optimum sits just above it.

7. **Hard-coded 50/50 casino prior.** Promoted to top-level `P_RED = 0.5`,
   threaded through `posterior_red` so prior-ablation experiments don't
   silently lie.

8. **`evaluate` docstring.** Now explicitly notes that the inner state
   distribution is the *outcome* marginal under the deterministic policy
   and the true reward probabilities — not the agent's posterior.

9. **"Omega will always..." (intro).** Reworded as a property of
   deterministic Bayesian tie-breaking, not of Bayesianism in general.

10. **§3 Feasibility claim.** Now says cross-product is
    `O(|F_succ| × |F_fail|)` per action per state and that scaling beyond
    `T=5` is not characterised.

11. **Timing measurement.** `timeit.repeat` (best of 5) instead of a single
    `time.time()`.

12. **§3 motivation.** Added a "why a scalar value function is not enough"
    paragraph at the top, so the jump from Bayesian DP (§2) to Pareto DP
    (§3) is obvious to a first-time reader.

13. **§8 paper alignment.** Now explicitly maps the implementation to the
    paper:
    - Infradistribution `Ψ = {m_Red, m_Blue}` (simplest non-trivial case)
    - Lower expectation = `min` over the two evaluators
    - Pareto frontier ≡ paper §2.5 finite extremal representation
    - Omega ≡ constructive instantiation of the worst-case infimum
    - Bayes recovery (paper §2.6): `P_RED = 1` collapses Ψ and reduces IB to
      Bayes
    - Notes scope: does not implement IB-style updates of `a`-measures or
      signed `a`-measures.

14. **`r_1^r, r_2^r, …` symbols (§3).** Inline legend added.

15. **Duplicate `from collections import defaultdict`** removed.

16. **"IB recovers 56% of gap"** removed. Output reports two clearly defined
    numbers instead.

## Headline numbers (unchanged)

|                           | V_Red  | V_Blue | min    |
|---------------------------|--------|--------|--------|
| Bayesian                  | 2.7800 | 3.0200 | 2.7800 |
| IB (Pareto-DP)            | 2.8584 | 2.8472 | 2.8472 |

## Alignment with the paper's premises

The notebook is consistent with the paper at the proof-of-concept level:
- `min(V^Red, V^Blue)` is the lower expectation under the two-element
  infradistribution `Ψ = {m_Red, m_Blue}`, matching the paper's §2 IB
  framework.
- The Pareto frontier is the paper's "finite set of non-dominated minimal
  points" (§2.5), specialised to two evaluators (so the frontier lives in
  ℝ²).
- The "Omega adversary" framing realises the worst-case quantifier — the
  paper's `inf` over admissible evaluators in §2.5.
- The notebook does **not** implement the §2.7 IB update rule for
  `a`-measures (offset/`b`-term carry-over). The §3.1 paper experiment
  evaluates a single-shot worst-case bandit; this notebook adds a sequential
  exploration twist (informative bimodal prior, T=5) but keeps the
  evaluator set fixed across steps. This is flagged in §8 "Scope and
  limitations".
