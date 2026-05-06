"""Generate a LaTeX table for canonical trap-bandit results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROWS = [
    ("0.99", "n/a", r"infra\_bayesian", "correct", "ib"),
    ("0.99", "n/a", r"bayes\_ucb", "correct", "bayes_ucb"),
    ("0.99", "0.99", r"bayes\_greedy", "correct", "bayes_greedy"),
    ("0.99", "0.5", r"bayes\_greedy", "misspecified", "bayes_greedy"),
    (
        "0.99",
        "0.01",
        r"bayes\_greedy",
        "severely_misspecified",
        "bayes_greedy",
    ),
    ("0.99", "0.99", r"bayes\_thompson", "correct", "bayes_thompson"),
    ("0.99", "0.5", r"bayes\_thompson", "misspecified", "bayes_thompson"),
    (
        "0.99",
        "0.01",
        r"bayes\_thompson",
        "severely_misspecified",
        "bayes_thompson",
    ),
    ("0.01", "n/a", r"infra\_bayesian", "mostly_safe_correct", "ib"),
    ("0.01", "n/a", r"bayes\_ucb", "mostly_safe_correct", "bayes_ucb"),
    ("0.01", "0.01", r"bayes\_greedy", "mostly_safe_correct", "bayes_greedy"),
    (
        "0.01",
        "0.01",
        r"bayes\_thompson",
        "mostly_safe_correct",
        "bayes_thompson",
    ),
]


def ci_cell(point: float, ci: list[float]) -> str:
    lo, hi = ci
    return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"


def build_table(results_dir: Path) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lll rcc}",
        r"\toprule",
        r"DGP $\alpha$ & Bayesian prior & Agent & Cat. rate & p50, 95\% CI & p95, 95\% CI \\",
        r"\midrule",
    ]

    for dgp_alpha, bayes_prior, agent_label, condition, agent in ROWS:
        summary = json.loads((results_dir / f"{condition}_summary.json").read_text())
        bootstrap = json.loads(
            (results_dir / f"{condition}_bootstrap_summary.json").read_text()
        )
        cat = summary[agent]["catastrophe_rate"]
        points = bootstrap[agent]["point"]
        cis = bootstrap[agent]["ci"]
        p50 = ci_cell(points[1], cis[1])
        p95 = ci_cell(points[2], cis[2])
        lines.append(
            f"{dgp_alpha} & {bayes_prior} & {agent_label} & {cat:.3f} & {p50} & {p95} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Final cumulative expected-regret percentiles with "
            r"bootstrap confidence intervals.}"
        ),
        r"\label{tab:trap-bandit-results}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/alaro/trap_bandit/results_separated_arms_200_pcat001"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/alaro/trap_bandit/results.tex"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_table(args.results_dir)
    args.output.write_text(table)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
