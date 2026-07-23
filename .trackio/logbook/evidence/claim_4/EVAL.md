# Claim 4 evaluation

Verdict: FALSIFIED

The exact official ViT-9 ablation grid contains 30 games, five budgets, and
the four Figure 3 GP variants. GP+Uncertainty has a better within-ablation
mean rank than ShaplEIG and lower arithmetic-mean MSE at budgets 16, 24, 32,
and 48. ShaplEIG is lower only at budget 64. A separate CSV parser recomputes
the full grid, aggregates, and ranks. Swapping the two method labels makes
the evidence inconsistent and is rejected.

This is a direct counterexample on one of the paper's own Figure 3 tasks. It
does not assert that GP+Uncertainty dominates on every task or metric.
