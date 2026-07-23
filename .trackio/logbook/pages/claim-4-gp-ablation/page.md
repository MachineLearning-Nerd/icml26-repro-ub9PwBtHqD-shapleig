# Claim 4 — GP-surrogate ablation

## Verdict: FALSIFIED

The exact 600-row grid contains 30 official ImageNet/ViT-9 games, five budgets,
and the four Figure 3 GP variants.

| Budget | ShaplEIG mean MSE | GP+Uncertainty mean MSE | Lower |
|---:|---:|---:|---|
| 16 | 0.0131612 | 0.0076601 | GP+Uncertainty |
| 24 | 0.0061901 | 0.0054042 | GP+Uncertainty |
| 32 | 0.0037685 | 0.0032461 | GP+Uncertainty |
| 48 | 0.0023682 | 0.0019719 | GP+Uncertainty |
| 64 | 0.0010531 | 0.0010797 | ShaplEIG |

GP+Uncertainty also has the better within-ablation mean rank (`2.207` versus
`2.227`). A second standard-library CSV parser reconstructs all keys,
aggregates, and ranks. Swapping the two method labels is rejected.

This is a direct counterexample on an official paper task. It does not assert
universal dominance by GP+Uncertainty. Near-tied floating-point acquisition
scores cause small run-to-run path changes, but the same four winning budgets
appear in the judged evidence and cumulative gate.

Evidence: `evidence/claim_4/`.
