# Negative controls


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_711b4410f29d", "created_at": "2026-07-17T05:50:07+00:00", "title": "Negative controls"}
-->
- **Repeat observation is near-redundant:** the EIG of observing an *already-seen* coalition S (a 2nd noisy read) is ≥0 but **much smaller** (<1/5) than the EIG of a fresh coalition — confirming EIG correctly down-weights redundant evaluations.
- **EIG > 0 always:** any observation strictly reduces posterior entropy (consistent with non-degenerate noise), ruling out degenerate zero-information designs being selected.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3e94d88b4703", "created_at": "2026-07-18T10:37:16+00:00", "title": "Repair controls and retained failures"}
-->
## Full-scale controls

1. **Posterior proxy rejected:** conclusions use realized MSE against exhaustive ground truth, not posterior trace or determinant.
2. **Toy scope rejected:** the old p=3–5 Gaussian experiment remains historical only; C2 now rests on the official p=9 ImageNet application and 30 games.
3. **Weak baseline set rejected:** random alone is insufficient; six named SOTA baselines and three GP ablations are included.
4. **Independent EIG check:** vectorized rank-one EIG matches direct covariance log determinants within 1e-7.
5. **Upstream parity:** LeverageSHAP and RegressionMSR reproduce pinned author code exactly.
6. **Retained failure:** KernelSHAP has two genuine numerical blow-ups at budget 16. They remain in raw data and arithmetic means; paired log statistics are reported for robustness.
7. **Retained negative ablation:** adaptive GP-uncertainty has lower aggregate MSE than adaptive ShaplEIG at budgets 16, 24, 32, and 48 (ShaplEIG wins at 64). GP-LeverageSHAP is also mixed. These are not SOTA estimator baselines, and the result limits the conclusion to the paper claim actually tested: improvement over the six named estimator baselines, not universal dominance over every GP acquisition heuristic.
