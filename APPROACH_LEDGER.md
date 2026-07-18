# C2 approach ledger — real costly application

Target: replace the toy-only comparison with realized Shapley-value accuracy on
the paper's ImageNet/ViT local-explanation application and competitive methods.

1. Pinned-data integrity: verify all 30 files against SHA-256 and require all
   512 coalitions. Acceptance: 30/30 pass.
2. Exact target: compute ground-truth SVs from exhaustive coalition values.
   Acceptance: linear Shapley map agrees with efficiency.
3. ShaplEIG adaptive ARD: MAP-fit nine Hamming lengthscales and exhaustively
   maximize closed-form EIG. Primary method.
4. ShaplEIG fixed isotropic: same EIG with a fixed prior-mode lengthscale.
   Ablates outcome-adaptive hyperparameters.
5. Independent EIG route: compare the rank-one/vectorized score with direct
   posterior log determinants. Acceptance: absolute error below `1e-7`.
6. KernelSHAP: state-of-the-art weighted-regression baseline from shapiq 1.4.1.
7. LeverageSHAP: authors' paired leverage design and PolySHAP regression.
8. UnbiasedKernelSHAP: state-of-the-art unbiased regression baseline.
9. RegressionMSR: the authors' XGBoost/TreeSHAP plus unbiased-residual baseline.
10. SVARM: stratified Shapley baseline from shapiq 1.4.1.
11. Permutation sampling: independent marginal-contribution implementation.
12. GP-random: same adaptive Hamming GP, uniformly random acquisition.
13. GP-uncertainty: same GP, maximum predictive-variance acquisition.
14. GP-LeverageSHAP: same GP, nonadaptive leverage acquisition.

The primary evidence is realized MSE across the 30 official games and matched
budgets 16, 24, 32, 48, and 64. Posterior trace/determinant is diagnostic only.
The two locally implemented author baselines are also compared directly with
the pinned upstream source at budgets 16, 32, and 64; acceptance is maximum
absolute difference below `1e-12`.

## Final disposition

| Route | Disposition | Evidence |
|---:|---|---|
| 1 | PASS | 30/30 SHA-256 checks; 512 unique coalitions each |
| 2 | PASS | exhaustive linear Shapley target and efficiency checks |
| 3 | PASS | adaptive ShaplEIG beats all six named SOTA baselines in paired curve analysis |
| 4 | PASS | fixed-isotropic sensitivity control is strongest overall |
| 5 | PASS | vectorized/direct EIG absolute difference `<1e-7` |
| 6 | RETAINED FAILURE | KernelSHAP has two singular-regression blow-ups at budget 16 |
| 7 | PASS | exact output parity with pinned PolySHAP (`0.0` max difference) |
| 8 | PASS | unbiased KernelSHAP executed on every matched cell |
| 9 | PASS | exact output parity with pinned RegressionMSR (`0.0` max difference) |
| 10 | PASS | SVARM executed on every matched cell |
| 11 | PASS | independent permutation estimator executed on every matched cell |
| 12 | MIXED CONTROL | GP-random beats adaptive ShaplEIG at some cells but is worse on mean rank |
| 13 | RETAINED NEGATIVE | GP-uncertainty beats adaptive ShaplEIG at 4/5 aggregate budgets |
| 14 | MIXED CONTROL | GP-LeverageSHAP wins at budgets 24/32; ShaplEIG wins at 16/48/64 |

Routes 12–14 are GP acquisition ablations, not the SOTA estimator baselines in
C2. Their mixed/negative outcomes are retained to delimit the claim: this run
supports superiority over the six named estimator baselines, not universal
dominance over every possible GP acquisition heuristic.
