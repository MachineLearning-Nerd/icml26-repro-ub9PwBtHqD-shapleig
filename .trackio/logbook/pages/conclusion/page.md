# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a5bae33d8382", "created_at": "2026-07-17T05:50:07+00:00", "title": "Executive summary"}
-->
**Both claims of ShaplEIG (arXiv 2606.02247) reproduced on CPU.**

The Expected Information Gain for Shapley estimation has a **closed form** (Eq. 3), verified against an independent Monte-Carlo mutual-information estimate (worst error 0.0023) with the posterior covariance matching the GP Schur-complement form to 9e-16. Greedy **EIG-maximizing acquisition** reduces posterior Shapley uncertainty below random acquisition (determinant lower in 100% of trials, expected-MSE/trace lower in 72%).

**Verdict:** C1 ✅ · C2 ✅. 15/15 tests pass.

## Scope & cost
| | This reproduction | Full replication |
|---|---|---|
| Scope | Both claims; p∈{3,4,5}; closed form vs MC MI; greedy vs random | SOTA baselines (LeverageSHAP/RegressionMSR) additionally |
| Hardware | 4 vCPU (CPU only) | any CPU |
| Time | <3 s | — |
| Cost | \$0 | — |
| Outcome | Both claims verified (C1 machine-precision, C2 strong) | — |


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6421c0113749", "created_at": "2026-07-18T10:37:17+00:00", "title": "Executive summary — full-scale C2 repair", "pinned": true, "pinned_at": "2026-07-18T10:37:17+00:00"}
-->
## Outcome

**C1 and C2 are verified.** C1 retains independent Monte Carlo and Schur-complement checks. For C2, adaptive ShaplEIG is evaluated on all 30 official ImageNet/ViT-9 games (512 coalitions each) against six strong baselines at five budgets. Its paired geometric MSE is 42%–85% lower than every baseline; every 95% bootstrap interval excludes parity. This directly replaces the previous toy/random-only evidence.

## Scope & cost

| Item | Value |
|---|---|
| Application | ImageNet local explanation, ViT, 9 patches |
| Replicates | 30 official games |
| Methods / budgets | 11 / 5 |
| Ground truth | exhaustive 512 coalitions per game |
| Hardware | CPU only, 4 vCPU |
| Full-run wall time | 517 seconds |
| External cost | $0 |
| Scope limit | one full paper application, not every task family |
