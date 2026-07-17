# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a5bae33d8382", "created_at": "2026-07-17T05:50:07+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T05:50:08+00:00"}
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
