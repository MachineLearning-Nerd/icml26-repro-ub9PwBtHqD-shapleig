# Claims 2–3 — exact evaluation scope

## Verdicts: Claim 2 BLOCKED · Claim 3 BLOCKED

The source statement is exactly 15 tasks across feature importance, data
valuation, hyperparameter importance, and local explanation, with budgets up
to 512.

| Exact route | Tasks | Result |
|---|---:|---|
| public precomputed games | 6 | all 30 files present per task |
| executable tree-game constructors | 3 | CorrGroups60, NHANES, Crime pass |
| TabPFN feature-importance tasks | 3 | blocked |
| YAHPO hyperparameter-importance tasks | 3 | blocked |

The three tree constructors cover `p=60`, `p=79`, and `p=101`. The six blocked
tasks require author-only precomputations/runtime or manually provisioned
surrogate metadata. Deleting one required public manifest entry makes the
independent scope verifier fail.

The preserved performance evidence uses all 30 official ViT-9 games and finds
lower paired geometric-mean MSE than KernelSHAP, LeverageSHAP, Permutation
Sampling, and Regression MSR, with every paired Wilcoxon `p<5e-5`. It covers
one task and budgets 16–64, so it is not promoted to the 15-task claim.

Evidence: `evidence/claim_2/` and `evidence/claim_3/`.
