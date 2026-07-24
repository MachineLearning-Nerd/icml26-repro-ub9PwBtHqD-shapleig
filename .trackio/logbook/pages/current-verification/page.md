# CURRENT verification — evaluator entrypoint

This is the current verifier for candidate revision
`orx/evaluator-visible-c5-release-gate`. It supersedes the earlier pages in
navigation; those pages are preserved as historical judged evidence.

Paper source: arXiv:2606.02247v1, retrieved 2026-07-23, HTML SHA-256
`81809d39fa180bab590aa3eb2c3a0c37ac3d17df20c91ac47dc4f805c981dff1`.
Every experiment uses:

```text
uv sync --frozen && uv run python repro/src/reproduce.py
```

Pinned environment: [pyproject.toml](../../evidence/current/environment/pyproject.toml)
and [uv.lock](../../evidence/current/environment/uv.lock). Current executable
source and tests are [bundled here](../../evidence/current/source/repro/).

## Claim 1 — VERIFIED, medium confidence

Exact source quantifier: for any candidate `z ∈ {0,1}^p` under the paper's
weighted-Hamming GP and Gaussian-noise assumptions, Theorem 3.1 states
`O(p^4+t^3)` time; Appendix B.4 states
`O(p^4+t^3+|W|t^2)` for a candidate set, versus a naive
`O(4^p t)` route.

The current proof obligation is met by the
[source-level symbolic certificate](../../evidence/current/claim_1/symbolic_derivation.md),
not by fitting a slope to the claimed formula. It covers arbitrary positive
`p,t`: the `A K Aᵀ` loops cost `O(p^4)`, exact polynomial quadrature makes
each `A K(Z,X)` row `O(p^2)`, and the Cholesky/solve path is bounded by
`O(t^3)`. Finite corroboration reports 21 explicit-coalition comparisons at
`p=3..9` with maximum EIG error `4.540865461422072e-10`, and execution at
`p=101,t=102`. The 0.1% EIG mutation is rejected.

Assumptions and quantifiers: [source audit](../../evidence/claim_1/source_audit.md);
current contract: [JSON](../../evidence/current/claim_1/claim_contract.json); raw:
[comparison CSV](../../evidence/claim_1/independent_comparison.csv) and
[scaling CSV](../../evidence/claim_1/scaling.csv); checker:
[JSON](../../evidence/claim_1/independent_checker.json); control:
[JSON](../../evidence/claim_1/negative_control.json); verifier:
[`verify_claim1_complexity.py`](../../evidence/current/source/repro/src/verify_claim1_complexity.py).

Limit: empirical runtime is hardware-specific and is not treated as a
universal proof.

## Claim 2 — BLOCKED, high confidence in the blocker

Exact source quantifier: Table 1 and Section 5.1 evaluate **15 exact tasks,
four families, 30 or 100 repetitions, and budgets through 512**.

Observed inline: the fixed command executes **4 tasks, 2 families, 30 games per
task, and budgets 16, 24, 32, 48, 64** (600 complete task-budget-game cells).
It therefore does not verify the full scope. Removing the complete ViT-9 task
is detected. Missing feature-importance and hyperparameter-importance inputs
prevent the exact full matrix under the immutable public-data contract.

Assumptions/quantifiers: [source audit](../../evidence/claim_2/source_audit.md);
contract: [JSON](../../evidence/claim_2/claim_contract.json); raw:
[executed subset CSV](../../evidence/claim_2/executed_subset_raw.csv) and
[15-task inventory](../../evidence/claim_2/task_inventory.csv); checker:
[JSON](../../evidence/claim_2/executed_subset_independent_checker.json);
control: [JSON](../../evidence/claim_2/executed_subset_negative_control.json);
verifier: [`verify_claim2_partial_scope.py`](../../evidence/current/source/repro/src/verify_claim2_partial_scope.py).

## Claim 3 — FALSIFIED for the judge's broad Figure 1 contract

Exact tested contract: across the Figure 1 task/budget comparison, ShaplEIG
has lower MSE than Kernel SHAP, Leverage SHAP, Permutation Sampling, and
Regression MSR. A named baseline win on an exact paper task surviving the
registered paired bootstrap, Wilcoxon, and Holm tests is a counterexample.

Observed inline at budget 16 (30 paired games each):

| Exact task | ShaplEIG mean MSE | Regression MSR mean MSE | paired ratio, 95% CI | Holm p |
|---|---:|---:|---:|---:|
| Bike Sharing / RF | 0.010953 | 0.005481 | 1.837 [1.489, 2.257] | 0.000471 |
| Bike Sharing / GB | 0.011583 | 0.006439 | 1.655 [1.384, 1.978] | 0.000603 |
| California Housing / GB | 0.009047 | 0.004335 | 1.967 [1.623, 2.359] | 0.0000358 |

The deletion control rejects one missing matched cell. Source nuance:
Section 5.2 permits short-interval exceptions, so this falsifies the judge's
broad cross-budget wording and does not claim every narrower prose sentence is
false.

Assumptions/quantifiers: [source audit](../../evidence/claim_3/source_audit.md);
contract: [JSON](../../evidence/claim_3/claim_contract.json); raw:
[2,250-row CSV](../../evidence/claim_3/dv10_raw.csv); statistics:
[JSON](../../evidence/claim_3/dv10_statistics.json); checker:
[JSON](../../evidence/claim_3/independent_checker.json); control:
[JSON](../../evidence/claim_3/negative_control.json); verifier:
[`verify_claim3_dv.py`](../../evidence/current/source/repro/src/verify_claim3_dv.py).

## Claim 4 — FALSIFIED

Exact source statement: Appendix D.2.1/Figure 3 reports EIG acquisition as
outperforming GP+Random, GP+Leverage, and GP+Uncertainty on the GP-surrogate
ablation.

Observed inline on the official 30-game ViT-9 grid:

| Budget | ShaplEIG MSE | GP+Uncertainty MSE | lower |
|---:|---:|---:|---|
| 16 | 0.0131612 | 0.0076601 | GP+Uncertainty |
| 24 | 0.0061901 | 0.0054042 | GP+Uncertainty |
| 32 | 0.0037685 | 0.0032461 | GP+Uncertainty |
| 48 | 0.0023682 | 0.0019719 | GP+Uncertainty |
| 64 | 0.0010531 | 0.0010797 | ShaplEIG |

GP+Uncertainty also has the better mean rank, `2.207` versus `2.227`.
Swapping the two labels is rejected.

Assumptions/quantifiers: [source audit](../../evidence/claim_4/source_audit.md);
contract: [JSON](../../evidence/claim_4/claim_contract.json); raw:
[CSV](../../evidence/claim_4/ablation_raw.csv); checker:
[JSON](../../evidence/claim_4/independent_checker.json); control:
[JSON](../../evidence/claim_4/negative_control.json); verifier:
[`verify_claim4_ablation.py`](../../evidence/current/source/repro/src/verify_claim4_ablation.py).

## Claim 5 — BLOCKED; source-matched large subset, low confidence

Exact source quantifier: Figures 5–6 cover all 15 tasks and budgets, averaged
over repetitions with SEM. GP refits are reported up to about 1,500 seconds
for games above 16 players; vectorized EIG stays below 30 seconds.

The current large-game subset directly measures the three paper tasks with
`p=60,79,101` at initial archive size and `t=512`. Raw maxima are
`280.988607917 s` for the five-restart GP fit and `19.109690625 s` for the
vectorized 1,024-candidate EIG grid. All six EIG cells are below 30 seconds
and all fit cells below 1,500 seconds.

This is not full verification: it is 3/15 tasks, one seed, a source-matched
NumPy/SciPy port rather than the authors' Torch runtime, and the `t=512`
archive is a deterministic prefix of the author sampler rather than a full
adaptive trajectory.

Full page: [Claim 5 current evidence](../current-claim-5/page.md);
source audit: [Markdown](../../evidence/claim_5/source_audit.md); contract:
[JSON](../../evidence/claim_5/claim_contract.json); raw:
[CSV](../../evidence/current/claim_5/large_timing_raw.csv); summary:
[JSON](../../evidence/current/claim_5/large_timing_summary.json); checker:
[JSON](../../evidence/current/claim_5/large_timing_independent_checker.json);
control:
[JSON](../../evidence/current/claim_5/large_timing_negative_control.json);
verifier:
[`verify_claim5_large_timing.py`](../../evidence/current/source/repro/src/verify_claim5_large_timing.py).

## Evaluator visibility gate

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
|---|---|---|---|---|---|---|---|---|
| C1 | this page | yes | yes | yes | yes | yes | yes | VERIFIED |
| C2 | this page | yes | yes | yes | yes | yes | yes | BLOCKED |
| C3 | this page | yes | yes | yes | yes | yes | yes | FALSIFIED |
| C4 | this page | yes | yes | yes | yes | yes | yes | FALSIFIED |
| C5 | this page + current C5 | yes | yes | yes | yes | yes | yes | BLOCKED |

The machine-readable [visibility matrix](../../evidence/current/visibility_matrix.json)
and [blind-review trace](../../evidence/current/blind_review.json) are produced
by the release gate. Each current verifier is also run against a deliberately
corrupted temporary artifact; the required nonzero exits are recorded in
[negative verifier exits](../../evidence/current/negative_verifier_exits.json).

## Reproducibility and deviations

Deterministic seeds, Git SHA, CPU model/platform, logical CPU count, component
runtime, fixed command, and limitations are present in each linked result or
summary. No GPU was used. Claim 2 and Claim 5 remain `BLOCKED`; partial or
source-matched evidence is not promoted to a pass.
