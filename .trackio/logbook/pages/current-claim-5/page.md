# CURRENT Claim 5 — large-game decomposed timing

## Full verdict: BLOCKED · Partial: source-matched large subset

The paper's Figures 5–6 quantify two separately timed author blocks:
five-restart `gp.fit()` and vectorized `EIGFunctionProperty`. The exact claim
covers all 15 tasks and budgets, with repetition means and SEM. The current
run measures the same two conceptual blocks for the three paper tasks above
16 players:

| Task | p | t | GP fit (s) | EIG grid (s) |
|---|---:|---:|---:|---:|
| CorrGroups60 | 60 | 61 | 1.054006792 | 1.532135042 |
| CorrGroups60 | 60 | 512 | 145.796542459 | 2.030829375 |
| NHANES | 79 | 80 | 2.495842875 | 4.460903000 |
| NHANES | 79 | 512 | 202.075294000 | 5.228882375 |
| Crime | 101 | 102 | 5.675786542 | 11.584338084 |
| Crime | 101 | 512 | 247.824729416 | 11.320505708 |

The six-cell parser independently reconstructs the grid and thresholds. Its
output reports `verifier_exit=0`, `maximum_players=101`, and
`maximum_archive_size=512`. Deleting the `p=101,t=512` cell makes the verifier
fail with “expected 6 rows, got 5” and “grid mismatch”.

Exact command:

```text
uv sync --frozen && uv run python repro/src/reproduce.py
```

Run Git SHA: `4f40cf95412121cb07d5e4a1c30f22e5ed890978`;
Python `3.12.11`; macOS arm64; 8 logical CPUs; large timing block
`696.706 s`; deterministic paper seed 1. No GPU.

Evidence: [raw CSV](../../evidence/current/claim_5/large_timing_raw.csv),
[summary](../../evidence/current/claim_5/large_timing_summary.json),
[independent checker](../../evidence/current/claim_5/large_timing_independent_checker.json),
[negative control](../../evidence/current/claim_5/large_timing_negative_control.json),
[source audit](../../evidence/claim_5/source_audit.md),
[contract](../../evidence/claim_5/claim_contract.json), and
[executable verifier](../../evidence/current/source/repro/src/verify_claim5_large_timing.py).

Limitations: 3/15 tasks, one seed, no SEM, NumPy/SciPy rather than author
Torch/BoTorch wall-clock implementation, and a deterministic source-sampler
prefix at `t=512` rather than a complete adaptive trajectory. These
limitations are why the full verdict remains `BLOCKED`.
