# 2026-07-23 claim-by-claim reproduction campaign

This additive campaign supersedes the aggregate verdict language on the
historical pages while preserving every judged page at revision
`85ca787e52cd4ba933883116d010d919bfe54fe7`.

| Claim | Exact result | Central evidence |
|---|---|---|
| 1 — closed-form complexity | **VERIFIED** | 21 explicit checks; max error `4.54e-10`; quartic operation slope `3.970`; `p=101` completes |
| 2 — 15-task evaluation scope | **BLOCKED** | 9/15 exact tasks runnable; six TabPFN/YAHPO tasks unavailable |
| 3 — Figure 1 performance ordering | **FALSIFIED** | Regression MSR wins at budget 16 on all three exact public data-valuation tasks; all survive correction over 60 comparisons |
| 4 — GP acquisition ablation | **FALSIFIED** | GP+Uncertainty has lower mean MSE at budgets 16, 24, 32, and 48 |
| 5 — per-iteration overhead | **BLOCKED** | exact timed code found; author runtime and manual data unavailable |

Every formal node used exactly:

```text
uv sync --frozen && uv run python repro/src/reproduce.py
```

The exact Claim 3 gate ran on local Apple-arm64 CPU, 8 logical CPUs, Python
3.12.11, no GPU, 990.35 seconds, and $0 external compute cost. All 32 tests,
independent checkers, negative controls, the judged one-task result, and
author-code parity passed.

No score increase is claimed. Only a new live judge verdict can change the
judged score.
