# Claim 3 — exact data-valuation counterexamples

## Verdict: FALSIFIED under the judge's broad Figure 1 claim contract

All 30 pinned games for each of the three exact 10-player data-valuation tasks
were evaluated with ShaplEIG and the four named baselines at budgets 16, 24,
32, 48, and 64. The registered search contains 2,250 raw MSE observations and
60 paired task/budget/baseline comparisons.

At budget 16, Regression MSR beats ShaplEIG on every task:

| Exact task | ShaplEIG mean MSE | Regression MSR mean MSE | Paired geometric ratio (95% bootstrap CI) | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Bike Sharing / RF | 0.010953 | 0.005481 | 1.837 [1.489, 2.257] | 0.000471 |
| Bike Sharing / GB | 0.011583 | 0.006439 | 1.655 [1.384, 1.978] | 0.000603 |
| California Housing / GB | 0.009047 | 0.004335 | 1.967 [1.623, 2.359] | 0.0000358 |

The author configuration is preserved: leverage initial design of `p+1`,
weighted-Hamming GP, five MAP restarts, refitting every iteration, exhaustive
candidate acquisition, and seeds 1–30. Realized error is measured against the
exhaustive 1,024-coalition Shapley value. A separate process reconstructs all
60 tests from raw CSV; deleting one matched cell is rejected.

Source nuance: Section 5.2 permits exceptions over very short intervals, and
budget 16 is the earliest tested point. This evidence falsifies the judge's
broad cross-budget wording, but it does not claim a complete 15-task rerun or
contradict every narrower prose sentence.

Evidence: `evidence/claim_3/`.
