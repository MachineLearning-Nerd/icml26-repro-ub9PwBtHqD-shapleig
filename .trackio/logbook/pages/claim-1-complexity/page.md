# Claim 1 — closed-form complexity

## Verdict: VERIFIED

The paper contrasts its `O(p^4 + t^3)` closed form with an explicit
`O(4^p · t)` route. This campaign checks both correctness and scaling.

- 21 independent explicit-coalition comparisons cover `p=3…9` and three
  deterministic seeds per size.
- Worst absolute EIG error: `4.540865461422072e-10`.
- Schur-complement regression error remains below `1e-9`.
- Modeled-operation log–log slope: `3.97015143`, `R²=0.99999348`.
- Explicit-memory exponential fit: `R²=0.99939116`.
- Efficient route: `p=101,t=102` in `3.510 s`, without a `2^p` object.
- Explicit `Kzz` projection at `p=20`: `8 TiB`.

The independent checker constructs every coalition, the Shapley matrix, the
kernel, and posterior covariance. Mutating one EIG by 0.1% is detected.

Evidence: `evidence/claim_1/`.
