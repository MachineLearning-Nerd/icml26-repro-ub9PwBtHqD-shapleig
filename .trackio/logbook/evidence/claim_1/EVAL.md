# Claim 1 evaluation

Verdict: VERIFIED

- Efficient and direct EIG agree over 21 cases at p=3..9;
  maximum absolute error `4.541e-10`.
- The ESP implementation completed p=101 without constructing a 2^p grid.
- Executable operation-count slope: `3.9702` (R² `1.0000`).
- Empirical runtime slope: `2.9950` (R² `0.9919`);
  runtime is diagnostic, while the executable loop count tests the asymptotic bound.
- Explicit-route memory grows exponentially: log(bytes) versus p R²
  `0.9994`; a dense p=20 Kzz matrix alone is
  `8,796,093,022,208` bytes.
- The 0.1% corruption negative control was detected.

Limitations: timings are hardware-specific and do not prove a Big-O statement
alone. The verdict rests on identity with an independent explicit computation,
the absence of any 2^p object in the inspected route, executable loop counts,
and successful execution at the paper's maximum p=101.
