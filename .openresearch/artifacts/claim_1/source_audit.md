# Claim 1 source audit

The retrieved paper HTML has SHA-256
`81809d39fa180bab590aa3eb2c3a0c37ac3d17df20c91ac47dc4f805c981dff1`.

- `#S3.SS4.SSS0.Px1`: the naive full-covariance route is said to be dominated
  by `O(4^p t)`.
- `#S3.Thmtheorem1`: for a candidate `z^(i) in {0,1}^p`, EIG is computable in
  `O(p^4 + t^3)`.
- `#A2.SS4`: evaluating a set `W` costs
  `O(p^4 + t^3 + |W| t^2)`.

The theorem assumes the paper's linear Shapley goal, product-form weighted
Hamming kernel, Gaussian-process posterior, and additive Gaussian observation
noise. The audit tests those assumptions directly and does not generalize the
bound to arbitrary kernels or non-Gaussian surrogates.
