# Current Claim 1 symbolic complexity certificate

This certificate audits the implementation in
[`complexity.py`](../source/repro/src/complexity.py) against Theorem 3.1 and
Appendix B.4 of arXiv:2606.02247v1. It is the basis for the asymptotic verdict;
the finite timing sweep is corroboration only.

## Assumptions

Let `p >= 1` be the number of binary players, `t >= 1` the number of observed
coalitions, and `W` a candidate set. The argument applies to the paper's
product weighted-Hamming kernel, linear Shapley operator, Gaussian-process
posterior, and additive Gaussian observation noise. It does not cover an
arbitrary kernel or a non-Gaussian surrogate.

## Independently reconstructed bound

`akazza_esp` computes `A K(Z,Z) Aᵀ`. Its outer `i` loop has `p` iterations.
For each `i`, its suffix construction and prefix updates each make `O(p)`
updates of arrays with `(p+1)^2 * 4 = O(p²)` entries. The contractions for at
most `p` values of `j` also touch `O(p²)` entries. Therefore this block costs
`O(p * p * p²) = O(p⁴)` time and `O(p³)` workspace.

For `n=p-1`, the reciprocal-binomial identity

`1 / C(n,k) = (n+1) ∫₀¹ x^k (1-x)^(n-k) dx`

turns each leave-one-factor-out Shapley sum into the integral of
`∏[gamma_r(1-x)+delta_r x]`, a degree-`p-1` polynomial. `akz_esp` evaluates
this integral with `ceil(p/2)`-node Gauss–Legendre quadrature, which is exact
for degree at most `2*ceil(p/2)-1`. At each node it forms all `p` factors once
and obtains every excluded product by division by its positive factor. The
work is `O(p²)` per input row, so `A K(Z,X)` costs `O(t p²)` and
`A K(Z,W)` costs `O(|W| p²)`. The quadrature result is tested against explicit
coalition enumeration for `p={3,4,5,8}`.

The remaining per-candidate operations are a `t x t` kernel and Cholesky
factorization, triangular solves, and products with `p` Shapley dimensions:
`O(t³ + p t² + p²t)`. For positive `p,t`,

- if `t <= p²`, then `t p² <= p⁴`; otherwise `t p² < t² <= t³`;
- if `t >= p`, then `p t² <= t³`; otherwise `p t² < p³ <= p⁴`;
- `p²t` is covered by the first case split.

Consequently one candidate costs `O(p⁴ + t³)`. Reusing the common
`A K(Z,Z) Aᵀ`, observation kernel, and factorization for a set `W` adds the
paper's `O(|W| t²)` candidate solves; the lower-order
`O(|W|(p²+pt))` construction terms are bounded by
`O(p⁴ + t³ + |W|t²)` under the same positive-integer domain.

The explicit checker constructs `2^p` coalitions and a dense `2^p x 2^p`
posterior covariance, giving `4^p` stored entries before the additional
observation-dependent work. This independently confirms the exponential
object used by the naive route; no exponential object is allocated by the
efficient implementation.

## Quantifier calibration

The source-level loop bound covers every positive integer `p,t` under the
listed model assumptions. The `p=3..9` equality checks and the `p=101`
execution are finite corroboration, not a proof of the universal quantifier.
