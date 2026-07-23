# Claim 1 method

The efficient checker reimplements the paper's Theorems B.1 and B.2 using
elementary-symmetric-polynomial coefficient tables. It computes
`A K(Z,W)` and `A K(Z,Z) A^T` without constructing the coalition grid `Z`.
The posterior covariance and one-candidate EIG then use dense matrices whose
dimensions are only `p`, `t`, and `|W|`.

The independent checker deliberately follows the naive route: it constructs all
`2^p` binary coalitions, the full Hamming kernel, the explicit Shapley matrix,
and the posterior covariance. This route is restricted to small `p` by design.
Agreement tests cover `p=3..9`, three deterministic seeds, quasi-noiseless
variance `1e-6`, ARD lengthscales, and non-unit output scale.

The performance sweep extends the efficient route through `p=101`, matching the
largest game size in the paper. Wall time and modeled live memory are recorded,
but the asymptotic decision uses executable scalar-operation counts from the
actual coefficient-table updates. The candidate-set term from Appendix B.4 is
kept explicit in the claim contract and is not silently collapsed into the
single-candidate theorem.

The negative control multiplies one efficient result by `1.001`; the same
`1e-7` identity threshold must reject it.
