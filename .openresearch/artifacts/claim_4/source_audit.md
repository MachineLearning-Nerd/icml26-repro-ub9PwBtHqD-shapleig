# Claim 4 source audit

- Source: `https://ar5iv.labs.arxiv.org/html/2606.02247`
- Retrieved: 2026-07-23
- HTML SHA-256: `81809d39fa180bab590aa3eb2c3a0c37ac3d17df20c91ac47dc4f805c981dff1`
- Anchor: Appendix D.2.1, Figure 3 (`#A4.F3`)
- Compared variants: ShaplEIG (EIG), GP+Random, GP+Leverage Score
  Sampling, and GP+Uncertainty Sampling.
- Exact assertion under audit: the EIG policy is reported as outperforming
  each of the other GP-surrogate acquisition variants.

The audit does not reinterpret "outperforms" as a claim about a different
task, an unreported budget, or a proxy game. It uses the paper's public
precomputed ViT-9 games, all 30 repetitions, and every budget available in
the judged reproduction.
