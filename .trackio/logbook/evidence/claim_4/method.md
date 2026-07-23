# Claim 4 method

The fixed cumulative command downloads the pinned public ViT-9 games and
evaluates the four Figure 3 GP variants at budgets 16, 24, 32, 48, and 64 for
all 30 games. The audit filters the resulting per-game MSE table to a complete
600-row ablation grid.

Two summaries are computed directly from raw MSE:

1. arithmetic-mean MSE for each method and budget; and
2. mean rank over the 150 matched game-budget cells.

A separate standard-library CSV parser reconstructs the complete key grid and
recomputes both summaries without reading the primary analysis output. The
negative control swaps the ShaplEIG and GP+Uncertainty labels; the verifier
must reject that mutated raw table against the original result.

The falsification is intentionally narrow: it contradicts the stated
within-ablation superiority on an official paper task, but it does not claim
universal dominance by GP+Uncertainty.
