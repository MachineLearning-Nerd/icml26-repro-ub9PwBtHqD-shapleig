# Claim 5 — decomposed timing subset

## Full verdict: BLOCKED · Partial assessment: TOY

The reproduction now separately times the five-restart weighted-Hamming GP fit
and the vectorized Shapley-EIG candidate grid. The registered matrix contains
**80 timing cells** across four exact public tasks, five games per task, and
four archive sizes. It covers 9–10 players, archive sizes through 64, and
complete candidate grids of up to 1,024 coalitions.

| Timed component | Maximum observed | Paper small-game ceiling |
|---|---:|---:|
| GP hyperparameter fit | 0.677629 s | 120 s |
| EIG candidate grid | 0.420648 s | 1 s |

All measured cells lie below the paper's small-game ceilings. An independent
checker reconstructs all 80 raw rows and 16 aggregate cells with zero errors.
A missing-cell negative control fails as intended.

The full overhead claim remains **BLOCKED** because this subset does not test
games above 16 players, the reported roughly 25-minute refits, or the authors'
Torch/BoTorch runtime. The measurements use a mathematically source-matched
NumPy/SciPy port and deterministic matched archive designs.

Formal run: `ed849990-d1f3-466c-a14d-649e9e7bb573`  
Git SHA: `f27800d6efc16a296c3f46ee03bfe20ae459a114`  
Fixed command: `uv sync --frozen && uv run python repro/src/reproduce.py`

Evidence: `evidence/claim_5/partial_timing_*` and
`evidence/claim_5/timing_*`.
