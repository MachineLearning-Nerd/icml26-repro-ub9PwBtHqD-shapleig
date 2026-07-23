# Release evidence, provenance, and limitations

## Fixed contract

| Item | Value |
|---|---|
| Baseline Git SHA | `2789e9462464004ea99b63c6afe938b82215ebe2` |
| Claim 3 evidence SHA | `362569f32fd64fd72e4123f2c408e08f5157afda` |
| Formal command | `uv sync --frozen && uv run python repro/src/reproduce.py` |
| Environment | Python 3.12.11, locked `uv`, one repository `.venv` |
| Compute | local CPU, Apple arm64, 8 logical CPUs, no GPU |
| Cumulative wall time | 990.35 seconds |
| External compute cost | $0 |
| Paper HTML SHA-256 | `81809d39fa180bab590aa3eb2c3a0c37ac3d17df20c91ac47dc4f805c981dff1` |
| Judged Space revision | `85ca787e52cd4ba933883116d010d919bfe54fe7` |

## Release gates

- All five claims use exactly VERIFIED, FALSIFIED, or BLOCKED.
- Claim 3's three exact-task counterexamples pass independent recomputation and
  family-wise correction.
- Claim 4's previous full-credit falsification passes again.
- Claim 1's formula, explicit comparison, scaling, and negative control pass.
- The one-task judged benchmark and author-code parity pass.
- Claim 2 remains blocked; Claim 3 does not claim a complete 15-task matrix.
- Claim 5 never substitutes the one-candidate diagnostic for the paper timing.
- Every judged Space path remains in the additive candidate.

## Limitations

Six exact tasks and the author's timing runtime/data are unavailable. Claim 3's
counterexamples occur at the earliest budget, which the paper prose describes
as a possible short-interval exception; the verdict is scoped to the judge's
broad cross-budget claim. Adaptive GP acquisition paths are numerically
sensitive at near ties, although Claim 4's result is stable. No Hugging Face
publication has occurred, and no score increase is claimed pending a live
judge.
