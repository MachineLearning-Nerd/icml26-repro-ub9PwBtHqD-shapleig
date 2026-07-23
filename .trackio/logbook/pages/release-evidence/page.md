# Release evidence, provenance, and limitations

## Fixed contract

| Item | Value |
|---|---|
| Baseline Git SHA | `2789e9462464004ea99b63c6afe938b82215ebe2` |
| Cumulative evidence SHA | `05258258c9e7da4b7e4eac0699c241b0060fd5fa` |
| Formal command | `uv sync --frozen && uv run python repro/src/reproduce.py` |
| Environment | Python 3.12.11, locked `uv`, one repository `.venv` |
| Compute | local CPU, Apple arm64, 8 logical CPUs, no GPU |
| Cumulative wall time | 365.15 seconds |
| External compute cost | $0 |
| Paper HTML SHA-256 | `81809d39fa180bab590aa3eb2c3a0c37ac3d17df20c91ac47dc4f805c981dff1` |
| Judged Space revision | `85ca787e52cd4ba933883116d010d919bfe54fe7` |

## Release gates

- All five claims use exactly VERIFIED, FALSIFIED, or BLOCKED.
- Claim 4's previous full-credit falsification passes again.
- Claim 1's formula, explicit comparison, scaling, and negative control pass.
- The one-task judged benchmark and author-code parity pass.
- Claims 2–3 never promote the one-task evidence to full scale.
- Claim 5 never substitutes the one-candidate diagnostic for the paper timing.
- Every judged Space path remains in the additive candidate.

## Limitations

Six exact tasks and the author's timing runtime/data are unavailable. Adaptive
GP acquisition paths are numerically sensitive at near ties, although the
claim-level falsification is stable. No Hugging Face publication has occurred,
and no score increase is claimed pending a live judge.
