# Claim 5 — per-iteration computational overhead

## Verdict: BLOCKED

The source and author code identify the exact timed blocks:

- GP hyperparameter refitting: `gp.fit()`
- EIG computation: `EIGFunctionProperty(candidate_set, ...)`

The exact timing claim cannot run under the frozen reproduction contract:

1. author Python is `>=3.11,<3.12`; the frozen environment is `3.12.11`;
2. `torch`, `gpytorch`, `botorch`, `hydra`, `yahpo_gym`, and `tabpfn` are
   absent; and
3. manual YAHPO metadata and TabPFN task directories are absent.

A diagnostic single-candidate NumPy calculation at `p=101` takes 3.510
seconds. It is not the paper's vectorized 1,024-candidate per-iteration
measurement and is not promoted to claim evidence. A negative control that
marks all prerequisites present removes the blocker and is rejected.

Evidence: `evidence/claim_5/`.
