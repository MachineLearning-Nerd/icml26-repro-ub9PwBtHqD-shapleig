# Claim 5 evaluation

Verdict: BLOCKED

The exact paper measurement cannot be run under the frozen reproduction
environment. The authors time `gp.fit()` and their vectorized
`EIGFunctionProperty` implementation with a Python 3.11-only Poetry lock,
BoTorch, GPyTorch, Torch, Hydra, and manually provisioned YAHPO/TabPFN data.
The fixed baseline is Python 3.12.11 and contains none of those runtime modules
or manual datasets.

The polynomial Claim 1 timing is deliberately not counted: it measures one
candidate and does not reproduce the paper's vectorized candidate-set call or
its GP fitting block. Changing the lock now would violate the campaign's fixed
environment contract. This is a reproducible blocker, not evidence for or
against the reported 30-second and 25-minute measurements.
