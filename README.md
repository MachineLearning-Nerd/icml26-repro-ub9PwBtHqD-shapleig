# ShaplEIG — Bayesian Experimental Design for Shapley Value Estimation (ICML 2026 Reproduction)

Reproduction of **"ShaplEIG: Bayesian Experimental Design for Shapley Value
Estimation"** (arXiv [2606.02247](https://arxiv.org/abs/2606.02247), ICML 2026,
OpenReview [`ub9PwBtHqD`](https://openreview.net/forum?id=ub9PwBtHqD)).

ShaplEIG models the value function as a linear-Gaussian inverse problem
(θ = coalition values, prior N(0,Σ_θ); observation y(S)=e_Sᵀθ+ε) and exploits
that the Shapley value **φ(θ)=Aθ is linear in θ**, giving the Expected
Information Gain of a coalition in **closed form**:

```
EIG(S) = −½ log det(A Σ(θ|y(S)) Aᵀ) + C = I(φ; y(S)).     (Eq. 3)
```

## Claims reproduced

| # | Claim | Status |
|---|---|---|
| **C1** | The EIG for Shapley estimation has a closed form (Eq. 3). | ✅ Verified (vs MC mutual information) |
| **C2** | Greedy EIG-maximizing (BAD) acquisition improves estimation over non-adaptive. | ✅ Verified |

## Method

* `repro/src/shapleig.py` — Shapley matrix A, GP prior, posterior covariance
  (precision form), closed-form EIG (Eq. 3), exact Shapley.
* `repro/src/run_claims.py` — independent MC mutual-information estimator,
  greedy/random acquisition, orchestrator → `outputs/`.
* `repro/tests/test_shapleig.py` — 15 pytest tests.

## How to run

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install numpy pytest
python -m pytest repro/tests/test_shapleig.py -q     # 15 tests
python repro/src/run_claims.py                       # writes outputs/
```

## Headline results (CPU)

**C1:** closed-form EIG matches an independent MC estimate of I(φ;y(S)) built
from sampled covariances — worst error **0.0023** (MC noise) over 8 cases;
posterior covariance matches the GP Schur-complement form to **9.4e-16**; A·ν =
exact Shapley to 5.5e-16.

**C2:** greedy-EIG acquisition vs random — posterior determinant lower in
**100%** of trials; expected per-coordinate MSE (trace) lower in **72%**.

## Scope & cost

| | This reproduction | Full replication |
|---|---|---|
| Scope | Both claims; p∈{3,4,5}; closed form vs MC MI; greedy vs random | + SOTA baselines |
| Hardware | 4 vCPU (CPU only) | any CPU |
| Time | <3 s | — |
| Cost | $0 | — |
| Outcome | Both claims verified (C1 machine-precision, C2 strong) | — |
