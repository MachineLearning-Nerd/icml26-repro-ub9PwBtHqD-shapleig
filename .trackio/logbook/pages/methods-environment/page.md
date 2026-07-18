# Methods & environment


---
<!-- trackio-cell
{"type": "code", "id": "cell_7e25a4d4e599", "created_at": "2026-07-17T05:49:02+00:00", "title": "Pytest suite (15 tests)", "command": ["python", "-m", "pytest", "repro/tests/test_shapleig.py", "-q"], "exit_code": 0, "duration_s": 1.068}
-->
````bash
$ python -m pytest repro/tests/test_shapleig.py -q
````

exit 0 · 1.1s


````python title=test_shapleig.py
"""Formal pytest suite: ShaplEIG (arXiv 2606.02247). Run: pytest -q repro/tests"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest
from shapleig import (shapley_matrix, prior_covariance, posterior_cov,
                      eig_closed_form, true_shapley)
from run_claims import (mc_mutual_information, greedy_acquire, random_acquire,
                        shapley_posterior_mean, shapley_posterior_det,
                        shapley_posterior_trace)


# -- Shapley matrix A: A @ nu == exact Shapley --
@pytest.mark.parametrize("p", [3, 4, 5])
def test_shapley_matrix(p):
    nu = lambda m: float((m * 13 + 7) % 17)
    phiA = shapley_matrix(p) @ np.array([nu(m) for m in range(1 << p)])
    assert np.max(np.abs(phiA - true_shapley(nu, p))) < 1e-12


# -- C1: posterior covariance (precision form) == GP Schur-complement form ----
@pytest.mark.parametrize("p", [3, 4])
def test_posterior_cov_two_forms(p):
    A = shapley_matrix(p); Sigma = prior_covariance(p); obs = [1, 4, 7]
    post1 = posterior_cov(Sigma, obs, 0.3)
    H = np.zeros((len(obs), 1 << p))
    for k, s in enumerate(obs): H[k, s] = 1.0
    Kyy = H @ Sigma @ H.T + 0.3 ** 2 * np.eye(len(obs))
    post2 = Sigma - Sigma @ H.T @ np.linalg.solve(Kyy, H @ Sigma)
    assert np.max(np.abs(post1 - post2)) < 1e-9


# -- C1: closed-form EIG == independent MC mutual information -----------------
@pytest.mark.parametrize("p,cand", [(3, 0), (3, 5), (4, 3), (4, 12)])
def test_eig_matches_mc(p, cand):
    A = shapley_matrix(p); Sigma = prior_covariance(p, lengthscale=2.0)
    cf = eig_closed_form(A, Sigma, [], cand, 0.4)
    mc = mc_mutual_information(A, Sigma, cand, 0.4, n_samples=200000, seed=cand)
    assert abs(cf - mc) < 0.03


# -- C1: EIG > 0 (observing always reduces entropy) and is symmetric-aware ----
def test_eig_positive():
    p = 4; A = shapley_matrix(p); Sigma = prior_covariance(p)
    for s in range(1 << p):
        assert eig_closed_form(A, Sigma, [], s, 0.3) > 0


# -- C2: greedy-EIG acquisition reduces posterior det below random -----------
@pytest.mark.parametrize("p", [3, 4, 5])
def test_c2_greedy_lower_det(p):
    A = shapley_matrix(p); Sigma = prior_covariance(p, lengthscale=2.0)
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(1 << p))
    theta_true = L @ rng.standard_normal(1 << p)
    nu_true = lambda m, t=theta_true: float(t[m])
    budget = p + 1
    gt = greedy_acquire(A, Sigma, 0.3, budget, nu_true, np.random.default_rng(0))
    obs_g, _ = gt[-1]
    det_g = shapley_posterior_det(A, Sigma, obs_g, 0.3)
    det_r = np.mean([shapley_posterior_det(A, Sigma,
                      random_acquire(A, Sigma, 0.3, budget, nu_true, np.random.default_rng(r))[-1][0], 0.3)
                     for r in range(8)])
    assert det_g < det_r


# -- C2: greedy reduces expected MSE (trace) on average ----------------------
def test_c2_greedy_lower_trace_average():
    wins = 0; total = 0
    for p in (3, 4, 5):
        A = shapley_matrix(p); Sigma = prior_covariance(p, lengthscale=2.0)
        for seed in range(6):
            rng = np.random.default_rng(seed)
            L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(1 << p))
            theta_true = L @ rng.standard_normal(1 << p)
            nu_true = lambda m, t=theta_true: float(t[m])
            budget = p + 1
            obs_g = greedy_acquire(A, Sigma, 0.3, budget, nu_true, np.random.default_rng(seed))[-1][0]
            tr_g = shapley_posterior_trace(A, Sigma, obs_g, 0.3)
            tr_r = np.mean([shapley_posterior_trace(A, Sigma,
                            random_acquire(A, Sigma, 0.3, budget, nu_true, np.random.default_rng(100*seed+r))[-1][0], 0.3)
                            for r in range(8)])
            wins += tr_g < tr_r; total += 1
    assert wins / total > 0.6


# -- Negative control: a repeat observation of an already-seen coalition has
#    much smaller EIG than a fresh coalition (it is near-redundant). ---------
def test_eig_redundant_observation_small():
    p = 4; A = shapley_matrix(p); Sigma = prior_covariance(p)
    e_repeat = eig_closed_form(A, Sigma, [3], 3, 0.3)        # 2nd noisy read of S=3
    e_fresh = eig_closed_form(A, Sigma, [3], 7, 0.3)         # a brand-new coalition
    assert 0 <= e_repeat < e_fresh / 5                        # repeat is near-redundant

````


````output
...............                                                          [100%]
15 passed in 0.85s

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8ee01f869fc5", "created_at": "2026-07-17T05:50:06+00:00", "title": "Method & environment"}
-->
**Paper:** "ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation" (arXiv 2606.02247, ICML 2026, ub9PwBtHqD). Clean-room from PDF.

**Model (Sec. 2):** value function θ=(ν(S)) over the 2^p coalitions, prior N(0,Σ_θ); observing coalition S gives y(S)=e_Sᵀθ+ε; Shapley φ=Aθ is linear (A = p×2^p Shapley matrix). Linear-Gaussian ⇒ posterior cov and EIG are closed form.

**Implementation (repro/src):** `shapleig.py` (Shapley matrix, prior, posterior, closed-form EIG, exact Shapley); `run_claims.py` (MC mutual-information, greedy/random acquisition, orchestrator). Python 3.12 + numpy, CPU, <3 s.

15/15 pytest tests: Shapley matrix, posterior two-form agreement, EIG-vs-MC (4 cases), EIG>0, greedy-lower-det (3 p), greedy-lower-trace, redundant-observation control.


---
<!-- trackio-cell
{"type": "code", "id": "cell_919abccfcdb2", "created_at": "2026-07-18T10:36:01+00:00", "title": "Verify 30 pinned games", "command": [".venv/bin/python", "repro/src/fetch_vit9_games.py"], "exit_code": 0, "duration_s": 0.055}
-->
````bash
$ .venv/bin/python repro/src/fetch_vit9_games.py
````

exit 0 · 0.1s


````python title=fetch_vit9_games.py
"""Fetch the exact 30 ImageNet/ViT-9 games used by the ShaplEIG paper.

The current shapiq main branch removed these historical assets, so every URL is
pinned to the last parent commit that contained them. SHA-256 checks prevent a
silent data change. The files contain all 2**9 coalition values, precomputed
from costly ViT inference exactly as allowed by the paper's protocol.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

COMMIT = "799cfd0f2c32f17446130247a7ac3519e68cce82"
BASE = (
    "https://raw.githubusercontent.com/mmschlk/shapiq/"
    f"{COMMIT}/data/precomputed_games/ImageClassifier_Game/9"
)
SHA256 = {
    1: "c8ea8c7a24b71432e96557035a78e634b61f98d56a3977228a59100843efb206",
    2: "c1f01f7c8e065ab85e73113550007137d7cc344fa315de4e35b32dcc94100ece",
    3: "b12268d3e39bab74b2c68d3f6a4cb34bb2a0ba06fb5b7ddccbbef95280e236b1",
    4: "31974d6fb290eaafaa72f1d5e36cef082b4a55a6382b8f9d61d75b659f9b0546",
    5: "e8c385d7ee172ffee3231a661be2bbf1068cce102c9e60cad091ddf9ed2c3710",
    6: "b843c0884ac2fa236964980ac61981f8afc85f158d5b95a851685404b0b4768f",
    7: "0c3fdb954775e41fb7b0e6b4ba74decbacfd6b1f3c538e024899901512b56726",
    8: "5afb4dd2ee02deb7dab6197c631e403b9078e4767f76d0b7ece819486bab0f4a",
    9: "55411dae679c302b40028c9a292fa7738c2159d1cf059629801e74c61b8221c9",
    10: "9d63121b9dfc7a7d4312785f6a0a680920fe7df979514e6b2c5885b91fce4035",
    11: "7c8d308327f070485789251de5734463d2a4acbc9b4f08cba71407278ca918a6",
    12: "7354c562b1947c11444b46fbdfb60771040bcae0a1140b17019de5a02250ecaa",
    13: "a6102b7f01335766ffd43c2dfec0b85bd571c1408baa09254c9ac44e831da38b",
    14: "93c8d82952b59b4c65d944116fb0d9ec6817ae2c5c1a4a8afb46da26589dcfcc",
    15: "9365985f6d1e8fcda2e66a5f199c98f87ec155bd474e5eea0fcb1932ac30a80a",
    16: "0564500762f6897ffd2febad3c2e118bb6511778067ce3b517983ab5e8c5d659",
    17: "55c166c5309cdf17d589201ed25c4f4e33faaddfba709fd2fed7d2460f4bce7d",
    18: "b26a535e2b0909682418aa0ade274f333eabb2af904dd60b8b48f441bca4aec9",
    19: "34668fd7b60eb1764423702a95dad87cff34629b01ff64d9bbe8b388742d546c",
    20: "2cb01699e6e9ef37a47738d500ccb5b286e0004e1bdf24e1c25074e5bad44021",
    21: "b0a7923492def3f14e9817664603473c2cf9938d91f405f4d3faff382c3a71f0",
    22: "58e89f5a1d319a558e7a2f38a01d755f11a072e4d8a43d3462dd4579aeff5e55",
    23: "cf12867cf63b9abf1466db5057573f44cf4488ed3d87eed8acbbe8f93b21ee17",
    24: "743c6dfd25c2e44cee10a566ff5fa17b98fb6216e013befe2936644f9ec3a153",
    25: "330bd55076bf4f21ba2a68faa48d010f3ed476662e59d2905a15179c743d25e1",
    26: "5d70be14b8cde253ed42d2f1c1c1d6c4177907820a527ac33fa36150f83e4585",
    27: "df03f1604cef78fd2fb867243257fd858022e7ab83b414bf70ad46ae5cf6e9b2",
    28: "1279b6d1726ad624d378b08150e2d50056aca3e929360dcecf27748d216f392c",
    29: "f9b8febd2a9dc82b10c17275ec4e62f7283eb80a793b7427817f64c63cd5ad65",
    30: "14545393c5f9921424584ed72e2bafdb315f73a372659866043da102b89d2070",
}


def fetch(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed, digest in SHA256.items():
        name = f"model_name=vit_9_patches_{seed}.npz"
        path = root / name
        payload = path.read_bytes() if path.exists() else urlopen(f"{BASE}/{name}").read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != digest:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
        if not path.exists():
            path.write_bytes(payload)
        paths.append(path)
    return paths


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "data" / "vit9"
    print(f"verified {len(fetch(target))} pinned games in {target}")

````


````output
verified 30 pinned games in data/vit9

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_ab3181a9eddf", "created_at": "2026-07-18T10:36:03+00:00", "title": "18-test independent verification suite", "command": [".venv/bin/pytest", "-q", "repro/tests"], "exit_code": 0, "duration_s": 1.401}
-->
````bash
$ .venv/bin/pytest -q repro/tests
````

exit 0 · 1.4s


````output
..................                                                       [100%]
18 passed in 1.15s

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_5aafc7e34156", "created_at": "2026-07-18T10:36:11+00:00", "title": "Exact parity with pinned author baselines", "command": ["env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", ".venv/bin/python", "repro/src/verify_upstream_parity.py"], "exit_code": 0, "duration_s": 3.66}
-->
````bash
$ env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python repro/src/verify_upstream_parity.py
````

exit 0 · 3.7s


````python title=verify_upstream_parity.py
"""Check local SOTA baselines against the authors' pinned source byte-for-byte."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from fetch_vit9_games import fetch
from real_application import (DATA, OUT, game_callable, leverage_estimate,
                              load_game, regression_msr_estimate, sv_from_result)

UPSTREAM = "162ce44fe380c7c11b959fc85206b5dcdeddff58"
BASE = f"https://raw.githubusercontent.com/slds-lmu/shapleig/{UPSTREAM}/src/xac/applications"


def load_remote(module: str):
    payload = urlopen(f"{BASE}/{module}.py").read()
    path = Path(tempfile.gettempdir()) / f"shapleig-{UPSTREAM}-{module}.py"
    path.write_bytes(payload)
    spec = importlib.util.spec_from_file_location(f"upstream_{module}", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def main():
    fetch(DATA)
    poly, rmsr = load_remote("polyshap"), load_remote("regressionMSR")
    frontier = poly.ExplanationFrontierGenerator(N=list(range(9))).generate_kadd(max_order=1)
    rows = []
    cases = [(1, 16), (1, 32), (1, 64), (12, 16), (16, 16)]
    for seed, budget in cases:
        _, values = load_game(DATA / f"model_name=vit_9_patches_{seed}.npz")
        game = game_callable(values)
        official_lev = poly.PolySHAP(
            n=9, explanation_frontier=frontier, sampling_weights=np.ones(10),
            pairing_trick=True, random_state=seed,
        ).approximate(budget=budget, game=game)
        official_reg = rmsr.RegressionMSR(
            n=9, pairing_trick=True, replacement=False,
            sampling_weights=np.ones(10), random_state=seed,
        ).approximate(budget=budget, game=game)
        rows.append({
            "replicate": seed,
            "budget": budget,
            "leverage_max_abs_diff": float(np.max(np.abs(
                sv_from_result(official_lev, 9) - leverage_estimate(values, budget, seed)
            ))),
            "regression_msr_max_abs_diff": float(np.max(np.abs(
                sv_from_result(official_reg, 9) - regression_msr_estimate(values, budget, seed)
            ))),
        })
    result = {
        "upstream": f"slds-lmu/shapleig@{UPSTREAM}",
        "budgets": rows,
        "pass": all(max(r["leverage_max_abs_diff"], r["regression_msr_max_abs_diff"]) < 1e-12
                    for r in rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "upstream_parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

````


````output
{
  "upstream": "slds-lmu/shapleig@162ce44fe380c7c11b959fc85206b5dcdeddff58",
  "budgets": [
    {
      "replicate": 1,
      "budget": 16,
      "leverage_max_abs_diff": 0.0,
      "regression_msr_max_abs_diff": 0.0
    },
    {
      "replicate": 1,
      "budget": 32,
      "leverage_max_abs_diff": 0.0,
      "regression_msr_max_abs_diff": 0.0
    },
    {
      "replicate": 1,
      "budget": 64,
      "leverage_max_abs_diff": 0.0,
      "regression_msr_max_abs_diff": 0.0
    },
    {
      "replicate": 12,
      "budget": 16,
      "leverage_max_abs_diff": 0.0,
      "regression_msr_max_abs_diff": 0.0
    },
    {
      "replicate": 16,
      "budget": 16,
      "leverage_max_abs_diff": 0.0,
      "regression_msr_max_abs_diff": 0.0
    }
  ],
  "pass": true
}

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bea73177859e", "created_at": "2026-07-18T10:37:15+00:00", "title": "Data, method, and parity"}
-->
## Paper-comparable method

- Data: 30 SHA-256-verified games pinned to `mmschlk/shapiq@799cfd0f2c32f17446130247a7ac3519e68cce82`.
- Primary: paired LeverageSHAP `p+1` initialization, quasi-noiseless Hamming GP, MAP ARD lengthscales, exhaustive closed-form EIG.
- Baselines: KernelSHAP, LeverageSHAP, RegressionMSR, UnbiasedKernelSHAP, SVARM, permutation sampling; plus GP-random, GP-uncertainty, GP-LeverageSHAP, and fixed-isotropic ShaplEIG ablations.
- Author parity: local LeverageSHAP and RegressionMSR match `slds-lmu/shapleig@162ce44fe380c7c11b959fc85206b5dcdeddff58` with maximum absolute difference 0.0 on five checks.
- Independent controls: direct-vs-vectorized EIG log determinant below 1e-7, quasi-noiseless interpolation, exact Shapley mapping, posterior Schur complement, and Monte Carlo mutual information.

The 14-route attempt ledger is published with the repository. The fixed-isotropic sensitivity control is reported separately and is not substituted for adaptive ShaplEIG.
