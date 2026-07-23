"""Real-application ShaplEIG benchmark on the paper's ImageNet/ViT-9 game.

This is a compact, independent NumPy/SciPy implementation of the paper's
Hamming-GP posterior and closed-form EIG. Baselines use shapiq 1.4.1 and the
same settings as the authors' public configuration. Every reported error is a
realized error against the exhaustive 512-coalition Shapley value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import binom

from fetch_vit9_games import COMMIT, SHA256, fetch
from shapleig import shapley_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "real_application"
DATA = ROOT / "data" / "vit9"
NOISE = 1e-6


def grid(p: int) -> np.ndarray:
    return ((np.arange(1 << p)[:, None] >> np.arange(p)) & 1).astype(bool)


def load_game(path: Path, expected_p: int | None = 9) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    p = int(d["n_players"])
    if expected_p is not None and p != expected_p:
        raise ValueError(f"expected {expected_p} players in {path}, got {p}")
    if d["coalitions"].shape != (1 << p, p):
        raise ValueError(f"unexpected game shape in {path}")
    masks = d["coalitions"].astype(np.int64) @ (1 << np.arange(p))
    if len(np.unique(masks)) != 1 << p or set(masks) != set(range(1 << p)):
        raise ValueError(f"coalitions are not exhaustive in {path}")
    values = np.empty(1 << p)
    values[masks] = d["values"]
    return grid(p), values


def sv_from_result(result, p: int) -> np.ndarray:
    return np.array([result[(i,)] for i in range(p)], dtype=float)


def game_callable(values: np.ndarray):
    weights = 1 << np.arange(int(round(math.log2(len(values)))))

    def game(coalitions):
        x = np.asarray(coalitions, dtype=np.int64)
        one = x.ndim == 1
        x = np.atleast_2d(x)
        ans = values[x @ weights]
        return ans[0] if one else ans

    return game


def hamming_kernel(X: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """Authors' implementation: exp(-mean_j([x_j!=x'_j]/ell_j))."""
    distance = np.sum(X[:, None, :] != X[None, :, :], axis=2) if ell.size == 1 else None
    if ell.size == 1:
        return np.exp(-distance / (X.shape[1] * ell[0]))
    weighted = np.sum((X[:, None, :] != X[None, :, :]) / ell, axis=2)
    return np.exp(-weighted / X.shape[1])


def _nll_and_grad(params: np.ndarray, Xo: np.ndarray, yo: np.ndarray, ard: bool):
    n_lengthscales = Xo.shape[1] if ard else 1
    log_ell, log_scale, mean = params[:n_lengthscales], params[-2], params[-1]
    ell, outputscale = np.exp(log_ell), math.exp(log_scale)
    base = hamming_kernel(Xo, ell)
    K = outputscale * base + NOISE * np.eye(len(Xo))
    residual = yo - mean
    try:
        cf = cho_factor(K, lower=True, check_finite=False)
        alpha = cho_solve(cf, residual, check_finite=False)
        fit = 0.5 * residual @ alpha
        logdet = np.sum(np.log(np.diag(cf[0])))
    except np.linalg.LinAlgError:
        return 1e100, np.zeros_like(params)
    # BoTorch default LogNormal(sqrt(2)+.5log(p), sqrt(3)) prior.
    mu, sigma = math.sqrt(2) + 0.5 * math.log(Xo.shape[1]), math.sqrt(3)
    prior = np.sum(log_ell + 0.5 * ((log_ell - mu) / sigma) ** 2)
    Kinv = cho_solve(cf, np.eye(len(Xo)), check_finite=False)
    common = 0.5 * (Kinv - np.outer(alpha, alpha))
    delta = Xo[:, None, :] != Xo[None, :, :]
    grad = np.empty_like(params)
    if ard:
        for j in range(n_lengthscales):
            dK = outputscale * base * delta[:, :, j] / (Xo.shape[1] * ell[j])
            grad[j] = np.sum(common * dK) + 1 + (log_ell[j] - mu) / sigma**2
    else:
        distance = delta.sum(axis=2)
        dK = outputscale * base * distance / (Xo.shape[1] * ell[0])
        grad[0] = np.sum(common * dK) + 1 + (log_ell[0] - mu) / sigma**2
    grad[-2] = np.sum(common * (outputscale * base))
    grad[-1] = -np.sum(alpha)
    return float(fit + logdet + prior), grad


def fit_gp(Xo: np.ndarray, values: np.ndarray, start: np.ndarray, ard: bool,
           seed: int, restarts: int = 5) -> tuple[np.ndarray, float, float]:
    y = values - np.mean(values)
    scale = np.std(y)
    y = y / scale if scale > 1e-12 else y
    initial_ell = np.log(start if ard else np.array([float(np.mean(start))]))
    rng = np.random.default_rng(seed)
    starts = [np.r_[initial_ell, 0.0, 0.0]]
    mu, sigma = math.sqrt(2) + 0.5 * math.log(Xo.shape[1]), math.sqrt(3)
    for _ in range(restarts - 1):
        random_ell = np.clip(rng.normal(mu, sigma, len(initial_ell)), math.log(1e-6), math.log(1e4))
        starts.append(np.r_[random_ell, rng.uniform(-2, 2), rng.normal(0, 0.25)])
    bounds = ([(math.log(1e-6), math.log(1e4))] * len(initial_ell)
              + [(math.log(1e-4), math.log(1e4)), (-5, 5)])
    best = None
    for x0 in starts:
        result = minimize(_nll_and_grad, x0, args=(Xo, y, ard), jac=True,
                          method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 100, "ftol": 1e-10, "gtol": 1e-7})
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    chosen = best.x if best is not None else starts[0]
    return np.exp(chosen[:-2]), math.exp(chosen[-2]), float(chosen[-1])


def gp_state(X: np.ndarray, values: np.ndarray, observed: list[int], ell: np.ndarray,
             A: np.ndarray, outputscale: float = 1.0, fitted_mean: float = 0.0):
    K = outputscale * hamming_kernel(X, ell)
    obs = np.asarray(observed, dtype=int)
    y = values[obs]
    mean, scale = float(np.mean(y)), float(np.std(y))
    scale = scale if scale > 1e-12 else 1.0
    yz = (y - mean) / scale - fitted_mean
    Koo = K[np.ix_(obs, obs)] + NOISE * np.eye(len(obs))
    cf = cho_factor(Koo, lower=True, check_finite=False)
    alpha = cho_solve(cf, yz, check_finite=False)
    posterior_mean = fitted_mean + K[:, obs] @ alpha
    phi = scale * (A @ posterior_mean)  # constant mean has zero Shapley value
    AK = A @ K
    solved = cho_solve(cf, K[obs, :], check_finite=False)
    Q = AK - AK[:, obs] @ solved
    C = A @ K @ A.T - AK[:, obs] @ cho_solve(cf, AK[:, obs].T, check_finite=False)
    C = (C + C.T) / 2 + 1e-10 * np.eye(A.shape[0])
    variance = np.maximum(outputscale + NOISE - np.sum(K[obs, :] * solved, axis=0), 1e-14)
    return phi, Q, C, variance


def leverage_sampler(p: int, budget: int, seed: int):
    from shapiq.approximator.sampling import CoalitionSampler
    sampler = CoalitionSampler(n_players=p, sampling_weights=np.ones(p + 1),
                               pairing_trick=True, random_state=seed)
    sampler.sample(budget)
    return sampler


def leverage_sample(p: int, budget: int, seed: int) -> np.ndarray:
    sampler = leverage_sampler(p, budget, seed)
    return sampler.coalitions_matrix.astype(np.int64) @ (1 << np.arange(p))


def gp_run(values: np.ndarray, policy: str, budgets: list[int], seed: int,
           adaptive: bool = True, ard: bool = True, refit_interval: int = 4) -> dict[int, np.ndarray]:
    p = int(round(math.log2(len(values))))
    if len(values) != 1 << p:
        raise ValueError("game values must enumerate a complete power-of-two coalition space")
    X, A = grid(p), shapley_matrix(p)
    observed = list(dict.fromkeys(leverage_sample(p, p + 1, seed).tolist()))
    rng = np.random.default_rng(seed)
    prior_mode = math.exp(math.sqrt(2) + 0.5 * math.log(p) - 3)
    ell = np.full(p if ard else 1, prior_mode)
    outputscale, fitted_mean = 1.0, 0.0
    estimates = {}
    initial_n = len(observed)
    for step in range(initial_n, max(budgets) + 1):
        if adaptive and (step - initial_n) % refit_interval == 0:
            ell, outputscale, fitted_mean = fit_gp(
                X[observed], values[observed], ell, ard=ard,
                seed=seed * 1000 + step, restarts=5,
            )
        phi, Q, C, variance = gp_state(
            X, values, observed, ell, A, outputscale=outputscale, fitted_mean=fitted_mean
        )
        if step in budgets:
            estimates[step] = phi
        if step == max(budgets):
            break
        unseen = np.ones(1 << p, dtype=bool); unseen[observed] = False
        if policy == "random":
            candidate = int(rng.choice(np.flatnonzero(unseen)))
        elif policy == "uncertainty":
            score = variance.copy(); score[~unseen] = -np.inf
            candidate = int(np.argmax(score))
        elif policy == "leverage":
            order = leverage_sample(p, 1 << p, seed)
            candidate = int(next(x for x in order if unseen[x]))
        elif policy == "eig":
            invCQ = np.linalg.solve(C, Q)
            rho = np.sum(Q * invCQ, axis=0) / variance
            score = -0.5 * np.log1p(-np.clip(rho, 0, 1 - 1e-12))
            score[~unseen] = -np.inf
            candidate = int(np.argmax(score))
        else:
            raise ValueError(policy)
        observed.append(candidate)
    return estimates


def permutation_estimate(values: np.ndarray, budget: int, seed: int) -> np.ndarray:
    p = int(round(math.log2(len(values))))
    rng = np.random.default_rng(seed)
    total, used = np.zeros(p), 0
    while used + p + 1 <= budget:
        perm = rng.permutation(p)
        mask, before = 0, values[0]
        for player in perm:
            mask |= 1 << int(player)
            after = values[mask]
            total[player] += after - before
            before = after
        used += p + 1
    return total / max(used // (p + 1), 1)


def baseline_estimates(values: np.ndarray, budget: int, seed: int) -> dict[str, np.ndarray]:
    from shapiq import KernelSHAP, SVARM, UnbiasedKernelSHAP
    p = int(round(math.log2(len(values))))
    game, weights = game_callable(values), np.ones(p + 1)
    methods = {
        "KernelSHAP": KernelSHAP(n=p, index="SV", max_order=1, pairing_trick=True,
                                  sampling_weights=weights, random_state=seed),
        "SVARM": SVARM(n=p, index="SV", max_order=1, pairing_trick=True,
                        sampling_weights=weights, random_state=seed),
        "UnbiasedKernelSHAP": UnbiasedKernelSHAP(n=p, pairing_trick=True,
                                                  replacement=False, random_state=seed),
    }
    ans = {name: sv_from_result(method.approximate(budget=budget, game=game), p)
           for name, method in methods.items()}
    ans["PermutationSampling"] = permutation_estimate(values, budget, seed)
    ans["RegressionMSR"] = regression_msr_estimate(values, budget, seed)
    return ans


def regression_msr_estimate(values: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Authors' RegressionMSR: XGBoost/TreeSHAP plus unbiased residual SV."""
    import shap
    from shapiq import UnbiasedKernelSHAP
    from xgboost import XGBRegressor

    p = int(round(math.log2(len(values))))
    sampler = leverage_sampler(p, budget, seed)
    X = sampler.coalitions_matrix.astype(float)
    y = game_callable(values)(X)
    model = XGBRegressor(random_state=seed, n_jobs=1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(
        model, feature_perturbation="interventional", data=np.zeros((1, p))
    )
    tree = np.asarray(explainer.shap_values(np.ones(p)), dtype=float)
    residual = y - model.predict(X)

    # This deliberately mirrors upstream residualGame: the residual estimator's
    # same-seed sampler requests the same coalitions, so values are returned in
    # their sampled order.
    residual_method = UnbiasedKernelSHAP(
        n=p, pairing_trick=True, replacement=False, random_state=seed
    )
    residual_result = residual_method.approximate(
        budget=budget, game=lambda coalitions: residual.copy()
    )
    return tree + sv_from_result(residual_result, p)


def leverage_estimate(values: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """LeverageSHAP regression from the authors' PolySHAP implementation."""
    p = int(round(math.log2(len(values))))
    sampler = leverage_sampler(p, budget, seed)
    X = sampler.coalitions_matrix.astype(bool)
    coals = X.astype(np.int64) @ (1 << np.arange(p))
    y = values[coals].copy()
    # Endpoints are always rows 0/1 in shapiq's paired CoalitionSampler.
    empty, full = y[0], y[1] - y[0]
    y -= empty
    sizes = X[2:].sum(axis=1)
    kernel = 1 / ((p - 1) * binom(p - 2, sizes - 1))
    W = np.sqrt(kernel * sampler.sampling_adjustment_weights[2:])
    Xt = W[:, None] * X[2:]
    P = np.eye(p) - np.ones((p, p)) / p
    beta = np.linalg.lstsq(Xt @ P, W * y[2:] - full / p * Xt.sum(axis=1), rcond=None)[0]
    return full / p + beta


def run(seeds: int, budgets: list[int]) -> dict:
    paths = fetch(DATA)[:seeds]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, started = [], time.perf_counter()
    for replicate, path in enumerate(paths, 1):
        X, values = load_game(path)
        truth = shapley_matrix(9) @ values
        seed = replicate
        gp_results = {
            "ShaplEIG": gp_run(values, "eig", budgets, seed, adaptive=True, ard=True, refit_interval=1),
            "ShaplEIG-fixed": gp_run(values, "eig", budgets, seed, adaptive=False, ard=False),
            "GP-random": gp_run(values, "random", budgets, seed, adaptive=True, ard=True),
            "GP-uncertainty": gp_run(values, "uncertainty", budgets, seed, adaptive=True, ard=True),
            "GP-LeverageSHAP": gp_run(values, "leverage", budgets, seed, adaptive=True, ard=True),
        }
        for budget in budgets:
            estimates = {name: by_budget[budget] for name, by_budget in gp_results.items()}
            estimates.update(baseline_estimates(values, budget, seed))
            estimates["LeverageSHAP"] = leverage_estimate(values, budget, seed)
            for name, estimate in estimates.items():
                rows.append({"application": "ImageNet ViT-9 local explanation",
                             "replicate": replicate, "budget": budget, "method": name,
                             "mse": float(np.mean((estimate - truth) ** 2)),
                             "efficiency_error": float(abs(estimate.sum() - (values[-1] - values[0]))),
                             "data_sha256": SHA256[replicate]})
        print(f"replicate {replicate}/{seeds} complete", flush=True)
    csv_path = OUT / "vit9_realized_mse.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    methods = sorted({r["method"] for r in rows})
    aggregate = {}
    for budget in budgets:
        aggregate[str(budget)] = {
            method: float(np.mean([r["mse"] for r in rows if r["budget"] == budget and r["method"] == method]))
            for method in methods
        }
    competitors = ["KernelSHAP", "LeverageSHAP", "RegressionMSR", "UnbiasedKernelSHAP",
                   "SVARM", "PermutationSampling"]
    comparisons = {}
    for other in competitors:
        pairs = []
        for replicate in range(1, seeds + 1):
            for budget in budgets:
                a = next(r["mse"] for r in rows if r["replicate"] == replicate and r["budget"] == budget and r["method"] == "ShaplEIG")
                b = next(r["mse"] for r in rows if r["replicate"] == replicate and r["budget"] == budget and r["method"] == other)
                pairs.append((a, b))
        comparisons[other] = {"win_fraction": float(np.mean([a < b for a, b in pairs])),
                              "mean_mse_ratio": float(np.mean([a / b for a, b in pairs]))}
    summary = {
        "application": "ImageNet local explanation with Vision Transformer and 9 image patches",
        "source": f"shapiq@{COMMIT}", "replicates": seeds, "coalitions_per_game": 512,
        "budgets": budgets, "methods": methods, "aggregate_mean_mse": aggregate,
        "shapleig_vs_sota": comparisons, "elapsed_seconds": time.perf_counter() - started,
        "data_integrity": "PASS", "realized_errors_not_posterior_proxy": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 24, 32, 48, 64])
    args = parser.parse_args()
    run(args.seeds, args.budgets)
