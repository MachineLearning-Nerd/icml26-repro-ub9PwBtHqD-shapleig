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
