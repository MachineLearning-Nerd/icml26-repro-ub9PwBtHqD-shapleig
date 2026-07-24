"""Independent checks for the real ImageNet/ViT benchmark machinery."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from real_application import gp_state, grid, hamming_kernel
from shapleig import shapley_matrix


def test_hamming_kernel_matches_product_definition():
    X = grid(3)
    ell = np.array([0.7, 1.2, 2.0])
    expected = np.ones((8, 8))
    for j in range(3):
        expected *= np.exp(-(X[:, None, j] != X[None, :, j]).astype(float) / (3 * ell[j]))
    assert np.allclose(hamming_kernel(X, ell), expected)


def test_vectorized_eig_matches_direct_covariance_update():
    p, X, A = 4, grid(4), shapley_matrix(4)
    values = np.sin(np.arange(16))
    obs, ell = [0, 15, 3, 12, 5], np.array([0.8, 1.1, 0.6, 2.0])
    _, Q, C, variance = gp_state(X, values, obs, ell, A)
    candidate = 7
    rho = Q[:, candidate] @ np.linalg.solve(C, Q[:, candidate]) / variance[candidate]
    fast = -0.5 * np.log1p(-rho)
    Cnew = C - np.outer(Q[:, candidate], Q[:, candidate]) / variance[candidate]
    direct = 0.5 * (np.linalg.slogdet(C)[1] - np.linalg.slogdet(Cnew)[1])
    assert abs(fast - direct) < 1e-7


def test_observed_coalitions_interpolate():
    p, X, A = 4, grid(4), shapley_matrix(4)
    values = np.cos(np.arange(16) / 3)
    obs = [0, 15, 1, 2, 4, 8]
    # Reconstruct the latent posterior directly and verify quasi-noiseless fit.
    ell = np.ones(4)
    K = hamming_kernel(X, ell)
    y = values[obs]; mean, scale = y.mean(), y.std()
    pred = mean + scale * K[:, obs] @ np.linalg.solve(
        K[np.ix_(obs, obs)] + 1e-6 * np.eye(len(obs)), (y - mean) / scale)
    assert np.max(np.abs(pred[obs] - y)) < 2e-5
