"""Independent identity checks for the polynomial Claim 1 implementation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from complexity import (
    akz_esp,
    akazza_esp,
    efficient_eig,
    efficient_eig_batch,
    kernel_parameters,
    kernel_points,
    naive_eig,
    shapley_matrix_direct,
)


@pytest.mark.parametrize("p", [3, 4, 5])
def test_esp_aka_matches_explicit_coalition_grid(p):
    rng = np.random.default_rng(260602247 + p)
    lengthscales = rng.uniform(0.6, 2.0, size=p)
    alpha, beta = kernel_parameters(lengthscales, outputscale=1.7)
    grid = ((np.arange(1 << p)[:, None] >> np.arange(p)) & 1).astype(np.int8)
    A = shapley_matrix_direct(p)
    explicit = A @ kernel_points(grid, grid, alpha, beta) @ A.T
    polynomial, _, _ = akazza_esp(alpha, beta)
    assert np.max(np.abs(polynomial - explicit)) < 1e-9


@pytest.mark.parametrize("p", [3, 4, 5, 8])
def test_synthetic_division_akz_matches_explicit_grid(p):
    rng = np.random.default_rng(260602247 + 100 * p)
    lengthscales = rng.uniform(0.6, 2.0, size=p)
    alpha, beta = kernel_parameters(lengthscales, outputscale=1.7)
    candidates = rng.integers(0, 2, size=(7, p), dtype=np.int8)
    grid = ((np.arange(1 << p)[:, None] >> np.arange(p)) & 1).astype(np.int8)
    explicit = shapley_matrix_direct(p) @ kernel_points(
        grid, candidates, alpha, beta
    )
    polynomial = akz_esp(candidates, alpha, beta)
    assert np.max(np.abs(polynomial - explicit)) < 1e-9


@pytest.mark.parametrize("p", [3, 4, 5])
def test_polynomial_eig_matches_explicit_posterior(p):
    rng = np.random.default_rng(260602247 + p)
    masks = rng.choice(1 << p, size=p + 2, replace=False)
    observed = ((masks[: p + 1, None] >> np.arange(p)) & 1).astype(np.int8)
    candidate = ((masks[p + 1] >> np.arange(p)) & 1).astype(np.int8)
    lengthscales = rng.uniform(0.6, 2.0, size=p)
    efficient = efficient_eig(observed, candidate, lengthscales).eig
    explicit, _ = naive_eig(observed, candidate, lengthscales)
    assert abs(efficient - explicit) < 1e-7


@pytest.mark.parametrize("p", [3, 4, 5])
def test_vectorized_eig_matches_independent_explicit_posterior(p):
    rng = np.random.default_rng(260602247 + 10 * p)
    masks = rng.choice(1 << p, size=p + 4, replace=False)
    observed = ((masks[: p + 1, None] >> np.arange(p)) & 1).astype(np.int8)
    candidates = (
        (masks[p + 1 :, None] >> np.arange(p)) & 1
    ).astype(np.int8)
    lengthscales = rng.uniform(0.6, 2.0, size=p)
    batch = efficient_eig_batch(observed, candidates, lengthscales).scores
    explicit = np.array(
        [naive_eig(observed, row, lengthscales)[0] for row in candidates]
    )
    assert np.max(np.abs(batch - explicit)) < 1e-7
