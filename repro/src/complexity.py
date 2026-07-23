"""Polynomial and naive EIG implementations for the Claim 1 audit.

The efficient route implements the elementary-symmetric-polynomial identities
from Theorems B.1/B.2 of arXiv:2606.02247. It never constructs the 2**p
coalition grid. The independent route explicitly constructs that grid and the
2**p by 2**p posterior covariance, so it is intentionally limited to small p.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class EIGResult:
    eig: float
    aka_operations: int
    efficient_peak_bytes: int


def shapley_weights(p: int, *, signed: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return Shapley weights indexed by coalition size."""
    w_in = np.zeros(p + 1)
    w_out = np.zeros(p + 1)
    for k in range(1, p + 1):
        w_in[k] = 1.0 / math.comb(p - 1, k - 1)
    for k in range(p):
        w_out[k] = 1.0 / math.comb(p - 1, k)
    if signed:
        w_out *= -1
    return w_in / p, w_out / p


def shapley_matrix_direct(p: int) -> np.ndarray:
    """Explicit p by 2**p Shapley matrix for the independent checker."""
    w_in, w_out = shapley_weights(p)
    masks = np.arange(1 << p, dtype=np.uint64)
    sizes = np.array([int(x).bit_count() for x in masks])
    A = np.empty((p, 1 << p))
    for j in range(p):
        inside = ((masks >> j) & 1).astype(bool)
        A[j] = np.where(inside, w_in[sizes], w_out[sizes])
    return A


def kernel_parameters(
    lengthscales: np.ndarray, outputscale: float
) -> tuple[np.ndarray, np.ndarray]:
    p = len(lengthscales)
    alpha = np.full(p, outputscale ** (1.0 / p))
    beta = alpha * np.exp(-1.0 / (lengthscales * p))
    return alpha, beta


def kernel_points(
    left: np.ndarray,
    right: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Product Hamming kernel between binary row matrices."""
    factors = np.where(
        left[:, None, :] == right[None, :, :],
        alpha[None, None, :],
        beta[None, None, :],
    )
    return factors.prod(axis=2)


def akz_esp(
    W: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Compute A K(Z,W) without enumerating Z, using Theorem B.1."""
    W = np.asarray(W, dtype=np.int8)
    p = len(alpha)
    w_in, w_out = shapley_weights(p)
    result = np.empty((p, len(W)))
    for row_idx, row in enumerate(W):
        gamma = np.where(row == 0, alpha, beta)
        delta = np.where(row == 0, beta, alpha)
        prefix = [np.ones(1)]
        for j in range(p):
            prefix.append(np.convolve(prefix[-1], [gamma[j], delta[j]]))
        suffix: list[np.ndarray] = [np.empty(0) for _ in range(p + 1)]
        suffix[p] = np.ones(1)
        for j in range(p - 1, -1, -1):
            suffix[j] = np.convolve([gamma[j], delta[j]], suffix[j + 1])
        for j in range(p):
            excluded = np.convolve(prefix[j], suffix[j + 1])
            result[j, row_idx] = (
                delta[j] * (excluded @ w_in[1:])
                + gamma[j] * (excluded @ w_out[:-1])
            )
    return result


def akazza_esp(
    alpha: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Compute A K(Z,Z) A.T in O(p**4), following Theorem B.2.

    Returns the matrix, an executable scalar-operation upper count for the
    tensor updates, and the maximum live workspace bytes of the algorithm.
    """
    p = len(alpha)
    ps = p + 1
    w_in, w_out_signed = shapley_weights(p, signed=False)
    # The theorem applies 1/p**2 outside, so remove the per-vector 1/p here.
    w_in *= p
    w_out = w_out_signed * p

    O0 = w_out.copy()
    O1 = np.zeros(ps)
    O1[:p] = w_out[1:]
    I1 = np.zeros(ps)
    I1[:p] = w_in[1:]
    I2 = np.zeros(ps)
    I2[: p - 1] = w_in[2:]
    left_t = np.stack([O0, O1, I1, I2], axis=1)
    right_t = left_t.copy()

    p_left = np.zeros((ps, ps, 4))
    suffix_all = np.zeros((p, ps, ps, 4))
    suffix_logs = np.zeros(p)
    aka = np.zeros((p, p))
    operations = 0

    live_bytes = (
        p_left.nbytes * 4
        + suffix_all.nbytes
        + suffix_logs.nbytes
        + aka.nbytes
        + left_t.nbytes
        + right_t.nbytes
    )

    for i in range(p):
        factors = [r for r in range(p) if r != i]
        nf = len(factors)
        s_prev = np.zeros_like(p_left)
        s_prev[0, :, :] = right_t
        suffix_all.fill(0)
        suffix_logs.fill(0)
        log_s = 0.0

        for k in range(nf - 1, -1, -1):
            r = factors[k]
            a, b = alpha[r], beta[r]
            s_new = a * s_prev
            s_new[1:, :, :] += b * s_prev[:-1, :, :]
            s_new[:, :-1, :] += b * s_prev[:, 1:, :]
            s_new[1:, :-1, :] += a * s_prev[:-1, 1:, :]
            operations += (
                s_prev.size
                + 2 * s_prev[:-1, :, :].size
                + 2 * s_prev[:, 1:, :].size
                + 2 * s_prev[:-1, 1:, :].size
            )
            scale = max(float(np.max(np.abs(s_new))), np.finfo(float).tiny)
            s_new /= scale
            log_s += math.log(scale)
            suffix_all[k] = s_new
            suffix_logs[k] = log_s
            s_prev = s_new

        p_curr = np.zeros_like(p_left)
        p_curr[:, 0, :] = left_t
        log_p = 0.0
        contracted = np.einsum("abl,abr->lr", p_curr, suffix_all[0])
        operations += 2 * p_curr.shape[0] * p_curr.shape[1] * 16
        ai, bi = alpha[i], beta[i]
        diagonal = ai * (contracted[2, 2] + contracted[0, 0]) - bi * (
            contracted[2, 0] + contracted[0, 2]
        )
        aka[i, i] = diagonal * math.exp(suffix_logs[0]) / (p * p)

        for m, j in enumerate(factors):
            if j > i:
                continue
            aj, bj = alpha[j], beta[j]
            if m < nf - 1:
                contracted = np.einsum(
                    "abl,abr->lr", p_curr, suffix_all[m + 1]
                )
                scale = log_p + suffix_logs[m + 1]
                operations += 2 * p_curr.shape[0] * p_curr.shape[1] * 16
            else:
                contracted = np.einsum("bl,br->lr", p_curr[0], right_t)
                scale = log_p
                operations += 2 * p_curr.shape[1] * 16

            value = (
                ai * aj * contracted[0, 0]
                + bi * aj * contracted[0, 1]
                - ai * bj * contracted[0, 2]
                - bi * bj * contracted[0, 3]
                + ai * bj * contracted[1, 0]
                + bi * bj * contracted[1, 1]
                - ai * aj * contracted[1, 2]
                - bi * aj * contracted[1, 3]
                - bi * aj * contracted[2, 0]
                - ai * aj * contracted[2, 1]
                + bi * bj * contracted[2, 2]
                + ai * bj * contracted[2, 3]
                - bi * bj * contracted[3, 0]
                - ai * bj * contracted[3, 1]
                + bi * aj * contracted[3, 2]
                + ai * aj * contracted[3, 3]
            ) * math.exp(scale) / (p * p)
            operations += 64
            aka[i, j] = value
            aka[j, i] = value

            p_new = aj * p_curr
            p_new[:-1, :, :] += bj * p_curr[1:, :, :]
            p_new[:, 1:, :] += bj * p_curr[:, :-1, :]
            p_new[:-1, 1:, :] += aj * p_curr[1:, :-1, :]
            operations += (
                p_curr.size
                + 2 * p_curr[1:, :, :].size
                + 2 * p_curr[:, :-1, :].size
                + 2 * p_curr[1:, :-1, :].size
            )
            scale_new = max(float(np.max(np.abs(p_new))), np.finfo(float).tiny)
            p_curr = p_new / scale_new
            log_p += math.log(scale_new)

    return (aka + aka.T) / 2, operations, live_bytes


def efficient_eig(
    observed: np.ndarray,
    candidate: np.ndarray,
    lengthscales: np.ndarray,
    *,
    outputscale: float = 1.7,
    noise: float = 1e-6,
) -> EIGResult:
    """Closed-form EIG with no 2**p objects."""
    observed = np.asarray(observed, dtype=np.int8)
    candidate = np.asarray(candidate, dtype=np.int8).reshape(1, -1)
    alpha, beta = kernel_parameters(lengthscales, outputscale)
    aka, operations, live_bytes = akazza_esp(alpha, beta)
    akx = akz_esp(observed, alpha, beta)
    akw = akz_esp(candidate, alpha, beta)
    kxx = kernel_points(observed, observed, alpha, beta)
    kxx.flat[:: len(kxx) + 1] += noise
    kxw = kernel_points(observed, candidate, alpha, beta)
    factor = cho_factor(kxx, lower=True, check_finite=False)
    solved_akx = cho_solve(factor, akx.T, check_finite=False)
    c = aka - akx @ solved_akx
    c = (c + c.T) / 2
    b = akw - akx @ cho_solve(factor, kxw, check_finite=False)
    variance = (
        outputscale
        + noise
        - (kxw.T @ cho_solve(factor, kxw, check_finite=False)).item()
    )
    c_factor = cho_factor(c, lower=True, check_finite=False)
    rho = (b.T @ cho_solve(c_factor, b, check_finite=False)).item() / variance
    eig = -0.5 * math.log1p(-min(max(rho, 0.0), 1.0 - 1e-12))
    live_bytes += (
        aka.nbytes
        + akx.nbytes
        + akw.nbytes
        + kxx.nbytes
        + kxw.nbytes
        + c.nbytes
        + b.nbytes
    )
    return EIGResult(eig=eig, aka_operations=operations, efficient_peak_bytes=live_bytes)


def naive_eig(
    observed: np.ndarray,
    candidate: np.ndarray,
    lengthscales: np.ndarray,
    *,
    outputscale: float = 1.7,
    noise: float = 1e-6,
) -> tuple[float, int]:
    """Independent explicit 2**p route used only for small p."""
    p = len(lengthscales)
    grid = ((np.arange(1 << p)[:, None] >> np.arange(p)) & 1).astype(np.int8)
    alpha, beta = kernel_parameters(lengthscales, outputscale)
    A = shapley_matrix_direct(p)
    kzz = kernel_points(grid, grid, alpha, beta)
    kxz = kernel_points(observed, grid, alpha, beta)
    kxx = kernel_points(observed, observed, alpha, beta)
    kxx.flat[:: len(kxx) + 1] += noise
    posterior = kzz - kxz.T @ np.linalg.solve(kxx, kxz)
    c = A @ posterior @ A.T
    mask = int(np.asarray(candidate, dtype=np.uint64) @ (1 << np.arange(p)))
    cross = A @ posterior[:, mask]
    variance = posterior[mask, mask] + noise
    c_new = c - np.outer(cross, cross) / variance
    eig = 0.5 * (
        np.linalg.slogdet(c)[1] - np.linalg.slogdet(c_new)[1]
    )
    peak_bytes = (
        grid.nbytes
        + A.nbytes
        + kzz.nbytes
        + kxz.nbytes
        + kxx.nbytes
        + posterior.nbytes
        + c.nbytes
        + c_new.nbytes
    )
    return float(eig), peak_bytes
