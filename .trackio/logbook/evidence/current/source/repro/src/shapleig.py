"""ShaplEIG: closed-form Expected Information Gain for Shapley-value estimation
(from "ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation",
arXiv 2606.02247, ub9PwBtHqD).

Model (linear-Gaussian inverse problem, Sec. 2):
  * parameter  theta = (nu(S))_{S subset [p]}   (the 2^p coalition values),
    prior theta ~ N(0, Sigma_theta);
  * observing coalition S yields  y(S) = e_S^T theta + eps,  eps ~ N(0, sigma_eps^2);
  * the Shapley value phi(theta) = A theta is a LINEAR functional of theta,
    where A is the p x 2^p Shapley matrix (Eq. 1).

Because both the observation model and the end-goal phi = A theta are linear and
all variables are Gaussian, the posterior covariance and the Expected
Information Gain admit exact closed forms:

    Sigma(theta | y(S)) = (Sigma_theta^{-1} + sigma_eps^{-2} e_S e_S^T)^{-1},
    EIG_phi(x=S) = -1/2 log det( A Sigma(theta|y(S)) A^T ) + C    (Eq. 3),

i.e. the mutual information I(phi(theta); y(S)).  Greedy BAD acquires, at each
step, the coalition S maximizing EIG_phi(S) (equivalently minimizing the
posterior entropy of the Shapley value).
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from itertools import combinations


def coalition_index(p: int):
    """Map each subset S of [p] (as a frozenset) to its index 0..2^p-1."""
    idx = {}
    for mask in range(1 << p):
        S = frozenset(i for i in range(p) if mask & (1 << i))
        idx[S] = mask
    return idx


def shapley_matrix(p: int) -> np.ndarray:
    r"""A (p x 2^p): phi_i = A theta where theta[S] = nu(S).
    phi_i = sum_{S subset [p]\{i}} (1/(p * C(p-1,|S|))) (nu(S u {i}) - nu(S))."""
    from math import comb
    A = np.zeros((p, 1 << p))
    for i in range(p):
        for mask in range(1 << p):
            if mask & (1 << i):
                continue
            S = mask
            Si = mask | (1 << i)
            s = bin(mask).count("1")
            w = 1.0 / (p * comb(p - 1, s))
            A[i, Si] += w
            A[i, S] -= w
    return A


def prior_covariance(p: int, scale: float = 1.0, lengthscale: float = 2.0) -> np.ndarray:
    """A kernel prior on coalition values: similarity by symmetric-difference
    size (coalitions with similar membership have correlated values)."""
    N = 1 << p
    K = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            d = bin(a ^ b).count("1")
            K[a, b] = scale * np.exp(-d / lengthscale)
    K += 1e-6 * np.eye(N)   # jitter
    return K


def posterior_cov(Sigma_theta: np.ndarray, observed: List[int], sigma_eps: float) -> np.ndarray:
    """Sigma(theta | y(D)) for observed coalitions D (list of masks), via
    Sigma^{-1} = Sigma_theta^{-1} + sigma_eps^{-2} sum_{S in D} e_S e_S^T."""
    N = Sigma_theta.shape[0]
    Prec = np.linalg.inv(Sigma_theta)
    for s in observed:
        e = np.zeros(N); e[s] = 1.0
        Prec += sigma_eps ** -2 * np.outer(e, e)
    return np.linalg.inv(Prec)


def eig_closed_form(A: np.ndarray, Sigma_theta: np.ndarray, observed: List[int],
                    candidate: int, sigma_eps: float) -> float:
    """EIG_phi(candidate S) (Eq. 3) = 1/2 [ log det(A Sigma_post A^T) -
    log det(A Sigma_{post + candidate} A^T) ]  (the entropy reduction of phi
    from observing S; the x-independent constant C cancels in the difference)."""
    Sigma_post = posterior_cov(Sigma_theta, observed, sigma_eps)
    Sigma_new = posterior_cov(Sigma_theta, observed + [candidate], sigma_eps)
    _, logdet_old = np.linalg.slogdet(A @ Sigma_post @ A.T)
    _, logdet_new = np.linalg.slogdet(A @ Sigma_new @ A.T)
    return 0.5 * (logdet_old - logdet_new)


def true_shapley(nu, p: int) -> np.ndarray:
    """Exact Shapley of a value function nu(mask)->float, by brute force."""
    from math import comb
    phi = np.zeros(p)
    for i in range(p):
        for mask in range(1 << p):
            if mask & (1 << i):
                continue
            s = bin(mask).count("1")
            w = 1.0 / (p * comb(p - 1, s))
            phi[i] += w * (nu(mask | (1 << i)) - nu(mask))
    return phi
