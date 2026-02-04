from __future__ import annotations

import numpy as np

from ..params import Params


def _solve_spd(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(mat)
    y = np.linalg.solve(chol, rhs)
    return np.linalg.solve(chol.T, y)


def _trait_matrices(
    params: Params,
    sigma2: np.ndarray,
    l_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = np.asarray(params.betas, dtype=float)
    trait_edges = np.asarray(params.trait_edges, dtype=float)
    n_traits = len(sigma2)
    if l_inv is None:
        l_mat = np.eye(n_traits, dtype=float) - trait_edges
        l_inv = np.linalg.solve(l_mat, np.eye(n_traits))
    m_mat = l_inv @ beta
    d_mat = np.diag(sigma2)
    sigma_t = l_inv @ d_mat @ l_inv.T
    return m_mat, sigma_t, l_inv


def posterior_moments(
    params: Params,
    betas_obs: np.ndarray,
    se2: np.ndarray,
    sigma2: np.ndarray,
    l_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    tau_inv = np.diag(1.0 / tau2)
    m_mat, sigma_t, l_inv = _trait_matrices(params, sigma2, l_inv=l_inv)

    v = sigma_t + np.diag(se2)
    v_inv_m = _solve_spd(v, m_mat)
    c_inv = tau_inv + m_mat.T @ v_inv_m
    c = np.linalg.inv(c_inv)
    v_inv_o = _solve_spd(v, betas_obs)
    m = c @ (tau_inv @ mu + m_mat.T @ v_inv_o)

    mu_t = m_mat @ m
    v_inv_r = _solve_spd(v, betas_obs - mu_t)
    t_mean = mu_t + sigma_t @ v_inv_r

    h = m_mat - sigma_t @ v_inv_m
    v_inv_sigma = _solve_spd(v, sigma_t)
    cov_t = sigma_t - sigma_t @ v_inv_sigma + h @ c @ h.T

    ee = c + np.outer(m, m)
    te = (t_mean[:, None] @ m[None, :]) + h @ c
    tt = cov_t + np.outer(t_mean, t_mean)
    return m, ee, te, tt, t_mean, l_inv


def log_marginal_observation(
    params: Params,
    betas_obs: np.ndarray,
    se2: np.ndarray,
    sigma2: np.ndarray,
    l_inv: np.ndarray | None = None,
) -> float:
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    sigma_e = np.diag(tau2)
    m_mat, sigma_t, l_inv = _trait_matrices(params, sigma2, l_inv=l_inv)
    v = sigma_t + np.diag(se2)
    cov = v + m_mat @ sigma_e @ m_mat.T
    diff = betas_obs - (m_mat @ mu)
    chol = np.linalg.cholesky(cov)
    sol = np.linalg.solve(chol, diff)
    quad = float(sol.T @ sol)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    n = len(diff)
    return -0.5 * (quad + logdet + n * np.log(2.0 * np.pi))
