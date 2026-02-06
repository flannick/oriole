from __future__ import annotations

import numpy as np

from ..params import Params
from ..sample.var_stats import SampledClassification


def _no_trait_edges(params: Params) -> bool:
    return not np.any(np.asarray(params.trait_edges, dtype=float))


def _solve_spd(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(mat)
    y = np.linalg.solve(chol, rhs)
    return np.linalg.solve(chol.T, y)


def trait_matrices_from_sigma(params: Params, sigma2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta = np.asarray(params.betas, dtype=float)
    sigma2_arr = np.asarray(sigma2, dtype=float)
    trait_edges = np.asarray(params.trait_edges, dtype=float)
    n_traits = len(sigma2_arr)
    l_mat = np.eye(n_traits, dtype=float) - trait_edges
    m_mat = np.linalg.solve(l_mat, beta)
    d_mat = np.diag(sigma2_arr)
    l_inv = np.linalg.solve(l_mat, np.eye(n_traits))
    sigma_t = l_inv @ d_mat @ l_inv.T
    return m_mat, sigma_t


def _trait_matrices(params: Params) -> tuple[np.ndarray, np.ndarray]:
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    return trait_matrices_from_sigma(params, sigma2)


def gls_information_and_score(
    m_mat: np.ndarray, solve_v, o: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    v_inv_m = solve_v(m_mat)
    i_mat = m_mat.T @ v_inv_m
    v_inv_o = solve_v(o)
    b_vec = m_mat.T @ v_inv_o
    return i_mat, b_vec


def gls_beta_se_from_Ib(
    i_mat: np.ndarray, b_vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        chol = np.linalg.cholesky(i_mat)
        y = np.linalg.solve(chol, b_vec)
        e_hat = np.linalg.solve(chol.T, y)
        cov = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(i_mat.shape[0])))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(i_mat)
        e_hat = cov @ b_vec
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return e_hat, se, cov


def gls_endophenotype_stats_chunk(
    params: Params,
    betas_obs: np.ndarray,
    ses_obs: np.ndarray,
    sigma2_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_vars = betas_obs.shape[0]
    n_endos = params.n_endos()
    e_beta_gls = np.zeros((n_vars, n_endos), dtype=float)
    e_se_gls = np.zeros((n_vars, n_endos), dtype=float)
    sigma2 = (
        np.asarray(sigma2_override, dtype=float)
        if sigma2_override is not None
        else np.asarray(params.sigmas, dtype=float) ** 2
    )
    if _no_trait_edges(params):
        b_mat = np.asarray(params.betas, dtype=float)
        for i in range(n_vars):
            o = betas_obs[i]
            v = sigma2 + ses_obs[i] ** 2
            weights = 1.0 / v
            i_mat = b_mat.T @ (b_mat * weights[:, None])
            b_vec = b_mat.T @ (o * weights)
            e_hat, se, _ = gls_beta_se_from_Ib(i_mat, b_vec)
            e_beta_gls[i] = e_hat
            e_se_gls[i] = se
        return e_beta_gls, e_se_gls
    m_mat, sigma_t = trait_matrices_from_sigma(params, sigma2)
    for i in range(n_vars):
        o = betas_obs[i]
        v = sigma_t + np.diag(ses_obs[i] ** 2)
        solve_v = lambda rhs: _solve_spd(v, rhs)
        i_mat, b_vec = gls_information_and_score(m_mat, solve_v, o)
        e_hat, se, _ = gls_beta_se_from_Ib(i_mat, b_vec)
        e_beta_gls[i] = e_hat
        e_se_gls[i] = se
    return e_beta_gls, e_se_gls


def _posterior_mean_cov(
    params: Params, betas: np.ndarray, ses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    tau_inv = np.diag(1.0 / tau2)
    if _no_trait_edges(params):
        B = np.asarray(params.betas, dtype=float)
        sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
        v = sigma2 + ses**2
        W = np.diag(1.0 / v)
        C = np.linalg.inv(tau_inv + B.T @ W @ B)
        m = C @ (tau_inv @ mu + B.T @ W @ betas)
        return m, C
    M, sigma_t = _trait_matrices(params)
    v = sigma_t + np.diag(ses**2)
    v_inv_m = _solve_spd(v, M)
    c_inv = tau_inv + M.T @ v_inv_m
    C = np.linalg.inv(c_inv)
    v_inv_o = _solve_spd(v, betas)
    m = C @ (tau_inv @ mu + M.T @ v_inv_o)
    return m, C


def analytical_classification(
    params: Params, betas: list[float], ses: list[float]
) -> SampledClassification:
    betas_arr = np.asarray(betas, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    m, C = _posterior_mean_cov(params, betas_arr, ses_arr)
    e_std = np.sqrt(np.diag(C))

    if _no_trait_edges(params):
        B = np.asarray(params.betas, dtype=float)
        sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
        v = sigma2 + ses_arr**2
        a = sigma2 / v
        b = ses_arr**2 / v
        t_means = a * betas_arr + b * (B @ m)
    else:
        M, sigma_t = _trait_matrices(params)
        v = sigma_t + np.diag(ses_arr**2)
        mu_t = M @ m
        v_inv_r = _solve_spd(v, betas_arr - mu_t)
        t_means = mu_t + sigma_t @ v_inv_r

    return SampledClassification(e_mean=m, e_std=e_std, t_means=t_means)


def analytical_classification_chunk(
    params: Params,
    betas_obs: np.ndarray,
    ses: np.ndarray,
) -> SampledClassification:
    n_vars = betas_obs.shape[0]
    n_endos = params.n_endos()
    n_traits = params.n_traits()
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    tau_inv = np.diag(1.0 / tau2)
    e_mean = np.zeros((n_vars, n_endos), dtype=float)
    e_std = np.zeros((n_vars, n_endos), dtype=float)
    t_means = np.zeros((n_vars, n_traits), dtype=float)
    if _no_trait_edges(params):
        B = np.asarray(params.betas, dtype=float)
        sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
        for i in range(n_vars):
            o = betas_obs[i]
            se2 = ses[i] ** 2
            v = sigma2 + se2
            w = np.diag(1.0 / v)
            c = np.linalg.inv(tau_inv + B.T @ w @ B)
            m = c @ (tau_inv @ mu + B.T @ w @ o)
            e_mean[i] = m
            e_std[i] = np.sqrt(np.diag(c))
            a = sigma2 / v
            b = se2 / v
            t_means[i] = a * o + b * (B @ m)
        return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_means)
    M, sigma_t = _trait_matrices(params)
    for i in range(n_vars):
        o = betas_obs[i]
        v = sigma_t + np.diag(ses[i] ** 2)
        v_inv_m = _solve_spd(v, M)
        c_inv = tau_inv + M.T @ v_inv_m
        c = np.linalg.inv(c_inv)
        v_inv_o = _solve_spd(v, o)
        m = c @ (tau_inv @ mu + M.T @ v_inv_o)
        mu_t = M @ m
        v_inv_r = _solve_spd(v, o - mu_t)
        e_mean[i] = m
        e_std[i] = np.sqrt(np.diag(c))
        t_means[i] = mu_t + sigma_t @ v_inv_r
    return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_means)


def calculate_mu_vec(params: Params, betas: list[float], ses: list[float]) -> np.ndarray:
    betas_arr = np.asarray(betas, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    m, _ = _posterior_mean_cov(params, betas_arr, ses_arr)
    return m


def calculate_mu_chunk(params: Params, betas_obs: np.ndarray, ses: np.ndarray) -> np.ndarray:
    n_vars = betas_obs.shape[0]
    out = np.zeros((n_vars, params.n_endos()), dtype=float)
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    tau_inv = np.diag(1.0 / tau2)
    if _no_trait_edges(params):
        B = np.asarray(params.betas, dtype=float)
        sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
        for i in range(n_vars):
            o = betas_obs[i]
            se2 = ses[i] ** 2
            v = sigma2 + se2
            w = np.diag(1.0 / v)
            c = np.linalg.inv(tau_inv + B.T @ w @ B)
            m = c @ (tau_inv @ mu + B.T @ w @ o)
            out[i] = m
        return out
    M, sigma_t = _trait_matrices(params)
    for i in range(n_vars):
        o = betas_obs[i]
        v = sigma_t + np.diag(ses[i] ** 2)
        v_inv_m = _solve_spd(v, M)
        c_inv = tau_inv + M.T @ v_inv_m
        c = np.linalg.inv(c_inv)
        v_inv_o = _solve_spd(v, o)
        m = c @ (tau_inv @ mu + M.T @ v_inv_o)
        out[i] = m
    return out
