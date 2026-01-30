from __future__ import annotations

import numpy as np

from ..params import Params
from ..sample.var_stats import SampledClassification


def _trait_matrices(params: Params) -> tuple[np.ndarray, np.ndarray]:
    beta = np.asarray(params.betas, dtype=float)
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    trait_edges = np.asarray(params.trait_edges, dtype=float)
    n_traits = len(sigma2)
    l_mat = np.eye(n_traits, dtype=float) - trait_edges
    m_mat = np.linalg.solve(l_mat, beta)
    d_mat = np.diag(sigma2)
    l_inv = np.linalg.solve(l_mat, np.eye(n_traits))
    sigma_t = l_inv @ d_mat @ l_inv.T
    return m_mat, sigma_t


def _posterior_mean_cov(
    params: Params, betas: np.ndarray, ses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    M, sigma_t = _trait_matrices(params)
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    v = sigma_t + np.diag(ses**2)
    tau_inv = np.diag(1.0 / tau2)
    v_inv_m = np.linalg.solve(v, M)
    c_inv = tau_inv + M.T @ v_inv_m
    C = np.linalg.inv(c_inv)
    v_inv_o = np.linalg.solve(v, betas)
    m = C @ (tau_inv @ mu + M.T @ v_inv_o)
    return m, C


def analytical_classification(
    params: Params, betas: list[float], ses: list[float]
) -> SampledClassification:
    betas_arr = np.asarray(betas, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    m, C = _posterior_mean_cov(params, betas_arr, ses_arr)
    e_std = np.sqrt(np.diag(C))

    M, sigma_t = _trait_matrices(params)
    v = sigma_t + np.diag(ses_arr**2)
    mu_t = M @ m
    v_inv_r = np.linalg.solve(v, betas_arr - mu_t)
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
    M, sigma_t = _trait_matrices(params)
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    tau_inv = np.diag(1.0 / tau2)
    e_mean = np.zeros((n_vars, n_endos), dtype=float)
    e_std = np.zeros((n_vars, n_endos), dtype=float)
    t_means = np.zeros((n_vars, n_traits), dtype=float)
    for i in range(n_vars):
        o = betas_obs[i]
        v = sigma_t + np.diag(ses[i] ** 2)
        v_inv_m = np.linalg.solve(v, M)
        c_inv = tau_inv + M.T @ v_inv_m
        c = np.linalg.inv(c_inv)
        v_inv_o = np.linalg.solve(v, o)
        m = c @ (tau_inv @ mu + M.T @ v_inv_o)
        mu_t = M @ m
        v_inv_r = np.linalg.solve(v, o - mu_t)
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
    M, sigma_t = _trait_matrices(params)
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    tau_inv = np.diag(1.0 / tau2)
    for i in range(n_vars):
        o = betas_obs[i]
        v = sigma_t + np.diag(ses[i] ** 2)
        v_inv_m = np.linalg.solve(v, M)
        c_inv = tau_inv + M.T @ v_inv_m
        c = np.linalg.inv(c_inv)
        v_inv_o = np.linalg.solve(v, o)
        m = c @ (tau_inv @ mu + M.T @ v_inv_o)
        out[i] = m
    return out
