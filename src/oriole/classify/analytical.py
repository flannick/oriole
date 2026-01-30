from __future__ import annotations

import numpy as np

from ..params import Params
from ..sample.var_stats import SampledClassification


def _posterior_mean_cov(
    params: Params, betas: np.ndarray, ses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    B = np.asarray(params.betas, dtype=float)
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    v = sigma2 + ses**2
    W = np.diag(1.0 / v)
    tau_inv = np.diag(1.0 / tau2)
    C = np.linalg.inv(tau_inv + B.T @ W @ B)
    m = C @ (tau_inv @ mu + B.T @ W @ betas)
    return m, C


def analytical_classification(
    params: Params, betas: list[float], ses: list[float]
) -> SampledClassification:
    betas_arr = np.asarray(betas, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    m, C = _posterior_mean_cov(params, betas_arr, ses_arr)
    e_std = np.sqrt(np.diag(C))

    B = np.asarray(params.betas, dtype=float)
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    v = sigma2 + ses_arr**2
    a = sigma2 / v
    b = ses_arr**2 / v
    t_means = a * betas_arr + b * (B @ m)

    return SampledClassification(e_mean=m, e_std=e_std, t_means=t_means)


def analytical_classification_chunk(
    params: Params,
    betas_obs: np.ndarray,
    ses: np.ndarray,
) -> SampledClassification:
    n_vars = betas_obs.shape[0]
    n_endos = params.n_endos()
    n_traits = params.n_traits()
    e_mean = np.zeros((n_vars, n_endos), dtype=float)
    e_std = np.zeros((n_vars, n_endos), dtype=float)
    t_means = np.zeros((n_vars, n_traits), dtype=float)
    for i in range(n_vars):
        sampled = analytical_classification(
            params, betas_obs[i].tolist(), ses[i].tolist()
        )
        e_mean[i] = sampled.e_mean
        e_std[i] = sampled.e_std
        t_means[i] = sampled.t_means
    return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_means)


def calculate_mu_vec(params: Params, betas: list[float], ses: list[float]) -> np.ndarray:
    betas_arr = np.asarray(betas, dtype=float)
    ses_arr = np.asarray(ses, dtype=float)
    m, _ = _posterior_mean_cov(params, betas_arr, ses_arr)
    return m


def calculate_mu_chunk(params: Params, betas_obs: np.ndarray, ses: np.ndarray) -> np.ndarray:
    n_vars = betas_obs.shape[0]
    out = np.zeros((n_vars, params.n_endos()), dtype=float)
    for i in range(n_vars):
        out[i] = calculate_mu_vec(params, betas_obs[i].tolist(), ses[i].tolist())
    return out
