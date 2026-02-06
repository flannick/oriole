from __future__ import annotations

import itertools
import numpy as np

from ..params import Params
from ..sample.var_stats import SampledClassification
from ..outliers.moments import posterior_moments, log_marginal_observation
from .analytical import (
    _no_trait_edges,
    _solve_spd,
    gls_beta_se_from_Ib,
    gls_information_and_score,
    trait_matrices_from_sigma,
)


def _log_prior_z(z: np.ndarray, pis: np.ndarray) -> float:
    return float(np.sum(z * np.log(pis) + (1.0 - z) * np.log(1.0 - pis)))


def _enumerate_z(n_traits: int):
    for bits in itertools.product([0, 1], repeat=n_traits):
        yield np.asarray(bits, dtype=float)


def outliers_analytic_classification(
    params: Params,
    betas: np.ndarray,
    ses: np.ndarray,
) -> SampledClassification:
    betas_arr = np.asarray(betas, dtype=float)
    se2 = np.asarray(ses, dtype=float) ** 2
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    pis = np.asarray(params.outlier_pis, dtype=float)
    kappa2 = params.outlier_kappa ** 2

    log_weights = []
    moments = []
    l_inv = None
    for z in _enumerate_z(len(sigma2)):
        c = np.where(z > 0.5, kappa2, 1.0)
        sigma2_z = sigma2 * c
        log_w = _log_prior_z(z, pis) + log_marginal_observation(
            params, betas_arr, se2, sigma2_z, l_inv=l_inv
        )
        m, ee, _, tt, t_mean, l_inv = posterior_moments(
            params, betas_arr, se2, sigma2_z, l_inv=l_inv
        )
        moments.append((m, ee, t_mean))
        log_weights.append(log_w)

    log_w = np.asarray(log_weights, dtype=float)
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    weights /= np.sum(weights)

    e_mean = np.zeros(params.n_endos(), dtype=float)
    e_second = np.zeros((params.n_endos(), params.n_endos()), dtype=float)
    t_mean = np.zeros(params.n_traits(), dtype=float)
    for weight, (m, ee, t_mean_z) in zip(weights, moments):
        e_mean += weight * m
        e_second += weight * ee
        t_mean += weight * t_mean_z

    e_var = np.diag(e_second - np.outer(e_mean, e_mean))
    e_std = np.sqrt(np.maximum(e_var, 0.0))
    return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_mean)


def outliers_analytic_gls(
    params: Params,
    betas: np.ndarray,
    ses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    betas_arr = np.asarray(betas, dtype=float)
    se2 = np.asarray(ses, dtype=float) ** 2
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    pis = np.asarray(params.outlier_pis, dtype=float)
    kappa2 = params.outlier_kappa ** 2

    log_weights = []
    i_blocks = []
    b_blocks = []
    l_inv = None
    for z in _enumerate_z(len(sigma2)):
        c = np.where(z > 0.5, kappa2, 1.0)
        sigma2_z = sigma2 * c
        log_w = _log_prior_z(z, pis) + log_marginal_observation(
            params, betas_arr, se2, sigma2_z, l_inv=l_inv
        )
        if _no_trait_edges(params):
            b_mat = np.asarray(params.betas, dtype=float)
            v = sigma2_z + se2
            weights = 1.0 / v
            i_mat = b_mat.T @ (b_mat * weights[:, None])
            b_vec = b_mat.T @ (betas_arr * weights)
        else:
            m_mat, sigma_t = trait_matrices_from_sigma(params, sigma2_z)
            v = sigma_t + np.diag(se2)
            solve_v = lambda rhs: _solve_spd(v, rhs)
            i_mat, b_vec = gls_information_and_score(m_mat, solve_v, betas_arr)
        i_blocks.append(i_mat)
        b_blocks.append(b_vec)
        log_weights.append(log_w)

    log_w = np.asarray(log_weights, dtype=float)
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    weights /= np.sum(weights)

    i_bar = np.zeros_like(i_blocks[0])
    b_bar = np.zeros_like(b_blocks[0])
    for weight, i_mat, b_vec in zip(weights, i_blocks, b_blocks):
        i_bar += weight * i_mat
        b_bar += weight * b_vec

    e_hat, se, _ = gls_beta_se_from_Ib(i_bar, b_bar)
    return e_hat, se


def outliers_analytic_classification_chunk(
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
        sampled = outliers_analytic_classification(params, betas_obs[i], ses[i])
        e_mean[i] = sampled.e_mean
        e_std[i] = sampled.e_std
        t_means[i] = sampled.t_means
    return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_means)


def outliers_analytic_gls_chunk(
    params: Params,
    betas_obs: np.ndarray,
    ses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_vars = betas_obs.shape[0]
    n_endos = params.n_endos()
    e_beta_gls = np.zeros((n_vars, n_endos), dtype=float)
    e_se_gls = np.zeros((n_vars, n_endos), dtype=float)
    for i in range(n_vars):
        e_hat, se = outliers_analytic_gls(params, betas_obs[i], ses[i])
        e_beta_gls[i] = e_hat
        e_se_gls[i] = se
    return e_beta_gls, e_se_gls
