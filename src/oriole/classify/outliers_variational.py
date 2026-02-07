from __future__ import annotations

import numpy as np

from ..params import Params
from ..sample.var_stats import SampledClassification
from ..outliers.variational import variational_posterior, _no_trait_edges, _variational_posterior_chunk_no_edges
from .analytical import gls_endophenotype_stats_chunk, gls_beta_se_from_Ib


def outliers_variational_classification(
    params: Params,
    betas: np.ndarray,
    ses: np.ndarray,
) -> SampledClassification:
    m, ee, _, _, t_mean, _, _ = variational_posterior(params, betas, ses)
    e_mean = m
    e_var = np.diag(ee - np.outer(e_mean, e_mean))
    e_std = np.sqrt(np.maximum(e_var, 0.0))
    return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_mean)


def outliers_variational_classification_chunk(
    params: Params,
    betas_obs: np.ndarray,
    ses: np.ndarray,
) -> SampledClassification:
    n_vars = betas_obs.shape[0]
    n_endos = params.n_endos()
    n_traits = params.n_traits()
    if _no_trait_edges(params):
        m, e_std, t_means, _ = _variational_posterior_chunk_no_edges(
            params, betas_obs, ses
        )
        return SampledClassification(e_mean=m, e_std=e_std, t_means=t_means)
    e_mean = np.zeros((n_vars, n_endos), dtype=float)
    e_std = np.zeros((n_vars, n_endos), dtype=float)
    t_means = np.zeros((n_vars, n_traits), dtype=float)
    for i in range(n_vars):
        sampled = outliers_variational_classification(params, betas_obs[i], ses[i])
        e_mean[i] = sampled.e_mean
        e_std[i] = sampled.e_std
        t_means[i] = sampled.t_means
    return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_means)


def outliers_variational_gls(
    params: Params,
    betas: np.ndarray,
    ses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, _, _, _, alpha = variational_posterior(params, betas, ses)
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    sigma2_eff = sigma2 / alpha
    e_beta, e_se = gls_endophenotype_stats_chunk(
        params, np.asarray([betas], dtype=float), np.asarray([ses], dtype=float), sigma2_eff
    )
    return e_beta[0], e_se[0]


def outliers_variational_gls_chunk(
    params: Params,
    betas_obs: np.ndarray,
    ses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_vars = betas_obs.shape[0]
    n_endos = params.n_endos()
    e_beta_gls = np.zeros((n_vars, n_endos), dtype=float)
    e_se_gls = np.zeros((n_vars, n_endos), dtype=float)
    if _no_trait_edges(params):
        _, _, _, alpha = _variational_posterior_chunk_no_edges(
            params, betas_obs, ses
        )
        sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
        sigma2_eff = sigma2[None, :] / alpha
        b_mat = np.asarray(params.betas, dtype=float)
        for i in range(n_vars):
            v = sigma2_eff[i] + ses[i] ** 2
            weights = 1.0 / v
            i_mat = b_mat.T @ (b_mat * weights[:, None])
            b_vec = b_mat.T @ (betas_obs[i] * weights)
            e_hat, se, _ = gls_beta_se_from_Ib(i_mat, b_vec)
            e_beta_gls[i] = e_hat
            e_se_gls[i] = se
        return e_beta_gls, e_se_gls
    for i in range(n_vars):
        e_hat, se = outliers_variational_gls(params, betas_obs[i], ses[i])
        e_beta_gls[i] = e_hat
        e_se_gls[i] = se
    return e_beta_gls, e_se_gls
