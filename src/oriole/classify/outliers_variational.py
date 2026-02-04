from __future__ import annotations

import numpy as np

from ..params import Params
from ..sample.var_stats import SampledClassification
from ..outliers.variational import variational_posterior


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
    e_mean = np.zeros((n_vars, n_endos), dtype=float)
    e_std = np.zeros((n_vars, n_endos), dtype=float)
    t_means = np.zeros((n_vars, n_traits), dtype=float)
    for i in range(n_vars):
        sampled = outliers_variational_classification(params, betas_obs[i], ses[i])
        e_mean[i] = sampled.e_mean
        e_std[i] = sampled.e_std
        t_means[i] = sampled.t_means
    return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_means)
