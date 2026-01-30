from __future__ import annotations

import numpy as np

from ..params import Params
from ..sample.var_stats import SampledClassification


def gibbs_classification_chunk(
    params: Params,
    betas_obs: np.ndarray,
    ses: np.ndarray,
    n_burn_in: int,
    n_samples: int,
    t_pinned: bool,
) -> SampledClassification:
    rng = np.random.default_rng()
    beta = np.asarray(params.betas, dtype=float)
    sigma = np.asarray(params.sigmas, dtype=float)
    sigma2 = sigma ** 2
    tau2 = params.tau ** 2
    n_vars, n_traits = betas_obs.shape

    e = np.full(n_vars, params.mu, dtype=float)
    t = e[:, None] * beta[None, :]

    inv_var_e = (1.0 / tau2) + np.sum((beta / sigma) ** 2)
    var_e = 1.0 / inv_var_e
    std_e = np.sqrt(var_e)

    se2 = ses ** 2
    var_t = 1.0 / (1.0 / sigma2[None, :] + 1.0 / se2)
    std_t = np.sqrt(var_t)

    def update_e():
        mean_e = var_e * (
            (params.mu / tau2) + np.sum(beta[None, :] * t / sigma2[None, :], axis=1)
        )
        return rng.normal(loc=mean_e, scale=std_e)

    def update_t(e_vals):
        if t_pinned:
            return betas_obs.copy()
        mean_t = var_t * (
            (beta[None, :] * e_vals[:, None] / sigma2[None, :])
            + (betas_obs / se2)
        )
        return rng.normal(loc=mean_t, scale=std_t)

    for _ in range(n_burn_in):
        e = update_e()
        t = update_t(e)

    e_sum = np.zeros(n_vars, dtype=float)
    e2_sum = np.zeros(n_vars, dtype=float)
    t_sum = np.zeros((n_vars, n_traits), dtype=float)

    for _ in range(n_samples):
        e = update_e()
        t = update_t(e)
        e_sum += e
        e2_sum += e ** 2
        t_sum += t

    e_mean = e_sum / n_samples
    e_var = (e2_sum / n_samples) - e_mean ** 2
    e_std = np.sqrt(np.maximum(e_var, 0.0))
    t_means = t_sum / n_samples
    return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_means)

