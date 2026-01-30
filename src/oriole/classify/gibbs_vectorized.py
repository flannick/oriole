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
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    n_vars, n_traits = betas_obs.shape
    n_endos = params.n_endos()

    e = np.tile(mu[None, :], (n_vars, 1))
    t = e @ beta.T

    sigma2_inv = 1.0 / sigma2
    tau2_inv = 1.0 / tau2
    c_inv = np.diag(tau2_inv) + beta.T @ (sigma2_inv[:, None] * beta)
    c = np.linalg.inv(c_inv)
    chol_c = np.linalg.cholesky(c)

    se2 = ses ** 2
    var_t = 1.0 / (1.0 / sigma2[None, :] + 1.0 / se2)
    std_t = np.sqrt(var_t)

    def update_e():
        rhs = (t / sigma2[None, :]) @ beta + (mu * tau2_inv)[None, :]
        mean_e = rhs @ c
        z = rng.normal(size=(n_vars, n_endos))
        return mean_e + z @ chol_c.T

    def update_t(e_vals):
        if t_pinned:
            return betas_obs.copy()
        mu_e = e_vals @ beta.T
        mean_t = var_t * ((mu_e / sigma2[None, :]) + (betas_obs / se2))
        return rng.normal(loc=mean_t, scale=std_t)

    for _ in range(n_burn_in):
        e = update_e()
        t = update_t(e)

    e_sum = np.zeros((n_vars, n_endos), dtype=float)
    e2_sum = np.zeros((n_vars, n_endos), dtype=float)
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
