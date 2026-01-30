from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..data import LoadedData
from ..params import Params


@dataclass
class AnalyticalMoments:
    e_mean: np.ndarray
    e2_mean: np.ndarray
    e_t_mean: np.ndarray
    t2_mean: np.ndarray


def analytical_moments_chunk(params: Params, betas_obs: np.ndarray, ses: np.ndarray) -> AnalyticalMoments:
    beta = np.asarray(params.betas, dtype=float)
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    tau2 = params.tau ** 2
    se2 = ses ** 2
    v = sigma2[None, :] + se2

    inv_var = (1.0 / tau2) + np.sum((beta[None, :] ** 2) / v, axis=1)
    v_e = 1.0 / inv_var
    mean_term = (params.mu / tau2) + np.sum(beta[None, :] * betas_obs / v, axis=1)
    m_e = v_e * mean_term
    e2 = v_e + m_e ** 2

    a = sigma2[None, :] / v
    b = (se2 / v) * beta[None, :]
    v_t = (sigma2[None, :] * se2) / v

    e_t = a * betas_obs * m_e[:, None] + b * e2[:, None]
    t2 = v_t + (a * betas_obs) ** 2 + 2 * a * betas_obs * b * m_e[:, None] + (b ** 2) * e2[:, None]

    return AnalyticalMoments(e_mean=m_e, e2_mean=e2, e_t_mean=e_t, t2_mean=t2)


def estimate_params_analytical(data: LoadedData, params: Params, chunk_size: int) -> Params:
    gwas = data.gwas_data
    n = gwas.meta.n_data_points()
    weights = np.asarray(data.weights.weights, dtype=float)
    w_sum = float(data.weights.sum)

    e_sum = 0.0
    e2_sum = 0.0
    e_t_sum = np.zeros(gwas.meta.n_traits(), dtype=float)
    t2_sum = np.zeros(gwas.meta.n_traits(), dtype=float)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        betas_obs = gwas.betas[start:end, :]
        ses = gwas.ses[start:end, :]
        w = weights[start:end]
        moments = analytical_moments_chunk(params, betas_obs, ses)
        e_sum += float((w * moments.e_mean).sum())
        e2_sum += float((w * moments.e2_mean).sum())
        e_t_sum += (w[:, None] * moments.e_t_mean).sum(axis=0)
        t2_sum += (w[:, None] * moments.t2_mean).sum(axis=0)

    mu = e_sum / w_sum
    tau2 = (e2_sum - 2.0 * mu * e_sum + (mu ** 2) * w_sum) / w_sum
    tau = float(np.sqrt(max(tau2, 0.0)))

    betas = (e_t_sum / e2_sum).tolist()

    betas_arr = np.asarray(betas, dtype=float)
    sigma2 = (t2_sum - 2.0 * betas_arr * e_t_sum + (betas_arr ** 2) * e2_sum) / w_sum
    sigmas = np.sqrt(np.maximum(sigma2, 0.0)).tolist()

    return Params(
        trait_names=gwas.meta.trait_names,
        mu=mu,
        tau=tau,
        betas=betas,
        sigmas=sigmas,
    )

