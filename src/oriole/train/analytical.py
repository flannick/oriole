from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..data import LoadedData
from ..params import Params


@dataclass
class AnalyticalMoments:
    e_mean: np.ndarray
    ee_mean: np.ndarray
    e_t_mean: np.ndarray
    t2_mean: np.ndarray


def analytical_moments_chunk(
    params: Params, betas_obs: np.ndarray, ses: np.ndarray
) -> AnalyticalMoments:
    beta = np.asarray(params.betas, dtype=float)
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    n_vars = betas_obs.shape[0]
    n_traits = params.n_traits()
    n_endos = params.n_endos()
    tau_inv = np.diag(1.0 / tau2)

    e_mean = np.zeros((n_vars, n_endos), dtype=float)
    ee_mean = np.zeros((n_vars, n_endos, n_endos), dtype=float)
    e_t_mean = np.zeros((n_vars, n_traits, n_endos), dtype=float)
    t2_mean = np.zeros((n_vars, n_traits), dtype=float)

    for i in range(n_vars):
        o = betas_obs[i]
        se2 = ses[i] ** 2
        v = sigma2 + se2
        w = np.diag(1.0 / v)
        c = np.linalg.inv(tau_inv + beta.T @ w @ beta)
        m = c @ (tau_inv @ mu + beta.T @ w @ o)
        m2 = c + np.outer(m, m)
        e_mean[i] = m
        ee_mean[i] = m2

        a = sigma2 / v
        b = se2 / v
        v_t = (sigma2 * se2) / v
        for i_trait in range(n_traits):
            b_i = beta[i_trait]
            e_t_mean[i, i_trait] = a[i_trait] * o[i_trait] * m + b[i_trait] * (m2 @ b_i)
            qf = float(b_i @ m2 @ b_i)
            t2_mean[i, i_trait] = (
                v_t[i_trait]
                + (a[i_trait] * o[i_trait]) ** 2
                + 2.0 * a[i_trait] * o[i_trait] * b[i_trait] * float(b_i @ m)
                + (b[i_trait] ** 2) * qf
            )

    return AnalyticalMoments(e_mean=e_mean, ee_mean=ee_mean, e_t_mean=e_t_mean, t2_mean=t2_mean)


def estimate_params_analytical(
    data: LoadedData, params: Params, chunk_size: int, mask: np.ndarray
) -> Params:
    gwas = data.gwas_data
    n = gwas.meta.n_data_points()
    weights = np.asarray(data.weights.weights, dtype=float)
    w_sum = float(data.weights.sum)

    n_traits = gwas.meta.n_traits()
    n_endos = params.n_endos()
    e_sum = np.zeros(n_endos, dtype=float)
    ee_sum = np.zeros((n_endos, n_endos), dtype=float)
    e_t_sum = np.zeros((n_traits, n_endos), dtype=float)
    t2_sum = np.zeros(n_traits, dtype=float)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        betas_obs = gwas.betas[start:end, :]
        ses = gwas.ses[start:end, :]
        w = weights[start:end]
        moments = analytical_moments_chunk(params, betas_obs, ses)
        e_sum += (w[:, None] * moments.e_mean).sum(axis=0)
        ee_sum += (w[:, None, None] * moments.ee_mean).sum(axis=0)
        e_t_sum += (w[:, None, None] * moments.e_t_mean).sum(axis=0)
        t2_sum += (w[:, None] * moments.t2_mean).sum(axis=0)

    mu = e_sum / w_sum
    diag_ee = np.diag(ee_sum)
    tau2 = (diag_ee - 2.0 * mu * e_sum + (mu ** 2) * w_sum) / w_sum
    tau2 = np.maximum(tau2, 0.0)

    betas = np.zeros((n_traits, n_endos), dtype=float)
    for i_trait in range(n_traits):
        active = np.asarray(mask[i_trait], dtype=bool)
        if not np.any(active):
            continue
        A_sub = ee_sum[np.ix_(active, active)]
        b_sub = e_t_sum[i_trait, active]
        betas_sub = np.linalg.solve(A_sub, b_sub)
        betas[i_trait, active] = betas_sub

    sigma2 = np.zeros(n_traits, dtype=float)
    for i_trait in range(n_traits):
        beta_row = betas[i_trait]
        sigma2[i_trait] = (
            t2_sum[i_trait]
            - 2.0 * float(beta_row @ e_t_sum[i_trait])
            + float(beta_row @ ee_sum @ beta_row)
        ) / w_sum
    sigma2 = np.maximum(sigma2, 0.0)
    sigmas = np.sqrt(sigma2)

    return Params(
        trait_names=gwas.meta.trait_names,
        endo_names=params.endo_names,
        mus=[float(value) for value in mu],
        taus=[float(value) for value in np.sqrt(tau2)],
        betas=betas.tolist(),
        sigmas=[float(value) for value in sigmas],
    )
