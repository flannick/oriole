from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..data import LoadedData
from ..params import Params


@dataclass
class AnalyticalSums:
    e_sum: np.ndarray
    ee_sum: np.ndarray
    te_sum: np.ndarray
    tt_sum: np.ndarray
    weight_sum: float


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


def analytical_moments_chunk(
    params: Params, betas_obs: np.ndarray, ses: np.ndarray, weights: np.ndarray
) -> AnalyticalSums:
    m_mat, sigma_t = _trait_matrices(params)
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    n_vars = betas_obs.shape[0]
    n_traits = params.n_traits()
    n_endos = params.n_endos()
    tau_inv = np.diag(1.0 / tau2)

    e_sum = np.zeros(n_endos, dtype=float)
    ee_sum = np.zeros((n_endos, n_endos), dtype=float)
    te_sum = np.zeros((n_traits, n_endos), dtype=float)
    tt_sum = np.zeros((n_traits, n_traits), dtype=float)
    weight_sum = float(np.sum(weights))

    for i in range(n_vars):
        w = float(weights[i])
        if w <= 0.0:
            continue
        o = betas_obs[i]
        v = sigma_t + np.diag(ses[i] ** 2)
        v_inv_m = np.linalg.solve(v, m_mat)
        c_inv = tau_inv + m_mat.T @ v_inv_m
        c = np.linalg.inv(c_inv)
        v_inv_o = np.linalg.solve(v, o)
        m = c @ (tau_inv @ mu + m_mat.T @ v_inv_o)

        mu_t = m_mat @ m
        v_inv_r = np.linalg.solve(v, o - mu_t)
        t_mean = mu_t + sigma_t @ v_inv_r

        h = m_mat - sigma_t @ v_inv_m
        v_inv_sigma = np.linalg.solve(v, sigma_t)
        cov_t = sigma_t - sigma_t @ v_inv_sigma + h @ c @ h.T

        ee = c + np.outer(m, m)
        te = (t_mean[:, None] @ m[None, :]) + h @ c
        tt = cov_t + np.outer(t_mean, t_mean)

        e_sum += w * m
        ee_sum += w * ee
        te_sum += w * te
        tt_sum += w * tt

    return AnalyticalSums(
        e_sum=e_sum,
        ee_sum=ee_sum,
        te_sum=te_sum,
        tt_sum=tt_sum,
        weight_sum=weight_sum,
    )


def estimate_params_analytical(
    data: LoadedData,
    params: Params,
    chunk_size: int,
    mask: np.ndarray,
    parent_mask: np.ndarray,
) -> Params:
    gwas = data.gwas_data
    n = gwas.meta.n_data_points()
    weights = np.asarray(data.weights.weights, dtype=float)

    n_traits = gwas.meta.n_traits()
    n_endos = params.n_endos()
    e_sum = np.zeros(n_endos, dtype=float)
    ee_sum = np.zeros((n_endos, n_endos), dtype=float)
    te_sum = np.zeros((n_traits, n_endos), dtype=float)
    tt_sum = np.zeros((n_traits, n_traits), dtype=float)
    w_sum = 0.0

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        betas_obs = gwas.betas[start:end, :]
        ses = gwas.ses[start:end, :]
        w = weights[start:end]
        moments = analytical_moments_chunk(params, betas_obs, ses, w)
        e_sum += moments.e_sum
        ee_sum += moments.ee_sum
        te_sum += moments.te_sum
        tt_sum += moments.tt_sum
        w_sum += moments.weight_sum

    mu = e_sum / w_sum
    diag_ee = np.diag(ee_sum)
    tau2 = (diag_ee - 2.0 * mu * e_sum + (mu**2) * w_sum) / w_sum
    tau2 = np.maximum(tau2, 0.0)

    betas = np.zeros((n_traits, n_endos), dtype=float)
    trait_edges = np.zeros((n_traits, n_traits), dtype=float)
    sigma2 = np.zeros(n_traits, dtype=float)

    for i_trait in range(n_traits):
        active_endos = [i for i, allowed in enumerate(mask[i_trait]) if allowed]
        parents = [i for i, allowed in enumerate(parent_mask[i_trait]) if allowed]
        if not active_endos and not parents:
            sigma2[i_trait] = tt_sum[i_trait, i_trait] / w_sum
            continue
        n_endos_active = len(active_endos)
        n_parents = len(parents)
        n_pred = n_endos_active + n_parents
        zz = np.zeros((n_pred, n_pred), dtype=float)
        tz = np.zeros(n_pred, dtype=float)
        if active_endos:
            zz[:n_endos_active, :n_endos_active] = ee_sum[np.ix_(active_endos, active_endos)]
            tz[:n_endos_active] = te_sum[i_trait, active_endos]
        if parents:
            tt_block = tt_sum[np.ix_(parents, parents)]
            zz[n_endos_active:, n_endos_active:] = tt_block
            tz[n_endos_active:] = tt_sum[i_trait, parents]
            if active_endos:
                te_block = te_sum[parents][:, active_endos]
                zz[n_endos_active:, :n_endos_active] = te_block
                zz[:n_endos_active, n_endos_active:] = te_block.T

        coeffs = np.linalg.solve(zz, tz)
        for i, k in enumerate(active_endos):
            betas[i_trait, k] = coeffs[i]
        for i, p in enumerate(parents):
            trait_edges[i_trait, p] = coeffs[n_endos_active + i]

        t2 = tt_sum[i_trait, i_trait]
        sigma2[i_trait] = (t2 - 2.0 * float(coeffs @ tz) + float(coeffs @ zz @ coeffs)) / w_sum

    sigma2 = np.maximum(sigma2, 0.0)
    sigmas = np.sqrt(sigma2)

    return Params(
        trait_names=gwas.meta.trait_names,
        endo_names=params.endo_names,
        mus=[float(value) for value in mu],
        taus=[float(value) for value in np.sqrt(tau2)],
        betas=betas.tolist(),
        sigmas=[float(value) for value in sigmas],
        trait_edges=trait_edges.tolist(),
    )
