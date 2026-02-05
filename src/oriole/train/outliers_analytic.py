from __future__ import annotations

import itertools
import numpy as np

from ..data import LoadedData
from ..params import Params
from ..outliers.moments import posterior_moments, log_marginal_observation


def _log_prior_z(z: np.ndarray, pis: np.ndarray) -> float:
    return float(np.sum(z * np.log(pis) + (1.0 - z) * np.log(1.0 - pis)))


def _enumerate_z(n_traits: int):
    for bits in itertools.product([0, 1], repeat=n_traits):
        yield np.asarray(bits, dtype=float)


def _analytic_variant_moments(
    params: Params,
    betas_obs: np.ndarray,
    ses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    se2 = np.asarray(ses, dtype=float) ** 2
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    pis = np.asarray(params.outlier_pis, dtype=float)
    kappa2 = params.outlier_kappa ** 2
    n_traits = len(sigma2)
    n_endos = params.n_endos()

    log_weights = []
    moments = []
    l_inv = None
    for z in _enumerate_z(n_traits):
        c = np.where(z > 0.5, kappa2, 1.0)
        sigma2_z = sigma2 * c
        log_w = _log_prior_z(z, pis) + log_marginal_observation(
            params, betas_obs, se2, sigma2_z, l_inv=l_inv
        )
        m, ee, te, tt, _, l_inv = posterior_moments(
            params, betas_obs, se2, sigma2_z, l_inv=l_inv
        )
        moments.append((m, ee, te, tt, c))
        log_weights.append(log_w)

    log_w = np.asarray(log_weights, dtype=float)
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    weights /= np.sum(weights)

    e_mean = np.zeros(params.n_endos(), dtype=float)
    ee_mix = np.zeros((n_endos, n_endos), dtype=float)
    te_mix = np.zeros((n_traits, n_endos), dtype=float)
    tt_mix = np.zeros((n_traits, n_traits), dtype=float)

    ee_w = np.zeros((n_traits, n_endos, n_endos), dtype=float)
    te_w = np.zeros((n_traits, n_traits, n_endos), dtype=float)
    tt_w = np.zeros((n_traits, n_traits, n_traits), dtype=float)
    t2_w = np.zeros(n_traits, dtype=float)
    alpha = np.zeros(n_traits, dtype=float)

    for weight, (m, ee, te, tt, c) in zip(weights, moments):
        e_mean += weight * m
        ee_mix += weight * ee
        te_mix += weight * te
        tt_mix += weight * tt

        inv_c = 1.0 / c
        alpha += weight * inv_c
        ee_w += inv_c[:, None, None] * (weight * ee)[None, :, :]
        te_w += inv_c[:, None, None] * (weight * te)[None, :, :]
        tt_w += inv_c[:, None, None] * (weight * tt)[None, :, :]
        t2_w += inv_c * (weight * np.diag(tt))

    return e_mean, ee_mix, te_mix, tt_mix, ee_w, te_w, tt_w, t2_w


def estimate_params_analytical_outliers(
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

    ee_sum_w = np.zeros((n_traits, n_endos, n_endos), dtype=float)
    te_sum_w = np.zeros((n_traits, n_traits, n_endos), dtype=float)
    tt_sum_w = np.zeros((n_traits, n_traits, n_traits), dtype=float)
    t2_sum_w = np.zeros(n_traits, dtype=float)

    if np.isnan(gwas.betas).any() or np.isnan(gwas.ses).any():
        for i in range(n):
            weight = float(weights[i])
            if weight <= 0.0:
                continue
            single, is_cols = gwas.only_data_point(i)
            if not is_cols:
                continue
            params_reduced = params.reduce_to(single.meta.trait_names, is_cols)
            e_mean, ee_mix, te_mix, tt_mix, ee_w, te_w, tt_w, t2_w = (
                _analytic_variant_moments(params_reduced, single.betas[0], single.ses[0])
            )
            e_sum += weight * e_mean
            ee_sum += weight * ee_mix
            w_sum += weight
            for local_i, global_i in enumerate(is_cols):
                te_sum[global_i] += weight * te_mix[local_i]
                ee_sum_w[global_i] += weight * ee_w[local_i]
                t2_sum_w[global_i] += weight * t2_w[local_i]
                for local_j, global_j in enumerate(is_cols):
                    tt_sum[global_i, global_j] += weight * tt_mix[local_i, local_j]
                    te_sum_w[global_i][global_j] += weight * te_w[local_i][local_j]
                    for local_k, global_k in enumerate(is_cols):
                        tt_sum_w[global_i][global_j][global_k] += (
                            weight * tt_w[local_i][local_j][local_k]
                        )
    else:
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            betas_obs = gwas.betas[start:end, :]
            ses = gwas.ses[start:end, :]
            w = weights[start:end]
            for i in range(betas_obs.shape[0]):
                weight = float(w[i])
                if weight <= 0.0:
                    continue
                e_mean, ee_mix, te_mix, tt_mix, ee_w, te_w, tt_w, t2_w = (
                    _analytic_variant_moments(params, betas_obs[i], ses[i])
                )
                e_sum += weight * e_mean
                ee_sum += weight * ee_mix
                te_sum += weight * te_mix
                tt_sum += weight * tt_mix
                w_sum += weight

                ee_sum_w += weight * ee_w
                te_sum_w += weight * te_w
                tt_sum_w += weight * tt_w
                t2_sum_w += weight * t2_w

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
            sigma2[i_trait] = t2_sum_w[i_trait] / w_sum
            continue
        n_endos_active = len(active_endos)
        n_parents = len(parents)
        n_pred = n_endos_active + n_parents
        zz = np.zeros((n_pred, n_pred), dtype=float)
        tz = np.zeros(n_pred, dtype=float)
        if active_endos:
            zz[:n_endos_active, :n_endos_active] = ee_sum_w[
                i_trait
            ][np.ix_(active_endos, active_endos)]
            tz[:n_endos_active] = te_sum_w[i_trait][i_trait, active_endos]
        if parents:
            zz[n_endos_active:, n_endos_active:] = tt_sum_w[i_trait][
                np.ix_(parents, parents)
            ]
            tz[n_endos_active:] = tt_sum_w[i_trait][i_trait, parents]
            if active_endos:
                te_block = te_sum_w[i_trait][parents, :][:, active_endos]
                zz[n_endos_active:, :n_endos_active] = te_block
                zz[:n_endos_active, n_endos_active:] = te_block.T

        coeffs = np.linalg.solve(zz, tz)
        for i, k in enumerate(active_endos):
            betas[i_trait, k] = coeffs[i]
        for i, p in enumerate(parents):
            trait_edges[i_trait, p] = coeffs[n_endos_active + i]

        sigma2[i_trait] = (
            t2_sum_w[i_trait]
            - 2.0 * float(coeffs @ tz)
            + float(coeffs @ zz @ coeffs)
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
        trait_edges=trait_edges.tolist(),
        outlier_kappa=params.outlier_kappa,
        outlier_pis=params.outlier_pis,
    )
