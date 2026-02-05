from __future__ import annotations

import numpy as np

from ..data import LoadedData
from ..params import Params
from ..outliers.variational import variational_posterior


def estimate_params_variational_outliers(
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
            m, ee, te, tt, _, _, alpha = variational_posterior(
                params_reduced, single.betas[0], single.ses[0]
            )
            e_sum += weight * m
            ee_sum += weight * ee
            w_sum += weight
            for local_i, global_i in enumerate(is_cols):
                te_sum[global_i] += weight * te[local_i]
                ee_sum_w[global_i] += (weight * alpha[local_i]) * ee
                t2_sum_w[global_i] += (weight * alpha[local_i]) * tt[local_i, local_i]
                for local_j, global_j in enumerate(is_cols):
                    tt_sum[global_i, global_j] += weight * tt[local_i, local_j]
                    te_sum_w[global_i][global_j] += (weight * alpha[local_i]) * te[local_j]
                    for local_k, global_k in enumerate(is_cols):
                        tt_sum_w[global_i][global_j][global_k] += (
                            (weight * alpha[local_i]) * tt[local_j, local_k]
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
                m, ee, te, tt, _, _, alpha = variational_posterior(
                    params, betas_obs[i], ses[i]
                )
                e_sum += weight * m
                ee_sum += weight * ee
                te_sum += weight * te
                tt_sum += weight * tt
                w_sum += weight

                ee_sum_w += (weight * alpha)[:, None, None] * ee[None, :, :]
                te_sum_w += (weight * alpha)[:, None, None] * te[None, :, :]
                tt_sum_w += (weight * alpha)[:, None, None] * tt[None, :, :]
                t2_sum_w += (weight * alpha) * np.diag(tt)

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
