from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import Meta, Weights
from ..params import Params
from .vars import Vars


@dataclass
class SampledClassification:
    e_mean: np.ndarray
    e_std: np.ndarray
    t_means: np.ndarray


class VarStats:
    def __init__(self, meta: Meta) -> None:
        self.meta = meta
        n_data_points = meta.n_data_points()
        n_traits = meta.n_traits()
        self.n = 0
        n_endos = meta.n_endos()
        self.e_sums = np.zeros((n_data_points, n_endos), dtype=float)
        self.ee_sums = np.zeros((n_data_points, n_endos, n_endos), dtype=float)
        self.e_t_sums = np.zeros((n_data_points, n_traits, n_endos), dtype=float)
        self.t_sums = np.zeros((n_data_points, n_traits), dtype=float)
        self.t2_sums = np.zeros((n_data_points, n_traits), dtype=float)
        self.t_t_sums = np.zeros((n_data_points, n_traits, n_traits), dtype=float)

    def reset(self) -> None:
        self.n = 0
        self.e_sums.fill(0.0)
        self.ee_sums.fill(0.0)
        self.e_t_sums.fill(0.0)
        self.t_sums.fill(0.0)
        self.t2_sums.fill(0.0)
        self.t_t_sums.fill(0.0)

    def add(self, vars: Vars) -> None:
        self.n += 1
        es = vars.es
        ts = vars.ts
        self.e_sums += es
        self.ee_sums += es[:, :, None] * es[:, None, :]
        self.e_t_sums += ts[:, :, None] * es[:, None, :]
        self.t_sums += ts
        self.t2_sums += ts**2
        self.t_t_sums += ts[:, :, None] * ts[:, None, :]

    def compute_new_params(
        self, weights: Weights, mask: np.ndarray, parent_mask: np.ndarray
    ) -> Params:
        meta = self.meta
        n_f = float(self.n)
        weights_arr = np.asarray(weights.weights, dtype=float)
        weights_sum = weights.sum
        mean_e = self.e_sums / n_f
        mean_ee = self.ee_sums / n_f
        mean_e2 = np.diagonal(mean_ee, axis1=1, axis2=2)
        mu = (weights_arr[:, None] * mean_e).sum(axis=0) / weights_sum

        tau2 = (
            weights_arr[:, None]
            * (mean_e2 - 2.0 * mu[None, :] * mean_e + (mu[None, :] ** 2))
        ).sum(axis=0) / weights_sum
        tau2 = np.maximum(tau2, 0.0)
        taus = np.sqrt(tau2)

        mean_e_t = self.e_t_sums / n_f
        mean_t_t = self.t_t_sums / n_f
        mean_t2 = self.t2_sums / n_f
        ee_sum = (weights_arr[:, None, None] * mean_ee).sum(axis=0)
        te_sum = (weights_arr[:, None, None] * mean_e_t).sum(axis=0)
        tt_sum = (weights_arr[:, None, None] * mean_t_t).sum(axis=0)
        t2_sum = (weights_arr[:, None] * mean_t2).sum(axis=0)

        n_traits = meta.n_traits()
        n_endos = meta.n_endos()
        betas = np.zeros((n_traits, n_endos), dtype=float)
        trait_edges = np.zeros((n_traits, n_traits), dtype=float)
        sigma2 = np.zeros(n_traits, dtype=float)
        for i_trait in range(n_traits):
            active_endos = [i for i, allowed in enumerate(mask[i_trait]) if allowed]
            parents = [i for i, allowed in enumerate(parent_mask[i_trait]) if allowed]
            if not active_endos and not parents:
                sigma2[i_trait] = t2_sum[i_trait] / weights_sum
                continue
            n_endos_active = len(active_endos)
            n_parents = len(parents)
            n_pred = n_endos_active + n_parents
            zz = np.zeros((n_pred, n_pred), dtype=float)
            tz = np.zeros(n_pred, dtype=float)
            if active_endos:
                zz[:n_endos_active, :n_endos_active] = ee_sum[
                    np.ix_(active_endos, active_endos)
                ]
                tz[:n_endos_active] = te_sum[i_trait, active_endos]
            if parents:
                zz[n_endos_active:, n_endos_active:] = tt_sum[np.ix_(parents, parents)]
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

            sigma2[i_trait] = (
                t2_sum[i_trait]
                - 2.0 * float(coeffs @ tz)
                + float(coeffs @ zz @ coeffs)
            ) / weights_sum
        sigma2 = np.maximum(sigma2, 0.0)
        sigmas = np.sqrt(sigma2)

        return Params(
            trait_names=meta.trait_names,
            endo_names=meta.endo_names,
            mus=[float(value) for value in mu],
            taus=[float(value) for value in taus],
            betas=betas.tolist(),
            sigmas=[float(value) for value in sigmas],
            trait_edges=trait_edges.tolist(),
        )

    def calculate_classification(self) -> SampledClassification:
        denom = float(self.n)
        e_mean = self.e_sums / denom
        mean_e2 = np.diagonal(self.ee_sums / denom, axis1=1, axis2=2)
        e_var = mean_e2 - e_mean ** 2
        e_std = np.sqrt(np.maximum(e_var, 0.0))
        t_means = self.t_sums / denom
        return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_means)
