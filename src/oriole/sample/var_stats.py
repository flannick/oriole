from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import Meta, Weights
from ..params import Params
from .vars import Vars


@dataclass
class SampledClassification:
    e_mean: float
    e_std: float
    t_means: list[float]


class VarStats:
    def __init__(self, meta: Meta) -> None:
        self.meta = meta
        n_data_points = meta.n_data_points()
        n_traits = meta.n_traits()
        self.n = 0
        self.e_sums = np.zeros(n_data_points, dtype=float)
        self.e2_sums = np.zeros(n_data_points, dtype=float)
        self.e_t_sums = np.zeros((n_data_points, n_traits), dtype=float)
        self.t_sums = np.zeros((n_data_points, n_traits), dtype=float)
        self.t2_sums = np.zeros((n_data_points, n_traits), dtype=float)

    def reset(self) -> None:
        self.n = 0
        self.e_sums.fill(0.0)
        self.e2_sums.fill(0.0)
        self.e_t_sums.fill(0.0)
        self.t_sums.fill(0.0)
        self.t2_sums.fill(0.0)

    def add(self, vars: Vars) -> None:
        self.n += 1
        es = vars.es
        ts = vars.ts
        self.e_sums += es
        self.e2_sums += es**2
        self.e_t_sums += es[:, None] * ts
        self.t_sums += ts
        self.t2_sums += ts**2

    def compute_new_params(self, weights: Weights) -> Params:
        meta = self.meta
        n_f = float(self.n)
        weights_arr = np.asarray(weights.weights, dtype=float)
        weights_sum = weights.sum
        mean_e = self.e_sums / n_f
        mean_e2 = self.e2_sums / n_f
        mu = float((weights_arr * mean_e).sum() / weights_sum)

        tau = float(
            ((weights_arr * (mean_e2 - 2.0 * mu * mean_e + mu**2)).sum() / weights_sum) ** 0.5
        )

        mean_e_t = self.e_t_sums / n_f
        mean_e2_sum = float((weights_arr * mean_e2).sum())
        betas_arr = (weights_arr[:, None] * mean_e_t).sum(axis=0) / mean_e2_sum
        betas = [float(beta) for beta in betas_arr]

        mean_t2 = self.t2_sums / n_f
        betas_vec = betas_arr[None, :]
        sigma_terms = mean_t2 - 2.0 * mean_e_t * betas_vec + mean_e2[:, None] * betas_vec**2
        sigma2 = (weights_arr[:, None] * sigma_terms).sum(axis=0) / weights_sum
        sigma2 = np.maximum(sigma2, 0.0)
        sigmas = [float(value) for value in np.sqrt(sigma2)]

        return Params(
            trait_names=meta.trait_names,
            mu=mu,
            tau=tau,
            betas=betas,
            sigmas=sigmas,
        )

    def calculate_classification(self) -> SampledClassification:
        denom = float(self.n * self.meta.n_data_points())
        e_mean = float(self.e_sums.sum() / denom)
        e2_mean = float(self.e2_sums.sum() / denom)
        t_means = [float(value) for value in self.t_sums.sum(axis=0) / denom]
        e_std = float((e2_mean - e_mean**2) ** 0.5)
        return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_means)
