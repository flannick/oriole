from __future__ import annotations

import numpy as np

from ..data import GwasData
from ..params import Params
from .vars import Vars


class GibbsSampler:
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def draw_e_component(
        self, vars: Vars, params: Params, i_data_point: int, i_endo: int
    ) -> float:
        betas = np.asarray(params.betas, dtype=float)
        sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
        beta_k = betas[:, i_endo]
        e_row = vars.es[i_data_point]
        t_row = vars.ts[i_data_point]
        mu_k = params.mus[i_endo]
        tau2_k = params.taus[i_endo] ** 2
        inv_var_sum = (1.0 / tau2_k) + np.sum((beta_k ** 2) / sigma2)
        variance = 1.0 / inv_var_sum
        std_dev = variance**0.5
        mu_t = betas @ e_row
        r = t_row - (mu_t - beta_k * e_row[i_endo])
        frac_sum = (mu_k / tau2_k) + np.sum(beta_k * r / sigma2)
        mean = variance * frac_sum
        return float(self._rng.normal(loc=mean, scale=std_dev))

    def draw_t(
        self,
        data: GwasData,
        vars: Vars,
        params: Params,
        i_data_point: int,
        i_trait: int,
        pinned: bool,
    ) -> float:
        betas_row = np.asarray(params.betas[i_trait], dtype=float)
        mu_e = float(betas_row @ vars.es[i_data_point])
        var_e = params.sigmas[i_trait] ** 2
        mu_o = data.betas[i_data_point, i_trait]
        se_o = data.ses[i_data_point, i_trait]
        if se_o > 0.0 and not pinned:
            var_o = se_o**2
            variance = 1.0 / (1.0 / var_e + 1.0 / var_o)
            std_dev = variance**0.5
            mean = variance * (mu_e / var_e + mu_o / var_o)
            return float(self._rng.normal(loc=mean, scale=std_dev))
        return float(mu_o)
