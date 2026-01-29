from __future__ import annotations

import numpy as np

from ..data import GwasData
from ..params import Params
from .vars import Vars


class GibbsSampler:
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def draw_e(self, vars: Vars, params: Params, i_data_point: int) -> float:
        n_traits = params.n_traits()
        tau = params.tau
        inv_var_sum = 1.0 / tau**2 + sum(
            (params.betas[i] / params.sigmas[i]) ** 2 for i in range(n_traits)
        )
        variance = 1.0 / inv_var_sum
        std_dev = variance**0.5
        frac_sum = params.mu / tau**2 + sum(
            params.betas[i] * vars.ts[i_data_point, i] / params.sigmas[i] ** 2
            for i in range(n_traits)
        )
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
        mu_e = params.betas[i_trait] * vars.es[i_data_point]
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

