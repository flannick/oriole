from __future__ import annotations

from math import sqrt

from ..params import Params
from ..sample.var_stats import SampledClassification


def analytical_classification(params: Params, betas: list[float], ses: list[float]) -> SampledClassification:
    tau2 = params.tau ** 2
    inv_var = 1.0 / tau2
    v_list = []
    for i in range(len(params.betas)):
        v = params.sigmas[i] ** 2 + ses[i] ** 2
        v_list.append(v)
        inv_var += (params.betas[i] ** 2) / v

    v_e = 1.0 / inv_var
    mean_term = params.mu / tau2
    for i in range(len(params.betas)):
        mean_term += params.betas[i] * betas[i] / v_list[i]
    m_e = v_e * mean_term
    e_std = sqrt(v_e)

    t_means = []
    for i in range(len(params.betas)):
        sigma2 = params.sigmas[i] ** 2
        se2 = ses[i] ** 2
        denom = sigma2 + se2
        a = sigma2 / denom
        b = (se2 / denom) * params.betas[i]
        t_means.append(a * betas[i] + b * m_e)

    return SampledClassification(e_mean=m_e, e_std=e_std, t_means=t_means)

