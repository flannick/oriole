from __future__ import annotations

from ..params import Params


def calculate_mu(params: Params, betas: list[float], ses: list[float]) -> float:
    tau2 = params.tau**2
    numerator = sum(
        params.betas[i] * betas[i] / (params.sigmas[i] ** 2 + ses[i] ** 2)
        for i in range(len(params.betas))
    ) + params.mu / tau2
    denominator = sum(
        params.betas[i] ** 2 / (params.sigmas[i] ** 2 + ses[i] ** 2)
        for i in range(len(params.betas))
    ) + 1.0 / tau2
    return numerator / denominator

