from __future__ import annotations

from ..data import GwasData
from ..error import new_error
from ..math.stats import Stats
from ..params import Params


def estimate_initial_params(data: GwasData, match_rust: bool = False) -> Params:
    meta = data.meta
    n_data_points = meta.n_data_points()
    n_traits = meta.n_traits()
    beta_stats = [Stats() for _ in range(n_traits)]
    se_stats = [Stats() for _ in range(n_traits)]
    for i_data_point in range(n_data_points):
        for i_trait in range(n_traits):
            beta_stats[i_trait].add(float(data.betas[i_data_point, i_trait]))
            if match_rust:
                se_stats[i_trait].add(float(data.betas[i_data_point, i_trait]))
            else:
                se_stats[i_trait].add(float(data.ses[i_data_point, i_trait]))

    sigmas: list[float] = []
    for stats in beta_stats:
        var = stats.variance()
        if var is None:
            raise new_error("Need at least two data points.")
        sigmas.append(var**0.5)

    beta_means: list[float] = []
    for stats in beta_stats:
        mean = stats.mean()
        if mean is None:
            raise new_error("Need at least one data point.")
        beta_means.append(mean)

    se_means: list[float] = []
    for stats in se_stats:
        mean = stats.mean()
        if mean is None:
            raise new_error("Need at least one data point.")
        se_means.append(mean)

    means_stats = Stats()
    for mean in beta_means:
        means_stats.add(mean)
    mu = means_stats.mean()
    if mu is None:
        raise new_error("Need at least one trait.")

    precision_stats = Stats()
    for se in se_means:
        precision_stats.add(se ** -2)
    precision_mean = precision_stats.mean()
    if precision_mean is None:
        raise new_error("Need at least one trait.")
    tau = 1.0 / (precision_mean**0.5)

    betas = [mean / (mu + tau * (1 if mu >= 0 else -1)) for mean in beta_means]
    return Params(trait_names=meta.trait_names, mu=mu, tau=tau, betas=betas, sigmas=sigmas)
