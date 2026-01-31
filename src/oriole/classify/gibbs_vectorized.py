from __future__ import annotations

import numpy as np

from ..params import Params
from ..sample.var_stats import SampledClassification


def gibbs_classification_chunk(
    params: Params,
    betas_obs: np.ndarray,
    ses: np.ndarray,
    n_burn_in: int,
    n_samples: int,
    t_pinned: bool,
) -> SampledClassification:
    rng = np.random.default_rng()
    beta = np.asarray(params.betas, dtype=float)
    sigma = np.asarray(params.sigmas, dtype=float)
    sigma2 = sigma ** 2
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    mu = np.asarray(params.mus, dtype=float)
    trait_edges = np.asarray(params.trait_edges, dtype=float)
    n_vars, n_traits = betas_obs.shape
    n_endos = params.n_endos()
    l_mat = np.eye(n_traits, dtype=float) - trait_edges
    has_edges = np.any(trait_edges)

    e = np.tile(mu[None, :], (n_vars, 1))
    m_mat = np.linalg.solve(l_mat, beta)
    t = e @ m_mat.T

    sigma2_inv = 1.0 / sigma2
    tau2_inv = 1.0 / tau2
    c_inv = np.diag(tau2_inv) + beta.T @ (sigma2_inv[:, None] * beta)
    c = np.linalg.inv(c_inv)
    chol_c = np.linalg.cholesky(c)

    se2 = ses ** 2
    def update_e():
        if not has_edges:
            rhs = (t / sigma2[None, :]) @ beta + (mu * tau2_inv)[None, :]
        else:
            y = t @ l_mat.T
            rhs = (y / sigma2[None, :]) @ beta + (mu * tau2_inv)[None, :]
        mean_e = rhs @ c
        z = rng.normal(size=(n_vars, n_endos))
        return mean_e + z @ chol_c.T

    if not has_edges:
        var_t = 1.0 / (1.0 / sigma2[None, :] + 1.0 / se2)
        std_t = np.sqrt(var_t)

        def update_t(e_vals):
            if t_pinned:
                return betas_obs.copy()
            mu_e = e_vals @ beta.T
            mean_t = var_t * ((mu_e / sigma2[None, :]) + (betas_obs / se2))
            return rng.normal(loc=mean_t, scale=std_t)

    else:
        parents_idx: list[np.ndarray] = []
        parents_coef: list[np.ndarray] = []
        children_idx: list[np.ndarray] = []
        children_coef: list[np.ndarray] = []
        for i_trait in range(n_traits):
            parents = np.where(np.abs(trait_edges[i_trait]) > 1e-12)[0]
            parents_idx.append(parents)
            parents_coef.append(trait_edges[i_trait, parents])
            children = np.where(np.abs(trait_edges[:, i_trait]) > 1e-12)[0]
            children_idx.append(children)
            children_coef.append(trait_edges[children, i_trait])

        def update_t(e_vals):
            if t_pinned:
                return betas_obs.copy()
            t_new = t.copy()
            for i_trait in range(n_traits):
                mu_e = e_vals @ beta[i_trait]
                parents = parents_idx[i_trait]
                if parents.size:
                    mu_parents = t_new[:, parents] @ parents_coef[i_trait]
                else:
                    mu_parents = 0.0
                mean_base = mu_e + mu_parents
                precision = (1.0 / sigma2[i_trait]) + np.where(
                    se2[:, i_trait] > 0.0, 1.0 / se2[:, i_trait], 0.0
                )
                num = (mean_base / sigma2[i_trait]) + np.where(
                    se2[:, i_trait] > 0.0, betas_obs[:, i_trait] / se2[:, i_trait], 0.0
                )
                for i_child, a_ci in zip(children_idx[i_trait], children_coef[i_trait]):
                    parents_child = parents_idx[i_child]
                    if parents_child.size:
                        parent_sum = t_new[:, parents_child] @ parents_coef[i_child]
                    else:
                        parent_sum = 0.0
                    mu_child = (e_vals @ beta[i_child]) + (
                        parent_sum - a_ci * t_new[:, i_trait]
                    )
                    resid = t_new[:, i_child] - mu_child
                    precision += (a_ci ** 2) / sigma2[i_child]
                    num += (a_ci / sigma2[i_child]) * resid
                variance = 1.0 / precision
                std_dev = np.sqrt(variance)
                mean = variance * num
                pinned_mask = se2[:, i_trait] <= 0.0
                t_new[:, i_trait] = rng.normal(loc=mean, scale=std_dev)
                if np.any(pinned_mask):
                    t_new[pinned_mask, i_trait] = betas_obs[pinned_mask, i_trait]
            return t_new

    for _ in range(n_burn_in):
        e = update_e()
        t = update_t(e)

    e_sum = np.zeros((n_vars, n_endos), dtype=float)
    e2_sum = np.zeros((n_vars, n_endos), dtype=float)
    t_sum = np.zeros((n_vars, n_traits), dtype=float)

    for _ in range(n_samples):
        e = update_e()
        t = update_t(e)
        e_sum += e
        e2_sum += e ** 2
        t_sum += t

    e_mean = e_sum / n_samples
    e_var = (e2_sum / n_samples) - e_mean ** 2
    e_std = np.sqrt(np.maximum(e_var, 0.0))
    t_means = t_sum / n_samples
    return SampledClassification(e_mean=e_mean, e_std=e_std, t_means=t_means)
