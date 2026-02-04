from __future__ import annotations

import numpy as np

from ..data import GwasData
from ..params import Params
from .vars import Vars


class GibbsSampler:
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._adj_cache_key: int | None = None
        self._parents: list[np.ndarray] = []
        self._children: list[np.ndarray] = []
        self._children_coef: list[np.ndarray] = []

    def _ensure_adjacency(self, params: Params) -> None:
        cache_key = id(params)
        if self._adj_cache_key == cache_key:
            return
        trait_edges = np.asarray(params.trait_edges, dtype=float)
        n_traits = trait_edges.shape[0]
        self._parents = []
        self._children = []
        self._children_coef = []
        for i_trait in range(n_traits):
            parents = np.where(np.abs(trait_edges[i_trait]) > 1e-12)[0]
            self._parents.append(parents)
            children = np.where(np.abs(trait_edges[:, i_trait]) > 1e-12)[0]
            self._children.append(children)
            self._children_coef.append(trait_edges[children, i_trait])
        self._adj_cache_key = cache_key

    def draw_e_component(
        self, vars: Vars, params: Params, i_data_point: int, i_endo: int
    ) -> float:
        betas = np.asarray(params.betas, dtype=float)
        sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
        trait_edges = np.asarray(params.trait_edges, dtype=float)
        kappa2 = params.outlier_kappa ** 2
        zs_row = vars.zs[i_data_point]
        beta_k = betas[:, i_endo]
        e_row = vars.es[i_data_point]
        t_row = vars.ts[i_data_point]
        mu_k = params.mus[i_endo]
        tau2_k = params.taus[i_endo] ** 2
        c = np.where(zs_row > 0, kappa2, 1.0)
        sigma2_eff = sigma2 * c
        inv_var_sum = (1.0 / tau2_k) + np.sum((beta_k ** 2) / sigma2_eff)
        variance = 1.0 / inv_var_sum
        std_dev = variance**0.5
        l_row = t_row - trait_edges @ t_row
        mu_t = betas @ e_row
        r = l_row - (mu_t - beta_k * e_row[i_endo])
        frac_sum = (mu_k / tau2_k) + np.sum(beta_k * r / sigma2_eff)
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
        self._ensure_adjacency(params)
        betas = np.asarray(params.betas, dtype=float)
        trait_edges = np.asarray(params.trait_edges, dtype=float)
        sigmas2 = np.asarray(params.sigmas, dtype=float) ** 2
        kappa2 = params.outlier_kappa ** 2
        t_row = vars.ts[i_data_point]
        e_row = vars.es[i_data_point]
        z_row = vars.zs[i_data_point]
        mu_o = data.betas[i_data_point, i_trait]
        se_o = data.ses[i_data_point, i_trait]
        if se_o <= 0.0 or pinned:
            return float(mu_o)
        sigma2_i = sigmas2[i_trait] * (kappa2 if z_row[i_trait] else 1.0)
        parents = self._parents[i_trait]
        if parents.size:
            mu_parents = float(trait_edges[i_trait, parents] @ t_row[parents])
        else:
            mu_parents = 0.0
        mu_e = float(betas[i_trait] @ e_row + mu_parents)
        precision = (1.0 / sigma2_i) + (1.0 / (se_o**2))
        num = (mu_e / sigma2_i) + (mu_o / (se_o**2))
        for i_child, a_ci in zip(self._children[i_trait], self._children_coef[i_trait]):
            parents_child = self._parents[i_child]
            if parents_child.size:
                parent_sum = float(trait_edges[i_child, parents_child] @ t_row[parents_child])
            else:
                parent_sum = 0.0
            mu_child = float(
                betas[i_child] @ e_row + (parent_sum - a_ci * t_row[i_trait])
            )
            resid = t_row[i_child] - mu_child
            sigma2_child = sigmas2[i_child] * (kappa2 if z_row[i_child] else 1.0)
            precision += (a_ci ** 2) / sigma2_child
            num += (a_ci / sigma2_child) * resid
        variance = 1.0 / precision
        std_dev = variance**0.5
        mean = variance * num
        return float(self._rng.normal(loc=mean, scale=std_dev))

    def draw_z(
        self, vars: Vars, params: Params, i_data_point: int, i_trait: int
    ) -> int:
        betas = np.asarray(params.betas, dtype=float)
        trait_edges = np.asarray(params.trait_edges, dtype=float)
        sigmas2 = np.asarray(params.sigmas, dtype=float) ** 2
        t_row = vars.ts[i_data_point]
        e_row = vars.es[i_data_point]
        pi = params.outlier_pis[i_trait]
        kappa2 = params.outlier_kappa ** 2
        if pi <= 0.0:
            return 0
        l_row = t_row - trait_edges @ t_row
        mu = float(betas[i_trait] @ e_row)
        eps = l_row[i_trait] - mu
        sigma2 = sigmas2[i_trait]
        log_p1 = (
            np.log(pi)
            - 0.5 * np.log(2.0 * np.pi * sigma2 * kappa2)
            - (eps**2) / (2.0 * sigma2 * kappa2)
        )
        log_p0 = (
            np.log(1.0 - pi)
            - 0.5 * np.log(2.0 * np.pi * sigma2)
            - (eps**2) / (2.0 * sigma2)
        )
        max_log = max(log_p0, log_p1)
        p1 = np.exp(log_p1 - max_log) / (
            np.exp(log_p0 - max_log) + np.exp(log_p1 - max_log)
        )
        return int(self._rng.random() < p1)
