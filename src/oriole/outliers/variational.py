from __future__ import annotations

import numpy as np

from ..params import Params
from .moments import posterior_moments


def _no_trait_edges(params: Params) -> bool:
    return not np.any(np.asarray(params.trait_edges, dtype=float))


def _logit(x: np.ndarray) -> np.ndarray:
    return np.log(x) - np.log(1.0 - x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _epsilon_second_diag(
    params: Params,
    ee: np.ndarray,
    te: np.ndarray,
    tt: np.ndarray,
) -> np.ndarray:
    beta = np.asarray(params.betas, dtype=float)
    trait_edges = np.asarray(params.trait_edges, dtype=float)
    n_traits = beta.shape[0]
    l_mat = np.eye(n_traits, dtype=float) - trait_edges

    lt = l_mat @ tt
    et = l_mat @ te
    et2 = te.T @ l_mat.T
    be = beta @ ee

    eps_second = lt @ l_mat.T - et @ beta.T - beta @ et2 + be @ beta.T
    return np.diag(eps_second)


def variational_posterior(
    params: Params,
    betas_obs: np.ndarray,
    ses: np.ndarray,
    max_iters: int = 25,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    betas_arr = np.asarray(betas_obs, dtype=float)
    se2 = np.asarray(ses, dtype=float) ** 2
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    pis = np.asarray(params.outlier_pis, dtype=float)
    kappa2 = params.outlier_kappa ** 2

    phi = np.clip(pis, 1e-6, 1.0 - 1e-6)
    l_inv = None
    for _ in range(max_iters):
        alpha = (1.0 - phi) + (phi / kappa2)
        sigma2_eff = sigma2 / alpha
        m, ee, te, tt, t_mean, l_inv = posterior_moments(
            params, betas_arr, se2, sigma2_eff, l_inv=l_inv
        )
        eps2 = _epsilon_second_diag(params, ee, te, tt)
        logit_phi = (
            _logit(pis)
            - 0.5 * np.log(kappa2)
            + 0.5 * (1.0 - 1.0 / kappa2) * (eps2 / sigma2)
        )
        phi_new = _sigmoid(logit_phi)
        if np.max(np.abs(phi_new - phi)) < tol:
            phi = phi_new
            break
        phi = phi_new

    alpha = (1.0 - phi) + (phi / kappa2)
    sigma2_eff = sigma2 / alpha
    m, ee, te, tt, t_mean, _ = posterior_moments(
        params, betas_arr, se2, sigma2_eff, l_inv=l_inv
    )
    return m, ee, te, tt, t_mean, phi, alpha


def _variational_posterior_chunk_no_edges(
    params: Params,
    betas_obs: np.ndarray,
    ses: np.ndarray,
    max_iters: int = 25,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    betas_arr = np.asarray(betas_obs, dtype=float)
    se2 = np.asarray(ses, dtype=float) ** 2
    sigma2 = np.asarray(params.sigmas, dtype=float) ** 2
    pis = np.asarray(params.outlier_pis, dtype=float)
    kappa2 = params.outlier_kappa ** 2
    beta = np.asarray(params.betas, dtype=float)
    mu = np.asarray(params.mus, dtype=float)
    tau2 = np.asarray(params.taus, dtype=float) ** 2
    tau_inv = np.diag(1.0 / tau2)

    n_vars, n_traits = betas_arr.shape
    n_endos = params.n_endos()

    phi = np.clip(pis, 1e-6, 1.0 - 1e-6)[None, :].repeat(n_vars, axis=0)
    beta_outer = np.einsum("ti,tj->tij", beta, beta)
    for _ in range(max_iters):
        alpha = (1.0 - phi) + (phi / kappa2)
        sigma2_eff = sigma2[None, :] / alpha
        v = sigma2_eff + se2
        inv_v = 1.0 / v

        i_mat = np.einsum("tij,nt->nij", beta_outer, inv_v)
        b_vec = np.einsum("ti,nt->ni", beta, betas_arr * inv_v)
        rhs = b_vec + (tau_inv @ mu)[None, :]
        try:
            c = np.linalg.inv(i_mat + tau_inv[None, :, :])
        except np.linalg.LinAlgError:
            c = np.linalg.pinv(i_mat + tau_inv[None, :, :])
        m = np.einsum("nij,nj->ni", c, rhs)
        mu_t = m @ beta.T
        v_inv_r = (betas_arr - mu_t) * inv_v
        t_mean = mu_t + sigma2_eff * v_inv_r

        h = beta[None, :, :] * (1.0 - sigma2_eff / v)[:, :, None]
        cov_add = np.einsum("ntk,nkl,ntl->nt", h, c, h)
        cov_t_diag = sigma2_eff - (sigma2_eff**2) * inv_v + cov_add
        tt_diag = cov_t_diag + t_mean**2

        hc = np.einsum("ntk,nkl->ntl", h, c)
        te = t_mean[:, :, None] * m[:, None, :] + hc
        ee = c + np.einsum("ni,nj->nij", m, m)

        term2 = 2.0 * np.einsum("ntk,tk->nt", te, beta)
        term3 = np.einsum("tk,nkl,tl->nt", beta, ee, beta)
        eps2 = tt_diag - term2 + term3

        logit_phi = (
            _logit(pis)[None, :]
            - 0.5 * np.log(kappa2)
            + 0.5 * (1.0 - 1.0 / kappa2) * (eps2 / sigma2[None, :])
        )
        phi_new = _sigmoid(logit_phi)
        if np.max(np.abs(phi_new - phi)) < tol:
            phi = phi_new
            break
        phi = phi_new

    alpha = (1.0 - phi) + (phi / kappa2)
    sigma2_eff = sigma2[None, :] / alpha
    v = sigma2_eff + se2
    inv_v = 1.0 / v

    i_mat = np.einsum("tij,nt->nij", beta_outer, inv_v)
    b_vec = np.einsum("ti,nt->ni", beta, betas_arr * inv_v)
    rhs = b_vec + (tau_inv @ mu)[None, :]
    try:
        c = np.linalg.inv(i_mat + tau_inv[None, :, :])
    except np.linalg.LinAlgError:
        c = np.linalg.pinv(i_mat + tau_inv[None, :, :])
    m = np.einsum("nij,nj->ni", c, rhs)
    mu_t = m @ beta.T
    v_inv_r = (betas_arr - mu_t) * inv_v
    t_mean = mu_t + sigma2_eff * v_inv_r
    e_std = np.sqrt(np.maximum(np.diagonal(c, axis1=1, axis2=2), 0.0))
    return m, e_std, t_mean, alpha
