from __future__ import annotations

import numpy as np

from ..params import Params
from .moments import posterior_moments


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
