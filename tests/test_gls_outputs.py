import math

import numpy as np

from oriole.classify.analytical import gls_endophenotype_stats_chunk
from oriole.classify.outliers_analytic import outliers_analytic_gls
from oriole.classify.outliers_variational import outliers_variational_gls
from oriole.data import GwasData, Meta
from oriole.params import Params
from oriole.sample.sampler import Sampler
from oriole.sample.vars import Vars


def _base_params(trait_names: list[str]) -> Params:
    n_traits = len(trait_names)
    return Params(
        trait_names=trait_names,
        endo_names=["E"],
        mus=[0.0],
        taus=[1.0],
        betas=[[0.5] for _ in range(n_traits)],
        sigmas=[0.5 for _ in range(n_traits)],
        trait_edges=[[0.0 for _ in range(n_traits)] for _ in range(n_traits)],
        outlier_kappa=4.0,
        outlier_pis=[0.1 for _ in range(n_traits)],
    )


def test_gls_closed_form_single_endo():
    params = Params(
        trait_names=["t1", "t2"],
        endo_names=["E"],
        mus=[0.0],
        taus=[1.0],
        betas=[[0.5], [1.2]],
        sigmas=[0.4, 0.6],
        trait_edges=[[0.0, 0.0], [0.0, 0.0]],
        outlier_kappa=1.0,
        outlier_pis=[0.0, 0.0],
    )
    betas = np.array([[0.2, -0.1]], dtype=float)
    ses = np.array([[0.1, 0.2]], dtype=float)
    e_beta, e_se = gls_endophenotype_stats_chunk(params, betas, ses)

    v1 = params.sigmas[0] ** 2 + ses[0, 0] ** 2
    v2 = params.sigmas[1] ** 2 + ses[0, 1] ** 2
    num = params.betas[0][0] * betas[0, 0] / v1 + params.betas[1][0] * betas[0, 1] / v2
    denom = (params.betas[0][0] ** 2) / v1 + (params.betas[1][0] ** 2) / v2
    expected_beta = num / denom
    expected_se = 1.0 / math.sqrt(denom)

    assert math.isclose(e_beta[0, 0], expected_beta, rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(e_se[0, 0], expected_se, rel_tol=1e-10, abs_tol=1e-10)


def test_variational_matches_enumeration_small():
    params = _base_params(["t1", "t2", "t3", "t4"])
    betas = np.array([0.15, -0.05, 0.08, 0.02], dtype=float)
    ses = np.array([0.1, 0.1, 0.12, 0.09], dtype=float)

    e_beta_enum, e_se_enum = outliers_analytic_gls(params, betas, ses)
    e_beta_var, e_se_var = outliers_variational_gls(params, betas, ses)

    z_enum = e_beta_enum[0] / e_se_enum[0]
    z_var = e_beta_var[0] / e_se_var[0]
    assert math.isclose(z_enum, z_var, rel_tol=5e-2, abs_tol=5e-2)


def test_gibbs_gls_matches_enumeration():
    params = _base_params(["t1", "t2", "t3"])
    betas = np.array([[0.12, -0.08, 0.05]], dtype=float)
    ses = np.array([[0.1, 0.11, 0.09]], dtype=float)

    meta = Meta(trait_names=params.trait_names, var_ids=["v1"], endo_names=params.endo_names)
    data = GwasData(meta=meta, betas=betas, ses=ses)
    vars = Vars.initial_vars(data, params)
    sampler = Sampler(meta, np.random.default_rng(1))

    sampler.sample_n(data, params, vars, 200, None, False)
    sampler.reset_stats()
    sampler.sample_n(data, params, vars, 400, None, False)

    mean_z = sampler.var_stats.z_sums[0] / max(1, sampler.var_stats.n)
    kappa2 = params.outlier_kappa ** 2
    alpha = (1.0 - mean_z) + (mean_z / kappa2)
    sigma2_eff = (np.asarray(params.sigmas, dtype=float) ** 2) / alpha
    e_beta_gibbs, e_se_gibbs = gls_endophenotype_stats_chunk(
        params, betas, ses, sigma2_override=sigma2_eff
    )

    e_beta_enum, e_se_enum = outliers_analytic_gls(params, betas[0], ses[0])
    z_gibbs = e_beta_gibbs[0, 0] / e_se_gibbs[0, 0]
    z_enum = e_beta_enum[0] / e_se_enum[0]
    assert math.isclose(z_gibbs, z_enum, rel_tol=2e-1, abs_tol=2e-1)
