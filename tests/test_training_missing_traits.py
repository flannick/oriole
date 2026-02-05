import json
from pathlib import Path

import numpy as np

from oriole.options.config import load_config
from oriole.train.train import train_or_check


def _write_gwas(path: Path, rows: list[tuple[str, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("VAR_ID\tBETA\tSE\n")
        for var_id, beta, se in rows:
            handle.write(f"{var_id}\t{beta}\t{se}\n")


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids))


def _write_config(path: Path, gwas1: Path, gwas2: Path, ids: Path, params: Path) -> None:
    path.write_text(
        f"""
[files]
params = "{params.as_posix()}"

[[gwas]]
name = "t1"
file = "{gwas1.as_posix()}"
[gwas.cols]
id = "VAR_ID"
effect = "BETA"
se = "SE"

[[gwas]]
name = "t2"
file = "{gwas2.as_posix()}"
[gwas.cols]
id = "VAR_ID"
effect = "BETA"
se = "SE"

[train]
ids_file = "{ids.as_posix()}"
normalize_mu_to_one = true
n_rounds = 1
n_iterations_per_round = 1
n_samples_per_iteration = 1
n_steps_burn_in = 1
"""
    )


def _load_params(path: Path) -> dict:
    return json.loads(path.read_text())


def test_training_tolerates_missing_traits(tmp_path):
    ids = ["v1", "v2", "v3"]
    gwas1 = tmp_path / "t1.tsv"
    gwas2 = tmp_path / "t2.tsv"
    ids_file = tmp_path / "ids.txt"

    _write_ids(ids_file, ids)

    rows1 = [("v1", 0.1, 0.01), ("v2", 0.2, 0.01), ("v3", -0.1, 0.02)]
    rows2_full = [("v1", -0.2, 0.03), ("v2", 0.05, 0.02), ("v3", 0.1, 0.02)]
    rows2_missing = [("v1", -0.2, 0.03), ("v3", 0.1, 0.02)]

    _write_gwas(gwas1, rows1)
    _write_gwas(gwas2, rows2_full)

    config_full = tmp_path / "config_full.toml"
    params_full = tmp_path / "params_full.json"
    _write_config(config_full, gwas1, gwas2, ids_file, params_full)

    cfg_full = load_config(str(config_full))
    train_or_check(cfg_full, match_rust=False, inference="analytic", dry=False)

    _write_gwas(gwas2, rows2_missing)
    config_missing = tmp_path / "config_missing.toml"
    params_missing = tmp_path / "params_missing.json"
    _write_config(config_missing, gwas1, gwas2, ids_file, params_missing)

    cfg_missing = load_config(str(config_missing))
    train_or_check(cfg_missing, match_rust=False, inference="analytic", dry=False)

    for params_path in (params_full, params_missing):
        params = _load_params(params_path)
        assert np.isfinite(params["mus"]).all()
        assert np.isfinite(params["taus"]).all()
        assert np.isfinite(params["betas"]).all()
        assert np.isfinite(params["sigmas"]).all()
