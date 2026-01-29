import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MOCASA_BIN = ROOT / "src" / "mocasa" / "target" / "release" / "mocasa"


def _write_sample_inputs():
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sample_ids.txt").write_text("VAR1 1.0\nVAR2 1.0\nVAR3 1.0\n")
    (RESULTS / "sample_trait1.tsv").write_text(
        "VAR_ID\tBETA\tSE\nVAR1\t0.1\t0.05\nVAR2\t-0.2\t0.07\nVAR3\t0.05\t0.04\n"
    )
    (RESULTS / "sample_trait2.tsv").write_text(
        "VAR_ID\tBETA\tSE\nVAR1\t0.2\t0.06\nVAR2\t-0.1\t0.08\nVAR3\t0.03\t0.05\n"
    )


def _write_config(path: Path, params_path: Path):
    config = f"""
[files]
params = "{params_path.as_posix()}"

[[gwas]]
name = "trait1"
file = "{(RESULTS / 'sample_trait1.tsv').as_posix()}"

[gwas.cols]
id = "VAR_ID"
effect = "BETA"
se = "SE"

[[gwas]]
name = "trait2"
file = "{(RESULTS / 'sample_trait2.tsv').as_posix()}"

[gwas.cols]
id = "VAR_ID"
effect = "BETA"
se = "SE"

[train]
ids_file = "{(RESULTS / 'sample_ids.txt').as_posix()}"
n_steps_burn_in = 5
n_samples_per_iteration = 5
n_iterations_per_round = 2
n_rounds = 1
normalize_mu_to_one = true

[classify]
params_override = {{}}
n_steps_burn_in = 5
n_samples = 10
out_file = "{(RESULTS / 'out_small.tsv').as_posix()}"
trace_ids = []
"""
    path.write_text(config.strip() + "\n")


def _run_oriole(config_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src" / "oriole" / "src")
    subprocess.run(
        [sys.executable, "-m", "oriole", "train", "-f", str(config_path)],
        cwd=ROOT,
        env=env,
        check=True,
    )


def _run_mocasa(config_path: Path):
    subprocess.run(
        [str(MOCASA_BIN), "train", "-f", str(config_path)],
        cwd=ROOT,
        check=True,
    )


def _load_params(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_scatter_compare_oriole_mocasa():
    if not MOCASA_BIN.exists():
        pytest.skip("mocasa binary not found")
    _write_sample_inputs()
    FIGURES.mkdir(exist_ok=True)

    oriole_params = RESULTS / "params_oriole.json"
    mocasa_params = RESULTS / "params_mocasa.json"
    oriole_config = RESULTS / "config_oriole.toml"
    mocasa_config = RESULTS / "config_mocasa.toml"
    _write_config(oriole_config, oriole_params)
    _write_config(mocasa_config, mocasa_params)

    _run_oriole(oriole_config)
    _run_mocasa(mocasa_config)

    oriole = _load_params(oriole_params)
    mocasa = _load_params(mocasa_params)

    oriole_betas = oriole["betas"]
    mocasa_betas = mocasa["betas"]

    fig_path = FIGURES / "compare_oriole_mocasa_betas.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(mocasa_betas, oriole_betas)
    plt.xlabel("mocasa betas")
    plt.ylabel("oriole betas")
    plt.title("Oriole vs Mocasa betas")
    plt.tight_layout()
    plt.savefig(fig_path)

    assert fig_path.exists()
    assert len(oriole_betas) == len(mocasa_betas)
