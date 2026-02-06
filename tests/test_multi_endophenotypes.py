import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"


def _run(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def test_multi_e_train_analytical(tmp_path):
    work_dir = tmp_path / "work"
    shutil.copytree(DATA_DIR, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    train_cfg = work_dir / "multi_e_config_train.toml"

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "train",
            "-f",
            str(train_cfg),
            "--chunk-size",
            "10",
        ],
        cwd=work_dir,
        env=env,
    )

    params_out = work_dir / "multi_e_params_out.json"
    assert params_out.exists()
    params = json.loads(params_out.read_text())
    assert params["endo_names"] == ["E1", "E2"]
    betas = params["betas"]
    assert len(betas) == 2
    assert len(betas[0]) == 2
    assert abs(betas[0][1]) < 1e-8
    assert abs(betas[1][0]) < 1e-8


def test_multi_e_classify_analytical(tmp_path):
    work_dir = tmp_path / "work"
    shutil.copytree(DATA_DIR, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    classify_cfg = work_dir / "multi_e_config_classify.toml"

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "classify",
            "-f",
            str(classify_cfg),
            "--chunk-size",
            "10",
        ],
        cwd=work_dir,
        env=env,
    )

    out_file = work_dir / "multi_e_classify_out.tsv"
    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip("\n")
        parts = header.split("\t")
        assert parts[:15] == [
            "id",
            "E1_mean_post",
            "E1_std_post",
            "E1_mean_calc",
            "E1_beta_gls",
            "E1_se_gls",
            "E1_z_gls",
            "E1_p_gls",
            "E2_mean_post",
            "E2_std_post",
            "E2_mean_calc",
            "E2_beta_gls",
            "E2_se_gls",
            "E2_z_gls",
            "E2_p_gls",
        ]
        assert parts[15:] == ["trait1", "trait2"]
        n_rows = sum(1 for _ in handle)
    assert n_rows == 3
