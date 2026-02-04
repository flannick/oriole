import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"


def _run(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def test_outliers_train_classify_analytic(tmp_path):
    work_dir = tmp_path / "work"
    shutil.copytree(DATA_DIR, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    train_cfg = work_dir / "outliers_config_train.toml"
    classify_cfg = work_dir / "outliers_config_classify.toml"

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "train",
            "-f",
            str(train_cfg),
            "--chunk-size",
            "200",
        ],
        cwd=work_dir,
        env=env,
    )

    params_out = work_dir / "outliers_params_out.json"
    assert params_out.exists()

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "classify",
            "-f",
            str(classify_cfg),
            "--chunk-size",
            "200",
        ],
        cwd=work_dir,
        env=env,
    )

    out_file = work_dir / "outliers_classify_out.tsv"
    assert out_file.exists()

    with out_file.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip("\n")
        assert header.startswith("id\tE_mean_samp\tE_std_samp\tE_mean_calc")
        n_rows = sum(1 for _ in handle)
    assert n_rows == 1000


def test_outliers_train_classify_variational(tmp_path):
    work_dir = tmp_path / "work"
    shutil.copytree(DATA_DIR, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    train_cfg = work_dir / "outliers_trait_edges_config_train.toml"
    classify_cfg = work_dir / "outliers_trait_edges_config_classify.toml"

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "train",
            "-f",
            str(train_cfg),
            "--chunk-size",
            "200",
        ],
        cwd=work_dir,
        env=env,
    )

    params_out = work_dir / "outliers_trait_edges_params_out.json"
    assert params_out.exists()

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "classify",
            "-f",
            str(classify_cfg),
            "--chunk-size",
            "200",
        ],
        cwd=work_dir,
        env=env,
    )

    out_file = work_dir / "outliers_trait_edges_classify_out.tsv"
    assert out_file.exists()

    with out_file.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip("\n")
        assert header.startswith("id\tE_mean_samp\tE_std_samp\tE_mean_calc")
        n_rows = sum(1 for _ in handle)
    assert n_rows == 5
