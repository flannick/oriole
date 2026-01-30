import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "oriole" / "tests" / "data"


def _run(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def test_train_and_classify_analytical(tmp_path):
    work_dir = tmp_path / "work"
    shutil.copytree(DATA_DIR, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src" / "oriole" / "src")

    train_cfg = work_dir / "sample_config_train.toml"
    classify_cfg = work_dir / "sample_config_classify.toml"

    # Train (analytical)
    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "train",
            "-f",
            str(train_cfg),
            "--analytical",
            "--chunk-size",
            "1000",
        ],
        cwd=work_dir,
        env=env,
    )

    params_out = work_dir / "sample_params_out.json"
    assert params_out.exists()

    # Classify (analytical) using bundled sample params
    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "classify",
            "-f",
            str(classify_cfg),
            "--analytical",
            "--chunk-size",
            "1000",
        ],
        cwd=work_dir,
        env=env,
    )

    out_file = work_dir / "sample_classify_out.tsv"
    assert out_file.exists()

    # quick sanity check: header + 1000 rows
    with out_file.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip("\n")
        assert header.startswith("id\te_mean_samp\te_std_samp\te_mean_calc")
        n_rows = sum(1 for _ in handle)
    assert n_rows == 1000
