import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from oriole.check import check_config
from oriole.options.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"


def _run(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def test_trait_edges_classify_analytical(tmp_path):
    work_dir = tmp_path / "work"
    shutil.copytree(DATA_DIR, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    classify_cfg = work_dir / "trait_edges_config_classify.toml"

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

    out_file = work_dir / "trait_edges_classify_out.tsv"
    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip("\n")
        assert header.startswith("id\tE_mean_samp\tE_std_samp\tE_mean_calc")
        parts = header.split("\t")
        assert parts[-3:] == ["trait1", "trait2", "trait3"]
        n_rows = sum(1 for _ in handle)
    assert n_rows == 5


def test_trait_edges_cycle_rejected():
    config = load_config(str(DATA_DIR / "trait_edges_config_cycle.toml"))
    with pytest.raises(Exception):
        check_config(config)
