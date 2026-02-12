import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"


def _run(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def test_min_trait_coverage_filter(tmp_path):
    work_dir = tmp_path / "work_cov"
    shutil.copytree(DATA_DIR, work_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    (work_dir / "cov_trait1.tsv").write_text(
        "VAR_ID;BETA;SE\nVAR_A;0.10;0.05\nVAR_B;0.20;0.05\n",
        encoding="utf-8",
    )
    (work_dir / "cov_trait2.tsv").write_text(
        "VAR_ID;BETA;SE\nVAR_B;0.30;0.05\nVAR_C;-0.40;0.05\n",
        encoding="utf-8",
    )
    (work_dir / "cov_params.json").write_text(
        "{\n"
        '  "trait_names": ["trait1", "trait2"],\n'
        '  "mu": 0.0,\n'
        '  "tau": 1.0,\n'
        '  "betas": [0.5, 0.5],\n'
        '  "sigmas": [1.0, 1.0]\n'
        "}\n",
        encoding="utf-8",
    )

    base_cfg = (
        "[files]\n"
        'params = "cov_params.json"\n\n'
        "[[gwas]]\n"
        'name = "trait1"\n'
        'file = "cov_trait1.tsv"\n'
        "[gwas.cols]\n"
        'id = "VAR_ID"\n'
        'effect = "BETA"\n'
        'se = "SE"\n\n'
        "[[gwas]]\n"
        'name = "trait2"\n'
        'file = "cov_trait2.tsv"\n'
        "[gwas.cols]\n"
        'id = "VAR_ID"\n'
        'effect = "BETA"\n'
        'se = "SE"\n\n'
        "[train]\n"
        'ids_file = "sample_1000_ids.txt"\n\n'
        "[classify]\n"
        'out_file = "cov_out.tsv"\n'
    )

    (work_dir / "cov_cfg_50.toml").write_text(
        base_cfg + "min_trait_coverage_pct = 50.0\n",
        encoding="utf-8",
    )
    (work_dir / "cov_cfg_100.toml").write_text(
        base_cfg + "min_trait_coverage_pct = 100.0\n",
        encoding="utf-8",
    )

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "classify",
            "-f",
            str(work_dir / "cov_cfg_50.toml"),
        ],
        cwd=work_dir,
        env=env,
    )
    rows_50 = (work_dir / "cov_out.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert rows_50[0].startswith("id\tn_traits_observed\t")
    assert len(rows_50) == 4
    observed_50 = {line.split("\t")[0]: int(line.split("\t")[1]) for line in rows_50[1:]}
    assert observed_50 == {"VAR_A": 1, "VAR_B": 2, "VAR_C": 1}

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "classify",
            "-f",
            str(work_dir / "cov_cfg_100.toml"),
        ],
        cwd=work_dir,
        env=env,
    )
    rows_100 = (work_dir / "cov_out.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows_100) == 2
    assert rows_100[1].split("\t")[0] == "VAR_B"
    assert rows_100[1].split("\t")[1] == "2"

