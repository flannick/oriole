import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"


def _run(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def test_gwas_ssf_output(tmp_path):
    work_dir = tmp_path / "work"
    shutil.copytree(DATA_DIR, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    cfg = work_dir / "ssf_config_classify.toml"

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "classify",
            "-f",
            str(cfg),
            "--chunk-size",
            "1000",
        ],
        cwd=work_dir,
        env=env,
    )

    full_out = work_dir / "ssf_full.tsv"
    ssf_out = work_dir / "ssf_out.tsv"
    assert full_out.exists()
    assert ssf_out.exists()

    with ssf_out.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip("\n")
        assert header == (
            "variant_id\tchromosome\tbase_pair_location\teffect_allele\tother_allele"
            "\teffect_allele_frequency\trsid\tbeta\tstandard_error\tp_value"
        )
        rows = [line.strip("\n") for line in handle if line.strip()]
    assert len(rows) == 3
    first = rows[0].split("\t")
    assert first[0] == "1:100:A:G"
    assert first[1] == "1"
    assert first[2] == "100"
    assert first[3] == "A"
    assert first[4] == "G"
    assert first[5] == "0.12"
    assert first[6] == "rs111"
