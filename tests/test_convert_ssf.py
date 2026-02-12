import csv
import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"


def _run(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def test_convert_ssf_single_endo(tmp_path):
    work_dir = tmp_path / "work_single"
    shutil.copytree(DATA_DIR, work_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    in_file = work_dir / "single_classify.tsv"
    with in_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "id",
                "E_mean_post",
                "E_std_post",
                "E_mean_calc",
                "E_beta_gls",
                "E_se_gls",
                "E_z_gls",
                "E_p_gls",
                "trait1",
            ]
        )
        writer.writerow(["1:100:A:G", "0.1", "0.2", "0.1", "0.15", "0.03", "5.0", "5e-7", "0.0"])

    out_file = work_dir / "single_ssf.tsv"
    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "convert-ssf",
            "-i",
            str(in_file),
            "-o",
            str(out_file),
        ],
        cwd=work_dir,
        env=env,
    )

    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as handle:
        rows = [line.strip() for line in handle if line.strip()]
    assert rows[0] == (
        "variant_id\tchromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error\tp_value"
    )
    assert rows[1] == "1:100:A:G\t1\t100\tA\tG\t0.15\t0.03\t5e-7"


def test_convert_ssf_multi_endo_gz_no_guess(tmp_path):
    work_dir = tmp_path / "work_multi"
    shutil.copytree(DATA_DIR, work_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    in_file = work_dir / "multi_classify.tsv.gz"
    with gzip.open(in_file, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
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
                "trait1",
                "trait2",
            ]
        )
        writer.writerow(
            [
                "rs123",
                "0.1",
                "0.2",
                "0.1",
                "0.11",
                "0.02",
                "5.5",
                "3e-8",
                "0.1",
                "0.2",
                "0.1",
                "-0.21",
                "0.04",
                "-5.25",
                "1.5e-7",
                "0.0",
                "0.0",
            ]
        )

    out_base = work_dir / "multi_ssf.tsv.gz"
    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "convert-ssf",
            "-i",
            str(in_file),
            "-o",
            str(out_base),
            "--no-guess-fields",
        ],
        cwd=work_dir,
        env=env,
    )

    out_e1 = work_dir / "multi_ssf_E1.tsv.gz"
    out_e2 = work_dir / "multi_ssf_E2.tsv.gz"
    assert out_e1.exists()
    assert out_e2.exists()
    with gzip.open(out_e1, "rt", encoding="utf-8") as handle:
        rows_e1 = [line.strip() for line in handle if line.strip()]
    with gzip.open(out_e2, "rt", encoding="utf-8") as handle:
        rows_e2 = [line.strip() for line in handle if line.strip()]
    assert rows_e1[0] == "variant_id\tbeta\tstandard_error\tp_value"
    assert rows_e2[0] == "variant_id\tbeta\tstandard_error\tp_value"
    assert rows_e1[1] == "rs123\t0.11\t0.02\t3e-8"
    assert rows_e2[1] == "rs123\t-0.21\t0.04\t1.5e-7"

