import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _write_params(path: Path) -> None:
    path.write_text(
        "{\n"
        '  "trait_names": ["t1", "t2"],\n'
        '  "mu": 0.0,\n'
        '  "tau": 1.0,\n'
        '  "betas": [0.4, 0.6],\n'
        '  "sigmas": [1.0, 1.0]\n'
        "}\n",
        encoding="utf-8",
    )


def _run_classify(work_dir: Path, cfg: Path, mode: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [sys.executable, "-m", "oriole", "classify", "-f", str(cfg)]
    if mode == "sorted":
        cmd.append("--sorted")
    elif mode == "unsorted":
        cmd.append("--unsorted")
    return subprocess.run(
        cmd,
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
    )


def _base_config_text(variant_mode: str = "auto", out_file: str = "out.tsv") -> str:
    return (
        "[files]\n"
        'params = "params.json"\n\n'
        "[data_access]\n"
        "max_memory_gb = 0.000001\n\n"
        "[variants]\n"
        f'id_mode = "{variant_mode}"\n\n'
        "[train]\n"
        'ids_file = "ids.tsv"\n\n'
        "[classify]\n"
        f'out_file = "{out_file}"\n'
        "min_trait_coverage_pct = 0.0\n"
        'input_order = "auto"\n\n'
    )


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").strip().splitlines()


@pytest.mark.parametrize("mode", ["auto", "unsorted", "sorted"])
def test_id_modes_sorted_and_unsorted(tmp_path: Path, mode: str) -> None:
    _write_params(tmp_path / "params.json")
    (tmp_path / "ids.tsv").write_text("v1\t1\nv2\t1\nv3\t1\n", encoding="utf-8")
    (tmp_path / "g1.tsv").write_text(
        "VAR_ID;BETA;SE\n"
        "v1;0.10;0.10\n"
        "v2;0.20;0.10\n"
        "v3;0.30;0.10\n",
        encoding="utf-8",
    )
    (tmp_path / "g2.tsv").write_text(
        "VAR_ID;BETA;SE\n"
        "v1;0.11;0.10\n"
        "v2;0.22;0.10\n"
        "v3;0.33;0.10\n",
        encoding="utf-8",
    )
    cfg = _base_config_text("id")
    cfg += (
        "[[gwas]]\nname = \"t1\"\nfile = \"g1.tsv\"\n[gwas.cols]\n"
        "id = \"VAR_ID\"\neffect = \"BETA\"\nse = \"SE\"\n\n"
        "[[gwas]]\nname = \"t2\"\nfile = \"g2.tsv\"\n[gwas.cols]\n"
        "id = \"VAR_ID\"\neffect = \"BETA\"\nse = \"SE\"\n"
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cfg, encoding="utf-8")

    proc = _run_classify(tmp_path, cfg_path, mode)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = _read_lines(tmp_path / "out.tsv")
    assert len(lines) == 4
    assert lines[0].startswith("id\tn_traits_observed\t")
    assert [line.split("\t")[0] for line in lines[1:]] == ["v1", "v2", "v3"]


@pytest.mark.parametrize("mode", ["auto", "unsorted", "sorted"])
def test_locus_modes_sorted_and_unsorted(tmp_path: Path, mode: str) -> None:
    _write_params(tmp_path / "params.json")
    (tmp_path / "ids.tsv").write_text("2:10:A:G\t1\n2:20:A:G\t1\n1:10:C:T\t1\n", encoding="utf-8")
    # Chromosome order is 2 then 1 in both files; positions increase within chromosome.
    (tmp_path / "g1.tsv").write_text(
        "VAR_ID;BETA;SE\n"
        "2:10:A:G;0.10;0.10\n"
        "2:20:A:G;0.20;0.10\n"
        "1:10:C:T;0.30;0.10\n",
        encoding="utf-8",
    )
    # Same loci but first row is flipped allele orientation.
    (tmp_path / "g2.tsv").write_text(
        "VAR_ID;BETA;SE\n"
        "2:10:G:A;0.11;0.10\n"
        "2:20:A:G;0.22;0.10\n"
        "1:10:C:T;0.33;0.10\n",
        encoding="utf-8",
    )
    cfg = _base_config_text("auto")
    cfg += (
        "[[gwas]]\nname = \"t1\"\nfile = \"g1.tsv\"\n[gwas.cols]\n"
        "id = \"VAR_ID\"\neffect = \"BETA\"\nse = \"SE\"\n\n"
        "[[gwas]]\nname = \"t2\"\nfile = \"g2.tsv\"\n[gwas.cols]\n"
        "id = \"VAR_ID\"\neffect = \"BETA\"\nse = \"SE\"\n"
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cfg, encoding="utf-8")

    proc = _run_classify(tmp_path, cfg_path, mode)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = _read_lines(tmp_path / "out.tsv")
    ids = [line.split("\t")[0] for line in lines[1:]]
    for expected in ["2:10:A:G", "2:20:A:G", "1:10:C:T"]:
        assert expected in ids


def test_sorted_mode_errors_for_unsorted_ids(tmp_path: Path) -> None:
    _write_params(tmp_path / "params.json")
    (tmp_path / "ids.tsv").write_text("v1\t1\nv2\t1\nv3\t1\n", encoding="utf-8")
    (tmp_path / "g1.tsv").write_text(
        "VAR_ID;BETA;SE\n"
        "v2;0.10;0.10\n"
        "v1;0.20;0.10\n"
        "v3;0.30;0.10\n",
        encoding="utf-8",
    )
    (tmp_path / "g2.tsv").write_text(
        "VAR_ID;BETA;SE\n"
        "v1;0.11;0.10\n"
        "v2;0.22;0.10\n"
        "v3;0.33;0.10\n",
        encoding="utf-8",
    )
    cfg = _base_config_text("id")
    cfg += (
        "[[gwas]]\nname = \"t1\"\nfile = \"g1.tsv\"\n[gwas.cols]\n"
        "id = \"VAR_ID\"\neffect = \"BETA\"\nse = \"SE\"\n\n"
        "[[gwas]]\nname = \"t2\"\nfile = \"g2.tsv\"\n[gwas.cols]\n"
        "id = \"VAR_ID\"\neffect = \"BETA\"\nse = \"SE\"\n"
    )
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cfg, encoding="utf-8")

    proc_sorted = _run_classify(tmp_path, cfg_path, "sorted")
    assert proc_sorted.returncode != 0
    assert "--unsorted" in (proc_sorted.stdout + proc_sorted.stderr)

    proc_unsorted = _run_classify(tmp_path, cfg_path, "unsorted")
    assert proc_unsorted.returncode == 0, proc_unsorted.stdout + proc_unsorted.stderr
    assert (tmp_path / "out.tsv").exists()
