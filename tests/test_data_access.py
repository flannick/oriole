import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"


def _run(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def test_gwas_base_uri_file_scheme(tmp_path):
    work_dir = tmp_path / "work"
    shutil.copytree(DATA_DIR, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    base_uri = f"file://{work_dir.as_posix()}"
    base_config = (work_dir / "sample_config_classify.toml").read_text(
        encoding="utf-8"
    )
    config_path = work_dir / "config_file_scheme.toml"
    config_path.write_text(
        f"[data_access]\n"
        f"gwas_base_uri = \"{base_uri}\"\n\n"
        f"{base_config}",
        encoding="utf-8",
    )

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "classify",
            "-f",
            str(config_path),
            "--chunk-size",
            "1000",
        ],
        cwd=work_dir,
        env=env,
    )

    out_file = work_dir / "sample_classify_out.tsv"
    assert out_file.exists()


def test_gwas_uri_file_scheme(tmp_path):
    work_dir = tmp_path / "work"
    shutil.copytree(DATA_DIR, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    base_config = (work_dir / "sample_config_classify.toml").read_text(
        encoding="utf-8"
    )
    config_path = work_dir / "config_uri_file_scheme.toml"
    rewritten = []
    for line in base_config.splitlines():
        if line.strip().startswith("file ="):
            rel_path = line.split("=", 1)[1].strip().strip('"')
            uri = f"file://{(work_dir / rel_path).as_posix()}"
            rewritten.append(f'uri = "{uri}"')
        else:
            rewritten.append(line)
    config_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "classify",
            "-f",
            str(config_path),
            "--chunk-size",
            "1000",
        ],
        cwd=work_dir,
        env=env,
    )

    out_file = work_dir / "sample_classify_out.tsv"
    assert out_file.exists()
