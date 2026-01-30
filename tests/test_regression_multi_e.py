import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG_DIR = ROOT / "tests" / "regression_multi_e"


def _run(cmd, cwd, env):
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _assert_close(actual, expected, rel=1e-8, abs_tol=1e-8):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual.keys()) == set(expected.keys())
        for key in expected:
            _assert_close(actual[key], expected[key], rel=rel, abs_tol=abs_tol)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_close(left, right, rel=rel, abs_tol=abs_tol)
        return
    if isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol)
        return
    assert actual == expected


def _read_classify(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        rows = []
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            var_id = parts[0]
            values = [float(value) for value in parts[1:]]
            rows.append((var_id, values))
        return header, rows


def test_regression_multi_e(tmp_path):
    work_dir = tmp_path / "regression_multi_e"
    shutil.copytree(REG_DIR, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "train",
            "-f",
            str(work_dir / "config_train.toml"),
            "--chunk-size",
            "10",
        ],
        cwd=work_dir,
        env=env,
    )

    params_actual = json.loads((work_dir / "params_out.json").read_text())
    params_expected = json.loads((work_dir / "expected" / "params.json").read_text())
    _assert_close(params_actual, params_expected)

    _run(
        [
            sys.executable,
            "-m",
            "oriole",
            "classify",
            "-f",
            str(work_dir / "config_classify.toml"),
            "--chunk-size",
            "10",
        ],
        cwd=work_dir,
        env=env,
    )

    header_actual, rows_actual = _read_classify(work_dir / "classify_out.tsv")
    header_expected, rows_expected = _read_classify(
        work_dir / "expected" / "classify.tsv"
    )
    assert header_actual == header_expected
    assert len(rows_actual) == len(rows_expected)
    for (var_id_a, values_a), (var_id_b, values_b) in zip(rows_actual, rows_expected):
        assert var_id_a == var_id_b
        assert len(values_a) == len(values_b)
        for left, right in zip(values_a, values_b):
            assert math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-8)
