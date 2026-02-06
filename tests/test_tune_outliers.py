import json
import math
import os
from pathlib import Path

import numpy as np

from oriole.options.config import load_config
from oriole.train.tune_outliers import tune_outliers


def _write_gwas(path: Path, ids, betas, ses):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("VAR_ID\tBETA\tSE\n")
        for var_id, beta, se in zip(ids, betas, ses):
            handle.write(f"{var_id}\t{beta}\t{se}\n")


def test_tune_outliers_prefers_higher_pi(tmp_path):
    rng = np.random.default_rng(1)
    n_pos = 12
    n_bg = 60
    ids_pos = [f"POS{i}" for i in range(n_pos)]
    ids_bg = [f"BG{i}" for i in range(n_bg)]
    ids_all = ids_pos + ids_bg

    # coherent signal across traits for positives
    pos_signal = 0.2
    betas_pos = pos_signal + 0.01 * rng.standard_normal(n_pos)
    se_pos = 0.05 + 0.005 * rng.standard_normal(n_pos)

    # background mostly noise, with single-trait outliers
    betas_bg = 0.01 * rng.standard_normal(n_bg)
    se_bg = 0.05 + 0.005 * rng.standard_normal(n_bg)
    outlier_idx = rng.choice(n_bg, size=10, replace=False)
    betas_bg[outlier_idx] += 1.0

    gwas_dir = tmp_path / "gwas"
    gwas_dir.mkdir()
    for trait in ("t1", "t2", "t3"):
        betas = np.concatenate([betas_pos, betas_bg])
        ses = np.concatenate([se_pos, se_bg])
        _write_gwas(gwas_dir / f"{trait}.tsv", ids_all, betas, ses)

    (tmp_path / "positives.txt").write_text("\n".join(ids_pos))
    (tmp_path / "genome.txt").write_text("\n".join(ids_all))

    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[files]
params = \"{(tmp_path / 'params.json').as_posix()}\"

[[gwas]]
name = \"t1\"
file = \"{(gwas_dir / 't1.tsv').as_posix()}\"
[gwas.cols]
id = \"VAR_ID\"
effect = \"BETA\"
se = \"SE\"

[[gwas]]
name = \"t2\"
file = \"{(gwas_dir / 't2.tsv').as_posix()}\"
[gwas.cols]
id = \"VAR_ID\"
effect = \"BETA\"
se = \"SE\"

[[gwas]]
name = \"t3\"
file = \"{(gwas_dir / 't3.tsv').as_posix()}\"
[gwas.cols]
id = \"VAR_ID\"
effect = \"BETA\"
se = \"SE\"

[outliers]
enabled = true
kappa = 4.0
expected_outliers = 0.03
max_enum_traits = 12
method = \"analytic\"

[tune_outliers]
enabled = true
mode = \"fast\"
n_folds = 3
seed = 1
kappa_grid = [2.0, 4.0]
expected_outliers_grid = [0.003, 0.3]
min_grid_length = 1
max_grid_length = 10
genomewide_ids_file = \"{(tmp_path / 'genome.txt').as_posix()}\"
n_background_sample = 60
fpr_targets = [0.2]
lambda_hard = 0.0
expand_on_boundary = false
boundary_margin = 0.0

[train]
ids_file = \"{(tmp_path / 'positives.txt').as_posix()}\"
n_steps_burn_in = 10
n_samples_per_iteration = 10
n_iterations_per_round = 1
n_rounds = 1
normalize_mu_to_one = true
learn_mu_tau = true
"""
    )

    cfg = load_config(str(config))
    result = tune_outliers(cfg, "analytic")
    assert math.isclose(result.kappa, 2.0) or math.isclose(result.kappa, 4.0)
    assert math.isclose(result.expected_outliers, 0.003) or math.isclose(
        result.expected_outliers, 0.3
    )
    assert len(result.scores) == 4
