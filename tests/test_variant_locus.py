from pathlib import Path

from oriole.data import load_data
from oriole.options.config import load_config
from oriole.options.action import Action


def test_locus_mode_flips_beta(tmp_path):
    gwas = tmp_path / "gwas.tsv"
    gwas.write_text(
        "\t".join(["VAR_ID", "CHR", "POS", "REF", "ALT", "BETA", "SE"]) + "\n"
        + "\t".join(["rs1", "1", "100", "A", "G", "1.0", "0.1"]) + "\n"
        + "\t".join(["rs2", "1", "200", "G", "C", "2.0", "0.2"]) + "\n",
        encoding="utf-8",
    )
    ids = tmp_path / "ids.tsv"
    ids.write_text(
        "\t".join(["CHROM", "POS", "REF", "ALT", "WEIGHT"]) + "\n"
        + "\t".join(["1", "100", "A", "G", "1.0"]) + "\n"
        + "\t".join(["1", "200", "C", "G", "1.0"]) + "\n",
        encoding="utf-8",
    )

    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[files]
params = "{(tmp_path / 'params.json').as_posix()}"

[variants]
id_mode = "locus"

[[gwas]]
name = "t1"
file = "{gwas.as_posix()}"
[gwas.cols]
id = "VAR_ID"
effect = "BETA"
se = "SE"
effect_allele = "ALT"
other_allele = "REF"
chrom = "CHR"
pos = "POS"

[train]
ids_file = "{ids.as_posix()}"
n_steps_burn_in = 1
n_samples_per_iteration = 1
n_iterations_per_round = 1
n_rounds = 1
normalize_mu_to_one = true
learn_mu_tau = false
""",
        encoding="utf-8",
    )

    cfg = load_config(str(config))
    data = load_data(cfg, Action.TRAIN).gwas_data

    key1 = "1:100:A:G"
    key2 = "1:200:C:G"
    idx1 = data.meta.var_ids.index(key1)
    idx2 = data.meta.var_ids.index(key2)

    assert data.betas[idx1, 0] == 1.0
    assert data.betas[idx2, 0] == -2.0


def test_auto_mode_detects_locus(tmp_path):
    gwas = tmp_path / "gwas.tsv"
    gwas.write_text(
        "\t".join(["VAR_ID", "CHR", "POS", "REF", "ALT", "BETA", "SE"]) + "\n"
        + "\t".join(["rs1", "1", "100", "A", "G", "1.0", "0.1"]) + "\n",
        encoding="utf-8",
    )
    ids = tmp_path / "ids.tsv"
    ids.write_text(
        "\t".join(["CHROM", "POS", "REF", "ALT", "WEIGHT"]) + "\n"
        + "\t".join(["1", "100", "A", "G", "1.0"]) + "\n",
        encoding="utf-8",
    )

    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[files]
params = "{(tmp_path / 'params.json').as_posix()}"

[variants]
id_mode = "auto"

[[gwas]]
name = "t1"
file = "{gwas.as_posix()}"
[gwas.cols]
id = "VAR_ID"
effect = "BETA"
se = "SE"
effect_allele = "ALT"
other_allele = "REF"
chrom = "CHR"
pos = "POS"

[train]
ids_file = "{ids.as_posix()}"
n_steps_burn_in = 1
n_samples_per_iteration = 1
n_iterations_per_round = 1
n_rounds = 1
normalize_mu_to_one = true
learn_mu_tau = false
""",
        encoding="utf-8",
    )

    cfg = load_config(str(config))
    data = load_data(cfg, Action.TRAIN).gwas_data
    assert data.meta.var_ids == ["1:100:A:G"]


def test_auto_mode_mixed_ids_fallbacks(tmp_path):
    gwas = tmp_path / "gwas.tsv"
    gwas.write_text(
        "\t".join(["VAR_ID", "BETA", "SE"]) + "\n"
        + "\t".join(["rs1", "0.5", "0.1"]) + "\n"
        + "\t".join(["1:200:C:T", "1.5", "0.2"]) + "\n",
        encoding="utf-8",
    )
    ids = tmp_path / "ids.tsv"
    ids.write_text(
        "\n".join(["rs1\t1.0", "1:200:C:T\t1.0"]) + "\n",
        encoding="utf-8",
    )

    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[files]
params = "{(tmp_path / 'params.json').as_posix()}"

[variants]
id_mode = "auto"

[[gwas]]
name = "t1"
file = "{gwas.as_posix()}"
[gwas.cols]
id = "VAR_ID"
effect = "BETA"
se = "SE"

[train]
ids_file = "{ids.as_posix()}"
n_steps_burn_in = 1
n_samples_per_iteration = 1
n_iterations_per_round = 1
n_rounds = 1
normalize_mu_to_one = true
learn_mu_tau = false
""",
        encoding="utf-8",
    )

    cfg = load_config(str(config))
    data = load_data(cfg, Action.TRAIN).gwas_data
    assert "rs1" in data.meta.var_ids
    assert "1:200:C:T" in data.meta.var_ids


def test_auto_mode_flips_across_multiple_gwas(tmp_path):
    ids = tmp_path / "ids.tsv"
    ids.write_text(
        "\n".join(["1:100:A:G\t1.0", "1:200:C:T\t1.0"]) + "\n",
        encoding="utf-8",
    )
    gwas1 = tmp_path / "g1.tsv"
    gwas1.write_text(
        "\t".join(["VAR_ID", "BETA", "SE"]) + "\n"
        + "\t".join(["1:100:A:G", "1.0", "0.1"]) + "\n"
        + "\t".join(["1:200:T:C", "2.0", "0.2"]) + "\n",
        encoding="utf-8",
    )
    gwas2 = tmp_path / "g2.tsv"
    gwas2.write_text(
        "\t".join(["VAR_ID", "BETA", "SE"]) + "\n"
        + "\t".join(["1:100:G:A", "3.0", "0.3"]) + "\n"
        + "\t".join(["1:200:C:T", "4.0", "0.4"]) + "\n",
        encoding="utf-8",
    )

    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[files]
params = "{(tmp_path / 'params.json').as_posix()}"

[variants]
id_mode = "auto"

[[gwas]]
name = "t1"
file = "{gwas1.as_posix()}"
[gwas.cols]
id = "VAR_ID"
effect = "BETA"
se = "SE"

[[gwas]]
name = "t2"
file = "{gwas2.as_posix()}"
[gwas.cols]
id = "VAR_ID"
effect = "BETA"
se = "SE"

[train]
ids_file = "{ids.as_posix()}"
n_steps_burn_in = 1
n_samples_per_iteration = 1
n_iterations_per_round = 1
n_rounds = 1
normalize_mu_to_one = true
learn_mu_tau = false
""",
        encoding="utf-8",
    )

    cfg = load_config(str(config))
    data = load_data(cfg, Action.TRAIN).gwas_data

    key1 = "1:100:A:G"
    key2 = "1:200:C:T"
    idx1 = data.meta.var_ids.index(key1)
    idx2 = data.meta.var_ids.index(key2)

    # g1: key1 direct, key2 flipped (T:C -> C:T)
    assert data.betas[idx1, 0] == 1.0
    assert data.betas[idx2, 0] == -2.0
    # g2: key1 flipped (G:A -> A:G), key2 direct
    assert data.betas[idx1, 1] == -3.0
    assert data.betas[idx2, 1] == 4.0
