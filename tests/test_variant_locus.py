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
