# ORIOLE

ORIOLE is a Python rewrite of the original Rust tool (`mocasa`). It fits a linear-Gaussian latent variable model with one or more endophenotypes (E) driving multiple traits (T), which in turn generate observed effects (O). The tool supports Gibbs sampling and closed-form (analytical) inference for classification, and analytical EM for training.

## Quick Start

### 1) Install

```bash
# from src/oriole
python -m pip install -e .
```

Dependencies are minimal: `numpy`, `tomli`, `tomli-w`. Tests use `pytest`.

### 2) Create a config

Copy the sample config and edit paths:

```
cp tests/data/sample_config_train.toml ./config.toml
```

Update the file paths to point at your GWAS files and variant ID list.

### 3) Train

```bash
oriole train -f config.toml
```

### 4) Classify

```bash
oriole classify -f config.toml
```

---

## Model Overview

For each variant *j* and trait *i* (with K endophenotypes):

- Endophenotype effect:  
  `E^j ~ Normal(mu, diag(tau^2))`

- True trait effects:  
  `T_i^j | E^j ~ Normal(B_i · E^j, sigma_i^2)`

- Observed effects:  
  `O_i^j | T_i^j ~ Normal(T_i^j, s_i^j^2)`

The observed standard errors `s_i^j` come from the GWAS files. The parameters to fit are:

- `mu_k`, `tau_k` for each endophenotype
- `beta_{i,k}` and `sigma_i` for each trait (betas are masked by config)

The model is linear-Gaussian, so the posterior for `E^j | O^j` is Gaussian and closed-form expectations are available for EM and analytical classification.

---

## Install & Dependencies

```bash
python -m pip install -e .
```

Required:
- `numpy`
- `tomli`
- `tomli-w`

Tests:
- `pytest`

---

## Running Oriole

### Training

```bash
oriole train -f config.toml
```

Analytical EM is the default.

Gibbs sampling (slower, stochastic):

```bash
oriole train -f config.toml --gibbs
```

### Classification

```bash
oriole classify -f config.toml
```

Analytical inference is the default.

Gibbs sampling (slower, stochastic):

```bash
oriole classify -f config.toml --gibbs
```

---

## Configuration File

Oriole uses TOML. A minimal template:

```toml
[files]
params = "path/to/params.json"  # written by train, read by classify

[[gwas]]
name = "trait1"
file = "path/to/trait1.tsv"

[gwas.cols]
id = "VAR_ID"
effect = "BETA"
se = "SE"

# repeat [[gwas]] blocks for each trait

[[endophenotypes]]
name = "E1"
traits = ["trait1"]  # subset of gwas names, or ["*"] for all traits

[[endophenotypes]]
name = "E2"
traits = ["trait2"]

[train]
ids_file = "path/to/variant_ids.txt"
n_steps_burn_in = 50
n_samples_per_iteration = 100
n_iterations_per_round = 100
n_rounds = 10
normalize_mu_to_one = true

[classify]
params_override = {}
n_steps_burn_in = 50
n_samples = 1000
out_file = "path/to/output.tsv"
trace_ids = []
```

If `[[endophenotypes]]` is omitted, Oriole defaults to a single endophenotype named `E` connected to all traits.

### Params JSON format

New format (multi-E):

```json
{
  "trait_names": ["trait1", "trait2"],
  "endo_names": ["E1", "E2"],
  "mus": [1.0, 0.5],
  "taus": [1.2, 0.8],
  "betas": [
    [0.2, 0.0],
    [0.0, -0.3]
  ],
  "sigmas": [0.4, 0.5]
}
```

Old single-E format is still accepted (it is interpreted as one endophenotype named `E`).

### GWAS file format

The GWAS files must have a header row containing the ID, effect, and SE columns. Supported delimiters: `;`, tab, `,`, or single spaces.

Example header:

```
VAR_ID;BETA;SE
```

### Variant ID list

The `train.ids_file` is a two-column file:

```
VAR_ID   WEIGHT
```

If no weight is provided, it defaults to 1.0.

---

## Flags

- `--gibbs`  
  Use Gibbs sampling instead of analytical inference. Supported for classify and train.

- `--match-rust`  
  Preserves Rust’s initial parameter estimation behavior (uses betas to estimate SE statistics).

- `--chunk-size <N>`  
  Number of variants to process per chunk. Default targets ~2GB RAM. Use `1` to disable vectorization.

- `-d, --dry`  
  Dry run (load data, check config, but do not train/classify).

---

## Test Data Included

The repo includes small test data under `tests/data/`:

- `aligned_*_1000.tsv` (9 traits, 1000 variants)
- `sample_1000_ids.txt`
- `sample_params.json`
- `sample_classify.tsv`
- `sample_config_train.toml`
- `sample_config_classify.toml`
- `multi_e_trait1.tsv`, `multi_e_trait2.tsv`
- `multi_e_ids.txt`
- `multi_e_params.json`
- `multi_e_config_train.toml`
- `multi_e_config_classify.toml`

---

## Tests

From `src/oriole`:

```bash
pytest -q
```

The test suite trains and classifies analytically (the default) using the bundled 1000-variant dataset.

---

## Full Usage Reference

### Train

```bash
oriole train -f config.toml [--gibbs] [--chunk-size N] [--match-rust]
```

### Classify

```bash
oriole classify -f config.toml [--gibbs] [--chunk-size N]
```

---

## More Complex Models

For multiple endophenotypes, add `[[endophenotypes]]` blocks to your config and assign each endo to a subset of traits. Traits listed under an endo will have free betas; missing traits are masked (their betas are fixed to 0).

Example:

```toml
[[endophenotypes]]
name = "E1"
traits = ["fi", "isiadj", "bmi"]  # subset of gwas names, or ["*"] for all traits

[[endophenotypes]]
name = "E2"
traits = ["bmi", "whradj", "bfp"]
```

If `[[endophenotypes]]` is omitted, Oriole defaults to a single endophenotype named `E` connected to all traits. Parameters are stored using the new multi-E JSON format (`endo_names`, `mus`, `taus`, `betas` matrix), but the old single-E format is still accepted.

---
