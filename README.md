# oriole (Python)

Oriole is a Python rewrite of the original Rust tool (`mocasa`). It fits a linear-Gaussian latent variable model with a single endophenotype (E) driving multiple traits (T), which in turn generate observed effects (O). The tool supports Gibbs sampling and closed-form (analytical) inference for classification, and analytical EM for training.

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
oriole train -f config.toml --analytical
```

### 4) Classify

```bash
oriole classify -f config.toml --analytical
```

---

## Model Overview

For each variant *j* and trait *i*:

- Endophenotype effect:  
  `E^j ~ Normal(mu, tau^2)`

- True trait effects:  
  `T_i^j | E^j ~ Normal(beta_i * E^j, sigma_i^2)`

- Observed effects:  
  `O_i^j | T_i^j ~ Normal(T_i^j, s_i^j^2)`

The observed standard errors `s_i^j` come from the GWAS files. The parameters to fit are:

- `mu`, `tau` (endophenotype mean and SD)
- `beta_i`, `sigma_i` for each trait

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

Analytical EM (fast, vectorized):

```bash
oriole train -f config.toml --analytical
```

### Classification

```bash
oriole classify -f config.toml
```

Analytical (fast, vectorized):

```bash
oriole classify -f config.toml --analytical
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

- `--analytical`  
  Use closed-form inference instead of Gibbs sampling. Supported for classify and train.

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

---

## Tests

From `src/oriole`:

```bash
pytest -q
```

The test suite trains and classifies analytically using the bundled 1000-variant dataset.

---

## Full Usage Reference

### Train

```bash
oriole train -f config.toml [--analytical] [--chunk-size N] [--match-rust]
```

### Classify

```bash
oriole classify -f config.toml [--analytical] [--chunk-size N]
```

---

## Publishing to GitHub (flannick)

1) Initialize git if needed:

```bash
git init
```

2) Create a GitHub repo (web or CLI), then add it as a remote:

```bash
git remote add origin git@github.com:flannick/<repo-name>.git
```

3) Stage and commit:

```bash
git add .
git commit -m "Initial public release of oriole"
```

4) Push:

```bash
git branch -M main
git push -u origin main
```

---

## Next Steps After Push

You indicated you’ll do a clean clone and test. Recommended commands:

```bash
git clone git@github.com:flannick/<repo-name>.git
cd <repo-name>/src/oriole
python -m pip install -e .
pytest -q
```

