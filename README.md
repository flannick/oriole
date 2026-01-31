# ORIOLE

ORIOLE is a linear-Gaussian model for joint inference of latent endophenotypes and
trait effects from GWAS summary statistics. It supports analytical inference and
Gibbs sampling, multiple endophenotypes, and directed edges between traits (a DAG).

## Quick start

### 1) Install

**Recommended Python:** 3.10+ (tested)
	•	Python 3.7 and older are not supported. They cannot install ORIOLE’s NumPy requirement (numpy>=1.23).
	•	If you’re using conda, create an environment with python=3.10 (recommended).
	•	If you’re using venv, make sure python3.10 (or newer) is the interpreter you use to create the venv.
  
Run installs from the **repository root** (the directory containing `pyproject.toml`).

#### Option A (recommended): conda

```bash
conda create -n oriole310 python=3.10 pip -y
conda activate oriole310
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

#### Option B: venv (requires `python3.10` available on your system)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

Dependencies: `numpy`, `tomli`, `tomli-w`. Tests use `pytest`.

> Developer note: avoid naming a local module/package `math` (it shadows the Python
> standard library). Internal math helpers live under `oriole/math_utils/`.

### 2) Run the included sample configs (recommended first run)

The sample TOML configs under `tests/data/` use **relative paths** (e.g.
`sample_1000_ids.txt`). The simplest way to run them is to change into that
directory first:

```bash
cd tests/data
oriole train -f sample_config_train.toml
oriole classify -f sample_config_classify.toml
```

### 3) Create your own config

Copy a sample config and edit paths:

```bash
cp tests/data/sample_config_train.toml ./config.toml
```

Update the GWAS file paths and the variant ID list.

### 4) Train

```bash
oriole train -f config.toml
```

Or via the wrapper script:

```bash
python run_oriole.py train -f config.toml
```

### 5) Classify

```bash
oriole classify -f config.toml
```

Or via the wrapper script:

```bash
python run_oriole.py classify -f config.toml
```

---

## Model overview

For each variant *j* and trait *i* (with K endophenotypes):

- Endophenotype effects:
  `E^j ~ Normal(mu, diag(tau^2))`

- Trait effects:
  `T^j | E^j ~ Normal(B E^j, Sigma)`

- Observed GWAS effects:
  `O^j | T^j ~ Normal(T^j, S_j)`

With trait-to-trait edges, the trait layer becomes:

- `T^j = A T^j + B E^j + eps^j`, where `eps^j ~ Normal(0, D)` and `A` is a DAG.

The observed standard errors come from the GWAS files. The parameters to fit are:

- `mu_k`, `tau_k` for each endophenotype
- `beta_{i,k}` and `sigma_i` for each trait
- `A_{i,p}` for allowed trait-to-trait edges

The model is linear-Gaussian, so analytical posterior moments are available and used
by analytical EM training and analytical classification.

---

## Install & dependencies

From the repository root:

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

## Running ORIOLE

### Training

```bash
oriole train -f config.toml
```

Or:

```bash
python run_oriole.py train -f config.toml
```

Analytical EM is the default. For Gibbs sampling:

```bash
oriole train -f config.toml --gibbs
```

Or:

```bash
python run_oriole.py train -f config.toml --gibbs
```

### Classification

```bash
oriole classify -f config.toml
```

Or:

```bash
python run_oriole.py classify -f config.toml
```

Analytical inference is the default. For Gibbs sampling:

```bash
oriole classify -f config.toml --gibbs
```

Or:

```bash
python run_oriole.py classify -f config.toml --gibbs
```

---

## Configuration file

ORIOLE uses TOML. A minimal template:

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

If `[[endophenotypes]]` is omitted, ORIOLE uses a single endophenotype named `E`
connected to all traits.

### Params JSON format

You can use a minimal format for the common case (one endophenotype, no trait edges):

```json
{
  "trait_names": ["trait1", "trait2"],
  "mu": 1.0,
  "tau": 0.8,
  "betas": [0.2, -0.3],
  "sigmas": [0.4, 0.5]
}
```

For multiple endophenotypes and/or trait-to-trait edges, use the full format:

```json
{
  "trait_names": ["trait1", "trait2", "trait3"],
  "endo_names": ["E1", "E2"],
  "mus": [1.0, 0.5],
  "taus": [1.2, 0.8],
  "betas": [
    [0.2, 0.0],
    [0.0, -0.3],
    [0.1, 0.2]
  ],
  "sigmas": [0.4, 0.5, 0.3],
  "trait_edges": [
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, -0.2, 0.0]
  ]
}
```

`trait_edges[i][j]` is the coefficient for the edge `trait_j -> trait_i`. If omitted,
ORIOLE treats all trait-to-trait edges as zero.

### GWAS file format

The GWAS files must have a header row containing the ID, effect, and SE columns.
Supported delimiters: `;`, tab, `,`, or single spaces.

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
  Preserves a legacy initialization behavior (uses betas to estimate SE statistics).

- `--chunk-size <N>`
  Number of variants to process per chunk. Default targets ~2GB RAM. Use `1` to disable vectorization.

- `-d, --dry`
  Dry run (load data, check config, but do not train/classify).

---

## Test data included

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
- `trait_edges_trait1.tsv`, `trait_edges_trait2.tsv`, `trait_edges_trait3.tsv`
- `trait_edges_ids.txt`
- `trait_edges_params.json`
- `trait_edges_config_classify.toml`

---

## Tests

From the repository root:

```bash
pytest -q
```

---

## Full usage reference

### Train

```bash
oriole train -f config.toml [--gibbs] [--chunk-size N] [--match-rust]
```

### Classify

```bash
oriole classify -f config.toml [--gibbs] [--chunk-size N]
```

---

## More complex models

### Multiple endophenotypes

Add `[[endophenotypes]]` blocks to your config and assign each endo to a subset of
traits. Traits listed under an endo have free betas; missing traits are masked.

```toml
[[endophenotypes]]
name = "E1"
traits = ["fi", "isiadj", "bmi"]  # subset of gwas names, or ["*"] for all traits

[[endophenotypes]]
name = "E2"
traits = ["bmi", "whradj", "bfp"]
```

### Trait-to-trait DAG edges

You can also specify directed edges between traits. These edges must form a DAG.
Add them to the config via `[[trait_edges]]` blocks:

```toml
[[trait_edges]]
parent = "trait1"
child = "trait2"

[[trait_edges]]
parent = "trait2"
child = "trait3"
```

The corresponding coefficients live in the `trait_edges` matrix in the params JSON.
When `[[trait_edges]]` are present, training estimates the coefficients for the
allowed edges alongside the endophenotype loadings. Edges not listed in the config
remain fixed at 0.
