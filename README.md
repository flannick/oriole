# ORIOLE

ORIOLE is a linear-Gaussian model for joint inference of latent endophenotypes and
trait effects from GWAS summary statistics. It supports analytical inference and
Gibbs sampling, multiple endophenotypes, and directed edges between traits (a DAG).

## Quick Start

### 1) Install

Tested with Python 3.10+ (recommended). Python 3.7 is not supported because
`numpy>=1.23` requires a newer interpreter.

```bash
# from src/oriole
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

If you do not have root (virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

Dependencies: `numpy`, `tomli`, `tomli-w`. Tests use `pytest`.

Note: avoid naming a local package `math` (it shadows the Python stdlib). ORIOLE
internal math helpers live under `oriole/math_utils/`.

### 2) Run the included sample configs (recommended first run)

The sample TOML configs under `tests/data/` use relative paths. The simplest
way to run them is to change into that directory first:

```bash
cd tests/data
oriole train -f sample_config_train.toml
oriole classify -f sample_config_classify.toml
```

### 3) Create a config

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

### Configuration reference (all options)

Below are all configuration options, their defaults if omitted, and when you
might override them.

`[files]`
- `params` (default: `params_out.json` next to the config file) sets where the
  trained parameters are written; override to keep runs separated.
- `trace` (default: none) writes per-iteration parameter traces; enable when
  diagnosing convergence or debugging optimization.

`[[gwas]]`
- `name` (required) names the trait; should match other sections and params.
- `file` (optional) GWAS TSV path on disk.
- `uri` (optional) GWAS source URI. Use `dig-open-data:<ancestry>:<trait>` for
  DIG Open Data, or a fully qualified URI like `file://`, `s3://`, or
  `registry://`. If both are set, `uri` takes precedence.
- `cols.id` (default: none) column name for variant ID; set if not `VAR_ID`.
- `cols.effect` (default: none) column name for effect size; set if not `BETA`.
- `cols.se` (default: none) column name for standard error; set if not `SE`.
  Intuition: correct column mapping is essential for accurate likelihoods.

`[data_access]`
- `retries` (default: `3`) retry count for remote streams.
- `download` (default: `false`) download remote files to a temp path before
  reading (useful for very large gz files or flaky streams).

`[endophenotypes]`
- If omitted, a single endophenotype `E` connects to all traits (`"*"`).
- `name` (required) names the endophenotype; useful when modeling multiple
  latent axes.
- `traits` (required) is a list of trait names or `["*"]` for all traits; use
  to restrict specific endophenotypes to subsets of traits.

`[trait_edges]`
- `parent` / `child` (default: none; list can be empty) define directed edges
  between traits. Intuition: encode known trait-to-trait causal ordering.

`[outliers]`
- `enabled` (default: `false`) toggles the outlier model.
- `kappa` (default: `4.0`) inflates variance for outlier traits; higher means
  more downweighting of isolated spikes.
- `expected_outliers` (default: `0.16 * n_traits`) expected number of outlier
  traits per variant; higher makes outlier handling more aggressive.
- `pi` (default: `0.16`) prior probability a trait is an outlier; use only if
  you prefer the per-trait parameterization (mutually exclusive with
  `expected_outliers`).
- `pi_by_trait` (default: none) per-trait overrides of `pi`; use if some traits
  are noisier than others.
- `max_enum_traits` (default: `12`) maximum traits for exact enumeration; lower
  values trade accuracy for speed in large trait sets.
- `method` (default: none) overrides inference method for outlier scoring
  (`analytic`, `variational`, or `gibbs`); use for reproducibility.

`[tune_outliers]`
- `enabled` (default: `false`) enables CV-based tuning.
- `mode` (default: `off`) controls tuning strategy: `off`, `fast`, `full`,
  `auto`, or `on`. `off` means no CV; `auto` chooses fast vs full based on size;
  `on` is shorthand for `auto`.
- `n_folds` (default: `5`) number of CV folds; higher is more stable but slower.
- `seed` (default: `1`) RNG seed for fold splits and sampling.
- `kappa_grid` (default: `[4.0]`) candidate kappas (centered on defaults).
- `expected_outliers_grid` (default: `[0.16 * n_traits]`) candidate expected
  outliers (centered on defaults).
- `min_grid_length` (default: `10`) minimum grid length per axis before boundary
  expansion; expands both directions from the center value.
- `max_grid_length` (default: `30`) max grid length per axis after expansion.
- `genomewide_ids_file` (default: none) list of IDs to sample background from.
- `negative_ids_file` (default: none) optional hard negatives for CV scoring.
- `n_background_sample` (default: `100000`) background sample size; larger gives
  more stable tail metrics but increases runtime.
- `n_negative_sample` (default: `50000`) hard-negative sample size.
- `fpr_targets` (default: `[1e-3]`) tail thresholds for the CV metric.
- `lambda_hard` (default: `0.0`) weight for hard-negative penalty.
- `expand_on_boundary` (default: `true`) expands the grid when the boundary is
  close to best.
- `max_expansions` (default: `0`, meaning no cap) max expansion rounds.
- `expansion_factor` (default: `2.0`) multiplicative step for expansion.
- `force_expansions` (default: `false`) always expand upward for validation.
- `boundary_margin` (default: `0.01`) expands if boundary is within 1% of best;
  use smaller values for stricter expansion.

`[train]`
- `ids_file` (required) list of positive variants to train on.
- `n_steps_burn_in` (default: `1000`) Gibbs burn-in steps per iteration; only
  used for Gibbs-based expectations.
- `n_samples_per_iteration` (default: `100`) Gibbs samples per EM iteration.
- `n_iterations_per_round` (default: `10`) Gibbs iterations per EM round.
- `n_rounds` (default: `1`) number of EM iterations; higher improves fit but
  increases runtime.
- `normalize_mu_to_one` (default: `false`) keeps endophenotype mean scaled; only
  meaningful when `learn_mu_tau = true`.
- `learn_mu_tau` (default: `false`) learns `mu` and `tau` during training; the
  recommended default is to keep them fixed.
- `mu` (default: `0.0`) fixed prior mean for endophenotypes when not learned.
- `tau` (default: `1.0`) fixed prior std for endophenotypes when not learned.
- `params_trace_file` (default: none) explicit trace output path.
- `t_pinned` (default: none) optionally pins the trait residual variance.
- `plot_convergence_out_file` (default: none) writes a convergence plot.
- `plot_cv_out_file` (default: none) writes a CV grid plot.
- `early_stop` (default: `false`) stops when parameters (and objective, if
  available) stabilize.
- `early_stop_patience` (default: `3`) consecutive stable iterations required.
- `early_stop_rel_tol` (default: `1e-4`) relative parameter change tolerance.
- `early_stop_obj_tol` (default: `1e-5`) objective change tolerance.
- `early_stop_min_iters` (default: `5`) minimum iterations before stopping.

`[classify]`
- `params_override` (default: none) overrides parameters during classification.
  Supports `mu`, `tau`, `mus`, `taus`, `mus_by_name`, `taus_by_name`; use to
  test sensitivity without retraining.
- `mu` (default: `0.0`) prior mean used during classification when no override
  is specified.
- `tau` (default: `1e6`) prior std used during classification when no override
  is specified (effectively non-informative).
- `n_steps_burn_in` (default: `1000`) Gibbs burn-in steps; only used for Gibbs.
- `n_samples` (default: `1000`) Gibbs samples for classification.
- `out_file` (default: `classify_out.tsv`) output scores path.
- `write_full` (default: `true`) writes the full ORIOLE classify TSV. Set to
  `false` if you only want the GWAS-SSF output.
- `gwas_ssf_out_file` (default: none) optional GWAS-SSF output path. When set,
  ORIOLE writes GWAS-SSF formatted results from the GLS estimates.
- `gwas_ssf_guess_fields` (default: `true`) guesses chromosome/position/alleles
  from `VAR_ID` and auto-detects common metadata column names. Set to `false`
  to only use explicitly configured columns.
- `gwas_ssf_variant_id_order` (default: `effect_other`) controls how ORIOLE
  interprets `VAR_ID` formats like `1_752566_A_G`. Use `other_effect` if your
  IDs are encoded as `chrom_pos_other_effect`.
- `trace_ids` (default: `[]`) variant IDs to trace during classification.
- `t_pinned` (default: none) optionally pins trait residual variance.

### CLI reference (all flags)

`oriole train` and `oriole classify`
- `-f/--conf-file` (required) config file path.
- `-d/--dry` (default: false) validate inputs without running.
- `--verbose` (default: false) more logging.
- `--match-rust` (default: false) match Rust implementation for parity checks.
- `--inference {auto,analytic,variational,gibbs}` (default: `auto`) controls
  expectation method; use `variational` for speed, `analytic` for determinism,
  `gibbs` for sampling-based accuracy.
- `--gibbs` (default: false) shortcut for `--inference gibbs`.
- `--chunk-size` (default: auto) controls batch size for analytic/variational;
  increase to speed up if you have RAM.
- `--plot-convergence-out-file` (default: none) writes a convergence plot.
- `--plot-cv-out-file` (default: none) writes the CV grid plot.

`oriole import-phenet`
- `-i/--phenet-file` (required) phenotype network input.
- `-p/--params-file` (required) ORIOLE params to write.
- `-f/--conf-file` (required) config file path.
- `-o/--out-file` (required) output path.
  Intuition: use this to convert external phenotype-network inputs into ORIOLE
  params for downstream classification.

`oriole scale-sigmas`
- `-i/--in-file` (required) input params JSON.
- `-s/--scale` (required) scale factor for trait residuals.
- `-o/--out-file` (required) output params JSON.
  Intuition: use to calibrate trait noise when integrating new datasets.

## Model Overview

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

## Classification Output Columns

Each classify output TSV includes one row per variant and the following fields:

`id`
- Variant identifier.

Per endophenotype `{endo}` (in this order):
- `{endo}_mean_post`: posterior mean of E.  
  In analytic/variational modes this is the closed‑form posterior mean.  
  In Gibbs mode this is the Monte Carlo posterior mean.
- `{endo}_std_post`: posterior standard deviation of E.  
  In analytic/variational modes this is the closed‑form posterior std.  
  In Gibbs mode this is the Monte Carlo posterior std.
- `{endo}_mean_calc`: analytically computed posterior mean of E.  
  In analytic/variational modes this matches `{endo}_mean_post` (up to numeric noise).  
  In Gibbs mode this stays analytic and is provided as a deterministic reference.
- `{endo}_beta_gls`: GLS estimate of E from the collapsed likelihood `p(O|E)` (no prior).
- `{endo}_se_gls`: GLS standard error.
- `{endo}_z_gls`: `{endo}_beta_gls / {endo}_se_gls`.
- `{endo}_p_gls`: two‑sided normal p‑value from `{endo}_z_gls`.

Trait means:
- one column per trait, containing the posterior mean of the latent trait value `T_i` for that variant.

Guidance:
- Use `{endo}_mean_post` / `{endo}_std_post` when you want full Bayesian posterior summaries (these depend on `mu/tau`).
- Use `{endo}_beta_gls` / `{endo}_z_gls` / `{endo}_p_gls` for GWAS‑style, prior‑free summaries.
- In Gibbs runs, compare `{endo}_mean_post` vs `{endo}_mean_calc` to assess sampling noise.

---

## Configuration File

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
# Optional GWAS-SSF metadata mappings:
# chrom = "CHR"
# pos = "BP"
# effect_allele = "A1"
# other_allele = "A2"
# eaf = "EAF"
# rsid = "RSID"

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
learn_mu_tau = true

[classify]
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

### Remote GWAS access (optional)

GWAS inputs can be local or remote **per file**. Use `[[gwas]].uri` to point to a
remote source and keep local files in `[[gwas]].file`.

For DIG Open Data, use the short URI form exposed by the `dig-open-data` package:

```toml
[[gwas]]
name = "bmi"
uri = "dig-open-data:EU:BMI"
```

You can mix local and remote entries in the same config. Local files still work
without any change.

If you need a fully qualified URI, you can also use `file://`, `s3://`, or
`registry://` in `[[gwas]].uri`.

The optional `[data_access]` block controls download behavior:

```toml
[data_access]
retries = 5
download = true
```

Optional metadata columns (used for GWAS-SSF output if present, or mapped via
`[gwas.cols]`):

- `CHR`: chromosome
- `BP`: base pair location
- `A1`: effect allele
- `A2`: other allele
- `EAF`: effect allele frequency
- `RSID`: rsID

If `gwas_ssf_guess_fields = true`, ORIOLE will also auto-detect common header
names and attempt to parse missing fields from `VAR_ID` formats like
`chr:pos:A1:A2` or `1_752566_A_G` using any non-alphanumeric delimiter (e.g.
`:`, `_`, `-`, `.`). Use `gwas_ssf_variant_id_order` to control whether the
last two tokens are treated as effect/other or other/effect.
If guessing is disabled, missing fields are omitted from the GWAS-SSF output.

### Variant ID list

The `train.ids_file` is a two-column file:

```
VAR_ID   WEIGHT
```

If no weight is provided, it defaults to 1.0.

---

## Flags

- `--inference {auto,analytic,variational,gibbs}`
  Select the inference method. Default is `auto` (analytic unless outliers are enabled
  and the model is large, in which case it uses variational).

- `--gibbs`
  Shortcut for `--inference gibbs`.

- `--plot-convergence-out-file <path>`
  Write a convergence plot of parameters across iterations.

- `--plot-cv-out-file <path>`
  Write a cross-validation score grid plot for outlier tuning.

- `--match-rust`
  Preserves a legacy initialization behavior (uses betas to estimate SE statistics).

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
- `ssf_trait1.tsv`
- `ssf_params.json`
- `ssf_config_classify.toml`
- `trait_edges_trait1.tsv`, `trait_edges_trait2.tsv`, `trait_edges_trait3.tsv`
- `trait_edges_ids.txt`
- `trait_edges_params.json`
- `trait_edges_config_classify.toml`
- `outliers_config_train.toml`
- `outliers_config_classify.toml`
- `outliers_trait_edges_config_train.toml`
- `outliers_trait_edges_config_classify.toml`

---

## Tests

From `src/oriole`:

```bash
pytest -q
```

---

## Full Usage Reference

### Train

```bash
oriole train -f config.toml [--inference auto|analytic|variational|gibbs] [--gibbs] [--chunk-size N] [--match-rust]
```

### Classify

```bash
oriole classify -f config.toml [--inference auto|analytic|variational|gibbs] [--gibbs] [--chunk-size N]
```

---

## More Complex Models

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

### Outlier trait indicators (variance inflation)

ORIOLE can add a binary indicator `Z_i^j` for each trait `i` and variant `j` that
inflates the trait-layer residual variance by `kappa^2` when the trait behaves as an
outlier. This downweights isolated trait spikes that do not match the endophenotype
pattern.

To enable, add an `[outliers]` block to your config:

```toml
[outliers]
enabled = true
kappa = 4.0
expected_outliers = 0.03
max_enum_traits = 12
method = "auto" # auto, analytic, variational, or gibbs
# Optional per-trait overrides:
# pi_by_trait = { fi = 0.01, bmi = 0.02 }
```

Notes:
- `kappa` must be >= 1.0 (values > 1.0 are required for any effect).
- `expected_outliers` is the expected number of outlier traits per variant.
- `pi` is the per-trait prior; it is mutually exclusive with `expected_outliers`.
- `pi_by_trait` overrides `pi` for named traits.
- `max_enum_traits` controls when analytic enumeration is allowed.

Inference controls:
- Use `--inference` to force `analytic`, `variational`, or `gibbs`.
- `auto` uses variational when outliers are enabled; otherwise it uses analytic.
- `--gibbs` is a shortcut for `--inference gibbs`.

### GWAS-style GLS outputs for endophenotypes

Classification outputs now include GWAS-style GLS summaries for each endophenotype:
`{endo}_beta_gls`, `{endo}_se_gls`, `{endo}_z_gls`, `{endo}_p_gls`.
These are derived from the collapsed likelihood `p(O | E)` by treating `E` as a
per-variant parameter and computing a generalized least squares estimate.

Intuition:
- `*_mean_*` columns are Bayesian posterior summaries that include the prior.
- `*_gls` columns are plug-in GLS summaries that are more comparable to standard
  GWAS effect estimates (no prior shrinkage in the GLS step).

Outlier handling:
- `analytic` enumeration computes GLS by averaging over `Z` states with exact
  posterior weights.
- `variational` uses an effective covariance with `E[1/c(Z)]` to compute GLS.
- `gibbs` uses the mean sampled `Z` (per trait) to build an effective covariance;
  this is a Monte Carlo approximation consistent with the sampling run.

### GWAS-SSF output (streamlined)

If you set `classify.gwas_ssf_out_file`, ORIOLE writes an additional GWAS-SSF
file derived from the GLS outputs. This file is useful for downstream tools
that expect standard GWAS summary statistics.

Columns included (when available):

- `variant_id` (always)
- `chromosome`, `base_pair_location`, `effect_allele`, `other_allele`, `rsid`
- `effect_allele_frequency` (averaged across traits when multiple values exist)
- `beta`, `standard_error`, `p_value` (from GLS)

If multiple endophenotypes are defined, ORIOLE writes one GWAS-SSF file per
endophenotype by appending `_{endo}` to the output filename (preserving `.gz`).
Set `classify.write_full = false` if you only want the GWAS-SSF output.

### Early stopping for training

Analytical/variational training can stop early when parameters (and the marginal
likelihood) stop changing. Add these fields under `[train]`:

```toml
[train]
early_stop = true
early_stop_patience = 3
early_stop_rel_tol = 1e-4
early_stop_obj_tol = 1e-5
early_stop_min_iters = 5
```

Notes:
- The relative-parameter change is checked every iteration.
- The marginal likelihood objective check is only available when outliers are
  disabled; otherwise ORIOLE falls back to the parameter criterion.

### Tuning kappa and expected outliers by cross-validation

To select `(kappa, expected_outliers)` using cross-validation, add a
`[tune_outliers]` block.
This uses a tail-focused metric based on the endophenotype score
`S = m_E^T V_E^{-1} m_E` and uses a one-standard-error rule to choose the
smallest `expected_outliers` and then the smallest `kappa` within one SE of
the best mean score.

```toml
[tune_outliers]
enabled = true
mode = "auto" # auto | full | fast | off
n_folds = 5
seed = 1
kappa_grid = [4.0]
expected_outliers_grid = [1.44]
min_grid_length = 10
max_grid_length = 30
genomewide_ids_file = "all_ids.txt"
negative_ids_file = "hard_negs.txt" # optional
n_background_sample = 200000
n_negative_sample = 50000
fpr_targets = [1e-3, 1e-4]
lambda_hard = 1.0
expand_on_boundary = true
max_expansions = 0
expansion_factor = 2.0
force_expansions = false
boundary_margin = 0.01
```

Notes:
- Tuning is off by default; set `enabled = true` and `mode` to `auto`, `fast`,
  `full`, or `on` to activate it. `on` is a shorthand for `auto`.
- `mode="full"` retrains within each fold (slow but principled).
- `mode="fast"` trains once and tunes `(kappa, expected_outliers)` with fixed
  parameters.
- `mode="auto"` uses `full` only for small, no-edge models; otherwise `fast`.
- If any boundary value is within `boundary_margin` (fractional, default 1%)
  of the best score, the grid expands by adding a full row and/or column in the
  direction of that boundary. Expansion continues until no boundary is within
  the margin or the grid reaches `max_grid_length` per axis. Set `max_expansions`
  to cap expansions (0 means no cap).
- Set `force_expansions = true` to expand the upper bounds every time (ignoring
  boundary checks). This is only for validation/debugging.
- If `mode` is not `off`, ORIOLE always runs CV. The center of the CV grid is
  the specified `kappa`/`expected_outliers` if present; otherwise it uses the
  built‑in defaults.
- The tuner errors out if the requested background/negative sample sizes exceed
  the available IDs, and it warns if a hard-negative file yields no usable IDs.

---
