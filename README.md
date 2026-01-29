# oriole (Python)

This is a Python rewrite of the original Rust `mocasa`/`oriole` tool.

## Layout

- `src/oriole/` — Python package source (src layout)
- `tests/` — tests
- `scripts/` — helper scripts (if needed)

## Install (editable)

```bash
python -m pip install -e .
```

## Usage

```bash
oriole train -f path/to/config.toml
oriole classify -f path/to/config.toml
oriole import-phenet -i phenet.opts -p params.json -f config.toml -o output.tsv
oriole scale-sigmas -i params.json -s 0.5 -o params_scaled.json
```

## Flags

- `--match-rust`: preserves the Rust behavior for initial parameter estimation
  (uses betas when estimating SE statistics).
- `--analytical`: use analytical posterior estimates for classify instead of Gibbs sampling.

## Notes

- Uses `numpy` for numeric operations and random sampling.
- Reads configs from TOML; writing TOML uses `tomli-w`.
