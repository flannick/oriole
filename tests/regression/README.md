# Regression Fixtures

This directory holds small, deterministic inputs and baseline outputs for regression testing.

- `inputs/` contains tiny GWAS and ID files.
- `config_train.toml` and `config_classify.toml` point at those inputs.
- `expected/` contains the baseline outputs produced by Oriole.

To refresh baselines (after an intentional model change):

```bash
# from src/oriole
/Users/flannick/codex-workspace/analysis/.venv/bin/python -m oriole train -f tests/regression/config_train.toml
/Users/flannick/codex-workspace/analysis/.venv/bin/python -m oriole classify -f tests/regression/config_classify.toml
cp params_out.json tests/regression/expected/params.json
cp classify_out.tsv tests/regression/expected/classify.tsv
```

Note: keep analytical inference as the default (no `--gibbs`) so baselines are deterministic.
