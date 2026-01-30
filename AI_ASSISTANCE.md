# AI Assistance

## Tools Used
- OpenAI Codex (CLI)

## Time Window
- January 2026

## Scope of AI-Assisted Changes
- Code and tests under `src/oriole/`
- Data/config/test fixtures under `src/oriole/tests/`
- Documentation updates under `src/oriole/README.md`
- Supporting artifacts under `reports/`, `results/`, and `figures/`

## Human Review Statement
All code reviewed, tested, and approved by maintainers.

## Reproducibility Evidence
Regression tests live under:
- `src/oriole/tests/regression/`
- `src/oriole/tests/regression_multi_e/`
- `src/oriole/tests/regression_trait_edges/`

Run them from the repo root:

```bash
PYTHONPATH=src/oriole/src /Users/flannick/codex-workspace/analysis/.venv/bin/python -m pytest -q src/oriole/tests
```

## Known Limitations
AI may introduce subtle bugs; treat as untrusted until reviewed.

