# Contributing to localcode

Contributions are welcome. Good changes make the default local workflow better, setup easier, or behaviour more reliable. Avoid adding complexity without a clear user benefit.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest ruff build
```

## Before opening a PR

```bash
ruff check src tests
pytest
python -m build --no-isolation
```

## Pull requests

- Keep each PR to one change.
- Say what the user will notice.
- Add tests for behaviour changes when practical.
- Do not mix refactors with bug fixes.
- Do not commit caches, model files or build output.

If a change touches the model or runtime setup, document the platform it supports, how it fails, and what happens when it fails.

If a change affects model/runtime setup, document:

- supported provider
- supported platform
- failure mode
- fallback behavior

## The Vendored llama.cpp Fork

`llama-cpp-turboquant/` is **latest upstream llama.cpp plus `patches/*.patch`**, not
a snapshot. Do not hand-merge upstream into it. Add or rebase a patch in
`patches/`, and let `.github/workflows/upstream-bump.yml` replay the series.

A bump PR is never merged on green CI: CI cannot load a model, so
`bash dev/verify_models.sh` must be run locally and pasted into the PR.

Full details, including how to add, rebase, and drop a patch: [docs/upstream-fork.md](docs/upstream-fork.md).

## Large Changes

For architecture changes, open an issue first so the direction can be agreed before you write code.
