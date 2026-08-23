# Contributing To LocalCode

## Scope

LocalCode is an alpha-stage local coding assistant. Contributions are welcome, but changes should keep the product pragmatic:

- improve the default local workflow
- reduce setup friction
- improve reliability and clarity
- avoid adding broad complexity without a clear user benefit

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Recommended extra tools:

```bash
pip install pytest ruff build
```

## Before Opening a PR

Run:

```bash
ruff check src tests
pytest
python -m build --no-isolation
```

## Pull Request Guidelines

- keep PRs focused
- explain user-facing impact
- include tests for behavior changes when practical
- avoid mixing unrelated refactors with bug fixes
- do not commit local caches, model files, or generated junk

## Runtime Changes

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

For architecture changes, open an issue first so the direction can be discussed before implementation.
