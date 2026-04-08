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

## Large Changes

For architecture changes, open an issue first so the direction can be discussed before implementation.
