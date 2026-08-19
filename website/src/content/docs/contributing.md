---
title: Contributing
description: How to set up a development environment and open a PR.
---

localcode is Apache-2.0. Contributions are welcome; the bar is that a change
keeps the product pragmatic — improve the default local workflow, reduce setup
friction, improve reliability and clarity, and avoid broad complexity without a
clear user benefit.

## Development setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest ruff build
```

## Before opening a PR

```sh
ruff check src tests
pytest
python -m build --no-isolation
```

## PR guidelines

- Keep PRs focused.
- Explain the user-facing impact.
- Include tests for behaviour changes when practical.
- Don't mix unrelated refactors with bug fixes.
- Don't commit local caches, model files, or generated artefacts.

If a change affects model or runtime setup, document the supported provider,
the supported platform, and the failure mode.

## Working on these docs

This site lives in `website/` and is built with Astro + Starlight:

```sh
cd website
npm install
npm run dev
```

See `website/README.md` for the full preview instructions, including which
pages are still marked as stubs.

The canonical files are [`CONTRIBUTING.md`](https://github.com/mjwsolo/localcode/blob/main/CONTRIBUTING.md)
and [`SECURITY.md`](https://github.com/mjwsolo/localcode/blob/main/SECURITY.md)
in the repository.
