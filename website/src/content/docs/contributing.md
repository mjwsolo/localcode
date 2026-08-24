---
title: Contributing
description: How to set up a development environment and open a PR.
---

localcode uses the Apache-2.0 license. Contributions are welcome. Changes should keep the product practical. They should improve the default local workflow, make setup easier, improve reliability and clarity, and avoid broad complexity unless it clearly helps users.

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
- Explain how the change affects users.
- Add tests for behaviour changes when practical.
- Do not mix unrelated refactors with bug fixes.
- Do not commit local caches, model files, or generated artefacts.

If a change affects model or runtime setup, document the supported provider, supported platform, and failure mode.

## Working on these docs

This site is in `website/`. It uses Astro + Starlight:

```sh
cd website
npm install
npm run dev
```

See `website/README.md` for full preview instructions. It also lists the pages that are still marked as stubs.

The official files in the repository are [`CONTRIBUTING.md`](https://github.com/mjwsolo/localcode/blob/main/CONTRIBUTING.md) and [`SECURITY.md`](https://github.com/mjwsolo/localcode/blob/main/SECURITY.md).
