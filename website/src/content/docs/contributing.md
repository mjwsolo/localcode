---
title: Contributing
description: How to set up a development environment and open a PR.
---

localcode is Apache-2.0. Contributions are welcome. Good changes make the default local workflow better, setup easier, or behaviour more reliable. Avoid adding complexity without a clear user benefit.

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

## Pull requests

- Keep each PR to one change.
- Say what the user will notice.
- Add tests for behaviour changes when practical.
- Do not mix refactors with bug fixes.
- Do not commit caches, model files or build output.

For architecture changes, open an issue first.

## Working on these docs

The site lives in `website/` and uses Astro with Starlight:

```sh
cd website
npm install
npm run dev
```

The canonical files are [`CONTRIBUTING.md`](https://github.com/mjwsolo/localcode/blob/main/CONTRIBUTING.md) and [`SECURITY.md`](https://github.com/mjwsolo/localcode/blob/main/SECURITY.md) in the repository.
