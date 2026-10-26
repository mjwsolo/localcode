---
title: Release process
description: One command takes a version from release candidate to published release, testing the package users install.
---

Bugs reached users because testing ran from the repository, on one big Mac, with
the developer's own settings. Users run the published package from a fresh install
on 16 to 128 GB Macs. A release therefore goes out as a **release candidate**
first, and every check runs against that candidate installed from PyPI, in a fresh
virtual environment with a fresh `HOME`.

## Cutting a release

On the Mac that has the models, from `main` (or a `release/*` branch) with a clean
tree that matches `origin`, and with `CHANGELOG.md` holding a `## 0.5.41` or
`## Unreleased` section:

```
scripts/release.sh 0.5.41 --model <installed model name>
scripts/release.sh 0.5.41 --model <name> --dry-run     # prints every step, runs nothing
```

It refuses to start if the tree is dirty, the branch is not `main` or `release/*`
or is not up to date with `origin`, another `llama-server` is running, the
changelog has no notes, or `v0.5.41` exists. Then:

1. **Candidate.** Picks the next free `0.5.41rcN` from the tags
   (`scripts/release_candidate.sh`), commits the bump, and pushes it to `main` with
   the tag `v0.5.41rcN`. This is the only push before promotion. `publish.yml`
   publishes it to PyPI as a pre-release and creates a GitHub release marked
   `--prerelease`. A candidate is never the default install: PEP 440 pre-releases
   are skipped by `pip install localcode` and `uv tool install localcode`. The
   release notes use the `rcN`, the final version or the `Unreleased` section of
   the changelog, so a candidate needs no heading of its own.
2. **Wait** for PyPI to serve it.
3. **Install** the candidate into a fresh venv. Everything below tests that install,
   each in a temporary `HOME`, XDG and `LOCALCODE_HOME`. The real models directory is
   passed explicitly, and the developer's `LOCALCODE_*` variables, config, UI
   preferences and sessions cannot change a result.
   - `localcode doctor`
   - the terminal snapshot suite against the installed `localcode-ui` and Python
     modules: `scripts/ui_e2e.sh --installed <venv python>`
   - the local model gate: `verify_models.py --installed --strict --publish-status`
     posts the commit status **local model gate** on the candidate commit
   - the release eval, `scripts/release_eval.sh`, posts **release eval**
4. **Promote.** When both statuses are green, the script dispatches
   `.github/workflows/release.yml` with the candidate. That run re-reads both
   statuses, runs the installed-package checks on a hosted macOS 15 runner (fresh
   venv and `HOME`, `localcode doctor`, the snapshot suite in installed mode;
   logs and diffs are uploaded), then bumps `pyproject.toml` and `__init__.py` to
   `0.5.41`, renames the changelog heading to `## 0.5.41 - <date>`, commits as
   `github-actions[bot]`, tags `v0.5.41` and starts `publish.yml`. If branch
   protection refuses the direct push, it opens a release PR, enables auto-merge and
   tags the merge commit (checks that run on pull requests must be able to start
   for the PR to merge).
5. **Post-publish checks** run automatically in `publish.yml` on macOS 15: the
   exact version is installed with pip and uv (including an upgrade from 0.4.9), and
   with pip into a fresh venv, where `localcode doctor` and the snapshot suite run
   against the installed package. `release.sh` waits for them.

If any check fails, the script stops before promotion, prints which check failed
and where its log is, and nothing is promoted. The candidate stays published as a
pre-release; fix the cause and run the script again, which uses the next `rcN`.

Pushing a `v*rc*` tag by hand also starts `release.yml`, which publishes the
candidate and runs the hosted checks, but never promotes: only the dispatch that
`release.sh` sends after both local statuses are green can.

## Using a candidate yourself

```
uv tool install --prerelease allow "localcode==0.5.41rc1"
pip install "localcode==0.5.41rc1"
```

An exact pin selects a pre-release without any flag in pip; uv needs
`--prerelease allow`. To go back to the stable release:
`uv tool install --reinstall localcode` (or `pip install --upgrade localcode`).

## The release eval

`scripts/release_eval.sh --model <name> [--candidate-venv DIR]` runs six fixed
Harness-Bench tasks (`002-monorepo-affected-build`, `010-layered-config-resolution`,
`035-workspace-dependency-resolution`, `092-bounded-cache-eviction-repair`,
`089-file-watcher-atomic-save-repair`, `120-http-cookie-jar-policy-repair`) twice
with the same local model: on the candidate, then on the previous release
installed from PyPI into a temporary venv. Each run has a fresh `HOME`. It prints
mean score, tasks passed and total wall time for both and exits 1 if the mean
score dropped by more than 0.05 or the run was more than 25% slower. It refuses to
start while any `llama-server` is running (GPU contention ruins timings) and never
downloads models; the Harness-Bench data must already be in `evals/data` (see
`evals/README.md`). `--dry-run` prints the plan.

## Why the model checks run on the Mac and not in CI

Hosted runners have about 7 GB of RAM and no models. The gate therefore runs on the
owner's Mac and records its result as a commit status, the same pattern the Model
promotion gate uses for pull requests. No self-hosted runner is registered and none
of the workflows can use one.

## What the workflows guard

`release.yml` has no `pull_request` trigger. Its jobs run only when
`github.repository_owner == github.actor` and the repository is not a fork. It uses
the workflow token only: tags and the release commit are pushed by
`github-actions[bot]`, and because refs pushed with that token do not start other
workflows, the promote job starts `publish.yml` explicitly with `workflow_dispatch`.
