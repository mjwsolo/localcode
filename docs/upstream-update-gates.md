# Updating the bundled runtimes

LocalCode ships pinned llama.cpp and OpenCode binaries. Updates become candidate
pull requests; the installed runtime does not download untested code at startup.
Daily hosted workflows detect changes and build candidates. A model gate runs
locally on an Apple Silicon Mac before promotion. GitHub never connects to that Mac.

## Local verification

Check out the candidate commit in a clean checkout and install the development
extra (`pip install -e ".[dev]"`). Download the required catalog models first;
the gate never downloads weights. Then run:

```bash
python scripts/verify_models.py --strict --publish-status
```

The gate launches the bundled server with `localcode.ui.server_cmd.server_command`,
including the configured context, vision projector, reasoning policy, cache and
warm-up flags. It replaces only the executable path with the candidate's bundled
binary. Servers have fresh authentication keys, private temporary directories,
unused loopback ports and bounded load/request times, and are reaped on failures.
The running application and its files are not changed.

Required keys: `gemma-12b`, `qwen`, `qwen38`, `diffusiongemma`, `north-mini-code`,
`muse-glimmer`, `kolibri`, `openjev`. Additional downloaded catalog quants run too.
Checks cover rejected unauthenticated inference, template rendering, chat,
well-formed tool calls with correct arguments, consumption of tool results,
System One choice/boolean/score response contracts, and an additional turbo4 run.
Qwen 3.8 and OpenJev also run through the shipping OpenCode binary and plugin.
Decision probabilities are checked for validity, not used as safety approvals.
This is a compatibility gate, not a coding-quality benchmark or exhaustive eval.

To test before pushing, run `--strict` without publishing, push the tested
commit, then run `--publish-receipt`. That verifies the clean commit, binary
hashes, model coverage, runtime checks and turbo4 result before publishing;
a stale or incomplete receipt is refused.

For a focused development run use `--models openjev`; that cannot publish a
promotion status. The pytest `real_models` tier tests downloaded models and may
skip on machines without weights; it is not a substitute for the strict gate.

Receipts go to `.localcode-gate/receipt.json` (gitignored), recording the exact
commit, binary hashes and checks. `--publish-status` requires the full strict
model set and a clean tracked checkout. It uses your local `gh` credentials to
publish a `local model gate` commit status, never a remote runner. After publishing,
rerun the **Model promotion gate** workflow for the PR. The check accepts only a
successful status on the current head commit. A new commit needs a fresh gate.
Maintainers should require **model promotion gate** in branch protection and
merge only when the standard CI checks pass too. Statuses attest the local
operator's run; they are not cryptographic proof of hardware execution.

## Candidate builds

- llama.cpp: `bash scripts/bump_upstream.sh --ref master --sync`, from a neutral
  build directory. Replays the maintained patch series and checks static linking,
  Metal registration and embedded paths before installing a candidate binary.
- OpenCode: `bash scripts/bump_opencode.sh localcode`. The maintained fork must
  already contain current upstream `dev`. Otherwise the workflow reports that
  the fork needs merging; it never drops LocalCode changes to force an update.
  The exact fork commit is built with a frozen lockfile and install scripts off.

The build job has read-only repository permissions and no stored checkout
credential. The separate writer job applies only permitted runtime/vendor paths
and creates a PR; it executes no upstream code. Build failures become a rolling
issue with a link to the log. No candidate auto-merges on build success.

## Release

After local verification and hosted CI pass, merge the candidate. Update both
package versions and the changelog, rebuild the UI if the embedded version
changes, test, and tag the merged main commit. The publish workflow creates the
PyPI package and GitHub release. Never tag an unmerged feature branch.
