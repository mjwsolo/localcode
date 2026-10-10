---
title: Compatibility and hardware verification
description: What CI proves, how to test a physical Mac, and how to diagnose installs.
---

## Automated checks

Every pull request runs the full non-model Python suite, including previously
excluded runtime, KV memory, thermal, recovery and loopback download tests.
Only the four external Hugging Face tests in `test_download.py` remain opt-in;
its five local tests run by default, including genuine skip-existing assertions. Real-model and
explicit opt-in network/audio tests remain separate. A stale speculative-decoding
flag assertion was corrected against the current shipped server flag.

- Linux: Python 3.10–3.14.
- Apple silicon hosted runners: macOS 15 and 26 with Python 3.10, 3.13 and 3.14.
- Built-wheel installations on both macOS images: pip and uv tool, fresh and
  upgrades from 0.4.9 and 0.5.2. Tests run outside the checkout, verify the resolved
  executable and package version, and execute both installed binaries.
- After publishing: pip and uv install the exact PyPI version, upgrading 0.4.9.
  A failed post-publication check marks the workflow failed; it cannot undo a
  PyPI upload. Fix forward or explicitly decide whether to yank the release.
- Simulated model/RAM tests cover 8–192 GiB, including the shipping server command.
  They verify configuration and definite over-capacity rejection, not physical fit.

The available images are listed in [GitHub's runner documentation](https://github.com/actions/runner-images).
The macOS 14 hosted jobs have been retired ahead of GitHub's November 2, 2026
shutdown. Standalone plugin, upstream-candidate and published-package checks
now run on macOS 15. macOS 13 remains the binary deployment target; macOS 13
and 14 require local verification because neither has a hosted row here. Hosted macOS versions are not a matrix of all M-series chips.
An actual Intel macOS runner verifies wheel rejection and the launcher error.
Rosetta and old macOS error paths are tested with simulated platform probes.

## Physical Mac gate (local only)

Download suitable models through LocalCode first. Close other memory-heavy apps,
then run from a source checkout with its dependencies installed:

```sh
caffeinate -i python scripts/verify_hardware.py \
  --models gemma-12b qwen --cycles 2 --prompt-tokens 2048
```

Use models that fit the machine. This command downloads nothing and registers
no runner. It launches only disposable local model servers and closes them on
completion. It uses the shipping Supervisor to load and switch models, executes
real chat/tool/runtime turns, sends a longer synthetic prompt, kills its own
model process, verifies the unavailable state, and reloads through the Supervisor.
It does not kill the user's existing LocalCode session or deliberately exhaust RAM.

Results stay in ignored `.localcode-gate/hardware.json`: chip, OS, physical RAM,
commit/dirty state, latency, checks, peak server RSS, and system swap growth.
RSS is sampled and does not measure every Metal allocation. Swap is system-wide
and can include other apps. These measurements are evidence, not a guarantee
that any model fitting the weight budget will run at every context length.
Optional `--max-rss-gb` and `--max-swap-growth-gb` enforce chosen limits.
Increase `--prompt-tokens` to test longer contexts within the actual per-slot budget.
Failures return nonzero and retain a local receipt.

Maintain real receipts for representative 8 GB, 16/24 GB, 32/48 GB and 64+ GB
machines, including an older chip and a current chip. An 8 GB run must use a
suitably small quant; it may correctly reject all locally downloaded catalog
choices. Do not label an untested hardware tier as verified. Simulating less
RAM on a large Mac is not equivalent to owning that smaller Mac.

The stricter model promotion gate remains:

```sh
python scripts/verify_models.py --strict --publish-status
```

It publishes only the exact-commit pass/fail status. The physical-Mac report
stays local unless you deliberately share it.

## Installation troubleshooting

```sh
localcode doctor
which -a localcode
python3 -m pip show localcode
uv tool list
```

`doctor` is offline and reports package/Python versions, executable paths,
architecture, macOS, RAM and whether proxy/index/certificate settings exist.
It does not print their values. Paths can identify your account, so redact
those before sharing outside your team. It is available in versions containing
this change; use the other commands when troubleshooting older installations.

A successful installation of an older release does not prove the latest one
is compatible. Pin the desired version to expose resolver errors. Check stale
PATH entries, cached corporate package indexes and architecture before changing
versions. Ask IT to configure approved proxy credentials and corporate CA trust;
do not disable certificate verification.
