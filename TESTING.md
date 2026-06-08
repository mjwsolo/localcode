# Testing

## Quick start

```bash
pytest tests/test_comprehensive_*.py        # the required suite — fast, no model download
pytest                                       # everything (incl. legacy — see triage below)
```

## The comprehensive suite (required in CI)

These run the **real** app — real agent loop, real tools, real disk, real
TUI — with only the *model* faked (a scripted `FakeRuntime`, see
`tests/e2e/fake_runtime.py`). Green here means core functionality works,
deterministically, on every push.

| File | Covers |
|---|---|
| `test_comprehensive_agent.py` | Agent loop, every tool, multi-round tool use, thinking, errors |
| `test_comprehensive_cli.py` | All non-interactive CLI commands (subprocess) |
| `test_comprehensive_tui.py` | Live Textual app via headless driver: keystroke → response, tool turns, slash commands |
| `test_comprehensive_vision.py` | Image wire-format per provider + capability flags |
| `test_comprehensive_voice.py` | STT/TTS logic (whisper + `say` mocked) |
| `test_comprehensive_machines.py` | Model recommendation / GPU-offload / promotion across 8→128 GB RAM |
| `test_comprehensive_download.py` | Download orchestration: retry, fallback, fatal-error fast-fail (network mocked) |
| `test_comprehensive_whisper.py` | **Real** whisper.cpp STT (gated) |
| `test_comprehensive_install.py` | **Real** `pip install` into a clean venv (gated) |

### Gated (opt-in) tests

Heavy tests skipped by default; enable with an env var:

```bash
LOCALCODE_RUN_INSTALL_TEST=1   pytest tests/test_comprehensive_install.py   # real venv install
LOCALCODE_RUN_WHISPER_TEST=1   pytest tests/test_comprehensive_whisper.py   # needs pywhispercpp + 540 MB model
LOCALCODE_RUN_REAL_DOWNLOAD=1  pytest tests/test_comprehensive_download.py  # downloads a real GGUF (multi-GB)
```

The macOS CI job runs the install test automatically (real platform check).

## CI layout (`.github/workflows/ci.yml`)

- **comprehensive (linux)** — required, py 3.11–3.13.
- **comprehensive (macos-arm64)** — required, runs on Apple Silicon (the user
  platform) + real isolated install.
- **build** — packaging must build.
- **legacy suite (non-blocking)** — runs the older tests for visibility; does
  not fail CI yet (see triage). Flip `continue-on-error: false` once resolved.

## Legacy test triage (22 pre-existing failures)

These predate the comprehensive suite and fail on the current branch. They
fall into three buckets — **do not bulk-delete**, two buckets are valuable:

### A. Possible REAL regressions — investigate before touching
- `test_runtime.py::test_large_qwen_on_16gb_*` (×3) — expect context clamped to
  **64K** on 16 GB Macs; code now emits **128K**. If the clamp was removed,
  16 GB machines running the 35B model may OOM. Decide: is 128K intended?
- `test_runtime.py::test_custom_binary_used` — expects a user-set
  `llama_cpp_binary` to be honored; code uses the bundled binary. Is custom
  binary override meant to still work?
- `test_fresh_install.py::test_download_uses_parallel`,
  `test_fresh_install.py::test_llama_server_command` — assert download path /
  server binary path that changed.

### B. Quality guardrails — fix the code or formally accept
- `test_architecture.py` (×4): 400-LoC file cap, no bare `print()` in prod,
  `agent/` submodules need `__all__`, legacy-big-file growth cap. These fail
  because of accumulated debt, not because they're wrong.

### C. Genuinely stale — safe to delete
- `test_agent_event_regressions.py` tool-set tests (×4) assert a fixed tool set
  including `append_file`, a tool that no longer exists.
- `test_agent_event_regressions.py` completion-gate / quality-monitor tests
  (×8) expect gate strings, but `completion_gate(...)` now returns `None` —
  the feature appears removed/disabled. Confirm it's gone, then delete.

Once each bucket is resolved (fix code for A/B, delete C), flip the legacy CI
job to blocking.
