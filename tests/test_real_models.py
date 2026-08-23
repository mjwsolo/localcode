"""Real-inference tier — load each downloaded GGUF and run a real turn.

This is the test that would have caught the live "DiffusionGemma returned no
usable response" bug: it drives every model through the SAME path the agent
loop uses, including `num_predict = MAX_OUTPUT_TOKENS = -1` (a reconstruction
that passed a positive num_predict hid the bug for days).

Opt-in and machine-aware:
  * marked `real_models` → only runs under `pytest -m real_models`
  * each model auto-SKIPS when its GGUF isn't downloaded, so this file is
    safe in CI and on any machine — it tests whatever you actually have.

What it asserts per model:
  * chat turn → non-empty, not the error sentinel
  * tool turn → a tool call is produced (or, at minimum, real content)
  * reasoning models (cohere2_moe) → `/thinking off` yields 0 thinking events
  * diffusion → repeated runs to shake out non-determinism

Run locally:  pytest -m real_models -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import models_catalog as catalog
from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway

pytestmark = pytest.mark.real_models

# The exact value the agent loop passes (MAX_OUTPUT_TOKENS). Using -1 here is
# the whole point — it's what broke diffusion in the live app.
LIVE_NUM_PREDICT = -1
TEST_PORT = 8199  # distinct from the default 8081 so a running app is untouched
# Degenerate-output error codes a turn must NOT return: E3107 (diffusion gave
# no usable output), E3108 (model collapsed into junk tokens).
ERR_SENTINELS = ("[E3107]", "[E3108]")


def _has_error(content: str) -> bool:
    return any(s in content for s in ERR_SENTINELS)


def _model_path(choice) -> Path:
    return Path(catalog.model_dir()) / choice.filename


def _present_choices():
    return [c for c in catalog.CHOICES if _model_path(c).is_file()]


def _ids(c):
    return f"{c.key}"


def _toolkit_schemas():
    from localcode.toolkit import LocalCodeToolkit
    tk = LocalCodeToolkit(repo_root=os.getcwd(), config=RuntimeConfig(), app=None)
    return tk.schemas()


SYS = {"role": "system", "content": "You are LocalCode.\nWorking directory: " + os.getcwd()}


def _collect(gw, messages, tools, *, think=False):
    """Drive a real turn; return (content, tool_calls, thinking_chars)."""
    content, tcs, think_chars = [], [], 0
    for ev in gw.stream_chat_events(
        messages, tools=tools, think=think, num_predict=LIVE_NUM_PREDICT
    ):
        t = ev.get("type")
        if t == "content":
            content.append(ev.get("content", ""))
        elif t == "tool_calls":
            tcs = ev.get("tool_calls", [])
        elif t == "thinking":
            think_chars += len(ev.get("content", ""))
    return "".join(content).strip(), tcs, think_chars


class _Server:
    """Start the right backend for a model, wait for health, tear it down.

    Diffusion has no server (one-shot CLI) so this is a no-op for it.
    """
    def __init__(self, gw, model_path):
        self.gw = gw
        self.model_path = model_path
        self.proc = None

    def __enter__(self):
        arch = (catalog.by_filename(Path(self.model_path).name).architecture or "").lower()
        if "diffusion" in arch:
            return self  # no server needed
        cmd = self.gw.llama_server_command(self.model_path)
        # Force our test port.
        if "--port" in cmd:
            cmd[cmd.index("--port") + 1] = str(TEST_PORT)
        self.gw.config.base_url = f"http://localhost:{TEST_PORT}"
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://localhost:{TEST_PORT}/health"
        for _ in range(180):
            if self.proc.poll() is not None:
                raise RuntimeError("server process exited during startup")
            try:
                urllib.request.urlopen(url, timeout=2)
                return self
            except Exception:
                time.sleep(1)
        raise RuntimeError("server did not become healthy in 180s")

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except Exception:
                self.proc.kill()


def _gateway_for(choice) -> LocalCodeRuntimeGateway:
    cfg = RuntimeConfig()
    cfg.provider = "llama_cpp"
    cfg.model = str(_model_path(choice))
    # Wire the bundled diffusion runner if present (so the right backend is used).
    from localcode.bootstrap import diffusion_cli_path
    dp = diffusion_cli_path()
    if dp:
        cfg.diffusion_cli_binary = str(dp)
    return LocalCodeRuntimeGateway(cfg)


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("choice", _present_choices() or [None], ids=lambda c: _ids(c) if c else "no-models")
def test_real_chat_turn(choice):
    if choice is None:
        pytest.skip("no models downloaded — nothing to exercise")
    gw = _gateway_for(choice)
    tools = _toolkit_schemas()
    with _Server(gw, gw.config.model):
        content, tcs, _ = _collect(gw, [SYS, {"role": "user", "content": "hi"}], tools)
    assert not _has_error(content), f"{choice.key}: degenerate-output error code returned"
    assert content or tcs, f"{choice.key}: empty chat turn"


@pytest.mark.parametrize("choice", _present_choices() or [None], ids=lambda c: _ids(c) if c else "no-models")
def test_real_tool_turn(choice):
    if choice is None:
        pytest.skip("no models downloaded — nothing to exercise")
    gw = _gateway_for(choice)
    tools = _toolkit_schemas()
    with _Server(gw, gw.config.model):
        content, tcs, _ = _collect(
            gw, [SYS, {"role": "user", "content": "List the files in the current directory."}], tools
        )
    assert not _has_error(content), f"{choice.key}: degenerate-output error code on tool turn"
    # A coding agent should call a tool here; at minimum it must not be empty.
    assert tcs or content, f"{choice.key}: empty tool turn"


@pytest.mark.parametrize(
    "choice",
    [c for c in _present_choices() if (catalog.by_filename(c.filename).architecture or "").lower() == "cohere2_moe"] or [None],
    ids=lambda c: _ids(c) if c else "no-reasoning-models",
)
def test_real_thinking_off_suppresses_reasoning(choice):
    if choice is None:
        pytest.skip("no reasoning model downloaded")
    gw = _gateway_for(choice)
    with _Server(gw, gw.config.model):
        _, _, think_on = _collect(gw, [SYS, {"role": "user", "content": "What is 2+2?"}], None, think=True)
        _, _, think_off = _collect(gw, [SYS, {"role": "user", "content": "What is 2+2?"}], None, think=False)
    assert think_off == 0, f"{choice.key}: /thinking off still displayed {think_off} reasoning chars"


@pytest.mark.parametrize(
    "choice",
    [c for c in _present_choices() if (catalog.by_filename(c.filename).architecture or "").lower() == "diffusion_gemma"] or [None],
    ids=lambda c: _ids(c) if c else "no-diffusion-models",
)
def test_real_diffusion_reliable_across_runs(choice):
    # Diffusion is the noisiest backend; run several times to ensure the
    # canvas/num_predict handling is robust (this is where -n -1 bit us).
    if choice is None:
        pytest.skip("no diffusion model downloaded")
    gw = _gateway_for(choice)
    tools = _toolkit_schemas()
    fails = 0
    for _ in range(3):
        content, tcs, _ = _collect(gw, [SYS, {"role": "user", "content": "hi"}], tools)
        if _has_error(content) or not (content or tcs):
            fails += 1
    assert fails == 0, f"{choice.key}: {fails}/3 diffusion chat runs failed"
