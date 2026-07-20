"""Scripted fake runtime — drive the whole agent loop with NO model server.

Why this exists
---------------
The real `LocalCodeRuntimeGateway` streams from a llama-server (a ~7GB
model, slow, and non-deterministic). That's unusable for "test every
feature on every commit." But the agent loop only ever talks to the
gateway through ONE method:

    engine.stream_chat_events(messages, tools=..., ...) -> Iterator[dict]

…yielding `{"type": "content"|"thinking"|"tool_calls"|"stage", ...}`
events. Tool calls are just data the model "emits"; the loop parses
them, runs the real tools, feeds results back, and calls
`stream_chat_events` again for the next round.

So a fake gateway that REPLAYS a scripted list of those events lets us
exercise the entire loop — prompts, every tool, multi-round tool use,
thinking, vision messages, errors — deterministically and instantly.

Usage
-----
    app = build_test_app(tmp_path)
    app.engine.script = [
        tool_round(("read_file", {"path": "main.py"})),  # round 1: call a tool
        say("Here is the file."),                         # round 2: final answer
    ]
    rec = EventRecorder()
    trace = run_one_turn(app, rec, "show me main.py")
    assert "read_file" in trace.tool_calls_made()

When the script is exhausted the fake yields a terminal text response so
the loop always terminates instead of hanging.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from localcode.runtime import LocalCodeRuntimeGateway


# ── Script builders ─────────────────────────────────────────────────
# Each "response" is a list of event dicts — exactly what one call to
# stream_chat_events would yield for one model turn.


def say(text: str, *, thinking: str | None = None) -> list[dict]:
    """A plain text answer (optionally preceded by a thinking block)."""
    events: list[dict] = []
    if thinking:
        events.append({"type": "thinking", "content": thinking})
    events.append({"type": "content", "content": text})
    return events


def tool_round(*calls: tuple[str, dict], text: str = "") -> list[dict]:
    """A model turn that calls one or more tools (Ollama tool-call shape).

    Each `call` is `(tool_name, args_dict)`. Optional `text` is any
    assistant prose emitted alongside the calls.
    """
    events: list[dict] = []
    if text:
        events.append({"type": "content", "content": text})
    events.append({
        "type": "tool_calls",
        # The loop json.loads() the arguments, so they must be a JSON
        # STRING here — matching how OpenAI/Ollama stream tool-call args.
        "tool_calls": [
            {"function": {"name": name, "arguments": json.dumps(dict(args))}}
            for name, args in calls
        ],
    })
    return events


def raise_error(exc: Exception) -> Exception:
    """Sentinel: when this response is reached, the fake raises `exc`
    from inside stream_chat_events — used to test the loop's error path.
    """
    return exc


# ── The fake gateway ────────────────────────────────────────────────


class FakeRuntime(LocalCodeRuntimeGateway):
    """A gateway whose `stream_chat_events` replays a script.

    Inherits everything else from the real gateway (config access,
    `_target_num_ctx`, etc.) so the loop sees a normal engine. No socket
    is ever opened — `_client` is a MagicMock and the streaming method is
    fully overridden.
    """

    def __init__(self, config, script: list | None = None) -> None:
        super().__init__(config)
        self._client = MagicMock()
        self.script: list = list(script or [])
        # Every messages payload the loop sent us, in order. Tests assert
        # on this to prove what actually reached "the model" (e.g. that an
        # image part or a tool result was included).
        self.calls: list[list[dict[str, Any]]] = []
        # The `think` flag passed on each call, in order. Tests assert on this
        # to prove decode-mode recovery (e.g. a loop abort retries with think
        # off).
        self.think_calls: list[bool] = []
        self._cursor = 0

    # The single seam the agent loop uses.
    def stream_chat_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        think: bool = False,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        recovery_mode: str = "",
        stream_policy: str = "",
    ) -> Iterator[dict[str, Any]]:
        # Defensive copy so later loop mutation can't rewrite history.
        self.calls.append([dict(m) for m in messages])
        self.think_calls.append(bool(think))

        if self._cursor < len(self.script):
            response = self.script[self._cursor]
            self._cursor += 1
        else:
            # Script exhausted — emit a terminal answer so the loop ends.
            response = say("Done.")

        if isinstance(response, Exception):
            raise response

        yield from response

    # Keep server-management no-ops harmless if anything reaches them.
    def _quick_server_probe(self) -> bool:  # pragma: no cover - defensive
        return True

    def _restart_server(self) -> bool:  # pragma: no cover - defensive
        return True


# ── Headless app builder ────────────────────────────────────────────


def _test_config(home: Path):
    """An AppConfig wired for tests: llama_cpp provider, fake server URL,
    everything pointed at a throwaway LOCALCODE_HOME."""
    from localcode.config import (
        AppConfig, LoggingConfig, RuntimeConfig,
        SafetyConfig, SearchConfig, UIConfig,
    )
    os.environ["LOCALCODE_HOME"] = str(home)
    return AppConfig(
        runtime=RuntimeConfig(
            provider="llama_cpp",
            base_url="http://localhost:9999",
            profile="e4b",
            model="test-model",
            mode="fast",
            temperature=0.1,
            # Large so trivial test conversations never trip auto-compaction
            # (compaction has its own dedicated test).
            max_context_chars=400000,
            llama_cpp_binary="/usr/local/bin/llama-server",
            kv_cache_type_k="q8_0",
            kv_cache_type_v="turbo4",
            laptop_26b_runtime_mode="speed",
        ),
        search=SearchConfig(),
        ui=UIConfig(),
        safety=SafetyConfig(),
        logging=LoggingConfig(),
    )


def build_test_app(tmp_path: Path, *, script: list | None = None, cwd: Path | None = None):
    """Construct a real `LocalCodeApp` with a FakeRuntime engine.

    `tmp_path` gives an isolated LOCALCODE_HOME (sessions/history/memory
    all land here, never the user's real home). `cwd` is the project
    root the agent operates on — defaults to a fresh tmp dir. The
    returned app's `.engine.script` can be reassigned per-turn.
    """
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    repo = cwd or (tmp_path / "project")
    repo.mkdir(parents=True, exist_ok=True)

    config = _test_config(home)

    # full_auto so the agent loop never blocks on an approval prompt.
    os.environ["LOCALCODE_AUTONOMY"] = "full_auto"

    from localcode.app import LocalCodeApp
    app = LocalCodeApp(config, cwd=repo)
    app.engine = FakeRuntime(config.runtime, script=script)
    return app
