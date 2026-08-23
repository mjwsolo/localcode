"""DiffusionGemma on the bundled llama-server.

DiffusionGemma is a block-diffusion LM: it denoises a whole 256-token canvas
per step instead of sampling one token per decode. The fork hosts that
denoiser INSIDE llama-server (llama-cpp-turboquant/PATCHES.md, patch 0005:
``diffusion_process_slot`` in tools/server/server-context.cpp), so localcode
serves it exactly like every other model: one bundled binary, one process that
keeps the 16 GB of weights resident across turns, ``/v1/chat/completions`` for
text, streaming and tool calls. There is no diffusion CLI, no second binary,
no per-turn model reload, and no architecture dispatch in the gateway.

These tests pin that contract from the Python side with a scripted SSE stream
(the real server is exercised by dev/verify_models.sh and
tests/test_real_models.py):

  * the diffusion-arch model goes through the SAME HTTP streaming path as any
    model, and its request body is the plain OpenAI chat-completions shape
    (OpenAI-format ``tools``, no hand-built prompt, no ``enable_thinking``
    kwarg -- the model has no thinking toggle and the server applies the
    GGUF's chat template);
  * block-sized chunks, reasoning_content and server-parsed tool_calls come
    out as the usual content / thinking / tool_calls events;
  * ``_restart_server`` launches the bundled llama-server for it (it used to
    short-circuit because "diffusion has no server");
  * the old side-channel (runtime_diffusion, diffusion_cli_path, the extra
    binaries, the setup-screen skip) is gone and stays gone;
  * catalog/picker wiring still mints the diffusion_gemma architecture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import models_catalog as catalog
from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "localcode"
DIFF_FILENAME = "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"


# ── Scripted llama-server SSE stream ─────────────────────────────────


class _FakeStreamResponse:
    """Stand-in for httpx's streaming response: yields scripted SSE lines."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def iter_lines(self):
        yield from self._lines


def _chunk(delta: dict, finish: str | None = None, **extra) -> str:
    body = {"choices": [{"delta": delta, "finish_reason": finish, "index": 0}]}
    body.update(extra)
    return "data: " + json.dumps(body)


def _done(finish: str = "stop", completion_tokens: int = 300) -> list[str]:
    return [
        _chunk({}, finish, usage={"prompt_tokens": 40, "completion_tokens": completion_tokens,
                                  "total_tokens": 40 + completion_tokens},
               timings={"predicted_n": completion_tokens, "predicted_per_second": 55.0}),
        "data: [DONE]",
    ]


def _block_stream() -> list[str]:
    """What the in-server diffusion path emits: ONE chunk per committed
    256-token block (the denoiser commits block by block), with the model's
    `<|channel>thought ... <channel|>` preamble already split out by the
    server's chat parser into reasoning_content."""
    return [
        _chunk({"reasoning_content": "The user wants a hash map definition. One sentence.\n"}),
        _chunk({"reasoning_content": "Keep it precise.", "content": "A hash map stores key-value pairs "}),
        _chunk({"content": "and uses a hash function to find them in constant time."}),
        *_done("stop"),
    ]


def _tool_stream() -> list[str]:
    """A server-parsed tool call (the GGUF's Gemma tool format, parsed by
    llama-server; nothing is parsed on the Python side)."""
    return [
        _chunk({"reasoning_content": "Need the weather tool."}),
        _chunk({"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]}),
        *_done("tool_calls", 60),
    ]


def _gateway(tmp_path: Path) -> LocalCodeRuntimeGateway:
    cfg = RuntimeConfig()
    cfg.provider = "llama_cpp"
    cfg.base_url = "http://localhost:8081"
    cfg.model = str(tmp_path / DIFF_FILENAME)
    turbo = tmp_path / "llama-server"
    turbo.write_text("#!/bin/sh\n")
    cfg.llama_cpp_binary = str(turbo)
    return LocalCodeRuntimeGateway(cfg)


def _drive(gw: LocalCodeRuntimeGateway, lines: list[str], **kw):
    """Run one stream_chat_events turn against scripted SSE; return
    (events, the JSON body that was POSTed)."""
    fake_client = MagicMock()
    fake_client.is_closed = False
    fake_client.stream = MagicMock(return_value=_FakeStreamResponse(lines))
    gw._client = fake_client
    with patch.object(gw, "_quick_server_probe", return_value=True):
        events = list(gw.stream_chat_events(
            [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}], **kw
        ))
    assert fake_client.stream.call_count == 1
    method, url = fake_client.stream.call_args.args[:2]
    assert method == "POST"
    assert url.endswith("/v1/chat/completions")
    return events, fake_client.stream.call_args.kwargs["json"]


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]


# ── One HTTP path for everything ─────────────────────────────────────


def test_diffusion_model_uses_the_plain_http_chat_path(tmp_path):
    gw = _gateway(tmp_path)
    events, body = _drive(gw, _block_stream())
    content = "".join(e["content"] for e in events if e["type"] == "content")
    assert content == ("A hash map stores key-value pairs and uses a hash function to find "
                       "them in constant time.")
    # The request is the ordinary OpenAI-compatible body: messages as given,
    # no hand-applied Gemma template, no raw `prompt` field.
    assert body["messages"][-1] == {"role": "user", "content": "hi"}
    assert "prompt" not in body
    assert body["stream"] is True
    assert events[-1]["type"] == "stream_done"


def test_diffusion_request_has_no_thinking_toggle(tmp_path):
    """DiffusionGemma reasons visibly in every turn and has no thinking
    switch; the Gemma-4 `enable_thinking` kwarg must NOT be sent (with
    enable_thinking=false the model emits end-of-generation at position 0,
    i.e. an empty reply -- verified on the real weights)."""
    gw = _gateway(tmp_path)
    for think in (True, False):
        _, body = _drive(gw, _block_stream(), think=think)
        assert "chat_template_kwargs" not in body
        assert "reasoning_effort" not in body


def test_diffusion_reasoning_split_comes_from_the_server(tmp_path):
    """The server keeps the `<|channel>thought ... <channel|>` markers in the
    generated text and its chat parser splits them into reasoning_content;
    localcode shows that as thinking when the policy is on and hides it when
    off -- the reasoning never leaks into the visible content either way."""
    gw = _gateway(tmp_path)
    events, _ = _drive(gw, _block_stream(), think=True)
    thinking = "".join(e["content"] for e in events if e["type"] == "thinking")
    content = "".join(e["content"] for e in events if e["type"] == "content")
    assert "hash map definition" in thinking
    assert "hash map definition" not in content

    events, _ = _drive(gw, _block_stream(), think=False)
    assert not [e for e in events if e["type"] == "thinking"]
    content = "".join(e["content"] for e in events if e["type"] == "content")
    assert "hash map definition" not in content
    assert content.startswith("A hash map")


def test_diffusion_tools_go_through_the_tools_key(tmp_path):
    gw = _gateway(tmp_path)
    events, body = _drive(gw, _tool_stream(), tools=TOOLS)
    # OpenAI-format tools in the body, applied by the server's chat template --
    # no plain-JSON tool block is appended to the prompt any more.
    assert body["tools"] == TOOLS
    assert "list_files(" not in json.dumps(body["messages"])
    tool_events = [e for e in events if e["type"] == "tool_calls"]
    assert tool_events, "server-parsed tool_calls must surface as a tool_calls event"
    call = tool_events[0]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


def test_block_chunks_stream_progressively(tmp_path):
    """Each committed block is one chunk; the consumer sees them as they
    land, not as a single dump at the end."""
    gw = _gateway(tmp_path)
    events, _ = _drive(gw, _block_stream())
    content_events = [e for e in events if e["type"] == "content"]
    assert len(content_events) >= 2
    assert content_events[0]["content"].startswith("A hash map")


def test_usage_comes_from_server_not_estimated(tmp_path):
    gw = _gateway(tmp_path)
    events, _ = _drive(gw, _block_stream())
    done = events[-1]
    assert done["type"] == "stream_done"
    assert done.get("completion_tokens") == 300
    assert not done.get("usage_estimated", False)


# ── Server lifecycle ─────────────────────────────────────────────────


def test_restart_server_launches_llama_server_for_diffusion(tmp_path, monkeypatch):
    """`_restart_server` used to return True without starting anything
    ("diffusion has no server"). Now it starts the bundled llama-server
    like for any model, and the command is the normal one."""
    gw = _gateway(tmp_path)
    model = tmp_path / DIFF_FILENAME
    model.write_bytes(b"GGUF")
    monkeypatch.setattr("localcode.bootstrap.get_model_path", lambda preferred=None: model)

    mgr = MagicMock()
    mgr.is_running.return_value = False
    mgr.restart.return_value = True
    mgr.port = 8081
    monkeypatch.setattr("localcode.server_manager.ServerManager.get", classmethod(lambda cls: mgr))

    assert gw._restart_server() is True
    mgr.restart.assert_called_once()
    cmd, model_arg = mgr.restart.call_args.args[:2]
    assert cmd[0] == gw.config.llama_cpp_binary and cmd[0].endswith("llama-server")
    assert model_arg == str(model)
    assert "--jinja" in cmd  # the GGUF's own chat template drives tools + reasoning


def test_server_command_for_diffusion_is_the_normal_one(tmp_path):
    gw = _gateway(tmp_path)
    cmd = gw.llama_server_command(gw.config.model)
    assert cmd[0].endswith("llama-server")
    assert "--model" in cmd and "--ctx-size" in cmd and "--jinja" in cmd
    # no diffusion-specific flag or binary (the model path itself is the only place the word appears)
    assert not any("diffusion" in c for c in cmd if c != gw.config.model)


# ── The side-channel is gone ─────────────────────────────────────────


def test_no_diffusion_side_channel_in_gateway(tmp_path):
    gw = _gateway(tmp_path)
    for name in ("_diffusion_choice", "_stream_diffusion_events", "_format_diffusion_prompt",
                 "_run_diffusion_cli", "_parse_diffusion_tool_calls", "_clean_diffusion_output",
                 "_diffusion_cli_binary"):
        assert not hasattr(gw, name), f"{name} must not come back"
    with pytest.raises(ImportError):
        import localcode.runtime_diffusion  # noqa: F401
    assert not hasattr(gw.config, "diffusion_cli_binary")


def test_one_shipped_binary():
    names = sorted(p.name for p in (SRC / "bin").iterdir() if p.is_file() and p.name != "__init__.py" and not p.name.startswith("."))
    assert names == ["llama-server"], f"exactly one shipped binary expected, got {names}"
    for text in ((ROOT / "MANIFEST.in").read_text(), (ROOT / "pyproject.toml").read_text()):
        assert "llama-diffusion" not in text


def test_tui_has_no_diffusion_special_cases():
    """Setup no longer skips the server launch for diffusion, and the status
    bar no longer shows a separate "diffusion runner" state: the server is
    probed for liveness like for every model."""
    setup_src = (SRC / "tui" / "screens" / "setup.py").read_text()
    assert "diffusion" not in setup_src.lower()
    chat_src = (SRC / "tui" / "screens" / "chat.py").read_text()
    assert "diffusion runner" not in chat_src


# ── Catalog / picker wiring ──────────────────────────────────────────


def test_diffusion_group_in_picker():
    g = catalog.by_group("diffusiongemma-26b-a4b")
    assert g is not None
    assert g.architecture == "diffusion_gemma"
    assert g.hf_repo == "unsloth/diffusiongemma-26B-A4B-it-GGUF"


def test_browsed_quant_mints_diffusion_choice():
    # A quant the curated CHOICES list doesn't know, picked via the
    # HF-style picker -- must still resolve with the diffusion arch.
    c = catalog.by_filename("diffusiongemma-26B-A4B-it-UD-Q4_K_XL.gguf")
    assert c is not None
    assert c.architecture == "diffusion_gemma"


def test_group_for_filename_no_false_positives():
    assert catalog.group_for_filename("totally-unrelated-model-Q4.gguf") is None
    g = catalog.group_for_filename("gemma-4-12b-it-UD-Q6_K.gguf")
    assert g is not None and g.key == "gemma-4-12b"


def test_curated_choices_still_win():
    c = catalog.by_filename(DIFF_FILENAME)
    assert c is not None
    assert c.key == "diffusiongemma"  # the curated entry, not a minted one


def test_catalog_prose_describes_the_server_path():
    c = catalog.by_key("diffusiongemma")
    g = catalog.by_group("diffusiongemma-26b-a4b")
    for notes in (c.notes, g.notes):
        assert "llama-diffusion-cli" not in notes
