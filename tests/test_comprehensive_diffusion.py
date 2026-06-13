"""DiffusionGemma runner coverage.

DiffusionGemma is a block-diffusion LM: llama-server cannot generate from
it, so the runtime dispatches (on the catalog's ``architecture`` field) to
the one-shot ``llama-diffusion-cli`` runner instead of HTTP. The real
runner is a one-time cmake build (bootstrap.ensure_diffusion_cli); tests
here use a stub executable so they run in milliseconds:

  * prompt formatting — Gemma chat template applied by hand (system fold,
    user/model roles, trailing model turn);
  * stream dispatch — a diffusion-arch model never touches HTTP, yields
    content events from the subprocess, strips <end_of_turn> and a
    prompt echo, raises with stderr context on a non-zero exit;
  * catalog — the picker group exists, and browsed quant filenames mint
    choices that carry architecture="diffusion_gemma" (the dispatch key).
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest


def _tool_args(call):
    """Tool-call arguments are a JSON STRING (OpenAI/Ollama convention, what
    the agent loop json.loads()es). Decode for assertions."""
    return json.loads(call["function"]["arguments"])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import bootstrap
from localcode import models_catalog as catalog
from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway, RuntimeErrorWithContext

DIFF_FILENAME = "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"


def _stub_cli(tmp_path: Path, script_body: str) -> Path:
    p = tmp_path / "llama-diffusion-cli"
    p.write_text("#!/bin/sh\n" + script_body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _gateway(tmp_path: Path, script_body: str) -> LocalCodeRuntimeGateway:
    cfg = RuntimeConfig()
    cfg.provider = "llama_cpp"
    cfg.model = str(tmp_path / DIFF_FILENAME)
    cfg.diffusion_cli_binary = str(_stub_cli(tmp_path, script_body))
    return LocalCodeRuntimeGateway(cfg)


# ── Prompt formatting ────────────────────────────────────────────────


def test_prompt_format_basic():
    msgs = [
        {"role": "system", "content": "Be terse.\nWorking directory: /tmp/proj"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]
    p = LocalCodeRuntimeGateway._format_diffusion_prompt(msgs)
    # UNIFIED: the SAME system prompt every model gets is folded in verbatim
    # (no bespoke concise substitute) — the full system text is preserved.
    assert "Be terse." in p
    assert "Working directory: /tmp/proj" in p
    # User / assistant turns are preserved with Gemma roles.
    assert "<start_of_turn>model\nhello<end_of_turn>" in p
    assert "<start_of_turn>user\nagain<end_of_turn>" in p
    # Generation slot comes last.
    assert p.endswith("<start_of_turn>model\n")


def test_prompt_format_includes_full_system_verbatim():
    # The full system prompt is folded in as-is (the old "discard verbose
    # system" workaround is gone — the empty-output bug was num_predict=-1,
    # not prompt length, so diffusion now shares the unified prompt).
    big = "UNIQUE_SYSTEM_MARKER " * 300
    p = LocalCodeRuntimeGateway._format_diffusion_prompt(
        [{"role": "system", "content": big}, {"role": "user", "content": "hi"}]
    )
    assert "UNIQUE_SYSTEM_MARKER" in p  # full system IS included now
    assert "<start_of_turn>user\n" in p and p.endswith("<start_of_turn>model\n")


# ── Stream dispatch through the stub runner ──────────────────────────


def test_diffusion_stream_yields_content(tmp_path):
    gw = _gateway(tmp_path, "printf 'Hello from diffusion!<end_of_turn>'\n")
    events = list(gw.stream_chat_events([{"role": "user", "content": "hi"}]))
    text = "".join(e["content"] for e in events if e["type"] == "content")
    assert text == "Hello from diffusion!"
    # Never went anywhere near HTTP / tool machinery.
    assert all(e["type"] in ("content", "stage", "stream_done") for e in events)
    # Terminal event is emitted so the UI can record token counts.
    assert any(e["type"] == "stream_done" for e in events)


def test_diffusion_strips_prompt_echo(tmp_path):
    # argv: -m <model> -p <prompt> ... → "$4" is the prompt. Echo it (like
    # a chatty runner) and then emit the generation.
    gw = _gateway(tmp_path, 'printf "%s" "$4"; printf "GEN"\n')
    events = list(gw.stream_chat_events([{"role": "user", "content": "hi"}]))
    text = "".join(e["content"] for e in events if e["type"] == "content")
    assert text == "GEN"


def test_diffusion_nonzero_exit_raises_with_stderr(tmp_path):
    gw = _gateway(tmp_path, "echo 'metal OOM' >&2; exit 3\n")
    with pytest.raises(RuntimeErrorWithContext) as ei:
        list(gw.stream_chat_events([{"role": "user", "content": "hi"}]))
    assert "metal OOM" in str(ei.value)


def test_chat_once_uses_diffusion_backend(tmp_path):
    gw = _gateway(tmp_path, "printf 'oneshot'\n")
    r = gw.chat_once([{"role": "user", "content": "hi"}])
    assert r["message"]["content"] == "oneshot"
    assert r["message"]["tool_calls"] == []


def test_diffusion_emits_plain_json_tool_call(tmp_path):
    # DiffusionGemma emits plain-JSON tool calls (wrapped in its channel/thought
    # reasoning); the backend must parse them and strip the scaffolding.
    # Quoted heredoc → literal output, identical under bash and dash (CI
    # runs /bin/sh = dash, where escaped printf quotes behaved differently).
    body = (
        "cat <<'STUBEOF'\n"
        '<|channel>thought\n'
        'reason<channel|>{"tool":"list_files","args":{"path":"."}}<tool_call|>\n'
        "STUBEOF\n"
    )
    gw = _gateway(tmp_path, body)
    tools = [{"function": {"name": "list_files",
                           "parameters": {"properties": {"path": {"type": "string"}}}}}]
    events = list(gw.stream_chat_events([{"role": "user", "content": "ls"}], tools=tools))
    tc = [e for e in events if e["type"] == "tool_calls"]
    assert tc, "expected a tool_calls event"
    fn = tc[0]["tool_calls"][0]["function"]
    assert fn["name"] == "list_files"
    assert _tool_args(tc[0]["tool_calls"][0]) == {"path": "."}
    # The raw JSON / channel scaffolding must NOT leak into visible content.
    content = "".join(e["content"] for e in events if e["type"] == "content")
    assert "<|channel>" not in content and '"tool"' not in content


def test_diffusion_repairs_malformed_tool_json():
    # DiffusionGemma is non-deterministic and often emits almost-valid JSON:
    # a bare `.` value, an unquoted path, or a trailing comma. The parser must
    # repair these and still surface the tool call (the real "ls" failure was
    # {"path":.} — invalid JSON — silently dropping the call). Valid JSON with
    # strings/numbers/booleans must pass through untouched.
    G = LocalCodeRuntimeGateway
    cases = {
        '{"tool":"list_files","args":{"path":.}}': {"path": "."},
        '{"tool":"read_file","args":{"path":src/main.py}}': {"path": "src/main.py"},
        '{"tool":"list_files","args":{"path":".",}}': {"path": "."},
        '{"tool":"bash","args":{"cmd":"ls -la","n":5}}': {"cmd": "ls -la", "n": 5},
        '{"tool":"x","args":{"enabled":true,"count":-3}}': {"enabled": True, "count": -3},
    }
    for raw, expected_args in cases.items():
        calls, _ = G._parse_diffusion_tool_calls(raw)
        assert calls, f"dropped tool call for {raw!r}"
        assert _tool_args(calls[0]) == expected_args, raw


def test_diffusion_json_repair_is_string_aware():
    # The repair must NOT corrupt string values that legitimately contain a
    # `:` or `,}` (e.g. shell commands) — a blind regex did, dropping the
    # call. It must also drop trailing commas and quote bare values, but only
    # OUTSIDE of strings.
    G = LocalCodeRuntimeGateway
    calls, _ = G._parse_diffusion_tool_calls(
        'Sure. {"tool":"run","args":{"cmd":"grep -n foo: bar","path":.}}'
    )
    assert calls and _tool_args(calls[0]) == {
        "cmd": "grep -n foo: bar", "path": "."
    }
    calls, _ = G._parse_diffusion_tool_calls(
        '{"tool":"x","args":{"k":"trailing comma here ,}","path":.}}'
    )
    assert calls and _tool_args(calls[0])["k"] == "trailing comma here ,}"


def test_diffusion_finds_all_brace_forms_in_order():
    # An earlier spaced-form `{ "tool"` call must not be skipped in favor of a
    # later compact `{"tool"` one.
    G = LocalCodeRuntimeGateway
    calls, _ = G._parse_diffusion_tool_calls(
        '{ "tool":"a","args":{}} and {"tool":"b","args":{}}'
    )
    assert [c["function"]["name"] for c in calls] == ["a", "b"]


def test_diffusion_tool_block_handles_none_parameters():
    # An MCP tool with inputSchema: null surfaces as parameters=None; must not
    # crash prompt formatting.
    G = LocalCodeRuntimeGateway
    block = G._diffusion_tool_block([{"function": {"name": "f", "parameters": None}}])
    assert "f(" in block


def test_diffusion_non_dict_args_coerced():
    # A model emitting "args":"foo" (or a list) must not produce a tool call
    # whose arguments crash the agent loop — coerce to {}.
    G = LocalCodeRuntimeGateway
    calls, _ = G._parse_diffusion_tool_calls('{"tool":"x","args":"oops"}')
    assert calls and _tool_args(calls[0]) == {}


def test_diffusion_canvas_clamps_nonpositive_num_predict(tmp_path):
    # The agent loop passes num_predict = MAX_OUTPUT_TOKENS = -1. The CLI's -n
    # is a CANVAS/token budget; `-n -1` yields an empty canvas ("no usable
    # response"). Non-positive num_predict MUST become the default 1024 budget
    # (room for a reasoning preamble AND a complete tool call); positive values
    # are capped at 1024. (Regression for the "returned no usable response" /
    # truncated-tool-call bugs.) The stub writes its argv to a file (the prompt
    # contains <end_of_turn>, which the output cleaner would otherwise truncate).
    argsfile = tmp_path / "argv.txt"
    gw = _gateway(tmp_path, f'printf "%s\\n" "$@" > "{argsfile}"\nprintf "ok"\n')

    def canvas_arg() -> str:
        toks = argsfile.read_text().splitlines()
        return toks[toks.index("-n") + 1]

    for np_in, want in [(-1, "1024"), (0, "1024"), (None, "1024"), (128, "128"), (9999, "1024")]:
        list(gw.stream_chat_events([{"role": "user", "content": "hi"}], num_predict=np_in))
        assert canvas_arg() == want, f"num_predict={np_in} → expected -n {want}"


def test_diffusion_bf16_strips_unmarked_thought_reasoning():
    # BF16 emits `thought\n<reasoning>.<answer>` with no channel markers and
    # no space at the reasoning→answer join (reasoning's own sentences use
    # ". "). The cleaner must keep only the answer. Real captured output.
    G = LocalCodeRuntimeGateway
    raw = (
        '\nthought\nThe user said "hi". I am LocalCode, a coding agent. '
        "I should greet briefly and wait for a task.Hello! How can I help you today?\n"
        "total time: 2156ms\nthroughput: 118 tok/s\n"
    )
    assert G._clean_diffusion_output(raw, "") == "Hello! How can I help you today?"


def test_diffusion_repairs_stray_quote_in_bare_value():
    # BF16 emitted {"path":."} (dropped the leading quote); the bare value `."`
    # must repair to "." not `."`.
    G = LocalCodeRuntimeGateway
    calls, _ = G._parse_diffusion_tool_calls('{"tool":"list_files","args":{"path":."}}')
    assert calls and _tool_args(calls[0]) == {"path": "."}


def test_diffusion_clean_never_blanks_out():
    # The cleaner must never empty a turn that contained real text, whatever
    # shape DiffusionGemma's non-deterministic output takes (this was the
    # BF16 "returned no usable response" bug).
    G = LocalCodeRuntimeGateway
    variants = [
        "<|channel>thought\nreason<channel|>Hello!",          # answer after channel
        "<|channel>thought\nuser said hi, I should greet",     # reasoning only, no answer
        "<end_of_turn>Hi there!",                              # early end_of_turn
        "Hello!<end_of_turn>Hello!Hello!",                     # answer then canvas padding
    ]
    for raw in variants:
        out = G._clean_diffusion_output(raw, "")
        assert out.strip(), f"cleaner blanked out: {raw!r}"
        assert "<|channel>" not in out and "<end_of_turn>" not in out


def test_non_diffusion_model_does_not_dispatch(tmp_path):
    cfg = RuntimeConfig()
    cfg.provider = "llama_cpp"
    cfg.model = str(tmp_path / "gemma-4-12b-it-UD-Q4_K_XL.gguf")
    gw = LocalCodeRuntimeGateway(cfg)
    assert gw._diffusion_choice() is None


# ── Catalog / picker wiring ──────────────────────────────────────────


def test_diffusion_group_in_picker():
    g = catalog.by_group("diffusiongemma-26b-a4b")
    assert g is not None
    assert g.architecture == "diffusion_gemma"
    assert g.hf_repo == "unsloth/diffusiongemma-26B-A4B-it-GGUF"


def test_browsed_quant_mints_diffusion_choice():
    # A quant the curated CHOICES list doesn't know, picked via the
    # HF-style picker — must still resolve with the diffusion arch so the
    # runtime dispatch works.
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


# ── Bootstrap runner discovery ───────────────────────────────────────


def test_diffusion_cli_path_prefers_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no cached binary
    fake = tmp_path / "bin" / "llama-diffusion-cli"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n")

    class _RT:
        diffusion_cli_binary = str(fake)

    class _Cfg:
        runtime = _RT()

    assert bootstrap.diffusion_cli_path(_Cfg()) == fake


def test_ensure_diffusion_cli_short_circuits_on_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cached = tmp_path / ".local" / "share" / "localcode" / "llama-diffusion-cli"
    cached.parent.mkdir(parents=True)
    cached.write_text("#!/bin/sh\n")
    ok, path = bootstrap.ensure_diffusion_cli()
    assert ok is True and Path(path) == cached
