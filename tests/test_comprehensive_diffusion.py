"""DiffusionGemma runner coverage.

DiffusionGemma is a block-diffusion LM: llama-server cannot generate from
it, so the runtime dispatches (on the catalog's ``architecture`` field) to
the one-shot ``llama-diffusion-cli`` runner instead of HTTP. The real
runner ships in the wheel (bootstrap.diffusion_cli_path); tests
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
import platform
import stat
import sys
from pathlib import Path

import pytest

# The DiffusionGemma runner is a macOS-only, cmake-built llama-diffusion-cli.
# These tests exercise that subprocess path; on the Linux CI runner (no native
# binary, 7 GB RAM) it hangs/OOMs and kills the runner (exit 143). The macOS CI
# leg runs them in full. Skipping here keeps the multi-version Linux matrix green
# without losing coverage.
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="DiffusionGemma runner/subprocess path is macOS-only",
)


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


def test_diffusion_clean_strips_unused_collapse_soup():
    # On a large prompt the entropy-bound-off retry can collapse into a stream
    # of <unused42> tokens. They're non-empty, so they were surfaced as "text"
    # (the `<unused26><unused27>…` soup in the chat). The cleaner must strip
    # them: mixed content keeps the real words; a collapse-ONLY turn cleans to
    # empty so _diffusion_turn_usable rejects it (-> honest E3107, not garbage).
    G = LocalCodeRuntimeGateway
    assert G._clean_diffusion_output("<unused26><unused27> hello <unused10>", "") == "hello"
    assert G._clean_diffusion_output("<unused26><unused27><unused30>", "") == ""
    assert G._diffusion_turn_usable(
        G._clean_diffusion_output("<unused1><unused2>", ""), [], None
    ) is False


def test_diffusion_adaptive_retry_forces_eb_off(monkeypatch, tmp_path):
    # The first attempt uses the fast entropy-bound `auto` decoder; a retry
    # after an unusable turn must force `--diffusion-eb off` (verified to
    # recover large-prompt turns the auto decoder denoises to empty). Re-running
    # the identical command could never recover a deterministic empty.
    G = LocalCodeRuntimeGateway
    cfg = RuntimeConfig()
    cfg.provider = "llama_cpp"
    cfg.model = str(tmp_path / DIFF_FILENAME)
    # Point at an EXISTING stub so the runtime uses it instead of trying to build
    # the real llama-diffusion-cli (a slow cmake build that times out on clean CI).
    cfg.diffusion_cli_binary = str(_stub_cli(tmp_path, "exit 0\n"))
    gw = G(cfg)
    seen_cmds: list[list] = []

    def fake_run(cmd, timeout):
        seen_cmds.append(cmd)
        # First call (no eb flag) returns empty; second (eb off) returns text.
        return "" if "--diffusion-eb" not in cmd else "recovered answer"

    monkeypatch.setattr(gw, "_run_diffusion_cli", fake_run)
    events = list(gw._stream_diffusion_events([{"role": "user", "content": "hi"}]))
    text = "".join(e["content"] for e in events if e["type"] == "content")
    assert text == "recovered answer"
    assert "--diffusion-eb" not in seen_cmds[0], "first attempt is the fast auto path"
    assert seen_cmds[1][-2:] == ["--diffusion-eb", "off"], "retry forces eb off"


def test_prompt_format_renders_tool_call_turn_and_labeled_result():
    # THE post-tool-result E3107 bug: an assistant turn that ONLY made tool
    # calls has empty content. The old formatter skipped it, leaving the
    # prompt as `user -> user -> model` (a tool result with no record the
    # model asked for one). The entropy-bound decoder then denoised to EMPTY
    # in ~2 steps -> E3107 on EVERY multi-step agentic task. The tool-call
    # turn must be rendered as a `model` turn, and the tool result as a
    # labeled `user` turn, so the conversation stays coherent.
    import re
    G = LocalCodeRuntimeGateway
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "search for X"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function",
             "function": {"name": "web_search",
                          "arguments": '{"query": "X"}'}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "result body"},
    ]
    p = G._format_diffusion_prompt(msgs, tools=None)
    # Coherent alternation — NOT user,user,model.
    assert re.findall(r"<start_of_turn>(\w+)", p) == ["user", "model", "user", "model"]
    # The empty-content tool-call turn is rendered as the model's JSON call.
    assert '{"tool": "web_search", "args": {"query": "X"}}' in p
    # The result is labeled with the tool it came from.
    assert "Tool result (web_search):" in p
    assert "result body" in p


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


def test_diffusion_repairs_invalid_string_escapes():
    # THE green-eggs-and-ham bug: DiffusionGemma emitted a perfect write_file
    # tool call, but the `content` string contained `\ ` (a stray backslash
    # before a space) — an INVALID JSON escape. json.loads rejected the whole
    # blob, the call was dropped, the span was still stripped from visible
    # text → empty turn → E3107 on EVERY prompt, even trivial ones. The repair
    # must sanitize invalid escapes INSIDE strings while preserving valid ones
    # (\n, \t, \", \uXXXX) so the tool call survives.
    G = LocalCodeRuntimeGateway
    raw = (
        '{"tool":"write_file","args":{"path":"book.txt",'
        r'"content":"line one.\nline two.\ stray backslash space.\nlast line."}}'
    )
    calls, text = G._parse_diffusion_tool_calls(raw)
    assert calls, "invalid \\ escape must be repaired, not drop the tool call"
    args = _tool_args(calls[0])
    assert args["path"] == "book.txt"
    # Valid \n escapes are preserved as real newlines; the stray backslash is
    # kept as a literal backslash (escaping, not dropping — so a Windows path
    # like C:\Users emitted as \U survives too).
    assert "line one.\nline two." in args["content"]
    assert "last line." in args["content"]
    # The arguments string must itself be valid JSON the agent loop can reparse.
    import json as _json
    _json.loads(calls[0]["function"]["arguments"])


def test_diffusion_repairs_invalid_escape_is_kept_not_dropped():
    # A stray backslash before a non-escape char must be ESCAPED (kept as a
    # literal backslash), not dropped — otherwise content/paths silently lose
    # characters. (\b \f \n \r \t \" \\ \/ \uXXXX stay valid escapes; we can't
    # disambiguate those from a literal-backslash intent, and don't try.)
    G = LocalCodeRuntimeGateway
    calls, _ = G._parse_diffusion_tool_calls(
        r'{"tool":"read_file","args":{"path":"C:\Xenon\queue.txt"}}'
    )
    assert calls, "invalid escapes in a path must not drop the call"
    assert _tool_args(calls[0])["path"] == r"C:\Xenon\queue.txt"


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
    # response"). Non-positive num_predict MUST become the default 2048 budget
    # (room for a reasoning preamble AND a complete tool call across blocks);
    # positive values are capped at 2048. (Regression for the "returned no
    # usable response" / truncated-tool-call bugs.) Also: NO --diffusion-blocks
    # flag — it caps output to one block regardless of -n. The stub writes its
    # argv to a file (the prompt has <end_of_turn>, which the cleaner truncates).
    argsfile = tmp_path / "argv.txt"
    gw = _gateway(tmp_path, f'printf "%s\\n" "$@" > "{argsfile}"\nprintf "ok"\n')

    def canvas_arg() -> str:
        toks = argsfile.read_text().splitlines()
        return toks[toks.index("-n") + 1]

    for np_in, want in [(-1, "2048"), (0, "2048"), (None, "2048"), (128, "128"), (9999, "2048")]:
        list(gw.stream_chat_events([{"role": "user", "content": "hi"}], num_predict=np_in))
        assert "--diffusion-blocks" not in argsfile.read_text(), "must not cap blocks"
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


def test_diffusion_preempts_oversized_prompt(tmp_path, monkeypatch):
    # A prompt large enough to reliably collapse the small canvas must
    # short-circuit to E3107 WITHOUT burning the 3 CLI retries (~75s).
    gw = _gateway(tmp_path, "printf 'should not run'\n")
    called = {"n": 0}

    def _counting_cli(*_a, **_k):
        called["n"] += 1
        return ""
    monkeypatch.setattr(gw, "_run_diffusion_cli", _counting_cli)

    big = "x" * 20000  # > the 16000-char pre-empt limit
    events = list(gw.stream_chat_events([{"role": "user", "content": big}]))
    content = "".join(e["content"] for e in events if e["type"] == "content")
    assert "E3107" in content
    assert called["n"] == 0, "oversized prompt must not invoke the CLI retries"


def test_diffusion_normal_prompt_still_runs_cli(tmp_path, monkeypatch):
    # A normal-size prompt must NOT be pre-empted — the CLI still runs.
    gw = _gateway(tmp_path, "printf 'hi there'\n")
    called = {"n": 0}

    def _counting_cli(*_a, **_k):
        called["n"] += 1
        return "hi there"
    monkeypatch.setattr(gw, "_run_diffusion_cli", _counting_cli)

    list(gw.stream_chat_events([{"role": "user", "content": "hello"}]))
    assert called["n"] >= 1, "normal prompt must still run the CLI"
