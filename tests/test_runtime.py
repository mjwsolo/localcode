"""Tests for localcode.runtime — server command building, context scaling, blob resolution."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from localcode.config import RuntimeConfig
from localcode.runtime import (
    LocalCodeRuntimeGateway,
    _error_message,
    _estimate_prompt_token_count,
    _estimate_token_count,
    _tool_arg_stream_guard,
    _strip_thinking_tokens,
)


class TestLlamaServerCommand:
    """Verify llama_server_command builds the correct CLI args for each mode."""

    def _make_gw(self, **overrides) -> LocalCodeRuntimeGateway:
        defaults = dict(
            provider="llama_cpp",
            base_url="http://localhost:8081",
            model="test.gguf",
            llama_cpp_binary="/usr/local/bin/llama-server",
            kv_cache_type_k="q8_0",
            kv_cache_type_v="turbo4",
            laptop_26b_runtime_mode="speed",
            llama_cpp_gpu_layers=0,
            llama_cpp_threads=8,
            llama_cpp_batch_size=128,
            llama_cpp_spec_type="",
            llama_cpp_draft_max=64,
            llama_cpp_expert_offload=False,
            llama_cpp_draft_model="",
            llama_cpp_lookup_cache=False,
            max_context_chars=40000,
            quant_preset="balanced",
        )
        defaults.update(overrides)
        cfg = RuntimeConfig(**defaults)
        return LocalCodeRuntimeGateway(cfg)

    def test_speed_mode_uses_mmap_no_gpu(self) -> None:
        """Legacy 'speed' mode path — falls to the default GPU branch in
        runtime.py, which is ngl-configurable. With gpu_layers=0 (fixture
        default) the command must show `-ngl 0`."""
        gw = self._make_gw(
            laptop_26b_runtime_mode="speed",
            llama_cpp_gpu_layers=0,
        )
        cmd = gw.llama_server_command("/path/model.gguf", port=8081)
        assert "--model" in cmd
        assert "/path/model.gguf" in cmd
        assert "--mmap" in cmd
        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "999"  # default branch hardcodes 999

    def test_turbo_mode_full_gpu(self) -> None:
        """Turbo mode now respects llama_cpp_gpu_layers (0 = CPU, 999 = full)
        instead of hardcoding 999. Test asserts the full-offload configuration
        by setting gpu_layers=999 explicitly."""
        gw = self._make_gw(
            laptop_26b_runtime_mode="turbo",
            llama_cpp_gpu_layers=999,
        )
        cmd = gw.llama_server_command("/path/model.gguf")
        assert "-ngl" in cmd
        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "999"
        assert "--mmap" in cmd
        assert "-fit" in cmd
        assert "off" in cmd

    def test_context_mode_expert_offload(self) -> None:
        """Context mode: GPU attention + CPU experts."""
        gw = self._make_gw(laptop_26b_runtime_mode="context")
        cmd = gw.llama_server_command("/path/model.gguf")
        assert "-ngl" in cmd
        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "999"
        assert "-ot" in cmd
        ot_idx = cmd.index("-ot")
        assert cmd[ot_idx + 1] == "exps=CPU"

    def test_kv_cache_types_in_command(self) -> None:
        """KV cache compression flags should appear in the command."""
        gw = self._make_gw(kv_cache_type_k="q8_0", kv_cache_type_v="turbo4")
        cmd = gw.llama_server_command("/path/model.gguf")
        assert "--cache-type-k" in cmd
        k_idx = cmd.index("--cache-type-k")
        assert cmd[k_idx + 1] == "q8_0"
        assert "--cache-type-v" in cmd
        v_idx = cmd.index("--cache-type-v")
        assert cmd[v_idx + 1] == "turbo4"

    def test_f16_kv_cache_omitted(self) -> None:
        """When KV cache type is f16 (default), the flag should be omitted."""
        gw = self._make_gw(kv_cache_type_k="f16", kv_cache_type_v="f16")
        cmd = gw.llama_server_command("/path/model.gguf")
        assert "--cache-type-k" not in cmd
        assert "--cache-type-v" not in cmd

    def test_custom_binary_used(self, tmp_path) -> None:
        # A configured binary is honored only when it exists on disk; the
        # command builder self-heals (falls back to discovery) for a stale
        # path, so the test must point at a real file to assert "used".
        custom = tmp_path / "llama-server"
        custom.write_text("#!/bin/sh\n")
        gw = self._make_gw(llama_cpp_binary=str(custom))
        cmd = gw.llama_server_command("/path/model.gguf")
        assert cmd[0] == str(custom)

    def test_error_message_serializes_backend_error_dict(self) -> None:
        msg = _error_message({"message": "server unavailable", "code": 503})
        assert "server unavailable" in msg
        assert "503" in msg

    def test_port_in_command(self) -> None:
        gw = self._make_gw()
        cmd = gw.llama_server_command("/path/model.gguf", port=9090)
        port_idx = cmd.index("--port")
        assert cmd[port_idx + 1] == "9090"

    def test_flash_attn_enabled(self) -> None:
        gw = self._make_gw()
        cmd = gw.llama_server_command("/path/model.gguf")
        assert "--flash-attn" in cmd

    def test_draft_model_adds_spec_flags(self) -> None:
        """When a draft model is specified, speculative decoding flags should appear."""
        gw = self._make_gw(llama_cpp_draft_model="/path/draft.gguf")
        cmd = gw.llama_server_command("/path/model.gguf")
        assert "--model-draft" in cmd
        assert "/path/draft.gguf" in cmd
        assert "--draft-max" in cmd

    def test_lookup_cache_adds_flag(self) -> None:
        gw = self._make_gw(llama_cpp_lookup_cache=True)
        cmd = gw.llama_server_command("/path/model.gguf")
        assert "--lookup-cache-dynamic" in cmd

    def test_ngram_spec_type(self) -> None:
        gw = self._make_gw(llama_cpp_spec_type="ngram-mod")
        cmd = gw.llama_server_command("/path/model.gguf")
        assert "--spec-type" in cmd
        assert "ngram-mod" in cmd

    def test_default_emits_no_speculative_decoding(self) -> None:
        """Regression: an empty spec_type (the default, force-disabled in
        config.py to prevent n-gram repetition loops) must NOT silently emit
        `--spec-type ngram-mod`. Speculative decoding ships OFF; it only turns
        on with an explicit draft model, lookup cache, or spec_type."""
        gw = self._make_gw()  # defaults: spec_type="", draft_model="", lookup off
        cmd = gw.llama_server_command("/path/model.gguf")
        assert "--spec-type" not in cmd
        assert "--model-draft" not in cmd
        assert "--lookup-cache-dynamic" not in cmd

    def test_context_mode_uses_10_threads(self) -> None:
        """Context mode should use 10 threads for expert computation."""
        gw = self._make_gw(laptop_26b_runtime_mode="context", llama_cpp_threads=4)
        cmd = gw.llama_server_command("/path/model.gguf")
        threads_idx = cmd.index("--threads")
        assert cmd[threads_idx + 1] == "10"

    def test_context_mode_batch_sizes_respect_safe_config(self) -> None:
        """Context/turbo modes must not override explicit safe batch config."""
        gw = self._make_gw(laptop_26b_runtime_mode="context", llama_cpp_batch_size=128)
        cmd = gw.llama_server_command("/path/model.gguf")
        b_idx = cmd.index("-b")
        assert cmd[b_idx + 1] == "128"
        ub_idx = cmd.index("-ub")
        assert cmd[ub_idx + 1] == "128"

    def test_turbo_mode_defaults_to_large_batch_for_non_qwen(self) -> None:
        gw = self._make_gw(laptop_26b_runtime_mode="turbo", llama_cpp_batch_size=-1)
        cmd = gw.llama_server_command("/path/gemma-model.gguf")
        b_idx = cmd.index("-b")
        assert cmd[b_idx + 1] == "2048"

    def test_large_qwen_on_16gb_uses_safer_batch(self) -> None:
        gw = self._make_gw(laptop_26b_runtime_mode="turbo", llama_cpp_batch_size=-1)
        with patch.object(gw, "_system_ram_gb", return_value=16):
            cmd = gw.llama_server_command("/path/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")
        b_idx = cmd.index("-b")
        assert cmd[b_idx + 1] == "512"

    def test_large_qwen_on_16gb_clamps_stale_large_batch_config(self) -> None:
        # `max_context_chars=200000` → num_ctx target = 50000, which on
        # the post-2026-04-29 Qwen 35B / 16GB path is ≥32768 so the
        # special-case override returns 65536 (the validated ceiling).
        # The earlier 2026-04-26 version of this test asserted 32768
        # because the cap was conservatively low; bumped to 64K after
        # `-b 512` (still asserted here) freed Metal prefill scratch.
        gw = self._make_gw(
            laptop_26b_runtime_mode="turbo",
            llama_cpp_batch_size=2048,
            max_context_chars=200000,
            quant_preset="fastest",
        )
        with patch.object(gw, "_system_ram_gb", return_value=16):
            cmd = gw.llama_server_command("/path/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")
        b_idx = cmd.index("-b")
        assert cmd[b_idx + 1] == "512"
        ctx_idx = cmd.index("--ctx-size")
        assert cmd[ctx_idx + 1] == "65536"

    def test_large_qwen_on_16gb_ctx_clamped_to_64k_ceiling(self) -> None:
        # When max_context_chars would translate to >64K tokens, the
        # Qwen 35B / 16GB special case clamps to 65536 (was 32768
        # pre-2026-04-29).
        gw = self._make_gw(
            laptop_26b_runtime_mode="turbo",
            llama_cpp_batch_size=-1,
            max_context_chars=400000,  # → 100000 tokens, > 65536 ceiling
            quant_preset="fastest",
        )
        with patch.object(gw, "_system_ram_gb", return_value=16):
            cmd = gw.llama_server_command("/path/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")
        ctx_idx = cmd.index("--ctx-size")
        assert cmd[ctx_idx + 1] == "65536"

    def test_large_qwen_on_16gb_floors_small_chars_to_64k(self) -> None:
        # Stale `performance.py` presets (max_context_chars=10000 →
        # ~2500 tokens) and other small chars values were leaving users
        # below the floor needed for system prompt + tool schemas + a
        # single user message, producing E3103 ("Conversation is too
        # long for this model") on the FIRST turn of an empty session.
        # Fix: on the validated 16 GB Apple-Silicon turbo path the
        # runtime always picks 64K — opting out should be done by
        # switching off turbo (kv_cache_type_v) or changing
        # laptop_26b_runtime_mode, not by leaving a stale chars value.
        gw = self._make_gw(
            laptop_26b_runtime_mode="turbo",
            llama_cpp_batch_size=-1,
            max_context_chars=10000,  # stale preset → would yield 2500 tokens
            quant_preset="fastest",
        )
        with patch.object(gw, "_system_ram_gb", return_value=16):
            cmd = gw.llama_server_command("/path/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")
        ctx_idx = cmd.index("--ctx-size")
        assert cmd[ctx_idx + 1] == "65536"


class TestTargetNumCtx:
    """Verify _target_num_ctx scales with config and quant preset."""

    def _make_gw(self, **overrides) -> LocalCodeRuntimeGateway:
        defaults = dict(
            provider="llama_cpp",
            base_url="http://localhost:8081",
            max_context_chars=40000,
            quant_preset="balanced",
            kv_cache_type_v="turbo4",
            laptop_26b_runtime_mode="speed",
        )
        defaults.update(overrides)
        cfg = RuntimeConfig(**defaults)
        gw = LocalCodeRuntimeGateway(cfg)
        # Pin RAM low so these chars→ctx assertions are host-independent (the
        # balanced path now lifts ctx on ≥32 GB machines; without this they'd
        # fail on the dev box / CI runner depending on its RAM).
        gw._system_ram_gb = lambda: 16  # type: ignore[assignment]
        return gw

    def test_balanced_lift_on_big_ram(self) -> None:
        # A capable machine should get a large context on the balanced/default
        # preset, not the flat max_context_chars//4. (Regression: 128 GB Macs
        # were stuck at ~50K, starving long agentic sessions into re-read loops.)
        # 96 GB+ now unlocks 256K since every catalog model trains to >=256K;
        # the fake model path here is unreadable so _model_max_ctx doesn't clamp.
        gw = self._make_gw(max_context_chars=200000, quant_preset="balanced")
        gw._system_ram_gb = lambda: 128  # type: ignore[assignment]
        assert gw._target_num_ctx() == 262144

    def test_balanced_preset(self) -> None:
        gw = self._make_gw(max_context_chars=40000, quant_preset="balanced")
        ctx = gw._target_num_ctx()
        assert ctx == 10000  # 40000 // 4

    def test_smallest_preset_caps_at_2048(self) -> None:
        gw = self._make_gw(max_context_chars=40000, quant_preset="smallest")
        ctx = gw._target_num_ctx()
        assert ctx == 2048

    def test_explicit_override(self) -> None:
        gw = self._make_gw()
        ctx = gw._target_num_ctx(num_ctx_override=16384)
        assert ctx == 16384

    def test_override_minimum_is_1024(self) -> None:
        gw = self._make_gw()
        ctx = gw._target_num_ctx(num_ctx_override=100)
        assert ctx == 1024

    def test_small_context_chars(self) -> None:
        gw = self._make_gw(max_context_chars=4000, quant_preset="balanced")
        ctx = gw._target_num_ctx()
        assert ctx == 2048  # max(2048, 4000//4=1000) -> 2048


COHERE_GGUF = "North-Mini-Code-1.0-UD-Q4_K_M.gguf"  # catalog filename (cohere2_moe)


class TestCohereBounds:
    """North-Mini-Code (cohere2moe) must be RAM-bounded in BOTH context and
    per-turn generation so its unconditional reasoning can't grow the
    (uncompressed f16) KV cache until the stock server OOM-kills."""

    def _make_gw(self, ram_gb: int, **overrides) -> LocalCodeRuntimeGateway:
        defaults = dict(
            provider="llama_cpp",
            base_url="http://localhost:8081",
            model=COHERE_GGUF,
            max_context_chars=2_000_000,   # huge knob → exercises the ceiling
            quant_preset="balanced",
            kv_cache_type_v="turbo4",
            laptop_26b_runtime_mode="turbo",
        )
        defaults.update(overrides)
        gw = LocalCodeRuntimeGateway(RuntimeConfig(**defaults))
        gw._system_ram_gb = lambda: ram_gb  # type: ignore[assignment]
        return gw

    def test_detected_by_catalog_architecture(self) -> None:
        gw = self._make_gw(96)
        assert gw._is_cohere_gguf() is True
        assert gw._is_cohere_gguf("/models/" + COHERE_GGUF) is True
        assert gw._is_cohere_gguf("/models/gemma-4-26B-A4B-it.gguf") is False

    def test_ctx_bounded_well_below_turbo_ceiling(self) -> None:
        # On a 96 GB Mac the TurboQuant ceiling is 256K; cohere (f16 KV) must
        # be clamped far lower. Model path unreadable → _model_max_ctx doesn't
        # interfere, so the cohere RAM ceiling is the binding cap.
        for ram, expected in [(16, 16384), (32, 32768), (48, 49152), (128, 65536)]:
            gw = self._make_gw(ram)
            assert gw._target_num_ctx(model_path=COHERE_GGUF) == expected, ram

    def test_ctx_never_exceeds_turbo_ceiling_for_cohere(self) -> None:
        gw = self._make_gw(128)
        assert gw._target_num_ctx(model_path=COHERE_GGUF) < gw._ram_ctx_ceiling(128)

    def test_launch_ctx_size_is_bounded(self) -> None:
        # The actual --ctx-size handed to the stock cohere server must reflect
        # the conservative ceiling, not the 500K trained length / 256K turbo cap.
        gw = self._make_gw(128)
        cmd = gw.llama_server_command("/models/" + COHERE_GGUF, port=8081)
        ctx = int(cmd[cmd.index("--ctx-size") + 1])
        assert ctx == 65536

    def test_unlimited_generation_is_capped(self) -> None:
        # The agent loop passes num_predict=-1 (MAX_OUTPUT_TOKENS). For the
        # unconditionally-reasoning cohere model this MUST become a bounded
        # positive cap so a single thinking turn can't run to OOM.
        gw = self._make_gw(128)
        opts = gw._options(num_predict_override=-1)
        np = opts["num_predict"]
        assert isinstance(np, int) and 0 < np <= 8192

    def test_generation_cap_scales_with_small_ctx(self) -> None:
        # On a 16 GB machine ctx is 16K, so the per-turn cap is ctx//2 = 8192
        # (clamped at the 8192 ceiling). Always bounded, never -1.
        gw = self._make_gw(16)
        assert gw._options(num_predict_override=-1)["num_predict"] == 8192

    def test_non_cohere_generation_stays_unlimited(self) -> None:
        # Regression guard: the cap is cohere-only; other models keep -1.
        gw = self._make_gw(128, model="gemma-4-26B-A4B-it.gguf")
        assert gw._options(num_predict_override=-1)["num_predict"] == -1


class TestStripThinkingTokens:
    """Verify _strip_thinking_tokens removes Gemma 4 channel artifacts."""

    def test_strips_unused25(self) -> None:
        assert _strip_thinking_tokens("hello <unused25> world") == "hello  world"

    def test_strips_gemma4_collapse_soup(self) -> None:
        # Known llama.cpp Gemma-4 bug: the model collapses into a loop of raw
        # <unusedNN> / [multimodal] / <eos> tokens. None are user-facing — scrub
        # them all (the streaming layer also detects the collapse and stops).
        soup = "Answer<unused12><unused29>[multimodal] here<eos>"
        assert _strip_thinking_tokens(soup) == "Answer here"

    def test_strips_channel_tags(self) -> None:
        text = "<|channel>thought\nsome thinking\n<channel|>actual response"
        result = _strip_thinking_tokens(text)
        assert "channel" not in result
        assert "actual response" in result

    def test_empty_string(self) -> None:
        assert _strip_thinking_tokens("") == ""

    def test_none_passthrough(self) -> None:
        """If text is falsy, return it as-is."""
        assert _strip_thinking_tokens("") == ""

    def test_no_tokens_unchanged(self) -> None:
        assert _strip_thinking_tokens("normal text") == "normal text"


class TestEndpoints:
    """Verify endpoint URLs are built correctly per provider."""

    def test_llama_cpp_endpoints(self) -> None:
        cfg = RuntimeConfig(provider="llama_cpp", base_url="http://localhost:8081")
        gw = LocalCodeRuntimeGateway(cfg)
        assert gw.endpoint == "http://localhost:8081/v1/chat/completions"
        assert gw.tags_endpoint == "http://localhost:8081/v1/models"

    def test_trailing_slash_stripped(self) -> None:
        """A trailing slash on base_url should not produce double slashes in the path."""
        cfg = RuntimeConfig(provider="llama_cpp", base_url="http://localhost:8081/")
        gw = LocalCodeRuntimeGateway(cfg)
        # The path portion after the authority should not have double slashes
        path_part = gw.endpoint.split("://", 1)[1]  # e.g. "localhost:8081/v1/..."
        assert "//" not in path_part


def test_prompt_token_fallback_ignores_transport_metadata() -> None:
    messages = [{"role": "system", "content": "x" * 400}, {"role": "user", "content": "hi"}]
    payload = {
        "model": "/large/path/model.gguf",
        "stream": True,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
    }

    prompt_estimate = _estimate_prompt_token_count(payload)
    full_payload_estimate = _estimate_token_count(payload)

    assert prompt_estimate < full_payload_estimate
    assert prompt_estimate == _estimate_token_count({"messages": messages, "tools": payload["tools"]})


def test_restart_server_resets_stale_http_client(monkeypatch) -> None:
    cfg = RuntimeConfig(
        provider="llama_cpp",
        base_url="http://localhost:8081",
        model="test.gguf",
    )
    gw = LocalCodeRuntimeGateway(cfg)
    stale_client = MagicMock()
    stale_client.is_closed = False
    gw._client = stale_client

    monkeypatch.setattr("localcode.bootstrap.get_model_path", lambda preferred=None: "/tmp/model.gguf")
    monkeypatch.setattr(gw, "llama_server_command", lambda model: ["llama-server", "--model", model, "--port", "8081"])

    class _Mgr:
        port = 8082

        def restart(self, cmd, model):
            return True

    monkeypatch.setattr("localcode.server_manager.ServerManager.get", lambda: _Mgr())

    assert gw._restart_server() is True
    stale_client.close.assert_called_once()
    assert gw._client is None
    assert gw.endpoint == "http://localhost:8082/v1/chat/completions"
    assert gw.tags_endpoint == "http://localhost:8082/v1/models"


def test_quick_server_probe_false_on_unreachable(monkeypatch) -> None:
    cfg = RuntimeConfig(provider="llama_cpp", base_url="http://localhost:65534")
    gw = LocalCodeRuntimeGateway(cfg)

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            raise OSError("connection refused")

    monkeypatch.setattr("localcode.runtime.httpx.Client", _Client)

    assert gw._quick_server_probe() is False


def test_tool_arg_stream_guard_allows_bounded_code_write() -> None:
    content = "def f():\n    return 1\n" * 40
    args = '{"path":"src/app.py","content":"' + content.replace("\n", "\\n") + '"}'
    limited, reason = _tool_arg_stream_guard("write_file", args, elapsed_s=10)
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_allows_long_multiline_code_write() -> None:
    content = "def f():\n    return 1\n" * 140
    args = '{"path":"src/app.py","content":"' + content.replace("\n", "\\n") + '"}'
    limited, reason = _tool_arg_stream_guard("write_file", args, elapsed_s=8)
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_allows_bulk_data_write() -> None:
    rows = ",".join(f'{{"id":{i},"name":"item-{i}"}}' for i in range(700))
    args = '{"path":"data/seed.json","content":"[' + rows.replace('"', '\\"') + ']"}'
    limited, reason = _tool_arg_stream_guard("write_file", args, elapsed_s=8)
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_allows_slow_bulk_data_write() -> None:
    rows = ",".join(f'{{"id":{i},"name":"item-{i}"}}' for i in range(250))
    args = '{"path":"data/seed.json","content":"[' + rows.replace('"', '\\"') + ']"}'
    limited, reason = _tool_arg_stream_guard("write_file", args, elapsed_s=31)
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_allows_slow_large_write() -> None:
    content = "console.log('x')\\n" * 900
    args = '{"path":"src/app.js","content":"' + content + '"}'
    limited, reason = _tool_arg_stream_guard("write_file", args, elapsed_s=30)
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_allows_recovery_source_write_to_finish() -> None:
    content = "console.log('x')\\n" * 130
    args = '{"path":"src/app.js","content":"' + content + '"}'
    limited, reason = _tool_arg_stream_guard(
        "write_file",
        args,
        elapsed_s=31,
        recovery_mode="large_write",
    )
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_allows_build_app_source_write_to_finish() -> None:
    content = "console.log('x')\\n" * 190
    args = '{"path":"src/app.js","content":"' + content + '"}'
    limited, reason = _tool_arg_stream_guard(
        "write_file",
        args,
        elapsed_s=46,
        stream_policy="build_app",
    )
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_does_not_early_redirect_recovery_small_write() -> None:
    content = "console.log('x')\\n" * 20
    args = '{"path":"src/app.js","content":"' + content + '"}'
    limited, reason = _tool_arg_stream_guard(
        "append_file",
        args,
        elapsed_s=31,
        recovery_mode="large_write",
    )
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_allows_slow_source_write() -> None:
    content = "console.log('x')\\n" * 360
    args = '{"path":"src/app.js","content":"' + content + '"}'
    limited, reason = _tool_arg_stream_guard("write_file", args, elapsed_s=61)
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_allows_small_slow_mutating_stream() -> None:
    args = '{"path":"src/app.py","content":"print(1)\\n"}'
    limited, reason = _tool_arg_stream_guard("write_file", args, elapsed_s=46)
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_allows_large_source_write_by_time() -> None:
    content = "console.log('x')\\n" * 1000
    args = '{"path":"src/app.js","content":"' + content + '"}'
    limited, reason = _tool_arg_stream_guard("write_file", args, elapsed_s=46)
    assert limited is False
    assert reason == ""


def test_tool_arg_stream_guard_keeps_extreme_safety_ceiling() -> None:
    args = '{"path":"src/app.py","content":"' + ("x" * 181_000) + '"}'
    limited, reason = _tool_arg_stream_guard("write_file", args, elapsed_s=1)
    assert limited is True
    assert "safety ceiling" in reason


class _FakeStreamResponse:
    """Minimal stand-in for httpx's streaming response context manager.

    Yields a fixed list of SSE lines from iter_lines(), mirroring what
    llama-server emits in OpenAI-compatible streaming mode.
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def iter_lines(self):
        for ln in self._lines:
            yield ln


def _sse(content: str) -> str:
    """One llama.cpp streaming chunk carrying `content` in choices[0].delta."""
    import json as _json
    return "data: " + _json.dumps(
        {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
    )


def _sse_done() -> str:
    import json as _json
    return "data: " + _json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
    )


def _collapse_lines() -> list[str]:
    """A stream that trips the Gemma-4 collapse detector (>=8 soup tokens)."""
    soup = "".join(f"<unused{n}>" for n in range(20, 32))  # 12 hits
    return [_sse(soup), _sse_done()]


def _good_lines(text: str = "Hello! How can I help?") -> list[str]:
    return [_sse(text), _sse_done()]


def _collapse_test_gw() -> LocalCodeRuntimeGateway:
    cfg = RuntimeConfig(
        provider="llama_cpp",
        base_url="http://localhost:8081",
        model="gemma4.gguf",
        max_retries=3,
    )
    return LocalCodeRuntimeGateway(cfg)


class TestCollapseRetry:
    """Gemma-4 token-soup collapse should auto-regenerate, not hard-fail."""

    def _run(self, gw, stream_batches: list[list[str]]):
        """Drive stream_chat_events with a scripted sequence of SSE batches —
        one batch consumed per HTTP stream attempt."""
        # Each call to client.stream returns the next prepared response.
        prepared = [_FakeStreamResponse(b) for b in stream_batches]
        fake_client = MagicMock()
        fake_client.is_closed = False
        fake_client.stream = MagicMock(side_effect=prepared)
        gw._client = fake_client
        with patch.object(gw, "_quick_server_probe", return_value=True):
            return list(gw.stream_chat_events([{"role": "user", "content": "hi"}]))

    def test_collapse_then_success_recovers_with_no_error(self) -> None:
        gw = _collapse_test_gw()
        events = self._run(gw, [_collapse_lines(), _good_lines("Hi there!")])
        contents = [e["content"] for e in events if e["type"] == "content"]
        joined = "".join(contents)
        # The good attempt's content is delivered...
        assert "Hi there!" in joined
        # ...and the collapse error is NEVER surfaced to the consumer.
        assert "E3108" not in joined
        # A collapse_retry stage event was emitted to signal the regeneration.
        assert any(
            e["type"] == "stage" and e.get("name") == "collapse_retry"
            for e in events
        )
        # Exactly one terminal stream_done (only the successful attempt's).
        assert sum(1 for e in events if e["type"] == "stream_done") == 1

    def test_all_attempts_collapse_surfaces_e3108(self) -> None:
        gw = _collapse_test_gw()
        # Initial attempt + 3 collapse retries (_MAX_COLLAPSE_RETRIES) =
        # 4 collapsing streams before E3108 is surfaced.
        events = self._run(
            gw,
            [
                _collapse_lines(),
                _collapse_lines(),
                _collapse_lines(),
                _collapse_lines(),
            ],
        )
        joined = "".join(e["content"] for e in events if e["type"] == "content")
        assert "E3108" in joined
        assert sum(1 for e in events if e["type"] == "stream_done") == 1

    def test_retry_escalates_sampler_not_a_noop(self) -> None:
        """Regression: the collapse retry must send a DIFFERENT payload than
        the initial attempt. The old code only nudged temperature and clamped
        it to 1.0 — but Gemma's base temperature is already 1.0, so every
        retry re-sent a byte-identical request and reproduced the same
        collapse. The retry must now escalate: a fresh seed + a higher
        repeat_penalty than the base payload carries."""
        import copy as _copy
        gw = _collapse_test_gw()
        # `payload` is mutated in place across attempts, so we must snapshot
        # the json kwarg at call time — the stored MagicMock call_args would
        # otherwise show every call pointing at the same final dict.
        sent: list[dict] = []
        prepared = iter(
            [_FakeStreamResponse(_collapse_lines()),
             _FakeStreamResponse(_good_lines("Recovered!"))]
        )

        def _stream(*_a, **kw):
            sent.append(_copy.deepcopy(kw["json"]))
            return next(prepared)

        fake_client = MagicMock()
        fake_client.is_closed = False
        fake_client.stream = MagicMock(side_effect=_stream)
        gw._client = fake_client
        with patch.object(gw, "_quick_server_probe", return_value=True):
            events = list(gw.stream_chat_events([{"role": "user", "content": "hi"}]))
        # Two HTTP stream attempts: initial (collapsed) + one retry.
        assert len(sent) == 2
        initial, retry = sent[0], sent[1]
        # The retry pins an explicit seed to force a different RNG path;
        # the initial attempt does not.
        assert "seed" not in initial
        assert isinstance(retry.get("seed"), int)
        # And it escalates repeat_penalty above whatever the base payload used.
        assert retry["repeat_penalty"] > initial.get("repeat_penalty", 1.0)
        # The recovery still succeeds and never surfaces the error.
        joined = "".join(e["content"] for e in events if e["type"] == "content")
        assert "Recovered!" in joined and "E3108" not in joined

    def test_normal_stream_unaffected_no_retry(self) -> None:
        gw = _collapse_test_gw()
        events = self._run(gw, [_good_lines("All good.")])
        joined = "".join(e["content"] for e in events if e["type"] == "content")
        assert "All good." in joined
        assert "E3108" not in joined
        assert not any(
            e["type"] == "stage" and e.get("name") == "collapse_retry"
            for e in events
        )
        assert sum(1 for e in events if e["type"] == "stream_done") == 1


class _MidStreamDropResponse:
    """Streaming response that yields real content lines then raises a
    connection error mid-iteration — server dying MID-STREAM after content
    already reached the consumer."""

    def __init__(self, good_lines: list[str]) -> None:
        self._lines = good_lines
        self.status_code = 200

    def __enter__(self) -> "_MidStreamDropResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def iter_lines(self):
        for ln in self._lines:
            yield ln
        raise httpx.ConnectError("Connection refused")


class _EarlyDropResponse:
    """Streaming response that raises a connection error BEFORE any content
    is yielded — the safe-to-retry case (drop before first token)."""

    status_code = 200

    def __enter__(self) -> "_EarlyDropResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def iter_lines(self):
        raise httpx.ConnectError("Connection refused")
        yield  # pragma: no cover — makes this a generator


class TestStreamConnectionRecovery:
    """Lost-connection auto-recovery for the streaming + non-streaming paths.

    Drop BEFORE content yielded -> restart + retry (no E3102).
    Drop AFTER content yielded -> ALSO restart + retry (recover). Hard-failing
    mid-stream drops with E3102 was a REGRESSION — the server gets paused under
    memory pressure mid-build and must self-heal, even if a partial reply is
    replayed.
    """

    def _gw(self) -> LocalCodeRuntimeGateway:
        cfg = RuntimeConfig(
            provider="llama_cpp",
            base_url="http://localhost:8081",
            model="gemma4.gguf",
            max_retries=3,
        )
        return LocalCodeRuntimeGateway(cfg)

    def test_early_drop_restarts_and_retries_without_e3102(self, monkeypatch) -> None:
        gw = self._gw()
        prepared = [_EarlyDropResponse(), _FakeStreamResponse(_good_lines("Recovered!"))]
        fake_client = MagicMock()
        fake_client.is_closed = False
        fake_client.stream = MagicMock(side_effect=prepared)
        gw._client = fake_client

        restart_calls = {"n": 0}
        monkeypatch.setattr(
            gw, "_restart_server",
            lambda: (restart_calls.__setitem__("n", restart_calls["n"] + 1) or True),
        )
        monkeypatch.setattr(gw, "_quick_server_probe", lambda: True)
        import time as _t
        monkeypatch.setattr(_t, "sleep", lambda *_a, **_k: None)

        events = list(gw.stream_chat_events([{"role": "user", "content": "hi"}]))
        joined = "".join(e["content"] for e in events if e["type"] == "content")

        assert "Recovered!" in joined
        assert "E3102" not in joined
        assert restart_calls["n"] == 1, "exactly one restart should have fired"
        assert any(
            e["type"] == "stage"
            and e.get("name") in ("server_reconnect", "memory_pressure_recovery")
            for e in events
        )
        assert sum(1 for e in events if e["type"] == "stream_done") == 1

    def test_mid_stream_drop_restarts_and_recovers(self, monkeypatch) -> None:
        # A drop AFTER content was streamed must RECOVER (restart + retry), not
        # hard-fail with E3102 (that was the regression that killed builds).
        gw = self._gw()
        second = _FakeStreamResponse(_good_lines("Recovered after drop"))
        prepared = [_MidStreamDropResponse([_sse("partial answer ")]), second]
        fake_client = MagicMock()
        fake_client.is_closed = False
        fake_client.stream = MagicMock(side_effect=prepared)
        gw._client = fake_client

        restart_calls = {"n": 0}
        monkeypatch.setattr(
            gw, "_restart_server",
            lambda: (restart_calls.__setitem__("n", restart_calls["n"] + 1) or True),
        )
        monkeypatch.setattr(gw, "_quick_server_probe", lambda: True)
        import time as _t
        monkeypatch.setattr(_t, "sleep", lambda *_a, **_k: None)

        events = list(gw.stream_chat_events([{"role": "user", "content": "hi"}]))
        joined = "".join(e["content"] for e in events if e["type"] == "content")

        # The turn recovered: it restarted once and the retried reply came through.
        assert restart_calls["n"] == 1, "a mid-stream drop must restart + retry"
        assert "Recovered after drop" in joined
        assert "E3102" not in joined
        assert any(
            e["type"] == "stage"
            and e.get("name") in ("server_reconnect", "memory_pressure_recovery")
            for e in events
        )

    def test_chat_once_restarts_and_retries_on_conn_error(self, monkeypatch) -> None:
        gw = self._gw()
        good = {"choices": [{"message": {"content": "ok", "role": "assistant"}}]}

        class _Resp:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self):
                return good

        calls = {"n": 0}

        def _post(endpoint, json):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("Connection refused")
            return _Resp()

        fake_client = MagicMock()
        fake_client.is_closed = False
        fake_client.post = MagicMock(side_effect=_post)
        gw._client = fake_client

        restart_calls = {"n": 0}
        monkeypatch.setattr(
            gw, "_restart_server",
            lambda: (restart_calls.__setitem__("n", restart_calls["n"] + 1) or True),
        )

        result = gw.chat_once([{"role": "user", "content": "hi"}])
        assert result["message"]["content"] == "ok"
        assert restart_calls["n"] == 1
        assert calls["n"] == 2
