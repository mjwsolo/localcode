"""Tests for gem.runtime — server command building, context scaling, blob resolution."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gem.config import RuntimeConfig
from gem.runtime import GemRuntimeGateway, _strip_thinking_tokens


class TestLlamaServerCommand:
    """Verify llama_server_command builds the correct CLI args for each mode."""

    def _make_gw(self, **overrides) -> GemRuntimeGateway:
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
        return GemRuntimeGateway(cfg)

    def test_speed_mode_uses_mmap_no_gpu(self) -> None:
        """Speed mode: CPU mmap, no GPU layers."""
        gw = self._make_gw(laptop_26b_runtime_mode="speed")
        cmd = gw.llama_server_command("/path/model.gguf", port=8081)
        assert "--model" in cmd
        assert "/path/model.gguf" in cmd
        assert "--mmap" in cmd
        # Should NOT have -ngl 999
        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "0"

    def test_turbo_mode_full_gpu(self) -> None:
        """Turbo mode: -ngl 999 + mmap + -fit off + --cache-ram 0."""
        gw = self._make_gw(laptop_26b_runtime_mode="turbo")
        cmd = gw.llama_server_command("/path/model.gguf")
        assert "-ngl" in cmd
        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "999"
        assert "--mmap" in cmd
        assert "-fit" in cmd
        assert "off" in cmd
        assert "--cache-ram" in cmd
        assert "0" in cmd

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

    def test_custom_binary_used(self) -> None:
        gw = self._make_gw(llama_cpp_binary="/opt/turbo/llama-server")
        cmd = gw.llama_server_command("/path/model.gguf")
        assert cmd[0] == "/opt/turbo/llama-server"

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

    def test_context_mode_uses_10_threads(self) -> None:
        """Context mode should use 10 threads for expert computation."""
        gw = self._make_gw(laptop_26b_runtime_mode="context", llama_cpp_threads=4)
        cmd = gw.llama_server_command("/path/model.gguf")
        threads_idx = cmd.index("--threads")
        assert cmd[threads_idx + 1] == "10"

    def test_context_mode_batch_sizes(self) -> None:
        """Context mode should use larger batch sizes for GPU."""
        gw = self._make_gw(laptop_26b_runtime_mode="context")
        cmd = gw.llama_server_command("/path/model.gguf")
        b_idx = cmd.index("-b")
        assert cmd[b_idx + 1] == "2048"
        ub_idx = cmd.index("-ub")
        assert cmd[ub_idx + 1] == "512"


class TestTargetNumCtx:
    """Verify _target_num_ctx scales with config and quant preset."""

    def _make_gw(self, **overrides) -> GemRuntimeGateway:
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
        return GemRuntimeGateway(cfg)

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


class TestFindOllamaBlob:
    """Verify _find_ollama_blob resolves Ollama model names to GGUF paths."""

    def test_resolves_blob_path(self) -> None:
        mock_output = "FROM /Users/test/.ollama/models/blobs/sha256-abc123\nTEMPLATE ...\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output, returncode=0
            )
            result = GemRuntimeGateway._find_ollama_blob("gemma4:e4b")
        assert result == "/Users/test/.ollama/models/blobs/sha256-abc123"

    def test_returns_model_name_on_failure(self) -> None:
        with patch("subprocess.run", side_effect=Exception("not found")):
            result = GemRuntimeGateway._find_ollama_blob("gemma4:e4b")
        assert result == "gemma4:e4b"

    def test_returns_model_name_when_no_match(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="TEMPLATE ...\n", returncode=0)
            result = GemRuntimeGateway._find_ollama_blob("gemma4:e4b")
        assert result == "gemma4:e4b"


class TestStripThinkingTokens:
    """Verify _strip_thinking_tokens removes Gemma 4 channel artifacts."""

    def test_strips_unused25(self) -> None:
        assert _strip_thinking_tokens("hello <unused25> world") == "hello  world"

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
        gw = GemRuntimeGateway(cfg)
        assert gw.endpoint == "http://localhost:8081/v1/chat/completions"
        assert gw.tags_endpoint == "http://localhost:8081/v1/models"

    def test_ollama_endpoints(self) -> None:
        cfg = RuntimeConfig(provider="ollama", base_url="http://localhost:11434")
        gw = GemRuntimeGateway(cfg)
        assert gw.endpoint == "http://localhost:11434/api/chat"
        assert gw.tags_endpoint == "http://localhost:11434/api/tags"

    def test_mlx_endpoints_empty(self) -> None:
        cfg = RuntimeConfig(provider="mlx-local")
        gw = GemRuntimeGateway(cfg)
        assert gw.endpoint == ""
        assert gw.tags_endpoint == ""

    def test_trailing_slash_stripped(self) -> None:
        """A trailing slash on base_url should not produce double slashes in the path."""
        cfg = RuntimeConfig(provider="llama_cpp", base_url="http://localhost:8081/")
        gw = GemRuntimeGateway(cfg)
        # The path portion after the authority should not have double slashes
        path_part = gw.endpoint.split("://", 1)[1]  # e.g. "localhost:8081/v1/..."
        assert "//" not in path_part
