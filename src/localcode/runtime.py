from __future__ import annotations

import json
import re
from typing import Any, Iterator

import httpx

from .config import RuntimeConfig


from .model_families import (
    ModelFamily, get_adapter, infer_family_from_profile,
    strip_thinking_tokens as _family_strip,
)


def _strip_thinking_tokens(text: str, family: ModelFamily | None = None) -> str:
    """Strip thinking-channel tokens that leak through in decoded text.

    Defaults to Gemma 4 (the original hardcoded behaviour) when family
    is None, so every existing callsite that didn't pass a family stays
    byte-identical. Qwen / DeepSeek / Llama adapters kick in when the
    caller threads the active family through.
    """
    return _family_strip(text, family)


from .tool_parsing import (
    inject_tool_schemas_into_prompt,
    parse_tool_calls,
)


def _estimate_token_count(value: Any) -> int:
    """Fallback token estimate when a backend omits usage metadata.

    Exact counts come from llama-server/OpenAI-compatible `usage`.
    Some local backends do not send it; returning zero makes telemetry
    useless, so use the same conservative chars/4 approximation already
    used by compaction and streaming UI.
    """
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        text = str(value)
    return max(1, len(text) // 4) if text else 0


def _estimate_prompt_token_count(payload: Any) -> int:
    """Estimate model input tokens without counting transport metadata.

    Fallback telemetry should represent the prompt the model saw, not the
    full HTTP request object. Counting `model`, sampler options, booleans,
    and other request metadata made tiny turns like "hi" display as ~3k
    input tokens even when the actual prompt text was much smaller.
    Native tool schemas are part of the model-facing request for OpenAI-style
    chat APIs, so include them; omit runtime-only options.
    """
    if not isinstance(payload, dict):
        return _estimate_token_count(payload)
    prompt_parts: dict[str, Any] = {}
    messages = payload.get("messages")
    if messages:
        prompt_parts["messages"] = messages
    tools = payload.get("tools")
    if tools:
        prompt_parts["tools"] = tools
    if prompt_parts:
        return _estimate_token_count(prompt_parts)
    prompt = payload.get("prompt")
    if prompt:
        return _estimate_token_count(prompt)
    return _estimate_token_count(payload)


def _tool_arg_stream_guard(
    tool_name: str,
    arguments: str,
    *,
    elapsed_s: float,
    recovery_mode: str = "",
    stream_policy: str = "",
) -> tuple[bool, str]:
    """Decide whether an in-flight tool argument stream is clearly malformed.

    Do not police normal file size here. Let the model finish tool calls
    and rely on tool errors / turn caps for recovery. This guard is only an
    emergency ceiling for broken JSON or runaway arguments that would
    otherwise consume the whole context window.
    """
    name = (tool_name or "").strip()
    arg_len = len(arguments or "")
    if arg_len <= 0:
        return False, ""

    if arg_len > 180_000:
        return True, "tool argument exceeded safety ceiling"
    return False, ""


def _pressure_kill_recent(within_seconds: int = 60) -> bool:
    """True if the pressure monitor SIGTERM'd llama-server in the last N seconds.

    The pressure-monitor thread (memory_guard.py) writes a marker file
    `~/.localcode/last-pressure-kill.txt` whenever it kills the server.
    Format: `level=<int>\\ntime=<unix_seconds>\\n`. We read the timestamp
    and decide whether the most recent kill is recent enough that the
    next "connection refused" is almost certainly the same incident,
    not an unrelated server crash.

    Used by the stream-chat retry path: if we get connection-refused AND
    a recent pressure kill is on record, we know the server died for a
    well-understood reason and can recover by restarting (after a brief
    cooldown so memory pressure subsides).
    """
    import time
    from .paths import pressure_kill_marker_path
    marker = pressure_kill_marker_path()
    try:
        if not marker.is_file():
            return False
        body = marker.read_text()
        for line in body.splitlines():
            if line.startswith("time="):
                try:
                    ts = int(line.split("=", 1)[1])
                except ValueError:
                    return False
                return (time.time() - ts) <= within_seconds
    except Exception:
        return False
    return False


def _clear_pressure_kill_marker() -> None:
    """Remove the pressure-kill marker after we've recovered from one.

    Without this, future unrelated connection failures (e.g. user
    restarted the server manually, sleep/wake, etc.) would still see the
    stale marker and incorrectly attribute themselves to memory
    pressure. Idempotent — silent no-op if the file is already gone."""
    from .paths import pressure_kill_marker_path
    try:
        pressure_kill_marker_path().unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


class RuntimeErrorWithContext(RuntimeError):
    pass


def _error_message(value: Any) -> str:
    """Return a safe one-line-ish error string for arbitrary backend payloads."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


class StreamEvent(dict):
    pass


class LocalCodeRuntimeGateway:
    """Talks to Ollama, llama.cpp, MLX, or HuggingFace local backends."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._hf_backend: Any | None = None
        self._mlx_backend: Any | None = None
        self._client: httpx.Client | None = None
        self.last_response_meta: dict[str, Any] = {}  # for token tracking
        self._last_thinking: str = ""  # thinking extracted from MLX output
        base = self.config.base_url.rstrip("/")
        if self.config.provider == "llama_cpp":
            self.endpoint = f"{base}/v1/chat/completions"
            self.tags_endpoint = f"{base}/v1/models"
        elif self.config.provider in ("mlx-local", "huggingface-local"):
            self.endpoint = ""
            self.tags_endpoint = ""
        else:
            self.endpoint = f"{base}/api/chat"
            self.tags_endpoint = f"{base}/api/tags"

    @property
    def client(self) -> httpx.Client:
        """Persistent connection-pooled HTTP client for speed."""
        if self._client is None or self._client.is_closed:
            # CPU-only mode (8GB) needs much longer timeout — inference is slow
            read_timeout = float(self.config.request_timeout_seconds)
            if self.config.llama_cpp_gpu_layers == 0:
                read_timeout = max(read_timeout, 600.0)  # 10 min for CPU mode
            self._client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=read_timeout,
                    write=30.0,
                    pool=10.0,
                ),
                limits=httpx.Limits(
                    max_connections=4,
                    max_keepalive_connections=2,
                    keepalive_expiry=300,
                ),
            )
        return self._client

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()

    def llama_server_command(self, model_path: str, port: int = 8081) -> list[str]:
        """Build the optimal llama-server launch command with all speed flags.

        Two runtime modes for 16GB Apple Silicon:
        - "speed": CPU mmap (ngl=0) — fastest decode (18.5 tok/s), 10K context
        - "context": GPU attention + CPU experts — 32K context, fast prompt eval (45 tok/s)

        Usage: subprocess.Popen(engine.llama_server_command("/path/to/model.gguf"))
        """
        from .bootstrap import _turboquant_binary_path
        from pathlib import Path as _P
        # Prefer the configured binary, BUT fall back to discovery if it
        # no longer exists on disk. This is the failure mode we hit when
        # the user moved the repo to ~/Desktop/GitHub and config still
        # pointed at the old ~/localcodetest path — Errno 2 surfaces as
        # "Failed to start" with no obvious cause. Self-healing here
        # silently re-discovers and persists the new path next time
        # `save_config` runs.
        configured = self.config.llama_cpp_binary or ""
        if configured and _P(configured).is_file():
            binary = configured
        else:
            discovered = _turboquant_binary_path()
            binary = str(discovered) if discovered else "llama-server"
            # Update the in-memory config so the next save_config persists
            # the corrected path (no need to wait for the user to re-run
            # `localcode setup`).
            if discovered and configured != str(discovered):
                try:
                    self.config.llama_cpp_binary = str(discovered)
                except Exception:
                    pass
        mode = self.config.laptop_26b_runtime_mode

        # Auto-detect threads and batch size if not explicitly set
        import os, platform
        if self.config.llama_cpp_threads <= 0:
            cpu_count = os.cpu_count() or 4
            if platform.system() == "Darwin":
                threads = max(4, min(cpu_count, 12))  # Apple Silicon: use perf cores
            else:
                threads = max(2, min(cpu_count, 8))
        else:
            threads = self.config.llama_cpp_threads
        if mode == "context":
            threads = max(threads, 10)  # context mode benefits from all cores

        # ── cohere2moe (North-Mini-Code) ──────────────────────────────
        # The TurboQuant fork can't load cohere2moe; LocalCode builds a
        # dedicated stock llama-server from llama.cpp PR #24260. It's a
        # stock binary, so it only accepts STOCK flags — none of the
        # TurboQuant extras (turbo4 KV, --spec-type, -fit, --ctx-checkpoints).
        try:
            from .models_catalog import by_filename as _bf
            _choice = _bf(_P(model_path).name)
            _arch = str(getattr(_choice, "architecture", "")) if _choice else ""
        except Exception:
            _arch = ""
        if "cohere" in _arch:
            from .bootstrap import cohere_server_path
            cbin = (self.config.cohere_server_binary or "").strip()
            if not (cbin and _P(cbin).is_file()):
                found = cohere_server_path()
                cbin = str(found) if found else ""
            if cbin:
                return [
                    cbin,
                    "--model", model_path,
                    "--port", str(port),
                    "--ctx-size", str(self._target_num_ctx(model_path=model_path)),
                    "--threads", str(threads),
                    "--flash-attn", "on",
                    "--mmap", "-ngl", "999",
                    "--jinja",
                    "-b", "512", "-ub", "512",
                ]
            # No cohere binary yet — fall through; setup builds it on select,
            # and _restart_server surfaces a clear error if it's truly missing.

        # ── Vanilla / stock llama.cpp compatibility (Linux CI, no Metal) ──
        # The bundled server is a TurboQuant fork whose flags (turbo4 KV,
        # -fit, --ctx-checkpoints, --spec-type) stock llama.cpp rejects.
        # When LOCALCODE_SERVER_FLAVOR=vanilla, emit a minimal stock-
        # compatible CPU command and return early. This is a FUNCTIONAL
        # path (does the agent run against a real model?) — NOT the
        # Apple-Silicon TurboQuant perf path.
        if os.environ.get("LOCALCODE_SERVER_FLAVOR", "").strip().lower() == "vanilla":
            ctx = min(8192, self._target_num_ctx(model_path=model_path))
            return [
                binary,
                "--model", model_path,
                "--port", str(port),
                "--ctx-size", str(ctx),
                "--threads", str(threads),
                "-ngl", "0",            # CPU only
                "--jinja",
                "-b", "256", "-ub", "256",
            ]

        # Flash attention helps on GPU (Metal/CUDA) but hurts on CPU —
        # the kernel does more arithmetic to save memory, which is
        # backwards when memory is plentiful and arithmetic is the
        # bottleneck. Only enable when we're actually offloading to GPU.
        ngl_for_flash = self.config.llama_cpp_gpu_layers if self.config.llama_cpp_gpu_layers >= 0 else 999
        flash_attn_on = ngl_for_flash > 0
        cmd = [
            binary,
            "--model", model_path,
            "--port", str(port),
            "--ctx-size", str(self._target_num_ctx(model_path=model_path)),
            "--threads", str(threads),
            "--flash-attn", "on" if flash_attn_on else "off",
        ]
        # Multimodal projector — if the currently-selected catalog entry
        # has an mmproj sidecar AND the sidecar is on disk, pass --mmproj
        # so llama-server loads the vision encoder alongside the text
        # decoder. We don't auto-download here; the chat layer prompts
        # the user on first vision use (see voice.py / vision flow).
        try:
            from .models_catalog import by_filename
            from pathlib import Path as _P
            choice = by_filename(_P(model_path).name)
            if choice is not None and choice.mmproj_path and choice.mmproj_path.is_file():
                cmd.extend(["--mmproj", str(choice.mmproj_path)])
        except Exception:
            # Never block server launch on mmproj lookup
            pass

        if mode in ("turbo", "turbo-think"):
            # Respect gpu_layers config (0 = CPU-only for 8GB machines)
            ngl = self.config.llama_cpp_gpu_layers if self.config.llama_cpp_gpu_layers >= 0 else 999
            cmd.extend(["--mmap", "-ngl", str(ngl)])
        elif mode == "context":
            # GPU mode: attention on Metal, experts on CPU, mmap for SSD paging
            cmd.extend(["--mmap", "-ngl", "999", "-ot", "exps=CPU"])
        elif self.config.llama_cpp_expert_offload:
            # Explicit expert offload (legacy config)
            cmd.extend(["--mmap", "-ngl", "999", "-ot", "exps=CPU"])
        else:
            # Default GPU mode
            cmd.extend(["--mmap", "-ngl", "999"])
        # KV cache compression (asymmetric: q8_0 K + turbo4 V recommended)
        ctk = self.config.kv_cache_type_k
        ctv = self.config.kv_cache_type_v
        if ctk and ctk != "f16":
            cmd.extend(["--cache-type-k", ctk])
        if ctv and ctv != "f16":
            cmd.extend(["--cache-type-v", ctv])
        # Speculative decoding (mutual exclusion: draft model > lookup > ngram).
        # Speculative decoding is LOSSLESS — every drafted token is verified
        # against the real model, so output is identical, just faster.
        if self.config.llama_cpp_draft_model:
            draft_path = self.config.llama_cpp_draft_model
            # Support Ollama blob paths (sha256-...)
            if not draft_path.startswith("/") and "sha256" not in draft_path:
                draft_path = self._find_ollama_blob(draft_path)
            cmd.extend(["--model-draft", draft_path,
                        "--draft-max", str(self.config.llama_cpp_draft_max)])
        elif self.config.llama_cpp_lookup_cache:
            # Prompt lookup decoding: matches n-grams from input in output (2-4x on edits)
            cmd.extend(["--lookup-cache-dynamic", "/tmp/localcode-lookup.bin"])
        elif self.config.llama_cpp_spec_type and self.config.llama_cpp_spec_type != "none":
            cmd.extend(["--spec-type", self.config.llama_cpp_spec_type,
                        "--draft-max", str(self.config.llama_cpp_draft_max)])
        elif self.config.llama_cpp_spec_type != "none":
            # DEFAULT (nothing configured): in-context n-gram speculative
            # decoding. Free and safe — no draft model (so none of the
            # vocab-mismatch / double-bandwidth problems of --model-draft),
            # no extra RAM, stateless (unlike lookup-cache's /tmp file).
            # It accelerates the repetitive token runs that dominate coding
            # output — identifiers, syntax, and re-emitted file content
            # during edits. Opt out with llama_cpp_spec_type = "none".
            cmd.extend(["--spec-type", "ngram-mod",
                        "--draft-max", str(self.config.llama_cpp_draft_max)])
        # Batch sizes — GPU benefits from bigger batches for prompt eval,
        # but large hybrid/MoE GGUFs on 16 GB Macs can OOM during prefill
        # with 2K batches. Respect explicit config first; otherwise clamp
        # the known-unstable Qwen-35B-on-16GB shape.
        if mode in ("turbo", "turbo-think", "context"):
            batch = self._effective_llama_batch_size(model_path)
            cmd.extend(["-b", str(batch), "-ub", str(min(512, batch))])
        else:
            batch = self.config.llama_cpp_batch_size
            if batch <= 0:
                batch = 2048 if platform.system() == "Darwin" else 128
            cmd.extend(["-b", str(batch), "-ub", str(min(512, batch))])
        # Honor the GGUF's embedded jinja chat template (Unsloth Apr-2026 GGUFs ship
        # the corrected Gemma 4 template that fixes tool-call special-token round-tripping)
        cmd.extend(["--jinja"])
        # Enable context shifting so sessions that exceed our ctx-size degrade
        # gracefully instead of hard-crashing the request. llama.cpp's
        # `--context-shift` updates RoPE frequencies of existing tokens and
        # evicts the oldest to make room for new ones, without re-processing
        # the rest of the KV cache. Default in llama-server is DISABLED — we
        # verified via `--help`. Our application-layer compaction
        # (_compact_messages in agent.py) is the primary defence, but when
        # a turn's own payload blows past the budget mid-decode, shifting is
        # the safety net. Cost is effectively zero (only fires on overflow).
        cmd.extend(["--context-shift"])
        # SSM context checkpoints — critical for hybrid-memory models like
        # Qwen 3.6 where the Mamba-2 state is recurrent and llama-server
        # cannot naturally reuse KV cache across requests. Checkpointing
        # serializes the ~63 MiB recurrent state at periodic boundaries so
        # subsequent multi-turn requests can restore and skip prefix
        # re-evaluation. Measured 19× speedup on turn 2 vs cold turn 1
        # (5932 ms → 302 ms prompt eval at 2K ctx).
        # --ctx-checkpoints 32     keep up to 32 rolling state snapshots
        # --checkpoint-every-n-tokens 2048  snapshot every 2K tokens during prefill
        # For pure-attention models these are no-ops; cheap to always set.
        # Note: --swa-full would additionally warm the attention SWA cache
        # but causes Metal OOM at our 14336 MiB sysctl on 16 GB Macs. Skip.
        cmd.extend([
            "--ctx-checkpoints", "32",
            "--checkpoint-every-n-tokens", "2048",
        ])
        # Single slot, disable fit check (we manage memory via sysctl)
        cmd.extend(["-np", "1", "-fit", "off"])
        return cmd

    @staticmethod
    def _find_ollama_blob(model_name: str) -> str:
        """Find the GGUF blob path for an Ollama model."""
        import subprocess
        try:
            result = subprocess.run(
                ["ollama", "show", model_name, "--modelfile"],
                capture_output=True, text=True, check=False,
            )
            for line in result.stdout.splitlines():
                if line.startswith("FROM ") and ".ollama" in line:
                    return line.split("FROM ", 1)[1].strip()
        except Exception:
            pass
        return model_name  # fallback: return as-is

    def _system_ram_gb(self) -> int:
        try:
            import subprocess
            mem_bytes = int(subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2
            ).stdout.strip())
            return max(1, mem_bytes // (1024 ** 3))
        except Exception:
            return 16

    def _is_large_qwen_gguf(self, model_path: str | None = None) -> bool:
        name = (model_path or self.config.model or "").lower()
        return "qwen" in name and any(token in name for token in ("30b", "32b", "35b"))

    def _is_large_gemma_gguf(self, model_path: str | None = None) -> bool:
        name = (model_path or self.config.model or "").lower()
        return "gemma" in name and any(token in name for token in ("26b", "27b", "30b"))

    def _is_large_local_gguf(self, model_path: str | None = None) -> bool:
        return self._is_large_qwen_gguf(model_path) or self._is_large_gemma_gguf(model_path)

    def _effective_llama_batch_size(self, model_path: str | None = None) -> int:
        configured = self.config.llama_cpp_batch_size
        if self._is_large_qwen_gguf(model_path) and self._system_ram_gb() <= 18:
            # This exact shape was observed to Metal-OOM on a 16 GB M4:
            # Qwen3.6-35B + full GPU offload + 64K/32K ctx + -b 2048.
            # Treat stale config as advisory here; a launch that cannot
            # complete prefill is worse than a slightly slower safe batch.
            if configured and configured > 0:
                return min(max(128, int(configured)), 512)
            return 512
        if configured and configured > 0:
            return max(128, int(configured))
        import platform
        return 2048 if platform.system() == "Darwin" else 128

    def generate_once(self, messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
        """Call model for a one-shot generation (no streaming, no tool calls).

        Routes to the correct endpoint per provider:
        - Ollama: /api/generate (bypasses chat template to avoid <|tool_response> tokens)
        - llama.cpp: /v1/chat/completions (OpenAI-compatible)
        - MLX/HF: in-process generation
        """
        if self.config.provider == "llama_cpp":
            from .thinking import should_use_thinking
            use_think = should_use_thinking(
                self.config.laptop_26b_runtime_mode,
                getattr(self.config, "internal_thinking_mode", "off"),
            )
            budget = max_tokens or 4096
            if use_think:
                budget = max(budget * 3, 4096)
            result = self.chat_once(messages, tools=None, think=use_think,
                                     num_predict=budget)
            msg = result.get("message", {})
            content = (msg.get("content", "") or "").strip()
            # Fallback: if content is empty, thinking may contain the real response
            if not content:
                content = (msg.get("thinking", "") or "").strip()
            # Strip Gemma 4 thinking channel tokens that leak through
            content = _strip_thinking_tokens(content)
            # If thinking stripped everything, retry without thinking
            if not content and use_think:
                result = self.chat_once(messages, tools=None, think=False,
                                         num_predict=max_tokens or 4096)
                msg = result.get("message", {})
                content = _strip_thinking_tokens((msg.get("content", "") or "").strip())
            return content

        if self.config.provider == "mlx-local":
            return self._mlx_generate(messages)

        if self.config.provider == "huggingface-local":
            return self._hf_generate(messages)

        # Ollama: use /api/generate to bypass chat template
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"\nUser: {content}")
            elif role == "assistant":
                parts.append(f"\nAssistant: {content}")
        prompt = "\n".join(parts) + "\nAssistant:"

        base = self.config.base_url.rstrip("/")
        opts = self._options(num_predict_override=max_tokens or 4096)

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": opts,
        }

        # Long timeout for code generation — 26B at 8 tok/s needs ~5min for 150 lines
        response = self.client.post(
            f"{base}/api/generate", json=payload,
            timeout=httpx.Timeout(connect=10, read=600, write=30, pool=10),
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def healthcheck(self) -> tuple[bool, str]:
        if self.config.provider == "mlx-local":
            try:
                self._get_mlx_backend()
                return True, self.config.mlx_model_id or self.config.model or "mlx local model"
            except Exception as exc:
                return False, str(exc)
        if self.config.provider == "huggingface-local":
            try:
                self._get_hf_backend()
                return True, self.config.huggingface_model_id or self.config.model or "local model"
            except Exception as exc:
                return False, str(exc)
        try:
            response = self.client.get(self.tags_endpoint)
            response.raise_for_status()
            return True, self.tags_endpoint
        except Exception as exc:
            return False, str(exc)

    def list_models(self) -> list[str]:
        if self.config.provider == "mlx-local":
            model_id = self.config.mlx_model_id or self.config.model
            return [model_id] if model_id else []
        if self.config.provider == "huggingface-local":
            model_id = self.config.huggingface_model_id or self.config.model
            return [model_id] if model_id else []
        response = self.client.get(self.tags_endpoint)
        response.raise_for_status()
        data = response.json()
        if self.config.provider == "llama_cpp":
            return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        return [m["name"] for m in data.get("models", []) if "name" in m]

    def _target_num_ctx(
        self,
        num_ctx_override: int | None = None,
        model_path: str | None = None,
    ) -> int:
        if num_ctx_override is not None:
            return max(1024, num_ctx_override)

        # `max_context_chars` is our primary policy knob. Convert it to an
        # approximate token budget without forcing a large 16k floor that
        # defeats the small-machine presets.
        num_ctx = max(2048, self.config.max_context_chars // 4)

        if self.config.quant_preset == "smallest":
            return min(num_ctx, 2048)
        if self.config.quant_preset == "fastest":
            turbo = self.config.kv_cache_type_v.startswith("turbo")
            if self.config.laptop_26b_runtime_mode in ("context", "turbo", "turbo-think") and turbo:
                # Scale context based on system RAM (via the shared helper so
                # tests can mock it — an inline sysctl call here read the real
                # machine's RAM regardless and produced 128K on big Macs).
                ram_gb = self._system_ram_gb()
                # Tier ladder MUST be monotonic — more RAM can only hold
                # more KV, never less. Pre-2026-04 the 16 GB branch was
                # 32K and the ladder (32K / 48K / 64K / 128K) made sense.
                # Then the multi-region mmap fix (llama-cpp-turboquant
                # 3d66675b8) freed ~3 GB of Metal wired budget on 16 GB
                # M4 and we bumped 16 GB to 64K — but forgot to update
                # the rest. Result was 24 GB = 48K, LESS than 16 GB.
                #
                # Conservative fix: floor every tier at the 16 GB value
                # (64K), since 16 GB is the only case we've actually
                # measured (peak wired ≈ 13.9 GB of 14.3 GB cap). Leave
                # the 64 GB upper bound at 128K since that was validated
                # earlier. The 24 GB / 32 GB tiers collapse to the
                # same 64K until someone runs the mmap-patched server
                # on real 24 / 32 GB hardware and measures their safe
                # ceiling — then bump explicitly, not by extrapolation.
                if self._is_large_local_gguf(model_path) and ram_gb <= 18:
                    # Validated path on 16 GB Apple Silicon: q8_0-K +
                    # turbo4-V at 64K ≈ 2.8 GB KV, fits the 14336 MB
                    # Metal wired budget. We always pick 64K here,
                    # regardless of whatever `max_context_chars` is in
                    # the user's config — a stale `performance.py`
                    # preset (max_context_chars=10000 → num_ctx=2500)
                    # was leaving users below the floor needed for
                    # system prompt + tool schemas + a single user
                    # message, producing E3103 ("Conversation is too
                    # long for this model") on the FIRST turn of an
                    # otherwise-empty session. Users who genuinely want
                    # less context should switch off turbo KV (set
                    # `kv_cache_type_v` to a non-turbo type) or change
                    # `laptop_26b_runtime_mode` away from "turbo" /
                    # "context" — those branches respect num_ctx as
                    # before. Inside the validated turbo branch, 64K
                    # is the floor.
                    return 65536
                if ram_gb >= 64:
                    return 131072  # 128K context (validated)
                return 65536       # 16-32 GB: 64K (validated on 16 GB)
            return min(num_ctx, 16384 if turbo else 3072)
        return num_ctx

    def _options(self, num_ctx_override: int | None = None, num_predict_override: int | None = None) -> dict[str, Any]:
        # Anti-repetition sampler stack — fixes the IQ3_S quantization-
        # collapse loop ("I'll use bash to run the commands." × 14)
        # WITHOUT a wall-clock cap. Three layers stacked because no
        # single sampler reliably breaks Gemma 4 IQ3_S MoE collapses:
        #
        #   1. DRY (Don't Repeat Yourself) — penalises N-token
        #      sequences that have appeared in recent output, scaling
        #      the penalty exponentially with the match length. Bumped
        #      multiplier 0.8 → 1.5 and allowed_length 2 → 1 to make
        #      the penalty hit single-token repeats too. Required for
        #      Gemma 4 IQ3_S where the router collapses onto a few
        #      experts and emits the same token over and over.
        #   2. repeat_penalty — independent of DRY; divides logits of
        #      tokens that have appeared in repeat_last_n by 1.15.
        #      Catches loops DRY misses (single tokens, short phrases
        #      below DRY threshold).
        #   3. min_p — cuts the long tail of low-probability tokens,
        #      forcing sampling to draw from the genuinely diverse
        #      head of the distribution rather than collapsing onto
        #      the single high-probability "loop" continuation.
        #
        # Sampler params follow Unsloth's official Gemma 4 config:
        # temperature=1.0, top_p=0.95, top_k=64. min_p=0.05 added
        # because Unsloth's repetition fix recommends it for
        # heavily-quantized MoE models.
        # Sampler tuning notes (2026-04-27):
        # - repeat_last_n was 256 → 64. With 256, `repeat_penalty`
        #   was punishing legitimate repeats of strings the model
        #   needs to emit verbatim (file paths containing the user's
        #   account name, recurring function names, JSON keys). Real
        #   failure: the model emitted several mangled variants of a
        #   username because the correct tokens were penalty-suppressed.
        #   64 is tight enough to break adjacent repetition without
        #   hitting same-string-far-back-in-prompt cases.
        # - dry_allowed_length was 1 → 2. allowed_length=1 made
        #   DRY penalise even single-token repeats, which broke
        #   identifiers that must repeat exactly. allowed_length=2
        #   still catches phrasal loops (which span many tokens)
        #   but lets single tokens like "marc" or path components
        #   repeat freely.
        # - dry_multiplier kept at 1.5 — when DRY does fire on a
        #   legit phrasal loop, we want it to bite hard.
        family = infer_family_from_profile(
            getattr(self.config, "model", "") or getattr(self.config, "profile", "") or ""
        )
        if family == ModelFamily.QWEN:
            # Sampler aligned to Unsloth's published Qwen3.6 IQ2_M
            # non-thinking spec — temperature=0.7, top_p=0.8, top_k=20.
            #
            # presence_penalty was 1.5 (Unsloth's documented fix for
            # the "planning-loop" failure: "Wait, I'll write the
            # response. Wait, I'll check…"). Removed 2026-04-29 after
            # this scenario manifested:
            #
            #   user: "hows the weather in spain?"
            #   model: bash → "It's 18°C in Spain right now."
            #   model: "It's currently 18 °C in Spain."
            #   model: bash → "The weather in Spain is around 18°C…"
            #   model: bash → "The weather in Spain is currently…"
            #
            # presence_penalty subtracts a fixed amount from the logit
            # of every token already in context. Once the model emits
            # an answer, its component tokens — INCLUDING the EOS /
            # sentence-end tokens that appeared after similar prior
            # sentences — get penalised for the rest of the round.
            # Instead of stopping, the model finds different words to
            # say the same thing, then a third paraphrase. Unsloth's
            # 1.5 was for the planning-preamble failure (a different
            # bug class) and we'd been masking it with the missing
            # sampler-forwarding bug fixed in commit 418b038 today.
            # DRY (re-enabled below) is the right tool against intra-
            # round repetition; presence_penalty fights itself.
            #
            # DRY itself: previously disabled citing a 2026-04-27
            # incident where it corrupted usernames inside
            # `/Users/<name>/...` paths. Root cause was
            # `dry_penalty_last_n=-1` (whole prompt) — already fixed
            # to 256 generated tokens. With DRY scoped to the decode
            # window, prompt-side identifiers are safe AND we catch
            # intra-round phrasal loops.
            temperature = min(float(self.config.temperature), 0.7)
            top_p = 0.8
            top_k = 20
            min_p = 0.0
            repeat_penalty = 1.10
            repeat_last_n = 64
            dry_multiplier = 1.5
            dry_base = 1.75
            dry_allowed_length = 2
            dry_penalty_last_n = 256
            presence_penalty = 0.0
        else:
            temperature = float(self.config.temperature)
            top_p = 0.95
            top_k = 64
            min_p = 0.05
            repeat_penalty = 1.15
            repeat_last_n = 64
            dry_multiplier = 1.5
            dry_base = 1.75
            dry_allowed_length = 2
            dry_penalty_last_n = 256
            # Google publishes no presence_penalty recommendation for
            # Gemma 4. Leaving at 0.0 (server default) until there's a
            # sourced reason to deviate.
            presence_penalty = 0.0

        opts: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": self._target_num_ctx(num_ctx_override),
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "dry_multiplier": dry_multiplier,
            "dry_base": dry_base,
            "dry_allowed_length": dry_allowed_length,
            # Limit DRY to the last 256 GENERATED tokens — NOT the whole
            # prompt (-1). With -1, DRY was penalising any string that
            # appeared anywhere in the system prompt, file paths, prior
            # tool results, etc. — including the user's username when
            # it tried to emit a path. Real failure 2026-04-27: model
            # produced several mangled username variants because the
            # username appeared dozens of times in context and DRY
            # refused to let the correct token sequence repeat. Scoping to 256
            # generated tokens means DRY catches loops the model
            # CREATES this round, while leaving legitimate prompt-side
            # repetition (paths, function names, identifiers) untouched.
            "dry_penalty_last_n": dry_penalty_last_n,
        }
        if self.config.mode == "fast":
            opts["num_predict"] = 4096  # cap generation for speed
        if num_predict_override is not None:
            if num_predict_override == -1:
                opts["num_predict"] = -1  # unlimited — model stops at EOS
            else:
                opts["num_predict"] = max(64, int(num_predict_override))
        return opts

    def chat_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        think: bool = False,
        format: dict[str, Any] | str | None = None,
        num_predict: int | None = None,
    ) -> dict[str, Any]:
        if self._diffusion_choice() is not None:
            # Same one-shot CLI backend as stream_chat_events, collected
            # into the non-streaming response shape.
            content_parts: list[str] = []
            tool_calls: list[Any] = []
            for ev in self._stream_diffusion_events(
                messages, tools=tools, num_predict=num_predict
            ):
                if ev.get("type") == "content":
                    content_parts.append(str(ev.get("content") or ""))
                elif ev.get("type") == "tool_calls":
                    tool_calls = ev.get("tool_calls") or []
            return {"message": {"content": "".join(content_parts), "tool_calls": tool_calls}}
        if self.config.provider == "mlx-local":
            # Inject tool schemas into system prompt for MLX
            effective_messages = messages
            if tools:
                effective_messages = self._inject_tools_into_messages(messages, tools)
            content = self._mlx_generate(effective_messages)
            if tools:
                parsed = parse_tool_calls(content)
                if parsed.has_tools:
                    return {"message": {"content": parsed.content, "tool_calls": parsed.to_ollama_format()}}
            return {"message": {"content": content, "tool_calls": []}}
        if self.config.provider == "huggingface-local":
            effective_messages = messages
            if tools:
                effective_messages = self._inject_tools_into_messages(messages, tools)
            content = self._hf_generate(effective_messages)
            if tools:
                parsed = parse_tool_calls(content)
                if parsed.has_tools:
                    return {"message": {"content": parsed.content, "tool_calls": parsed.to_ollama_format()}}
            return {"message": {"content": content, "tool_calls": []}}

        # Reset idle-suspend countdown — a chat is incoming.
        try:
            from .server_manager import ServerManager as _SM
            _SM.get().mark_activity()
        except Exception:
            pass
        payload = self._payload(messages, stream=False, tools=tools, think=think, num_predict=num_predict)
        # Structured output: Ollama grammar-constrains output to match this schema
        if format is not None:
            payload["format"] = format
        last_error: Exception | None = None
        for _ in range(max(1, self.config.max_retries + 1)):
            try:
                response = self.client.post(self.endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    raise RuntimeErrorWithContext(_error_message(data["error"]))
                if self.config.provider == "llama_cpp":
                    return {"message": self._extract_message(data)}
                self.last_response_meta = {
                    k: data.get(k, 0) for k in
                    ("prompt_eval_count", "eval_count", "total_duration", "load_duration")
                }
                return data
            except Exception as exc:
                last_error = exc
                # Server error or connection lost — try restarting
                is_500 = hasattr(exc, 'response') and getattr(exc.response, 'status_code', 0) == 500
                is_conn_err = "connect" in str(exc).lower() or "refused" in str(exc).lower()
                if is_500 or is_conn_err:
                    try:
                        self._restart_server()
                    except Exception:
                        pass
        raise RuntimeErrorWithContext(str(last_error) if last_error else "runtime request failed")

    def _restart_server(self) -> bool:
        """Kill and relaunch llama-server through the single lifecycle owner.

        Returns True iff healthy after restart.

        CRITICAL: after restart, RE-READ the actual port from
        ServerManager and update self.config.base_url + self.endpoint.
        ServerManager has port-fallback (default 8081 → 8082 → 8083 if
        the default is held by a stuck process from a prior session).
        Without propagating the fallback port back here, every HTTP
        request continues hitting 8081 — connection refused — while a
        perfectly healthy server runs on 8082. That was the recurring
        `[E3102] Lost connection to the model server` after model
        swaps. (See RESUME.md port-isolation TODO.)
        """
        # Diffusion models have NO persistent server — each turn spawns the
        # one-shot llama-diffusion-cli. Trying to start a llama-server for
        # them just times out (E1002). Nothing to restart → report ready so
        # the /model hot-swap and recovery paths don't choke on them.
        if self._diffusion_choice() is not None:
            return True

        from .bootstrap import get_model_path
        from .server_manager import ServerManager
        from pathlib import Path

        preferred = Path(self.config.model).name if self.config.model else None
        model = get_model_path(preferred)
        if not model:
            return False
        try:
            if self._client is not None and not self._client.is_closed:
                self._client.close()
        except Exception:
            pass
        self._client = None
        cmd = self.llama_server_command(str(model))
        mgr = ServerManager.get()
        ok = mgr.restart(cmd, str(model))
        # Propagate the (possibly fallback) port back to our config and
        # endpoint URL so downstream HTTP requests hit the live server.
        try:
            actual_port = mgr.port
            new_base = f"http://localhost:{actual_port}"
            if self.config.base_url != new_base:
                self.config.base_url = new_base
                # Endpoint URLs are derived from base_url at __init__;
                # rebuild them here so the rest of the gateway uses
                # the new port.
                if self.config.provider == "llama_cpp":
                    self.endpoint = f"{new_base}/v1/chat/completions"
                    self.tags_endpoint = f"{new_base}/v1/models"
                elif self.config.provider not in ("mlx-local", "huggingface-local"):
                    self.endpoint = f"{new_base}/api/chat"
                    self.tags_endpoint = f"{new_base}/api/tags"
            if self._client is not None and not self._client.is_closed:
                self._client.close()
            self._client = None
        except Exception:
            pass
        return ok

    _LAST_PROBE_OK_TS: float = 0.0
    _PROBE_FRESHNESS_SECONDS: float = 30.0

    def _quick_server_probe(self) -> bool:
        """Fast liveness probe before starting a streaming request.

        The normal client uses long read timeouts because generation can be
        slow. That is wrong for a preflight: if the server PID exists but the
        HTTP listener is wedged/non-responsive, we need to restart before the
        agent blocks a round for minutes.

        Cached across rounds: a successful probe is trusted for the next
        `_PROBE_FRESHNESS_SECONDS`. The probe was firing every round
        (one HTTP GET + ~100-300 ms wall clock per round) even though the
        server can't realistically wedge inside a 30-second window
        between two consecutive POSTs. Real failure mode (wedged server)
        is detected by the streaming layer's exception handling on the
        actual `POST /v1/chat/completions` — the probe is ONLY a
        cold-start safety net.
        """
        if self.config.provider in {"mlx-local", "huggingface-local"}:
            return True
        import time as _time
        now = _time.monotonic()
        if now - self._LAST_PROBE_OK_TS < self._PROBE_FRESHNESS_SECONDS:
            return True
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=1.0, read=2.0, write=1.0, pool=1.0)) as client:
                response = client.get(self.tags_endpoint)
                ok = response.status_code < 500
                if ok:
                    self._LAST_PROBE_OK_TS = now
                return ok
        except Exception:
            return False

    # ── DiffusionGemma backend (experimental) ───────────────────────
    #
    # Block-diffusion models denoise a whole block of tokens in parallel
    # instead of decoding token-by-token, so llama-server (and its HTTP
    # streaming API) can't drive them. Generation goes through the
    # one-shot `llama-diffusion-cli` runner from llama.cpp PR #24423
    # (built once by `bootstrap.ensure_diffusion_cli`). Consequences:
    #   - the model weights are (re)mapped per turn — first turn is slow,
    #     later turns are faster via the OS page cache;
    #   - output arrives in coarse chunks (denoised blocks), not tokens;
    #   - we apply the Gemma chat template ourselves (-p takes raw text).

    def _diffusion_choice(self):
        """Catalog entry for the configured model IF it's a diffusion arch."""
        try:
            from pathlib import Path as _Path

            from .models_catalog import by_filename
            c = by_filename(_Path(self.config.model or "").name)
        except Exception:
            return None
        if c is not None and str(getattr(c, "architecture", "")).startswith("diffusion"):
            return c
        return None

    @staticmethod
    def _format_diffusion_prompt(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Gemma chat template, applied by hand.

        llama-diffusion-cli's `-p` is a raw prompt — unlike llama-server
        there is no /v1/chat/completions layer to apply the GGUF's
        embedded Jinja template. Gemma's convention: system text is
        folded into the first user turn; roles are `user` / `model`.

        Tools are described in PLAIN JSON (not the Gemma special-token
        format), which is the one form DiffusionGemma reliably emits.

        CRITICAL: LocalCode's full agent system prompt (~10K chars with
        reasoning rules, notebook rules, skills, project instructions)
        OVERFLOWS DiffusionGemma's fixed canvas and makes it emit
        degenerate/empty output. Block-diffusion can't use most of that
        instruction text anyway. So for the diffusion path we DISCARD the
        verbose system prompt and substitute a concise one — keeping only
        the role, the working directory, and the (plain-JSON) tool block.
        """
        # Pull the working directory out of whatever big system prompt we
        # were handed, so path-relative tool calls still resolve.
        cwd_line = ""
        for m in messages:
            if m.get("role") == "system":
                for line in str(m.get("content") or "").splitlines():
                    if line.strip().lower().startswith("working directory:"):
                        cwd_line = line.strip()
                        break
        system_bits = [
            "You are LocalCode, a coding agent on the user's machine with "
            "filesystem access through the tools below. Be brief and act "
            "directly — when a task needs to read, write, or run something, "
            "call a tool. Reply with ONLY your final answer — do not write a "
            "'thought' preamble, reasoning, or narration."
        ]
        if cwd_line:
            system_bits.append(cwd_line)
        if tools:
            system_bits.append(LocalCodeRuntimeGateway._diffusion_tool_block(tools))
        pending_system = "\n\n".join(system_bits)
        parts: list[str] = []
        for m in messages:
            role = m.get("role")
            text = str(m.get("content") or "").strip()
            if role == "system" or not text:
                continue
            if role == "user" and pending_system:
                text = f"{pending_system}\n\n{text}"
                pending_system = ""
            gemma_role = "model" if role == "assistant" else "user"
            parts.append(f"<start_of_turn>{gemma_role}\n{text}<end_of_turn>\n")
        if pending_system:
            # System-only conversations (rare): emit it as a user turn so
            # the instructions reach the model at all.
            parts.append(f"<start_of_turn>user\n{pending_system}<end_of_turn>\n")
        parts.append("<start_of_turn>model\n")
        return "".join(parts)

    @staticmethod
    def _diffusion_tool_block(tools: list[dict[str, Any]]) -> str:
        """Plain-JSON tool instructions for DiffusionGemma.

        Verified that DiffusionGemma emits `{"tool":"NAME","args":{...}}`
        cleanly with this format, whereas the Gemma-4 special-token format
        (<|tool_call>…<tool_call|>) makes it collapse to empty output.
        """
        lines = []
        for t in tools:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            name = fn.get("name", "")
            if not name:
                continue
            desc = (fn.get("description", "") or "").strip().split("\n")[0][:120]
            params = (fn.get("parameters") or {}).get("properties", {}) or {}
            pnames = ", ".join(params.keys())
            lines.append(f'- {name}({pnames}): {desc}')
        return (
            "You can call tools. To call one, reply with ONLY a single JSON "
            'object on its own line, e.g. {"tool":"list_files","args":'
            '{"path":"."}} — every string value MUST be in double quotes '
            "(write \"path\":\".\", never path:.), and emit no other text. "
            "After the tool result comes back, continue. Available tools:\n"
            + "\n".join(lines)
        )

    @staticmethod
    def _parse_diffusion_tool_calls(text: str) -> tuple[list, str]:
        """Extract plain-JSON tool calls from DiffusionGemma output.

        Returns (tool_calls_in_ollama_format, remaining_visible_text).
        Finds `{"tool":"NAME","args":{...}}` objects (balanced braces),
        converts them to the {function:{name,arguments}} shape the agent
        loop expects, and removes them from the visible text.
        """
        import json as _json
        import re as _re
        calls = []
        spans: list[tuple[int, int]] = []
        # Locate every tool object opener in order. A single regex matches all
        # whitespace variants (`{"tool"`, `{ "tool"`, `{\n"tool"`) so an early
        # spaced-form call is never skipped in favor of a later compact one.
        opener = _re.compile(r'\{\s*"tool"')
        i = 0
        while True:
            m = opener.search(text, i)
            if not m:
                break
            j = m.start()
            # Balanced-brace scan from j.
            depth = 0
            k = j
            in_str = False
            esc = False
            while k < len(text):
                c = text[k]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
            blob = text[j:k]
            obj = None
            try:
                obj = _json.loads(blob)
            except Exception:
                # DiffusionGemma is non-deterministic and frequently emits
                # *almost*-valid JSON — a bare `.` value ({"path":.}), an
                # unquoted bareword, or a trailing comma. Repair the common
                # cases and retry rather than dropping a real tool call.
                try:
                    obj = _json.loads(
                        LocalCodeRuntimeGateway._repair_diffusion_json(blob)
                    )
                except Exception:
                    obj = None
            if isinstance(obj, dict):
                name = obj.get("tool")
                args = obj.get("args", {})
                if not isinstance(args, dict):
                    # The agent loop expects an arguments dict; a model that
                    # emits "args":"foo" or a list would otherwise crash it.
                    args = {}
                if name:
                    calls.append({
                        "id": f"diff_{len(calls)}",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
                    spans.append((j, k))
            i = k if k > j else j + 1
        # Remove parsed tool-call JSON from the visible text.
        if spans:
            out = []
            last = 0
            for s, e in spans:
                out.append(text[last:s])
                last = e
            out.append(text[last:])
            text = "".join(out)
        return calls, text.strip()

    @staticmethod
    def _repair_diffusion_json(blob: str) -> str:
        """Best-effort repair of almost-valid JSON from DiffusionGemma.

        The diffusion model often emits a recognizable tool-call object with
        one malformed value — most commonly a bare ``.`` ({"path":.}), an
        unquoted bareword ({"path":src/main.py}), or a trailing comma. We
        quote bare values (leaving real numbers and the literals
        true/false/null alone) and drop trailing commas, then let json.loads
        validate the result. If the repair doesn't yield valid JSON the
        caller's try/except discards it — this never fabricates a tool call.

        This is a single-pass char scanner, NOT a regex, because it MUST be
        string-aware: a blind regex would fire on a ``:`` or ``,}`` that
        lives inside a legitimate string value (e.g. a shell command
        ``"grep foo: bar"``) and corrupt it. Here, repairs only ever apply to
        value positions OUTSIDE of strings.
        """
        import re as _re

        out: list[str] = []
        n = len(blob)
        i = 0
        in_str = False
        esc = False
        while i < n:
            c = blob[i]
            if in_str:
                out.append(c)
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                i += 1
                continue
            if c == '"':
                in_str = True
                out.append(c)
                i += 1
                continue
            if c == ",":
                # Drop a trailing comma (comma followed only by ws then } or ]).
                j = i + 1
                while j < n and blob[j] in " \t\r\n":
                    j += 1
                if j < n and blob[j] in "}]":
                    i += 1  # skip the comma
                    continue
                out.append(c)
                i += 1
                continue
            if c == ":":
                out.append(c)
                i += 1
                j = i
                while j < n and blob[j] in " \t\r\n":
                    j += 1
                out.append(blob[i:j])
                i = j
                if i < n:
                    nxt = blob[i]
                    # Leave already-valid values (string/object/array/number)
                    # untouched; only quote a bare value.
                    if nxt not in '"{[' and not (nxt.isdigit() or nxt == "-"):
                        k = i
                        while k < n and blob[k] not in ",}]":
                            k += 1
                        # A bare (unquoted) value shouldn't contain quotes; a
                        # stray one means the model dropped a delimiter (it
                        # emitted `."` meaning `"."`). Strip surrounding quotes
                        # before re-quoting so {"path":."} → {"path":"."}.
                        token = blob[i:k].strip().strip('"')
                        if token in ("true", "false", "null") or _re.fullmatch(
                            r"-?\d+(\.\d+)?([eE][+-]?\d+)?", token
                        ):
                            out.append(blob[i:k])
                        else:
                            esc_tok = token.replace("\\", "\\\\").replace('"', '\\"')
                            out.append('"' + esc_tok + '"')
                        i = k
                continue
            out.append(c)
            i += 1
        return "".join(out)

    def _diffusion_cli_binary(self) -> str | None:
        p = (getattr(self.config, "diffusion_cli_binary", "") or "").strip()
        from pathlib import Path as _Path
        if p and _Path(p).is_file():
            return p
        from .bootstrap import diffusion_cli_path
        found = diffusion_cli_path()
        return str(found) if found is not None else None

    def _stream_diffusion_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        num_predict: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        import subprocess
        import time as _time
        from pathlib import Path as _Path

        binary = self._diffusion_cli_binary()
        if binary is None:
            # Setup normally builds the runner before the first turn; this
            # is the headless / edge path. Build now (one-time, minutes).
            yield {
                "type": "stage",
                "name": "diffusion_build",
                "message": "Building the diffusion runner (one-time, a few minutes)...",
            }
            from .bootstrap import ensure_diffusion_cli
            ok, result = ensure_diffusion_cli()
            if not ok:
                raise RuntimeErrorWithContext(
                    f"DiffusionGemma needs the llama-diffusion-cli runner and the build failed: {result}"
                )
            binary = result

        model_path = str(_Path(self.config.model or "").expanduser())
        # DiffusionGemma chokes on the Gemma-4 special-token tool format
        # (<|tool_call>…<tool_call|>) — it collapses to near-empty output.
        # It DOES reliably emit a plain JSON tool call when asked in plain
        # text, so for the diffusion path we inject tools as plain JSON and
        # parse that (see _diffusion_tool_block / _parse_diffusion_tool_call).
        prompt = self._format_diffusion_prompt(messages, tools=tools)

        # The agent loop passes num_predict = MAX_OUTPUT_TOKENS = -1 ("no
        # server-side cap, let it run"). For the HTTP server that means
        # unlimited, but llama-diffusion-cli's -n is a fixed CANVAS SIZE —
        # passing `-n -1` produces a degenerate/empty canvas ("returned no
        # usable response"). So treat any non-positive num_predict as "use
        # the default canvas", and cap at 512 (the canvas we validated).
        # NOTE: `num_predict or 512` does NOT work here — -1 is truthy.
        _n = int(num_predict) if (num_predict and int(num_predict) > 0) else 512
        _canvas = min(_n, 512)

        cmd = [
            binary,
            "-m", model_path,
            "-p", prompt,
            "-no-cnv",          # CRITICAL: the GGUF ships a chat template, so the
                                # CLI auto-enables conversation mode and would apply
                                # the template AGAIN on top of the one we built in
                                # _format_diffusion_prompt — double-templating that
                                # produced empty output. -no-cnv treats -p as the
                                # already-formatted raw prompt and runs one-shot.
            "-ngl", "99",
            "-n", str(_canvas),
        ]
        deadline = _time.monotonic() + max(60, int(self.config.request_timeout_seconds or 600))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        # Drain stderr CONCURRENTLY. Reading only stdout while stderr is a
        # PIPE deadlocks the moment the runner writes >64 KB of logs to
        # stderr (pipe buffer fills → child blocks on write → stdout goes
        # silent → we block on read forever). llama.cpp binaries are
        # chatty on stderr during model load, so this is the common case,
        # not the edge case.
        import threading as _threading
        stderr_tail: list[str] = []

        def _drain_stderr() -> None:
            try:
                assert proc.stderr is not None
                for line in proc.stderr:
                    stderr_tail.append(line)
                    if len(stderr_tail) > 50:
                        stderr_tail.pop(0)
            except Exception:
                pass

        _threading.Thread(target=_drain_stderr, daemon=True,
                          name="diffusion-stderr").start()
        # Block-diffusion isn't token-streamed — the runner denoises a whole
        # canvas and prints it in one go — so we COLLECT the full stdout, then
        # clean it once. Streaming 512-byte chunks straight through leaked the
        # prompt echo, the runner's "total time:/throughput:" stats lines (it
        # prints those to stdout, not stderr), and canvas padding after the
        # answer. We still read in a loop to honour the timeout.
        raw_parts: list[str] = []
        try:
            assert proc.stdout is not None
            while True:
                if _time.monotonic() > deadline:
                    proc.kill()
                    raise RuntimeErrorWithContext(
                        "diffusion generation timed out "
                        f"({self.config.request_timeout_seconds}s)"
                    )
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                raw_parts.append(chunk)
            rc = proc.wait(timeout=30)
            if rc != 0:
                err_tail = "".join(stderr_tail[-6:]).strip()
                raise RuntimeErrorWithContext(
                    f"llama-diffusion-cli exited with code {rc}:\n{err_tail}"
                )
        finally:
            if proc.poll() is None:
                proc.kill()

        _raw_joined = "".join(raw_parts)
        text = self._clean_diffusion_output(_raw_joined, prompt)

        # Diagnostic dump of the REAL live turn (prompt the model actually saw,
        # raw stdout, cleaned text, tool names offered). This is the only way
        # to see why a live turn differs from an isolated reconstruction.
        # Opt-in via LOCALCODE_DIFFUSION_DEBUG=1; overwrites each turn.
        import os as _os
        if _os.environ.get("LOCALCODE_DIFFUSION_DEBUG"):
            try:
                _dbg = _os.path.join(
                    _os.path.expanduser("~/.local/share/localcode"),
                    "diffusion_last.log",
                )
                _tool_names = [
                    (t.get("function", t) or {}).get("name", "?")
                    for t in (tools or [])
                ]
                with open(_dbg, "w", errors="replace") as _f:
                    _f.write(
                        f"=== PROMPT ({len(prompt)} chars, {len(_tool_names)} tools) ===\n"
                        f"tools: {_tool_names}\n"
                        f"num_predict={num_predict}\n"
                        f"{prompt}\n"
                        f"=== RAW STDOUT ({len(_raw_joined)} chars) ===\n"
                        f"{_raw_joined!r}\n"
                        f"=== CLEANED ===\n{text!r}\n"
                    )
            except Exception:
                pass

        # Tool calls first: DiffusionGemma emits plain JSON ({"tool":...,
        # "args":...}) when tools were offered in plain-text form. Parse and
        # surface those as a tool_calls event; strip them from the visible
        # content so the user doesn't see raw JSON.
        tool_calls = []
        if tools:
            tool_calls, text = self._parse_diffusion_tool_calls(text)

        if not text.strip() and not tool_calls:
            text = (
                "⚠ DiffusionGemma returned no usable response this turn. "
                "It's an experimental diffusion model — for heavier coding, "
                "a Gemma 26B-A4B quant (via /model) is more reliable."
            )
        # Emit in modest chunks so the chat log renders progressively.
        for i in range(0, len(text), 160):
            yield {"type": "content", "content": text[i:i + 160]}

        if tool_calls:
            yield {"type": "tool_calls", "tool_calls": tool_calls}

        # Terminal event — the agent/streaming consumer needs this to record
        # token counts and finish the round. The HTTP path emits it from the
        # server's usage; diffusion-cli gives none, so estimate from chars
        # (chars/4). Without this the UI showed "14.4s" with NO token count.
        _ct = max(1, len(text) // 4)
        yield {
            "type": "stream_done",
            "finish_reason": "stop",
            "content_chars": len(text),
            "completion_tokens": _ct,
            "total_tokens": _ct,
            "usage_estimated": True,
        }

    @staticmethod
    def _clean_diffusion_output(raw: str, prompt: str) -> str:
        """Turn raw llama-diffusion-cli stdout into a clean assistant reply.

        DiffusionGemma's output is non-deterministic in SHAPE — it may wrap
        deliberation in `<|channel>thought … <channel|>` then give the answer,
        or spend its whole canvas on reasoning, or emit `<end_of_turn>` early.
        Each cleaning step is therefore applied ONLY IF it leaves real text,
        so we never blank out a turn that actually contained content.
        """
        import re as _re

        def _strip_tokens(s: str) -> str:
            for tok in ("<|channel>", "<channel|>", "<tool_call|>", "<|tool_call>",
                        "<start_of_turn>model", "<start_of_turn>", "<end_of_turn>"):
                s = s.replace(tok, "")
            return s

        text = raw
        if text.startswith(prompt):
            text = text[len(prompt):]
        # Always drop the runner's stdout stats/progress lines.
        text = _re.sub(
            r"(?m)^\s*(total time:|throughput:|time per step:|diffusion step:|diffusion_).*$",
            "", text,
        )
        # Prefer the reply BEFORE the first <end_of_turn> (drops canvas
        # padding) — but only if that leaves content; the model sometimes
        # emits <end_of_turn> first, which would otherwise empty the turn.
        head = text.split("<end_of_turn>", 1)[0]
        if head.strip():
            text = head
        # Remove the channel/thought reasoning block (keeping the answer after
        # <channel|>) — but only if the answer is non-empty; if the model
        # spent its whole canvas reasoning, keep that rather than blank out.
        deblocked = _re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=_re.DOTALL)
        if deblocked.strip():
            text = deblocked
        # BF16 DiffusionGemma emits reasoning WITHOUT channel markers — a bare
        # `thought` line, the deliberation, then the answer, with no closing
        # marker. Observed shape: the reasoning's own sentences are separated
        # by ". " (period + SPACE), but the join from the last reasoning
        # sentence to the answer has NO space ("...wait for a task.Hello!").
        # So: strip the `thought` marker, then split on a sentence-end
        # immediately followed by a capital (no space) and keep the final
        # segment as the answer. Each step only applies if it leaves real
        # text, so a reasoning-only turn is never blanked.
        m = _re.match(r"(?is)^\s*thought\b[ \t]*\n?(.*)$", text)
        if m and m.group(1).strip():
            body = m.group(1)
            joins = list(_re.finditer(r"[.!?](?=[A-Z])", body))
            if joins:
                answer = body[joins[-1].end():].strip()
                if answer:
                    body = answer
            if body.strip():
                text = body
        return _strip_tokens(text).strip()

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
        # Diffusion models (architecture="diffusion_gemma") cannot be
        # served by llama-server — they generate via the one-shot
        # llama-diffusion-cli runner. Dispatch on the catalog's
        # architecture field BEFORE any HTTP machinery runs.
        if self._diffusion_choice() is not None:
            yield from self._stream_diffusion_events(
                messages, tools=tools, num_predict=num_predict
            )
            return
        if self.config.provider == "mlx-local":
            effective_messages = messages
            if tools:
                effective_messages = self._inject_tools_into_messages(messages, tools)
            # Stream with thinking/content separation
            raw_parts: list[str] = []
            buffer = ""
            in_thinking = False
            self._last_thinking = ""
            for chunk in self._mlx_stream_generate(effective_messages):
                buffer += chunk
                # Buffer partial special tokens
                if "<|" in buffer and "|>" not in buffer:
                    continue
                if "<tool_call" in buffer and "tool_call|>" not in buffer:
                    continue
                if "<|channel>" in buffer and "<channel|>" not in buffer:
                    # Inside thinking block — accumulate but don't yield as content
                    if "<|channel>thought" in buffer:
                        in_thinking = True
                    continue
                raw_parts.append(buffer)
                # Check if thinking block ended in this chunk
                if in_thinking and "<channel|>" in buffer:
                    # Extract thinking text
                    parts = buffer.split("<channel|>", 1)
                    thinking_part = parts[0].replace("<|channel>thought", "").strip()
                    if thinking_part:
                        yield {"type": "thinking", "content": thinking_part}
                    content_after = parts[1] if len(parts) > 1 else ""
                    content_after = self._clean_mlx_output(content_after)
                    in_thinking = False
                    buffer = ""
                    if content_after.strip():
                        yield {"type": "content", "content": content_after}
                    continue
                if not in_thinking:
                    cleaned = self._clean_mlx_output(buffer)
                    buffer = ""
                    if cleaned.strip():
                        yield {"type": "content", "content": cleaned}
                else:
                    buffer = ""
            # Flush
            if buffer:
                raw_parts.append(buffer)
                if in_thinking:
                    yield {"type": "thinking", "content": buffer}
                else:
                    cleaned = self._clean_mlx_output(buffer)
                    if cleaned.strip():
                        yield {"type": "content", "content": cleaned}
            # Parse tool calls from raw
            raw_full = "".join(raw_parts)
            if tools:
                parsed = parse_tool_calls(raw_full)
                if parsed.has_tools:
                    yield {"type": "tool_calls", "tool_calls": parsed.to_ollama_format()}
            return
        if self.config.provider == "huggingface-local":
            content = self._hf_generate(messages)
            for chunk in self._chunk_text(content, 180):
                if chunk:
                    yield {"type": "content", "content": chunk}
            return

        payload = self._payload(messages, stream=True, tools=tools, think=think, num_ctx=num_ctx, num_predict=num_predict)
        last_error: Exception | None = None
        # Cap server restarts across the WHOLE stream, not per-attempt.
        # Bringing up a ~10 GB server costs seconds + a GPU-memory spike;
        # without a cap, a server that dies on every reconnect turned the
        # retry loop into a restart storm (each of N attempts fired its
        # own restart). One restart per stream is enough to recover from
        # a pressure-kill/OOM; if that doesn't bring it back, surface the
        # error instead of thrashing.
        _MAX_STREAM_RESTARTS = 1
        _restarts_done = 0
        for attempt in range(max(1, self.config.max_retries + 1)):
            try:
                if attempt == 0 and _restarts_done < _MAX_STREAM_RESTARTS and not self._quick_server_probe():
                    _restarts_done += 1
                    yield {
                        "type": "stage",
                        "name": "server_reconnect",
                        "message": "Model server stopped responding — restarting it now and continuing.",
                    }
                    if not self._restart_server():
                        raise RuntimeErrorWithContext("model server is not responding and restart failed")
                    payload = self._payload(
                        messages,
                        stream=True,
                        tools=tools,
                        think=think,
                        num_ctx=num_ctx,
                        num_predict=num_predict,
                    )
                # Accumulate tool call deltas (OpenAI streams them incrementally)
                pending_tools: dict[int, dict] = {}  # index -> {id, name, arguments}
                # Per-index throttle: only emit tool_preview when args grow by
                # PREVIEW_STEP chars since the last preview. Without this, the
                # UI sees an event per SSE chunk for tools whose `content` arg
                # is a 5K-line file — fine functionally, noisy in logs.
                preview_marks: dict[int, int] = {}  # index -> last-emitted args length
                PREVIEW_STEP = 256
                tool_args_oversize = False
                oversize_tool_name = ""
                oversize_args_chars = 0
                oversize_args_snippet = ""
                oversize_reason = ""
                # Stateful thinking detection for llama.cpp
                # Gemma 4 thinking tokens decode as <unused25> through llama.cpp
                in_thinking = False  # True while inside thinking block
                # Diagnostic capture for the round_end event. Set as the
                # stream progresses; flushed via a stream_done event at
                # the end so agent/loop.py can attribute outcome correctly
                # (clean EOS vs length-cap vs tool_calls vs cut off).
                last_finish_reason: str = ""
                content_chars_total = 0
                reasoning_chars_total = 0
                raw_content_tail = ""  # last 500 chars of `content` deltas
                # Timing markers — captured here, surfaced on stream_done.
                # `_stream_started_at` lets us compute total stream wall.
                # `_first_token_at` is the moment the first usable token
                # (content, reasoning, or tool_call delta) arrives — this
                # is THE TTFT we care about (prompt-eval finished, decode
                # started). `_stream_ended_at` is set just before yielding
                # the stream_done event. Together they let agent/loop
                # split round duration into ttft_ms vs decode_ms cleanly.
                import time as _time_mod_runtime
                _stream_started_at = _time_mod_runtime.monotonic()
                _first_token_at: float | None = None
                # llama-server emits a final SSE chunk with `usage:
                # {prompt_tokens, completion_tokens, total_tokens}`.
                # Capture them so the chat-screen summary can show real
                # input/output token counts instead of estimating from
                # message char length.
                prompt_tokens_seen: int = 0
                completion_tokens_seen: int = 0
                usage_estimated = False
                with self.client.stream("POST", self.endpoint, json=payload) as response:
                    if response.status_code >= 400:
                        try:
                            response.read()
                            body = response.text
                        except Exception:
                            body = ""
                        snippet = body.strip()[:500] if body else ""
                        # Transient 5xx (502/503/504): the server is up
                        # but the inference path is briefly unavailable
                        # (warmup race after a model swap, slot busy
                        # finishing a prior request, kv-cache rebuild).
                        # Retry with backoff INSIDE the gateway so the
                        # user never sees the transient — they just
                        # experience a few extra hundred ms of latency.
                        # 4xx errors are bugs in our request shape and
                        # bubble out immediately.
                        is_transient = 500 <= response.status_code < 600
                        if is_transient and attempt < max(1, self.config.max_retries):
                            import time as _time
                            # Backoff capped at 3 s per attempt — the
                            # earlier exponential 1/2/4/8 s schedule
                            # accumulated up to 15 s of user-visible
                            # latency on bad turns. Qwen cold-swap warmup
                            # still fits in 2–3 s total which is well
                            # within the new cap × max_retries budget.
                            _time.sleep(min(3.0, 1.0 * (2 ** attempt)))
                            raise RuntimeErrorWithContext(
                                f"HTTP {response.status_code} from {self.endpoint}"
                                + (f" — {snippet}" if snippet else "")
                            )
                        raise RuntimeErrorWithContext(
                            f"HTTP {response.status_code} from {self.endpoint}"
                            + (f" — {snippet}" if snippet else "")
                        )
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if self.config.provider == "llama_cpp" and line.startswith("data: "):
                            line = line[6:]
                            if line == "[DONE]":
                                break
                        data = json.loads(line)
                        if "error" in data:
                            raise RuntimeErrorWithContext(_error_message(data["error"]))
                        # Capture finish_reason if the chunk carries one. Most
                        # chunks have `null`; only the terminating chunk has
                        # the real value (`stop` / `length` / `tool_calls` /
                        # `content_filter`). Read defensively — Ollama and
                        # llama.cpp both put it under choices[0] in OpenAI
                        # mode but the field name varies.
                        try:
                            _choices = data.get("choices") or []
                            if _choices:
                                _fr = _choices[0].get("finish_reason")
                                if _fr:
                                    last_finish_reason = str(_fr)
                            # Usage info — present on the final chunk
                            # for OpenAI-compatible servers including
                            # llama-server. Read defensively; not all
                            # backends populate it.
                            _usage = data.get("usage") or {}
                            if _usage:
                                pt = _usage.get("prompt_tokens")
                                ct = _usage.get("completion_tokens")
                                if isinstance(pt, int) and pt > 0:
                                    prompt_tokens_seen = pt
                                if isinstance(ct, int) and ct > 0:
                                    completion_tokens_seen = ct
                        except Exception:
                            pass
                        message = self._extract_message(data)
                        # Accumulate diagnostic counters BEFORE we yield events
                        # downstream — gives us per-stream stats (raw chars
                        # in, reasoning vs content split) for round_end.
                        _delta_content = message.get("content", "") or ""
                        if _delta_content:
                            content_chars_total += len(_delta_content)
                            raw_content_tail = (
                                raw_content_tail + _delta_content
                            )[-500:]
                        _delta_reasoning = message.get("thinking", "") or ""
                        if _delta_reasoning:
                            reasoning_chars_total += len(_delta_reasoning)
                        # Record first-token timestamp on the first
                        # arriving delta (content, reasoning, or any
                        # tool-call deltas — caught a few lines below).
                        # That's the moment prompt-eval finished and
                        # decode began — our TTFT marker.
                        if _first_token_at is None and (
                            _delta_content or _delta_reasoning
                            or message.get("tool_calls")
                        ):
                            _first_token_at = _time_mod_runtime.monotonic()
                        # Accumulate tool call deltas
                        tool_calls = message.get("tool_calls") or []
                        for tc in tool_calls:
                            idx = tc.get("index", 0)
                            fn = tc.get("function", {})
                            if idx not in pending_tools:
                                pending_tools[idx] = {
                                    "id": tc.get("id", f"call_{idx}"),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if fn.get("name"):
                                # Strip whitespace at the SOURCE — IQ2-quantized
                                # models routinely emit names with a leading or
                                # trailing space ("list_files "), which then
                                # propagates through every downstream dispatcher
                                # and causes opaque KeyError reports. Sanitizing
                                # here means the tool name is correct everywhere
                                # in the pipeline, not just in the dispatcher
                                # we happen to remember to patch.
                                prev_name = pending_tools[idx]["function"]["name"]
                                pending_tools[idx]["function"]["name"] = fn["name"].strip()
                                # Mid-stream UI signal: the model has committed
                                # to a tool name. Without this, large `content`
                                # args (5K-line file writes) take 1-3 minutes
                                # to stream and the UI shows "thinking…" the
                                # whole time. Emit `tool_preview` so the chat
                                # screen can render `▪ Write` as a placeholder
                                # immediately, then update args as they grow.
                                # The agent loop's later `out.log_tool(...)`
                                # call is unchanged — it still fires the real
                                # `tool_start` once we're ready to execute.
                                if not prev_name and pending_tools[idx]["function"]["name"]:
                                    yield {
                                        "type": "tool_preview",
                                        "index": idx,
                                        "name": pending_tools[idx]["function"]["name"],
                                        "args_chars": 0,
                                        "args_snippet": "",
                                    }
                                    preview_marks[idx] = 0
                            if fn.get("arguments"):
                                pending_tools[idx]["function"]["arguments"] += fn["arguments"]
                                # Throttled progress update so the UI can show
                                # a growth indicator while the args stream in.
                                args_len = len(pending_tools[idx]["function"]["arguments"])
                                if args_len - preview_marks.get(idx, 0) >= PREVIEW_STEP:
                                    preview_marks[idx] = args_len
                                    name = pending_tools[idx]["function"]["name"]
                                    if name:
                                        # Send the FULL accumulated args so the
                                        # TUI can decode and live-stream the
                                        # `content` / `new_string` field for
                                        # write_file / edit_file / append_file
                                        # / multi_edit. Earlier we trimmed to
                                        # the leading 256 chars (enough for
                                        # `"path": "x"` regex extraction); the
                                        # cost of sending the full args is
                                        # negligible vs decode wall-clock and
                                        # the UX win is huge — the user sees
                                        # the file appearing line by line
                                        # instead of staring at "12 KB
                                        # streaming…" for 5 minutes. The TUI
                                        # tracks its own decoded position so
                                        # only NEW bytes get rendered each tick.
                                        accum = pending_tools[idx]["function"]["arguments"]
                                        yield {
                                            "type": "tool_preview",
                                            "index": idx,
                                            "name": name,
                                            "args_chars": args_len,
                                            "args_snippet": accum,
                                        }
                                _elapsed_arg_s = (
                                    _time_mod_runtime.monotonic() - _stream_started_at
                                )
                                _limit, _limit_reason = _tool_arg_stream_guard(
                                    pending_tools[idx]["function"].get("name", ""),
                                    pending_tools[idx]["function"].get("arguments", ""),
                                    elapsed_s=_elapsed_arg_s,
                                    recovery_mode=recovery_mode,
                                    stream_policy=stream_policy,
                                )
                                if _limit:
                                    tool_args_oversize = True
                                    last_finish_reason = "tool_args_limit"
                                    oversize_args_chars = args_len
                                    oversize_tool_name = pending_tools[idx]["function"].get("name", "")
                                    oversize_args_snippet = pending_tools[idx]["function"].get("arguments", "")[:500]
                                    oversize_reason = _limit_reason
                                    break
                        if tool_args_oversize:
                            break
                        # Check for explicit reasoning_content first.
                        # Reasoning models (e.g. North-Mini-Code) reason
                        # UNCONDITIONALLY — the server returns it in
                        # reasoning_content even when we asked for no thinking.
                        # Honor `/thinking off` by displaying it only when
                        # think is on. We must NOT `continue` here: a single
                        # delta can carry BOTH reasoning_content and the first
                        # fragment of visible content, and skipping would drop
                        # that answer text. Falling through processes the
                        # co-arriving content below (and is a no-op when the
                        # reasoning delta has no content).
                        thinking = message.get("thinking")
                        if thinking and think:
                            yield {"type": "thinking", "content": thinking}
                        content = message.get("content", "")
                        if not content:
                            continue
                        # Stateful detection of thinking-channel markers in content.
                        # Markers come from the active model family's adapter — for
                        # Gemma 4 both open and close are literally `<unused25>`
                        # (same token); Qwen uses `<think>` / `</think>`. Defaults
                        # to Gemma when family is unset so prior behaviour holds.
                        _family = infer_family_from_profile(
                            getattr(self.config, "profile", "") or ""
                        )
                        _adapter = get_adapter(_family)
                        _open = _adapter.thinking_open
                        _close = _adapter.thinking_close
                        if self.config.provider == "llama_cpp" and think:
                            if not in_thinking and _open in content:
                                # Thinking block starts
                                in_thinking = True
                                after = content.split(_open, 1)[1]
                                cleaned = _strip_thinking_tokens(after, _family)
                                if cleaned.strip():
                                    yield {"type": "thinking", "content": cleaned}
                                continue
                            if in_thinking:
                                if _close in content:
                                    # Thinking block ends
                                    before = content.split(_close, 1)[0]
                                    cleaned = _strip_thinking_tokens(before, _family)
                                    if cleaned.strip():
                                        yield {"type": "thinking", "content": cleaned}
                                    in_thinking = False
                                    after = content.split(_close, 1)[1]
                                    cleaned_after = _strip_thinking_tokens(after, _family)
                                    if cleaned_after.strip():
                                        yield {"type": "content", "content": cleaned_after}
                                else:
                                    # Still inside thinking block
                                    cleaned = _strip_thinking_tokens(content, _family)
                                    if cleaned.strip():
                                        yield {"type": "thinking", "content": cleaned}
                                continue
                        cleaned = _strip_thinking_tokens(content)
                        if cleaned:
                            yield {"type": "content", "content": cleaned}
                    # If stream ends while still in thinking, that's fine —
                    # thinking_done will be emitted by the agent loop
                        # Ollama signals done
                        if data.get("done"):
                            final_msg = data.get("message", {})
                            final_tools = final_msg.get("tool_calls")
                            if final_tools:
                                for ft in final_tools:
                                    idx = ft.get("index", len(pending_tools))
                                    pending_tools[idx] = ft
                # Diagnostic event for the agent loop — outcome of THIS
                # stream (per-round, not per-turn). Lets round_end log
                # WHY the model stopped (clean EOS / length cap / tool
                # commit / cut off) and a sample of the raw tail when a
                # round terminates with content but zero tools (the
                # "said let me check, did nothing" failure mode). Yielded
                # BEFORE tool_calls so the agent sees the diagnostic
                # alongside the action.
                _stream_ended_at = _time_mod_runtime.monotonic()
                # ttft_ms = time from request-start to first token.
                # decode_ms = time from first token to stream end.
                # If first_token_at is None (e.g. server returned no
                # content at all), ttft_ms == total stream wall and
                # decode_ms == 0 — that's the right framing for an
                # empty round.
                _ttft_ms = int(
                    ((_first_token_at or _stream_ended_at) - _stream_started_at) * 1000
                )
                _decode_ms = int(
                    (_stream_ended_at - (_first_token_at or _stream_ended_at)) * 1000
                )
                if not prompt_tokens_seen:
                    prompt_tokens_seen = _estimate_prompt_token_count(payload)
                    usage_estimated = True
                if not completion_tokens_seen:
                    tool_arg_chars = 0
                    for _tc in pending_tools.values():
                        try:
                            _fn = _tc.get("function") or {}
                            tool_arg_chars += len(str(_fn.get("name") or ""))
                            tool_arg_chars += len(str(_fn.get("arguments") or ""))
                        except Exception:
                            pass
                    completion_tokens_seen = _estimate_token_count(
                        "x" * (content_chars_total + reasoning_chars_total + tool_arg_chars)
                    )
                    usage_estimated = True
                yield {
                    "type": "stream_done",
                    "finish_reason": last_finish_reason,
                    "content_chars": content_chars_total,
                    "reasoning_chars": reasoning_chars_total,
                    "raw_tail": raw_content_tail,
                    "pending_tool_count": len(pending_tools),
                    "prompt_tokens": prompt_tokens_seen,
                    "completion_tokens": completion_tokens_seen,
                    "total_tokens": (
                        (prompt_tokens_seen or 0) + (completion_tokens_seen or 0)
                    ) if (prompt_tokens_seen or completion_tokens_seen) else 0,
                    "usage_estimated": usage_estimated,
                    "tool_args_limited": tool_args_oversize,
                    "limited_tool_name": oversize_tool_name,
                    "limited_args_chars": oversize_args_chars,
                    "limited_args_snippet": oversize_args_snippet,
                    "limited_reason": oversize_reason,
                    "ttft_ms": _ttft_ms,
                    "decode_ms": _decode_ms,
                }
                # Yield accumulated tool calls at the end
                if pending_tools:
                    yield {"type": "tool_calls", "tool_calls": list(pending_tools.values())}
                return
            except Exception as exc:
                last_error = exc
                # Memory-pressure recovery. The pressure monitor SIGTERMs
                # llama-server when macOS reports CRITICAL memory pressure
                # (kern.memorystatus_vm_pressure_level >= 4) for ~2s. The
                # next request lands on a dead socket → "connection
                # refused". Without recovery the user sees E3102, the
                # session is effectively dead, they have to relaunch.
                # With recovery: detect the pattern, wait for memory to
                # settle, restart the server, retry the stream once.
                err_str = _error_message(exc).lower()
                is_conn_err = "connect" in err_str or "refused" in err_str
                # Auto-recovery on connection-refused: try ONE server
                # restart before surfacing the error to the user. Any
                # time the server is dead (pressure-kill, OOM, crash,
                # external kill) the next request lands on a closed
                # socket — we shouldn't need a pressure marker to
                # trigger recovery; the connection error itself is the
                # proof that action is needed.
                #
                # Previous version required `_pressure_kill_recent()`
                # which meant restarts only fired when the pressure
                # monitor had written its marker. A natural OOM or
                # external kill (jetsam, Activity Monitor force-quit)
                # writes no marker → recovery didn't fire → user saw
                # E3102 and the session was effectively dead (evidence:
                # e2e run 2026-04-23T23:54, long_coding_session failed
                # 25/25 turns at 0.5 s each on a dead server).
                _pressure_related = _pressure_kill_recent()
                if is_conn_err and attempt < self.config.max_retries and _restarts_done < _MAX_STREAM_RESTARTS:
                    _restarts_done += 1
                    recovery_label = (
                        "memory_pressure_recovery"
                        if _pressure_related
                        else "server_reconnect"
                    )
                    msg = (
                        "macOS flagged memory pressure and your local model "
                        "server was paused to protect the system. Restarting "
                        "now and continuing your request — no action needed."
                        if _pressure_related
                        else "Lost connection to the model server — "
                             "restarting it now and retrying your request."
                    )
                    yield {"type": "stage", "name": recovery_label, "message": msg}
                    # Cooldown: let macOS reclaim memory and the kernel's
                    # pressure level drop back to NORMAL before we try to
                    # bring up a fresh ~10 GB server. Without the wait we
                    # often re-trigger the kill within seconds.
                    import time as _time
                    _time.sleep(5.0)
                    try:
                        self._restart_server()
                    except Exception:
                        # If restart itself fails, fall through to the
                        # normal retry/error path so the user still sees
                        # an error rather than silent hang.
                        pass
                    _clear_pressure_kill_marker()
                    # Reset stream state for retry. pending_tools is the
                    # only mutable across the retry — clear it so the
                    # restart starts clean.
                    pending_tools = {}
                    in_thinking = False
                    # Allow one more attempt past the normal cap for
                    # this specific recovery path — the user shouldn't
                    # eat a hard failure for a known-recoverable cause.
                    continue
                if attempt >= self.config.max_retries:
                    break
        raise RuntimeErrorWithContext(str(last_error) if last_error else "runtime stream failed")

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        num_predict: int | None = None,
    ) -> Iterator[str]:
        for event in self.stream_chat_events(messages, tools, num_predict=num_predict):
            if event["type"] == "content":
                yield str(event["content"])

    def _payload(
        self,
        messages: list[dict[str, Any]],
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
        think: bool = True,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ) -> dict[str, Any]:
        opts = self._options(num_ctx_override=num_ctx, num_predict_override=num_predict)
        if self.config.provider == "llama_cpp":
            extra: dict[str, Any] = {
                "n_gpu_layers": self.config.llama_cpp_gpu_layers,
                "n_threads": self.config.llama_cpp_threads,
                "n_batch": self.config.llama_cpp_batch_size,
            }
            # Speed: expert offloading — keep attention on GPU, experts on CPU
            if self.config.llama_cpp_expert_offload:
                extra["ot"] = "exps=CPU"
            # Speed: KV cache compression (asymmetric K/V)
            if self.config.kv_cache_type_k and self.config.kv_cache_type_k != "f16":
                extra["cache_type_k"] = self.config.kv_cache_type_k
            if self.config.kv_cache_type_v and self.config.kv_cache_type_v != "f16":
                extra["cache_type_v"] = self.config.kv_cache_type_v
            payload: dict[str, Any] = {
                "model": self.config.model,
                "stream": stream,
                "messages": messages,
                # v0.2.12 baseline: forward only `temperature` and let
                # llama-server's defaults handle every other sampler.
                #
                # Why we walked back the "forward the full sampler
                # stack" change: with `repeat_penalty=1.10` /
                # `repeat_last_n=64` / DRY all forwarded, the model
                # produced 2-3 paraphrased answers in one round on
                # simple questions ("hows weather in london" →
                # three different formattings of the same data, 2 m 30 s,
                # observed 2026-04-29). v0.2.12 only forwarded
                # `temperature`; it did not show this loop. So our
                # carefully-tuned sampler config was *causing* the
                # rephrase rather than preventing it — most likely
                # `repeat_penalty` penalising `<|im_end|>` after the
                # first complete answer. Server defaults win for now;
                # any future re-introduction must come with a single-
                # variable A/B and a kept measurement, not a bundle.
                "temperature": opts["temperature"],
                "chat_template_kwargs": {"enable_thinking": think},
            }
            # Single-knob anti-rephrase: `repeat_penalty=1.05` only.
            # Added 2026-04-29 after stripping the full sampler stack
            # killed multi-round looping but left intra-decode rephrase
            # intact ("hows weather in tokyo" → 3 paraphrased sentences
            # in one 80-token decode). Why ONLY this one and at 1.05
            # (not the previous 1.10): the loop with the full stack was
            # penalty=1.10 + top_p=0.8 + top_k=20 + DRY, which trapped
            # the model into rephrasing because EOS was penalised AND
            # the candidate pool was tiny. Server defaults give us
            # top_p=0.95 / top_k=40 — plenty of escape room — so a
            # mild 1.05 penalty discourages exact phrase repeats
            # without starving EOS. If this re-introduces a loop,
            # revert this hunk and accept occasional rephrase as a
            # model property of Qwen3.6 IQ2_M.
            payload["repeat_penalty"] = 1.05
            if "num_predict" in opts:
                _np = opts["num_predict"]
                # llama-server treats max_tokens=-1 as "use default", which on
                # some builds caps at ~2048 and silently truncates write_file
                # args mid-JSON for large file writes. Only forward positive
                # caps; for -1 ("unlimited") omit the field so the server
                # uses its true unlimited default and the model can finish
                # large tool-call argument streams.
                if isinstance(_np, int) and _np > 0:
                    payload["max_tokens"] = _np
            if tools:
                payload["tools"] = tools
            # One-shot diagnostic: log the sampler subset of the FIRST
            # outbound chat-completions payload so we can verify DRY is
            # actually being forwarded. Set LOCALCODE_DEBUG_SAMPLERS=1
            # to enable. Self-disables after one log to keep noise down.
            import os, logging
            if os.environ.get("LOCALCODE_DEBUG_SAMPLERS") == "1" and not getattr(self, "_logged_samplers", False):
                sampler_keys = (
                    "temperature", "top_p", "top_k", "min_p",
                    "repeat_penalty", "repeat_last_n",
                    "dry_multiplier", "dry_base", "dry_allowed_length",
                    "dry_penalty_last_n",
                    "presence_penalty", "frequency_penalty",
                )
                snapshot = {k: payload.get(k, "<absent>") for k in sampler_keys}
                logging.getLogger("localcode.samplers").info(
                    "outbound_samplers tools=%s %s",
                    bool(tools), snapshot,
                )
                self._logged_samplers = True
            return payload

        payload = {
            "model": self.config.model,
            "stream": stream,
            "messages": messages,
            "options": opts,
            "keep_alive": "30m",
        }
        if not think:
            payload["think"] = False
        if tools:
            payload["tools"] = tools
        # Timeout: 120s for streaming, 300s for non-streaming
        return payload

    def _extract_message(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.config.provider == "llama_cpp":
            choices = data.get("choices", [])
            if not choices:
                return {}
            delta = choices[0].get("delta") or choices[0].get("message") or {}
            result: dict[str, Any] = {
                "content": delta.get("content", ""),
                "tool_calls": delta.get("tool_calls") or [],
            }
            # Gemma 4 thinking: llama.cpp returns reasoning in reasoning_content
            thinking = delta.get("reasoning_content", "")
            if thinking:
                result["thinking"] = thinking
            return result
        return data.get("message", {})

    # ── Local backends ───────────────────────────────────────────────────

    def _get_hf_backend(self) -> Any:
        if self._hf_backend is not None:
            return self._hf_backend
        model_id = self.config.huggingface_model_id or self.config.model
        if not model_id:
            raise RuntimeErrorWithContext("No Hugging Face model configured.")
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError:
            try:
                import torch
                from transformers import AutoModelForCausalLM as AutoModelForMultimodalLM, AutoTokenizer as AutoProcessor
            except Exception as exc:
                raise RuntimeErrorWithContext("transformers + torch required for HF backend.") from exc
        torch_dtype = None
        if self.config.huggingface_dtype not in {"", "auto"}:
            torch_dtype = getattr(torch, self.config.huggingface_dtype, None)
        model_kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        else:
            model_kwargs["dtype"] = "auto"
        device_map = self.config.huggingface_device if self.config.huggingface_device else "auto"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map=device_map, **model_kwargs)
        self._hf_backend = (model, processor)
        return self._hf_backend

    def _get_mlx_backend(self) -> Any:
        if self._mlx_backend is not None:
            return self._mlx_backend
        model_id = self.config.mlx_model_id or self.config.model
        if not model_id:
            raise RuntimeErrorWithContext("No MLX model configured.")
        try:
            from mlx_lm import load, generate
        except Exception as exc:
            raise RuntimeErrorWithContext("mlx-lm required for MLX backend.") from exc
        model, tokenizer = load(model_id)
        self._mlx_backend = (model, tokenizer, generate)
        return self._mlx_backend

    def _mlx_generate(self, messages: list[dict[str, Any]]) -> str:
        model, tokenizer, generate = self._get_mlx_backend()
        prompt = self._messages_to_prompt(messages)
        try:
            if getattr(tokenizer, "chat_template", None) is not None:
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            pass
        return str(generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max(512, min(8192, self.config.max_context_chars // 4)),
            # temperature handled by mlx defaults
            verbose=False,
        )).strip()

    # Gemma 4 special tokens to strip from MLX output
    _GEMMA4_TOKENS = re.compile(
        r'<\|tool_call\>.*?<tool_call\|>'
        r'|<\|"\|>'
        r'|<\|[^>]*\|>'
        r'|<turn\|>'
        r'|<\|turn\>',
        re.DOTALL
    )

    def _clean_mlx_output(self, text: str) -> str:
        """Strip Gemma 4 special tokens, split thinking from content."""
        # Split on thinking channel markers
        if "<|channel>thought" in text:
            # Everything before <|channel>thought = content prefix (usually empty)
            # Everything between <|channel>thought and <channel|> = thinking
            # Everything after <channel|> = actual content
            parts = re.split(r'<\|channel\>thought', text, maxsplit=1)
            before = parts[0]
            if len(parts) > 1:
                rest = parts[1]
                channel_end = rest.find("<channel|>")
                if channel_end >= 0:
                    thinking = rest[:channel_end]
                    content = rest[channel_end + len("<channel|>"):]
                    # Store thinking for the indicator
                    self._last_thinking = thinking
                    text = before + content
                else:
                    # Haven't seen end of thinking yet — buffer it
                    self._last_thinking = rest
                    text = before
        # Clean remaining tokens
        text = self._GEMMA4_TOKENS.sub('', text)
        # Also strip <channel|> and <|channel> fragments
        text = text.replace("<channel|>", "").replace("<|channel>", "")
        return text

    def _mlx_stream_generate(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        """True streaming generation for MLX."""
        model, tokenizer, _ = self._get_mlx_backend()
        prompt = self._messages_to_prompt(messages)
        try:
            if getattr(tokenizer, "chat_template", None) is not None:
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            pass
        try:
            from mlx_lm import stream_generate
            for response in stream_generate(
                model, tokenizer, prompt=prompt,
                max_tokens=max(512, min(8192, self.config.max_context_chars // 4)),
            ):
                text = response.text if hasattr(response, 'text') else (response.get("text", "") if isinstance(response, dict) else str(response))
                if text:
                    cleaned = self._clean_mlx_output(text)
                    if cleaned:
                        yield cleaned
        except (ImportError, AttributeError):
            content = self._mlx_generate(messages)
            cleaned = self._clean_mlx_output(content)
            for chunk in self._chunk_text(cleaned, 80):
                yield chunk

    def _hf_generate(self, messages: list[dict[str, Any]]) -> str:
        model, processor = self._get_hf_backend()
        try:
            # Use Gemma 4's apply_chat_template for proper formatting
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            if hasattr(model, "device"):
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max(256, min(4096, self.config.max_context_chars // 4)),
                # temperature handled by mlx defaults
            )
            generated = outputs[0][input_len:]
            # Try parse_response for thinking mode support
            raw = processor.decode(generated, skip_special_tokens=False)
            if hasattr(processor, "parse_response"):
                parsed = processor.parse_response(raw)
                return str(parsed.get("content", parsed.get("text", raw))).strip()
            return processor.decode(generated, skip_special_tokens=True).strip()
        except Exception:
            # Fallback to simple prompt-based generation
            prompt = self._messages_to_prompt(messages)
            inputs = processor(prompt, return_tensors="pt")
            if hasattr(model, "device"):
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=2048)
            return processor.decode(outputs[0], skip_special_tokens=True).strip()

    @staticmethod
    def _inject_tools_into_messages(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Inject tool schemas into the system/first message for non-native backends."""
        result = list(messages)
        for i, msg in enumerate(result):
            if msg.get("role") == "system":
                result[i] = {
                    **msg,
                    "content": inject_tool_schemas_into_prompt(msg["content"], tools),
                }
                return result
        # No system message — prepend one
        return [
            {"role": "system", "content": inject_tool_schemas_into_prompt("", tools)},
            *result,
        ]

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = str(msg.get("content", ""))
            if content:
                parts.append(f"{role}:\n{content}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> Iterator[str]:
        for start in range(0, len(text), chunk_size):
            yield text[start:start + chunk_size]
