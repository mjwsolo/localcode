from __future__ import annotations

import json
import re
from typing import Any, Iterator

import httpx

from .config import RuntimeConfig
from .runtime_diffusion import _DiffusionMixin


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


def _log_disconnect_context(stage: str, error: Any, *, mid_stream: bool = False) -> dict:
    """Capture the disconnect CLASS around a lost-connection event.

    Pulls a snapshot from ServerManager (exit code, memory-guard fired?,
    running state) and records it to the structured event log
    (events.jsonl, type server_disconnect) AND as a human-readable line in
    last_error.log, so the disconnect class is visible next time instead of
    a bare "[E3102] Lost connection". Returns the diag dict. Never raises.
    """
    diag: dict = {}
    try:
        from .server_manager import ServerManager as _SM
        diag = _SM.get().disconnect_diagnostics()
    except Exception:
        diag = {}
    err_str = _error_message(error)
    diag["stage"] = stage
    diag["mid_stream"] = mid_stream
    diag["error"] = err_str[:500]
    # Capture the SERVER's own last words. llama-server prints the real cause
    # (ggml/Metal allocation failure, GGML_ASSERT, OOM) to server.log right
    # before it dies; without this we only see httpx's "connection dropped"
    # and are left guessing. The tail is the actual diagnosis.
    _server_tail = ""
    try:
        from .paths import global_state_dir
        _lp = global_state_dir() / "server.log"
        if _lp.exists():
            _lines = _lp.read_text(errors="replace").splitlines()
            _server_tail = "\n".join(_lines[-15:])[-1500:]
    except Exception:
        _server_tail = ""
    if _server_tail:
        diag["server_log_tail"] = _server_tail
    try:
        from .server_manager import _lifecycle_log as _ll
        _ll("server_disconnect", **diag)
    except Exception:
        pass
    try:
        import time as _t
        from .paths import last_error_log_path
        path = last_error_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                f"[{_t.strftime('%Y-%m-%dT%H:%M:%S')}] server_disconnect "
                f"stage={stage} class={diag.get('disconnect_class', 'unknown')} "
                f"exit_code={diag.get('exit_code')} "
                f"pressure_kill={diag.get('pressure_kill')} "
                f"running={diag.get('running')} mid_stream={mid_stream} "
                f"free_mb={diag.get('free_mb')} error={err_str[:300]}\n"
                + (f"  server.log tail:\n    "
                   + _server_tail.replace("\n", "\n    ") + "\n" if _server_tail else "")
            )
    except Exception:
        pass
    return diag


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


def apply_param_overrides(cmd: list[str], env: dict | None = None) -> list[str]:
    """Rewrite llama-server flags from ``LOCALCODE_OVERRIDE_*`` env vars.

    Lets the offline model-optimizer (dev/eval/model_opt.py) sweep launch
    parameters — GPU layers, context size, threads, batch — WITHOUT editing
    config or the catalog. Only flags already present in ``cmd`` are
    rewritten; an unset env var leaves the command untouched, so the default
    path (and every existing test) is byte-for-byte unchanged.

    Recognised: LOCALCODE_OVERRIDE_NGL / _NCTX / _THREADS / _BATCH.
    """
    import os
    e = env if env is not None else os.environ
    rewrites = {
        "-ngl": e.get("LOCALCODE_OVERRIDE_NGL"),
        "--ctx-size": e.get("LOCALCODE_OVERRIDE_NCTX"),
        "--threads": e.get("LOCALCODE_OVERRIDE_THREADS"),
        "-b": e.get("LOCALCODE_OVERRIDE_BATCH"),
    }
    for flag, val in rewrites.items():
        if val and flag in cmd:
            cmd[cmd.index(flag) + 1] = str(val)
    return cmd


class LocalCodeRuntimeGateway(_DiffusionMixin):
    """Talks to the local llama.cpp (llama-server) backend."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._client: httpx.Client | None = None
        self.last_response_meta: dict[str, Any] = {}  # for token tracking
        # Circuit-breaker state for the stream-recovery loop: consecutive
        # server deaths since the last successful stream. Persists across
        # calls/turns (this gateway is one per session) so a deterministic
        # crash can't restart-loop forever. Reset to 0 on any success.
        self._consecutive_stream_deaths = 0
        # llama_cpp is the only HTTP runtime (ollama/mlx/hf removed); diffusion
        # models are architecture-routed to the one-shot CLI, not this endpoint.
        base = self.config.base_url.rstrip("/")
        self.endpoint = f"{base}/v1/chat/completions"
        self.tags_endpoint = f"{base}/v1/models"

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
        # Prefix-cache reuse: llama-server already reuses the longest common
        # prefix of the KV cache per slot, so an unchanged system-prompt prefix
        # is free across turns. But this agent mutates mid-context (compaction,
        # read/write redaction in agent/context.py) which shifts every token
        # after the edit point and defeats naive prefix matching. --cache-reuse
        # lets llama-server salvage KV chunks AFTER such a gap instead of
        # re-prefilling the whole tail — directly cuts time-to-first-token on
        # turns that follow a compaction. 0 disables.
        if self.config.llama_cpp_cache_reuse and self.config.llama_cpp_cache_reuse > 0:
            cmd.extend(["--cache-reuse", str(self.config.llama_cpp_cache_reuse)])
        # Speculative decoding (mutual exclusion: draft model > lookup > ngram).
        # Speculative decoding is LOSSLESS — every drafted token is verified
        # against the real model, so output is identical, just faster.
        if self.config.llama_cpp_draft_model:
            draft_path = self.config.llama_cpp_draft_model
            cmd.extend(["--model-draft", draft_path,
                        "--draft-max", str(self.config.llama_cpp_draft_max)])
        elif self.config.llama_cpp_lookup_cache:
            # Prompt lookup decoding: matches n-grams from input in output (2-4x on edits)
            cmd.extend(["--lookup-cache-dynamic", "/tmp/localcode-lookup.bin"])
        elif self.config.llama_cpp_spec_type and self.config.llama_cpp_spec_type != "none":
            cmd.extend(["--spec-type", self.config.llama_cpp_spec_type,
                        "--draft-max", str(self.config.llama_cpp_draft_max)])
        # NO speculative decoding by default. An empty `llama_cpp_spec_type`
        # means OFF — NOT a cue to fall back to in-context n-gram decoding.
        # The previous default here emitted `--spec-type ngram-mod`, which
        # directly contradicted config.py's force-disable of spec_type (added
        # 2026-04-26 because n-gram/lookup decoding causes INFINITE VERBATIM
        # REPETITION: once the model emits a phrase, the n-gram drafter re-
        # proposes it and greedy verification accepts the whole block, so the
        # loop accelerates instead of breaking). config.py disabled it at the
        # config layer; this branch silently re-enabled it at launch. Ship OFF;
        # opt into a real draft model (lossless) or an explicit spec_type.
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
        # Tuned launch params: a stored model-opt recommendation applies first,
        # then explicit LOCALCODE_OVERRIDE_* env vars win. Both no-op by
        # default (no store file, no env), so the default path is unchanged.
        try:
            from .recommendations import load_overrides as _rec_overrides
            merged_env = {**_rec_overrides(_P(model_path).name), **os.environ}
        except Exception:
            merged_env = dict(os.environ)
        return apply_param_overrides(cmd, env=merged_env)

    @staticmethod
    def _ram_ctx_ceiling(ram_gb: int) -> int:
        """Single source of truth for the per-RAM context-window ceiling —
        used by BOTH the fastest+turbo path and the balanced/default lift so
        a given machine gets the SAME window on every preset (a 48 GB Mac was
        getting 64K on one path, 96K on another). Monotonic + gap-free across
        all real Mac sizes. The CAP here is the RAM/KV budget; the model's
        trained length is a separate cap applied via `_model_max_ctx`.

        The tier ladder lives in `model_config.RAM_CTX_CEILING_TIERS` (the
        central per-Mac config); this delegates so the numbers are edited in
        one place. Every current catalog model trains to >=256K, so on a big
        machine RAM is the binding constraint — 96 GB+ unlocks 256K; small-RAM
        tiers stay conservative (16 GB→64K, 64 GB→128K hardware-measured).
        """
        from .model_config import ram_ctx_ceiling
        return ram_ctx_ceiling(ram_gb)

    @staticmethod
    def _cohere_ctx_ceiling(ram_gb: int) -> int:
        """Conservative per-RAM context ceiling for the cohere2moe
        (North-Mini-Code) model.

        Why MUCH tighter than `_ram_ctx_ceiling`: that ladder is sized for
        the TurboQuant path, where the KV cache is compressed (q8_0 K +
        turbo4 V). North-Mini runs on the STOCK PR-#24260 server with
        *uncompressed f16* KV, so per-token KV is ~4x heavier, and there's
        no -fit guard. A 256K f16 KV on this model is multiple-tens-of-GB
        and OOM-kills the server mid-turn (the observed `zsh: killed` after
        the agent "thinking..." for 11 minutes). These tiers keep the
        allocated KV well within unified memory while still leaving plenty
        of room for this model's long unconditional reasoning. Monotonic.

        Tier ladder lives in `model_config.COHERE_CTX_CEILING_TIERS`.
        """
        from .model_config import cohere_ctx_ceiling
        return cohere_ctx_ceiling(ram_gb)

    def _model_max_ctx(self, model_path: str | None = None) -> int:
        """The model's trained context length (`*.context_length` in the GGUF
        header) — the hard cap we must never exceed, else the model decodes
        past where it was trained and output degrades. Read once per file and
        cached (this is called per-request via _options). Returns a large
        sentinel if the file can't be read, so an unreadable GGUF never
        wrongly clamps the window below the RAM ceiling.
        """
        import struct
        from pathlib import Path as _Path
        from .model_config import CTX_NO_CLAMP_SENTINEL
        path = str(model_path or getattr(self.config, "model", "") or "")
        if not path:
            return CTX_NO_CLAMP_SENTINEL
        cache = getattr(self, "_model_ctx_cache", None)
        if cache is None:
            cache = self._model_ctx_cache = {}
        if path in cache:
            return cache[path]
        result = CTX_NO_CLAMP_SENTINEL  # unknown → don't clamp
        try:
            if _Path(path).is_file():
                with open(path, "rb") as f:
                    if f.read(4) == b"GGUF":
                        f.read(4)            # version
                        f.read(8)            # tensor_count
                        n_kv, = struct.unpack("<Q", f.read(8))
                        _S = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
                              5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

                        def _rs():
                            ln, = struct.unpack("<Q", f.read(8))
                            return f.read(ln).decode("utf-8", "replace")

                        def _rv(t):
                            if t == 8:
                                return _rs()
                            if t == 9:
                                et, = struct.unpack("<I", f.read(4))
                                ln, = struct.unpack("<Q", f.read(8))
                                return [_rv(et) for _ in range(ln)]
                            fmt = _S[t]
                            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

                        for _ in range(n_kv):
                            k = _rs()
                            t, = struct.unpack("<I", f.read(4))
                            v = _rv(t)
                            if k.endswith(".context_length") and isinstance(v, int):
                                result = v
                                break
        except Exception:
            result = CTX_NO_CLAMP_SENTINEL
        cache[path] = result
        return result

    # ── General RAM-aware KV-cache cap (model-agnostic OOM guard) ───────
    #
    # The per-model `_cohere_ctx_ceiling` was a point-fix. The systematic
    # guard: never launch a `--ctx-size` whose KV cache won't fit in RAM,
    # for ANY model present or future. We compute the KV bytes/token from
    # the GGUF's own attention metadata × the dtype actually launched, then
    # cap ctx so weights + KV-at-full-ctx + headroom fit in unified memory.
    #
    # CRITICAL design choice — fail OPEN, only tighten when CONFIDENT:
    # the cap is applied only when the launched KV dtype has a known byte
    # size (f16/f32/q8_0/…). The TurboQuant path uses compressed KV types
    # (turbo4-V etc.) whose exact byte size we don't model; for those the
    # cap is a NO-OP and the existing hardware-validated ceilings
    # (`_ram_ctx_ceiling`) govern — so this never regresses the tuned
    # 64K/128K/256K turbo tiers, it only catches the uncompressed-f16
    # stock-path class that actually OOM'd (cohere/North-Mini).

    # Bytes per KV element by quant type. None-on-miss is deliberate:
    # an unknown (e.g. "turbo4") type means "don't clamp — defer to the
    # validated ceilings", never "guess and risk a wrong cap". The table is
    # the central `model_config.KV_DTYPE_BYTES`; aliased here so the existing
    # `self._KV_DTYPE_BYTES.get(...)` call sites are unchanged.
    from .model_config import KV_DTYPE_BYTES as _KV_DTYPE_BYTES

    def _effective_kv_dtypes(self, model_path: str | None) -> tuple[str, str]:
        """The KV (K, V) dtypes the server is ACTUALLY launched with for this
        model — not just config, which can disagree with the launch path.
        The cohere2moe stock path ignores --cache-type and runs uncompressed
        f16; every other path uses the configured types (default f16)."""
        if self._is_cohere_gguf(model_path):
            return ("f16", "f16")
        k = (getattr(self.config, "kv_cache_type_k", "") or "f16").lower()
        v = (getattr(self.config, "kv_cache_type_v", "") or "f16").lower()
        return (k, v)

    def _gguf_kv_meta(self, model_path: str | None = None) -> dict | None:
        """Read the attention metadata needed to size the KV cache:
        n_layers (block_count), n_kv_heads (head_count_kv, falling back to
        head_count), and head_dim (key_length, falling back to
        embedding_length // head_count). Returns None if the file can't be
        read or the keys are missing. Cached per path."""
        import struct
        from pathlib import Path as _Path
        path = str(model_path or getattr(self.config, "model", "") or "")
        if not path:
            return None
        cache = getattr(self, "_gguf_meta_cache", None)
        if cache is None:
            cache = self._gguf_meta_cache = {}
        if path in cache:
            return cache[path]
        meta: dict | None = None
        found: dict[str, int] = {}
        try:
            if _Path(path).is_file():
                with open(path, "rb") as f:
                    if f.read(4) == b"GGUF":
                        f.read(4); f.read(8)
                        n_kv, = struct.unpack("<Q", f.read(8))
                        _S = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
                              5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

                        def _rs():
                            ln, = struct.unpack("<Q", f.read(8))
                            return f.read(ln).decode("utf-8", "replace")

                        def _rv(t):
                            if t == 8:
                                return _rs()
                            if t == 9:
                                et, = struct.unpack("<I", f.read(4))
                                ln, = struct.unpack("<Q", f.read(8))
                                return [_rv(et) for _ in range(ln)]
                            fmt = _S[t]
                            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

                        wanted = (
                            ".block_count", ".attention.head_count_kv",
                            ".attention.head_count", ".attention.key_length",
                            ".embedding_length",
                        )
                        for _ in range(n_kv):
                            k = _rs()
                            t, = struct.unpack("<I", f.read(4))
                            v = _rv(t)
                            for suf in wanted:
                                if k.endswith(suf) and isinstance(v, int):
                                    found[suf] = v
                        n_layers = found.get(".block_count")
                        n_heads = found.get(".attention.head_count")
                        n_kv_heads = found.get(".attention.head_count_kv", n_heads)
                        head_dim = found.get(".attention.key_length")
                        if head_dim is None and n_heads and found.get(".embedding_length"):
                            head_dim = found[".embedding_length"] // n_heads
                        if n_layers and n_kv_heads and head_dim:
                            meta = {
                                "n_layers": n_layers,
                                "n_kv_heads": n_kv_heads,
                                "head_dim": head_dim,
                            }
        except Exception:
            meta = None
        cache[path] = meta
        return meta

    def _kv_bytes_per_token(self, model_path: str | None = None) -> float | None:
        """KV-cache bytes per token = n_layers × n_kv_heads × head_dim ×
        (bytes_K + bytes_V). Returns None when the metadata or the launched
        KV dtype is unknown (→ caller does not clamp)."""
        meta = self._gguf_kv_meta(model_path)
        if not meta:
            return None
        kt, vt = self._effective_kv_dtypes(model_path)
        kb = self._KV_DTYPE_BYTES.get(kt)
        vb = self._KV_DTYPE_BYTES.get(vt)
        if kb is None or vb is None:
            return None  # compressed/unknown (e.g. turbo) → defer to ceilings
        return meta["n_layers"] * meta["n_kv_heads"] * meta["head_dim"] * (kb + vb)

    def _model_file_bytes(self, model_path: str | None = None) -> int:
        from pathlib import Path as _Path
        try:
            return _Path(str(model_path or self.config.model or "")).stat().st_size
        except Exception:
            return 0

    def _kv_aware_ctx_ceiling(self, model_path: str | None, ram_gb: int) -> int:
        """Largest context whose KV cache fits in RAM beside the weights, with
        headroom for the OS, the app, and compute activations. Returns a large
        sentinel (no clamp) when KV size isn't confidently computable, so the
        existing validated ceilings govern the turbo-compressed path."""
        from .model_config import (
            CTX_NO_CLAMP_SENTINEL, KV_FIT_CTX_MULTIPLE, KV_FIT_MIN_CTX,
            KV_FIT_RESERVE_FRACTION, KV_FIT_RESERVE_GB,
        )
        bpt = self._kv_bytes_per_token(model_path)
        if not bpt or bpt <= 0:
            return CTX_NO_CLAMP_SENTINEL
        total = ram_gb * (1024 ** 3)
        weights = self._model_file_bytes(model_path)
        # Reserve the larger of 3 GB or 15% of RAM for OS + app + activations.
        reserve = max(KV_FIT_RESERVE_GB * 1024 ** 3, int(total * KV_FIT_RESERVE_FRACTION))
        kv_budget = total - weights - reserve
        if kv_budget <= 0:
            return KV_FIT_MIN_CTX  # weights barely fit; give the minimum workable ctx
        max_ctx = int(kv_budget / bpt)
        # Round down to a 2048 multiple; never below a 2048 floor.
        return max(KV_FIT_MIN_CTX, (max_ctx // KV_FIT_CTX_MULTIPLE) * KV_FIT_CTX_MULTIPLE)

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

    def _is_cohere_gguf(self, model_path: str | None = None) -> bool:
        """True for the cohere2moe (North-Mini-Code) model.

        This model is served by the STOCK PR-#24260 llama-server, NOT the
        TurboQuant fork — so it gets f16 KV (no q8_0-K / turbo4-V
        compression) and no -fit/ctx-checkpoint memory management. Combined
        with its 500K trained context and UNCONDITIONAL reasoning (every
        turn emits a long <think> preamble), it needs its own conservative
        context + generation caps so a single long turn can't grow the KV
        cache until the OS OOM-kills the server. Detection is architecture-
        based (catalog), with a name fallback for unreadable/renamed files.
        """
        try:
            from .models_catalog import by_filename as _bf
            from pathlib import Path as _P
            name = (model_path or self.config.model or "")
            choice = _bf(_P(name).name) if name else None
            if choice is not None and "cohere" in str(getattr(choice, "architecture", "")):
                return True
        except Exception:
            pass
        name = (model_path or self.config.model or "").lower()
        return "north-mini" in name or "cohere2" in name

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

        Uses llama.cpp's /v1/chat/completions (OpenAI-compatible) endpoint.
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
        # llama_cpp is the only provider; the branch above always returns.
        return ""

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = self.client.get(self.tags_endpoint)
            response.raise_for_status()
            return True, self.tags_endpoint
        except Exception as exc:
            return False, str(exc)

    def list_models(self) -> list[str]:
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
        raw = self._target_num_ctx_uncapped(num_ctx_override, model_path)
        if num_ctx_override is not None:
            return raw  # explicit, deliberate value — respect it as-is
        # Universal RAM-aware KV cap on EVERY computed path: no model, present
        # or future, gets a `--ctx-size` whose KV cache can't fit (the OOM
        # never-again guard). No-op when KV size isn't confidently computable
        # (turbo-compressed KV) — the validated ceilings govern there.
        return min(
            raw, self._kv_aware_ctx_ceiling(model_path, self._system_ram_gb())
        )

    def _target_num_ctx_uncapped(
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

        # cohere2moe (North-Mini-Code): served by the STOCK server with
        # UNCOMPRESSED f16 KV, so it must NOT inherit the large TurboQuant
        # context ceilings (256K on a big Mac). Clamp to a conservative
        # RAM-aware ceiling BEFORE the preset branches — this bounds the
        # KV cache the launched `--ctx-size` allocates and is the primary
        # OOM guard for this model's long unconditional-reasoning turns.
        # Still never exceed the model's trained length.
        if self._is_cohere_gguf(model_path):
            return min(
                num_ctx,
                self._cohere_ctx_ceiling(self._system_ram_gb()),
                self._model_max_ctx(model_path),
            )

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
                return min(
                    self._ram_ctx_ceiling(ram_gb),
                    self._model_max_ctx(model_path),
                )
            return min(num_ctx, 16384 if turbo else 3072)
        # RAM-aware lift for the balanced/default path. Without this, ctx was
        # a flat `max_context_chars // 4` (~50K) on EVERY machine — a 128 GB
        # Mac got the same window as a 16 GB one. On a capable machine that
        # starved long agentic sessions: once the window filled, the model
        # lost earlier turns and re-read files it had already read (the
        # observed "re-reading App.tsx forever" loop). A big machine can hold
        # a big KV cache, so give it one. Small machines keep their sizing.
        # Lift only on >=32 GB machines (lots of headroom regardless of KV
        # type). Below that we keep the chars-based value to avoid an OOM on
        # an unvalidated small-RAM × KV-type combination.
        from .model_config import BALANCED_RAM_LIFT_MIN_GB
        ram_gb = self._system_ram_gb()
        if ram_gb >= BALANCED_RAM_LIFT_MIN_GB:
            num_ctx = max(num_ctx, self._ram_ctx_ceiling(ram_gb))
        # Never exceed the model's trained context length (degrades past it).
        return min(num_ctx, self._model_max_ctx(model_path))

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
            # Sequence breakers RESET DRY's n-gram matching at these tokens, so
            # a repeated file PATH / identifier isn't treated as a penalizable
            # loop. Without them (llama.cpp's default breakers don't include
            # path chars), a path repeated within one round — `cd X && mkdir X
            # && ls X` — got its username subtokens suppressed and the model
            # emitted mangled variants of the home-dir username. Breaking on
            # `/ . _ - space` keeps DRY for phrasal loops but frees paths/idents.
            "dry_sequence_breakers": ["\n", "/", ".", "_", "-", " ", ":", "\"", "'"],
        }
        if self.config.mode == "fast":
            opts["num_predict"] = 4096  # cap generation for speed
        if num_predict_override is not None:
            if num_predict_override == -1:
                opts["num_predict"] = -1  # unlimited — model stops at EOS
            else:
                opts["num_predict"] = max(64, int(num_predict_override))
        # cohere2moe (North-Mini-Code) reasons UNCONDITIONALLY — every turn
        # emits a long <think> preamble — so an unbounded ("-1") per-turn
        # budget lets a single "thinking" turn grow the KV cache until the
        # OS OOM-kills the stock server (observed: 11 min of "thinking..."
        # then `zsh: killed`). Cap each turn to a generous-but-bounded
        # ceiling that still fits reasoning + a complete tool call, scaled
        # so it can never approach the (already-bounded) context window.
        # This only TIGHTENS an otherwise-unlimited/oversized budget.
        if self._is_cohere_gguf():
            from .model_config import cohere_generation_cap
            cap = cohere_generation_cap(self._target_num_ctx())
            cur = opts.get("num_predict")
            if not isinstance(cur, int) or cur <= 0 or cur > cap:
                opts["num_predict"] = cap
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
                # Server error or connection lost — try restarting. chat_once
                # is NON-streaming: nothing has been handed to the caller yet,
                # so a clean restart + full retry is always safe here (unlike
                # the streaming path, which must guard a partial stream).
                is_500 = hasattr(exc, 'response') and getattr(exc.response, 'status_code', 0) == 500
                is_conn_err = "connect" in str(exc).lower() or "refused" in str(exc).lower()
                if is_500 or is_conn_err:
                    if is_conn_err:
                        # Capture the disconnect class before restarting so
                        # the cause is visible later.
                        _log_disconnect_context("chat_once", exc)
                    try:
                        # _restart_server() blocks on the health probe, so by
                        # the time it returns the model is loaded and the retry
                        # lands on a ready server (not a still-loading 503).
                        if self._restart_server():
                            _clear_pressure_kill_marker()
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
                else:
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
        # Circuit-breaker ceiling: how many consecutive server deaths (across
        # calls/turns, reset on any successful stream) we'll keep restarting
        # through before declaring the server unstable and stopping the loop.
        _MAX_CONSECUTIVE_STREAM_DEATHS = 3
        # STREAM-LEVEL "have we handed real content to the consumer yet?"
        # flag. The per-attempt `_emitted_real` is reset each try; this one
        # survives across attempts and is the streaming-safety gate for
        # connection-drop recovery: a drop BEFORE any content was yielded is
        # safe to restart+retry; a drop MID-STREAM (content already emitted)
        # must NOT silently replay — we surface E3102 instead.
        _emitted_real_stream = False
        # Gemma-4 token-soup collapse is INTERMITTENT (the same "hi" that
        # collapses now answers cleanly on a re-sample). The collapse path
        # used to hard-fail with E3108 and no retry. Instead, re-generate
        # up to _MAX_COLLAPSE_RETRIES times before giving up — each retry
        # ESCALATING the sampler (fresh seed + higher repeat_penalty +
        # temperature push; see the collapse-retry block below) so the
        # resample can't reproduce the same degenerate path. We only retry
        # a collapse that fires BEFORE any real content has been streamed
        # to the consumer (the soup tokens are stripped, so the consumer
        # has seen nothing yet) — that keeps the retry clean with no
        # double-emit, and the "hi" collapse always falls in this window.
        # A collapse mid-answer (rare) still surfaces E3108 rather than
        # re-streaming a partially-consumed turn.
        _MAX_COLLAPSE_RETRIES = 3
        _collapse_retries_done = 0
        # The attempt budget must cover BOTH connection retries and collapse
        # retries — otherwise a low `max_retries` would starve collapse
        # recovery. Collapse retries are gated separately by
        # `_collapse_retries_done` so this only RAISES the ceiling; it never
        # forces extra connection attempts.
        _attempt_budget = max(
            1, self.config.max_retries + 1, _MAX_COLLAPSE_RETRIES + 1
        )
        for attempt in range(_attempt_budget):
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
                _collapse_hits = 0     # raw <unusedNN>/[multimodal] tokens seen
                _collapsed = False     # known llama.cpp Gemma-4 token-soup loop
                # True once a real (non-soup) content/thinking/tool delta has
                # been yielded downstream this attempt. Gates collapse-retry:
                # we can only cleanly re-generate a collapse that fired before
                # anything reached the consumer (no double-emit).
                _emitted_real = False
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
                        # Known llama.cpp Gemma-4 bug: the 26B-A4B MoE (and 31B
                        # dense) collapse into a loop of raw <unusedNN> /
                        # [multimodal] tokens during longer generation. The
                        # tokens are stripped from the display, but the model
                        # keeps spending the whole budget on soup — so once the
                        # collapse is unmistakable, STOP reading.
                        _collapse_hits += (
                            (_delta_content + _delta_reasoning).count("<unused")
                            + (_delta_content + _delta_reasoning).count("[multimodal]")
                        )
                        if _collapse_hits >= 8:
                            _collapsed = True
                            last_finish_reason = "collapse"
                            break
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
                                    _emitted_real = True
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
                            _emitted_real = True
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
                                    _emitted_real = True
                                    yield {"type": "thinking", "content": cleaned}
                                continue
                            if in_thinking:
                                if _close in content:
                                    # Thinking block ends
                                    before = content.split(_close, 1)[0]
                                    cleaned = _strip_thinking_tokens(before, _family)
                                    if cleaned.strip():
                                        _emitted_real = True
                                        yield {"type": "thinking", "content": cleaned}
                                    in_thinking = False
                                    after = content.split(_close, 1)[1]
                                    cleaned_after = _strip_thinking_tokens(after, _family)
                                    if cleaned_after.strip():
                                        _emitted_real = True
                                        yield {"type": "content", "content": cleaned_after}
                                else:
                                    # Still inside thinking block
                                    cleaned = _strip_thinking_tokens(content, _family)
                                    if cleaned.strip():
                                        _emitted_real = True
                                        yield {"type": "thinking", "content": cleaned}
                                continue
                        cleaned = _strip_thinking_tokens(content)
                        if cleaned:
                            _emitted_real = True
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
                # Gemma-4 collapse recovery. The collapse is intermittent, so
                # re-generate the SAME request before surfacing E3108 — but
                # only while nothing real has reached the consumer yet, so the
                # retry can't double-emit content. (The collapse-on-"hi" case
                # always lands here: the soup is stripped, so _emitted_real is
                # still False when the collapse trips.) On the retry we nudge
                # sampling slightly to break out of the bad sample: a small
                # temperature bump plus the existing repeat_penalty knob. Only
                # if EVERY attempt collapses do we fall through to E3108.
                if (
                    _collapsed
                    and not _emitted_real
                    and _collapse_retries_done < _MAX_COLLAPSE_RETRIES
                ):
                    _collapse_retries_done += 1
                    yield {
                        "type": "stage",
                        "name": "collapse_retry",
                        "message": (
                            "The model produced repeated junk tokens — "
                            "re-generating the response."
                        ),
                    }
                    # Actively break the collapse on the retry instead of
                    # re-sending a byte-identical request. The old code only
                    # bumped temperature and clamped it to 1.0 — but Gemma's
                    # base temperature IS 1.0 (Unsloth spec, see _options), so
                    # the clamp made the "nudge" a no-op: every retry re-sampled
                    # the exact same degenerate path and we burned the whole
                    # budget before surfacing E3108. Three independent levers:
                    #   1. seed — force a different RNG trajectory so an
                    #      intermittent collapse can't reproduce
                    #      deterministically (defeats a pinned server seed).
                    #   2. repeat_penalty — escalate hard (1.30 → 1.55 → 1.60);
                    #      it divides the logit of the looping token directly,
                    #      the exact antidote to "junk × N".
                    #   3. temperature — allow a modest push ABOVE the 1.0
                    #      quality ceiling. We're already collapsed, so trading
                    #      a little coherence for escape velocity is correct.
                    # Written both top-level (llama_cpp / OpenAI-compat payload)
                    # and into `options` (ollama-style payload) so the escalation
                    # lands whichever shape `_payload` produced.
                    import random as _rand_mod
                    _new_seed = _rand_mod.randint(1, 2**31 - 1)
                    _new_rp = min(1.60, 1.05 + 0.25 * _collapse_retries_done)
                    # Read the base temperature from whichever shape _payload
                    # produced: top-level for llama_cpp/OpenAI-compat, nested
                    # under `options` for ollama. Reading only top-level would
                    # misread the ollama base as 1.0 and escalate off the wrong
                    # anchor.
                    _opts = payload.get("options")
                    _base_t = payload.get("temperature")
                    if _base_t is None and isinstance(_opts, dict):
                        _base_t = _opts.get("temperature")
                    try:
                        _t = float(_base_t if _base_t is not None else 1.0)
                    except (TypeError, ValueError):
                        _t = 1.0
                    _new_t = min(1.2, _t + 0.08 * _collapse_retries_done)
                    for _tgt in (payload, _opts if isinstance(_opts, dict) else None):
                        if _tgt is None:
                            continue
                        _tgt["seed"] = _new_seed
                        _tgt["repeat_penalty"] = _new_rp
                        _tgt["temperature"] = _new_t
                    continue
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
                if _collapsed:
                    # Surface the canonical error code (E3108) instead of the
                    # stripped-to-near-empty soup or ad-hoc prose.
                    from .errors import (
                        LocalCodeError as _LCE,
                        by_code as _by_code,
                        format_for_user as _fmt_err,
                    )
                    yield {
                        "type": "content",
                        "content": _fmt_err(_LCE(_by_code("E3108"))),
                    }
                # Reached the end of a stream without an exception — the server
                # answered. Clear the circuit breaker so a later transient death
                # gets a fresh recovery budget (only CONSECUTIVE deaths count).
                self._consecutive_stream_deaths = 0
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
                # Carry this attempt's emission state up to the stream level.
                # `_emitted_real` is defined at the top of the try; if the
                # exception fired before that line it won't exist, so guard.
                try:
                    if _emitted_real:
                        _emitted_real_stream = True
                except NameError:
                    pass
                # Memory-pressure recovery. The pressure monitor SIGTERMs
                # llama-server when macOS reports CRITICAL memory pressure
                # (kern.memorystatus_vm_pressure_level >= 4) for ~2s. The
                # next request lands on a dead socket → "connection
                # refused". Without recovery the user sees E3102, the
                # session is effectively dead, they have to relaunch.
                # With recovery: detect the pattern, wait for memory to
                # settle, restart the server, retry the stream once.
                err_str = _error_message(exc).lower()
                # Classify by EXCEPTION TYPE (with a text fallback), not a bare
                # `"connect" in err_str` — that substring also matched httpx's
                # RemoteProtocolError ("peer closed CONNECTION … incomplete
                # chunked read"), but that's fine on its own; both a dead socket
                # AND a mid-body drop are recoverable by restart. The real
                # defect was the ABSENCE of a stop condition: a deterministic
                # failure (an oversized request that kills the server every
                # time) restarted + re-POSTed forever. The circuit breaker
                # below (_consecutive_stream_deaths) bounds that.
                try:
                    import httpx as _httpx
                    _is_protocol_drop = isinstance(exc, _httpx.RemoteProtocolError)
                    _is_connect_err = isinstance(
                        exc, (_httpx.ConnectError, _httpx.ConnectTimeout)
                    )
                except Exception:
                    _is_protocol_drop = "incomplete chunked read" in err_str or "peer closed" in err_str
                    _is_connect_err = False
                if not _is_connect_err and not _is_protocol_drop:
                    _is_connect_err = "connect" in err_str or "refused" in err_str
                # Both a connect failure and a mid-stream protocol drop are
                # "server went away" — recoverable by restart + retry.
                is_conn_err = _is_connect_err or _is_protocol_drop
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
                # STREAMING-SAFETY: a mid-stream drop that already yielded
                # content may replay a partial duplicate on retry — acceptable
                # (the recovery stage explains it) and far better than killing a
                # turn that would otherwise self-heal (a hard E3102 here used to
                # kill builds on a transient memory-pressure pause).
                if is_conn_err:
                    # Capture the disconnect class for diagnostics (both a dead
                    # socket and a mid-body protocol drop land here).
                    _log_disconnect_context(
                        "stream_chat_events", exc,
                        mid_stream=_emitted_real_stream or _is_protocol_drop,
                    )
                    # CIRCUIT BREAKER — the fix for the restart loop. Count
                    # consecutive server deaths across calls (reset on any
                    # successful stream, see below). A transient death recovers
                    # on the first restart; a DETERMINISTIC one — a request too
                    # large for the server, killing it every time — would
                    # otherwise restart + re-POST forever. Once the deaths pile
                    # up we stop restarting and surface actionable guidance.
                    self._consecutive_stream_deaths = (
                        getattr(self, "_consecutive_stream_deaths", 0) + 1
                    )
                    if self._consecutive_stream_deaths > _MAX_CONSECUTIVE_STREAM_DEATHS:
                        yield {
                            "type": "stage", "name": "server_unstable",
                            "message": (
                                "The model server keeps crashing on this request — "
                                "it's likely too large for the current settings. "
                                "Try /clear to shrink the context, or /model to "
                                "switch to a lighter model."
                            ),
                        }
                        break  # stop the restart loop; fall through to E3102
                if (
                    is_conn_err
                    and attempt < self.config.max_retries
                    and _restarts_done < _MAX_STREAM_RESTARTS
                ):
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
            # Forward the EOS-NEUTRAL anti-loop samplers that were previously
            # computed into `opts` and then DROPPED (they only went to the dead
            # ollama-shaped payload below). DRY penalises repeated N-GRAMS —
            # not single tokens — so unlike repeat_penalty it does NOT suppress
            # the EOS/sentence-end tokens that caused the 2026-04-29 paraphrase
            # regression; min_p trims the low-probability tail that token-
            # collapse loops feed on. llama-server has DRY OFF by default, so
            # NOT forwarding these is a real root cause of the repeat-collapse
            # loops. We deliberately do NOT re-add the aggressive repeat_penalty
            # /top_k/top_p bundle (that was the regression). A/B back to
            # temperature-only with LOCALCODE_SAMPLER_MINIMAL=1.
            import os as _os
            if _os.environ.get("LOCALCODE_SAMPLER_MINIMAL") != "1":
                payload["dry_multiplier"] = opts["dry_multiplier"]
                payload["dry_base"] = opts["dry_base"]
                payload["dry_allowed_length"] = opts["dry_allowed_length"]
                payload["dry_penalty_last_n"] = opts["dry_penalty_last_n"]
                if opts.get("dry_sequence_breakers"):
                    payload["dry_sequence_breakers"] = opts["dry_sequence_breakers"]
                if opts.get("min_p", 0):
                    payload["min_p"] = opts["min_p"]
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
            import logging
            # Always log the ACTUAL forwarded samplers once per session (was
            # opt-in) so we can correlate looping with the real server config —
            # the "control this better" record. Cheap: fires exactly once.
            if not getattr(self, "_logged_samplers", False):
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
