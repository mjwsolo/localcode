from __future__ import annotations

import json
import re
from typing import Any, Iterator

import httpx

from .config import RuntimeConfig


def _strip_thinking_tokens(text: str) -> str:
    """Strip Gemma 4 thinking channel tokens that leak through at IQ3_S quant."""
    if not text:
        return text
    # <unused25> is the raw decode of <|channel> / <channel|> tokens
    text = text.replace("<unused25>", "")
    # Strip actual channel tags if present
    text = re.sub(r"<\|channel>thought\n?", "", text)
    text = re.sub(r"<channel\|>\n?", "", text)
    return text
from .tool_parsing import (
    inject_tool_schemas_into_prompt,
    parse_tool_calls,
)


class RuntimeErrorWithContext(RuntimeError):
    pass


class StreamEvent(dict):
    pass


class GemRuntimeGateway:
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
            self._client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=float(self.config.request_timeout_seconds),
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
        binary = self.config.llama_cpp_binary or str(_turboquant_binary_path() or "") or "llama-server"
        mode = self.config.laptop_26b_runtime_mode

        # Context mode benefits from all CPU cores for expert computation
        threads = 10 if mode == "context" else self.config.llama_cpp_threads
        cmd = [
            binary,
            "--model", model_path,
            "--port", str(port),
            "--ctx-size", str(self._target_num_ctx()),
            "--threads", str(threads),
            "--flash-attn", "on",
        ]

        if mode in ("turbo", "turbo-think"):
            # Full GPU: all layers on Metal via mmap shared buffers, 2 graph splits
            # Requires: sudo sysctl iogpu.wired_limit_mb=14336
            cmd.extend(["--mmap", "-ngl", "999", "-fit", "off", "--cache-ram", "0"])
        elif mode == "context":
            # GPU mode: attention on Metal, experts on CPU, mmap for SSD paging
            cmd.extend(["--mmap", "-ngl", "999", "-ot", "exps=CPU"])
        elif self.config.llama_cpp_expert_offload:
            # Explicit expert offload (legacy config)
            cmd.extend(["--mmap", "-ngl", "999", "-ot", "exps=CPU"])
        else:
            # Speed mode (default): CPU mmap, no GPU — fastest decode
            cmd.extend(["--mmap", "-ngl", str(self.config.llama_cpp_gpu_layers)])
        # KV cache compression (asymmetric: q8_0 K + turbo4 V recommended)
        ctk = self.config.kv_cache_type_k
        ctv = self.config.kv_cache_type_v
        if ctk and ctk != "f16":
            cmd.extend(["--cache-type-k", ctk])
        if ctv and ctv != "f16":
            cmd.extend(["--cache-type-v", ctv])
        # Speculative decoding (mutual exclusion: draft model > lookup > ngram)
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
        elif self.config.llama_cpp_spec_type:
            cmd.extend(["--spec-type", self.config.llama_cpp_spec_type,
                        "--draft-max", str(self.config.llama_cpp_draft_max)])
        # Batch sizes — larger batches for GPU mode, smaller for CPU
        if mode == "context":
            cmd.extend(["-b", "2048", "-ub", "512"])  # GPU benefits from bigger batches
        else:
            cmd.extend(["-b", str(self.config.llama_cpp_batch_size),
                        "-ub", str(min(512, self.config.llama_cpp_batch_size))])
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

    def generate_once(self, messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
        """Call model for a one-shot generation (no streaming, no tool calls).

        Routes to the correct endpoint per provider:
        - Ollama: /api/generate (bypasses chat template to avoid <|tool_response> tokens)
        - llama.cpp: /v1/chat/completions (OpenAI-compatible)
        - MLX/HF: in-process generation
        """
        if self.config.provider == "llama_cpp":
            use_think = self.config.laptop_26b_runtime_mode.endswith("-think")
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

    def _target_num_ctx(self, num_ctx_override: int | None = None) -> int:
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
                # Scale context based on system RAM
                import subprocess
                try:
                    mem_bytes = int(subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True, timeout=2
                    ).stdout.strip())
                    ram_gb = mem_bytes // (1024 ** 3)
                except Exception:
                    ram_gb = 16
                if ram_gb >= 64:
                    return 131072  # 128K context
                elif ram_gb >= 32:
                    return 65536   # 64K context
                elif ram_gb >= 24:
                    return 49152   # 48K context
                return 32768       # 16GB: 32K context
            return min(num_ctx, 16384 if turbo else 3072)
        return num_ctx

    def _options(self, num_ctx_override: int | None = None, num_predict_override: int | None = None) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "temperature": self.config.temperature,
            "num_ctx": self._target_num_ctx(num_ctx_override),
            "top_p": 0.95,
            "top_k": 64,
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
        think: bool = True,
        format: dict[str, Any] | str | None = None,
        num_predict: int | None = None,
    ) -> dict[str, Any]:
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
                    raise RuntimeErrorWithContext(data["error"])
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

    def _restart_server(self) -> None:
        """Kill and relaunch llama-server with correct binary and flags."""
        import os
        import subprocess
        import time
        from .bootstrap import get_model_path
        # Kill by port first (catches zombies pkill misses)
        try:
            r = subprocess.run(["lsof", "-ti", ":8081"], capture_output=True, text=True, timeout=3)
            for pid in r.stdout.strip().split():
                if pid:
                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=2)
        except Exception:
            pass
        try:
            subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True, timeout=3)
        except Exception:
            pass
        time.sleep(1)
        model = get_model_path()
        if not model:
            return
        cmd = self.llama_server_command(str(model))
        env = dict(os.environ)
        env["GGML_BACKEND_PATH"] = ""
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, env=env)
            # Wait for healthcheck
            for _ in range(30):
                time.sleep(1)
                try:
                    ok, _ = self.healthcheck()
                    if ok:
                        return
                except Exception:
                    pass
        except Exception:
            pass

    def stream_chat_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        think: bool = True,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ) -> Iterator[dict[str, Any]]:
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
        for attempt in range(max(1, self.config.max_retries + 1)):
            try:
                # Accumulate tool call deltas (OpenAI streams them incrementally)
                pending_tools: dict[int, dict] = {}  # index -> {id, name, arguments}
                # Stateful thinking detection for llama.cpp
                # Gemma 4 thinking tokens decode as <unused25> through llama.cpp
                in_thinking = False  # True while inside thinking block
                with self.client.stream("POST", self.endpoint, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if self.config.provider == "llama_cpp" and line.startswith("data: "):
                            line = line[6:]
                            if line == "[DONE]":
                                break
                        data = json.loads(line)
                        if "error" in data:
                            raise RuntimeErrorWithContext(data["error"])
                        message = self._extract_message(data)
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
                                pending_tools[idx]["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                pending_tools[idx]["function"]["arguments"] += fn["arguments"]
                        # Check for explicit reasoning_content first
                        thinking = message.get("thinking")
                        if thinking:
                            yield {"type": "thinking", "content": thinking}
                            continue
                        content = message.get("content", "")
                        if not content:
                            continue
                        # Stateful detection of <unused25> thinking tags in content
                        # Gemma 4: <unused25> = opening thinking tag, second <unused25> = closing
                        if self.config.provider == "llama_cpp" and think:
                            if not in_thinking and "<unused25>" in content:
                                # Thinking block starts
                                in_thinking = True
                                after = content.split("<unused25>", 1)[1]
                                cleaned = _strip_thinking_tokens(after)
                                if cleaned.strip():
                                    yield {"type": "thinking", "content": cleaned}
                                continue
                            if in_thinking:
                                if "<unused25>" in content:
                                    # Thinking block ends
                                    before = content.split("<unused25>", 1)[0]
                                    cleaned = _strip_thinking_tokens(before)
                                    if cleaned.strip():
                                        yield {"type": "thinking", "content": cleaned}
                                    in_thinking = False
                                    after = content.split("<unused25>", 1)[1]
                                    cleaned_after = _strip_thinking_tokens(after)
                                    if cleaned_after.strip():
                                        yield {"type": "content", "content": cleaned_after}
                                else:
                                    # Still inside thinking block
                                    cleaned = _strip_thinking_tokens(content)
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
                # Yield accumulated tool calls at the end
                if pending_tools:
                    yield {"type": "tool_calls", "tool_calls": list(pending_tools.values())}
                return
            except Exception as exc:
                last_error = exc
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
                "temperature": opts["temperature"],
                "chat_template_kwargs": {"enable_thinking": think},
            }
            if "num_predict" in opts:
                payload["max_tokens"] = opts["num_predict"]
            if tools:
                payload["tools"] = tools
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
