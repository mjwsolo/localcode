"""DiffusionGemma backend, extracted from runtime.py.

Block-diffusion models denoise a whole block of tokens in parallel
instead of decoding token-by-token, so llama-server (and its HTTP
streaming API) can't drive them. Generation goes through the one-shot
`llama-diffusion-cli` runner (upstream llama.cpp PR #24423), which ships
in the wheel next to llama-server (see `bootstrap.diffusion_cli_path`).

This module hosts ``_DiffusionMixin`` — a behaviour-preserving extraction
of the diffusion methods that previously lived on
``LocalCodeRuntimeGateway``. The gateway inherits from this mixin, so the
method names and call sites (``self._stream_diffusion_events`` etc.) and
the static-method entry points
(``LocalCodeRuntimeGateway._format_diffusion_prompt`` etc.) are unchanged.

Import hygiene: this module does NOT import ``runtime`` at module level,
to avoid an import cycle (runtime imports ``_DiffusionMixin`` at module
level). ``RuntimeErrorWithContext`` is imported lazily inside the methods
that raise it.
"""

from __future__ import annotations

from typing import Any, Iterator


class _DiffusionMixin:
    """DiffusionGemma backend (experimental) for LocalCodeRuntimeGateway.

    Block-diffusion models denoise a whole block of tokens in parallel
    instead of decoding token-by-token, so llama-server (and its HTTP
    streaming API) can't drive them. Generation goes through the
    one-shot `llama-diffusion-cli` runner (upstream llama.cpp PR #24423),
    shipped in the wheel next to llama-server. Consequences:
      - the model weights are (re)mapped per turn — first turn is slow,
        later turns are faster via the OS page cache;
      - output arrives in coarse chunks (denoised blocks), not tokens;
      - we apply the Gemma chat template ourselves (-p takes raw text).
    """

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

        UNIFIED PROMPT: diffusion uses the SAME system prompt as every
        other model — it's already in `messages` as the system turn,
        fully formatted by the agent loop. (The old code substituted a
        bespoke "concise" prompt under the belief that the full one
        overflowed the canvas and produced empty output; that was a
        misdiagnosis — the real cause was `num_predict=-1` sent as the
        CLI's `-n` canvas size. With that fixed the full prompt works,
        verified on BF16 + Q4.)

        Two things are still diffusion-specific and APPENDED, because they
        are delivery mechanics, not prompt content:
          * the PLAIN-JSON tool block — diffusion has no server/template to
            deliver tools natively, and the Gemma special-token tool format
            collapses it to empty output;
          * a one-line nudge to skip the `thought` preamble some quants
            (notably BF16) emit ahead of the answer.
        """
        system_text = ""
        for m in messages:
            if m.get("role") == "system":
                system_text = str(m.get("content") or "").strip()
                break
        suffix_bits: list[str] = []
        if tools:
            suffix_bits.append(_DiffusionMixin._diffusion_tool_block(tools))
        suffix_bits.append(
            "Reply with ONLY your final answer or a tool call — no 'thought' "
            "preamble, reasoning, or narration."
        )
        pending_system = "\n\n".join(
            ([system_text] if system_text else []) + suffix_bits
        )
        # Map tool_call_id → tool name so a tool RESULT can be labeled with the
        # tool it came from (the model reads "Tool result (web_search):" and
        # knows it's its own search coming back, not a fresh user request).
        import json as _json
        id_to_name: dict[str, str] = {}
        for m in messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    tcid, nm = tc.get("id"), (tc.get("function") or {}).get("name")
                    if tcid and nm:
                        id_to_name[tcid] = nm

        parts: list[str] = []
        for m in messages:
            role = m.get("role")
            text = str(m.get("content") or "").strip()
            if role == "system":
                continue
            if role == "assistant":
                # CRITICAL: an assistant turn that ONLY made tool calls has
                # empty content. Skipping it (the old `not text: continue`)
                # left the next prompt as `user → user → model` — the model
                # saw a tool result with no record it had asked for one, and
                # the entropy-bound decoder denoised to EMPTY in ~2 steps
                # (E3107 on every multi-step agentic task). Render the tool
                # calls as a model turn, in the same plain-JSON shape the model
                # itself emits, so the conversation stays coherent.
                bits = [text] if text else []
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    nm = fn.get("name")
                    if not nm:
                        continue
                    try:
                        a = _json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        a = {}
                    bits.append(_json.dumps(
                        {"tool": nm, "args": a if isinstance(a, dict) else {}}
                    ))
                rendered = "\n".join(b for b in bits if b)
                if not rendered:
                    continue
                parts.append(f"<start_of_turn>model\n{rendered}<end_of_turn>\n")
                continue
            if role == "tool":
                # Tool results are fed back as a labeled user turn (Gemma has
                # no dedicated tool role in this hand-applied template).
                nm = id_to_name.get(m.get("tool_call_id"))
                label = f"Tool result ({nm}):" if nm else "Tool result:"
                parts.append(
                    f"<start_of_turn>user\n{label}\n{text or '(no output)'}<end_of_turn>\n"
                )
                continue
            # user
            if not text:
                continue
            if pending_system:
                text = f"{pending_system}\n\n{text}"
                pending_system = ""
            parts.append(f"<start_of_turn>user\n{text}<end_of_turn>\n")
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
        # Every `{"tool"...}` region we locate is a tool-call ATTEMPT, not
        # prose — strip it from the visible text whether or not it parses, so
        # a malformed/truncated call (the model's stray `}` / JSON fragments)
        # never leaks into the chat. `calls` only gets the ones that parse.
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
            spans.append((j, k))  # strip the attempt from visible regardless
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
                        _DiffusionMixin._repair_diffusion_json(blob)
                    )
                except Exception:
                    obj = None
            if isinstance(obj, dict):
                name = obj.get("tool")
                args = obj.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                if name:
                    calls.append({
                        "id": f"diff_{len(calls)}",
                        "type": "function",
                        # arguments MUST be a JSON STRING, not a dict — the
                        # agent loop (and OpenAI/Ollama convention) does
                        # json.loads(arguments) everywhere. Emitting a dict
                        # crashed with "the JSON object must be str... not
                        # dict" (E9001) on the first diffusion tool call.
                        "function": {"name": name, "arguments": _json.dumps(args)},
                    })
            i = k if k > j else j + 1
        # Remove every located tool-call region from the visible text.
        if spans:
            out = []
            last = 0
            for s, e in spans:
                out.append(text[last:s])
                last = e
            out.append(text[last:])
            text = "".join(out)
        # Strip orphaned structural braces left by tool-call scaffolding — the
        # model sometimes wraps its plan in braces or leaves a dangling `}`
        # next to a stripped call (the "...directory.}" leak). Only when braces
        # are unbalanced, and only at the edges, so balanced braces in real
        # code/prose are untouched.
        opens, closes = text.count("{"), text.count("}")
        while closes > opens and text.rstrip().endswith("}"):
            text = text.rstrip()[:-1]
            closes -= 1
        while opens > closes and text.lstrip().startswith("{"):
            text = text.lstrip()[1:]
            opens -= 1
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

        # Valid JSON string-escape initiators. A backslash inside a string
        # followed by anything else is INVALID JSON — `json.loads` rejects
        # the whole blob. DiffusionGemma routinely emits these in long
        # `content` values (observed: `\ ` — a stray backslash before a
        # space, inside the text of a write_file payload). The book/code it
        # was asked to write parsed perfectly except for one bad escape,
        # which dropped the entire tool call → empty turn → E3107. We
        # repair by ESCAPING the stray backslash (`\` → `\\`, a literal
        # backslash) rather than dropping it, so a real backslash in a
        # Windows path (`C:\Users` → emitted `\U`, also invalid) survives.
        _VALID_ESC = set('"\\/bfnrtu')

        def _is_valid_escape(at: int) -> bool:
            """True if blob[at] is a backslash beginning a valid JSON escape."""
            nxt = blob[at + 1] if at + 1 < n else ""
            if nxt not in _VALID_ESC:
                return False
            if nxt == "u":
                # \u must be followed by exactly 4 hex digits.
                hexpart = blob[at + 2 : at + 6]
                return len(hexpart) == 4 and all(
                    ch in "0123456789abcdefABCDEF" for ch in hexpart
                )
            return True

        out: list[str] = []
        n = len(blob)
        i = 0
        in_str = False
        while i < n:
            c = blob[i]
            if in_str:
                if c == "\\":
                    if _is_valid_escape(i):
                        # Keep the escape sequence verbatim (copy both chars).
                        out.append(blob[i : i + 2])
                        i += 2
                        continue
                    # Stray/invalid backslash — escape it to a literal one.
                    out.append("\\\\")
                    i += 1
                    continue
                out.append(c)
                if c == '"':
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

    def _run_diffusion_cli(self, cmd: list[str], timeout_secs: int) -> str:
        """Run the one-shot diffusion CLI once and return its full stdout.

        Block-diffusion isn't token-streamed — the runner denoises a whole
        canvas and prints it at once — so we collect all stdout, then clean it
        once. stderr is drained CONCURRENTLY: reading only stdout while stderr
        is a full PIPE deadlocks (llama.cpp is chatty on stderr during load).
        """
        from .runtime import RuntimeErrorWithContext
        import subprocess
        import threading as _threading
        import time as _time
        deadline = _time.monotonic() + timeout_secs
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace",
        )
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

        # Watchdog: `proc.stdout.read(4096)` below BLOCKS until 4096 bytes or
        # EOF, so a silent/wedged child (e.g. hung loading the ~15 GB model)
        # would never let the in-loop deadline check run — the turn hung on
        # "thinking…" forever. Enforce the deadline from a timer thread that
        # kills the process; the kill closes the pipe, read() returns "" (EOF),
        # and the loop exits. `_timed_out` distinguishes it from a clean finish.
        _timed_out = {"v": False}

        def _watchdog() -> None:
            while proc.poll() is None:
                if _time.monotonic() > deadline:
                    _timed_out["v"] = True
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return
                _time.sleep(0.25)

        _threading.Thread(target=_watchdog, daemon=True,
                          name="diffusion-watchdog").start()
        raw_parts: list[str] = []
        try:
            assert proc.stdout is not None
            while True:
                if _timed_out["v"] or _time.monotonic() > deadline:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    raise RuntimeErrorWithContext(
                        f"diffusion generation timed out ({timeout_secs}s)"
                    )
                chunk = proc.stdout.read(4096)
                if not chunk:
                    if _timed_out["v"]:
                        raise RuntimeErrorWithContext(
                            f"diffusion generation timed out ({timeout_secs}s)"
                        )
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
        return "".join(raw_parts)

    @staticmethod
    def _diffusion_turn_usable(
        text: str, tool_calls: list, tools: list | None
    ) -> bool:
        """Whether a diffusion turn is good enough to surface (vs re-sample).

        Usable = real visible text, OR tool calls whose REQUIRED args are all
        present. A tool call missing a required arg (e.g. bash with no
        `command`) means the canvas truncated the call — re-sampling usually
        produces a complete one, so treat it as unusable.
        """
        import json as _json
        if not tool_calls:
            return bool(text.strip())
        required_by_name: dict[str, list] = {}
        for t in (tools or []):
            fn = t.get("function", t) if isinstance(t, dict) else {}
            name = fn.get("name")
            if name:
                required_by_name[name] = (
                    fn.get("parameters") or {}
                ).get("required", []) or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                args = _json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            if any(r not in args for r in required_by_name.get(fn.get("name"), [])):
                return False
        return True

    @staticmethod
    def _diffusion_stats(raw: str) -> dict[str, Any]:
        """Parse the runner's own stats line from stdout.

        The CLI prints e.g. `... (33 steps over 1 blocks, entropy-bound)` and
        `throughput: 48.0 tok/s (...)`. We surface steps/blocks/tok_s into
        telemetry — a 2-step turn is the entropy-bound-empty signature, so
        this is how we DETECT (not guess) the empty-denoise in the wild.
        """
        import re as _re
        st: dict[str, Any] = {"steps": None, "blocks": None, "tok_s": None}
        m = _re.search(r"\((\d+)\s+steps over\s+(\d+)\s+blocks", raw)
        if m:
            st["steps"], st["blocks"] = int(m.group(1)), int(m.group(2))
        t = _re.search(r"throughput:\s*([\d.]+)\s*tok/s", raw)
        if t:
            st["tok_s"] = float(t.group(1))
        return st

    def _log_diffusion_telemetry(
        self,
        *,
        model_path: str,
        prompt: str,
        n_canvas: int,
        num_predict: int | None,
        tools: list[dict[str, Any]] | None,
        attempts: list[dict],
        final_text: str,
        final_tool_calls: list,
    ) -> None:
        """Append one JSONL line per diffusion turn — the real telemetry for
        what actually happened (prompt size, per-attempt eb mode + steps +
        tok/s, retries, final outcome). Best-effort; never breaks a turn."""
        try:
            import json as _json
            import os as _os
            from pathlib import Path as _Path
            outcome = (
                "tool" if final_tool_calls
                else ("text" if final_text.strip() else "empty_e3107")
            )
            rec = {
                "model": _Path(model_path).name,
                "prompt_chars": len(prompt),
                "prompt_tokens_est": len(prompt) // 4,
                "n_canvas": n_canvas,
                "num_predict": num_predict,
                "n_tools_offered": len(tools or []),
                "attempts": len(attempts),
                "attempts_detail": attempts,
                "recovered_on_retry": (
                    len(attempts) > 1 and bool(attempts) and attempts[-1].get("usable")
                ),
                "outcome": outcome,
                "final_clean_len": len(final_text.strip()),
            }
            path = _Path(_os.path.expanduser(
                "~/.local/share/localcode/diffusion_telemetry.jsonl"
            ))
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", errors="replace") as f:
                f.write(_json.dumps(rec) + "\n")
        except Exception:
            pass

    def _stream_diffusion_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        num_predict: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        from .runtime import RuntimeErrorWithContext
        from pathlib import Path as _Path

        binary = self._diffusion_cli_binary()
        if binary is None:
            # The runner ships in the wheel next to llama-server; nothing is
            # built or downloaded at runtime. Missing means a broken install.
            raise RuntimeErrorWithContext(
                "DiffusionGemma needs the bundled llama-diffusion-cli runner and it "
                "is missing from this install. Reinstall localcode."
            )

        model_path = str(_Path(self.config.model or "").expanduser())
        # DiffusionGemma chokes on the Gemma-4 special-token tool format
        # (<|tool_call>…<tool_call|>) — it collapses to near-empty output.
        # It DOES reliably emit a plain JSON tool call when asked in plain
        # text, so for the diffusion path we inject tools as plain JSON and
        # parse that (see _diffusion_tool_block / _parse_diffusion_tool_call).
        prompt = self._format_diffusion_prompt(messages, tools=tools)

        # llama-diffusion-cli generates block-autoregressively in 256-token
        # blocks; `-n` is the TOTAL token budget across blocks (verified:
        # -n 256 → 1 block ≈1K chars, -n 768 → 3 blocks, -n 2048 → ~8 blocks).
        # The agent loop passes num_predict=MAX_OUTPUT_TOKENS=-1; `-1 or 512`
        # is -1 (truthy!) → `-n -1` → degenerate, so treat non-positive as a
        # generous default. We DON'T pass --diffusion-blocks: it CAPS output
        # to one block regardless of -n, which truncated long turns mid-
        # reasoning before the tool call ever appeared (→ E3107). Without it,
        # -n alone controls length and the entropy-bound decoder still stops
        # early on short turns, so simple replies stay fast. 2048 leaves room
        # for a reasoning preamble AND a complete tool call on agentic turns.
        from .model_config import DIFFUSION_DEFAULT_CANVAS, DIFFUSION_MAX_CANVAS
        _n = int(num_predict) if (num_predict and int(num_predict) > 0) else DIFFUSION_DEFAULT_CANVAS
        _canvas = min(_n, DIFFUSION_MAX_CANVAS)

        cmd = [
            binary,
            "-m", model_path,
            "-p", prompt,
            "-no-cnv",   # GGUF ships a chat template; -no-cnv stops the CLI
                         # re-applying it on top of the prompt we built (double
                         # templating produced empty output).
            "-ngl", "99",
            "-n", str(_canvas),
        ]
        timeout_secs = max(60, int(self.config.request_timeout_seconds or 600))

        # Diffusion is NON-DETERMINISTIC in output SHAPE — one sample gives a
        # clean tool call, the next dumps a visible plan and a truncated
        # empty-args call. So generate, validate, and RETRY until the turn is
        # usable: real text, or tool calls whose required args are all present.
        #
        # ADAPTIVE retry (data-driven, see diffusion_telemetry.jsonl): the
        # FIRST attempt uses the entropy-bound decoder's fast `auto` mode
        # (early-stops on short turns → snappy simple replies). But on a large
        # prompt (a big tool result, a long spec) the entropy-bound decoder
        # CONFIDENTLY denoises the first block to EMPTY in ~2 steps — a
        # deterministic empty, so re-running the identical command (the old
        # behaviour) could NEVER recover it. Verified: `--diffusion-eb off`
        # forces full denoising and produces real content on the exact prompts
        # that `auto` empties. So every RETRY forces the entropy bound off.
        # Pre-empt the small-canvas failure: a prompt this large makes
        # DiffusionGemma denoise to empty / <unused> collapse even with the
        # eb-off retry (verified ~16K+ chars). Don't burn ~75s on 3 futile
        # retries — surface E3107 with the model-switch guidance immediately,
        # and record WHY in telemetry so the threshold is tunable from data.
        from .model_config import DIFFUSION_PROMPT_CHAR_LIMIT as _DIFFUSION_PROMPT_CHAR_LIMIT
        if len(prompt) > _DIFFUSION_PROMPT_CHAR_LIMIT:
            self._log_diffusion_telemetry(
                model_path=model_path, prompt=prompt, n_canvas=_canvas,
                num_predict=num_predict, tools=tools,
                attempts=[{"eb": "skipped", "reason": "prompt_over_limit",
                           "prompt_chars": len(prompt),
                           "limit": _DIFFUSION_PROMPT_CHAR_LIMIT}],
                final_text="", final_tool_calls=[],
            )
            from .errors import LocalCodeError, by_code, format_for_user
            _msg = format_for_user(LocalCodeError(by_code("E3107")))
            for i in range(0, len(_msg), 160):
                yield {"type": "content", "content": _msg[i:i + 160]}
            yield {
                "type": "stream_done", "finish_reason": "stop",
                "content_chars": len(_msg),
                "completion_tokens": max(1, len(_msg) // 4),
                "total_tokens": max(1, len(_msg) // 4), "usage_estimated": True,
            }
            return

        text = ""
        tool_calls: list = []
        _raw_joined = ""
        _attempts_meta: list[dict] = []
        for _attempt in range(3):
            _cmd = cmd if _attempt == 0 else cmd + ["--diffusion-eb", "off"]
            _raw_joined = self._run_diffusion_cli(_cmd, timeout_secs)
            text = self._clean_diffusion_output(_raw_joined, prompt)
            tool_calls = []
            if tools:
                tool_calls, text = self._parse_diffusion_tool_calls(text)
            _usable = self._diffusion_turn_usable(text, tool_calls, tools)
            _attempts_meta.append({
                "eb": "auto" if _attempt == 0 else "off",
                **self._diffusion_stats(_raw_joined),
                "raw_len": len(_raw_joined),
                "clean_len": len(text.strip()),
                "tool_calls": len(tool_calls),
                "usable": _usable,
            })
            if _usable:
                break

        # Persistent telemetry — one JSONL line per diffusion turn so we can
        # SEE what actually happens (steps, blocks, tok/s, retries, outcome)
        # instead of guessing. Inspect with:
        #   tail -f ~/.local/share/localcode/diffusion_telemetry.jsonl
        self._log_diffusion_telemetry(
            model_path=model_path,
            prompt=prompt,
            n_canvas=_canvas,
            num_predict=num_predict,
            tools=tools,
            attempts=_attempts_meta,
            final_text=text,
            final_tool_calls=tool_calls,
        )

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

        # (tool_calls + text were already parsed/cleaned in the retry loop
        # above — DiffusionGemma's plain-JSON calls are surfaced as a
        # tool_calls event with the scaffolding stripped from the content.)
        if not text.strip() and not tool_calls:
            from .errors import LocalCodeError, by_code, format_for_user
            text = format_for_user(LocalCodeError(by_code("E3107")))
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
            # Strip Gemma collapse tokens (<unused42>, [multimodal], <eos>).
            # On a large prompt the entropy-bound-off retry can collapse into
            # a stream of these — non-empty, so it was surfaced as "text"
            # (the `<unused26><unused27>…` soup). Removing them here means a
            # collapse-only turn cleans to EMPTY → unusable → honest E3107
            # instead of garbage in the chat. (Same regex the HTTP path uses.)
            s = _re.sub(r"<unused\d+>|\[multimodal\]|<eos>", "", s)
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
