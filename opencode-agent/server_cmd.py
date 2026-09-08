#!/usr/bin/env python3
"""Print the llama-server command localcode itself would launch on THIS machine
for a given GGUF: context size from the RAM tier ladder, KV-cache compression,
flash-attn, batch sizes, checkpoints. Nothing hardcoded in the front ends.

    server_cmd.py <gguf> <port> [alias]   -> one argument per line
    server_cmd.py --ctx <gguf>             -> just the context size
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from localcode.config import load_config  # noqa: E402
from localcode.runtime import LocalCodeRuntimeGateway  # noqa: E402


def server_command(gguf: str, port: int, alias: str | None = None) -> list[str]:
    cfg = load_config()
    g = LocalCodeRuntimeGateway(cfg.runtime)
    cmd = g.llama_server_command(gguf, port)
    cmd[1:1] = ["--host", "127.0.0.1"]
    if alias:
        cmd += ["--alias", alias]
    # Hidden thinking is a SERVER property, one switch for every front end and
    # every model. Per-wire hacks (chat_template_kwargs, reasoning effort
    # "none", --chat-template-kwargs) silently failed on Muse Glimmer; the
    # server flags worked on every wire. localcode's default is off.
    # Per-model policy from the catalog (models_catalog.ModelChoice.reasoning_control):
    #   "server" / "chat_template" -> the flags below silence it (gemma 4, Qwen 3.x)
    #   "always"                    -> no off switch exists (Muse Glimmer); don't send flags
    #   "none"                      -> no reasoning channel at all
    control = "server"
    try:
        from localcode.models_catalog import by_filename
        choice = by_filename(Path(gguf).name)
        if choice is not None:
            control = choice.reasoning_control or "server"
    except Exception:  # noqa: BLE001
        pass
    mode = os.environ.get("LOCALCODE_INTERNAL_THINKING_MODE") or cfg.runtime.internal_thinking_mode or "off"
    if control in ("server", "chat_template") and str(mode).strip().lower() != "on":
        cmd += ["--reasoning", "off", "--reasoning-budget", "0"]
    return cmd


def context_size(gguf: str) -> int:
    cmd = server_command(gguf, 0)
    return int(cmd[cmd.index("--ctx-size") + 1])


if __name__ == "__main__":
    if sys.argv[1] == "--ctx":
        print(context_size(sys.argv[2]))
    else:
        gguf, port = sys.argv[1], int(sys.argv[2])
        alias = sys.argv[3] if len(sys.argv) > 3 else None
        print("\n".join(server_command(gguf, port, alias)))
