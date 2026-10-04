"""Every flag localcode passes to llama-server must exist in the bundled binary.

The 2026-10 llama.cpp bump removed `--mmap` upstream. The model gate launches
the server with its own short flag list, so it stayed green while every real
launch from localcode would have died with "invalid argument". This test asks
the shipped binary itself, so a renamed or removed flag fails here first.
"""
from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parents[1] / "src" / "localcode" / "bin" / "llama-server"


def _help_flags() -> set[str]:
    out = subprocess.run([str(BIN), "--help"], capture_output=True, text=True, timeout=60)
    text = out.stdout + out.stderr
    # option lines look like: "-lm,   --load-mode MODE   description"
    flags: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("-"):
            continue
        head = re.split(r"\s{2,}", line.strip(), maxsplit=2)
        for part in head[:2]:
            for tok in part.replace(",", " ").split():
                if re.fullmatch(r"--?[A-Za-z][A-Za-z0-9-]*", tok):
                    flags.add(tok)
    return flags


def _emitted(cmd: list[str]) -> list[str]:
    return [a for a in cmd[1:] if re.fullmatch(r"--?[A-Za-z][A-Za-z0-9-]*", a)]


pytestmark = pytest.mark.skipif(
    not (platform.system() == "Darwin" and platform.machine() == "arm64" and BIN.is_file()),
    reason="the bundled llama-server only runs on Apple Silicon",
)


@pytest.mark.parametrize("mode", ["turbo", "turbo-think", "context", "speed"])
def test_every_emitted_flag_is_accepted_by_the_bundled_binary(mode: str) -> None:
    from localcode.config import RuntimeConfig
    from localcode.runtime import LocalCodeRuntimeGateway

    known = _help_flags()
    assert "--ctx-size" in known, "could not parse llama-server --help"

    cfg = RuntimeConfig(provider="llama_cpp", base_url="http://127.0.0.1:8081", model="model.gguf")
    cfg.laptop_26b_runtime_mode = mode
    cmd = LocalCodeRuntimeGateway(cfg).llama_server_command("/path/model.gguf", 8081)
    # flags the UI launcher appends on top of the gateway command (ui/server_cmd.py)
    cmd += ["--host", "--api-key", "--alias", "--reasoning", "--reasoning-budget", "--mmproj", "--slot-save-path"]

    unknown = sorted({f for f in _emitted(cmd) if f not in known})
    assert not unknown, f"llama-server no longer accepts: {unknown}"
