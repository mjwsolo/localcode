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
def test_every_emitted_flag_is_accepted_by_the_bundled_binary(mode, monkeypatch, tmp_path):
    from localcode.config import AppConfig, RuntimeConfig, UIConfig
    from localcode.models_catalog import CHOICES
    from localcode.ui import server_cmd

    known = _help_flags()
    assert "--ctx-size" in known, "could not parse llama-server --help"
    cfg = AppConfig(runtime=RuntimeConfig(provider="llama_cpp"), ui=UIConfig())
    cfg.runtime.laptop_26b_runtime_mode = mode
    monkeypatch.setattr(server_cmd, "load_config", lambda: cfg)
    monkeypatch.setenv("LOCALCODE_SERVER_KEY", "test-key")
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(server_cmd, "mmproj_for", lambda gguf: tmp_path / "projector.gguf")
    for choice in CHOICES:
        cmd = server_cmd.server_command(str(tmp_path / choice.filename), 8081, choice.key)
        unknown = sorted({f for f in _emitted(cmd) if f not in known})
        assert not unknown, f"{choice.key}: llama-server no longer accepts: {unknown}"
