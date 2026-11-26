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


def test_streamed_expert_flags_are_accepted_by_the_bundled_binary(monkeypatch, tmp_path):
    """--moe-cache-mib and -ot exps=CPU, as added for a streamed MoE launch."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from gguf_builder import moe_tensors, write_gguf
    from localcode.config import AppConfig, RuntimeConfig, UIConfig
    from localcode.runtime import LocalCodeRuntimeGateway
    from localcode.ui import server_cmd

    kv = {"qwen3moe.expert_count": 256, "qwen3moe.expert_used_count": 8}
    gguf = write_gguf(tmp_path / "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf", moe_tensors(40, 200_000_000, dense_bytes=int(0.05e9)), kv)
    monkeypatch.setattr(server_cmd, "load_config", lambda: AppConfig(runtime=RuntimeConfig(provider="llama_cpp"), ui=UIConfig()))
    monkeypatch.setattr(server_cmd, "mmproj_for", lambda g: None)
    monkeypatch.setattr(LocalCodeRuntimeGateway, "_system_ram_gb", lambda _: 36)
    monkeypatch.setenv("LOCALCODE_RAM_GB", "36")
    monkeypatch.setenv("LOCALCODE_PROMPT_WARMUP", "0")
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path / "run"))
    cmd = server_cmd.server_command(str(gguf), 8081)
    assert "--moe-cache-mib" in cmd
    unknown = sorted({f for f in _emitted(cmd) if f not in _help_flags()})
    assert not unknown, f"bundled llama-server does not accept: {unknown}"
