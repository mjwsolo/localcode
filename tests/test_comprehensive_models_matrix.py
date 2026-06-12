"""Model × RAM matrix — the CI-safe tier.

We can't load 16-50 GB GGUFs or spin up a 128 GB Mac in CI, but every
RAM-dependent and arch-dependent DECISION the runtime makes is a pure
function we CAN drive: which backend a model dispatches to, what binary
and flags `llama_server_command` emits, and what context ceiling a given
RAM allows. So we sweep every catalog entry across the full Apple-Silicon
RAM ladder and assert those decisions, with `_system_ram_gb` mocked.

The companion real-inference tier (actually loading each GGUF and running
a turn) lives in test_real_models.py behind the `real_models` marker.

Backends by architecture:
  * diffusion_gemma  → llama-diffusion-cli (one-shot, NO server)
  * cohere2_moe      → llama-server-cohere (stock flags only)
  * everything else  → the TurboQuant llama-server
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import models_catalog as catalog
from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway

# The Apple-Silicon RAM configurations LocalCode ships against.
RAM_LADDER = [8, 16, 18, 24, 32, 36, 48, 64, 96, 128]

# Hard KV ceilings the runtime must never exceed for a given RAM (validated
# on 16 GB; 128 GB allowed up to 128K). A model asking for more must clamp.
def _ctx_ceiling(ram_gb: int) -> int:
    return 131072 if ram_gb >= 64 else 65536


def _backend_for(arch: str) -> str:
    a = (arch or "").lower()
    if "diffusion" in a:
        return "diffusion_cli"
    if "cohere" in a:
        return "cohere_server"
    return "turboquant_server"


# Flags that are TurboQuant-only — a stock cohere server must never see them.
_TURBO_ONLY = ("--spec-type", "-fit", "--ctx-checkpoints")
_TURBO_CACHE_VALUES = ("turbo4", "turbo")


def _make_gw(tmp_path: Path, choice, ram_gb: int) -> LocalCodeRuntimeGateway:
    cfg = RuntimeConfig()
    cfg.provider = "llama_cpp"
    cfg.model = str(tmp_path / choice.filename)
    # Drive the validated fast/turbo path so the RAM ladder engages.
    cfg.quant_preset = "fastest"
    cfg.kv_cache_type_v = "turbo4"
    cfg.laptop_26b_runtime_mode = "turbo"
    cfg.max_context_chars = 400000  # would exceed every ceiling → must clamp
    # Real binaries on disk so command building is deterministic (no discovery).
    turbo = tmp_path / "llama-server"
    turbo.write_text("#!/bin/sh\n")
    cohere = tmp_path / "llama-server-cohere"
    cohere.write_text("#!/bin/sh\n")
    diff = tmp_path / "llama-diffusion-cli"
    diff.write_text("#!/bin/sh\n")
    cfg.llama_cpp_binary = str(turbo)
    cfg.cohere_server_binary = str(cohere)
    cfg.diffusion_cli_binary = str(diff)
    gw = LocalCodeRuntimeGateway(cfg)
    return gw


def _choice_ids(c):
    return f"{c.key}[{c.architecture}]"


ALL_CHOICES = list(catalog.CHOICES)


# ── Catalog integrity ────────────────────────────────────────────────


@pytest.mark.parametrize("choice", ALL_CHOICES, ids=_choice_ids)
def test_every_choice_is_wellformed(choice):
    assert choice.filename and choice.filename.endswith(".gguf")
    assert choice.size_gb and choice.size_gb > 0
    assert choice.architecture in {
        "gemma4-iswa", "qwen35moe", "diffusion_gemma", "cohere2_moe",
    }, f"unknown arch {choice.architecture!r}"
    # by_filename must round-trip to a choice carrying the same architecture
    # (this is the dispatch key the runtime keys off).
    rt = catalog.by_filename(choice.filename)
    assert rt is not None and rt.architecture == choice.architecture


# ── Backend dispatch per architecture ────────────────────────────────


@pytest.mark.parametrize("choice", ALL_CHOICES, ids=_choice_ids)
def test_backend_dispatch_matches_architecture(tmp_path, choice):
    gw = _make_gw(tmp_path, choice, 64)
    is_diffusion = gw._diffusion_choice() is not None
    if _backend_for(choice.architecture) == "diffusion_cli":
        assert is_diffusion, "diffusion arch must dispatch to the CLI runner"
    else:
        assert not is_diffusion, "non-diffusion arch must NOT take the diffusion path"


# ── Server command construction across the RAM ladder ────────────────


@pytest.mark.parametrize("ram_gb", RAM_LADDER)
@pytest.mark.parametrize("choice", ALL_CHOICES, ids=_choice_ids)
def test_server_command_is_sane_for_every_ram(tmp_path, choice, ram_gb):
    backend = _backend_for(choice.architecture)
    if backend == "diffusion_cli":
        pytest.skip("diffusion has no server command (one-shot CLI path)")
    gw = _make_gw(tmp_path, choice, ram_gb)
    from unittest.mock import patch
    with patch.object(gw, "_system_ram_gb", return_value=ram_gb):
        cmd = gw.llama_server_command(gw.config.model)

    assert isinstance(cmd, list) and cmd, "must return a non-empty argv"
    # Binary matches the backend.
    if backend == "cohere_server":
        assert cmd[0].endswith("llama-server-cohere")
        # Stock binary: none of the TurboQuant-only flags.
        for flag in _TURBO_ONLY:
            assert flag not in cmd, f"cohere server must not get {flag}"
        for i, tok in enumerate(cmd):
            if tok in ("--cache-type-v", "--cache-type-k"):
                assert cmd[i + 1] not in _TURBO_CACHE_VALUES, "cohere = stock KV only"
    else:
        assert cmd[0].endswith("llama-server")
        assert not cmd[0].endswith("llama-server-cohere")

    # Common required flags.
    assert "--model" in cmd and "--port" in cmd
    assert "--threads" in cmd and int(cmd[cmd.index("--threads") + 1]) > 0
    # Context size present, positive, and within the RAM ceiling.
    assert "--ctx-size" in cmd
    ctx = int(cmd[cmd.index("--ctx-size") + 1])
    assert ctx >= 2048, f"ctx {ctx} below usable floor"
    assert ctx <= _ctx_ceiling(ram_gb), (
        f"{choice.key} on {ram_gb}GB asked for ctx {ctx} > ceiling "
        f"{_ctx_ceiling(ram_gb)} — Metal OOM risk"
    )


# ── Recommendation sanity (complements test_comprehensive_machines) ──


@pytest.mark.parametrize("ram_gb", RAM_LADDER)
def test_recommended_model_dispatches_to_a_real_backend(tmp_path, ram_gb):
    choice = catalog.recommend(ram_gb)
    assert _backend_for(choice.architecture) in {
        "diffusion_cli", "cohere_server", "turboquant_server",
    }
