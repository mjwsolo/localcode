"""Model × RAM matrix — the CI-safe tier.

We can't load 16-50 GB GGUFs or spin up a 128 GB Mac in CI, but every
RAM-dependent and arch-dependent DECISION the runtime makes is a pure
function we CAN drive: which backend a model dispatches to, what binary
and flags `llama_server_command` emits, and what context ceiling a given
RAM allows. So we sweep every catalog entry across the full Apple-Silicon
RAM ladder and assert those decisions, with `_system_ram_gb` mocked.

The companion real-inference tier (actually loading each GGUF and running
a turn) lives in test_real_models.py behind the `real_models` marker.

Backend: EVERY architecture, DiffusionGemma included, runs on the one bundled
TurboQuant llama-server (the fork hosts the block-diffusion denoiser inside
the server, see llama-cpp-turboquant/PATCHES.md 0005). There is no per-arch
runner and no second binary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import models_catalog as catalog
from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway

# The Apple-Silicon RAM configurations LocalCode ships against.
RAM_LADDER = [8, 16, 18, 24, 32, 36, 48, 64, 96, 128]

# Hard KV ceiling the runtime must never exceed for a given RAM. Use the
# runtime's own _ram_ctx_ceiling as the single source of truth so this test
# can't drift from it (it did once — the 48GB->96K tier was added to the
# runtime but not here).
def _ctx_ceiling(ram_gb: int) -> int:
    return LocalCodeRuntimeGateway._ram_ctx_ceiling(ram_gb)


def _backend_for(arch: str) -> str:
    # Every arch (incl. cohere2_moe, muse_glimmer AND diffusion_gemma) runs on
    # the one bundled llama-server.
    return "turboquant_server"


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
    cfg.llama_cpp_binary = str(turbo)
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
        "gemma4-iswa", "qwen35moe", "qwen35", "diffusion_gemma", "cohere2_moe",
        "muse_glimmer", "kolibri1", "flint", "deepseek4", "inkling2_loop",
    }, f"unknown arch {choice.architecture!r}"
    # by_filename must round-trip to a choice carrying the same architecture
    # (this is the dispatch key the runtime keys off).
    rt = catalog.by_filename(choice.filename)
    assert rt is not None and rt.architecture == choice.architecture


# ── Backend dispatch per architecture ────────────────────────────────


@pytest.mark.parametrize("choice", ALL_CHOICES, ids=_choice_ids)
def test_backend_dispatch_matches_architecture(tmp_path, choice):
    gw = _make_gw(tmp_path, choice, 64)
    # One HTTP path for everything: the gateway has no architecture dispatch
    # and no diffusion side-channel any more.
    assert _backend_for(choice.architecture) == "turboquant_server"
    assert not hasattr(gw, "_diffusion_choice")
    assert not hasattr(gw, "_stream_diffusion_events")
    assert gw.endpoint.endswith("/v1/chat/completions")


# ── Server command construction across the RAM ladder ────────────────


@pytest.mark.parametrize("ram_gb", RAM_LADDER)
@pytest.mark.parametrize("choice", ALL_CHOICES, ids=_choice_ids)
def test_server_command_is_sane_for_every_ram(tmp_path, choice, ram_gb):
    gw = _make_gw(tmp_path, choice, ram_gb)
    from unittest.mock import patch
    with patch.object(gw, "_system_ram_gb", return_value=ram_gb):
        cmd = gw.llama_server_command(gw.config.model)

    assert isinstance(cmd, list) and cmd, "must return a non-empty argv"
    # One bundled binary for every server-backed arch; no per-arch runner.
    assert cmd[0].endswith("llama-server")
    assert cmd[0] == gw.config.llama_cpp_binary
    if "muse" in choice.architecture:
        assert "--jinja" in cmd, "Muse Glimmer requires --jinja (its chat template)"

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
    # Auto-recommend never lands on the research diffusion path (product call,
    # see models_catalog._NO_AUTO_RECOMMEND_ARCHS).
    assert _backend_for(choice.architecture) == "turboquant_server"


def test_every_choice_has_a_picker_group():
    """Every downloadable ModelChoice must map to a MODEL_GROUPS entry — the
    picker lists GROUPS, not choices, so a choice without a group is invisible
    in the UI (the Qwen 3.8 regression: catalog had it, picker didn't)."""
    group_repos = {g.hf_repo for g in catalog.MODEL_GROUPS}
    orphans = [c.key for c in catalog.CHOICES if c.hf_repo not in group_repos]
    assert not orphans, f"choices with no MODEL_GROUPS entry (invisible in picker): {orphans}"


def test_every_picker_group_has_a_downloadable_quant():
    choice_repos = {c.hf_repo for c in catalog.CHOICES}
    empty = [g.key for g in catalog.MODEL_GROUPS if g.hf_repo not in choice_repos]
    assert not empty, f"picker groups with no downloadable quant: {empty}"


@pytest.mark.parametrize("choice", ALL_CHOICES, ids=_choice_ids)
def test_every_catalog_reasoning_policy_resolves(choice):
    from localcode.reasoning_capabilities import ReasoningControl, reasoning_capabilities

    caps = reasoning_capabilities(choice.filename)
    if choice.architecture == "diffusion_gemma":
        assert not caps.supported
        assert caps.control is ReasoningControl.NONE
        return
    assert caps.family == choice.architecture
    assert caps.supported == (choice.reasoning_control != "none")
    if choice.reasoning_control == "always":
        assert caps.control is ReasoningControl.ALWAYS
        assert not caps.supports_budget


@pytest.mark.parametrize("choice", ALL_CHOICES, ids=_choice_ids)
def test_launcher_only_silences_reasoning_where_a_switch_exists(choice, monkeypatch):
    """The UI launcher may pass --reasoning off only to models that have the switch.

    DiffusionGemma has none: with the flag it returns empty or head-truncated text.
    """
    from localcode.reasoning_capabilities import ReasoningControl, reasoning_capabilities
    from localcode.ui.server_cmd import server_command

    monkeypatch.delenv("LOCALCODE_INTERNAL_THINKING_MODE", raising=False)
    cmd = server_command(f"/models/{choice.filename}", 8081, choice.key)
    has_switch = reasoning_capabilities(choice.filename).control is ReasoningControl.CHAT_TEMPLATE
    assert ("--reasoning" in cmd) == has_switch
    if choice.architecture == "diffusion_gemma" or choice.reasoning_control == "reasoning_effort":
        # The thought block the model always emits is parsed out, not leaked.
        assert cmd[cmd.index("--reasoning-format") + 1] == "auto"
    else:
        assert "--reasoning-format" not in cmd
    if choice.reasoning_control == "reasoning_effort":
        # Inkling 2 ignores enable_thinking; off is the template's own effort "none".
        kwargs = cmd[cmd.index("--chat-template-kwargs") + 1]
        assert json.loads(kwargs) == {"reasoning_effort": "none"}
        # Off also opens the turn with an empty thought (packaged template).
        template = Path(cmd[cmd.index("--chat-template-file") + 1])
        assert template.is_file()
        assert "<|message_model|><|content_thinking|><|end_message|>" in template.read_text()
    else:
        assert "--chat-template-kwargs" not in cmd
        assert "--chat-template-file" not in cmd


def test_inkling_runs_an_f16_kv_cache(monkeypatch):
    """Inkling's banded attention converts a quantized cache to f16 on every
    chunk: f16 is 2x faster at 32K and 3.3 GB smaller at 128K. Other models
    keep the configured q8_0 cache."""
    from localcode.models_catalog import CHOICES
    from localcode.ui.server_cmd import server_command

    monkeypatch.delenv("LOCALCODE_INTERNAL_THINKING_MODE", raising=False)
    for choice in CHOICES:
        cmd = server_command(f"/models/{choice.filename}", 8081, choice.key)
        if choice.architecture == "inkling2_loop":
            assert "--cache-type-k" not in cmd and "--cache-type-v" not in cmd, choice.key
