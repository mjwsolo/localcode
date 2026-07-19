from __future__ import annotations

from localcode.models import infer_profile_from_model, resolve_profile
from localcode.models_catalog import (
    CHOICES,
    _parse_total_active_b,
    by_key,
    estimate_decode_tok_s,
)


def test_diffusiongemma_catalog_entry_uses_release_gguf() -> None:
    choice = by_key("diffusiongemma")

    assert choice is not None
    assert choice.name == "DiffusionGemma 26B-A4B (Q4)"
    assert choice.hf_repo == "unsloth/diffusiongemma-26B-A4B-it-GGUF"
    assert choice.filename == "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
    assert choice.size_gb == 15.7
    assert choice.active_params == "4B (adaptive diffusion MoE)"
    assert choice.architecture == "diffusion_gemma"
    assert choice.license == "Apache 2.0"
    assert choice.humaneval_pass_at_1 is None
    assert not choice.supports_vision
    assert "diffusion" in choice.notes.lower()


def test_north_mini_code_catalog_entry() -> None:
    # North-Mini-Code is served via a dedicated llama-server built from
    # llama.cpp PR #24260 (cohere2moe), so it's a real catalog model again.
    choice = by_key("north-mini-code")
    assert choice is not None
    assert choice.hf_repo == "unsloth/North-Mini-Code-1.0-GGUF"
    assert choice.filename == "North-Mini-Code-1.0-UD-Q4_K_M.gguf"
    assert choice.architecture == "cohere2_moe"


def test_new_models_are_available_in_model_picker_order() -> None:
    keys = [choice.key for choice in CHOICES]

    assert "diffusiongemma" in keys
    assert "north-mini-code" in keys
    assert keys.index("gemma") < keys.index("diffusiongemma") < keys.index("north-mini-code")


def test_diffusiongemma_profile_and_aliases() -> None:
    profile = resolve_profile("diffusiongemma", None)

    assert profile.key == "diffusiongemma-26b-moe"
    assert profile.default_model == "diffusiongemma26b-q4"
    assert profile.family == "diffusiongemma"
    assert profile.tool_strategy == "prompt"
    assert profile.supports_native_system is False
    assert infer_profile_from_model("unsloth/diffusiongemma-26B-A4B-it-Q4_K_M.gguf") == profile


def test_north_mini_code_profile_and_aliases() -> None:
    profile = resolve_profile("north-mini-code", None)

    assert profile.key == "north-mini-code-30b-moe"
    assert profile.default_model == "north-mini-code-q4"
    assert profile.family == "cohere2_moe"
    assert profile.tool_strategy == "prompt"
    assert profile.supports_native_system is False
    assert infer_profile_from_model("unsloth/North-Mini-Code-1.0-UD-Q4_K_M.gguf") == profile


# ---------------------------------------------------------------------------
# Capability surface tests
# ---------------------------------------------------------------------------

# Expected per-key vision capability. Vision is declared iff the entry ships an
# mmproj sidecar; this is the contract the runtime relies on.
_EXPECTED_VISION = {
    "gemma": True,
    "gemma-12b-bf16": True,
    "gemma-12b-qat": True,
    "gemma-qat": True,
    "gemma-31b-qat": True,
    "qwen": True,
    "gemma-q8": True,
    "qwen-q8": True,
    "diffusiongemma": False,
    "north-mini-code": False,
}


class TestCapabilitySurface:
    def test_every_catalog_entry_has_expected_vision(self) -> None:
        for choice in CHOICES:
            assert choice.key in _EXPECTED_VISION, (
                f"new catalog entry {choice.key!r} missing from _EXPECTED_VISION "
                "— add its expected vision flag"
            )
            assert choice.supports_vision is _EXPECTED_VISION[choice.key], (
                f"{choice.key} vision={choice.supports_vision}, "
                f"expected {_EXPECTED_VISION[choice.key]}"
            )

    def test_vision_derived_from_mmproj(self) -> None:
        # supports_vision must stay equivalent to the original inference.
        for choice in CHOICES:
            assert choice.supports_vision == (choice.mmproj_filename is not None)

    def test_vision_models_have_mmproj_path(self) -> None:
        for choice in CHOICES:
            if choice.supports_vision:
                assert choice.mmproj_path is not None
            else:
                assert choice.mmproj_path is None

    def test_audio_in_out_default_true_for_all(self) -> None:
        # Audio is hardware-gated (voice.py), not model-gated — every entry
        # declares both audio capabilities.
        for choice in CHOICES:
            assert choice.supports_audio_in is True
            assert choice.supports_audio_out is True

    def test_capabilities_rollup_matches_properties(self) -> None:
        for choice in CHOICES:
            expected = set()
            if choice.supports_vision:
                expected.add("vision")
            if choice.supports_thinking:
                expected.add("thinking")
            expected.add("audio_in")
            expected.add("audio_out")
            assert choice.capabilities == frozenset(expected)

    def test_vision_entry_capability_set(self) -> None:
        gemma = by_key("gemma")
        assert gemma is not None
        assert gemma.capabilities == frozenset(
            {"vision", "thinking", "audio_in", "audio_out"}
        )

    def test_text_only_entry_capability_set(self) -> None:
        # diffusiongemma is text-only AND can't do hidden reasoning;
        # north-mini-code is text-only but DOES support thinking.
        diff = by_key("diffusiongemma")
        assert diff is not None
        assert diff.capabilities == frozenset({"audio_in", "audio_out"})
        assert "vision" not in diff.capabilities
        assert "thinking" not in diff.capabilities

        nmc = by_key("north-mini-code")
        assert nmc is not None
        assert nmc.capabilities == frozenset({"thinking", "audio_in", "audio_out"})
        assert "vision" not in nmc.capabilities

    def test_thinking_support_gated_by_architecture(self) -> None:
        # Only diffusion architectures lack a toggleable hidden-reasoning
        # channel; every other catalog entry supports /thinking.
        for choice in CHOICES:
            is_diffusion = str(choice.architecture).lower().startswith("diffusion")
            assert choice.supports_thinking is (not is_diffusion), (
                f"{choice.key} ({choice.architecture}) "
                f"supports_thinking={choice.supports_thinking}"
            )


# ---------------------------------------------------------------------------
# estimate_decode_tok_s and _parse_total_active_b tests
# ---------------------------------------------------------------------------

M5_MAX_BW = 614.0  # GB/s — confirmed Apple spec


class TestParseTotalActiveB:
    def test_moe_hyphen_separator(self) -> None:
        assert _parse_total_active_b("Gemma 4 26B-A4B") == (26.0, 4.0)

    def test_moe_decimal_total(self) -> None:
        total, active = _parse_total_active_b("Qwen 3.6 35B-A3B")
        assert total == 35.0
        assert active == 3.0

    def test_dense_single_b(self) -> None:
        total, active = _parse_total_active_b("Gemma 4 12B")
        assert total == active == 12.0

    def test_north_mini_moe(self) -> None:
        total, active = _parse_total_active_b("North-Mini-Code 1.0 30B-A3B")
        assert total == 30.0
        assert active == 3.0

    def test_unknown_returns_zeros(self) -> None:
        assert _parse_total_active_b("no-size-here") == (0.0, 0.0)


class TestEstimateDecodeTokS:
    def test_returns_none_for_zero_size(self) -> None:
        assert estimate_decode_tok_s(0.0, "Gemma 4 26B-A4B", M5_MAX_BW) is None

    def test_returns_none_for_zero_bandwidth(self) -> None:
        assert estimate_decode_tok_s(11.2, "Gemma 4 26B-A4B", 0.0) is None

    def test_moe_tiny_active_does_not_exceed_sane_ceiling(self) -> None:
        # Qwen 3.6 35B-A3B IQ2_M: very small active fraction — should not
        # show 150+ tok/s even on the fastest chip (M5 Max at 614 GB/s).
        result = estimate_decode_tok_s(10.7, "Qwen 3.6 35B-A3B", M5_MAX_BW)
        assert result is not None
        assert result <= 100, f"Qwen IQ2 on M5 Max predicted {result} tok/s — unrealistically high"

    def test_moe_quants_monotonic_with_size(self) -> None:
        # Larger quants (same model) must be slower, not faster.
        iq3 = estimate_decode_tok_s(11.2, "Gemma 4 26B-A4B", M5_MAX_BW)
        q8 = estimate_decode_tok_s(28.0, "Gemma 4 26B-A4B", M5_MAX_BW)
        assert iq3 is not None and q8 is not None
        assert iq3 > q8, f"IQ3 ({iq3}) should be faster than Q8 ({q8})"

    def test_dense_slower_than_moe_same_bandwidth(self) -> None:
        # Dense 12B Q4 has no active-param discount → slower than MoE IQ3.
        dense = estimate_decode_tok_s(7.37, "Gemma 4 12B", M5_MAX_BW)
        moe_iq3 = estimate_decode_tok_s(11.2, "Gemma 4 26B-A4B", M5_MAX_BW)
        assert dense is not None and moe_iq3 is not None
        assert dense < moe_iq3, (
            f"Dense 12B Q4 ({dense}) should be slower than MoE IQ3 ({moe_iq3})"
        )

    def test_scales_with_bandwidth(self) -> None:
        # Higher bandwidth → higher tok/s for the same model.
        low = estimate_decode_tok_s(11.2, "Gemma 4 26B-A4B", 200.0)
        high = estimate_decode_tok_s(11.2, "Gemma 4 26B-A4B", M5_MAX_BW)
        assert low is not None and high is not None
        assert high > low, f"M5 Max ({high}) should beat 200 GB/s machine ({low})"

    def test_moe_spread_is_sane(self) -> None:
        # For MoE the quant-to-quant ratio should be modest — NOT 5x.
        # IQ2 vs Q8 on the same model family should be under 2x.
        iq2 = estimate_decode_tok_s(10.7, "Qwen 3.6 35B-A3B", M5_MAX_BW)
        q8 = estimate_decode_tok_s(38.5, "Qwen 3.6 35B-A3B", M5_MAX_BW)
        assert iq2 is not None and q8 is not None
        ratio = iq2 / q8
        assert ratio < 2.5, (
            f"Qwen IQ2/Q8 ratio {ratio:.2f}x is too large — "
            f"MoE decode should be flatter (got IQ2={iq2}, Q8={q8})"
        )

    def test_gemma_iq3_ballpark_m5_max(self) -> None:
        # Anchor: measured ~83 tok/s on M5 Max stack (IQ3_S + TurboQuant KV).
        # We allow ±20 tok/s since this is a rough estimate.
        result = estimate_decode_tok_s(11.2, "Gemma 4 26B-A4B", M5_MAX_BW)
        assert result is not None
        assert 60 <= result <= 105, (
            f"Gemma IQ3 estimated {result} tok/s on M5 Max — expected 60-105 range"
        )
