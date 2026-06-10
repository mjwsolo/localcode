from __future__ import annotations

from localcode.models import MLX_MODEL_IDS, infer_profile_from_model, resolve_profile
from localcode.models_catalog import CHOICES, by_key


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


def test_north_mini_code_catalog_entry_uses_release_gguf() -> None:
    choice = by_key("north-mini-code")

    assert choice is not None
    assert choice.name == "North Mini Code 30B-A3B (Q4)"
    assert choice.hf_repo == "unsloth/North-Mini-Code-1.0-GGUF"
    assert choice.filename == "North-Mini-Code-1.0-UD-Q4_K_M.gguf"
    assert choice.size_gb == 17.9
    assert choice.active_params == "3B active (30B total MoE)"
    assert choice.architecture == "cohere2_moe"
    assert choice.license == "Apache 2.0"
    assert choice.humaneval_pass_at_1 is None
    assert not choice.supports_vision
    assert "Artificial Analysis Coding Index" in choice.notes


def test_new_models_are_available_in_model_picker_order() -> None:
    keys = [choice.key for choice in CHOICES]

    assert "diffusiongemma" in keys
    assert "north-mini-code" in keys
    assert keys.index("gemma") < keys.index("diffusiongemma") < keys.index("north-mini-code") < keys.index("gemma-q8")


def test_diffusiongemma_profile_and_aliases() -> None:
    profile = resolve_profile("diffusiongemma", None)

    assert profile.key == "diffusiongemma-26b-moe"
    assert profile.default_model == "diffusiongemma26b-q4"
    assert profile.family == "diffusiongemma"
    assert profile.tool_strategy == "prompt"
    assert profile.supports_native_system is False
    assert infer_profile_from_model("unsloth/diffusiongemma-26B-A4B-it-Q4_K_M.gguf") == profile
    assert MLX_MODEL_IDS[profile.key] == "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def test_north_mini_code_profile_and_aliases() -> None:
    profile = resolve_profile("north-mini-code", None)

    assert profile.key == "north-mini-code-30b-moe"
    assert profile.default_model == "north-mini-code-q4"
    assert profile.family == "cohere2_moe"
    assert profile.tool_strategy == "prompt"
    assert profile.supports_native_system is False
    assert infer_profile_from_model("unsloth/North-Mini-Code-1.0-UD-Q4_K_M.gguf") == profile
    assert MLX_MODEL_IDS[profile.key] == "mlx-community/North-Mini-Code-1.0-4bit"
