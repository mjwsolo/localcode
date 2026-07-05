from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    key: str
    display_name: str
    default_model: str
    family: str
    feature_variant: str
    recommended_context_chars: int
    advertised_context_window: int
    summary: str
    supports_vision: bool = False
    supports_audio: bool = False
    supports_native_system: bool = False
    tool_strategy: str = "prompt"
    agent_steps: int = 3
    approval_style: str = "balanced"
    verification_bias: str = "balanced"
    retrieval_budget: int = 4
    context_strategy: str = "balanced"


GEMMA_PROFILES: dict[str, ModelProfile] = {
    "gemma4-e2b": ModelProfile(
        key="gemma4-e2b",
        display_name="Gemma 4 E2B",
        default_model="gemma4:e2b",
        family="gemma4",
        feature_variant="compact",
        recommended_context_chars=12000,
        advertised_context_window=128000,
        summary="Fastest edge profile for small machines and tight latency budgets.",
        supports_vision=True,
        supports_audio=True,
        supports_native_system=True,
        tool_strategy="native",
        agent_steps=2,
        approval_style="tight",
        verification_bias="fast",
        retrieval_budget=2,
        context_strategy="tight",
    ),
    "gemma4-e4b": ModelProfile(
        key="gemma4-e4b",
        display_name="Gemma 4 E4B",
        default_model="gemma4:e4b",
        family="gemma4",
        feature_variant="balanced",
        recommended_context_chars=20000,
        advertised_context_window=128000,
        summary="Best default for local coding on most laptops.",
        supports_vision=True,
        supports_audio=True,
        supports_native_system=True,
        tool_strategy="native",
        agent_steps=3,
        approval_style="balanced",
        verification_bias="balanced",
        retrieval_budget=4,
        context_strategy="balanced",
    ),
    "gemma4-12b": ModelProfile(
        key="gemma4-12b",
        display_name="Gemma 4 12B",
        default_model="gemma4:12b",
        family="gemma4",
        feature_variant="balanced",
        recommended_context_chars=24000,
        advertised_context_window=128000,
        summary="Mid-sized dense Gemma 4 for 16 GB Apple Silicon: stronger than E4B, native vision + audio.",
        supports_vision=True,
        supports_audio=True,
        supports_native_system=True,
        tool_strategy="native",
        agent_steps=3,
        approval_style="balanced",
        verification_bias="balanced",
        retrieval_budget=5,
        context_strategy="balanced",
    ),
    "gemma4-26b-moe": ModelProfile(
        key="gemma4-26b-moe",
        display_name="Gemma 4 26B MoE",
        default_model="gemma26b-iq3",
        family="gemma4",
        feature_variant="expanded",
        recommended_context_chars=36000,
        advertised_context_window=256000,
        summary="High-throughput reasoning profile for strong local workstations.",
        supports_vision=True,
        supports_native_system=True,
        tool_strategy="native",
        agent_steps=4,
        approval_style="balanced",
        verification_bias="thorough",
        retrieval_budget=6,
        context_strategy="broad",
    ),
    "gemma4-26b-laptop": ModelProfile(
        key="gemma4-26b-laptop",
        display_name="Gemma 4 26B Laptop",
        default_model="gemma26b-iq3",
        family="gemma4",
        feature_variant="balanced",
        recommended_context_chars=14000,
        advertised_context_window=256000,
        summary="Disciplined 26B profile for 16 GB Apple Silicon laptops: tight context, patch-first execution.",
        supports_vision=True,
        supports_native_system=True,
        tool_strategy="native",
        agent_steps=3,
        approval_style="balanced",
        verification_bias="balanced",
        retrieval_budget=4,
        context_strategy="tight",
    ),
    "gemma4-31b": ModelProfile(
        key="gemma4-31b",
        display_name="Gemma 4 31B Dense",
        default_model="gemma4:31b",
        family="gemma4",
        feature_variant="full",
        recommended_context_chars=50000,
        advertised_context_window=256000,
        summary="Highest-quality local Gemma 4 profile for serious coding sessions.",
        supports_vision=True,
        supports_native_system=True,
        tool_strategy="native",
        agent_steps=5,
        approval_style="rich",
        verification_bias="thorough",
        retrieval_budget=8,
        context_strategy="broad",
    ),
    "diffusiongemma-26b-moe": ModelProfile(
        key="diffusiongemma-26b-moe",
        display_name="DiffusionGemma 26B-A4B",
        default_model="diffusiongemma26b-q4",
        family="diffusiongemma",
        feature_variant="experimental",
        recommended_context_chars=24000,
        advertised_context_window=256000,
        summary="Experimental diffusion-generation Gemma model; faster block-wise decode potential, unmeasured locally.",
        tool_strategy="prompt",
        agent_steps=3,
        approval_style="balanced",
        verification_bias="thorough",
        retrieval_budget=5,
        context_strategy="balanced",
    ),
    "north-mini-code-30b-moe": ModelProfile(
        key="north-mini-code-30b-moe",
        display_name="North Mini Code 30B-A3B",
        default_model="north-mini-code-q4",
        family="cohere2_moe",
        feature_variant="coding",
        recommended_context_chars=32000,
        advertised_context_window=256000,
        summary="Cohere's open agentic coding MoE: 30B total, ~3B active, Apache 2.0.",
        tool_strategy="prompt",
        agent_steps=4,
        approval_style="balanced",
        verification_bias="thorough",
        retrieval_budget=6,
        context_strategy="broad",
    ),
}


ALIASES = {
    "e2b": "gemma4-e2b",
    "e4b": "gemma4-e4b",
    "12b": "gemma4-12b",
    "26b": "gemma4-26b-moe",
    "26b-laptop": "gemma4-26b-laptop",
    "26blaptop": "gemma4-26b-laptop",
    "26b-moe": "gemma4-26b-moe",
    "31b": "gemma4-31b",
    "diffusiongemma": "diffusiongemma-26b-moe",
    "diffusion-gemma": "diffusiongemma-26b-moe",
    "diffusiongemma26b": "diffusiongemma-26b-moe",
    "diffusiongemma:26b": "diffusiongemma-26b-moe",
    "north-mini-code": "north-mini-code-30b-moe",
    "north-mini": "north-mini-code-30b-moe",
    "northminicode": "north-mini-code-30b-moe",
    "north-mini-code:30b": "north-mini-code-30b-moe",
    "gemma4:e2b": "gemma4-e2b",
    "gemma4:e4b": "gemma4-e4b",
    "gemma4:12b": "gemma4-12b",
    "gemma4:26b": "gemma4-26b-moe",
    "gemma4:26b-moe": "gemma4-26b-moe",
    "gemma4:26b-laptop": "gemma4-26b-laptop",
    "gemma4:31b": "gemma4-31b",
}


def infer_profile_from_model(model_name: str) -> ModelProfile | None:
    name = model_name.lower().strip()
    if name in ALIASES:
        return GEMMA_PROFILES[ALIASES[name]]
    for profile in GEMMA_PROFILES.values():
        if profile.default_model in name or name in profile.default_model:
            return profile
    if "diffusiongemma" in name or "diffusion-gemma" in name:
        return GEMMA_PROFILES["diffusiongemma-26b-moe"]
    if "north-mini-code" in name or "north mini code" in name or "northminicode" in name:
        return GEMMA_PROFILES["north-mini-code-30b-moe"]
    if "31b" in name:
        return GEMMA_PROFILES["gemma4-31b"]
    if "12b" in name:
        return GEMMA_PROFILES["gemma4-12b"]
    if "26b-laptop" in name:
        return GEMMA_PROFILES["gemma4-26b-laptop"]
    if "26b" in name:
        return GEMMA_PROFILES["gemma4-26b-moe"]
    return None


def resolve_profile(profile_name: str | None, explicit_model: str | None) -> ModelProfile:
    normalized = (profile_name or "gemma4-e4b").strip().lower()
    if normalized in ALIASES:
        return GEMMA_PROFILES[ALIASES[normalized]]
    if normalized in GEMMA_PROFILES:
        return GEMMA_PROFILES[normalized]
    inferred = infer_profile_from_model(explicit_model or "")
    return inferred or GEMMA_PROFILES["gemma4-e4b"]


def get_runtime_model(profile: ModelProfile, explicit_model: str | None) -> str:
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()
    return profile.default_model
