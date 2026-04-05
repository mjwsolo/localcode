from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelTagRecommendation:
    model_tag: str
    backend: str
    quant_preset: str
    model_id_field: str
    note: str


def recommend_for_model_tag(model_tag: str) -> ModelTagRecommendation:
    lowered = model_tag.lower()
    backend = "ollama"
    quant_preset = "balanced"
    model_id_field = "model"
    note = "No special recommendation."
    if ".gguf" in lowered or any(token in lowered for token in ("q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0")):
        backend = "llama_cpp"
        model_id_field = "model"
        if "q2_" in lowered or "q3_" in lowered:
            quant_preset = "smallest"
            note = "Very low memory, but expect visible quality loss."
        elif "q4_" in lowered:
            quant_preset = "fastest"
            note = "Best small-machine default for speed and memory."
        elif "q5_" in lowered or "q6_" in lowered:
            quant_preset = "balanced"
            note = "Good tradeoff between quality and speed."
        elif "q8_" in lowered:
            quant_preset = "best"
            note = "High quality but much heavier on RAM/VRAM."
    elif "awq" in lowered:
        backend = "huggingface-local"
        model_id_field = "huggingface_model_id"
        quant_preset = "balanced"
        note = "AWQ is a strong advanced-user choice for local GPU inference."
    elif "gptq" in lowered:
        backend = "huggingface-local"
        model_id_field = "huggingface_model_id"
        quant_preset = "balanced"
        note = "GPTQ can work well locally, but backend support varies more than GGUF."
    elif "mlx" in lowered:
        backend = "mlx-local"
        model_id_field = "mlx_model_id"
        quant_preset = "fastest"
        note = "MLX quantized variants are the preferred Apple Silicon local path."
    return ModelTagRecommendation(model_tag=model_tag, backend=backend, quant_preset=quant_preset, model_id_field=model_id_field, note=note)
