"""Model catalog — single source of truth for available models.

Every entry must be transparent: name, source, exact filename, size, license,
where it'll be downloaded, and any caveats. The picker UI reads from this list
so users see exactly what they're choosing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_DIR = Path.home() / ".local" / "share" / "localcode" / "models"


def model_dir() -> Path:
    """Resolve where GGUFs should live. Honors `runtime.model_dir` in config
    when set (expanded + normalized); otherwise falls back to the default
    under `~/.local/share/localcode/models`.
    """
    try:
        from .config import load_config
        raw = (load_config().runtime.model_dir or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    except Exception:
        pass
    return DEFAULT_MODEL_DIR


# Backwards-compat name for code that imports the constant. Lookups on this
# object still resolve through `model_dir()`, not the module-load snapshot.
# Most callers should use `model_dir()` or `ModelChoice.local_path` directly.
CANONICAL_MODEL_DIR = DEFAULT_MODEL_DIR


@dataclass(frozen=True)
class ModelChoice:
    key: str               # short id used in config (e.g. "gemma", "qwen")
    name: str              # display name
    hf_repo: str           # HuggingFace repo (e.g. "unsloth/gemma-4-26B-A4B-it-GGUF")
    filename: str          # exact GGUF filename in the repo
    size_gb: float         # download size
    active_params: str     # active parameter count (for MoE)
    architecture: str      # architecture identifier (matches llama.cpp arch enum)
    license: str           # license tag
    # HumanEval pass@1 measured ON THIS STACK (quantized weights + our
    # TurboQuant KV cache + llama-server path). NOT the paper/upstream
    # number. None when we haven't measured. The earlier field
    # `swe_bench_verified` was the paper number of the full-precision
    # upstream model — removed because quoting it next to our config
    # was misleading.
    humaneval_pass_at_1: float | None
    notes: str             # honest caveats / fit info

    @property
    def hf_url(self) -> str:
        return f"https://huggingface.co/{self.hf_repo}/blob/main/{self.filename}"

    @property
    def local_path(self) -> Path:
        return model_dir() / self.filename


CHOICES: list[ModelChoice] = [
    ModelChoice(
        key="gemma",
        name="Gemma 4 26B-A4B (Unsloth UD-IQ3_S)",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        filename="gemma-4-26B-A4B-it-UD-IQ3_S.gguf",
        size_gb=11.2,
        active_params="3.8B (top-8 of 128 experts)",
        architecture="gemma4-iswa",
        license="Gemma (Google) — research + commercial w/ attribution",
        humaneval_pass_at_1=0.951,  # 156/164 measured on this stack 2026-04-22
        notes="Comfortable fit on 16 GB Mac. Has Apr-11 chat template + Apr-8 tokenizer fixes. Tool calling via Gemma 4 native format. Measured 95.1% HumanEval pass@1 at IQ3_S + TurboQuant KV on our harness.",
    ),
    ModelChoice(
        key="qwen",
        name="Qwen 3.6 35B-A3B (Unsloth UD-IQ2_M)",
        hf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        filename="Qwen3.6-35B-A3B-UD-IQ2_M.gguf",
        size_gb=10.7,
        active_params="3.0B (top-8 + 1 shared of 256 experts)",
        architecture="qwen35moe",
        license="Apache 2.0",
        humaneval_pass_at_1=0.947,  # 126/133 measured on this stack 2026-04-22 (stopped early at 81%)
        notes="Full GPU offload on 16 GB M4. Hybrid attn + Mamba-2 SSM. Native 262K context. Requires the multi-region mmap patch (llama-cpp-turboquant commit 3d66675b8).",
    ),
]


def by_key(key: str) -> ModelChoice | None:
    for c in CHOICES:
        if c.key == key:
            return c
    return None


def by_filename(filename: str) -> ModelChoice | None:
    for c in CHOICES:
        if c.filename == filename:
            return c
    return None


def current(config) -> ModelChoice | None:
    """Resolve the currently selected model from config."""
    model_path = (config.runtime.model or "").strip()
    if not model_path:
        return None
    name = Path(model_path).name
    return by_filename(name)


def format_choice_long(c: ModelChoice, *, downloaded: bool, current_marker: bool = False) -> str:
    """Multi-line formatted description for the picker UI."""
    marker = " (current)" if current_marker else ""
    status = "downloaded" if downloaded else f"will download {c.size_gb:.1f} GB"
    if c.humaneval_pass_at_1 is not None:
        bench = f"{c.humaneval_pass_at_1*100:.1f}% HumanEval pass@1 (measured on this stack)"
    else:
        bench = "no benchmark (untested on this stack)"
    # Warn if this model is too big for the current Mac's RAM budget.
    # Short inline warning; the picker screen also dims models with a
    # bad fit so users can see at a glance.
    try:
        from .health import estimate_fit
        fits, reason = estimate_fit(c.size_gb)
    except Exception:
        fits, reason = True, ""
    fit_line = "" if fits else f"  ⚠ fit:      {reason}\n"
    return (
        f"[{c.key}] {c.name}{marker}\n"
        f"  source:    {c.hf_repo}\n"
        f"  url:       {c.hf_url}\n"
        f"  filename:  {c.filename}\n"
        f"  size:      {c.size_gb:.1f} GB ({status})\n"
        f"{fit_line}"
        f"  active:    {c.active_params}\n"
        f"  arch:      {c.architecture}\n"
        f"  license:   {c.license}\n"
        f"  benchmark: {bench}\n"
        f"  download path: {c.local_path}\n"
        f"  note:      {c.notes}"
    )
