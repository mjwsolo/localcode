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
    # Optional vision sidecar. When set, llama-server can be launched
    # with `--mmproj <path>` to enable image input. The mmproj is a
    # separate file from the text-decoder GGUF; it lives in the same
    # HF repo and the same model_dir on disk. Identical for all quants
    # of the same model family (the projector quality is independent
    # of the text-decoder quant).
    #
    # `mmproj_filename` is the LOCAL filename — must be unique across
    # model families so Gemma's mmproj and Qwen's mmproj don't overwrite
    # each other on download. `mmproj_hf_filename` is what HF calls it
    # in the repo (commonly just `mmproj-F16.gguf` for both families,
    # which is why we need the separate local name).
    mmproj_filename: str | None = None
    mmproj_size_gb: float = 0.0  # download size of the mmproj sidecar
    mmproj_hf_filename: str | None = None  # HF-side filename; defaults to mmproj_filename

    @property
    def hf_url(self) -> str:
        return f"https://huggingface.co/{self.hf_repo}/blob/main/{self.filename}"

    @property
    def local_path(self) -> Path:
        return model_dir() / self.filename

    @property
    def mmproj_path(self) -> Path | None:
        """Where the vision projector lives on disk (None if model has no
        vision sidecar)."""
        if not self.mmproj_filename:
            return None
        return model_dir() / self.mmproj_filename

    @property
    def supports_vision(self) -> bool:
        return self.mmproj_filename is not None


CHOICES: list[ModelChoice] = [
    ModelChoice(
        key="gemma",
        name="Gemma 4 26B-A4B (Q3)",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        filename="gemma-4-26B-A4B-it-UD-IQ3_S.gguf",
        size_gb=11.2,
        active_params="3.8B (top-8 of 128 experts)",
        architecture="gemma4-iswa",
        license="Gemma (Google) — research + commercial w/ attribution",
        humaneval_pass_at_1=0.951,  # 156/164 measured on this stack 2026-04-22
        notes="Comfortable fit on 16 GB Mac. Has Apr-11 chat template + Apr-8 tokenizer fixes. Tool calling via Gemma 4 native format. Measured 95.1% HumanEval pass@1 at IQ3_S + TurboQuant KV on our harness.",
        mmproj_filename="mmproj-gemma-4-26B-A4B-F16.gguf",
        mmproj_size_gb=1.2,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelChoice(
        key="gemma-12b",
        name="Gemma 4 12B (Q4)",
        hf_repo="unsloth/gemma-4-12b-it-GGUF",
        filename="gemma-4-12b-it-UD-Q4_K_XL.gguf",
        size_gb=7.37,
        active_params="12B (dense)",
        architecture="gemma4-iswa",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes=(
            "Mid-sized dense Gemma 4 — sits between E4B and the 26B MoE. "
            "UD-Q4_K_XL fits 16 GB unified memory (~7.4 GB weights + KV + mmproj). "
            "Native multimodal (vision + audio per Google's Gemma 4 announcement). "
            "Apache 2.0 licensed. Untested on this stack — no HumanEval number yet. "
            "TODO: confirm exact GGUF filename, arch enum, and mmproj filename/size against the repo before release."
        ),
        mmproj_filename="mmproj-gemma-4-12b-F16.gguf",
        mmproj_size_gb=0.86,  # 862 MB; repo offers BF16 + F16, both 862 MB
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelChoice(
        key="gemma-12b-bf16",
        name="Gemma 4 12B (BF16, full)",
        hf_repo="unsloth/gemma-4-12b-it-GGUF",
        filename="gemma-4-12b-it-BF16.gguf",
        size_gb=23.8,
        active_params="12B (dense)",
        architecture="gemma4-iswa",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes=(
            "Full-precision BF16 — reference quality, zero quantization loss (this is the 100% "
            "baseline every quant's '% of BF16' is measured against). ~24 GB weights → needs "
            "≥48 GB unified memory (sweet spot 64-128 GB Apple Silicon). 16/24 GB Macs CANNOT run "
            "this — use the [gemma-12b] Q4 entry instead. Same dense 12B as the Q4 entry, just "
            "unquantized. Untested on this stack. TODO: confirm exact BF16 filename (may be split into "
            "shards) + mmproj filename/size against the repo before release."
        ),
        mmproj_filename="mmproj-gemma-4-12b-F16.gguf",
        mmproj_size_gb=0.86,  # 862 MB; repo offers BF16 + F16, both 862 MB
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelChoice(
        key="qwen",
        name="Qwen 3.6 35B-A3B (Q2)",
        hf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        filename="Qwen3.6-35B-A3B-UD-IQ2_M.gguf",
        size_gb=10.7,
        active_params="3.0B (top-8 + 1 shared of 256 experts)",
        architecture="qwen35moe",
        license="Apache 2.0",
        humaneval_pass_at_1=0.947,  # 126/133 measured on this stack 2026-04-22 (stopped early at 81%)
        notes="Full GPU offload on 16 GB M4. Hybrid attn + Mamba-2 SSM. Native 262K context. Requires the multi-region mmap patch (llama-cpp-turboquant commit 3d66675b8).",
        mmproj_filename="mmproj-Qwen3.6-35B-A3B-F16.gguf",
        mmproj_size_gb=0.9,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelChoice(
        key="diffusiongemma",
        name="DiffusionGemma 26B-A4B (Q4)",
        hf_repo="unsloth/diffusiongemma-26B-A4B-it-GGUF",
        filename="diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
        size_gb=15.7,
        active_params="4B (adaptive diffusion MoE)",
        architecture="diffusion_gemma",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes=(
            "Google DiffusionGemma 26B-A4B instruction model — experimental diffusion/denoising "
            "text generation that can generate blocks in parallel instead of strictly token-by-token. "
            "Apache 2.0. GGUF Q4_K_M is ~15.7 GiB, so treat as a 32 GB+ unified-memory pick until "
            "this stack has measured load/runtime behavior. No LocalCode HumanEval number yet."
        ),
    ),
    ModelChoice(
        key="north-mini-code",
        name="North Mini Code 30B-A3B (Q4)",
        hf_repo="unsloth/North-Mini-Code-1.0-GGUF",
        filename="North-Mini-Code-1.0-UD-Q4_K_M.gguf",
        size_gb=17.9,
        active_params="3B active (30B total MoE)",
        architecture="cohere2_moe",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes=(
            "Cohere North Mini Code 1.0 — agentic coding model; 30B total parameters with ~3B "
            "active. Cohere reports 33.4 on the Artificial Analysis Coding Index. Apache 2.0. "
            "UD-Q4_K_M is ~17.9 GiB, so recommend 32 GB+ unified memory. Untested on this stack."
        ),
    ),
    ModelChoice(
        key="gemma-q8",
        name="Gemma 4 26B-A4B (Q8)",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        filename="gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf",
        size_gb=28.0,
        active_params="3.8B (top-8 of 128 experts)",
        architecture="gemma4-iswa",
        license="Gemma (Google) — research + commercial w/ attribution",
        humaneval_pass_at_1=None,
        notes="High-RAM workstation pick — same MoE architecture as the small Gemma entry, just near-lossless Q8 quant instead of IQ3_S. 3.8B active params = same decode speed class as the IQ3 version (~95 tok/s on M5 Max) at ~99% of BF16 quality vs ~80% on IQ3. Native multimodal — pair with the F16 mmproj for image input. Requires ≥48 GB unified memory; sweet spot on 128 GB Apple Silicon.",
        mmproj_filename="mmproj-gemma-4-26B-A4B-F16.gguf",
        mmproj_size_gb=1.2,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelChoice(
        key="north-mini-code",
        name="North-Mini-Code 1.0 (30B-A3B MoE)",
        hf_repo="unsloth/North-Mini-Code-1.0-GGUF",
        filename="North-Mini-Code-1.0-UD-IQ1_M.gguf",
        size_gb=9.38,
        active_params="3B active / 30B total (MoE, 128 experts, 8 active per token)",
        architecture="cohere2_moe",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes=(
            "Cohere Labs' code-specialized sparse MoE — 30B total / ~3B active (128 experts, "
            "top-8), interleaved sliding-window + global (NoPE) attention 3:1, 256K context. "
            "DOES NOT fit a 16 GB Mac's ~8.8 GB budget: smallest published Unsloth quant is "
            "UD-IQ1_M at 9.38 GB (~0.6 GB over), and IQ1-level quant degrades code quality "
            "noticeably — do not auto-select for 16 GB Macs without warning the user. Wants a "
            "24 GB+ Mac (UD-Q3_K_XL ~14.3 GB or UD-IQ4_XS ~15.2 GB). Architecture is "
            "Cohere2MoeForCausalLM / model_type \"cohere2_moe\", a NEW MoE variant of cohere2 "
            "(Command R7B) — verify your llama.cpp/TurboQuant fork is new enough to load it. "
            "Tool calls use Cohere's command-R style parser, not gemma/qwen/llama. Text-only, "
            "no vision/mmproj. Untested on this stack — no HumanEval number yet."
        ),
    ),
    ModelChoice(
        key="qwen-q8",
        name="Qwen 3.6 35B-A3B (Q8)",
        hf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        filename="Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",
        size_gb=38.5,
        active_params="3.0B (top-8 + 1 shared of 256 experts)",
        architecture="qwen35moe",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes="High-RAM workstation pick (≥64 GB unified memory; sweet spot on 128 GB Apple Silicon). UD-Q8_K_XL keeps attention/router/embedding layers above 8-bit while bulk experts stay at Q8 → ~99.3% of BF16 quality vs ~75-80% on IQ2_M. Same 3B active params → same decode speed class as the IQ2 entry. Native 262K context. Requires the multi-region mmap patch (llama-cpp-turboquant commit 3d66675b8).",
        mmproj_filename="mmproj-Qwen3.6-35B-A3B-F16.gguf",
        mmproj_size_gb=0.9,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
]


def _system_ram_gb() -> int:
    """Standalone RAM probe — mirrors LocalCodeRuntimeGateway._system_ram_gb so
    the picker doesn't need a runtime instance. Falls back to 16 on any failure
    (conservative — favors the lightweight model)."""
    try:
        import platform, subprocess
        if platform.system() == "Darwin":
            mem_bytes = int(subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip())
            return max(1, mem_bytes // (1024 ** 3))
        # Linux / others: read /proc/meminfo
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return max(1, kb // (1024 * 1024))
    except Exception:
        pass
    return 16


def recommend(ram_gb: int | None = None) -> ModelChoice:
    """Pick the catalog entry best suited to the host's RAM.

    Rule of thumb: model weights + KV + OS headroom should fit in ~70% of
    unified memory. We bias conservatively — being slightly under-quantized
    is worse than crashing on load. Falls back to the first entry if nothing
    matches (shouldn't happen with the current catalog).
    """
    if ram_gb is None:
        ram_gb = _system_ram_gb()
    # Sort candidates by size_gb; pick the biggest whose weights fit
    # in ~55% of RAM (leaves room for KV cache, activations, OS).
    budget = ram_gb * 0.55
    fits = [c for c in CHOICES if c.size_gb <= budget]
    if fits:
        return max(fits, key=lambda c: c.size_gb)
    # Nothing fits the budget — return the smallest entry so at least
    # something is suggested rather than a recommendation the user can't run.
    return min(CHOICES, key=lambda c: c.size_gb)


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


# ---------------------------------------------------------------------------
# Curated MODEL-GROUP layer (ADDITIVE — built on top of the catalog above).
#
# CHOICES is a hand-picked shortlist: one or two quants per model, chosen to
# fit common Macs. The browsing UI wants something richer — like HuggingFace's
# GGUF page, it lists EVERY quant a repo ships for a given model VERSION, each
# with its exact size and a fit badge for this Mac's unified memory.
#
# A ModelGroup describes one model VERSION (repo + family + arch + license +
# optional vision sidecar). The browser fetches the repo's quant list at
# runtime, then calls `choice_for_quant(group, filename, size_gb)` to mint a
# ModelChoice for whichever quant the user picks. That ModelChoice flows
# through the EXISTING download/runtime path unchanged — it carries the same
# hf_repo / architecture / license / mmproj fields the rest of the code reads.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelGroup:
    key: str               # short id for the group (e.g. "gemma-4-12b")
    display_name: str      # SPECIFIC version, e.g. "Gemma 4 12B" — never bare "gemma"
    maker: str             # Google / Alibaba / Cohere
    hf_repo: str           # HuggingFace repo that ships the quants
    family: str            # model family bucket (gemma4 / qwen / cohere)
    architecture: str      # llama.cpp arch enum, copied onto each ModelChoice
    license: str           # license tag, copied onto each ModelChoice
    notes: str             # honest caveats, copied onto each minted ModelChoice
    # Optional vision sidecar — mirrors ModelChoice's mmproj fields. None / 0
    # for text-only models. `mmproj_filename` is the unique LOCAL name (so two
    # families' mmprojs don't collide on disk); `mmproj_hf_filename` is what HF
    # calls it in the repo (commonly just "mmproj-F16.gguf").
    mmproj_filename: str | None = None
    mmproj_size_gb: float = 0.0
    mmproj_hf_filename: str | None = None

    @property
    def supports_vision(self) -> bool:
        return self.mmproj_filename is not None


MODEL_GROUPS: list[ModelGroup] = [
    ModelGroup(
        key="gemma-4-26b-a4b",
        display_name="Gemma 4 26B-A4B",
        maker="Google",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        family="gemma4",
        architecture="gemma4-iswa",
        license="Gemma (Google) — research + commercial w/ attribution",
        notes=(
            "Google's sparse MoE — ~3.8B active (top-8 of 128 experts). Native "
            "multimodal: pair any quant with the F16 mmproj for image input. "
            "Tool calling via Gemma 4 native format."
        ),
        mmproj_filename="mmproj-gemma-4-26B-A4B-F16.gguf",
        mmproj_size_gb=1.2,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelGroup(
        key="gemma-4-12b",
        display_name="Gemma 4 12B",
        maker="Google",
        hf_repo="unsloth/gemma-4-12b-it-GGUF",
        family="gemma4",
        architecture="gemma4-iswa",
        license="Apache 2.0",
        notes=(
            "Mid-sized dense Gemma 4 — sits between E4B and the 26B MoE. Native "
            "multimodal (vision + audio per Google's Gemma 4 announcement); pair "
            "any quant with the F16 mmproj for image input. Apache 2.0 licensed."
        ),
        mmproj_filename="mmproj-gemma-4-12b-F16.gguf",
        mmproj_size_gb=0.86,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelGroup(
        key="qwen-3.6-35b-a3b",
        display_name="Qwen 3.6 35B-A3B",
        maker="Alibaba",
        hf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        family="qwen",
        architecture="qwen35moe",
        license="Apache 2.0",
        notes=(
            "Alibaba's sparse MoE — ~3.0B active (top-8 + 1 shared of 256 experts). "
            "Hybrid attn + Mamba-2 SSM, native 262K context. Native multimodal: "
            "pair any quant with the F16 mmproj for image input. Requires the "
            "multi-region mmap patch (llama-cpp-turboquant commit 3d66675b8)."
        ),
        mmproj_filename="mmproj-Qwen3.6-35B-A3B-F16.gguf",
        mmproj_size_gb=0.9,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelGroup(
        key="north-mini-code-1.0",
        display_name="North-Mini-Code 1.0 30B-A3B",
        maker="Cohere",
        hf_repo="unsloth/North-Mini-Code-1.0-GGUF",
        family="cohere",
        architecture="cohere2_moe",
        license="Apache 2.0",
        notes=(
            "Cohere Labs' code-specialized sparse MoE — 30B total / ~3B active "
            "(128 experts, top-8), interleaved sliding-window + global (NoPE) "
            "attention 3:1, 256K context. Architecture is Cohere2MoeForCausalLM "
            "(\"cohere2_moe\") — verify your llama.cpp/TurboQuant fork is new "
            "enough to load it. Tool calls use Cohere's command-R style parser. "
            "Text-only — no vision/mmproj."
        ),
        # Text-only: no vision sidecar.
    ),
]


def _quant_label(filename: str) -> str:
    """Derive a readable quant label from a GGUF filename.

    Strips the repo/model prefix and the `.gguf` suffix, leaving the quant
    descriptor, e.g.:
        "gemma-4-12b-it-UD-Q4_K_XL.gguf" -> "UD-Q4_K_XL"
        "Qwen3.6-35B-A3B-UD-IQ2_M.gguf"  -> "UD-IQ2_M"
        "gemma-4-12b-it-BF16.gguf"       -> "BF16"
    Falls back to the bare stem if no quant token is recognizable.
    """
    import re
    stem = filename[:-5] if filename.lower().endswith(".gguf") else filename
    # Match a trailing quant descriptor: optional UD-/IQ/Q tier with K/_ parts,
    # or a plain precision tag like BF16 / F16 / F32.
    m = re.search(
        r"(?:^|[-_.])((?:UD-)?(?:I?Q\d+[A-Z0-9_]*|BF16|F16|F32))$",
        stem,
    )
    if m:
        return m.group(1)
    # Fallback: take the last hyphen-delimited chunk that looks quant-ish.
    tail = stem.rsplit("-", 1)[-1]
    return tail or stem


def choice_for_quant(group: ModelGroup, filename: str, size_gb: float) -> ModelChoice:
    """Mint a ModelChoice for a specific quant of `group` so the EXISTING
    download/runtime path (download_model / get_model_path / launch) just
    works. Copies repo / arch / license / mmproj from the group; the quant's
    filename + size come from the (live-fetched) repo listing.
    """
    label = _quant_label(filename)
    return ModelChoice(
        key=f"{group.key}:{label}",
        name=f"{group.display_name} ({label})",
        hf_repo=group.hf_repo,
        filename=filename,
        size_gb=size_gb,
        active_params="",  # not known per-quant from the listing; group-level
        architecture=group.architecture,
        license=group.license,
        humaneval_pass_at_1=None,  # browsed quants are unmeasured on this stack
        notes=group.notes,
        mmproj_filename=group.mmproj_filename,
        mmproj_size_gb=group.mmproj_size_gb,
        mmproj_hf_filename=group.mmproj_hf_filename,
    )


def by_group(key: str) -> ModelGroup | None:
    for g in MODEL_GROUPS:
        if g.key == key:
            return g
    return None
