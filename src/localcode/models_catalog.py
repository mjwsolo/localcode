"""Model catalog — single source of truth for available models.

Every entry must be transparent: name, source, exact filename, size, license,
where it'll be downloaded, and any caveats. The picker UI reads from this list
so users see exactly what they're choosing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    # Integrity pinning (both optional, default = unpinned = current behavior).
    # `revision` pins the download to a specific git revision (commit SHA or
    # tag) instead of the mutable `main` branch tip — upstream silently swaps
    # weights under the same filename (see the Gemma notes), and a pinned
    # revision makes a swap visible. `sha256` is the expected hex digest of the
    # GGUF; when set, the download is verified after completion and DELETED +
    # rejected on mismatch. Populate these as hashes/revisions are captured;
    # any entry left unpinned downloads exactly as before.
    revision: str = "main"
    sha256: str | None = None
    # Exact byte size from the HF API. The old completeness check allowed a
    # 3% tolerance, which cannot distinguish a good file from a stale one:
    # upstream re-published gemma-4-26B-A4B UD-Q8_K_XL with the same name and
    # a 1,984-byte difference, and the stale copy was silently kept for months
    # while emitting <unused17> garbage instead of text. An exact size catches
    # that for free; sha256 above catches a same-size swap.
    size_bytes: int | None = None
    reasoning_control: str = "chat_template"
    reasoning_budget_tokens: int = 8192
    preserves_reasoning: bool = True
    supports_parallel_tools: bool = False

    @property
    def hf_url(self) -> str:
        return f"https://huggingface.co/{self.hf_repo}/blob/{self.revision}/{self.filename}"

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

    # ------------------------------------------------------------------
    # Declarative capability surface.
    #
    # These properties make a model's intended modalities EXPLICIT so the
    # picker/runtime can reason about them and a misconfigured future entry
    # can't silently lose a capability. They describe the model's INTENT,
    # not whether the host can currently exercise it — the actual gates are:
    #   * vision: requires an mmproj sidecar AND the launch path passing
    #     `--mmproj` to llama-server (see runtime.py / mmproj_path below).
    #   * audio in/out: hardware-gated only (macOS mic/speakers, handled in
    #     voice.py). It is model-AGNOSTIC today — every text model can be
    #     spoken to / read aloud because audio is transcribed to / synthesized
    #     from text at the voice layer, not inside the model. So we default
    #     both audio flags True for all entries and let voice.py do the real
    #     hardware gating.
    # ------------------------------------------------------------------

    @property
    def supports_vision(self) -> bool:
        """True iff the model ships a vision projector (mmproj) sidecar.

        Derived from `mmproj_filename` — this is the single, obvious place
        that defines vision capability. Unchanged in behavior from the
        original inference (`mmproj_filename is not None`); now documented
        as the declarative source of truth. Note the runtime must still pass
        `--mmproj` at launch for image input to actually work.
        """
        return self.mmproj_filename is not None

    @property
    def supports_thinking(self) -> bool:
        """True iff the model has a toggleable hidden-reasoning channel.

        Gate for the `/thinking` policy. Block-diffusion models (`diffusion_*`)
        generate spans in parallel, not token-by-token, so they have no
        hidden-reasoning stream to toggle (the status bar shows `n/a`); every
        other catalog arch supports it. Derived from the architecture.
        """
        return self.reasoning_control != "none"

    @property
    def supports_audio_in(self) -> bool:
        """True if the model may be addressed by voice (speech -> text).

        Model-agnostic today: speech is transcribed to text before it reaches
        the model, so every entry supports it. The real gate is microphone
        hardware availability (macOS), enforced in voice.py.
        """
        return True

    @property
    def supports_audio_out(self) -> bool:
        """True if the model's replies may be spoken (text -> speech).

        Model-agnostic today: replies are synthesized from text after the
        model produces them, so every entry supports it. The real gate is
        speaker/TTS hardware availability (macOS), enforced in voice.py.
        """
        return True

    @property
    def capabilities(self) -> frozenset[str]:
        """The set of capability tags this model declares.

        A convenience rollup of the individual `supports_*` properties for
        callers that prefer set membership (e.g. `"vision" in choice.capabilities`).
        Always derived from the properties above so the two views can't drift.
        """
        caps = set()
        if self.supports_vision:
            caps.add("vision")
        if self.supports_thinking:
            caps.add("thinking")
        if self.supports_audio_in:
            caps.add("audio_in")
        if self.supports_audio_out:
            caps.add("audio_out")
        return frozenset(caps)


CHOICES: list[ModelChoice] = [
    ModelChoice(
        key="gemma",
        sha256="878be93f9c238ea853b3fd1eb602637ce3cf1cddea56dc345d9a7bf2d6093e29",
        size_bytes=11289671136,
        name="Gemma 4 26B-A4B (Q3)",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        filename="gemma-4-26B-A4B-it-UD-IQ3_S.gguf",
        size_gb=11.2,
        active_params="3.8B (top-8 of 128 experts)",
        architecture="gemma4-iswa",
        license="Gemma (Google) — research + commercial w/ attribution",
        humaneval_pass_at_1=0.951,  # 156/164 measured on this stack 2026-04-22
        notes="Comfortable fit on 16 GB Mac. Has Apr-11 chat template + Apr-8 tokenizer fixes. Tool calling via Gemma 4 native format. Measured 95.1% HumanEval pass@1 at IQ3_S + TurboQuant KV on our harness. Upstream silently refreshed these weights 2026-07-16 (tool-calling + vision fixes, same filenames) — copies downloaded before then are stale; delete and re-download to pick up the fixes.",
        mmproj_filename="mmproj-gemma-4-26B-A4B-F16.gguf",
        mmproj_size_gb=1.2,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelChoice(
        key="gemma-12b",
        sha256="90fd944d227e9d9b68e7e2c7d5b57b79d4c66ed521b0919fbbd932cf834f6f8e",
        size_bytes=7366423360,
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
            "Native multimodal (vision + audio). Apache 2.0. Upstream silently "
            "refreshed these weights 2026-07-16 (tool-calling + vision fixes, same "
            "filenames) — copies downloaded before then are stale; delete and "
            "re-download to pick up the fixes."
        ),
        mmproj_filename="mmproj-gemma-4-12b-F16.gguf",
        mmproj_size_gb=0.86,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelChoice(
        key="gemma-12b-bf16",
        sha256="5a5eefea73350705c6753105b725a51301d653e8d6646173db29a7e8da8e6efd",
        size_bytes=23832066656,
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
            "unquantized. Output quality untested on this stack, but the BF16 GGUF is verified to "
            "exist as a SINGLE file (not sharded) and the mmproj filename is confirmed against the "
            "repo (2026-06-12), so the download resolves."
        ),
        mmproj_filename="mmproj-gemma-4-12b-F16.gguf",
        mmproj_size_gb=0.86,  # 862 MB; repo offers BF16 + F16, both 862 MB
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelChoice(
        key="qwen",
        sha256="2be7ef1ed7e1af8b10d3829102cf9a6c2bd5ddb64d675b4ece23a60799403d43",
        size_bytes=11522702304,
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
        sha256="24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5",
        size_bytes=16806810208,
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
        reasoning_control="none",
        reasoning_budget_tokens=0,
    ),
    ModelChoice(
        key="north-mini-code",
        sha256="d59f4cb3abec1bd3e5e185b99cdd811e3be3402cbc2bab8223081a460d5724bf",
        size_bytes=19203186784,
        name="North Mini Code 30B-A3B (Q4)",
        hf_repo="unsloth/North-Mini-Code-1.0-GGUF",
        filename="North-Mini-Code-1.0-UD-Q4_K_M.gguf",
        size_gb=17.9,
        active_params="3B active (30B total MoE)",
        architecture="cohere2_moe",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes=(
            "Cohere North Mini Code 1.0 — agentic coding model; 30B total / ~3B "
            "active. Its cohere2moe arch isn't in the TurboQuant server, so "
            "LocalCode builds a dedicated llama-server from llama.cpp PR #24260 "
            "on first use (one-time, ~5-12 min) and serves it with stock flags. "
            "Apache 2.0. ~17.9 GiB; recommend 32 GB+ unified memory."
        ),
        reasoning_budget_tokens=0,
    ),
    ModelChoice(
        key="muse-glimmer",
        sha256="82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38",
        size_bytes=15878222368,
        name="Muse Glimmer 30B (Q4)",
        hf_repo="unsloth/Muse-Glimmer-30B-GGUF",
        filename="Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        size_gb=15.9,
        active_params="30B dense (+ perception encoder for vision)",
        architecture="muse_glimmer",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes=(
            "Meta's Muse Glimmer 30B (Meta Superintelligence Lab, Aug 2026) — a "
            "multimodal agentic model purpose-built for local deployment: multi-step "
            "reasoning, tool use, failure recovery, and vision in one model. Unsloth "
            "Dynamic UD-Q4_K_XL quant. Its muse_glimmer arch isn't in the TurboQuant "
            "server, so LocalCode builds a dedicated stock llama-server from current "
            "llama.cpp (PR #26841) on first use (one-time, ~5-12 min) and serves it "
            "with stock flags + --jinja (required by its chat template). Pair with the "
            "BF16 perception encoder for image input. ~15.9 GiB; recommend 32 GB+ "
            "unified memory."
        ),
        mmproj_filename="mmproj-Muse-Glimmer-30B-BF16.gguf",
        mmproj_size_gb=3.85,
        mmproj_hf_filename="mmproj-Muse-Glimmer-30B-BF16.gguf",
        reasoning_budget_tokens=0,
    ),
    ModelChoice(
        key="gemma-q8",
        sha256="d4bf9791d727d7b88aeea89aba309c68086a4d51cf337047c4e51dde7e243058",
        size_bytes=27636232928,
        name="Gemma 4 26B-A4B (Q8)",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        filename="gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf",
        size_gb=28.0,
        active_params="3.8B (top-8 of 128 experts)",
        architecture="gemma4-iswa",
        license="Gemma (Google) — research + commercial w/ attribution",
        humaneval_pass_at_1=None,
        notes="High-RAM workstation pick — same MoE architecture as the small Gemma entry, just near-lossless Q8 quant instead of IQ3_S. 3.8B active params = same decode speed class as the IQ3 version (~95 tok/s on M5 Max) at ~99% of BF16 quality vs ~80% on IQ3. Native multimodal — pair with the F16 mmproj for image input. Requires ≥48 GB unified memory; sweet spot on 128 GB Apple Silicon. Upstream silently refreshed these weights 2026-07-16 (tool-calling + vision fixes, same filenames) — copies downloaded before then are stale; delete and re-download.",
        mmproj_filename="mmproj-gemma-4-26B-A4B-F16.gguf",
        mmproj_size_gb=1.2,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelChoice(
        key="qwen-q8",
        sha256="b762215c5f507f4865df4ac3d1afa803828afa41e05ecac3fac431a67bbd88e8",
        size_bytes=38451182560,
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
    ModelChoice(
        key="qwen38",
        sha256="3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e",
        size_bytes=17559178144,
        name="Qwen 3.8 27B (Q4)",
        hf_repo="unsloth/Qwen3.8-27B-GGUF",
        filename="Qwen3.8-27B-UD-Q4_K_XL.gguf",
        size_gb=17.9,
        active_params="27B dense (hybrid attention + Mamba-2 SSM, 1 MTP layer)",
        architecture="qwen35",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes="Dense 27B — every parameter active each token (no MoE routing), so it's heavier per token than the 35B-A3B MoE but denser in capability. Hybrid attention + Mamba-2 SSM with a multi-token-prediction (MTP) layer; the TurboQuant fork loads the transformer stack and skips the MTP block at inference. UD-Q4_K_XL keeps attention/embedding above 4-bit. Needs ~32 GB unified memory to fit with context; comfortable on 48-64 GB Apple Silicon. Pair with the F16 mmproj for image input.",
        mmproj_filename="mmproj-Qwen3.8-27B-F16.gguf",
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


# Architectures we never AUTO-recommend (users can still pick them explicitly
# in the model picker). Every catalog architecture now loads on the bundled
# binaries with no runtime build or download, so cohere2_moe (North-Mini-Code)
# and muse_glimmer (Muse Glimmer) are recommendable like any other model.
#
# diffusion_gemma stays excluded on PRODUCT grounds, not technical ones: it is
# a research block-denoising model. It is served by the same bundled
# llama-server as everything else, but output arrives one 256-token block at a
# time (not token by token) and it reasons visibly every turn. It works, but
# it is not the coding-agent experience a first-run user should land on by
# default. Owner can override by emptying this set.
_NO_AUTO_RECOMMEND_ARCHS = {"diffusion_gemma"}

# Capability order for auto-recommend, best → worst for coding-agent use. This
# is deliberately NOT raw file size: the big MoEs measure ~95% HumanEval here
# (even at low bit) and must outrank the 12B dense, and a higher-bit quant of a
# family wins. A 12B-BF16 must never be recommended over a 26B/35B MoE just
# because its file happens to be larger.
_RECOMMEND_ORDER = [
    "qwen-q8",         # 35B-A3B MoE, near-lossless Q8
    "gemma-q8",        # 26B-A4B MoE, near-lossless Q8
    "qwen",            # 35B-A3B MoE Q2 — 94.7% HumanEval
    "gemma",           # 26B-A4B MoE Q3 — 95.1% HumanEval
    "gemma-12b-bf16",  # 12B dense, full precision
    "gemma-12b",       # 12B dense Q4
]


def _capability_rank(choice) -> int:
    """Lower = more capable. Curated models first; any other production model
    sorts after them — so a newly-added quant is still recommendable, but never
    outranks a curated MoE."""
    order = {k: i for i, k in enumerate(_RECOMMEND_ORDER)}
    return order.get(choice.key, len(_RECOMMEND_ORDER) + 1)


def recommend(ram_gb: int | None = None) -> ModelChoice:
    """Pick the best model for THIS machine's RAM — capability-ranked, never the
    research diffusion architecture, and never a hardcoded default.

    Weights must fit in ~55% of unified memory (leaves room for KV cache,
    activations, OS). Among the production-ready models that fit, return the
    most capable (see ``_RECOMMEND_ORDER``) so the recommendation scales with
    the user's hardware instead of defaulting to any one model.
    """
    if ram_gb is None:
        ram_gb = _system_ram_gb()
    budget = ram_gb * 0.55
    candidates = [
        c for c in CHOICES
        if c.architecture not in _NO_AUTO_RECOMMEND_ARCHS and c.size_gb <= budget
    ]
    if candidates:
        # Most capable that fits; tie-break toward the larger (better-quant) file.
        return min(candidates, key=lambda c: (_capability_rank(c), -c.size_gb))
    # Nothing fits the budget — smallest production-ready model so the user still
    # gets something runnable rather than an impossible recommendation.
    prod = [c for c in CHOICES if c.architecture not in _NO_AUTO_RECOMMEND_ARCHS]
    return min(prod or CHOICES, key=lambda c: c.size_gb)


def by_key(key: str) -> ModelChoice | None:
    for c in CHOICES:
        if c.key == key:
            return c
    return None


def by_filename(filename: str) -> ModelChoice | None:
    for c in CHOICES:
        if c.filename == filename:
            return c
    # Not one of the curated CHOICES — maybe a quant the user browsed via
    # the HF-style picker (which can select ANY quant of a MODEL_GROUPS
    # repo). Unsloth GGUF filenames are `<repo-stem>-<QUANT>.gguf`, so a
    # prefix match on the repo stem identifies the group; mint a choice so
    # downstream consumers (status bar, fit checks, and CRITICALLY the
    # runtime's architecture dispatch for diffusion models) see correct
    # metadata instead of None.
    return _minted_for_filename(filename)


@lru_cache(maxsize=64)
def _minted_for_filename(filename: str) -> ModelChoice | None:
    g = group_for_filename(filename)
    if g is None:
        return None
    # Fill the real size from the picker's 24h quant-listing cache when
    # available (cache-only — by_filename runs on the status bar's 2 s
    # tick, so NO network here). With a real size, completeness checks
    # treat partial downloads correctly.
    size_gb = 0.0
    try:
        from .hf_quants import _read_cache
        cached = _read_cache(g.hf_repo)
        if cached:
            for q in cached[0]:
                if q.filename == filename and not q.is_mmproj:
                    size_gb = q.size_gb
                    break
    except Exception:
        size_gb = 0.0
    return choice_for_quant(g, filename, size_gb)


def group_for_filename(filename: str) -> ModelGroup | None:
    """Match a GGUF filename to the MODEL_GROUPS repo it came from."""
    for g in MODEL_GROUPS:
        stem = g.hf_repo.rsplit("/", 1)[-1]
        if stem.endswith("-GGUF"):
            stem = stem[: -len("-GGUF")]
        if stem and filename.startswith(stem):
            return g
    return None


def current(config) -> ModelChoice | None:
    """Resolve the currently selected model from config."""
    model_path = (config.runtime.model or "").strip()
    if not model_path:
        return None
    name = Path(model_path).name
    return by_filename(name)


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
        key="qwen-3.8-27b",
        display_name="Qwen 3.8 27B",
        maker="Alibaba",
        hf_repo="unsloth/Qwen3.8-27B-GGUF",
        family="qwen",
        architecture="qwen35",
        license="Apache 2.0",
        notes=(
            "Alibaba's dense 27B — all params active per token (no MoE routing). "
            "Hybrid attention + Mamba-2 SSM with a multi-token-prediction layer; "
            "the TurboQuant fork serves the transformer stack and skips the MTP "
            "block. Native multimodal: pair any quant with the F16 mmproj for "
            "image input. Tool calling via Qwen/Hermes format."
        ),
        mmproj_filename="mmproj-Qwen3.8-27B-F16.gguf",
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
            "Cohere Labs' code-specialized sparse MoE — 30B total / ~3B active. "
            "Its cohere2moe arch isn't in the TurboQuant server, so LocalCode "
            "builds a dedicated llama-server from llama.cpp PR #24260 on first "
            "use and serves it with stock flags. 256K context. Text-only."
        ),
    ),
    ModelGroup(
        key="muse-glimmer-30b",
        display_name="Muse Glimmer 30B",
        maker="Meta",
        hf_repo="unsloth/Muse-Glimmer-30B-GGUF",
        family="muse",
        architecture="muse_glimmer",
        license="Apache 2.0",
        notes=(
            "Meta's multimodal agentic model for local deployment (Meta "
            "Superintelligence Lab, Aug 2026): multi-step reasoning, tool use, "
            "failure recovery, native vision. Unsloth Dynamic quants. Its "
            "muse_glimmer arch isn't in the TurboQuant server — LocalCode builds a "
            "dedicated stock llama-server from llama.cpp PR #26841 on first use and "
            "serves it with stock flags + --jinja. Pair with the BF16 perception "
            "encoder for image input."
        ),
        mmproj_filename="mmproj-Muse-Glimmer-30B-BF16.gguf",
        mmproj_size_gb=3.85,
        mmproj_hf_filename="mmproj-Muse-Glimmer-30B-BF16.gguf",
    ),
    ModelGroup(
        key="diffusiongemma-26b-a4b",
        display_name="DiffusionGemma 26B-A4B",
        maker="Google",
        hf_repo="unsloth/diffusiongemma-26B-A4B-it-GGUF",
        family="gemma4",
        architecture="diffusion_gemma",
        license="Apache 2.0",
        notes=(
            "EXPERIMENTAL block-diffusion model — denoises whole 256-token "
            "blocks in parallel instead of decoding token-by-token. Served by "
            "the bundled llama-server like every other model (the fork hosts "
            "the denoiser from llama.cpp PR #24423 inside the server), so the "
            "model loads once and stays resident; output streams one block "
            "at a time rather than token by token. The repo ships "
            "plain quants only (Q4_K_M/Q5_K_M/Q6_K/Q8_0/BF16, no UD-* "
            "dynamic quants); Q4_K_M (~15.7 GB) is the recommended pick. "
            "Text-only."
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


def _parse_total_active_b(name: str) -> tuple[float, float]:
    """(total_params_B, active_params_B) parsed from a model name.

    "Gemma 4 26B-A4B" -> (26, 4); "Qwen 3.6 35B-A3B" -> (35, 3);
    a dense "Gemma 4 12B" -> (12, 12). (0, 0) if no size is found.

    Handles optional decimal totals ("Qwen 3.6 35B-A3B" → total=35.0) and
    both hyphen and space separators between the total-B and A<n>B tokens.
    The active figure parsed here may differ slightly from the exact active
    count advertised per model (e.g. Gemma 26B-A4B has 3.8B actual active
    but the name encodes "4B"); callers rely on the ratio active/total for
    the bandwidth fraction, so small rounding in the name is acceptable.
    """
    import re
    # MoE pattern: "<total>B[-_ ]A<active>B" (e.g. "26B-A4B", "35B-A3B")
    m = re.search(r"(\d+(?:\.\d+)?)\s*B\s*[-_ ]?A(\d+(?:\.\d+)?)", name, re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    # Dense pattern: single "<n>B" (active == total)
    m = re.search(r"(\d+(?:\.\d+)?)\s*B", name, re.I)
    if m:
        v = float(m.group(1))
        return v, v
    return 0.0, 0.0


def estimate_decode_tok_s(size_gb: float, name: str, bandwidth_gbps: float) -> int | None:
    """Rough estimated decode speed (tokens/sec) for a quant on this machine.

    Uses a saturating two-term model:

        tok_s = 1 / (effective_gb / realized_bw + compute_overhead)

    where:
      effective_gb   = size_gb × (active / total)   — bytes read per token
                       (for MoE only the active experts are read)
      realized_bw    = bandwidth_gbps × 0.70         — Apple Silicon realises
                       ~65-75 % of peak spec bandwidth during decode
      compute_overhead                               — per-token floor that
                       CAPS tiny-active MoE quants at a realistic ceiling.
                       MoE routing and expert dispatch add ~8 ms/token of
                       irreducible compute on Apple Silicon regardless of
                       quant size; dense models are lower (~2 ms/token) so
                       the model remains more bandwidth-bound there.

    Calibration (M5 Max, 614 GB/s):
      Gemma-4-26B-A4B IQ3_S (~1.7 GB effective) → ~83 tok/s  [measured]
      Gemma-4-26B-A4B Q8    (~4.3 GB effective) → ~55 tok/s  [estimate]
      Dense Gemma-4-12B Q4  (~7.4 GB effective) → ~35 tok/s  [estimate]

    The ABSOLUTE number is approximate — it ignores attention/KV-cache reads,
    context length, and per-session speculative decoding gains (which can add
    3-8× on repetitive/code output but are invisible to this static formula).
    The PRIMARY signal is the RATIO between quants of the same model family:
    a 3× size difference in effective_gb produces a roughly 2× speed ratio
    once compute overhead is included, matching observed MoE flatness.
    Returns None when inputs are invalid.
    """
    if size_gb <= 0 or bandwidth_gbps <= 0:
        return None
    total, active = _parse_total_active_b(name)
    frac = (active / total) if (total > 0 and active > 0) else 1.0
    effective_gb = size_gb * frac
    if effective_gb <= 0:
        return None

    # tok/s estimate inputs live in the central per-Mac config (model_config)
    # so the calibration constants are edited in one place.
    from .model_config import (
        DECODE_COMPUTE_OVERHEAD_DENSE_S,
        DECODE_COMPUTE_OVERHEAD_MOE_S,
        DECODE_MOE_ACTIVE_FRAC_THRESHOLD,
        DECODE_REALIZED_BW_FRACTION,
    )

    # Apple Silicon realises roughly 65-75% of advertised peak bandwidth
    # during LLM decode (cache misses, memory controller overhead, etc.).
    realized_bw = bandwidth_gbps * DECODE_REALIZED_BW_FRACTION

    # Per-token compute overhead (seconds) — the irreducible floor that
    # prevents tiny-active MoE quants from predicting unrealistic speeds.
    # MoE models have significant routing/dispatch overhead (~8 ms/token);
    # dense models are lower (~2 ms/token) so bandwidth dominates there.
    is_moe = frac < DECODE_MOE_ACTIVE_FRAC_THRESHOLD
    compute_overhead_s = (
        DECODE_COMPUTE_OVERHEAD_MOE_S if is_moe else DECODE_COMPUTE_OVERHEAD_DENSE_S
    )

    tok_s = 1.0 / (effective_gb / realized_bw + compute_overhead_s)

    if tok_s >= 100:
        return int(round(tok_s / 10) * 10)
    if tok_s >= 20:
        return int(round(tok_s / 5) * 5)
    return max(1, int(round(tok_s)))


def by_group(key: str) -> ModelGroup | None:
    for g in MODEL_GROUPS:
        if g.key == key:
            return g
    return None
