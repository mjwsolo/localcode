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
class Drafter:
    """A vendor-trained speculative drafter for ONE target model (MTP head or
    DFlash block drafter). It ships with the model: the download that installs
    the model installs the drafter too, and the launcher passes it as
    `--model-draft` with its `--spec-type`. Every drafted token is verified by
    the target, so the output is the target's own; measured 1.5x to 2.3x decode
    on Apple Silicon. Only drafters trained for the exact model are listed:
    a generic small model as drafter is what failed in the past.
    """
    filename: str          # local name (unique across families)
    hf_repo: str           # repo that ships it (may differ from the model's)
    hf_filename: str       # path inside that repo
    spec_type: str         # llama-server --spec-type: "draft-mtp" | "draft-dflash"
    size_gb: float
    size_bytes: int        # exact, from the HF API (download completeness)
    sha256: str            # verified after download
    revision: str = "main"

    @property
    def local_path(self) -> Path:
        return model_dir() / self.filename


# Verified live 2026-10-09 (greedy, same coding prompt, bundled v0.6.0 server):
# gemma-4-12b Q4_K_XL 55 -> 120-139 tok/s; Qwen3.6-35B IQ2_M 129 -> 224 (DFlash);
# Muse Glimmer Q4_K_XL 30 -> 45 (DFlash). Qwen 3.8 27B carries its head inside
# the GGUF (`nextn_predict_layers`) and needs no file.
DRAFTER_GEMMA_12B = Drafter(
    filename="mtp-gemma-4-12b-it-Q8_0.gguf", hf_repo="unsloth/gemma-4-12b-it-GGUF",
    hf_filename="MTP/mtp-gemma-4-12b-it-Q8_0.gguf", spec_type="draft-mtp",
    size_gb=0.47, size_bytes=465109248,
    sha256="145db9094bc0f85f1701e255a2ed216dcc9800fc8bc8631ad00905b456bd451b",
    revision="fc034cfff751157913579611efad8462ac1be606",
)
DRAFTER_GEMMA_26B = Drafter(
    filename="mtp-gemma-4-26B-A4B-it-Q8_0.gguf", hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
    hf_filename="MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf", spec_type="draft-mtp",
    size_gb=0.46, size_bytes=461766816,
    sha256="6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b",
    revision="c099eb48e663fd284577b04978a94ffccb261841",
)
DRAFTER_QWEN36 = Drafter(
    filename="dflash-Qwen3.6-35B-A3B-Q8_0.gguf", hf_repo="ggml-org/Qwen3.6-35B-A3B-GGUF",
    hf_filename="dflash-Qwen3.6-35B-A3B-Q8_0.gguf", spec_type="draft-dflash",
    size_gb=0.42, size_bytes=421060800,
    sha256="339030c9dad4f410c7ef2d6cbd6e38251a84fabb59c44db992538c2451a0efee",
    revision="baec3ebee244827cda0f4557eafa8b28f7545fa6",
)
# DFlash 2 from the DFlash authors (z-lab mirror of incoai). Measured against
# ggml-org's DFlash v1 on the same Q4_K_XL Muse, greedy, 600 tokens: accept
# rate 0.74 vs 0.60, decode +12%, identical output, vision unaffected.
DRAFTER_MUSE = Drafter(
    filename="dflash2-Muse-Glimmer-30B-Q4_K_M.gguf", hf_repo="z-lab/Muse-Glimmer-30B-DFlash2-GGUF",
    hf_filename="Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf", spec_type="draft-dflash",
    size_gb=1.65, size_bytes=1645657280,
    sha256="93dbfb6f88e4645dec1347cf93f9d6fc80b90d413038722385b2a8e53565c949",
    revision="880882627431093d99d3b2368efb4a6fcf12d4cb",
)
DRAFTERS: list[Drafter] = [DRAFTER_GEMMA_12B, DRAFTER_GEMMA_26B, DRAFTER_QWEN36, DRAFTER_MUSE]
# Sidecar files earlier releases downloaded on the user's behalf (the user never
# picked them) that a newer Drafter record has replaced. The supervisor deletes
# them from the models directory; nothing else ever references these names.
RETIRED_SIDECARS: tuple[str, ...] = (
    "dflash-Muse-Glimmer-30B-Q4_0.gguf",   # DFlash v1, replaced by DFlash 2 in 0.5.9
)


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
    # How hidden reasoning is controlled for this model. ONE switch, applied by
    # the server command (server_cmd / runtime), never by per-wire request hints:
    #   "server"        llama-server --reasoning off --reasoning-budget 0 silences it
    #                   (gemma 4, Qwen 3.x: verified 4 tokens, empty reasoning channel)
    #   "always"        the model always reasons and ships no off switch (Muse
    #                   Glimmer: only "Reasoning strength: low|medium|high|xhigh" in the
    #                   system prompt; Meta recommends high/xhigh for agentic work)
    #   "none"          no reasoning channel at all (diffusion)
    #   "chat_template" legacy name for "server" (kept for old configs)
    reasoning_control: str = "chat_template"
    reasoning_budget_tokens: int = 8192
    preserves_reasoning: bool = True
    supports_parallel_tools: bool = False
    supports_systemone: bool = False
    # Vendor-trained speculative drafter shipped with the model (see Drafter).
    drafter: Drafter | None = None

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
        drafter=DRAFTER_GEMMA_26B,
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
        mmproj_size_gb=0.18,  # 175 MB on HF as of 2026-09 (repo re-exported the projector)
        mmproj_hf_filename="mmproj-F16.gguf",
        drafter=DRAFTER_GEMMA_12B,
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
        mmproj_size_gb=0.18,  # 175 MB on HF as of 2026-09
        mmproj_hf_filename="mmproj-F16.gguf",
        drafter=DRAFTER_GEMMA_12B,
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
        drafter=DRAFTER_QWEN36,
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
        reasoning_control="chat_template",
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
        reasoning_control="always",
        reasoning_budget_tokens=0,
        drafter=DRAFTER_MUSE,
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
        drafter=DRAFTER_GEMMA_26B,
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
        drafter=DRAFTER_QWEN36,
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
    ModelChoice(
        key="qwen38-q8",
        sha256="af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377",
        size_bytes=31457991680,
        name="Qwen 3.8 27B (Q8)",
        hf_repo="unsloth/Qwen3.8-27B-GGUF",
        filename="Qwen3.8-27B-UD-Q8_K_XL.gguf",
        size_gb=31.5,
        active_params="27B dense (hybrid attention + Mamba-2 SSM, 1 MTP layer)",
        architecture="qwen35",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes="Same dense 27B as the Q4 entry, at much higher fidelity: UD-Q8_K_XL keeps attention/embedding above 8-bit while the bulk stays Q8, ~99% of BF16 quality. Same 27B dense compute per token, so decode speed is in the same class as the Q4 while quality is far higher. The high-RAM pick for this model — fits ~96 GB unified memory with context, comfortable on 128 GB Apple Silicon. Pair with the F16 mmproj for image input.",
        mmproj_filename="mmproj-Qwen3.8-27B-F16.gguf",
        mmproj_size_gb=0.9,
        mmproj_hf_filename="mmproj-F16.gguf",
    ),
    ModelChoice(
        key="saluki",
        name="Underdog Saluki 27B 1.0 (IQ2-mix)",
        hf_repo="ConwayResearch/Underdog-Saluki-27B-1.0",
        filename="Underdog-Saluki-27B-1.0-IQ2-mix.gguf",
        revision="4f60eba09f4b6db63c72915d328d97e1d93edb8c",
        sha256="4a673518b11b1c4445f9b9a9d3c40356f5ba39f6a6260dcb25f3fbfd475d9efb",
        size_bytes=7898369152,
        size_gb=7.9,
        active_params="27B dense",
        architecture="qwen35",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes="Mixed 2-bit quantization of Qwen 3.8 27B by Underdog. 7.9 GB weights plus context, runtime and OS memory; file size is not total RAM use. Coding quality has not been benchmarked on LocalCode; opt-in, not an automatic recommendation. Optional F16 projector adds image input. No speculative drafter is bundled.",
        mmproj_filename="mmproj-Underdog-Saluki-27B-1.0-F16.gguf",
        mmproj_size_gb=0.928,
    ),
    ModelChoice(
        key="kolibri",
        # Pinned: a community quant under a mutable `main`. A re-upload must be
        # a visible change here, never a silent swap under the same filename.
        revision="b08405e141706457126fa75b9f2f9e0729b30d0a",
        sha256="c2ac1301424441ef210b6de50ce25e8ccf69f86494df53d6ba52ed558456062e",
        size_bytes=47454113472,
        name="Kolibri 1 78B-A3B (Q4, experimental)",
        hf_repo="Hob-forge/Kolibri-1-GGUF",
        filename="Kolibri-1-Q4_K_M.gguf",
        size_gb=47.5,
        active_params="3.5B active (78B total MoE)",
        architecture="kolibri1",
        license="Apache 2.0",
        humaneval_pass_at_1=None,
        notes=(
            "Aleph Alpha's open-weight English/German MoE — 78B total, ~3.5B "
            "active per token (top-6 of 384 experts plus one shared expert), "
            "sliding-window / full attention interleave, 256K native context. "
            "EXPERIMENTAL: the kolibri1 architecture is a community patch carried "
            "by the bundled server (not yet in upstream llama.cpp) and this GGUF "
            "is a community quant, not an Aleph Alpha release. Verified here: "
            "load, generation, tool calling, reasoning on/off, and exact recall "
            "at 60K tokens. Not benchmarked for coding quality. ~47.5 GB weights: "
            "needs 64 GB unified memory at minimum, comfortable on 96-128 GB. "
            "Text-only. Never auto-recommended."
        ),
        # Qwen-style template: <think> reasoning, Hermes <tool_call> JSON, and
        # the standard server switch silences reasoning (measured: 2 completion
        # tokens, empty reasoning channel). That is the default policy
        # ("chat_template"), the same one the Gemma 4 and Qwen entries use.
    ),
    ModelChoice(
        key="openjev",
        supports_systemone=True,
        # Pinned to the revision and digest that were verified live.
        revision="10840f375658dea7afc5ff4711127bca8218b560",
        sha256="38b512277edaeec6cd251d146bff6d97bc328fa9db19755401d6e9ab1ccbcba5",
        size_bytes=18973872288,
        name="OpenJev 27B (Q4, decision model)",
        hf_repo="ggml-org/OpenJev-GGUF",
        filename="OpenJev-Q4_K_M.gguf",
        size_gb=19.0,
        active_params="27B dense (Qwen 3.5 base, tuned for typed decisions)",
        architecture="qwen35",
        license="CC BY-NC 4.0 (non-commercial use only)",
        humaneval_pass_at_1=None,
        notes=(
            "A DECISION model, not a coding model. It answers typed questions "
            "about a state (text, JSON, a web page, a screenshot) with a choice, "
            "a yes/no probability or a score, in one forward pass per question, "
            "through the bundled server's /v1/systemone endpoint; `localcode api` "
            "prints the endpoint and key. It also answers on the chat endpoint, "
            "but it was tuned for decisions and is not benchmarked for coding. "
            "The weights are CC BY-NC 4.0: research and non-commercial use only. "
            "Needs ~32 GB unified memory. Pair with the mmproj for screenshot "
            "input. Never auto-recommended."
        ),
        mmproj_filename="mmproj-OpenJev-Q8_0.gguf",
        mmproj_size_gb=0.6,
        mmproj_hf_filename="mmproj-OpenJev-Q8_0.gguf",
    ),
]


def _forced_ram_gb() -> int | None:
    import os
    raw = os.environ.get("LOCALCODE_RAM_GB", "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _system_ram_gb() -> int:
    """Standalone RAM probe — mirrors LocalCodeRuntimeGateway._system_ram_gb so
    the picker doesn't need a runtime instance. Falls back to 16 on any failure
    (conservative — favors the lightweight model).

    LOCALCODE_RAM_GB makes the probe report that figure instead, so the evals
    can run the choices a 16 or 24 GB Mac would make on a larger machine. It
    is an eval knob, not a setting: every RAM-derived choice (quant fit,
    context ceiling, slots) follows it, so set it only on a machine that can
    actually hold what the smaller tier would pick."""
    forced = _forced_ram_gb()
    if forced is not None:
        return forced
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
_NO_AUTO_RECOMMEND_ARCHS = {"diffusion_gemma", "kolibri1"}
# Entries excluded by key, for models that share an architecture with a
# recommended one: OpenJev is a qwen35 like Qwen 3.8, but it is a decision
# model and must never be a first-run coding default. Saluki remains opt-in
# until coding quality has been measured on this stack.
_NO_AUTO_RECOMMEND_KEYS = {"openjev", "saluki"}


def _auto_recommendable(c: "ModelChoice") -> bool:
    return c.architecture not in _NO_AUTO_RECOMMEND_ARCHS and c.key not in _NO_AUTO_RECOMMEND_KEYS

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


def recommend(ram_gb: int | None = None, bandwidth_gbps: float | None = None) -> ModelChoice:
    """Pick the best model for THIS machine's RAM — capability-ranked, never the
    research diffusion architecture, and never a hardcoded default.

    Groups are tried in capability order (see ``_RECOMMEND_ORDER``); each gets
    the quant ``recommended_quant_index`` stars on this Mac, from the repo's
    real file sizes. The first group with a starred quant wins, so a degraded
    or non-fitting quant is never the recommendation while a better model fits.
    Returns the curated (pinned) entry when the starred file is one.
    """
    if ram_gb is None:
        ram_gb = _system_ram_gb()
    for g in _groups_by_capability():
        rows = known_quants(g)
        speed = (lambda size, g=g: estimate_decode_tok_s(size, g.display_name, bandwidth_gbps)) if bandwidth_gbps else None
        idx = recommended_quant_index(rows, g, ram_gb, speed)
        if idx is None:
            continue
        filename, size_gb = rows[idx]
        curated = by_filename_curated(filename)
        return curated or choice_for_quant(g, filename, size_gb)
    # Nothing fits this Mac — smallest production-ready model so the user still
    # gets something runnable rather than an impossible recommendation.
    prod = [c for c in CHOICES if _auto_recommendable(c)]
    return min(prod or CHOICES, key=lambda c: c.size_gb)


def by_filename_curated(filename: str) -> ModelChoice | None:
    return next((c for c in CHOICES if c.filename == filename), None)


def _groups_by_capability() -> list["ModelGroup"]:
    """Auto-recommendable groups, most capable first: the curated ranking, then
    any other production group in catalog order."""
    ordered: list[ModelGroup] = []
    for key in _RECOMMEND_ORDER:
        c = by_key(key)
        g = next((g for g in MODEL_GROUPS if c is not None and g.hf_repo == c.hf_repo), None)
        if g is not None and g not in ordered:
            ordered.append(g)
    for g in MODEL_GROUPS:
        if g not in ordered:
            ordered.append(g)
    return [g for g in ordered if _group_recommendable(g)]


def _group_recommendable(g: "ModelGroup") -> bool:
    if g.architecture in _NO_AUTO_RECOMMEND_ARCHS:
        return False
    curated = [c for c in CHOICES if c.hf_repo == g.hf_repo]
    return bool(curated) and all(c.key not in _NO_AUTO_RECOMMEND_KEYS for c in curated)


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

    supports_systemone: bool = False
    # Vendor-trained speculative drafter, same file for every quant of the model.
    drafter: Drafter | None = None
    # KV cache geometry, the same for every quant of the model, at the default
    # q8_0 K / q8_0 V cache, as llama.cpp allocates it (runtime._kv_bytes_per_token
    # and _kv_fixed_bytes on the GGUF headers): bytes per context token over the
    # layers whose cache grows with the context, and fixed bytes (sliding-window
    # caches, recurrent state). Decides which quants fit a Mac before download.
    kv_bytes_per_token: float = 0.0
    kv_fixed_bytes: int = 0

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
        drafter=DRAFTER_GEMMA_26B,
        kv_bytes_per_token=10880, kv_fixed_bytes=167116800,  # 5 global layers x 2 KV heads; 25 sliding-window layers hold 1536 cells
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
        mmproj_size_gb=0.18,  # 175 MB on HF as of 2026-09 (repo re-exported the projector)
        mmproj_hf_filename="mmproj-F16.gguf",
        drafter=DRAFTER_GEMMA_12B,
        kv_bytes_per_token=8704, kv_fixed_bytes=267386880,  # 8 global layers x 1 KV head; 40 sliding-window layers hold 1536 cells
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
        drafter=DRAFTER_QWEN36,
        kv_bytes_per_token=10880, kv_fixed_bytes=65861059,  # 10 attention layers of 40 (server.log: 10 layers); 62.81 MiB recurrent state
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
        kv_bytes_per_token=34816, kv_fixed_bytes=156893184,  # 16 attention layers of 64; recurrent state of the other 48
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
        kv_bytes_per_token=26624, kv_fixed_bytes=339738624,  # f16 cache: 13 global, 36 sliding-window layers
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
        drafter=DRAFTER_MUSE,
        kv_bytes_per_token=28288, kv_fixed_bytes=0,  # every layer full attention
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
        kv_bytes_per_token=10880, kv_fixed_bytes=167116800,  # Gemma 4 26B-A4B layout
    ),
    ModelGroup(
        key="saluki-27b",
        display_name="Underdog Saluki 27B 1.0",
        maker="Underdog",
        hf_repo="ConwayResearch/Underdog-Saluki-27B-1.0",
        family="qwen",
        architecture="qwen35",
        license="Apache 2.0",
        notes="Mixed 2-bit Qwen 3.8 27B: 7.9 GB weights, with additional RAM needed for context and runtime. Coding quality unmeasured on LocalCode; opt-in. Optional F16 projector enables images. No bundled speculative drafter.",
        mmproj_filename="mmproj-Underdog-Saluki-27B-1.0-F16.gguf",
        mmproj_size_gb=0.928,
        kv_bytes_per_token=34816, kv_fixed_bytes=156893184,  # Qwen 3.8 27B layout
    ),
    ModelGroup(
        key="kolibri-1",
        display_name="Kolibri 1 78B-A3B",
        maker="Aleph Alpha",
        hf_repo="Hob-forge/Kolibri-1-GGUF",
        family="kolibri",
        architecture="kolibri1",
        license="Apache 2.0",
        notes=(
            "Aleph Alpha's open-weight English/German sparse MoE — 78B total, "
            "~3.5B active. Experimental: community-patched architecture and a "
            "community quant. Qwen-style reasoning tags and Hermes tool calls. "
            "Text-only."
        ),
        kv_bytes_per_token=10880, kv_fixed_bytes=55705600,  # 10 global layers, 40 sliding-window layers
    ),
    ModelGroup(
        key="openjev",
        supports_systemone=True,
        display_name="OpenJev 27B (decision model)",
        maker="OpenJev",
        hf_repo="ggml-org/OpenJev-GGUF",
        family="qwen",
        architecture="qwen35",
        license="CC BY-NC 4.0 (non-commercial use only)",
        notes=(
            "Decision model, not a coding model: typed choice / yes-no / score "
            "answers about text, web pages and screenshots through /v1/systemone. "
            "Q4_K_M, Q8_0 and BF16 quants. Non-commercial licence."
        ),
        mmproj_filename="mmproj-OpenJev-Q8_0.gguf",
        mmproj_size_gb=0.6,
        mmproj_hf_filename="mmproj-OpenJev-Q8_0.gguf",
        kv_bytes_per_token=34816, kv_fixed_bytes=156893184,  # Qwen 3.5 27B layout: 16 attention layers of 64
    ),
]


# Exact file sizes (bytes) of the model-weight quants each vendor repo ships,
# from the Hugging Face API (2026-10-18). The picker's live listing replaces
# these when it has one; this copy lets the recommendation (and its ★) work
# offline, on a first run, and in tests with the same real sizes.
KNOWN_QUANT_BYTES: dict[str, dict[str, int]] = {
    "unsloth/Muse-Glimmer-30B-GGUF": {
        "Muse-Glimmer-30B-UD-IQ2_XXS.gguf": 10746373152,
        "Muse-Glimmer-30B-UD-IQ2_XS.gguf": 11513104416,
        "Muse-Glimmer-30B-UD-IQ2_M.gguf": 12255421472,
        "Muse-Glimmer-30B-UD-Q2_K_XL.gguf": 12444212256,
        "Muse-Glimmer-30B-UD-IQ3_XXS.gguf": 13130658848,
        "Muse-Glimmer-30B-UD-Q3_K_XL.gguf": 13360983072,
        "Muse-Glimmer-30B-UD-IQ3_M.gguf": 14122705696,
        "Muse-Glimmer-30B-UD-Q4_K_XL.gguf": 15878222368,
        "Muse-Glimmer-30B-UD-Q5_K_M.gguf": 19194274848,
        "Muse-Glimmer-30B-UD-Q5_K_L.gguf": 19768527904,
        "Muse-Glimmer-30B-UD-Q5_K_XL.gguf": 21789618976,
        "Muse-Glimmer-30B-UD-Q6_K_XL.gguf": 26265362976,
        "Muse-Glimmer-30B-Q8_0.gguf": 29612957984,
        "Muse-Glimmer-30B-UD-Q8_K_XL.gguf": 32300651040,
    },
    "unsloth/North-Mini-Code-1.0-GGUF": {
        "North-Mini-Code-1.0-UD-IQ1_M.gguf": 9379799136,
        "North-Mini-Code-1.0-UD-IQ2_XXS.gguf": 9782452320,
        "North-Mini-Code-1.0-UD-IQ2_M.gguf": 9863438432,
        "North-Mini-Code-1.0-UD-Q2_K_XL.gguf": 10480001120,
        "North-Mini-Code-1.0-UD-IQ3_XXS.gguf": 11708375136,
        "North-Mini-Code-1.0-UD-IQ3_S.gguf": 12765339744,
        "North-Mini-Code-1.0-UD-Q3_K_M.gguf": 14213013600,
        "North-Mini-Code-1.0-UD-Q3_K_XL.gguf": 14343037024,
        "North-Mini-Code-1.0-UD-IQ4_XS.gguf": 15230132320,
        "North-Mini-Code-1.0-UD-IQ4_NL.gguf": 15532122208,
        "North-Mini-Code-1.0-UD-Q4_K_S.gguf": 18045558880,
        "North-Mini-Code-1.0-MXFP4_MOE.gguf": 18662121568,
        "North-Mini-Code-1.0-UD-Q4_K_M.gguf": 19203186784,
        "North-Mini-Code-1.0-UD-Q4_K_XL.gguf": 19253518432,
        "North-Mini-Code-1.0-UD-Q5_K_S.gguf": 21619105888,
        "North-Mini-Code-1.0-UD-Q5_K_M.gguf": 22946603104,
        "North-Mini-Code-1.0-UD-Q5_K_XL.gguf": 23000080480,
        "North-Mini-Code-1.0-UD-Q6_K.gguf": 25513517152,
        "North-Mini-Code-1.0-UD-Q6_K_XL.gguf": 27853938784,
        "North-Mini-Code-1.0-Q8_0.gguf": 32437264480,
        "North-Mini-Code-1.0-UD-Q8_K_XL.gguf": 33209933920,
    },
    "unsloth/Qwen3.6-35B-A3B-GGUF": {
        "Qwen3.6-35B-A3B-UD-IQ1_M.gguf": 10047749088,
        "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf": 10756586464,
        "Qwen3.6-35B-A3B-UD-IQ2_M.gguf": 11522702304,
        "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf": 12290628576,
        "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf": 13211155424,
        "Qwen3.6-35B-A3B-UD-IQ3_S.gguf": 13676723168,
        "Qwen3.6-35B-A3B-UD-Q3_K_S.gguf": 15359196128,
        "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf": 16600710112,
        "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf": 16845511648,
        "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf": 17730509792,
        "Qwen3.6-35B-A3B-UD-IQ4_NL.gguf": 18040888288,
        "Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf": 19500506080,
        "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf": 20893015008,
        "Qwen3.6-35B-A3B-MXFP4_MOE.gguf": 21706144736,
        "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf": 22134528992,
        "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf": 22360456160,
        "Qwen3.6-35B-A3B-UD-Q5_K_S.gguf": 24942050272,
        "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf": 26456194016,
        "Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf": 26592508896,
        "Qwen3.6-35B-A3B-UD-Q6_K.gguf": 29308320736,
        "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf": 31843777504,
        "Qwen3.6-35B-A3B-Q8_0.gguf": 36903140320,
        "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf": 38451182560,
    },
    "unsloth/Qwen3.8-27B-GGUF": {
        "Qwen3.8-27B-UD-IQ1_S.gguf": 6192222208,
        "Qwen3.8-27B-UD-IQ1_M.gguf": 6729166848,
        "Qwen3.8-27B-UD-IQ2_XXS.gguf": 7266070528,
        "Qwen3.8-27B-UD-IQ2_S.gguf": 8371970048,
        "Qwen3.8-27B-UD-Q2_K_XL.gguf": 9828981664,
        "Qwen3.8-27B-UD-IQ3_XXS.gguf": 10934860704,
        "Qwen3.8-27B-UD-IQ3_S.gguf": 12040883104,
        "Qwen3.8-27B-UD-Q3_K_XL.gguf": 13146393504,
        "Qwen3.8-27B-UD-IQ4_XS.gguf": 14252845984,
        "Qwen3.8-27B-UD-Q4_K_S.gguf": 15358213024,
        "Qwen3.8-27B-Q4_0.gguf": 16056478688,
        "Qwen3.8-27B-UD-Q4_K_M.gguf": 16464440224,
        "Qwen3.8-27B-Q4_1.gguf": 17540705248,
        "Qwen3.8-27B-UD-Q4_K_XL.gguf": 17559178144,
        "Qwen3.8-27B-UD-Q5_K_S.gguf": 18665753504,
        "Qwen3.8-27B-UD-Q5_K_M.gguf": 19771509664,
        "Qwen3.8-27B-UD-Q5_K_XL.gguf": 20876938144,
        "Qwen3.8-27B-UD-Q6_K.gguf": 21983677344,
        "Qwen3.8-27B-UD-Q6_K_M.gguf": 23088409504,
        "Qwen3.8-27B-UD-Q6_K_L.gguf": 24193919904,
        "Qwen3.8-27B-UD-Q6_K_XL.gguf": 25299061664,
        "Qwen3.8-27B-UD-Q8_K_L.gguf": 28045695904,
        "Qwen3.8-27B-Q8_0.gguf": 29047086048,
        "Qwen3.8-27B-UD-Q8_K_XL.gguf": 31457991680,
    },
    "unsloth/gemma-4-12b-it-GGUF": {
        "gemma-4-12b-it-UD-IQ2_M.gguf": 4213353280,
        "gemma-4-12b-it-UD-IQ3_XXS.gguf": 4639716160,
        "gemma-4-12b-it-UD-Q2_K_XL.gguf": 4661419840,
        "gemma-4-12b-it-Q3_K_S.gguf": 5135014720,
        "gemma-4-12b-it-Q3_K_M.gguf": 5693872960,
        "gemma-4-12b-it-UD-Q3_K_XL.gguf": 6022684480,
        "gemma-4-12b-it-IQ4_XS.gguf": 6375734080,
        "gemma-4-12b-it-IQ4_NL.gguf": 6716357440,
        "gemma-4-12b-it-Q4_0.gguf": 6738475840,
        "gemma-4-12b-it-Q4_K_S.gguf": 6764526400,
        "gemma-4-12b-it-Q4_K_M.gguf": 7121861440,
        "gemma-4-12b-it-UD-Q4_K_XL.gguf": 7366423360,
        "gemma-4-12b-it-Q4_1.gguf": 7397604160,
        "gemma-4-12b-it-Q5_K_S.gguf": 8204680000,
        "gemma-4-12b-it-Q5_K_M.gguf": 8413576000,
        "gemma-4-12b-it-UD-Q5_K_XL.gguf": 8606021440,
        "gemma-4-12b-it-Q6_K.gguf": 9786022720,
        "gemma-4-12b-it-UD-Q6_K_XL.gguf": 10685012800,
        "gemma-4-12b-it-Q8_0.gguf": 12669647680,
        "gemma-4-12b-it-UD-Q8_K_XL.gguf": 13637327680,
        "gemma-4-12b-it-BF16.gguf": 23832066656,
    },
    "unsloth/gemma-4-26B-A4B-it-GGUF": {
        "gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf": 9922480608,
        "gemma-4-26B-A4B-it-UD-IQ2_M.gguf": 10014755296,
        "gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf": 10546934240,
        "gemma-4-26B-A4B-it-UD-IQ3_S.gguf": 11289671136,
        "gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf": 11416548832,
        "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf": 12728497888,
        "gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf": 12907280096,
        "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf": 13597177568,
        "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf": 13613037280,
        "gemma-4-26B-A4B-it-UD-Q4_K_S.gguf": 16487610080,
        "gemma-4-26B-A4B-it-MXFP4_MOE.gguf": 16551048928,
        "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf": 16947541728,
        "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf": 17010980576,
        "gemma-4-26B-A4B-it-UD-Q5_K_S.gguf": 18850707168,
        "gemma-4-26B-A4B-it-UD-Q5_K_M.gguf": 21150365408,
        "gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf": 21217769184,
        "gemma-4-26B-A4B-it-UD-Q6_K.gguf": 23172478688,
        "gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf": 23295391456,
        "gemma-4-26B-A4B-it-Q8_0.gguf": 26859861728,
        "gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf": 27636232928,
    },
}


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
        r"(?:^|[-_.])((?:UD-)?(?:I?Q\d+[A-Z0-9_]*(?:-mix)?|BF16|F16|F32))$",
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
    # Reasoning policy is a property of the MODEL, not the quant: copy it from
    # the group's canonical catalog entry so a browsed quant of Muse Glimmer is
    # still "always" and a browsed Qwen quant is still "server".
    canonical = next((c for c in CHOICES if c.hf_repo == group.hf_repo), None)
    return ModelChoice(
        key=f"{group.key}:{label}",
        revision=canonical.revision if canonical and canonical.filename == filename else "main",
        sha256=canonical.sha256 if canonical and canonical.filename == filename else None,
        size_bytes=canonical.size_bytes if canonical and canonical.filename == filename else None,
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
        supports_systemone=group.supports_systemone,
        drafter=group.drafter,
        reasoning_control=canonical.reasoning_control if canonical else "chat_template",
        reasoning_budget_tokens=canonical.reasoning_budget_tokens if canonical else 8192,
        preserves_reasoning=canonical.preserves_reasoning if canonical else True,
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


# ---------------------------------------------------------------------------
# Fit and quant recommendation (shared by recommend(), the picker and the
# launcher). Memory math lives in gpu_memory.
# ---------------------------------------------------------------------------


def known_quants(group: "ModelGroup") -> list[tuple[str, float]]:
    """(filename, size_gb) of the group's quants, smallest first: the recorded
    repo listing plus the curated entries."""
    rows = {f: b / 1e9 for f, b in KNOWN_QUANT_BYTES.get(group.hf_repo, {}).items()}
    for c in CHOICES:
        if c.hf_repo == group.hf_repo:
            rows.setdefault(c.filename, (c.size_bytes / 1e9) if c.size_bytes else c.size_gb)
    return sorted(rows.items(), key=lambda r: r[1])


def quant_bits(label: str) -> float:
    """Nominal bits per weight of a quant label ("UD-IQ3_XXS" -> 3, "BF16" -> 16)."""
    import re
    t = label.upper().removeprefix("UD-")
    if t.startswith(("BF16", "F16")):
        return 16.0
    if t.startswith("F32"):
        return 32.0
    if t.startswith("MXFP4"):
        return 4.0
    m = re.match(r"I?Q(\d)", t)
    if not m:
        return 16.0
    bits = float(m.group(1))
    if bits == 2 and t.startswith("Q2_K"):
        return 2.5  # K-quant 2-bit mixes keep attention and embeddings higher
    return bits


def is_degraded(label: str, group: "ModelGroup | None" = None) -> bool:
    """Below Unsloth's UD-Q2_K_XL: 1-bit and IQ2 quants of models under 40B.
    That is where agent quality falls off a cliff (Qwen 3.8: Divergence-300
    25% at UD-Q2_K_XL vs 8-10% at UD-IQ2_S; 2-bit experts corrupt tool-call
    JSON), so the picker marks them and never stars them."""
    t = label.upper().removeprefix("UD-")
    if t.startswith(("IQ1", "Q1", "TQ1")):
        return True
    if not t.startswith("IQ2"):
        return False
    total_b = _parse_total_active_b(group.display_name)[0] if group is not None else 0.0
    return not total_b or total_b < 40


def _quality_tier(label: str, group: "ModelGroup") -> int:
    """0 degraded, 1 low-bit (2.5-3.x), 2 near-lossless (4-bit and up)."""
    if is_degraded(label, group):
        return 0
    return 2 if quant_bits(label) >= 4 else 1


def quant_plan(group: "ModelGroup", size_gb: float, ram_gb: float) -> dict:
    """How a quant of `group` would run on a Mac with `ram_gb`: its fit badge
    (weights + a working context + compute in the GPU working set) and whether
    its speculative drafter also fits. The launcher applies the same rule, so a
    drafter that does not fit is left out rather than squeezing the context."""
    from . import gpu_memory
    weights = int(size_gb * 1e9)
    bpt, fixed = group.kv_bytes_per_token, group.kv_fixed_bytes
    badge = gpu_memory.fit(weights, ram_gb, bpt, fixed)
    drafter = group.drafter
    with_drafter = False
    if drafter is not None and badge == "fits":
        with_drafter = gpu_memory.fit(weights + drafter.size_bytes, ram_gb, bpt, fixed) == "fits"
    return {"fit": badge, "drafter": with_drafter,
            "size_gb": size_gb + (drafter.size_gb if with_drafter and drafter else 0.0)}


def recommended_quant_index(rows, group: "ModelGroup", ram_gb: float,
                            speed=None, min_speed_fraction: float = 0.5) -> int | None:
    """Index of the quant to star among `rows` ((filename or label, size_gb)
    pairs, any object with .filename/.size_gb works too), or None when no
    non-degraded quant fits this Mac.

    1. Only quants that fit (gpu_memory.fit) and are not degraded.
    2. Below FULL_PRECISION_MIN_RAM_GB, at most NEAR_LOSSLESS_BITS bits.
    3. Keep the speculative drafter, unless dropping it reaches a better
       quality tier (e.g. 2.5-3 bit -> 4 bit): a near-lossless quant without
       drafting beats a low-bit one with it; within a tier the drafter's
       1.5-2x decode wins.
    4. Among the rest, the largest file that still decodes at least
       `min_speed_fraction` of the fastest one (`speed(size_gb)` -> tok/s).
    """
    from .model_config import FULL_PRECISION_MIN_RAM_GB, NEAR_LOSSLESS_BITS

    def parts(row):
        if isinstance(row, tuple):
            return row[0], row[1]
        return getattr(row, "filename", getattr(row, "label", "")), row.size_gb

    cands = []
    for i, row in enumerate(rows):
        name, size = parts(row)
        label = _quant_label(name) if name.lower().endswith(".gguf") else name
        if is_degraded(label, group):
            continue
        plan = quant_plan(group, size, ram_gb)
        if plan["fit"] == "fits":
            cands.append((i, label, size, plan["drafter"]))
    if not cands:
        return None
    if ram_gb < FULL_PRECISION_MIN_RAM_GB:
        capped = [c for c in cands if quant_bits(c[1]) <= NEAR_LOSSLESS_BITS]
        cands = capped or cands
    best_tier = max(_quality_tier(c[1], group) for c in cands)
    if group.drafter is not None:
        drafted = [c for c in cands if c[3]]
        if drafted and max(_quality_tier(c[1], group) for c in drafted) == best_tier:
            cands = drafted
        else:
            cands = [c for c in cands if _quality_tier(c[1], group) == best_tier]
    if speed is not None:
        speeds = {c[0]: (speed(c[2]) or 0) for c in cands}
        fastest = max(speeds.values()) or 0
        if fastest > 0:
            responsive = [c for c in cands if speeds[c[0]] >= fastest * min_speed_fraction]
            cands = responsive or cands
    return max(cands, key=lambda c: c[2])[0]
