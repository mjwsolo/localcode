# Fork-local patches on top of upstream `llama.cpp`

`llama-cpp-turboquant/` is a vendored copy of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
plus a small set of fork-local changes. This file is the durable inventory of
those changes, so that bumping to a newer upstream is a mechanical replay
instead of an archaeology project.

The machine-readable form of this inventory lives at the repo root:

| file | meaning |
| --- | --- |
| `patches/PINNED_UPSTREAM` | the upstream commit the series applies to |
| `patches/000N-*.patch` | numbered patches, applied in order with `git apply` |

**Pinned upstream:** `3f545beccee69d9975f466ec7e45fd9aacd8ba90` (2026-08-22,
`vulkan : added the PAD_REFLECT_1D operation (#26586)`).

**Previous pin:** `c08d28d08871715fd68accffaeeb76ddcaede658` (2026-04-05). That
commit was never recorded anywhere; it was recovered for this bump by
blob-matching the vendored tree against upstream history. Do not let that
happen again — update `patches/PINNED_UPSTREAM` on every bump.

## How to bump

```sh
git clone https://github.com/ggml-org/llama.cpp upstream && cd upstream
git checkout <new-sha>
for p in /path/to/patches/*.patch; do git apply --index "$p" || echo "STALE: $p"; done
```

Then copy the result over `llama-cpp-turboquant/`, honouring the vendoring
exclusions below, update `patches/PINNED_UPSTREAM`, and regenerate the patches
with `git diff <new-sha> HEAD -- <paths>`.

### Vendoring exclusions

The vendored copy deliberately omits these upstream top-level paths:
`tests/`, `examples/`, `docs/`, `.github/`, `models/`, `media/`, `grammars/`,
`benches/`, `pocs/`, `ci/`, `build-xcframework.sh`.

## Hard gates before any bump is considered done

1. Static build links nothing from Homebrew:
   `otool -L build/bin/llama-server | grep /opt/homebrew` must be **empty**.
   A previous rebuild linked Homebrew `libssl` and would have broken every user
   without Homebrew installed.
2. `bash dev/verify_models.sh` — **all 8** bundled-server configs pass
   (load + generate + tool-call, including the `turbo4` KV path).

A bump that regresses a working model is strictly worse than staying stale.

---

## The patches

### `0001-turboquant.patch` — TurboQuant KV cache + TQ weight quants

The product's headline feature. Zero upstream equivalent.

Two sub-features that are **inseparable in practice** — they share the WHT
rotation machinery, the ggml type-traits tables, and the Metal/CUDA kernel
files, so splitting them yields patches that do not independently compile:

**(a) TurboQuant KV cache** — `--cache-type-k/-v turbo2|turbo3|turbo4`
- `GGML_TYPE_TURBO2_0` / `TURBO3_0` / `TURBO4_0`, `GGML_OP_TURBO_WHT`
- `ggml/src/ggml-metal/turbo-matrices.h`, `turbo-wht.h`, `ggml-metal.metal`
- `ggml/src/ggml-turbo-quant.c`, `src/turbo-rotation-data{,-32}.h`
- `src/llama-kv-cache.{h,cpp}`, `src/llama-graph.cpp`, `src/llama-context.cpp`
- CLI plumbing in `common/arg.cpp`, `tools/llama-bench`

**(b) TQ3_1S / TQ4_1S weight quants** — WHT-rotated Lloyd-Max
- `GGML_TYPE_TQ3_1S` / `TQ4_1S`, `LLAMA_FTYPE_MOSTLY_TQ3_1S` / `TQ4_1S`
- `ggml/src/ggml-quants.{c,h}`, `src/llama-quant.cpp`, `tools/quantize`

#### ⚠ Enum renumbering — read this before touching the type ids

A `ggml_type` id is **serialised into the GGUF tensor header**, and a
`llama_ftype` is serialised as `general.file_type`. If upstream takes an id we
already use, a merge does not fail to compile — it silently *reinterprets
tensor data*.

That is exactly what happened at this bump: the fork held 41/42/43 for the
TURBO types and 44/45 for the TQ types, and upstream took 41 (`GGML_TYPE_Q1_0`)
and 42 (`GGML_TYPE_Q2_0`).

The fork types were therefore moved into **reserved high blocks** with ~20 ids
of runway, so this class of collision cannot recur:

| symbol | old | new |
| --- | --- | --- |
| `GGML_TYPE_TURBO3_0` | 41 | **64** |
| `GGML_TYPE_TURBO4_0` | 42 | **65** |
| `GGML_TYPE_TURBO2_0` | 43 | **66** |
| `GGML_TYPE_TQ3_1S` | 44 | **67** |
| `GGML_TYPE_TQ4_1S` | 45 | **68** |
| `GGML_TYPE_COUNT` | 46 | **69** |
| `LLAMA_FTYPE_MOSTLY_TQ3_1S` | 43 | **200** |
| `LLAMA_FTYPE_MOSTLY_TQ4_1S` | 44 | **201** |

`gguf-py/gguf/constants.py` mirrors all of the above and must be kept in sync.

The three TURBO types are KV-cache-only and never appear in a GGUF, so
renumbering them is free. TQ3_1S/TQ4_1S *are* weight types, but no catalog
model and no code in localcode's `src/` uses them, so renumbering them now was
free too — and it was the only cheap moment to do it. **Any pre-existing
TQ3_1S/TQ4_1S GGUF must be requantized.** None are known to exist.

#### Known gaps deliberately accepted at the 2026-08-22 bump

- **Metal TQ-weight `mul_mm` fast path dropped.** Upstream restructured the
  `mul_mm` dispatch (pipelines now carry `nr0`/`nr1`/`nsg`); the fork's
  "rotate activations → rotated mul_mm → un-rotate" path did not survive.
  TQ3_1S/TQ4_1S weights now take the generic path. Safe because no shipped
  model uses these types. `get_pipeline_mul_mm_tq_rotated` and
  `get_pipeline_tq3_rotate_act` are no longer called.
- **Upstream's own activation rotation is defaulted OFF.** Master gained
  graph-level `attn_rot_k`/`attn_rot_v` (default on for quantized KV). This
  fork rotates inside the KV kernels, so upstream's rotation would be a second,
  redundant rotation. `LLAMA_ATTN_ROT_DISABLE` now defaults to `true`;
  set it to `0` to opt back in. Worth re-evaluating for non-turbo KV types.
- **CUDA RDNA `DKQ=640` flash-attention tuning** now reuses upstream's
  `576,512` RDNA numbers rather than the fork's own tuned values.
- **Vulkan `rte` shader variants** for TURBO3_0 are gone; upstream removed the
  `float_controls_rte_fp16` split they hung off.

Only the Metal path is compiled in the shipped macOS wheel
(`-DGGML_METAL=ON`), so the CUDA/Vulkan/WebGPU notes above are not
product-affecting today.

### `0002-server-no-context-cap.patch` — do not cap `n_ctx` to the training context

`tools/server/server-context.cpp`. Upstream clamps a requested slot context to
the model's training context. localcode configures rope scaling
(`--rope-scale` / `--rope-scaling yarn`) to run beyond it, so the clamp is
replaced with a warning. No upstream equivalent.

### `0003-hunyuan-ocr-mtmd.patch` — HunyuanOCR vision model

`tools/mtmd/models/hunyuanocr.cpp`. Fork-local; **partially superseded** — the
*chat template* half of this support was absorbed by upstream, which
generalised it to `LLM_CHAT_TEMPLATE_HUNYUAN_VL` ("tencent/HunyuanOCR &
tencent/HunyuanVL"). Only the mtmd vision model file remains fork-local.
Re-check on the next bump whether upstream has taken this too.

### `0004-fork-local-bench-scripts.patch` — bench artifacts

`scripts/turbo-quality-gate.sh`, `scripts/bench-smem-m5.sh` and two recorded
benchmark outputs. Development aids, not product code. Safe to drop if they
ever conflict.

### `0005-diffusion-gemma-pr24423.patch` — DiffusionGemma (upstream PR #24423)

**Status: OPEN upstream PR, unmerged.** [ggml-org/llama.cpp#24423](https://github.com/ggml-org/llama.cpp/pull/24423),
vendored at head SHA `daca8075d871483545dd85d58ce11970b304b541` (the same SHA
`src/localcode/bootstrap.py` pins as `_DIFFUSION_PR_SHA`).
**DROP WHEN MERGED** — once the PR lands, bump past it and delete this patch.

Adds the `diffusion-gemma` architecture (`src/models/diffusion-gemma.cpp`,
`src/models/gemma4-common.h`, `LLM_ARCH_DIFFUSION_GEMMA` plumbing in
`llama-arch`/`llama-model`/`llama-context`, `llama_set_causal_attn` support,
`diffusion.*` GGUF keys in `gguf-py`), the entropy-bound block-diffusion
decoder, the `--diffusion-eb*` / `--diffusion-blocks` / `--diffusion-kv-cache`
/ `--diffusion-gpu-sampling*` CLI flags, and a CUDA on-device sampling kernel
(`ggml-cuda/diffusion-sampling.cu`, not compiled in the Metal wheel).

Why this is a patch and not "just use llama-server": the PR touches
`tools/server` **zero** times. Diffusion generates by iterative denoising of a
whole canvas, not by autoregressive decode, so `llama-server`'s slot/completion
loop has nowhere to host it. The PR instead ships its own programs, and so do we:

| binary | what it is |
| --- | --- |
| `llama-diffusion-cli` | one-shot generation; what `runtime_diffusion.py` drives today |
| `llama-diffusion-gemma-visual-server` | persistent process: loads once, prints `READY`, then one JSON chat request per stdin line, streams `F`/`C`/`STATS`/`DONE` records on stdout |
| `llama-diffusion-gemma-server` | persistent **raw-logits** forward server (one forward pass per request, returns `C × n_vocab` float32). For external denoising drivers; not a text server |

#### Relocation: `examples/diffusion*` → `tools/diffusion*`

Upstream keeps these under `examples/`, which this vendored tree omits
entirely (see *Vendoring exclusions*). They are product code for us, so they
live under `tools/diffusion/` and `tools/diffusion-gemma-server/` and are wired
into `tools/CMakeLists.txt`, exactly like `tools/server`. The shipped build
therefore stays `-DLLAMA_BUILD_EXAMPLES=OFF` and one cmake invocation produces
`llama-server` and all three diffusion binaries. Everything they need is in
this tree; nothing is fetched at build or run time.

`tools/diffusion/` is the upstream `examples/diffusion/` at the pinned SHA plus
the PR's hunks; `diffusion-gemma-eval` was dropped (eval harness, not product).

#### Fork-local edits on top of the PR

- `common/arg.cpp`: `-no-cnv` gained `LLAMA_EXAMPLE_DIFFUSION` (the PR put it
  on the `-cnv` option; upstream had since narrowed that option's example set,
  so the hunk needed re-targeting).
- `diffusion-gemma-visual-server.cpp`: `chat.h` now takes `common_json`, not
  nlohmann; the request is parsed once more with `common_json::parse` for the
  chat-template calls. Re-check whether the PR has caught up on the next bump.

#### Verified (2026-08-23, M-series, `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`)

`llama-diffusion-cli -no-cnv -ngl 99 -n 2048` and the `--diffusion-eb off`
retry both produce coherent text; the visual server answered two consecutive
requests from one resident process. `otool -L` on all four binaries shows only
system frameworks.

---

## Retired at the 2026-08-22 bump

Patches that no longer exist because upstream implemented the same thing,
usually better. **Do not resurrect these.**

### qwen35 MTP guard — retired, upstream is now strictly better

This was the fork's most fragile patch and the one that broke Qwen 3.8 27B on a
previous bump attempt (`missing tensor 'blk.64.ssm_conv1d.weight'`). It marked
the trailing Multi-Token-Prediction block as not-a-transformer-layer by hand,
in the giant `switch (arch)` blocks of `llama-model.cpp`.

Upstream now models MTP as a first-class concept:

- `hparams.n_layer_nextn`, and `hparams.n_layer()` (effective, excludes MTP)
  versus `hparams.n_layer_all`
- `is_recr_impl[i] = (i < hparams.n_layer()) && ...` — literally our guard
- `src/models/qwen35.cpp::load_block_mtp()` actually *loads* the `nextn.*`
  tensors instead of merely claiming them
- `LLM_GRAPH_TYPE_DECODER_MTP` + `graph_mtp` — it can *run* the MTP head

Upstream also independently reclassified the `NEXTN_*` tensors as
`LLM_TENSOR_LAYER_REPEATING`, which was the other half of our patch.

### Cherry-picks absorbed by upstream

- **Gemma 4 tokenizer fixes** (`add_bos` override, BPE byte-fallback in
  `token_to_byte`, `LLAMA_TOKEN_ATTR_BYTE` in `token_to_piece`) — all in master.
- **HunyuanOCR chat template** — generalised upstream to `HUNYUAN_VL`.

### Reverse drift discarded

Hunks that looked like fork changes but were only the vendored tree being
*older* than upstream. These were dropped in favour of upstream:
`src/unicode.cpp` (would have deleted upstream's newline regex splitter),
`ggml/src/ggml-webgpu/ggml-webgpu.cpp`, `tools/server/server-task.h` and the
timing-field initialisers in `server-context.cpp`, and `common/download.cpp`
(upstream's version adds GGUF-split handling).

This category is the reason a bump must diff against a *known* base commit
rather than against whatever upstream happens to be today.

## Watch list for the next bump

- **Muse Glimmer thinking tags.** Upstream's
  `common_chat_params_init_muse_glimmer()` sets `supports_thinking = true` and
  a PEG grammar for ` to=self<|message|>` … `<|eom|>`, but does **not** set
  `thinking_start_tag` / `thinking_end_tags`. Upstream PR **#27475** proposes
  exactly that. If Muse returns empty `content`, that is the first thing to
  check; add it as a fork patch and drop it when #27475 merges.
- **`ggml_mul_mat_aux()`** in `src/llama-kv-cache.cpp` may now be dead —
  upstream replaced its call site with `llama_mul_mat_hadamard()`.
- **CUDA `GET_ROWS` claims TQ4_1S/TQ3_1S support** in `supports_op`, but
  `ggml-cuda/getrows.cu` has no TQ case and would `GGML_ABORT`. Pre-existing,
  not introduced by the bump, but it should be decided one way or the other.
