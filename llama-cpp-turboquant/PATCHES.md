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

**Pinned upstream:** `338fce6511c744893cc0d3d958b275f40552c7a4` (2026-10-27,
master, `HIP/CUDA: remove pragma unroll from loops in conv2d ... (#31255)`,
310 commits after `62df1815b`, the previous pin).

**Previous pins:** `62df1815bc053066761f4eac58e288f23778e164` (2026-10-14),
`5c8854fd88319125125dc4b836c98b1b215b0f23` (2026-10-11),
`62a6f74bab748b1755f4cd8dfc0871caf865c6c8` (2026-10-11), `836d57176dc699a726c55418e4f96b8ca628e1bf` (2026-10-03),
`3f545beccee69d9975f466ec7e45fd9aacd8ba90` (2026-08-22), and before it `c08d28d08871715fd68accffaeeb76ddcaede658` (2026-04-05). That
commit was never recorded anywhere; it was recovered for this bump by
blob-matching the vendored tree against upstream history. Do not let that
happen again — update `patches/PINNED_UPSTREAM` on every bump.

## How to bump

`scripts/bump_upstream.sh --ref <new-sha> --sync` does the whole thing when
every patch still applies: replay, source assertions, static build,
self-containment check, sync into this directory (honouring the vendoring
exclusions below), and the `patches/PINNED_UPSTREAM` update.

When a patch no longer applies, do **not** hand-edit reject hunks out of
`git apply`. It has no merge base, so six weeks of drift produced 26 rejects
across 12 files at the 2026-10-03 bump. Turn the series into real history and
let git do a 3-way merge instead; that cut the same bump to 10 conflicted files,
most of them one hunk each:

```sh
git clone https://github.com/ggml-org/llama.cpp upstream && cd upstream
git checkout -b fork-old "$(cat /path/to/patches/PINNED_UPSTREAM)"
# If `upstream` was cloned from a local blobless clone, git drops the promisor
# settings and the checkout fails with "unable to read sha1 file". Restore them
# first: git config remote.origin.promisor true &&
#        git config remote.origin.partialclonefilter blob:none
# `git apply --3way` cannot be used with these patches (no preimage blobs);
# the fork-old -> rebase route below is the one that works.
for p in /path/to/patches/*.patch; do
  git apply --index "$p" && git commit -qm "$(basename "$p" .patch)"
done
git rebase <new-sha>            # resolve, `git add`, `git rebase --continue`
# build + test here, committing fixes with `git commit --fixup=<patch commit>`
GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash <new-sha>
# one plain diff per commit, in order (no author headers end up in the repo):
git diff --binary <new-sha> <c1> > 0001-….patch ; git diff --binary <c1> <c2> > 0002-….patch ; …
```

Then run `scripts/bump_upstream.sh --ref <new-sha> --sync --force` so the
regenerated series is proven to replay from a pristine clone before it lands.
`UPSTREAM_REPO=file:///path/to/that/clone` avoids a second network clone.

### Vendoring exclusions

The vendored copy deliberately omits these upstream top-level paths:
`tests/`, `examples/`, `docs/`, `.github/`, `models/`, `media/`, `grammars/`,
`benches/`, `pocs/`, `ci/`, `build-xcframework.sh`.

## Hard gates before any bump is considered done

1. Static build links nothing from Homebrew:
   `otool -L build/bin/llama-server | grep /opt/homebrew` must be **empty**.
   A previous rebuild linked Homebrew `libssl` and would have broken every user
   without Homebrew installed. `-DLLAMA_OPENSSL=OFF` is what prevents it
   (upstream defaults it ON); the bump script was missing that flag until the
   2026-10-03 bump, where this gate caught exactly that binary.
2. `bash dev/verify_models.sh` — **all 11** bundled-server configs pass
   (load + generate + tool-call, including the `turbo4` KV path and
   DiffusionGemma on the same binary).

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

#### Metal layout since the 2026-10-03 bump

Upstream deleted the monolithic `ggml-metal.metal` (#26561, #29329): Metal is
now one library per `ggml/src/ggml-metal/kernels/<name>.metal`, compiled in
parallel, with shared code in `kernels/*.h`. The TurboQuant shader delta
(2,015 lines in the old file) was redistributed, not rewritten:

| where | what |
| --- | --- |
| `kernels/common.h` | WHT sign tables, `turbo_fwht_128*`, `turbo_rotate_*`, all centroid / midpoint LUTs (every kernel file sees them) |
| `kernels/quantize.h` | `quantize_turbo{2,3,4}_0` |
| `kernels/dequantize.h` | `dequantize_turbo{2,3,4}_0` (+ `_t4`), and the TQ3_1S / TQ4_1S constants and dequantizers |
| `kernels/turbo.metal` (new) | `kernel_turbo_wht`, `kernel_turbo4_dequant_f16`, `kernel_set_rows_turbo{,2,4}` and their instantiations |
| `kernels/fa_turbo.metal` (new) | non-vec flash-attention instantiations for every turbo / turbo-q8_0 K/V pair |
| `kernels/fa_vec_turbo.metal` (new) | the same for the vec kernel, baseline config only (see below) |
| `kernels/fa_vec_common.metal` | the `TURBO_SPARSE_V` skip in the quantized-V loop |
| `kernels/mul_mv.metal`, `mul_mm.metal`, `quantize.metal` | TQ3_1S / TQ4_1S mat-vec kernels and `mul_mm` / `mul_mm_id` / `mul_mv_ext` / `mul_mv_id` / `cpy` / `get_rows` instantiations; larger `mul_mm_id_map0` tiles (32…256) |

**A new kernel file must be registered twice**: in `METALLIB_KERNEL_SOURCES`
(`ggml-metal/CMakeLists.txt`) and in the `GGML_METAL_LIBS` X-macro
(`ggml-metal-device.m`). The bump script asserts both for the three turbo files.

Host-side decisions that differ from the pre-bump fork:

- **Flash-attention pipeline names.** The fork used to rename every FA kernel to
  `k<K>_v<V>`. Now symmetric K/V keeps upstream's single-type name
  (`kernel_flash_attn_ext_turbo4_dk128_dv128`) and only mixed pairs encode both
  (`…_kturbo4_vq8_0_…`), so upstream's own instantiation files are untouched.
- **Vec kernel: baseline only.** Upstream instantiates tuned `_q<Q>_ne<NE>`
  variants per dtype and picks them from `ggml_metal_tuning::fa_vec_pick`. The
  table has no turbo entries, so the pick always falls back to the baseline
  (`Q=1`, `NE=fa_vec_baseline_ne(dk,dv)`), which is what `fa_vec_turbo.metal`
  instantiates. If the baseline NE table changes upstream, regenerate that file.
- **`TURBO_SPARSE_V`** now skips a cache position only when the attention weight
  is negligible for *every* query in the threadgroup (upstream batches `Q`
  queries per threadgroup; the old check assumed one).
- **`use_kv_f16`** (upstream's dequantize-KV-to-F16 prefill path, #27390) is
  disabled for mixed K/V pairs; it assumes one type for both.
- **`FC_TURBO_WHT`** moved 1700 → 2300 → 2400 → 2500; upstream took 1700 (`FC_NORM`),
  then 2300 (`FC_MUL_MV_MMA`, the few-row MMA mat-mul added in v0.6.0), then
  2400 (`FC_TOP_K_MASK`, at the 2026-10-27 bump).
  The TQ3_1S / TQ4_1S dispatch condition in `ggml_metal_op_mul_mat_mv` is the
  only other v0.6.0 touch point: upstream moved that body into a helper.
- **Compile-time knobs** (`TURBO_USE_4MAG`, `TURBO_SPARSE_V`,
  `TURBO_PROFILE_MODE`) are set on the shared `prep` dictionary in
  `ggml_metal_library_init`, so every per-kind library sees them.

Measured after the port (gemma-4-12b Q4_K_XL, 16 prompts, mean KL of the first
generated token's top-20 distribution against the same binary's f16 KV):

| K / V | pre-bump binary | post-bump binary |
| --- | --- | --- |
| q8_0 / q8_0 | 0.0000 | 0.0000 |
| turbo4 / turbo4 | 0.0005 | 0.0014 |
| turbo3 / turbo3 | 0.0699 | 0.0009 |
| turbo3 / q8_0 | 0.0448 | 0.0527 |
| turbo4 / turbo2 | 0.0017 | 0.0099 |
| q8_0 / turbo2 | 0.0023 | 0.0017 |
| turbo2 / turbo2 | 5.58 | 4.97 |

**turbo2 as the K type is a precision limit, not a kernel bug** (re-measured
2026-10-04). The first-token metric above is too weak to judge this: with short
prompts the cache barely matters, and the same metric puts f16 on CPU 0.58 KL
away from f16 on Metal. A teacher-forced run settles it (gemma-4-12b Q4_K_XL,
96 positions of one passage, incremental decode so every step reads the cache):

| K / V | backend | perplexity | top-1 same as f16 Metal |
| --- | --- | --- | --- |
| f16 / f16 | Metal | 24.7 | 96/96 |
| f16 / f16 | CPU | 23.4 | 83/96 |
| q8_0 / q8_0 | Metal | 24.0 | 93/96 |
| turbo4 / turbo4 | Metal | 25.2 | 75/96 |
| turbo3 / q8_0 | Metal | 38.8 | 69/96 |
| turbo3 / turbo3 | Metal | 52.8 | 54/96 |
| q8_0 / turbo2 | Metal | 67.1 | 50/96 |
| turbo2 / q8_0 | Metal | 65.5 | 56/96 |
| turbo2 / q8_0 | CPU | 55.3 | 52/96 |
| turbo2 / turbo2 | Metal | 54.2 | 44/96 |
| turbo2 / turbo2 | CPU | 72.5 | 40/96 |

Metal and the CPU reference path lose the same amount, so the Metal kernels are
doing what the design says. The design is the limit: a 2-bit Lloyd-Max code on
the rotated unit vector reaches cosine 0.94 with the original (0.98 at 3 bits),
about a third of each key's length in error, and attention logits take that
error directly. There is nothing to repair in a kernel. localcode therefore
never passes turbo2 as the key type: `kv_cache_type_k = "turbo2"` launches with
q8_0 keys (`runtime.py`, covered by `test_turbo2_keys_fall_back_to_q8_0`).

The same run shows every type below turbo4 costs real quality on this model,
turbo2 and turbo3 values included. The shipped default (q8_0 / q8_0) and
turbo4 are the only pairs within noise of f16. One passage of 96 tokens is a
small sample; treat the ranking as solid and the exact numbers as rough.

#### Rebased at the 2026-10-27 bump

Every conflict was an append-at-the-same-spot collision with new upstream code;
no TurboQuant behaviour changed.

- Upstream added the `GGML_OP_TOP_K_MASK` op (CPU, Metal) and the BF16
  tensor flash-attention kernel `kernels/fa_tensor.metal`. `GGML_OP_TURBO_WHT`
  now follows `TOP_K_MASK` in the op enum, name and symbol tables
  (`GGML_OP_COUNT` 102 → 103), and `fa_turbo` follows `fa_tensor` in both
  Metal registration lists. `FC_TURBO_WHT` moved to 2500 (see above).
- Upstream moved KV buffer allocation into `llama_memory_bufs::emplace()`
  (#31141, #31200), which allocates and clears the buffer itself, and moved
  `no_alloc` out of `llama_hparams`. The turbo rotation matrices and the InnerQ
  identity scale are now written right after `emplace()` (so still after the
  clear), gated on `ctxs_bufs.no_alloc` instead of `model.hparams.no_alloc`.

#### Known gaps deliberately accepted at the 2026-10-03 bump

- **Metal TQ-weight rotated fast path removed entirely.** The previous bump
  left `get_pipeline_mul_mm_tq_rotated` / `tq3_rotate_act` and the in-place
  activation rotation in `mul_mat_id` half-wired; they are gone, along with the
  `*_rotated` dequantizers and the concurrency-reset hack they needed.
  TQ3_1S / TQ4_1S weights run the generic `mul_mv` / `mul_mm` kernels. No
  catalog model uses these types.
- **CUDA turbo flash-attention instances are no longer compiled.** Upstream
  replaced the hand-listed `fattn-vec-instance-*.cu` files with a generator
  driven by `GGML_CUDA_FA_QUANTS`, which only accepts upstream's type names.
  Turbo K/V on CUDA therefore takes upstream's convert-to-F16 fallback (correct
  but slow), and the fork's `template-instances/*turbo*.cu` files are dead until
  the generator is taught the turbo types. **Not built or tested**: only Metal
  ships.
- **The extra 512 bytes of FA-vec threadgroup memory** ("turbo block cache") was
  dropped; nothing in the kernel read it.

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
  At the 2026-10-14 bump upstream added a per-model switch
  (`llama_model_set_attn_rot`, `model.attn_rot`); the two compose as
  `!model.attn_rot || (env ? atoi(env) != 0 : true)`, so the fork default
  (off) and the env opt-in are unchanged and the model switch can only turn
  it further off.
- **`mul_mm_id_map0` threadgroup memory cap (32 KB).** At the 2026-10-14 bump
  upstream rewrote `kernel_mul_mm_id_map0` (#30373, duplicate expert ids): its
  smem is now `2*ne02*4 + ne02*ne20*2` (per-expert count + base, then id
  staging). The fork keeps upstream's formula and its own clamp to 32768
  after `GGML_PAD`, and the fork's larger `map0` instantiations (32..256)
  still apply to the new kernel.
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
replaced with a warning. No upstream equivalent. Since the 2026-10-03 bump the
clamp lives in `n_ctx_slot()` (the `std::min` against `n_ctx_train`), and
`load_model()` only reports it; the patch edits both.

### `0004-fork-local-bench-scripts.patch` — bench artifacts

`scripts/turbo-quality-gate.sh`, `scripts/bench-smem-m5.sh` and two recorded
benchmark outputs. Development aids, not product code. Safe to drop if they
ever conflict.

### `0005-diffusion-gemma-pr24423.patch` — DiffusionGemma (upstream PR #24423) inside llama-server

**Status: OPEN upstream PR, unmerged.** [ggml-org/llama.cpp#24423](https://github.com/ggml-org/llama.cpp/pull/24423),
vendored at head SHA `daca8075d871483545dd85d58ce11970b304b541`.
**When it merges, keep only the fork-local server integration below** (the PR
itself does not touch `tools/server`); bump past it and shrink this patch to
those hunks.

Adds the `diffusion-gemma` architecture (`src/models/diffusion-gemma.cpp`,
`src/models/gemma4-common.h`, `LLM_ARCH_DIFFUSION_GEMMA` plumbing in
`llama-arch`/`llama-model`/`llama-context`, `llama_set_causal_attn` support,
`diffusion.*` GGUF keys in `gguf-py`), the entropy-bound block-diffusion
decoder, the `--diffusion-eb*` / `--diffusion-blocks` / `--diffusion-kv-cache`
/ `--diffusion-gpu-sampling*` CLI flags, and a CUDA on-device sampling kernel
(`ggml-cuda/diffusion-sampling.cu`, not compiled in the Metal wheel).

#### Fork-local: the denoiser runs inside `llama-server`

The PR ships its own programs (`llama-diffusion-cli`, two stdin "servers") and
touches `tools/server` zero times: diffusion generates by iterative denoising
of a whole 256-token canvas, not by autoregressive decode, so the server's
slot/completion loop has nowhere to host it. The product mandate is **one
binary, one process, `/v1/chat/completions` for everything**, so this patch
hosts the denoiser in the server instead. Shipped binary list: exactly
`llama-server`.

| file | what |
| --- | --- |
| `common/common.cpp` | `common_init_result` enables `llama_diffusion_set_sc()` between model load and context creation for canvas diffusion models (the graph reserve must size the self-conditioning input). No-op for every other arch. |
| `tools/server/server-context.cpp` | `diffusion_probe()` (vocab-only GGUF peek before the context exists: single slot, `ctx_shift`/`cache_reuse`/`fit`/`warmup` off, `n_ubatch >= canvas`, `n_outputs_max = canvas`), `diffusion_init()` (entropy-bound params from GGUF metadata, prompt-KV cache + device SC auto on single GPU, `llama_set_causal_attn(false)`), and `diffusion_process_slot()`: `update_slots()` routes every task there. It applies the **same** chat template / tokenizer as any model (so OpenAI `tools` and `<\|channel>thought…` reasoning work through the normal chat parser), maps `max_tokens` to `n_blocks = ceil(max_tokens / 256)`, runs `diffusion_generate_entropy_bound()` block by block, commits each finished block and feeds its text through `process_token()` (stop strings, budget, EOG) as ONE streaming chunk, and finishes through `send_final_response()`. `finish_reason` is `stop` on EOG / repetition stop, `length` when the block budget is exhausted. `seed` is honoured; unset draws a fresh one except at `temperature: 0`, where it is pinned so the turn is reproducible. |
| `tools/server/CMakeLists.txt` | links the `llama-diffusion` static library (the loop) into `server-context`. |
| `tools/server/server-queue.{h,cpp}` | `server_queue::has_pending_cancel(id)`: the denoiser's per-step callback polls it (4x/s) so a client disconnect aborts the turn within a step, instead of the server denoising the abandoned request to completion before the queued CANCEL is even looked at. |
| `tools/diffusion/*` | the denoising loop, now a library plus an **opt-in** CLI (`-DLLAMA_BUILD_DIFFUSION_TOOLS=ON`, dev aid, not shipped). `diffusion_trim_canvas()` moved here and **fixed**: the PR's repetition-loop detector compared only every other token at stride 2, so any comma-separated list (`, tides, salt, blue`) was cut as a "loop". It now requires a genuinely periodic run. |
| `tools/diffusion-gemma-server/*` | the PR's stdin servers, opt-in only (same flag), not shipped. |

Constraints, by design: single slot, strictly synchronous (one request at a
time, `n_parallel` is forced to 1), text only (no mtmd). A client disconnect
is honoured between denoise steps (~100 ms), not mid-step. `/props`, `/v1/models`, `/health` and `/slots`
are the stock endpoints and need no special-casing. The request must leave
room for one canvas: `prompt + 256 <= n_ctx`, else `ERROR_TYPE_EXCEED_CONTEXT_SIZE`.

Known model behaviour, not a server bug: with `chat_template_kwargs:
{"enable_thinking": false}` the model emits end-of-generation at canvas
position 0 (empty reply) on every seed. The Gemma-4 template variant is one
this checkpoint was not trained on. localcode never sends it for this arch
(`reasoning_capabilities` → `NONE`); `dev/verify_models.sh` omits it for the
diffusion line.

#### Relocation: `examples/diffusion*` → `tools/diffusion*`

Upstream keeps these under `examples/`, which this vendored tree omits
entirely (see *Vendoring exclusions*). They are product code for us (the
library is linked into the server), so they live under `tools/diffusion/` and
`tools/diffusion-gemma-server/` and are wired into `tools/CMakeLists.txt`.
The shipped build stays `-DLLAMA_BUILD_EXAMPLES=OFF` and one cmake invocation
produces `llama-server` only. `diffusion-gemma-eval` was dropped (eval harness,
not product).

#### Other fork-local edits on top of the PR

- `common/arg.cpp`: `-no-cnv` gained `LLAMA_EXAMPLE_DIFFUSION` (the PR put it
  on the `-cnv` option; upstream had since narrowed that option's example set,
  so the hunk needed re-targeting).
- `diffusion-gemma-visual-server.cpp`: `chat.h` now takes `common_json`, not
  nlohmann; the request is parsed once more with `common_json::parse` for the
  chat-template calls.
- `common/parsers/gemma4.cpp` (was `common/chat.cpp` until upstream moved the
  per-template inits into `common/parsers/`): the Gemma 4 format's `preserved_tokens` gained `<|"|>`,
  the string delimiter of the tool-call syntax (`call:f{path:<|"|>.<|"|>}`).
  The server renders a CONTROL token into the generated text only if the
  format preserves it. The official Gemma 4 GGUFs and our
  `diffusiongemma-*-Q4_K_M` export it as USER_DEFINED (always rendered), but
  `diffusiongemma-26B-A4B-it-BF16.gguf` exports it as CONTROL, so the text
  handed to the parser read `{path:.}`, the PEG grammar rejected the call and
  `/v1/chat/completions` returned empty content and null `tool_calls` for
  every tool turn (the model's own output was correct; the loss happened in
  token-to-text). Applies to the autoregressive path too. Suitable upstream.

#### API drift fixed at the 2026-10-03 bump

- `hparams.n_ff_exp` became per-layer (`n_ff_exp_arr` / `n_ff_exp()`);
  `diffusion-gemma.cpp` now loads it with `get_key_or_arr(…, false)` exactly as
  upstream `gemma4.cpp` does.
- `llama_batch_allocr::init()` takes `(batch, vocab, output_all)` and
  `llama_batch_ext` has no `n_tokens`; the memoryless-encode `output_all`
  computation uses `batch_inp.tokens.size()`.

#### Verified (2026-08-23, M-series, `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`)

`llama-server -m … -c 8192 -ngl 999 --jinja -fa on` loads in ~18 s and answers
`/v1/chat/completions` from one resident process: text (`Reply with exactly:
OK` → `OK`, reproducible at temperature 0), a second request with no reload,
streaming one SSE chunk per committed block, a `get_weather` tool call through
the OpenAI `tools` key, `max_tokens` honoured exactly, ~55 tok/s over a
4-block answer. `otool -L` shows only system frameworks.

**Thought-channel handling (2026-10-09).** DiffusionGemma opens `<|channel>thought`
itself and cannot be told not to (`--reasoning off` ends generation at canvas
position 0). Two server-side rules in `update_slots_diffusion` keep the reply
usable with the Gemma 4 parser on (`--reasoning-format auto`): a leading EMPTY
thought block (`<|channel>thought\n<channel|>`, what the model emits after a
tool result) is dropped before parsing, since the parser otherwise filed the
answer after it as reasoning; and a one-block reply that opens a thought and
ends without closing it is treated as the answer. `server_chat_params::
no_forced_thinking` additionally strips a thought opener the template appends
after a tool response, so the model always decides itself. Verified: plain
question, one-word reply, tool call, post-tool answer, streamed and not.

**2026-10-11 bump:** upstream's `chat : refactor API` (#30210) replaced
`common_chat_templates_apply()` in `oaicompat_chat_params_parse` with a
`common_chat_session` whose prompt and generation prompt are private and
already tokenised, so the `no_forced_thinking` opener strip can no longer live
in `tools/server/server-common.cpp`. It is now a `bool no_forced_thinking` on
`common_chat_session_params` (`common/chat.h`) applied to `applied.prompt` /
`applied.generation_prompt` at the top of the `common_chat_session` constructor
(`common/chat.cpp`), before the parser's generation prompt is tokenised, so
prompt, parser prefix and the emitted generation prompt stay consistent;
`server-common.cpp` only forwards the flag. Everything else in the patch
rebased unchanged (upstream still has no DiffusionGemma, no denoiser, no
thought-channel rules).

**2026-10-14 bump:** upstream's `chat: replace output string with parts`
(#30389) turned `common_chat_params::prompt` into a `common_chat_prompt`
(template parts and user-input parts). The `no_forced_thinking` strip now
trims the opener from the last prompt part, and only when that part is
template text (the opener is never user input). `tools/diffusion/diffusion-cli.cpp`
and `tools/diffusion-gemma-server/diffusion-gemma-visual-server.cpp` take
`.prompt.str()`. Upstream's no-cache attention now pads K/V to the mask's KV
dimension (#30416); the DiffusionGemma masks are built at the exact K/V
length, so no padding is added for them.

### `0007-dflash-mtmd-zero-fill.patch` — images with a DFlash drafter

**Status: open upstream.** Still needed at the 2026-10-11 pin: upstream's #30257
(`spec: properly handle mtmd input for mtp`) fixes only the MTP drafter, not
DFlash. Minimal port of
[ggml-org/llama.cpp#30220](https://github.com/ggml-org/llama.cpp/pull/30220)
(issue [#27408](https://github.com/ggml-org/llama.cpp/issues/27408)). **When
upstream merges a fix for #27408, delete this patch** and re-run the model gate's
vision check on every DFlash model.

mtmd image chunks are decoded into the target context by the mtmd helper and
never pass through the DFlash `process()` hook, so the draft KV cache is left
with a positional hole and the next injection fails the consecutive-position
check: `llama_process(ctx_dft) failed rc=-1` and HTTP 500 on every image
request when `--spec-type draft-dflash` is loaded. Upstream's earlier #28587
(in the pinned tree) skips M-RoPE-pinned image batches but leaves the hole.
The patch seeds the hole with zero target-layer features. Drafted tokens are
still verified by the target, so output is unchanged; only draft acceptance
right after an image drops.

Found by the shipping gate's vision check (`scripts/verify_models.py`), which
sends one image to every model that pairs a projector: Qwen 3.6 35B-A3B with
its ggml-org DFlash drafter answered 500 on both quants, text requests and
partial-prefix reuse were fine, and the same request without the drafter or
with Gemma's MTP drafter passed.

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

## Mixed-KV Metal vector selection

Mixed q8/TurboQuant and symmetric TurboQuant caches have baseline vector
kernels only. Select by both K and V types, keeping these caches on baseline
Q/NE variants. Standard symmetric caches retain their upstream tuning. A
missing pipeline is checked before querying its thread limit. The regression
case is an 8K-token Qwen prompt followed by decode with q8 keys/TurboQuant
values and the tensor API disabled; it previously SIGSEGVed at the pipeline
thread-limit query.

### `0010-metal-precompiled-library.patch` — load precompiled Metal kernels when they match

The shipped binary embeds its Metal sources, and compiling them took 20-30 s on
every launch with a cold macOS shader cache (first run after an install or
update). `ggml_metal_library_init` now first looks next to the executable for
`ggml-metal-<key>.bundle`, where `<key>` is an FNV-1a hash of every embedded
kind's source plus the exact runtime macro set and language version. A match
loads each kind's precompiled library from that one file (the kinds share helper symbols, so they cannot be linked into a single metallib); anything else (other sources,
other hardware, an override environment variable, a load error) falls back to
the runtime compile. `scripts/build_metallibs.py` builds the two variants
(Metal 4 tensor devices, and the rest) on the CI runner and `--check` fails the
build when the files in `src/localcode/bin` no longer match the sources.

### `0011-kolibri1-community-gguf-compat.patch` - load the catalog Kolibri-1 GGUF on upstream's loader

Upstream added the `kolibri1` architecture (#30673, refined in #30865), which
retired `0006-kolibri1.patch`. The catalog GGUF (`Hob-forge/Kolibri-1-GGUF`,
`Kolibri-1-Q4_K_M.gguf`, sha256 `c2ac1301...`) was converted with the community
patch that 0006 carried, and differs from upstream's converter in two metadata
values only (every other key and tensor name matches):

- `tokenizer.ggml.pre = "kolibri1"`; upstream writes `qwen2` and rejects the
  unknown name. `src/llama-vocab.cpp` maps `kolibri1` to the qwen2 split, which
  is what the community patch did too.
- `kolibri1.expert_gating_func = 5` (the community `SIGMOID_LOGIT_ADD`:
  select top-k on `logits + bias`, weight by the unbiased `sigmoid(logits)`).
  Upstream's `kolibri1` graph builds exactly that biased selection itself and
  passes `SIGMOID` for the weights; it has no type 5. `src/models/kolibri1.cpp`
  reads 5 as `SIGMOID`.

**Delete this patch** when the catalog points at a GGUF converted by upstream's
`conversion/kolibri1.py` (or when Kolibri leaves the catalog).

### `0012-decision-models-keep-chat.patch` - OpenJev keeps answering on the chat endpoint

Upstream's MiniCPM-OCR-Jev support (#30980, 2026-10-27) made the server refuse
every completion request (HTTP 501, "decision model without text generation")
for each decision type whose output modalities are only `decisions`, which
includes OpenJev. The stated reason is specific to MiniCPM-OCR-Jev: its
checkpoint is a pooled value head with no LM head. OpenJev is a Qwen 3.5 causal
LM that generated text fine before #30980, and LocalCode serves it on
`/v1/chat/completions` too (the catalog says so, and the model gate checks
chat and an OpenCode runtime turn on it). Found by the gate at this bump.

`server_decision_context::can_generate()` now refuses only
`COMMON_DECISION_TYPE_MINICPM_OCR_JEV`; every other decision type generates
as it did before #30980. `/v1/models` still reports upstream's output
modalities. **Delete this patch** if upstream narrows the check to head-less
checkpoints, or if OpenJev chat is dropped from the product.

## Retired at the 2026-10-27 bump

**Do not resurrect these.**

- **`0003-hunyuan-ocr-mtmd.patch`**: a dead file. `tools/mtmd/models/hunyuanocr.cpp`
  was referenced by nothing; upstream builds HunyuanOCR through `hunyuanvl.cpp`
  (`PROJECTOR_TYPE_HUNYUANVL`).
- **`0006-kolibri1.patch`**: upstream ships `kolibri1` (#30673, #30865). The
  community GGUF needs two metadata aliases, carried as `0011` above.
- **`0009-flint-pr30928.patch`**: upstream merged Flint (#30928, `a37cc4610`)
  together with `build_amax_src1_f16` (#30861), the Jinja hex escapes (#30929)
  and the namespace attribute fix (#30941). Upstream now also applies the amax
  rescale to Mistral 4, Mellum and Laguna, replacing the `GGML_PREC_F32` hint.
