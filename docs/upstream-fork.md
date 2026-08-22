# The Vendored llama.cpp Fork

`llama-cpp-turboquant/` is the source of the `llama-server` binary shipped in
the wheel. It is **not** a snapshot of llama.cpp that we occasionally
hand-merge. It is defined as:

```
fork = latest upstream llama.cpp master + patches/*.patch
```

Everything in this document exists because that definition was not enforced for
four months.

## Why

The tree's last upstream merge was 2026-04-22. By August the consequences were
concrete:

- `cohere2moe` and `muse_glimmer` — two architectures the product ships — existed
  upstream but not in our tree, so each needs a **separate source-built runner**
  on the user's machine (see `.github/workflows/runners.yml`). That is pure
  staleness tax; upstream already had the code.
- A manual catch-up attempt silently dropped the qwen35 Multi-Token-Prediction
  guard, so Qwen 3.8 27B stopped loading, and produced a binary linked against
  Homebrew's `libssl`, which is broken on any machine without Homebrew OpenSSL.
- Nobody noticed any of it, because nothing was watching.

The fork's actual delta is small. Carrying a small delta as a replayable patch
series costs a weekly PR. Carrying it as a frozen snapshot costs a quarter of
drift and a broken model.

## The delta we carry

| Patch | What it is | Ends when |
|---|---|---|
| TurboQuant KV cache | the product's headline feature; `GGML_TYPE_TURBO2_0/3_0/4_0`, `GGML_OP_TURBO_WHT`, `turbo-matrices.h`, `turbo-wht.h` | never — not upstreamable as-is |
| qwen35 MTP guard | stops the loader treating Qwen 3.8's trailing Multi-Token-Prediction head as a transformer layer | upstream handles the head |
| TQ3_1S / TQ4_1S | fork-local weight quant types | never |
| Fused MoE router | performance | upstream fuses it |
| Muse thinking-tag fix | 3 lines, from **still-open upstream PR #27475** | **the day #27475 merges — delete it then** |

`llama-cpp-turboquant/PATCHES.md` is the authoritative, human-readable
inventory. This table is orientation, not the contract.

## The contract

Three things define the fork. Automation reads all three; if any is missing,
`scripts/bump_upstream.sh` exits 20 with an explanation rather than crashing.

```
patches/PINNED_UPSTREAM             one line: the upstream SHA the series applies to
patches/0001-*.patch, 0002-…        numbered, individually applicable via `git apply`, in order
llama-cpp-turboquant/PATCHES.md     what each patch is, why we carry it, when it can go
```

Patches must be **individually applicable**. A series where 0003 only applies
after 0002 is fine (they are applied in order), but a patch that cannot be
reasoned about or rebased on its own defeats the point.

## The weekly loop

`.github/workflows/upstream-bump.yml` runs every Monday (and on
`workflow_dispatch`):

1. clone `ggml-org/llama.cpp` at master, record the SHA
2. if it equals `patches/PINNED_UPSTREAM` → exit success, nothing to do
3. replay `patches/*.patch` in order with `git apply`
4. assert the delta is still in the source, build static with the shipped flags,
   assert `otool -L` names nothing under `/opt/homebrew`
5. open a PR updating the vendored tree and `PINNED_UPSTREAM`

If step 3 fails, it stops there and files (or comments on) a GitHub issue naming
the exact patch, the upstream SHA, and the reject hunks. That issue is the
signal that a patch needs rebasing — it is the *expected* outcome sometimes, not
a malfunction.

Reproduce any CI run locally, byte for byte:

```bash
bash scripts/bump_upstream.sh                 # full: replay + build + assert
bash scripts/bump_upstream.sh --skip-build    # fast: replay + source assertions only
bash scripts/bump_upstream.sh --ref <sha>     # bump to a specific upstream commit
bash scripts/bump_upstream.sh --sync          # also update the vendored tree + PINNED_UPSTREAM
```

Exit codes: `0` success, `10` already current, `20` patch series missing,
`30` a patch failed to apply, `40` delta or self-containment assertion failed,
`50` build failed.

## What CI cannot check — and why

**A green build does not mean the models work.** GitHub's macOS runners have no
usable Metal compute device, and the models are 7–38 GB. Nothing is ever loaded
in CI. A green run proves the patches apply, the tree compiles, the turbo
symbols and the MTP guard are present in the source, and the binary is
self-contained. It does not prove a single token can be generated.

That gap is exactly what bit us: the source-level MTP-guard check is *necessary
but not sufficient*, and a bump that compiled cleanly still broke Qwen 3.8 27B
in the field.

So every bump PR carries a required manual gate. On a real Apple Silicon Mac,
against a binary built from the PR branch:

```bash
bash dev/verify_models.sh
```

Eight bundled-server configs: load, generate, tool-calling, including the
turbo4 KV-cache path. Paste the full output into the PR. **Do not merge a bump
on green CI alone.**

## Adding a patch

1. Work in a clean clone of upstream at `patches/PINNED_UPSTREAM`, not in the
   vendored tree — the vendored tree is an output, not a workspace.
2. Make the change as one focused commit.
3. Export it into the series with the next free number:
   ```bash
   git format-patch -1 --no-signature --start-number 7 -o /path/to/localcode/patches
   ```
4. Add a row to `llama-cpp-turboquant/PATCHES.md`: what it does, why upstream
   does not have it, and **the condition under which it gets deleted**. A patch
   with no exit condition is a patch nobody will ever dare remove.
5. Re-run `bash scripts/bump_upstream.sh --force --skip-build` to confirm the
   whole series still replays in order.

Keep patches small and single-purpose. A 2000-line omnibus patch is a patch that
will never be rebased; it will be re-snapshotted, and then we are back where we
started.

## Rebasing a patch when CI says it stopped applying

The issue names the patch and includes the reject hunks. Then:

```bash
git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp
git checkout <the upstream SHA from the issue>

# replay everything before the broken one, then apply it by hand
for p in /path/to/localcode/patches/0001-*.patch /path/to/localcode/patches/0002-*.patch; do
  git apply "$p"
done
git apply --3way /path/to/localcode/patches/0004-the-broken-one.patch   # resolve conflicts
git add -A && git commit
git format-patch -1 --no-signature --start-number 4 -o /path/to/localcode/patches
```

Then verify the full series and the build:

```bash
bash scripts/bump_upstream.sh --force
```

Before you rebase, always ask the cheaper question first: **has upstream merged
the equivalent?** A conflict very often means upstream implemented the same
thing their own way. In that case you delete the patch instead of fixing it.

## Dropping a patch upstream has absorbed

The live example is the Muse thinking-tag fix — three lines we carry only
because upstream PR #27475 has not merged yet.

1. Check the upstream state: is the equivalent code in master?
   ```bash
   gh pr view 27475 --repo ggml-org/llama.cpp --json state,mergedAt
   ```
2. Delete the patch file from `patches/`.
3. **Do not renumber the rest.** Gaps in the numbering are fine and cheap;
   renumbering invalidates every reference in issues, PRs, and PATCHES.md.
4. Strike the row from `llama-cpp-turboquant/PATCHES.md` (keep a one-line note
   saying it was dropped and when — that is what stops someone re-adding it).
5. Re-run `bash scripts/bump_upstream.sh --force` and confirm the behaviour the
   patch provided is still correct, via `dev/verify_models.sh`.

Dropping patches is the point. The series should get shorter over time, and
anything that can be upstreamed should be, so that this file eventually
describes a fork that is nearly nothing at all.

## When a bump makes a dedicated runner redundant

`.github/workflows/runners.yml` builds separate binaries for architectures the
bundled server cannot load. Some of those exist only because the fork was stale.
Whenever the vendored tree moves forward, re-check whether the bundled server
can now load `cohere2moe`, `muse_glimmer`, or `diffusion_gemma` — if it can,
delete the runner, its pin in `src/localcode/bootstrap.py`, and its matrix entry.
Every runner deleted is one fewer download and one fewer thing to keep verified.
