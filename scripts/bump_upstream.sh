#!/usr/bin/env bash
# Replay the TurboQuant fork's patch series onto the latest llama.cpp master,
# build it with the shipped flags, and assert the fork delta survived.
#
# The fork is NOT a snapshot. It is: latest upstream + patches/*.patch. Keeping
# it that way is the whole point - the last time this tree was left to rot for
# four months, two architectures the product ships (cohere2moe, muse_glimmer)
# had to be built as separate source runners on the user's machine, and a manual
# bump attempt silently dropped the qwen35 MTP guard (breaking Qwen 3.8 27B) and
# produced a binary linked against Homebrew's libssl.
#
# CI (.github/workflows/upstream-bump.yml) calls this script. Run it locally to
# reproduce a CI failure exactly.
#
# Usage:
#   scripts/bump_upstream.sh [options]
#
#   --workdir DIR      where to clone/build (default: a mktemp -d, kept on failure)
#   --ref REF          upstream ref to bump to (default: master)
#   --patches DIR      patch series dir (default: <repo>/patches)
#   --vendor DIR       vendored fork dir (default: <repo>/llama-cpp-turboquant)
#   --skip-build       replay + source assertions only, no cmake (fast dry run)
#   --sync             on success, rsync the patched tree over the vendored dir
#                      and update patches/PINNED_UPSTREAM
#   --force            replay even when upstream already equals PINNED_UPSTREAM
#   -j N               build parallelism (default: sysctl hw.ncpu, else 4)
#
# Exit codes (CI branches on these - do not renumber casually):
#   0   success: patches replayed, build + assertions passed
#   10  already current: upstream master == patches/PINNED_UPSTREAM, nothing to do
#   20  contract missing: no patches/PINNED_UPSTREAM or no patches/*.patch
#   30  a patch failed to apply - it needs rebasing onto the new upstream
#   40  a fork-delta / self-containment assertion failed
#   50  the build failed
#
# On exit 30 the script writes $WORKDIR/patch-failure.txt (patch name, upstream
# SHA, reject output) - that file is what CI pastes into the tracking issue.

set -uo pipefail

UPSTREAM_REPO="${UPSTREAM_REPO:-https://github.com/ggml-org/llama.cpp.git}"
REF="master"
WORKDIR=""
PATCHES_DIR=""
VENDOR_DIR=""
SKIP_BUILD=0
DO_SYNC=0
FORCE=0
JOBS=""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log()  { printf '\n=== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$1" >&2; exit "${2:-1}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --workdir) WORKDIR="$2"; shift 2 ;;
    --ref)     REF="$2"; shift 2 ;;
    --patches) PATCHES_DIR="$2"; shift 2 ;;
    --vendor)  VENDOR_DIR="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --sync)    DO_SYNC=1; shift ;;
    --force)   FORCE=1; shift ;;
    -j)        JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

PATCHES_DIR="${PATCHES_DIR:-$REPO_ROOT/patches}"
VENDOR_DIR="${VENDOR_DIR:-$REPO_ROOT/llama-cpp-turboquant}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

if [ -z "$WORKDIR" ]; then
  WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/localcode-bump.XXXXXX")"
fi
mkdir -p "$WORKDIR"
SRC="$WORKDIR/llama.cpp"
RESULT_ENV="$WORKDIR/bump-result.env"
: > "$RESULT_ENV"

emit() { printf '%s=%s\n' "$1" "$2" >> "$RESULT_ENV"; }

echo "repo root:  $REPO_ROOT"
echo "workdir:    $WORKDIR"
echo "patches:    $PATCHES_DIR"
echo "vendor:     $VENDOR_DIR"
echo "upstream:   $UPSTREAM_REPO @ $REF"
emit WORKDIR "$WORKDIR"

# --------------------------------------------------------------------------
# 0. The contract. Another moving part produces patches/; if it is not there
#    yet we say so plainly instead of dying inside git apply on a glob that
#    never expanded.
# --------------------------------------------------------------------------
log "Checking the patch-series contract"

PINNED_FILE="$PATCHES_DIR/PINNED_UPSTREAM"
missing=""
[ -f "$PINNED_FILE" ] || missing="${missing}
  - $PINNED_FILE (one line: the upstream commit SHA the series applies to)"

PATCHES=()
if [ -d "$PATCHES_DIR" ]; then
  while IFS= read -r p; do
    [ -n "$p" ] && PATCHES+=("$p")
  done < <(find "$PATCHES_DIR" -maxdepth 1 -name '*.patch' -type f | LC_ALL=C sort)
fi
[ "${#PATCHES[@]}" -gt 0 ] || missing="${missing}
  - $PATCHES_DIR/NNNN-*.patch (the numbered, individually-applicable series)"

if [ -n "$missing" ]; then
  cat >&2 <<EOF

ERROR: the patch series is not in place yet, so there is nothing to replay.

Missing:$missing

The fork is defined as "latest upstream + patches/". Until the series exists,
llama-cpp-turboquant/ is still an unmanaged snapshot and this automation cannot
do its job. See docs/upstream-fork.md for how the series is built.
EOF
  emit STATUS contract-missing
  exit 20
fi

PINNED="$(tr -d '[:space:]' < "$PINNED_FILE")"
[ -n "$PINNED" ] || die "$PINNED_FILE is empty - it must hold one upstream commit SHA" 20
echo "pinned upstream: $PINNED"
echo "patches in series: ${#PATCHES[@]}"
for p in "${PATCHES[@]}"; do echo "  $(basename "$p")"; done
emit PINNED "$PINNED"
emit PATCH_COUNT "${#PATCHES[@]}"

# --------------------------------------------------------------------------
# 1. Clone upstream, record the SHA.
# --------------------------------------------------------------------------
log "Cloning $UPSTREAM_REPO @ $REF"
rm -rf "$SRC"
git clone --filter=blob:none --no-checkout "$UPSTREAM_REPO" "$SRC" \
  || die "clone failed" 1
git -C "$SRC" checkout --quiet "$REF" || die "checkout $REF failed" 1
NEW_SHA="$(git -C "$SRC" rev-parse HEAD)"
echo "upstream $REF is $NEW_SHA"
emit NEW_SHA "$NEW_SHA"

if [ "$NEW_SHA" = "$PINNED" ] && [ "$FORCE" -eq 0 ]; then
  log "Already current - upstream $REF equals patches/PINNED_UPSTREAM"
  emit STATUS already-current
  exit 10
fi

# The commit list that goes in the PR body. Only computable when the pinned SHA
# is an ancestor we actually fetched; a shallow/rewritten history just skips it.
COMMITS_FILE="$WORKDIR/upstream-commits.txt"
if git -C "$SRC" cat-file -e "$PINNED^{commit}" 2>/dev/null; then
  git -C "$SRC" log --oneline --no-decorate "$PINNED..$NEW_SHA" > "$COMMITS_FILE" 2>/dev/null \
    || : > "$COMMITS_FILE"
  COMMIT_COUNT="$(grep -c . "$COMMITS_FILE" 2>/dev/null || echo 0)"
else
  echo "(pinned commit $PINNED not present in the fetched history - commit list unavailable)" \
    > "$COMMITS_FILE"
  COMMIT_COUNT=0
fi
emit COMMITS_FILE "$COMMITS_FILE"
emit COMMIT_COUNT "$COMMIT_COUNT"

# --------------------------------------------------------------------------
# 2. Replay the series. A failure here is the signal we built this for:
#    upstream moved under one of our patches and it needs a rebase. Stop at the
#    first failure - continuing past it produces a meaningless tree.
# --------------------------------------------------------------------------
log "Replaying ${#PATCHES[@]} patches onto $NEW_SHA"
FAILURE_FILE="$WORKDIR/patch-failure.txt"
rm -f "$FAILURE_FILE"

for p in "${PATCHES[@]}"; do
  name="$(basename "$p")"
  printf '  applying %s ... ' "$name"
  out="$(git -C "$SRC" apply --verbose --reject --whitespace=nowarn "$p" 2>&1)"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED"
    rejects="$(find "$SRC" -name '*.rej' -type f | LC_ALL=C sort)"
    {
      echo "Patch that failed:  $name"
      echo "Upstream commit:    $NEW_SHA"
      echo "Previously pinned:  $PINNED"
      echo
      echo "git apply output:"
      echo "$out"
      if [ -n "$rejects" ]; then
        echo
        echo "Reject hunks:"
        while IFS= read -r r; do
          echo
          echo "--- ${r#"$SRC"/}"
          sed -n '1,120p' "$r"
        done <<< "$rejects"
      fi
    } > "$FAILURE_FILE"
    cat "$FAILURE_FILE" >&2
    emit STATUS patch-failed
    emit FAILED_PATCH "$name"
    emit FAILURE_FILE "$FAILURE_FILE"
    cat >&2 <<EOF

Patch $name no longer applies to upstream $NEW_SHA.

This is not a bug in the automation. Upstream changed the code this patch
touches, so the patch needs rebasing. See docs/upstream-fork.md
("Rebasing a patch"). Nothing was built and nothing was synced.
EOF
    exit 30
  fi
  echo "ok"
done
echo "all ${#PATCHES[@]} patches applied cleanly"

# --------------------------------------------------------------------------
# 3. Source-level assertions. These run before the build so a dropped delta is
#    reported in seconds rather than after a 40-minute compile.
# --------------------------------------------------------------------------
log "Asserting the fork delta survived the replay"
assert_fail=0
af() { echo "ASSERTION FAILED: $*" >&2; assert_fail=1; }

# The KV type enum values are serialised into GGUF, so a silent renumber is a
# data-format break, not just a missing feature.
for sym in GGML_TYPE_TURBO2_0 GGML_TYPE_TURBO3_0 GGML_TYPE_TURBO4_0 GGML_OP_TURBO_WHT; do
  grep -q "$sym" "$SRC/ggml/include/ggml.h" || af "$sym missing from ggml/include/ggml.h"
done
for f in ggml/src/ggml-metal/turbo-matrices.h ggml/src/ggml-metal/turbo-wht.h; do
  [ -f "$SRC/$f" ] || af "$f missing"
done

# Qwen 3.8 27B carries a trailing Multi-Token-Prediction head. Without this
# guard the loader treats it as a transformer layer and dies on
# "missing tensor 'blk.64.ssm_conv1d.weight'". A manual bump dropped it once and
# nothing caught it until the model failed on a user's machine. Upstream renamed
# the field to n_layer_nextn at some point, so accept either spelling.
if ! grep -qE 'nextn_predict_layers|n_layer_nextn' "$SRC/src/llama-model.cpp"; then
  af "qwen35 MTP guard missing from src/llama-model.cpp - Qwen 3.8 27B will fail to load"
fi

if [ "$assert_fail" -ne 0 ]; then
  emit STATUS delta-missing
  die "the patch series applied but the fork delta is not in the tree - a patch is a no-op or applied to the wrong place" 40
fi
echo "fork delta present"

if [ "$SKIP_BUILD" -eq 1 ]; then
  log "--skip-build: stopping before cmake"
  emit STATUS replayed-no-build
  exit 0
fi

# --------------------------------------------------------------------------
# 4. Build, with exactly the flags the shipped binary uses. LLAMA_CURL=OFF and
#    BUILD_SHARED_LIBS=OFF are load-bearing: they are how the artifact stays
#    self-contained on a machine with no Homebrew.
# --------------------------------------------------------------------------
log "Configuring (Release, static, Metal embedded)"
cmake -S "$SRC" -B "$SRC/build-ci" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_CURL=OFF || { emit STATUS build-failed; die "cmake configure failed" 50; }

log "Building with -j $JOBS"
cmake --build "$SRC/build-ci" --config Release -j "$JOBS" \
  || { emit STATUS build-failed; die "build failed" 50; }

# --------------------------------------------------------------------------
# 5. Self-containment. The wheel ships this binary to machines with no
#    Homebrew; a build that picked up /opt/homebrew libssl is broken for them.
# --------------------------------------------------------------------------
log "Asserting the binary is self-contained"
BIN="$SRC/build-ci/bin/llama-server"
[ -x "$BIN" ] || { emit STATUS delta-missing; die "llama-server was not produced at $BIN" 40; }
if command -v otool >/dev/null 2>&1; then
  otool -L "$BIN"
  if otool -L "$BIN" | grep -q '/opt/homebrew'; then
    otool -L "$BIN" | grep '/opt/homebrew' >&2
    emit STATUS not-self-contained
    die "llama-server links Homebrew libraries - it will not run on a machine without them" 40
  fi
  echo "self-contained"
else
  warn "otool not available (not macOS?) - skipping the self-containment check"
fi
emit BINARY "$BIN"

# --------------------------------------------------------------------------
# 6. Optionally land the result in the working tree. CI does this, then commits
#    and opens a PR. Never committed by this script.
# --------------------------------------------------------------------------
if [ "$DO_SYNC" -eq 1 ]; then
  log "Syncing the patched tree into $VENDOR_DIR"
  [ -d "$VENDOR_DIR" ] || die "vendor dir $VENDOR_DIR does not exist" 1
  rm -rf "$SRC/build-ci"
  rsync -a --delete \
    --exclude '.git/' \
    --exclude '.github/' \
    --exclude 'build/' \
    --exclude 'build-static/' \
    --exclude '.cache/' \
    --exclude 'models/ggml-vocabs/' \
    "$SRC/" "$VENDOR_DIR/" || die "rsync into the vendored tree failed" 1
  printf '%s\n' "$NEW_SHA" > "$PINNED_FILE"
  echo "vendored tree updated; patches/PINNED_UPSTREAM -> $NEW_SHA"
  emit SYNCED 1
fi

emit STATUS success
log "Success: upstream $NEW_SHA + ${#PATCHES[@]} patches builds clean and self-contained"
cat <<EOF

NOT VERIFIED BY THIS SCRIPT: whether the models still work.

Loading a model needs a real Metal device and 7-38 GB of weights. Before this
bump is merged, run locally and paste the output into the PR:

    bash dev/verify_models.sh

That covers the 8 bundled-server configs - load, generate, tool-calling,
including the turbo4 KV cache path.
EOF
exit 0
