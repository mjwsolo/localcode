#!/usr/bin/env bash
# Rebuild the bundled UI binary (src/localcode/bin/localcode-ui) from the exact
# fork commit in src/localcode/ui/FORK_COMMIT, in a neutral directory so no
# developer path is embedded. Prints the sha256; CI compares it.
#
#   scripts/build_ui_binary.sh            # build + install into src/localcode/bin
#   scripts/build_ui_binary.sh --check    # build only, compare sha256 with the bundled one
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT="$(tr -d '[:space:]' < "$ROOT/src/localcode/ui/FORK_COMMIT")"
REPO="${LOCALCODE_UI_FORK_REPO:-https://github.com/mjwsolo/opencode.git}"
WORK="${LOCALCODE_UI_BUILD_DIR:-/tmp/localcode-ui-build}"
command -v bun >/dev/null || { echo "bun is required (https://bun.sh)"; exit 1; }
# The v2 runtime's build scripts require the bun release named in its package.json (1.4.2 or later).
case "$(bun --version)" in 0.*|1.[0-3].*|1.4.[01]) echo "bun 1.4.2 or later is required (found $(bun --version))"; exit 1;; esac
mkdir -p "$WORK"
if [ ! -d "$WORK/opencode/.git" ]; then git clone -q "$REPO" "$WORK/opencode"; fi
git -C "$WORK/opencode" fetch -q origin
# Undo complete patch stacks first, including later edits to patch-created files.
PATCHES=("$ROOT"/patches/ui/*.patch)
for ((i=${#PATCHES[@]}-1; i>=0; i--)); do
  patch="${PATCHES[$i]}"
  [ -f "$patch" ] || continue
  if git -C "$WORK/opencode" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$WORK/opencode" apply --reverse "$patch"
  fi
done
# Reset cannot remove untracked files introduced by product patches. Remove only
# exact patch-created copies; never discard a modified file or follow a symlink.
python3 - "$ROOT/patches/ui" "$WORK/opencode" <<'PY'
import pathlib, re, subprocess, sys
patches, repo = map(pathlib.Path, sys.argv[1:])
for patch in patches.glob("*.patch"):
    for section in patch.read_text().split("diff --git ")[1:]:
        if "new file mode 100644\n" not in section or "--- /dev/null\n" not in section:
            continue
        match = re.search(r"^\+\+\+ b/(.+)$", section, re.M)
        if not match:
            raise RuntimeError(f"Unrecognized new-file patch: {patch.name}")
        relative = pathlib.Path(match[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe patch path")
        tracked = subprocess.run(["git", "cat-file", "-e", f"HEAD:{relative}"],
                                 cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        target = repo / relative
        if tracked.returncode == 0:
            continue
        if target.is_symlink() or not target.resolve().is_relative_to(repo.resolve()):
            raise RuntimeError(f"Refusing a symlink at patch-created path: {relative}")
        if not target.exists():
            continue
        body = section[section.index("\n@@") + 1:]
        expected = "".join(line[1:] for line in body.splitlines(keepends=True) if line.startswith("+")).encode()
        if target.read_bytes() != expected:
            raise RuntimeError(f"Patch-created file was modified; preserve it before rebuilding: {relative}")
        target.unlink()
PY
# A previous build leaves the product patches applied; start from a clean tree
# or the checkout of a different commit refuses.
git -C "$WORK/opencode" reset -q --hard
# Files a retired patch set created stay behind as untracked files and would block checking out a
# commit that tracks them (the v2 fork carries 0001-0008 itself). This clone is build-only.
git -C "$WORK/opencode" clean -fdq
git -C "$WORK/opencode" checkout -q "$COMMIT"
# Product UI changes kept alongside the pinned source until upstreamed. Patches 0001-0008 are
# folded into the v2 fork itself, so this set is empty until a new product change needs one.
for patch in "$ROOT"/patches/ui/*.patch; do
  [ -f "$patch" ] || continue
  if ! git -C "$WORK/opencode" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$WORK/opencode" apply "$patch"
  fi
done
( cd "$WORK/opencode" && bun install --frozen-lockfile --ignore-scripts >/dev/null )
# Validate the native render-buffer contracts before bundling the UI.
( cd "$WORK/opencode/packages/tui" && bun test test/ui/solid-quote.test.ts test/ui/message-panel.test.ts test/ui/first-response-note.test.ts )
# The runtime shows this in its footer; keep it equal to the wheel version (semver form).
PYVER="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
export OPENCODE_VERSION="$(echo "$PYVER" | sed -E 's/([0-9])(a|b|rc)([0-9]+)$/\1-\2\3/')"
# opencode v2: the CLI (and the UI it embeds) builds from packages/cli; the browser app is not bundled.
( cd "$WORK/opencode/packages/cli" && bun run script/build.ts --single --skip-install --skip-web-ui >/dev/null )
OUT="$(ls "$WORK"/opencode/packages/cli/dist/*/bin/localcode | head -1)"
file "$OUT" | grep -q 'arm64' || { echo "unexpected architecture: $(file "$OUT")"; exit 1; }
# The binary is committed to this repository, whose host rejects files of 100 MiB or more.
SIZE="$(stat -f %z "$OUT" 2>/dev/null || stat -c %s "$OUT")"
[ "$SIZE" -lt 104857600 ] || { echo "UI binary is $SIZE bytes; it must stay under 100 MiB"; exit 1; }
python3 "$ROOT/scripts/check_ui_paths.py" "$OUT"
SHA="$(shasum -a 256 "$OUT" | cut -d' ' -f1)"
echo "built $OUT"; echo "sha256 $SHA (fork $COMMIT)"
if [ "${1:-}" = "--check" ]; then
  HAVE="$(shasum -a 256 "$ROOT/src/localcode/bin/localcode-ui" | cut -d' ' -f1)"
  [ "$SHA" = "$HAVE" ] && echo "bundled binary matches" || { echo "bundled binary sha256 $HAVE differs"; exit 1; }
  exit 0
fi
# Install as a new file: overwriting a binary that has already run in place
# leaves macOS with a stale code-signature cache, and the new one is killed.
cp "$OUT" "$ROOT/src/localcode/bin/localcode-ui.new"; chmod 755 "$ROOT/src/localcode/bin/localcode-ui.new"
mv -f "$ROOT/src/localcode/bin/localcode-ui.new" "$ROOT/src/localcode/bin/localcode-ui"
echo "installed into src/localcode/bin/localcode-ui"
