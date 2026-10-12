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
git -C "$WORK/opencode" checkout -q "$COMMIT"
# Product UI changes kept alongside the pinned source until upstreamed.
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
( cd "$WORK/opencode/packages/opencode" && bun run script/build.ts --single --skip-install --skip-embed-web-ui >/dev/null )
OUT="$(ls "$WORK"/opencode/packages/opencode/dist/*/bin/localcode | head -1)"
file "$OUT" | grep -q 'arm64' || { echo "unexpected architecture: $(file "$OUT")"; exit 1; }
python3 "$ROOT/scripts/check_ui_paths.py" "$OUT"
SHA="$(shasum -a 256 "$OUT" | cut -d' ' -f1)"
echo "built $OUT"; echo "sha256 $SHA (fork $COMMIT)"
if [ "${1:-}" = "--check" ]; then
  HAVE="$(shasum -a 256 "$ROOT/src/localcode/bin/localcode-ui" | cut -d' ' -f1)"
  [ "$SHA" = "$HAVE" ] && echo "bundled binary matches" || { echo "bundled binary sha256 $HAVE differs"; exit 1; }
  exit 0
fi
cp "$OUT" "$ROOT/src/localcode/bin/localcode-ui"; chmod 755 "$ROOT/src/localcode/bin/localcode-ui"
echo "installed into src/localcode/bin/localcode-ui"
