#!/usr/bin/env bash
# Prepare the next release candidate of a version.
#
#   scripts/release_candidate.sh 0.5.41            bump the files to the next free 0.5.41rcN
#   scripts/release_candidate.sh 0.5.41 --dry-run  only say which rcN that would be
#
# It looks at the tags that exist (local, and on origin when reachable), picks the
# next free rcN, sets that version in pyproject.toml and src/localcode/__init__.py,
# and prints the exact commands to commit, land, tag and push. It never commits,
# tags or pushes by itself.
set -eu
ROOT="${RELEASE_CANDIDATE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VERSION="" DRY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $a" >&2; exit 2 ;;
    *) VERSION="$a" ;;
  esac
done
case "$VERSION" in
  [0-9]*.[0-9]*.[0-9]*) ;;
  *) echo "usage: $0 <next final version, e.g. 0.5.41> [--dry-run]" >&2; exit 2 ;;
esac
case "$VERSION" in *[!0-9.]*) echo "give the final version (0.5.41), not a candidate" >&2; exit 2 ;; esac

git() { command git -C "$ROOT" "$@"; }

if [ -n "$(git tag -l "v$VERSION")" ] || { git ls-remote --exit-code --tags origin "refs/tags/v$VERSION" >/dev/null 2>&1; }; then
  echo "v$VERSION is already released; pick a newer version" >&2; exit 2
fi

tags="$(git tag -l "v$VERSION""rc*")"
if remote="$(git ls-remote --tags origin "refs/tags/v$VERSION""rc*" 2>/dev/null)"; then
  tags="$tags
$(printf '%s\n' "$remote" | sed -n 's|.*refs/tags/\([^^]*\).*|\1|p')"
else
  echo "note: origin not reachable, using local tags only" >&2
fi
last="$(printf '%s\n' "$tags" | sed -n "s/^v$VERSION""rc\([0-9][0-9]*\)\$/\1/p" | sort -n | tail -1)"
N=$(( ${last:-0} + 1 ))
RC="${VERSION}rc${N}"
echo "next candidate: $RC (last existing: $([ -n "$last" ] && echo "rc$last" || echo none))"
[ "$DRY" = 1 ] && exit 0

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "tracked files have uncommitted changes; commit or stash them first" >&2; exit 2
fi
current="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/pyproject.toml" | head -1)"
[ -n "$current" ] || { echo "no version in pyproject.toml" >&2; exit 2; }
sed -i.bak "s/^version = \"$current\"\$/version = \"$RC\"/" "$ROOT/pyproject.toml"
sed -i.bak "s/^\\( *__version__ = \\)\"$current\"\$/\\1\"$RC\"/" "$ROOT/src/localcode/__init__.py"
rm -f "$ROOT/pyproject.toml.bak" "$ROOT/src/localcode/__init__.py.bak"
grep -q "^version = \"$RC\"\$" "$ROOT/pyproject.toml" && grep -q "__version__ = \"$RC\"\$" "$ROOT/src/localcode/__init__.py" \
  || { echo "could not set the version in both files" >&2; exit 2; }
echo "set $current -> $RC in pyproject.toml and src/localcode/__init__.py"

cat <<EOF

Next, from a clean checkout of the commit you mean to test:

  git switch -c release/$RC
  git commit -am "$RC"
  git push -u origin release/$RC
  gh pr create --title "$RC" --body "Release candidate $RC"

The package is published when this lands on main: publish.yml sees the version
change, runs the release gate, uploads $RC to PyPI as a pre-release, creates the
GitHub release marked --prerelease and then installs it from PyPI into a fresh
environment and runs the post-publish checks. To publish a commit that is
already on main without a PR, tag it instead (the workflow refuses a commit
that is not on main):

  git tag -a v$RC -m "$RC"
  git push origin v$RC

Install it (it is never the default install):

  uv tool install --prerelease allow "localcode==$RC"
  pip install "localcode==$RC"
EOF
