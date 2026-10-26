#!/usr/bin/env bash
# One-command release, run by the owner on the Mac that has the models.
#
#   scripts/release.sh <version> --model <name> [--dry-run] [--work DIR] [--wait-minutes N]
#
# 1. checks: clean tree, on main (or a release/* branch) and up to date with origin,
#    no llama-server running, CHANGELOG has a "## <version>" section, v<version> unused
# 2. bumps to the next free <version>rcN, commits, and pushes it to main with the tag v<version>rcN
#    (this is the only push before promotion; publish.yml then publishes a PyPI pre-release)
# 3. waits for PyPI to serve the candidate
# 4. installs it into a fresh venv and tests THAT, each with a fresh HOME:
#      localcode doctor, the terminal snapshot suite (scripts/ui_e2e.sh --installed),
#      the local model gate (scripts/verify_models.py --installed --strict --publish-status,
#      which posts the commit status "local model gate"), and the release eval against the
#      previous PyPI release (scripts/release_eval.sh, posts the status "release eval")
# 5. when both statuses are green it dispatches .github/workflows/release.yml, which re-checks
#    them, runs the installed-package checks on a hosted Mac, and tags the tested commit as
#    v<version> (no new commit; publish.yml rewrites the version in its build only);
#    this script waits for that exact run and its post-publish checks
# 6. if a check fails it stops before promotion, says which one and where its log is,
#    and leaves the candidate published as a pre-release
#
# --dry-run prints every step and command and does not tag, push, install or run a model.
# Needs: git, gh (logged in), python3, node, `inspect` on PATH (pip install -e evals).
set -eu
ROOT="${RELEASE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${RELEASE_PYTHON:-python3}"
cd "$ROOT"

VERSION="" MODEL="" DRY=0 WORK="" WAIT_MIN=60
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --wait-minutes) WAIT_MIN="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) VERSION="$1"; shift ;;
  esac
done
case "$VERSION" in *[!0-9.]*|"") echo "usage: $0 <final version, e.g. 0.5.41> --model <name> [--dry-run]" >&2; exit 2 ;; esac
[ -n "$MODEL" ] || { echo "--model is required (the local model for the release eval)" >&2; exit 2; }

say() { printf '%s\n' "$*"; }
g() { command git -C "$ROOT" "$@"; }
run() { printf '+ %s\n' "$*"; if [ "$DRY" = 0 ]; then "$@"; fi; }
problems=0
refuse() { # <message>: a hard stop, or in a dry run a note of what would stop a real run
  if [ "$DRY" = 1 ]; then say "WOULD REFUSE: $1"; problems=$((problems + 1)); else echo "refusing to start: $1" >&2; exit 2; fi
}

# ---- 1. checks -----------------------------------------------------------------
say "== checks =="
branch="$(g rev-parse --abbrev-ref HEAD)"
case "$branch" in main|release/*) ;; *) refuse "branch is $branch; use main or a release/* branch" ;; esac
[ -z "$(g status --porcelain --untracked-files=no)" ] || refuse "the working tree has uncommitted changes"
if g fetch --quiet origin 2>/dev/null; then
  if [ "$branch" = main ]; then
    [ "$(g rev-parse HEAD)" = "$(g rev-parse origin/main)" ] || refuse "main is not identical to origin/main (pull or push first)"
  else
    g merge-base --is-ancestor origin/main HEAD || refuse "$branch does not contain origin/main"
    if g rev-parse --verify --quiet "origin/$branch" >/dev/null; then
      [ "$(g rev-parse HEAD)" = "$(g rev-parse "origin/$branch")" ] || refuse "$branch is not identical to origin/$branch"
    else
      refuse "$branch is not on origin"
    fi
  fi
else
  refuse "cannot reach origin to check the branch is up to date"
fi
pgrep -x llama-server >/dev/null 2>&1 && refuse "a llama-server process is running (it would distort the gate and the eval timings)"
grep -Eq "^## ($VERSION( |rc[0-9]|\$)|Unreleased)" "$ROOT/CHANGELOG.md" || refuse "CHANGELOG.md has no '## $VERSION', '## ${VERSION}rcN' or '## Unreleased' section (the release needs its notes)"
if [ -n "$(g tag -l "v$VERSION")" ] || g ls-remote --exit-code --tags origin "refs/tags/v$VERSION" >/dev/null 2>&1; then
  refuse "v$VERSION already exists"
fi
say "checks done ($problems problem(s))"

if [ -z "$WORK" ]; then
  if [ "$DRY" = 1 ]; then WORK="${TMPDIR:-/tmp}/localcode-release-$VERSION"; else WORK="$(mktemp -d "${TMPDIR:-/tmp}/localcode-release.XXXXXX")"; fi
fi
LOGS="$WORK/logs"
RC="$(RELEASE_CANDIDATE_ROOT="$ROOT" bash "$ROOT/scripts/release_candidate.sh" "$VERSION" --dry-run | sed -n 's/^next candidate: \([^ ]*\).*/\1/p')"
[ -n "$RC" ] || { echo "could not work out the candidate number" >&2; exit 2; }
say ""; say "release $VERSION via candidate $RC (model $MODEL), work directory $WORK"
[ "$DRY" = 1 ] || mkdir -p "$LOGS"

setver() { # <version>: set it in both files
  run sed -i.bak "s/^version = \".*\"\$/version = \"$1\"/" "$ROOT/pyproject.toml"
  run sed -i.bak "s/^\\( *__version__ = \\)\".*\"\$/\\1\"$1\"/" "$ROOT/src/localcode/__init__.py"
  run rm -f "$ROOT/pyproject.toml.bak" "$ROOT/src/localcode/__init__.py.bak"
}

# Fail early when a publish.yml run for the commit has failed.
publish_runs_failed() { # <sha>
  [ "$(gh run list --workflow publish.yml --commit "$1" --json conclusion --jq '[.[] | select(.conclusion == "failure")] | length' 2>/dev/null || echo 0)" != 0 ]
}

# ---- 2. candidate --------------------------------------------------------------
say ""; say "== 2. publish candidate $RC =="
setver "$RC"
run g commit -qam "$RC"
run g tag -a "v$RC" -m "$RC"
# main is protected (required checks, admins included), so the candidate goes on
# release/<version> with a PR to main; its checks and the local statuses land on
# this exact commit, and the PR merges once the final release is out.
REL_BRANCH="release/$VERSION"
run g push -f origin "HEAD:refs/heads/$REL_BRANCH" "refs/tags/v$RC"
if [ "$DRY" = 1 ]; then
  say "+ gh pr create --base main --head $REL_BRANCH (unless one is open)"
else
  gh pr view "$REL_BRANCH" --json number >/dev/null 2>&1 || \
    gh pr create --base main --head "$REL_BRANCH" --title "$VERSION" --body "Release $VERSION. The candidate $RC on this exact commit is tested by scripts/release.sh; this merges after the final release is published." >/dev/null
fi
SHA="$( [ "$DRY" = 1 ] && echo "<candidate commit>" || g rev-parse HEAD )"

# ---- 3. wait for PyPI ----------------------------------------------------------
say ""; say "== 3. wait for PyPI to serve $RC =="
if [ "$DRY" = 1 ]; then
  say "+ poll https://pypi.org/pypi/localcode/$RC/json every 30s for up to $WAIT_MIN min, stopping early if a publish.yml run for the commit fails"
else
  deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
  until curl -fsS "https://pypi.org/pypi/localcode/$RC/json" >/dev/null 2>&1; do
    publish_runs_failed "$SHA" && { echo "publish.yml failed for $RC; see: gh run list --workflow publish.yml --commit $SHA" >&2; exit 1; }
    [ "$(date +%s)" -lt "$deadline" ] || { echo "$RC did not appear on PyPI in $WAIT_MIN minutes" >&2; exit 1; }
    sleep 30
  done
  say "$RC is on PyPI"
fi

# ---- 4. test the installed candidate -------------------------------------------
say ""; say "== 4. install $RC from PyPI into a fresh venv and test it =="
VENV="$WORK/rc-venv"
HOME_DIR="$WORK/rc-home"
run "$PY" -m venv "$VENV"
if [ "$DRY" = 1 ]; then
  say "+ $VENV/bin/python -m pip install --no-cache-dir localcode==$RC   (retried while the index catches up)"
else
  n=0
  until "$VENV/bin/python" -m pip install --quiet --no-cache-dir "localcode==$RC" >"$LOGS/install.log" 2>&1; do
    n=$((n + 1)); [ "$n" -lt 8 ] || { echo "could not install localcode==$RC; log: $LOGS/install.log" >&2; exit 1; }
    sleep 30
  done
fi

REPO_SLUG="$( [ "$DRY" = 1 ] && echo "<owner>/<repo>" || gh repo view --json nameWithOwner --jq .nameWithOwner )"
post_status() { # <context> <state> <description>: a commit status on the candidate commit
  run gh api "repos/$REPO_SLUG/statuses/$SHA" -f "state=$2" -f "context=$1" -f "description=$3"
}

check() { # <name> <command...>   (stops at the first failure)
  local name="$1"; shift
  say ""; say "-- check: $name"
  if [ "$DRY" = 1 ]; then printf '+ %s\n' "$*"; return 0; fi
  if "$@" >"$LOGS/$name.log" 2>&1; then say "   passed (log: $LOGS/$name.log)"; else
    tail -n 15 "$LOGS/$name.log" | sed 's/^/   | /'
    say ""; say "FAILED: $name. Log: $LOGS/$name.log"
    say "Not promoted. $RC stays published as a pre-release; v$VERSION was not created."
    [ "$name" != release-eval ] || post_status "release eval" failure "Release eval failed against the previous release" >/dev/null
    exit 1
  fi
}
[ "$DRY" = 1 ] || mkdir -p "$HOME_DIR/state" "$HOME_DIR/.config" "$HOME_DIR/.local/share"
check doctor env -i PATH="$PATH" HOME="$HOME_DIR" XDG_CONFIG_HOME="$HOME_DIR/.config" XDG_DATA_HOME="$HOME_DIR/.local/share" LOCALCODE_HOME="$HOME_DIR/state" "$VENV/bin/localcode" doctor
check ui-snapshots "$ROOT/scripts/ui_e2e.sh" --installed "$VENV/bin/python"
# Posts the commit status "local model gate" (pending, then success or failure) on the candidate commit.
check model-gate "$VENV/bin/python" "$ROOT/scripts/verify_models.py" --installed --strict --publish-status
post_status "release eval" pending "Release eval running against the previous release"
check release-eval "$ROOT/scripts/release_eval.sh" --model "$MODEL" --candidate-venv "$VENV" --work "$WORK/release-eval"
post_status "release eval" success "Candidate matches the previous release on the fixed Harness-Bench subset"

# ---- 5. promote (the workflow does it) -------------------------------------------
say ""; say "== 5. all checks passed: ask the release workflow to promote $RC to $VERSION =="
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
MAIN_SHA="$( [ "$DRY" = 1 ] && echo "<tip of main>" || { g fetch -q origin main; g rev-parse origin/main; } )"
run gh workflow run release.yml --ref main -f "candidate=$RC"
if [ "$DRY" = 1 ]; then
  say "+ wait up to $WAIT_MIN min for that release.yml run (it re-checks both statuses and the hosted installed-package checks, then tags the tested commit as v$VERSION, with no new commit, and starts publish.yml on that tag)"
  say "+ wait for the publish.yml run on ref v$VERSION whose head is the candidate commit (release gate, PyPI, GitHub release, post-publish install, doctor and snapshot checks on macos-15)"
  say ""; say "dry run: nothing was tagged, pushed, installed or run ($problems check problem(s) a real run would stop on)"
  exit 0
fi
wait_run() { # <workflow> <branch> <head sha> <not before>: wait for that exact run, succeed only on success
  local wf="$1" branch="$2" head="$3" since="$4" deadline line
  deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
  sleep 20
  while :; do
    line="$(gh run list --workflow "$wf" --branch "$branch" --limit 30 --json headSha,createdAt,status,conclusion \
      --jq "[.[] | select(.headSha == \"$head\" and .createdAt >= \"$since\")] | sort_by(.createdAt) | last | \"\(.status) \(.conclusion)\"" 2>/dev/null || true)"
    case "$line" in
      "completed success") return 0 ;;
      "completed "*) echo "$wf ($branch, $head) ended with: $line" >&2; return 1 ;;
    esac
    [ "$(date +%s)" -lt "$deadline" ] || { echo "$wf ($branch, $head) did not finish in $WAIT_MIN minutes" >&2; return 1; }
    sleep 30
  done
}
wait_run release.yml main "$MAIN_SHA" "$STARTED" || { say "FAILED: promotion. $RC stays a pre-release; see: gh run list --workflow release.yml"; exit 1; }
wait_run publish.yml "v$VERSION" "$SHA" "$STARTED" || { say "FAILED: the final release or its post-publish checks; see: gh run list --workflow publish.yml --branch v$VERSION"; exit 1; }
# The release PR's head is the tested commit; rerun the promotion check now that
# the local gate status is posted, then merge it with a merge commit.
for r in $(gh run list --workflow "Model promotion gate" --branch "$REL_BRANCH" --json databaseId,headSha --jq "[.[] | select(.headSha == \"$SHA\")][0].databaseId" 2>/dev/null); do gh run rerun "$r" >/dev/null 2>&1 || true; done
sleep 60
gh pr checks "$REL_BRANCH" --watch --interval 30 >/dev/null 2>&1 || true
if gh pr merge "$REL_BRANCH" --merge >/dev/null 2>&1; then
  say "merged the release PR for $VERSION into main"
else
  say "NOTE: the release PR for $VERSION did not merge automatically (main moved, or a check failed); merge it by hand: gh pr view $REL_BRANCH"
fi
say "released $VERSION; post-publish checks passed"
