"""The build-completion typecheck gate: command selection AND behaviour.

Three guarantees are pinned here:

1. TS PROJECT-REFERENCE scaffolds (Vite/React/Vue) are actually type-checked.
   A modern scaffold's root tsconfig.json is a solution file (`"references":
   [...]`, no `include`); `tsc --noEmit` on it type-checks NOTHING and reports a
   false clean. The referenced projects must be checked.
2. Verification is READ-ONLY. It runs `tsc -p <ref> --noEmit`, never `tsc -b`
   (build mode emits JS/.d.ts/source maps and writes .tsbuildinfo into the
   user's repo), and redirects tsc's incremental state outside the repo.
3. A result that is not a green run is never reported as clean — JSONC configs
   parse, unparseable configs fail loudly, and timeouts / execution failures /
   nonzero-without-output are distinguishable from clean.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from localcode.tools.project_check import (
    _STATUS_RANK,
    CheckOutcome,
    _detect_commands,
    _strip_jsonc,
    run_project_check,
    run_project_check_result,
)


def _scaffold(tmp: Path, tsconfig, *, tsc_script: str = "#!/bin/sh\nexit 0\n") -> Path:
    """A minimal node project. `tsconfig` may be a dict or raw (JSONC) text."""
    (tmp / "package.json").write_text(json.dumps({"name": "x", "scripts": {}}))
    text = tsconfig if isinstance(tsconfig, str) else json.dumps(tsconfig)
    (tmp / "tsconfig.json").write_text(text)
    binp = tmp / "node_modules" / ".bin"
    binp.mkdir(parents=True)
    tsc = binp / "tsc"
    tsc.write_text(tsc_script)
    os.chmod(tsc, 0o755)
    return tsc


def _labels(tmp: Path) -> list[str]:
    return [label for label, _argv, _cwd in _detect_commands(str(tmp))]


def _argv_for(tmp: Path, label: str) -> list[str]:
    return next(a for lbl, a, _ in _detect_commands(str(tmp)) if lbl == label)


# ── 1. reference scaffolds are checked, read-only ───────────────────────────

def test_project_references_check_the_referenced_project(tmp_path: Path):
    _scaffold(tmp_path, {"files": [], "references": [{"path": "./tsconfig.app.json"}]})
    (tmp_path / "tsconfig.app.json").write_text(json.dumps({"include": ["src"]}))
    labels = _labels(tmp_path)
    assert "tsc -p tsconfig.app.json --noEmit" in labels
    argv = _argv_for(tmp_path, "tsc -p tsconfig.app.json --noEmit")
    assert "--noEmit" in argv


def test_build_mode_is_never_used(tmp_path: Path):
    """`tsc -b` is a BUILD: it emits JS/.d.ts/maps into the user's repo."""
    _scaffold(tmp_path, {"files": [], "references": [{"path": "./tsconfig.app.json"}]})
    (tmp_path / "tsconfig.app.json").write_text(json.dumps({"include": ["src"]}))
    for _label, argv, _cwd in _detect_commands(str(tmp_path)):
        assert argv is None or "-b" not in argv


def test_tsbuildinfo_is_written_outside_the_repo(tmp_path: Path):
    _scaffold(tmp_path, {"files": [], "references": [{"path": "./tsconfig.app.json"}]})
    (tmp_path / "tsconfig.app.json").write_text(json.dumps({"include": ["src"]}))
    argv = _argv_for(tmp_path, "tsc -p tsconfig.app.json --noEmit")
    assert "--tsBuildInfoFile" in argv
    dest = Path(argv[argv.index("--tsBuildInfoFile") + 1])
    assert tmp_path not in dest.parents and dest != tmp_path


def test_nested_references_are_resolved(tmp_path: Path):
    _scaffold(tmp_path, {"references": [{"path": "./pkg"}]})
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "tsconfig.json").write_text(json.dumps({"references": [{"path": "./tsconfig.lib.json"}]}))
    (pkg / "tsconfig.lib.json").write_text(json.dumps({"include": ["src"]}))
    assert any("pkg/tsconfig.lib.json" in lbl.replace(os.sep, "/") for lbl in _labels(tmp_path))


def test_plain_tsconfig_checks_itself(tmp_path: Path):
    _scaffold(tmp_path, {"compilerOptions": {"strict": True}, "include": ["src"]})
    argv = _argv_for(tmp_path, "tsc -p tsconfig.json --noEmit")
    assert "--noEmit" in argv and "-b" not in argv


# ── 2. JSONC ────────────────────────────────────────────────────────────────

JSONC_SOLUTION = """{
  // TypeScript has supported comments here since 1.8
  "files": [],
  /* and block comments too */
  "references": [{ "path": "./tsconfig.app.json" }]
}
"""

JSONC_TRAILING_COMMA = """{
  "files": [],
  "references": [{ "path": "./tsconfig.app.json" },],
}
"""


@pytest.mark.parametrize("text", [JSONC_SOLUTION, JSONC_TRAILING_COMMA])
def test_commented_solution_config_is_not_downgraded(tmp_path: Path, text: str):
    """Regression: strict json.load() failed here, silently reverting to the
    inadequate `tsc --noEmit` false-clean behaviour."""
    _scaffold(tmp_path, text)
    (tmp_path / "tsconfig.app.json").write_text(json.dumps({"include": ["src"]}))
    labels = _labels(tmp_path)
    assert "tsc -p tsconfig.app.json --noEmit" in labels
    assert "tsc -p tsconfig.json --noEmit" not in labels


def test_strip_jsonc_preserves_string_contents():
    src = '{"a": "http://x/y // not a comment", "b": "/* nor this */"}'
    assert json.loads(_strip_jsonc(src)) == {
        "a": "http://x/y // not a comment", "b": "/* nor this */"}


def test_unparseable_tsconfig_fails_loudly(tmp_path: Path):
    _scaffold(tmp_path, '{ "references": [ BROKEN }')
    argvs = [argv for _lbl, argv, _cwd in _detect_commands(str(tmp_path))]
    assert argvs == [None]  # sentinel: no safe/adequate command
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status == "failed"
    assert not outcome.is_verified
    assert run_project_check(str(tmp_path)) is None  # not surfaced as code errors


# ── 3. outcomes: clean vs unavailable vs timed_out vs failed vs errors ───────

def test_failing_checker_reaches_the_gate_end_to_end(tmp_path: Path):
    _scaffold(tmp_path, {"files": [], "references": [{"path": "./tsconfig.app.json"}]},
              tsc_script="#!/bin/sh\n"
                         "echo \"src/a.ts(3,7): error TS2339: Property 'x' does not exist.\"\n"
                         "exit 2\n")
    (tmp_path / "tsconfig.app.json").write_text(json.dumps({"include": ["src"]}))
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status == "errors" and outcome.is_red
    assert "TS2339" in outcome.detail
    assert "TS2339" in (run_project_check(str(tmp_path)) or "")


def test_clean_checker_is_clean(tmp_path: Path):
    _scaffold(tmp_path, {"include": ["src"]})
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status == "clean" and outcome.is_verified
    assert run_project_check(str(tmp_path)) is None


def test_absent_tsc_is_unavailable_not_clean(tmp_path: Path):
    (tmp_path / "package.json").write_text(json.dumps({"name": "x"}))
    (tmp_path / "tsconfig.json").write_text(json.dumps({"include": ["src"]}))
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status == "unavailable"
    assert not outcome.is_verified  # absent checker must not read as verified


def test_nonzero_with_empty_output_is_failed_not_clean(tmp_path: Path):
    _scaffold(tmp_path, {"include": ["src"]}, tsc_script="#!/bin/sh\nexit 2\n")
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status == "failed" and not outcome.is_verified
    assert "exited 2" in outcome.detail


def test_timeout_is_timed_out_not_clean(tmp_path: Path, monkeypatch):
    _scaffold(tmp_path, {"include": ["src"]}, tsc_script="#!/bin/sh\nsleep 30\n")
    monkeypatch.setattr("localcode.tools.project_check._TIMEOUT", 0.5)
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status == "timed_out" and not outcome.is_verified


def test_execution_failure_is_failed_not_clean(tmp_path: Path):
    _scaffold(tmp_path, {"include": ["src"]}, tsc_script="")  # not a valid executable
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status in ("failed", "timed_out") and not outcome.is_verified


def test_composite_noemit_toolchain_error_is_not_reported_as_user_errors(tmp_path: Path):
    _scaffold(tmp_path, {"include": ["src"]},
              tsc_script="#!/bin/sh\n"
                         "echo 'error TS6304: Composite projects may not disable emit.'\n"
                         "exit 1\n")
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status == "failed" and not outcome.is_red


def test_a_worse_status_outranks_a_clean_one(tmp_path: Path):
    """One green checker must not mask another that timed out or failed."""
    assert _STATUS_RANK["timed_out"] > _STATUS_RANK["clean"]
    assert _STATUS_RANK["failed"] > _STATUS_RANK["clean"]
    assert _STATUS_RANK["errors"] > _STATUS_RANK["failed"]
    assert not CheckOutcome("timed_out").is_verified
