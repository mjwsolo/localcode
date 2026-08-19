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
    _MAX_TS_PROJECTS,
    _STATUS_RANK,
    CheckOutcome,
    _detect_commands,
    _strip_jsonc,
    _ts_check_targets,
    _tsbuildinfo_dir,
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


def _fake_commands(monkeypatch, *outcomes):
    """Drive run_project_check_result with scripted checkers.

    Each outcome is (label, returncode, stdout) or the string "sentinel" for an
    argv=None entry.
    """
    import subprocess as _sp

    plan = {}
    cmds = []
    for i, o in enumerate(outcomes):
        if o == "sentinel":
            cmds.append((f"fake{i} (verification incomplete: scripted)", None, "."))
            continue
        label, rc, stdout = o
        cmds.append((label, [f"__fake{i}__"], "."))
        plan[f"__fake{i}__"] = (rc, stdout)

    monkeypatch.setattr("localcode.tools.project_check._detect_commands",
                        lambda _root: cmds)

    def _run(argv, **kw):
        rc, stdout = plan[argv[0]]
        if rc == "timeout":
            raise _sp.TimeoutExpired(argv, 1)
        if rc == "boom":
            raise OSError("cannot execute")
        return _sp.CompletedProcess(argv, rc, stdout, "")

    monkeypatch.setattr("localcode.tools.project_check.subprocess.run", _run)


def test_a_clean_checker_does_not_mask_a_timed_out_one(monkeypatch):
    _fake_commands(monkeypatch, ("green", 0, ""), ("slow", "timeout", ""))
    assert run_project_check_result(".").status == "timed_out"


def test_a_clean_checker_does_not_mask_a_failed_one(monkeypatch):
    _fake_commands(monkeypatch, ("green", 0, ""), ("broken", "boom", ""))
    assert run_project_check_result(".").status == "failed"


def test_a_clean_checker_does_not_mask_a_truncation_sentinel(monkeypatch):
    _fake_commands(monkeypatch, ("green", 0, ""), "sentinel")
    outcome = run_project_check_result(".")
    assert outcome.status == "failed" and not outcome.is_verified


def test_real_errors_outrank_a_failed_checker(monkeypatch):
    _fake_commands(monkeypatch, ("broken", "boom", ""),
                   ("tsc", 2, "src/a.ts(1,1): error TS2322: nope."))
    outcome = run_project_check_result(".")
    assert outcome.status == "errors" and "TS2322" in outcome.detail


def test_status_ranking_constants(tmp_path: Path):
    assert _STATUS_RANK["timed_out"] > _STATUS_RANK["clean"]
    assert _STATUS_RANK["failed"] > _STATUS_RANK["clean"]
    assert _STATUS_RANK["errors"] > _STATUS_RANK["failed"]
    assert not CheckOutcome("timed_out").is_verified


# ── trailing commas must not corrupt string literals (BLOCKING 2) ────────────

@pytest.mark.parametrize("raw,expected", [
    ('{"references":[{"path":"./bad,}"}]}', "./bad,}"),
    ('{"references":[{"path":"./bad,]"}],}', "./bad,]"),
    ('{"references":[{"path":"esc \\" ,} still string"}]}', 'esc " ,} still string'),
])
def test_trailing_comma_pass_preserves_string_literals(raw: str, expected: str):
    """A regex `,(\\s*[}\\]])` pass rewrote `"./bad,}"` to `"./bad}"`, pointing
    reference resolution at a file that does not exist — a false clean."""
    assert json.loads(_strip_jsonc(raw))["references"][0]["path"] == expected


def test_comma_in_path_does_not_lose_the_referenced_project(tmp_path: Path):
    _scaffold(tmp_path, '{"files": [], "references": [{"path": "./bad,}.json"},]}')
    (tmp_path / "bad,}.json").write_text(json.dumps({"include": ["src"]}))
    labels = _labels(tmp_path)
    assert any("bad,}" in lbl for lbl in labels)
    assert not any("incomplete" in lbl for lbl in labels)


# ── reference-graph coverage (BLOCKING 3) ───────────────────────────────────

def test_config_owning_source_AND_references_is_itself_checked(tmp_path: Path):
    """Proven false clean: its own `include` was skipped as if it were a pure
    solution file."""
    _scaffold(tmp_path, {"include": ["rootsrc"],
                         "references": [{"path": "./tsconfig.lib.json"}]})
    (tmp_path / "tsconfig.lib.json").write_text(json.dumps({"include": ["libsrc"]}))
    labels = _labels(tmp_path)
    assert "tsc -p tsconfig.json --noEmit" in labels
    assert "tsc -p tsconfig.lib.json --noEmit" in labels


def test_config_with_neither_include_nor_files_owns_source(tmp_path: Path):
    """TypeScript defaults to including everything when both keys are absent."""
    _scaffold(tmp_path, {"compilerOptions": {"strict": True},
                         "references": [{"path": "./tsconfig.lib.json"}]})
    (tmp_path / "tsconfig.lib.json").write_text(json.dumps({"include": ["libsrc"]}))
    assert "tsc -p tsconfig.json --noEmit" in _labels(tmp_path)


def test_pure_solution_config_is_not_checked_itself(tmp_path: Path):
    _scaffold(tmp_path, {"files": [], "references": [{"path": "./tsconfig.lib.json"}]})
    (tmp_path / "tsconfig.lib.json").write_text(json.dumps({"include": ["src"]}))
    assert "tsc -p tsconfig.json --noEmit" not in _labels(tmp_path)


def test_missing_reference_is_not_silently_skipped(tmp_path: Path):
    """`tsc -b` reports TS5083 here; a resolving sibling must not excuse it."""
    _scaffold(tmp_path, {"files": [], "references": [
        {"path": "./tsconfig.lib.json"}, {"path": "./tsconfig.missing.json"}]})
    (tmp_path / "tsconfig.lib.json").write_text(json.dumps({"include": ["src"]}))
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status == "failed" and not outcome.is_verified
    assert "reference not found" in outcome.detail


def test_invalid_reference_entry_is_rejected(tmp_path: Path):
    _scaffold(tmp_path, {"files": [], "references": [{"pathe": "./typo.json"}]})
    assert run_project_check_result(str(tmp_path)).status == "failed"


def test_target_cap_overflow_is_reported_not_swallowed(tmp_path: Path):
    """The projects past the cap are not checked, so the run is UNVERIFIED —
    silently doing less work must never read as clean."""
    refs = []
    for i in range(_MAX_TS_PROJECTS + 2):
        (tmp_path / f"tsconfig.p{i}.json").write_text(json.dumps({"include": [f"p{i}"]}))
        refs.append({"path": f"./tsconfig.p{i}.json"})
    _scaffold(tmp_path, {"files": [], "references": refs})
    targets, reason = _ts_check_targets(str(tmp_path))
    assert len(targets) == _MAX_TS_PROJECTS and "only the first" in reason
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status == "failed" and not outcome.is_verified


def test_reference_depth_is_bounded_and_reported(tmp_path: Path):
    for i in range(1, 20):
        (tmp_path / f"c{i}.json").write_text(
            json.dumps({"files": [], "references": [{"path": f"./c{i + 1}.json"}]}))
    (tmp_path / "c20.json").write_text(json.dumps({"include": ["leaf"]}))
    _scaffold(tmp_path, {"files": [], "references": [{"path": "./c1.json"}]})
    _targets, reason = _ts_check_targets(str(tmp_path))
    assert "deeper than" in reason
    assert run_project_check_result(str(tmp_path)).status == "failed"


def test_reference_cycle_terminates(tmp_path: Path):
    _scaffold(tmp_path, {"files": [], "references": [{"path": "./a.json"}]})
    (tmp_path / "a.json").write_text(
        json.dumps({"include": ["a"], "references": [{"path": "./b.json"}]}))
    (tmp_path / "b.json").write_text(
        json.dumps({"include": ["b"], "references": [{"path": "./a.json"}]}))
    targets, reason = _ts_check_targets(str(tmp_path))
    assert sorted(targets) == ["a.json", "b.json"] and reason == ""


# ── scratch directory (BLOCKING 4) ──────────────────────────────────────────

def test_no_scratch_dir_means_no_run_at_all(tmp_path: Path, monkeypatch):
    """Running without --tsBuildInfoFile put a .tsbuildinfo back in the user's
    repo while reporting clean. Refuse instead."""
    _scaffold(tmp_path, {"compilerOptions": {"composite": True}, "include": ["src"]})
    monkeypatch.setattr("localcode.tools.project_check._tsbuildinfo_dir",
                        lambda _d: None)
    argvs = [argv for _lbl, argv, _cwd in _detect_commands(str(tmp_path))]
    assert argvs == [None]
    outcome = run_project_check_result(str(tmp_path))
    assert outcome.status == "failed" and "scratch directory" in outcome.detail


def test_every_tsc_command_redirects_buildinfo(tmp_path: Path):
    _scaffold(tmp_path, {"files": [], "references": [{"path": "./tsconfig.lib.json"}]})
    (tmp_path / "tsconfig.lib.json").write_text(json.dumps({"include": ["src"]}))
    for _lbl, argv, _cwd in _detect_commands(str(tmp_path)):
        if argv is None:
            continue
        assert "--tsBuildInfoFile" in argv
        dest = Path(argv[argv.index("--tsBuildInfoFile") + 1]).resolve()
        assert tmp_path.resolve() not in dest.parents


def test_scratch_dir_inside_the_repo_is_refused(tmp_path: Path, monkeypatch):
    inside = tmp_path / "scratch"
    inside.mkdir()
    monkeypatch.setattr("localcode.tools.project_check.tempfile.gettempdir",
                        lambda: str(inside))
    assert _tsbuildinfo_dir(str(tmp_path)) is None
