"""Scenario orchestrator + report writer.

Runs every scenario in order, captures results, writes a structured
report to `~/.localcode/test-results/<timestamp>/` and updates the
`latest/` symlink so external tools (or terminal coding tools) can find the newest
report at a stable path.

Each scenario is a Python module under `tests/e2e/scenarios/` exposing
a `run() -> ScenarioResult` function. The runner imports them lazily
so a syntax error in one scenario doesn't prevent the others from
running.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from .harness import ScenarioResult


SCENARIO_MODULES = [
    # Order matters a LOT. The long coding sessions each need a live,
    # healthy server for 20-40 min; they can't share a server with a
    # scenario that restarts it mid-run. Previous order put
    # `server_lifecycle` FIRST, which shut down the shared server mid-
    # memory-squeeze on 16 GB Macs and left the next two scenarios to
    # fail 25/25 turns at 0.5 s each on a dead socket (2026-04-23
    # e2e run: 0/3 passed, root cause = scenario ordering + no auto-
    # recovery on connection-refused).
    #
    # New order:
    #   1. Long fast-mode coding session  — needs live server, ~20 min
    #   2. Long reasoning-mode session    — same, ~30 min
    #   3. server_lifecycle (LAST)        — intentionally destroys the
    #      server it just spawned, so any failure here can't cascade
    #      into other scenarios.
    "tests.e2e.scenarios.long_coding_session",
    "tests.e2e.scenarios.long_coding_session_reasoning",
    "tests.e2e.scenarios.server_lifecycle",
]

def _results_root() -> Path:
    """Per-project test-results root: `<project_root>/.localcode/test-results/`.
    Resolved live so the runner picks up the right project even if invoked
    from a subdirectory.
    """
    from localcode.paths import test_results_root
    return test_results_root()


# Lazy property for legacy code that imports the constant. Most callers
# should prefer `_results_root()` so the lookup follows cwd correctly.
RESULTS_ROOT = _results_root()


def _import_scenario(modpath: str):
    """Import a scenario module by dotted path. Returns (module, error)."""
    try:
        return importlib.import_module(modpath), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def _run_scenario(modpath: str) -> ScenarioResult:
    """Run a single scenario and convert any exception into a failed result."""
    mod, err = _import_scenario(modpath)
    if mod is None:
        return ScenarioResult(
            name=modpath.split(".")[-1],
            passed=False,
            assertions=[("import", False, err or "unknown import error")],
        )
    short = modpath.split(".")[-1]
    print(f"\n── {short} ─────────────────────────────────────────────")
    try:
        result = mod.run()
    except Exception as exc:
        return ScenarioResult(
            name=short,
            passed=False,
            assertions=[("scenario_raised", False,
                         f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[:1000]}")],
        )
    status = "PASS" if result.passed else "FAIL"
    print(f"  {status} ({result.wall_clock_s:.1f}s)")
    for n, ok, detail in result.assertions:
        mark = "✓" if ok else "✗"
        line = f"    {mark} {n}"
        if detail and not ok:
            line += f" — {detail}"
        print(line)
    return result


def _write_report(results: list[ScenarioResult], outdir: Path) -> None:
    """Emit `report.md`, `report.json`, and `failures.md` (failures only).

    The markdown is human-skimmable; the JSON is for diffing across runs
    and for me (terminal coding tools) to read deterministically. `failures.md` is the
    one to paste at me when something breaks — it contains only the
    failed cases plus their evidence + truncated event traces.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    # ── report.md ──
    md = [f"# LocalCode E2E Report",
          f"",
          f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
          f"",
          f"**Result: {passed}/{len(results)} scenarios passed**",
          f""]
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        md.append(f"## {status} — `{r.name}` ({r.wall_clock_s:.1f}s)")
        md.append("")
        md.append("| check | result | detail |")
        md.append("|---|---|---|")
        for n, ok, detail in r.assertions:
            mark = "✓" if ok else "✗"
            md.append(f"| `{n}` | {mark} | {detail or '-'} |")
        if r.evidence:
            md.append("")
            md.append("**Evidence:**")
            md.append("```json")
            md.append(json.dumps(r.evidence, indent=2, default=str))
            md.append("```")
        md.append("")
    (outdir / "report.md").write_text("\n".join(md))

    # ── failures.md (only the failed scenarios + full event trace) ──
    fail_md = [f"# Failed scenarios — {datetime.now().isoformat(timespec='seconds')}",
               f""]
    failures = [r for r in results if not r.passed]
    if not failures:
        fail_md.append("_All scenarios passed._")
    for r in failures:
        fail_md.append(f"## `{r.name}` ({r.wall_clock_s:.1f}s)")
        fail_md.append("")
        fail_md.append("### Failed assertions")
        for n, ok, detail in r.assertions:
            if not ok:
                fail_md.append(f"- `{n}` — {detail or '-'}")
        fail_md.append("")
        fail_md.append("### Evidence")
        fail_md.append("```json")
        fail_md.append(json.dumps(r.evidence, indent=2, default=str))
        fail_md.append("```")
        fail_md.append("")
        if r.turns:
            fail_md.append("### Event traces (first 100 events per turn)")
            for i, t in enumerate(r.turns):
                fail_md.append(f"#### Turn {i+1}: `{t.turn_text[:80]}`")
                fail_md.append(f"_wall clock: {t.wall_clock_s:.2f}s, error: {t.error or 'none'}_")
                fail_md.append("```")
                for e in t.events[:100]:
                    payload_brief = json.dumps(e.payload, default=str)[:160]
                    fail_md.append(f"  {e.t:6.2f}  {e.event_type:20}  {payload_brief}")
                if len(t.events) > 100:
                    fail_md.append(f"  ... +{len(t.events)-100} more events")
                fail_md.append("```")
                fail_md.append("")
    (outdir / "failures.md").write_text("\n".join(fail_md))

    # ── report.json (machine-readable) ──
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "totals": {"passed": passed, "failed": failed, "total": len(results)},
        "scenarios": [
            {
                "name": r.name,
                "passed": r.passed,
                "wall_clock_s": r.wall_clock_s,
                "assertions": [{"name": n, "passed": ok, "detail": d}
                               for n, ok, d in r.assertions],
                "evidence": r.evidence,
                "turn_count": len(r.turns),
            }
            for r in results
        ],
    }
    (outdir / "report.json").write_text(json.dumps(data, indent=2, default=str))

    # ── per-scenario event log ──
    for r in results:
        for i, t in enumerate(r.turns):
            log_path = outdir / f"{r.name}.turn{i+1}.events.jsonl"
            with log_path.open("w") as f:
                for e in t.events:
                    f.write(json.dumps({
                        "t": e.t, "type": e.event_type, "payload": e.payload,
                    }, default=str) + "\n")


def _update_latest_symlink(outdir: Path) -> None:
    """Point `~/.localcode/test-results/latest/` at the newest run."""
    latest = RESULTS_ROOT / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(outdir.name)
    except Exception:
        # Symlink is a convenience, not load-bearing.
        pass


def main() -> int:
    print("LocalCode E2E test run")
    print("=" * 60)
    t0 = time.time()
    results: list[ScenarioResult] = []
    for mod in SCENARIO_MODULES:
        results.append(_run_scenario(mod))

    # Where to land the report.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = RESULTS_ROOT / stamp
    _write_report(results, outdir)
    _update_latest_symlink(outdir)

    passed = sum(1 for r in results if r.passed)
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"DONE — {passed}/{len(results)} scenarios passed in {elapsed:.1f}s")
    print(f"Report: {outdir}")
    print(f"Latest: {RESULTS_ROOT / 'latest'}")
    if passed < len(results):
        print(f"Failures detail: {outdir / 'failures.md'}")
    return 0 if passed == len(results) else 1
