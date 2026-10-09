"""Architecture test — structural rules enforced on every commit.

Fails CI if the codebase violates any rule below. New rules get added
here instead of enforced by convention, because convention drifts.

Philosophy: the test shouldn't block legitimate work. Rules are scoped
tightly — a rule about tool layering doesn't fire on eval/ code, a
rule about file size has an allowlist for files we've already chosen
to tolerate, etc. When a rule catches something that IS legitimate,
the answer is to refine the rule, not to silence it.

## Rules

  1. No import cycles in src/localcode/
  2. Layer direction:
       - tools/ must not import from tui/
       - eval/ may import from src/localcode/ but not vice-versa
  3. Model-specific isolation: Gemma-specific regex literals
     (<|tool_call>, <unused25>) only live in model_families.py
  4. File size cap: no NEW file > 400 LoC. Existing big files are
     grandfathered in LEGACY_BIG_FILES below with their current
     line count; CI fails if any of them grows by more than 20%
     OR if any new file exceeds the cap.
  5. agent/ submodules declare __all__
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "localcode"
EVAL_DIR = REPO / "eval"


# Rule 4 (file-size cap) removed 2026-07-07 — self-imposed guardrail, deleted.


# ── Helpers ─────────────────────────────────────────────────────────


def _iter_py(root: Path):
    """Yield every .py file under `root`, skipping __pycache__."""
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def _imports_in(path: Path, top_level_only: bool = False) -> list[str]:
    """Return the list of import targets from a file, like
    ['localcode.agent', 'localcode.tools.bash']. Relative imports
    inside the package are resolved to their absolute form.

    `top_level_only`: if True, skip imports nested inside function
    or class bodies (deferred imports). That's what you want for
    cycle detection — Python only fails at module-load time when
    BOTH sides of a cycle do the import at module level. Deferred
    imports inside function bodies resolve at call time and are a
    deliberate workaround for breaking the cycle.
    """
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return []
    pkg_path = path.relative_to(REPO).with_suffix("")
    pkg_parts = pkg_path.parts
    if pkg_parts[:2] == ("src", "localcode"):
        module_parts = ("localcode",) + pkg_parts[2:]
    else:
        module_parts = pkg_parts

    # Build a set of node-ids that live inside function/class bodies
    # or inside `if TYPE_CHECKING:` guards so we can skip them when
    # top_level_only is True. `TYPE_CHECKING` imports are stripped by
    # Python at runtime — they exist only for type checkers — so they
    # can't cause import-time cycles.
    nested_ids: set[int] = set()
    if top_level_only:
        def _is_type_checking_guard(node: ast.If) -> bool:
            t = node.test
            if isinstance(t, ast.Name) and t.id == "TYPE_CHECKING":
                return True
            if (isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"):
                return True
            return False

        def _walk(scope, inside: bool):
            for child in ast.iter_child_nodes(scope):
                if isinstance(child, (ast.Import, ast.ImportFrom)) and inside:
                    nested_ids.add(id(child))
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    _walk(child, True)
                elif isinstance(child, ast.If) and _is_type_checking_guard(child):
                    # Body + orelse of a TYPE_CHECKING guard are
                    # deferred, treat like function body.
                    for sub in child.body + child.orelse:
                        _walk(sub, True)
                        if isinstance(sub, (ast.Import, ast.ImportFrom)):
                            nested_ids.add(id(sub))
                else:
                    _walk(child, inside)
        _walk(tree, False)

    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if id(node) in nested_ids:
                continue
            if node.module is None:
                continue
            if node.level == 0:
                out.append(node.module)
            else:
                base = module_parts[:-node.level]
                if not base:
                    continue
                out.append(".".join(list(base) + [node.module]))
        elif isinstance(node, ast.Import):
            if id(node) in nested_ids:
                continue
            for alias in node.names:
                out.append(alias.name)
    return out


# ── Rule 1 — no cycles ──────────────────────────────────────────────


def test_no_import_cycles_in_src():
    """Build the import graph of src/localcode/**/*.py (node = module
    name like 'localcode.agent.loop'). Depth-first search; any back
    edge to a node currently on the stack is a cycle.

    Runs only on `localcode.*` edges — third-party and stdlib are
    assumed cycle-free on our behalf.
    """
    graph: dict[str, set[str]] = {}
    for path in _iter_py(SRC):
        mod = ".".join(path.relative_to(REPO / "src").with_suffix("").parts)
        graph.setdefault(mod, set())
        for imp in _imports_in(path, top_level_only=True):
            if imp.startswith("localcode."):
                graph[mod].add(imp)

    # Normalise: trim to modules actually in the graph (others resolve
    # to packages whose __init__ exists; they count as nodes).
    nodes = set(graph.keys())
    for deps in graph.values():
        nodes |= deps
    for n in nodes:
        graph.setdefault(n, set())

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in graph}

    def visit(n: str, stack: list[str]) -> list[str] | None:
        color[n] = GRAY
        stack.append(n)
        for m in graph.get(n, ()):
            if color.get(m, WHITE) == GRAY:
                # Found a back-edge; truncate stack to the cycle start.
                i = stack.index(m)
                return stack[i:] + [m]
            if color.get(m, WHITE) == WHITE:
                c = visit(m, stack)
                if c is not None:
                    return c
        stack.pop()
        color[n] = BLACK
        return None

    for n in list(graph):
        if color[n] == WHITE:
            cycle = visit(n, [])
            if cycle is not None:
                pytest.fail(
                    "Import cycle detected:\n  " + " → ".join(cycle)
                )


# ── Rule 2 — layer direction ────────────────────────────────────────


def test_tools_layer_does_not_import_tui():
    """The tools/ layer executes file-system / network / shell
    actions — pure backend. Importing from tui/ would mean tool
    behaviour depends on the presentation layer, which breaks
    headless mode (eval, tests, scripted use) and couples two
    concerns that should stay separable.
    """
    violations: list[str] = []
    for path in _iter_py(SRC / "tools"):
        for imp in _imports_in(path):
            if imp.startswith("localcode.tui") or imp.startswith(".tui"):
                violations.append(f"{path.relative_to(REPO)}: imports {imp}")
    assert not violations, (
        "tools/ must not import tui/ — the tool layer must stay "
        "headless-safe.\n  " + "\n  ".join(violations)
    )


def test_eval_is_not_imported_by_prod_code():
    """eval/ contains test scaffolding (scenarios, benchmarks, judge
    wiring). It's fine for eval/ to reach into src/localcode/, but
    the reverse means prod has a dependency on its own test suite,
    which is backwards.
    """
    violations: list[str] = []
    for path in _iter_py(SRC):
        for imp in _imports_in(path):
            if imp.startswith("eval.") or imp == "eval":
                violations.append(f"{path.relative_to(REPO)}: imports {imp}")
    assert not violations, (
        "src/localcode/ must not import from eval/ — prod can't "
        "depend on its test scaffolding.\n  " + "\n  ".join(violations)
    )


# ── Rule 3 — model-specific isolation ───────────────────────────────


_GEMMA_LITERALS = [
    r"<\|tool_call\>",
    r"<unused25>",
    r"<\|channel>thought",
    r"<channel\|>",
]


def _contains_gemma_literal_in_code(path: Path) -> list[str]:
    """Return string-literal values in `path` that contain Gemma
    markers AND are NOT a docstring. Docstrings / comments
    legitimately reference the patterns to explain what the live
    code does — we only want to catch actual runtime usage.

    Returns the list of offending literal values (for a useful
    error message). Empty list means clean.
    """
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return []
    pattern = re.compile("|".join(_GEMMA_LITERALS))

    # Collect node-ids of every docstring so we can skip them.
    # A docstring is the first statement of a Module / FunctionDef /
    # AsyncFunctionDef / ClassDef body, wrapped in an `Expr` whose
    # `value` is an `ast.Constant[str]`.
    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not body or not isinstance(body, list):
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            docstring_ids.add(id(first.value))

    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstring_ids:
                continue
            if pattern.search(node.value):
                hits.append(node.value[:60])
    return hits


def test_gemma_specific_literals_live_only_in_model_families():
    """The point of `model_families.py` was to put every model-
    specific regex/string in one place so swapping Gemma → Qwen
    doesn't mean grepping six files. This test catches drift: if
    a new `<|tool_call>` regex sneaks into e.g. runtime.py outside
    the adapter path, the adapter pattern has been bypassed.

    Detection uses AST, so docstrings / comments that mention the
    markers (e.g. "# strip <unused25>") don't trigger — only real
    runtime use of the literal does.

    Allowed locations:
      • src/localcode/model_families.py (the adapter itself)
    """
    ALLOWED = {
        (SRC / "model_families.py").resolve(),
        # The cross-family tool-call repair parser legitimately needs every
        # family's markup (it recovers tool calls the server failed to parse,
        # for ALL families) — the one place besides the adapters that must know
        # the literals directly.
        (SRC / "tools" / "tool_call_repair.py").resolve(),
    }
    violations: list[str] = []
    for path in _iter_py(SRC):
        if path.resolve() in ALLOWED:
            continue
        hits = _contains_gemma_literal_in_code(path)
        if hits:
            violations.append(
                f"{path.relative_to(REPO)}: " + " | ".join(repr(h) for h in hits)
            )
    assert not violations, (
        "Gemma-specific literals used at runtime outside model_families.py:\n  "
        + "\n  ".join(violations)
        + "\nRoute through model_families.get_adapter(family) to pick the "
          "right markers per active model, or widen ALLOWED above with a "
          "justification comment."
    )


# ── Rule 4 — file size caps ─────────────────────────────────────────


# File-size cap tests removed 2026-07-07 — they were a self-imposed guardrail
# that blocked landing real features (repeatedly forcing comment-trimming to
# satisfy an arbitrary LoC ceiling). Split files when it aids readability, not
# because a test demands it.


# ── Rule 5 — agent/ submodules declare __all__ ─────────────────────


def test_agent_submodules_declare_all():
    """Every agent/*.py declares an `__all__`, even if empty (meaning
    "nothing public" — all internal). The empty-list pattern is
    fine for context.py / helpers.py where everything is
    underscore-prefixed and the loop reaches in by name. What we
    don't tolerate is NO declaration at all, which leaves the
    public surface undefined.
    """
    agent_dir = SRC / "agent"
    violations: list[str] = []
    for path in _iter_py(agent_dir):
        if path.name == "__init__.py":
            # __init__ is the public umbrella; checked separately
            pass
        text = path.read_text(errors="replace")
        if "__all__" not in text:
            violations.append(str(path.relative_to(REPO)))
    assert not violations, (
        "agent/ submodules missing __all__:\n  " + "\n  ".join(violations)
        + "\nAdd `__all__ = [...]` near the top of the file to declare "
          "its public surface, or `__all__: list[str] = []` if nothing "
          "is meant to be public."
    )


# ── Rule 6 — observability consistency ─────────────────────────────


# Files where `print()` at module or function level is legitimate —
# CLI entry points (`python -m localcode.errors --emit-docs`),
# standalone utility scripts, or debug-only verbose output. Every
# file listed here has a justification comment below. New entries
# need a justification; the pattern says we'd rather expand the
# allowlist with a reason than silently let printfs leak into the
# agent loop.
PRINT_ALLOWLIST = {
    "diagnostics.py",  # Offline `localcode doctor` CLI emits its JSON report.
    "launch.py",  # UI launcher: prints to the terminal BEFORE the UI runtime starts (no event bus yet)
    "ports.py",  # UI launcher: prints to the terminal BEFORE the UI runtime starts (no event bus yet)
    "server_cmd.py",  # UI launcher: prints to the terminal BEFORE the UI runtime starts (no event bus yet)
    # CLI entry point: `python -m localcode.errors` emits the
    # error-code docs table. Not reachable from the agent loop.
    "errors.py",
    # Standalone recovery script — `localcode-unstick` binary target.
    # Runs in isolation, prints to terminal to surface what it cleaned
    # up; not invoked from the TUI / app.py path.
    "recovery.py",
    # Turn-diff module: prints() shown in the class docstring as a
    # USAGE EXAMPLE; the AST walker skips docstrings but keep the
    # file here as defence-in-depth against someone moving the
    # example out of the docstring later.
    "turn_diff.py",
    # `python -m localcode.hf_quants` self-test under `if __name__ ==
    # "__main__"` — prints quant-fetch results to the terminal; not
    # reachable from the agent loop.
    "hf_quants.py",
    # CLI entry point (`localcode`/`python -m localcode.entrypoint`):
    # prints the restart/resume help banner. Not invoked from the
    # agent loop.
    "entrypoint.py",
    # CLI entry point (`localcode api`): prints the local endpoint details
    # to the terminal for other tools. Not invoked from the agent loop.
    "api.py",
    # CLI entry point (`localcode decide`): JSON results on stdout and errors
    # on stderr. Its inference helpers do not print into the TUI.
    "decision.py",
}


def test_no_bare_print_in_prod_code():
    """Production code routes user-facing output through
    `OutputManager` / `events.emit()`, never through bare `print()`.
    Ad-hoc prints skip the event bus, don't appear in events.jsonl,
    and can't be captured by the TUI — which means:

      • Debugging production issues gets harder (missing events).
      • The TUI draws over the print, corrupting the screen.
      • Tests / eval can't assert on what the model "said" via
        print, only via the recorded event stream.

    This rule catches drift before it ships. Docstring examples
    containing `print()` are filtered out (AST-based detection;
    same trick as the Gemma-literals rule). CLI entry points and
    standalone utility scripts are allowlisted with justification.
    """
    violations: list[str] = []
    for path in _iter_py(SRC):
        if path.name in PRINT_ALLOWLIST:
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue

        # Same docstring-skipping trick as rule 3.
        docstring_ids: set[int] = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not body or not isinstance(body, list):
                continue
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstring_ids.add(id(first.value))

        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                # Check if this call is inside a docstring — already
                # skipped by the `ast.Constant` path, but defensive.
                if id(node) in docstring_ids:
                    continue
                line = getattr(node, "lineno", "?")
                violations.append(f"{path.relative_to(REPO)}:{line}")
    assert not violations, (
        "Bare `print()` calls in production code — route through "
        "OutputManager / events.emit() instead:\n  "
        + "\n  ".join(violations)
        + "\nAllowlist legit sites (CLI entry points, utility scripts) "
          "in PRINT_ALLOWLIST above with a justification comment."
    )


# ── Rule 7 — smoke-call representative functions ───────────────────


def _collect_names(target: ast.expr, out: set[str]) -> None:
    """Recursively extract Name ids from an assignment LHS
    (handles tuple/list unpacking, starred targets)."""
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for el in target.elts:
            _collect_names(el, out)
    elif isinstance(target, ast.Starred):
        _collect_names(target.value, out)


def _check_module_resolves(path: Path) -> None:
    """Same logic as test_agent_loop_name_references_resolve, but
    parameterised so we can run it over every agent/ submodule.
    Fails the calling test on any unresolved reference."""
    import builtins
    tree = ast.parse(path.read_text())
    bound: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                bound.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.FunctionDef):
            bound.add(n.name)
            for arg in n.args.args: bound.add(arg.arg)
            for arg in n.args.posonlyargs: bound.add(arg.arg)
            for arg in n.args.kwonlyargs: bound.add(arg.arg)
            if n.args.vararg: bound.add(n.args.vararg.arg)
            if n.args.kwarg: bound.add(n.args.kwarg.arg)
        elif isinstance(n, ast.AsyncFunctionDef):
            bound.add(n.name)
            for arg in n.args.args: bound.add(arg.arg)
        elif isinstance(n, ast.ClassDef):
            bound.add(n.name)
        elif isinstance(n, ast.Lambda):
            for arg in n.args.args: bound.add(arg.arg)
            for arg in n.args.posonlyargs: bound.add(arg.arg)
            for arg in n.args.kwonlyargs: bound.add(arg.arg)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                _collect_names(t, bound)
        elif isinstance(n, ast.AugAssign):
            _collect_names(n.target, bound)
        elif isinstance(n, ast.AnnAssign):
            _collect_names(n.target, bound)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            _collect_names(n.target, bound)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.withitem) and n.optional_vars:
            _collect_names(n.optional_vars, bound)
        elif isinstance(n, ast.comprehension):
            _collect_names(n.target, bound)
        elif isinstance(n, ast.NamedExpr):
            _collect_names(n.target, bound)

    builtin_names = set(dir(builtins))
    unresolved: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id in bound or n.id in builtin_names:
                continue
            unresolved.add(f"{path.name}:{n.lineno} — unresolved {n.id!r}")

    assert not unresolved, (
        f"Unresolved name references in {path.relative_to(REPO)}:\n  "
        + "\n  ".join(sorted(unresolved))
        + "\nAdd to imports or bind locally."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
