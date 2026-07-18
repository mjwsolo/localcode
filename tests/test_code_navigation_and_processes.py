from __future__ import annotations

import re
import time
from pathlib import Path
from types import SimpleNamespace

from localcode.process_registry import load_records
from localcode.tools import background_process, code_navigation
from localcode.tools.base import ToolContext


def _ctx(root: Path):
    # Real ToolContext (not a bare namespace) so tools get resolve_path —
    # the corrupted-absolute-path healer every file tool now routes through.
    app = SimpleNamespace(repo_root=root)
    return ToolContext(app=app, out=None)  # type: ignore[arg-type]


def test_code_navigation_python_symbols_definitions_and_references(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text(
        "VALUE = 1\n\nclass Widget:\n    pass\n\ndef build_widget():\n    return Widget()\n"
    )
    ctx = _ctx(tmp_path)

    symbols = code_navigation.execute(ctx, {"action": "symbols", "path": "mod.py"})
    assert "mod.py:3: class Widget" in symbols
    assert "mod.py:6: function build_widget" in symbols
    assert "mod.py:1: variable VALUE" in symbols

    definition = code_navigation.execute(
        ctx, {"action": "definition", "symbol": "Widget"}
    )
    assert definition == "mod.py:3: class Widget"

    references = code_navigation.execute(
        ctx, {"action": "references", "symbol": "Widget"}
    )
    assert "mod.py:3:" in references
    assert "mod.py:7:" in references


def test_code_navigation_rejects_paths_outside_repo(tmp_path: Path) -> None:
    result = code_navigation.execute(
        _ctx(tmp_path), {"action": "symbols", "path": "../outside.py"}
    )
    assert result.startswith("Error: path must stay inside")


def test_background_process_has_stable_id_owner_and_incremental_poll(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    started = background_process.execute(
        ctx,
        {
            "action": "start",
            "command": "printf first; sleep 2; printf second",
            "owner": "test-agent",
        },
    )
    match = re.search(r"(proc-[0-9a-f]+)", started)
    assert match
    process_id = match.group(1)
    try:
        deadline = time.time() + 1
        first = ""
        while time.time() < deadline and "first" not in first:
            first = background_process.execute(
                ctx, {"action": "poll", "process_id": process_id, "offset": 0}
            )
            time.sleep(0.02)
        assert "status=running" in first
        assert "first" in first
        offset = int(re.search(r"next_offset=(\d+)", first).group(1))
        second_poll = background_process.execute(
            ctx, {"action": "poll", "process_id": process_id, "offset": offset}
        )
        assert "first" not in second_poll
        records = load_records(tmp_path)
        assert records[-1].process_id == process_id
        assert records[-1].owner == "test-agent"
        assert str(tmp_path / ".localcode" / "process-logs") in records[-1].log_path
    finally:
        assert background_process.execute(
            ctx, {"action": "stop", "process_id": process_id}
        ) == f"Stopped {process_id}."


def test_background_process_requires_id_for_poll(tmp_path: Path) -> None:
    result = background_process.execute(_ctx(tmp_path), {"action": "poll"})
    assert result == "Error: process_id is required for poll."
