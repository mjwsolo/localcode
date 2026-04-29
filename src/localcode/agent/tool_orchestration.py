"""Tool-call orchestration helpers for the agent loop.

Keeps the loop focused on agent control flow while this module owns the
question "which tool calls from this round can be prefetched in parallel?".
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from typing import Any

from ..tools import is_concurrency_safe


def prefetch_parallel_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    should_skip: callable,
    execute_tool: callable,
) -> tuple[dict[int, object], ThreadPoolExecutor | None]:
    """Dispatch concurrency-safe tool calls in parallel.

    `should_skip(idx, name, args) -> bool` lets the loop keep ownership of
    dedup/cache policy while this module owns concurrency policy.
    """
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for idx, tc in enumerate(tool_calls):
        fn = tc.get("function", {}) or {}
        name = (fn.get("name", "") or "").strip()
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except Exception:
            continue
        if not is_concurrency_safe(name, args):
            continue
        if should_skip(idx, name, args):
            continue
        candidates.append((idx, name, args))

    if len(candidates) < 2:
        return {}, None

    pool = ThreadPoolExecutor(
        max_workers=min(len(candidates), 6),
        thread_name_prefix="lc-parallel-tool",
    )
    futures: dict[int, object] = {}
    for idx, name, args in candidates:
        futures[idx] = pool.submit(execute_tool, name, args)
    return futures, pool
