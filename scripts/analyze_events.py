#!/usr/bin/env python3
"""Analyze LocalCode events.jsonl turn outcomes.

Focuses on dynamic skill injection effectiveness:
skill_injection -> completion_status/tool_count/rounds.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _skill_key(skills: list[str]) -> str:
    return ",".join(skills) if skills else "(none)"


def analyze(path: Path) -> str:
    turns: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "skills": [],
        "skill_chars": 0,
        "skill_event_seen": False,
        "skill_candidates": [],
        "completion_status": "",
        "tools": 0,
        "rounds": 0,
        "duration_s": 0.0,
        "task_kind": "",
        "goal_type": "",
        "read_file_chars": 0,
        "edit_errors": 0,
        "write_existing_rejections": 0,
        "prompt_chars": [],
        "prompt_tokens": [],
        "ttft_ms": [],
        "decode_ms": [],
        "tool_arg_limits": 0,
        "tool_schemas": [],
        "redaction_saved": 0,
    })
    legacy_turns = 0
    for event in _load_events(path):
        turn_id = str(event.get("turn_id") or "")
        if not turn_id:
            continue
        rec = turns[turn_id]
        typ = event.get("type")
        if typ == "turn_start":
            rec["task_kind"] = event.get("task_kind", "")
            rec["goal_type"] = event.get("goal_type", "")
        elif typ == "skill_injection":
            rec["skill_event_seen"] = True
            rec["skills"] = list(event.get("skills") or [])
            rec["skill_chars"] = int(event.get("chars") or 0)
        elif typ == "skill_selection":
            rec["skill_candidates"] = list(event.get("candidates") or [])
        elif typ == "turn_end" and "completion_status" in event:
            rec["completion_status"] = event.get("completion_status", "")
            rec["tools"] = int(event.get("tools_called_count") or event.get("tools_count") or 0)
            rec["rounds"] = int(event.get("rounds") or 0)
            rec["duration_s"] = float(event.get("duration_s") or event.get("duration_ms") or 0)
            if event.get("duration_ms") and not event.get("duration_s"):
                rec["duration_s"] = rec["duration_s"] / 1000.0
        elif typ == "round_start":
            rec["prompt_chars"].append(int(event.get("prompt_chars") or 0))
            schemas = event.get("tool_schemas") or []
            if schemas:
                rec["tool_schemas"] = list(schemas)
        elif typ == "round_end":
            rec["prompt_tokens"].append(int(event.get("prompt_tokens") or 0))
            rec["ttft_ms"].append(int(event.get("ttft_ms") or 0))
            rec["decode_ms"].append(int(event.get("decode_ms") or 0))
            if event.get("tool_args_limited"):
                rec["tool_arg_limits"] += 1
        elif typ == "redaction":
            rec["redaction_saved"] += int(event.get("saved_total") or 0)
        elif typ == "tool_result":
            name = str(event.get("name") or "")
            preview = str(event.get("preview") or "")
            if name == "read_file":
                rec["read_file_chars"] += int(event.get("chars") or 0)
            if name in {"edit_file", "multi_edit", "edit_diff"} and (
                "old_string not found" in preview
                or "applied 0/" in preview.lower()
                or str(event.get("error")).lower() == "true"
            ):
                rec["edit_errors"] += 1
            if name == "write_file" and "already exists" in preview:
                rec["write_existing_rejections"] += 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in turns.values():
        if not rec.get("completion_status"):
            continue
        if not rec.get("skill_event_seen"):
            legacy_turns += 1
        grouped[_skill_key(rec["skills"])].append(rec)

    lines = [
        f"events: {path}",
        f"completed turn records: {sum(len(v) for v in grouped.values())}",
        f"turns without skill_injection: {legacy_turns}",
        "",
        "skill_combo\tturns\tcompleted\tblocked\tincomplete_or_error\tavg_tools\tavg_rounds\tavg_duration_s\tavg_skill_chars",
    ]
    for key, records in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        completed = sum(1 for r in records if r["completion_status"] == "completed")
        blocked = sum(1 for r in records if r["completion_status"] == "blocked_user_input")
        bad = len(records) - completed - blocked
        lines.append(
            "\t".join([
                key,
                str(len(records)),
                str(completed),
                str(blocked),
                str(bad),
                f"{mean(r['tools'] for r in records):.1f}",
                f"{mean(r['rounds'] for r in records):.1f}",
                f"{mean(r['duration_s'] for r in records):.1f}",
                f"{mean(r['skill_chars'] for r in records):.0f}",
            ])
        )
    candidate_counts: dict[str, int] = defaultdict(int)
    selected_counts: dict[str, int] = defaultdict(int)
    for rec in turns.values():
        for item in rec.get("skill_candidates") or []:
            name = str(item.get("name") or "")
            if name:
                candidate_counts[name] += 1
        for name in rec.get("skills") or []:
            selected_counts[str(name)] += 1
    if candidate_counts or selected_counts:
        lines.extend(["", "skill\tcandidate_turns\tselected_turns"])
        for name in sorted(set(candidate_counts) | set(selected_counts)):
            lines.append(f"{name}\t{candidate_counts.get(name, 0)}\t{selected_counts.get(name, 0)}")
    editing_turns = [
        rec for rec in turns.values()
        if rec.get("read_file_chars") or rec.get("edit_errors") or rec.get("write_existing_rejections")
    ]
    if editing_turns:
        lines.extend([
            "",
            "editing telemetry:",
            f"turns_with_file_reads: {len(editing_turns)}",
            f"avg_read_file_chars: {mean(r['read_file_chars'] for r in editing_turns):.0f}",
            f"edit_errors: {sum(r['edit_errors'] for r in editing_turns)}",
            f"write_existing_rejections: {sum(r['write_existing_rejections'] for r in editing_turns)}",
        ])
    round_records = [rec for rec in turns.values() if rec.get("prompt_chars") or rec.get("ttft_ms")]
    if round_records:
        all_prompt_chars = [v for r in round_records for v in r["prompt_chars"] if v]
        all_prompt_tokens = [v for r in round_records for v in r["prompt_tokens"] if v]
        all_ttft = [v for r in round_records for v in r["ttft_ms"] if v]
        all_decode = [v for r in round_records for v in r["decode_ms"] if v]
        lines.extend(["", "prompt/runtime telemetry:"])
        if all_prompt_chars:
            lines.append(f"avg_prompt_chars: {mean(all_prompt_chars):.0f}")
            lines.append(f"max_prompt_chars: {max(all_prompt_chars)}")
        if all_prompt_tokens:
            lines.append(f"avg_prompt_tokens: {mean(all_prompt_tokens):.0f}")
            lines.append(f"max_prompt_tokens: {max(all_prompt_tokens)}")
        if all_ttft:
            lines.append(f"avg_ttft_ms: {mean(all_ttft):.0f}")
            lines.append(f"max_ttft_ms: {max(all_ttft)}")
        if all_decode:
            lines.append(f"avg_decode_ms: {mean(all_decode):.0f}")
            lines.append(f"max_decode_ms: {max(all_decode)}")
        lines.append(f"tool_arg_limit_events: {sum(r['tool_arg_limits'] for r in round_records)}")
        lines.append(f"redaction_saved_chars: {sum(r['redaction_saved'] for r in round_records)}")

        slow = sorted(
            (
                r for r in round_records
                if (max(r["ttft_ms"] or [0]) > 10_000 or max(r["decode_ms"] or [0]) > 60_000)
            ),
            key=lambda r: max(max(r["ttft_ms"] or [0]), max(r["decode_ms"] or [0])),
            reverse=True,
        )[:5]
        if slow:
            lines.append("")
            lines.append("slow_turns:")
            for r in slow:
                lines.append(
                    f"- kind={r.get('task_kind') or '?'} status={r.get('completion_status') or '?'} "
                    f"max_prompt_chars={max(r['prompt_chars'] or [0])} "
                    f"max_ttft_ms={max(r['ttft_ms'] or [0])} "
                    f"max_decode_ms={max(r['decode_ms'] or [0])} "
                    f"tool_schemas={','.join(r.get('tool_schemas') or [])}"
                )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=".localcode/events.jsonl",
        help="Path to events.jsonl (default: .localcode/events.jsonl)",
    )
    args = parser.parse_args()
    print(analyze(Path(args.path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
