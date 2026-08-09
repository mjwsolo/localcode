"""Tests for localcode.protocol — the public run --json parser/normalizer.

Golden JSONL streams live in tests/protocol_fixtures/ and are the shared
compatibility fixtures the integration plan (§7.2.1, Phase 1 exit criteria)
requires: LocalCode core, the native eval harness, and the future Harbor plugin
must all produce identical normalized results from these same files.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localcode.protocol import (
    FailureCategory,
    SchemaVersion,
    UnsupportedSchemaError,
    normalize_reason,
    outcome_from_parse,
    parse_stream,
    redact,
    SCHEMA_VERSION,
)

FIXTURES = Path(__file__).parent / "protocol_fixtures"


def _parse(name: str):
    return parse_stream((FIXTURES / name).read_text().splitlines())


# ── happy path ──────────────────────────────────────────────────────

def test_success_stream_parses_and_normalizes():
    parsed = _parse("success.jsonl")
    assert parsed.ok
    assert parsed.terminal is not None
    assert str(parsed.schema) == "1.0"
    out = outcome_from_parse(parsed)
    assert out.category == FailureCategory.COMPLETED
    assert out.clean_finish is True
    assert out.rounds == 2
    assert out.tool_calls == 2
    assert out.tool_failures == 0
    assert (out.prompt_tokens, out.completion_tokens, out.total_tokens) == (1200, 340, 1540)
    assert out.final_text == "Done."


def test_clean_finish_is_not_task_pass():
    # The parser reports LocalCode's own completion, never task success —
    # the category is COMPLETED, never TASK_PASSED (verifier-owned).
    out = outcome_from_parse(_parse("success.jsonl"))
    assert out.category != FailureCategory.TASK_PASSED


# ── failure categories ──────────────────────────────────────────────

def test_reasoning_loop_stream_categorized():
    out = outcome_from_parse(_parse("reasoning_loop.jsonl"))
    assert out.category == FailureCategory.REASONING_LOOP_ABORT
    assert out.clean_finish is False
    assert out.raw_reason == "thinking_loop_exhausted"


def test_tool_failure_stream_counts_and_categorizes():
    parsed = _parse("tool_failure.jsonl")
    out = outcome_from_parse(parsed)
    assert out.tool_failures == 1
    assert out.category == FailureCategory.TOOL_FAILURE


# ── protocol violations ─────────────────────────────────────────────

def test_missing_terminal_is_protocol_error():
    parsed = _parse("no_terminal.jsonl")
    assert not parsed.ok
    assert any(p.kind == "missing_terminal" for p in parsed.problems)
    out = outcome_from_parse(parsed)
    assert out.category == FailureCategory.EVENT_PROTOCOL_ERROR
    assert out.protocol_ok is False


def test_duplicate_terminal_is_protocol_error():
    parsed = _parse("duplicate_terminal.jsonl")
    assert any(p.kind == "duplicate_terminal" for p in parsed.problems)
    assert outcome_from_parse(parsed).category == FailureCategory.EVENT_PROTOCOL_ERROR


# ── resilience ──────────────────────────────────────────────────────

def test_malformed_lines_tolerated_but_recorded():
    parsed = _parse("malformed.jsonl")
    # Two junk lines skipped, but the terminal + real events still parsed.
    assert parsed.terminal is not None
    assert parsed.ok  # terminal present, no missing/duplicate problems
    assert sum(1 for p in parsed.problems if p.kind == "malformed_line") == 2
    assert outcome_from_parse(parsed).category == FailureCategory.COMPLETED


def test_unknown_event_types_and_fields_preserved():
    parsed = _parse("unknown_future.jsonl")
    # A newer producer's event type is kept, not dropped.
    novel = [e for e in parsed.events if e.type == "quantum_reasoning_step"]
    assert novel and novel[0].get("novel_field") == "future"
    # Additive field on a known event is preserved on the envelope.
    call = [e for e in parsed.events if e.type == "tool_call"][0]
    assert call.get("brand_new_field") == 42
    # Same-major (1.9) is still readable and normalizes cleanly.
    out = outcome_from_parse(parsed)
    assert out.category == FailureCategory.COMPLETED


def test_unsupported_major_schema_raises():
    with pytest.raises(UnsupportedSchemaError):
        _parse("unsupported_major.jsonl")


# ── unit: schema version ────────────────────────────────────────────

@pytest.mark.parametrize("raw,major,minor", [
    (1, 1, 0), ("1", 1, 0), ("1.0", 1, 0), ("1.9", 1, 9), ("2.3", 2, 3),
    ("", 1, 0), ("garbage", 1, 0),
])
def test_schema_version_parse(raw, major, minor):
    sv = SchemaVersion.parse(raw)
    assert (sv.major, sv.minor) == (major, minor)


def test_schema_readable_by_major_only():
    assert SchemaVersion(1, 5).readable_by(1) is True
    assert SchemaVersion(2, 0).readable_by(1) is False
    assert SCHEMA_VERSION == "1.0"


# ── unit: reason normalization ──────────────────────────────────────

@pytest.mark.parametrize("status,reason,expected", [
    ("ok", "completed", FailureCategory.COMPLETED),
    ("ok", "verified_run_or_launch", FailureCategory.COMPLETED),
    ("timeout", "run exceeded 300s", FailureCategory.AGENT_TIMEOUT),
    ("interrupted", "keyboard interrupt", FailureCategory.AGENT_INCOMPLETE),
    ("error", "no model found on disk", FailureCategory.SETUP_FAILURE),
    ("error", "could not start the model server", FailureCategory.MODEL_SERVER_FAILURE),
    ("error", "stream_error:ConnectionError", FailureCategory.MODEL_SERVER_FAILURE),
    ("incomplete", "thinking_loop_exhausted", FailureCategory.REASONING_LOOP_ABORT),
    ("incomplete", "max_output_recovery_exhausted", FailureCategory.CONTEXT_EXHAUSTED),
    ("incomplete", "truncated_tool_call_exhausted", FailureCategory.TOOL_FAILURE),
    ("incomplete", "stall_exhausted", FailureCategory.AGENT_INCOMPLETE),
    ("incomplete", "max_rounds", FailureCategory.AGENT_INCOMPLETE),
    ("error", "something we never named", FailureCategory.AGENT_CRASH),
])
def test_normalize_reason(status, reason, expected):
    assert normalize_reason(status, reason) == expected


# ── unit: redaction ─────────────────────────────────────────────────

def test_redact_masks_secret_fields_recursively():
    cfg = {
        "model": "qwen",
        "api_key": "sk-live-xxx",
        "headers": {"Authorization": "Bearer abc", "X-Trace": "ok"},
        "servers": [{"token": "t-123", "url": "http://x"}],
    }
    red = redact(cfg)
    assert red["model"] == "qwen"                       # untouched
    assert red["api_key"] == "***REDACTED***"
    assert red["headers"]["Authorization"] == "***REDACTED***"
    assert red["headers"]["X-Trace"] == "ok"
    assert red["servers"][0]["token"] == "***REDACTED***"
    assert red["servers"][0]["url"] == "http://x"
    # input not mutated
    assert cfg["api_key"] == "sk-live-xxx"


# ── regression: missing terminal fields must be "", never "None" ─────

def test_missing_terminal_fields_are_empty_not_none_string():
    """A result event may omit optional fields (final_text especially). The
    normalized outcome must surface "" for a missing field, never the literal
    string "None" — regression: str(None) is truthy, so a trailing `or ""`
    could not rescue it."""
    # final_text omitted (the common case — it's optional)
    out = outcome_from_parse(parse_stream(
        ['{"type":"result","status":"ok","reason":"completed"}']))
    assert out.final_text == ""
    # status omitted entirely
    out2 = outcome_from_parse(parse_stream(
        ['{"type":"result","reason":"completed"}']))
    assert out2.status == ""
    assert out2.raw_reason == "completed"
    # present values still pass through untouched
    out3 = outcome_from_parse(parse_stream(
        ['{"type":"result","status":"ok","reason":"completed","final_text":"hi"}']))
    assert out3.final_text == "hi"
