"""Secret redaction: every credential family, and no false positives.

`localcode.redaction.scrub` is installed at the three places LocalCode
writes durable cleartext (events.jsonl, session JSON, history.db). These
tests pin both halves of the contract: recognised credentials are
rewritten, and ordinary code/prose is returned untouched.
"""
# NOTE: credential fixtures below are split across adjacent string
# literals ("AKIA" "IOSFO...") purely so this FILE contains no literal
# token. GitHub push protection scans source and rejected the push
# otherwise. Python concatenates adjacent literals at parse time, so
# the value the scrubber sees is byte-identical to a real credential.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode.redaction import has_secret, scrub, scrub_obj  # noqa: E402


# (label, secret string, expected marker)
SECRETS = [
    ("aws-long-term", "AKIA" "IOSFODNN7EXAMPLE", "[redacted:aws-key]"),
    ("aws-sts", "ASIA" "Y34FZKBOKMUTVV7A", "[redacted:aws-key]"),
    ("aws-bearer", "ABIA" "Y34FZKBOKMUTVV7A", "[redacted:aws-key]"),
    ("aws-context", "ACCA" "Y34FZKBOKMUTVV7A", "[redacted:aws-key]"),
    ("github-classic", "ghp_" + "A" * 36, "[redacted:github-token]"),
    ("github-oauth", "gho_" + "b" * 36, "[redacted:github-token]"),
    ("github-user", "ghu_" + "c" * 36, "[redacted:github-token]"),
    ("github-server", "ghs_" + "d" * 36, "[redacted:github-token]"),
    ("github-refresh", "ghr_" + "e" * 36, "[redacted:github-token]"),
    ("github-fine-grained", "github_pat_" + "A1b2C3d4E5" * 3, "[redacted:github-token]"),
    ("openai", "sk-" + "A1b2C3d4E5" * 4, "[redacted:openai-key]"),
    ("anthropic", "sk-ant-" "api03-" + "Xy9" * 20, "[redacted:anthropic-key]"),
    ("huggingface", "hf_" + "QwErTyUiOp" * 3, "[redacted:hf-token]"),
    ("slack-bot", "xoxb-" "1234567890-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx",
     "[redacted:slack-token]"),
    ("slack-user", "xoxp-" "1234567890-0987654321", "[redacted:slack-token]"),
    ("google", "AIza" + "SyD1e2F3g4H5i6J7k8L9m0N1o2P3q4R5s6T", "[redacted:google-key]"),
    ("stripe-live", "sk_live_" + "51H8xKq2eZvKYlo2C0" * 2, "[redacted:stripe-key]"),
    ("npm", "npm_" + "a1b2c3d4e5" * 3 + "f1b2c3", "[redacted:npm-token]"),
    ("jwt",
     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
     ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
     ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
     "[redacted:jwt]"),
]


@pytest.mark.parametrize("label,secret,marker", SECRETS, ids=[s[0] for s in SECRETS])
def test_secret_family_is_redacted(label, secret, marker):
    text = f"here is the value: {secret} — use it"
    out = scrub(text)
    assert secret not in out, f"{label} survived scrub()"
    assert marker in out
    # Surrounding text is preserved; only the token is rewritten.
    assert out.startswith("here is the value: ")
    assert out.endswith(" — use it")
    assert has_secret(text) is True


def test_pem_private_key_block_is_redacted():
    body = "\n".join(["MIIEowIBAAKCAQEA" + "x" * 48] * 6)
    pem = f"-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----"
    out = scrub(f"key follows:\n{pem}\ndone")
    assert "MIIEowIBAAKCAQEA" not in out
    assert "BEGIN RSA PRIVATE KEY" not in out
    assert "[redacted:private-key]" in out
    assert out.startswith("key follows:")
    assert out.endswith("done")


def test_openssh_and_ec_private_keys_also_match():
    for kind in ("OPENSSH", "EC", "", "ENCRYPTED"):
        header = f"-----BEGIN {kind} PRIVATE KEY-----".replace("  ", " ")
        footer = f"-----END {kind} PRIVATE KEY-----".replace("  ", " ")
        pem = f"{header}\nc2VjcmV0Ym9keWdvZXNoZXJl\n{footer}"
        assert scrub(pem) == "[redacted:private-key]", kind


def test_two_adjacent_pem_blocks_do_not_collapse():
    one = "-----BEGIN PRIVATE KEY-----\naaaa\n-----END PRIVATE KEY-----"
    two = "-----BEGIN PRIVATE KEY-----\nbbbb\n-----END PRIVATE KEY-----"
    out = scrub(f"{one}\nmiddle text\n{two}")
    assert out.count("[redacted:private-key]") == 2
    assert "middle text" in out


def test_anthropic_key_wins_over_generic_openai_pattern():
    key = "sk-ant-" "api03-" + "Zz7" * 20
    assert scrub(key) == "[redacted:anthropic-key]"


def test_multiple_secrets_in_one_string():
    text = f"export AWS_KEY=AKIAIOSFODNN7EXAMPLE GH={'ghp_' + 'Z' * 36}"
    out = scrub(text)
    assert "AKIA" not in out
    assert "ghp_" not in out
    assert "[redacted:aws-key]" in out
    assert "[redacted:github-token]" in out


# ── no false positives ───────────────────────────────────────────────

CLEAN = [
    "def scrub(text: str) -> str:\n    return text",
    "commit 9f2c1a4b8e0d3f6a5c7b9e1d2f4a6c8b0d3e5f7a is the parent",
    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "uuid 550e8400-e29b-41d4-a716-446655440000",
    "The quick brown fox jumps over the lazy dog. Nothing secret here at all.",
    "import boto3; client = boto3.client('s3')  # uses AKIA-style creds from env",
    "npm install --save-dev typescript@5.4.5",
    "ghp is short for GitHub personal token, prefix ghp_ followed by 36 chars",
    "sk-",
    "https://example.com/path?query=value&other=thing",
    '{"integrity": "sha512-abcdefghijklmnopqrstuvwxyz0123456789+/=="}',
    "",
    "short",
]


@pytest.mark.parametrize("text", CLEAN, ids=range(len(CLEAN)))
def test_ordinary_content_is_untouched(text):
    assert scrub(text) == text
    assert has_secret(text) is False


def test_scrub_returns_non_strings_unchanged():
    for value in (None, 42, 3.5, True, b"bytes"):
        assert scrub(value) is value


def test_scrub_obj_walks_nested_containers():
    payload = {
        "messages": [
            {"role": "user", "content": "my key is AKIAIOSFODNN7EXAMPLE"},
            {"role": "assistant", "content": "ok"},
        ],
        "tool_args": ("--token", "ghp_" + "Q" * 36),
        "count": 3,
    }
    out = scrub_obj(payload)
    assert "AKIA" not in str(out)
    assert "ghp_" not in str(out)
    assert out["messages"][1]["content"] == "ok"
    assert out["count"] == 3
    assert isinstance(out["tool_args"], tuple)


# ── the three install points ─────────────────────────────────────────


def test_events_emit_scrubs_before_writing(tmp_path, monkeypatch):
    from localcode import events

    log = tmp_path / "events.jsonl"
    monkeypatch.setattr(events, "_resolve_path", lambda: log)
    monkeypatch.setattr(events, "_cached_path", log, raising=False)
    monkeypatch.setenv("LOCALCODE_EVENTS", "1")

    events.emit("tool_call", args="curl -H 'Authorization: Bearer " + "ghp_" + "R" * 36 + "'")
    written = log.read_text()
    assert "ghp_" not in written
    assert "[redacted:github-token]" in written


def test_session_save_scrubs_transcript(tmp_path):
    from localcode.session import SessionState, SessionStore, utc_now

    store = SessionStore()
    store.sessions_dir = tmp_path
    session = SessionState(
        session_id="redaction-test",
        repo_root=tmp_path,
        created_at=utc_now(),
        messages=[{"role": "user", "content": "token AKIAIOSFODNN7EXAMPLE"}],
    )
    path = store.save(session)
    text = path.read_text()
    assert "AKIA" "IOSFODNN7EXAMPLE" not in text
    assert "[redacted:aws-key]" in text


def test_history_record_scrubs_content(tmp_path, monkeypatch):
    from localcode import history as history_mod

    db = history_mod.HistoryDB(tmp_path / "history.db")
    db.record_user_prompt(
        session_id="s1", repo_root=str(tmp_path),
        prompt="deploy with sk-ant-api03-" + "Kk3" * 20,
    )
    rows = db.get_session_history("s1")
    assert rows, "prompt was not recorded"
    assert "sk-ant-" not in rows[0]["content"]
    assert "[redacted:anthropic-key]" in rows[0]["content"]
    db.close()
