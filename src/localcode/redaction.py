"""Value-level secret redaction for anything LocalCode persists to disk.

Why this exists
---------------
LocalCode writes three durable, cleartext artifacts:

  • `.localcode/events.jsonl`   — every tool call, arg and result summary
  • `.localcode/sessions/*.json` — the verbatim transcript
  • `history.db`                — prompts, responses, tool args/results

A user pasting an API key into a prompt, a `cat .env` in a bash tool
result, or a model echoing a token back all end up in those files
permanently. `protocol/outcomes.redact()` only masks *dict keys* named
like secrets, which does nothing for a key sitting inside a blob of
text. `scrub()` here works on the VALUE: it rewrites the token itself,
wherever it appears in a string.

Scope — deliberately narrow
---------------------------
Only high-precision, self-identifying credential formats are matched:
vendor-prefixed tokens (`AKIA…`, `ghp_…`, `sk-ant-…`), structural JWTs,
and PEM private-key blocks. Each has a distinctive literal prefix, so a
false positive is essentially impossible.

We deliberately do NOT add generic entropy heuristics ("any 32+ char
high-entropy string is a secret"). On a coding agent's transcript those
fire constantly on legitimate content — git SHAs, content hashes,
base64 test fixtures, minified JS, UUIDs, lockfile integrity hashes —
and silently corrupting a user's own source code in their session log
is far worse than missing an unprefixed secret. Precision over recall
is the intentional trade-off here.

Cost
----
Patterns are compiled once at import. `scrub()` short-circuits on
strings with no candidate substring, so the common case (ordinary code
and prose) costs one `any()` over a handful of `in` checks.
"""
from __future__ import annotations

import re
from typing import Any


__all__ = ["scrub", "scrub_obj", "has_secret"]


# (compiled pattern, replacement marker). Order matters only where one
# family is a prefix of another — `sk-ant-` must be tried before the
# generic `sk-`, so it lands first in the list.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # PEM private keys — whole block, header through footer. DOTALL so
    # the body's newlines are consumed. Non-greedy so two adjacent keys
    # don't collapse into one match.
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                re.DOTALL),
     "[redacted:private-key]"),

    # AWS access key ids (long-term, temporary/STS, bearer, context).
    (re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"),
     "[redacted:aws-key]"),

    # GitHub: personal/oauth/user-to-server/server-to-server/refresh
    # tokens, plus the newer fine-grained `github_pat_` form.
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
     "[redacted:github-token]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
     "[redacted:github-token]"),

    # Anthropic before OpenAI: `sk-ant-…` also matches the `sk-…` rule.
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
     "[redacted:anthropic-key]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
     "[redacted:openai-key]"),

    # Stripe live secret keys. Before nothing in particular, but note
    # `sk_live_` uses an underscore so the `sk-` rule never sees it.
    (re.compile(r"\bsk_live_[0-9a-zA-Z]{20,}"),
     "[redacted:stripe-key]"),

    # HuggingFace user access tokens.
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
     "[redacted:hf-token]"),

    # Slack bot/app/user/refresh/legacy tokens.
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
     "[redacted:slack-token]"),

    # Google API keys — fixed 39-char total length.
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
     "[redacted:google-key]"),

    # npm automation/publish tokens — fixed 36-char body.
    (re.compile(r"\bnpm_[A-Za-z0-9]{36}"),
     "[redacted:npm-token]"),

    # JWTs: three base64url segments. Anchored on the `eyJ` header
    # prefix ( `{"` base64-encoded ), which is what makes this safe to
    # match structurally rather than by entropy.
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
     "[redacted:jwt]"),
]


# Cheap pre-filter. If none of these substrings occur, no pattern above
# can possibly match, so we skip the regex pass entirely. Keep this list
# in sync with the prefixes used in `_PATTERNS`.
_CANDIDATES: tuple[str, ...] = (
    "AKIA", "ASIA", "ABIA", "ACCA",
    "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_",
    "sk-", "sk_live_", "hf_", "xox", "AIza", "npm_", "eyJ",
    "PRIVATE KEY",
)

# Shortest thing any pattern can match (`xox?-` + 10). Strings below
# this can't contain a secret, so don't even run the substring scan.
_MIN_LEN = 14


def has_secret(text: str) -> bool:
    """True if `text` contains at least one recognised credential."""
    if not text or len(text) < _MIN_LEN:
        return False
    if not any(c in text for c in _CANDIDATES):
        return False
    return any(pat.search(text) for pat, _ in _PATTERNS)


def scrub(text: str) -> str:
    """Replace recognised credentials in `text` with stable markers.

    Returns `text` unchanged (same object) when nothing matches, which
    is the overwhelmingly common case. Never raises: a non-`str` input
    is returned as-is so callers can map this over mixed payloads
    without type-checking first.
    """
    if not isinstance(text, str):
        return text
    if len(text) < _MIN_LEN:
        return text
    if not any(c in text for c in _CANDIDATES):
        return text
    for pattern, marker in _PATTERNS:
        text = pattern.sub(marker, text)
    return text


def scrub_obj(obj: Any) -> Any:
    """Recursively `scrub()` every string inside a JSON-shaped value.

    Dict keys are scrubbed too — a secret can end up as a key when a
    tool result is turned into a mapping. Non-container, non-string
    leaves pass through untouched. Depth is bounded by the caller's
    data; our payloads are shallow (events, session dicts).
    """
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, dict):
        return {scrub_obj(k): scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(scrub_obj(v) for v in obj)
    return obj
