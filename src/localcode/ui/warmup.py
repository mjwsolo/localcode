"""Prompt warm-up: prefill the first request's constant prefix at model load.

The first message of a session pays for the whole system prompt + tool
schemas (3-9k tokens: ~3 s on an M5, far longer on a 16 GB M1). Later turns
are free because llama-server keeps the longest common prefix per slot, and
its host prompt cache (--cache-ram) carries that state across the
title-generation request the runtime fires alongside the first message.

A fresh process has nothing cached. So, once per session, after the first
turn settles, the supervisor asks llama-server to save the slot
(--slot-save-path), reads the TOKEN IDS out of the head of that file, throws
the multi-hundred-MB state away and keeps the ids (a few KB). On the next
load of the same model it replays those ids through /completion with
n_predict=1: the slot (and the host cache) then hold the exact prefix
before the user has typed anything.

Why tokens and not the saved state: restoring the state file skips the
prompt-checkpoints that sliding-window models (Gemma) need for prefix reuse,
so a restored slot re-evaluates everything; replaying tokens rebuilds the
checkpoints (measured: 5,176-token prompt, 3,900 ms cold -> 540 ms after
replay + an intervening title request).

Two sessions' captures are merged to their common prefix, so the stored
tokens converge on the part that never changes (system prompt, tool
schemas) and drop the user's message. The environment block carries the
date, so a capture from another day shares only the prompt before it; when
the common prefix is under half of the new capture the new one replaces it.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import struct
import time
import urllib.request
from pathlib import Path

from localcode.ui import fork_commit, run_dir

MIN_CAPTURE_TOKENS = 512     # below this it is the title request, not the agent prompt
MAX_TOKENS = 12288           # never replay more than this (bounds the launch-time prefill)
SETTLE_S = 3.0               # the slot must stay idle on the same task this long before we capture
CAPTURE_WINDOW_S = 20 * 60   # give up capturing if no turn happens within this
SLOT_FILE = "warm.bin"


def enabled() -> bool:
    return os.environ.get("LOCALCODE_PROMPT_WARMUP", "1").strip().lower() not in {"0", "false", "no", "off"}


def slot_save_dir() -> Path:
    return run_dir() / "slots"


def warm_dir() -> Path:
    return run_dir() / "warm"


def _warm_file(alias: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in alias)
    return warm_dir() / f"{safe}.json"


# ---- pure helpers ----------------------------------------------------------

def parse_slot_tokens(path: Path) -> list[int]:
    """Token ids from a llama-server slot save file (llama_state_seq_save_file).

    Layout: u32 magic, u32 version, u32 n, then n u32 entries, then the KV
    state. The entries are server_tokens::serialize's packed form
    (LLAMA_TOKEN_NULL marker, format version, u32 count, the ids, media
    keys) or, from older servers, the plain id list.
    """
    with path.open("rb") as f:
        magic, _version, n = struct.unpack("<III", f.read(12))
        if magic != 0x67677371:  # 'ggsq'
            raise ValueError("not a llama slot save file")
        packed = list(struct.unpack(f"<{n}I", f.read(4 * n)))
    if packed and packed[0] == 0xFFFFFFFF:
        if len(packed) < 3:
            raise ValueError("truncated token header")
        count = packed[2]
        toks = packed[3:3 + count]
        if len(toks) != count:
            raise ValueError("truncated token list")
        return toks
    return packed


def common_prefix(a: list[int], b: list[int]) -> list[int]:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def merge_tokens(old: list[int] | None, new: list[int]) -> list[int]:
    """Keep what two captures share; replace when they share too little."""
    new = new[:MAX_TOKENS]
    if not old:
        return new
    lcp = common_prefix(old, new)
    if len(lcp) >= MIN_CAPTURE_TOKENS and len(lcp) * 2 >= len(new):
        return lcp
    return new


class CaptureTracker:
    """Decides when the first turn has settled enough to capture the slot.

    Feed it /slots snapshots; it answers with the slot id to save once a slot
    has been idle on the same task, with a prompt of agent size, for
    SETTLE_S. Tasks seen before `baseline` (the warm-up replay itself) are
    ignored.
    """

    def __init__(self, baseline_task: int | None = None, settle_s: float = SETTLE_S) -> None:
        self.baseline_task = baseline_task
        self.settle_s = settle_s
        self._idle_since: float | None = None
        self._idle_task: int | None = None

    def observe(self, slots: list[dict], now: float) -> int | None:
        if any(s.get("is_processing") for s in slots):
            self._idle_since = None
            return None
        best = None
        for s in slots:
            try:
                n = int(s.get("n_prompt_tokens") or 0)
                task = int(s.get("id_task") if s.get("id_task") is not None else -1)
            except (TypeError, ValueError):
                continue
            if n < MIN_CAPTURE_TOKENS or task < 0:
                continue
            if self.baseline_task is not None and task <= self.baseline_task:
                continue
            if best is None or n > best[1]:
                best = (int(s.get("id", 0)), n, task)
        if best is None:
            self._idle_since = None
            return None
        if self._idle_task != best[2]:
            self._idle_task, self._idle_since = best[2], now
            return None
        if self._idle_since is not None and now - self._idle_since >= self.settle_s:
            return best[0]
        return None


# ---- store -------------------------------------------------------------------

def load(alias: str) -> list[int] | None:
    try:
        data = json.loads(_warm_file(alias).read_text())
    except (OSError, ValueError):
        return None
    if data.get("fork") != fork_commit():
        return None
    toks = data.get("tokens")
    if not isinstance(toks, list) or not all(isinstance(t, int) for t in toks) or len(toks) < MIN_CAPTURE_TOKENS:
        return None
    return toks[:MAX_TOKENS]


def store(alias: str, tokens: list[int]) -> None:
    path = _warm_file(alias)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"fork": fork_commit(), "saved": int(time.time()), "tokens": tokens}))
    os.replace(tmp, path)


def forget(alias: str) -> None:
    try:
        _warm_file(alias).unlink()
    except OSError:
        pass


# ---- server I/O --------------------------------------------------------------

def _post(port: int, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def slots(port: int) -> list[dict] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=2) as r:
            data = json.loads(r.read().decode())
        return data if isinstance(data, list) else None
    except Exception:  # noqa: BLE001
        return None


class Replay:
    """One in-flight prefill of the stored prefix, cancellable from another thread.

    llama-server drops a task whose client went away (checked between
    batches), so closing the socket is the cancel: the tokens read so far
    stay in the slot as a partial prefix. The plugin asks for this the moment
    the user submits, so a still-running warm-up never delays a real turn by
    more than one batch.
    """

    def __init__(self, port: int, timeout: float = 900.0) -> None:
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        self.cancelled = False

    def run(self, tokens: list[int]) -> dict:
        body = json.dumps({"prompt": tokens, "n_predict": 1, "cache_prompt": True, "temperature": 0})
        self.conn.request("POST", "/completion", body=body, headers={"Content-Type": "application/json"})
        r = json.loads(self.conn.getresponse().read().decode())
        return r.get("timings") or {}

    def cancel(self) -> None:
        self.cancelled = True
        # shutdown() is what unblocks the reader thread; close() alone is
        # deferred while http.client's file wrapper holds the socket.
        sock = getattr(self.conn, "sock", None)
        try:
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001, S110
            pass


def replay(port: int, tokens: list[int], timeout: float = 900.0) -> dict:
    """Prefill the stored prefix. Returns llama-server's timings."""
    return Replay(port, timeout).run(tokens)


def capture(port: int, slot_id: int) -> list[int]:
    """Save the slot, keep its token ids, drop the state file."""
    d = slot_save_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / SLOT_FILE
    try:
        _post(port, f"/slots/{slot_id}?action=save", {"filename": SLOT_FILE}, 120)
        return parse_slot_tokens(path)
    finally:
        try:
            path.unlink()
        except OSError:
            pass
