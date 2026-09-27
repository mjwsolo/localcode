"""How much context a session should use before compacting, measured on this machine.

The server is launched with the largest context that fits in memory (the RAM
ladder in runtime.py). That is the right ceiling for *fitting*, not for *speed*:
on the same Mac a model's decode speed halves and its prefill speed drops four
to six times between a 16k and a 64k prompt, so a session that fills 120k
tokens spends minutes on every cache miss and crawls on every step.

The budget is derived from the server's own timing lines (llama-server prints
one per request into server.log, which the supervisor owns):

    slot print_timing: id 0 | task 42 | prompt eval time = 727 ms / 650 tokens (...)
    slot print_timing: id 0 | task 42 |        eval time = 16471 ms / 852 tokens (...)
    slot release: id 0 | task 42 | stop processing: n_tokens = 31577, truncated = 0

Each request gives one sample: how long the prompt was, how fast prefill ran
at that depth, how fast decode ran at that depth. The budget is the largest
prompt length at which a full re-read still finishes within REREAD_MAX_S and
decode keeps at least DECODE_FLOOR of its short-prompt speed. Both constants
are latencies a person feels, never token counts: every token number here is
relative to the context the server actually loaded.
"""
from __future__ import annotations

import re
from statistics import median
from typing import NamedTuple

REREAD_MAX_S = 60.0    # a cache miss (compaction, restart, model switch) may cost this long
DECODE_FLOOR = 0.5     # keep at least half of the short-prompt decode speed
PP_MIN_CHUNK = 256     # prefill chunks smaller than this measure overhead, not speed
TG_MIN_TOKENS = 16     # decode timings on fewer tokens are noise
PRIOR_DEPTH_FACTOR = 0.6  # prefill at the budget depth is slower than the warm-up's cold prefill
GRID = 16              # candidate budgets: n_ctx/16 steps
FLOOR_DIV = 4          # never budget below n_ctx/4: the fixed prefix must fit with room to work

_TIMING = re.compile(r"task +(\d+) \| +(prompt eval time|eval time) = +([\d.]+) ms / +(\d+) tokens")
_RELEASE = re.compile(r"task +(\d+) \| stop processing: n_tokens = (\d+)")


class Sample(NamedTuple):
    n_tokens: int          # prompt length when the request finished
    pp_tps: float | None   # prefill tokens/s for this request (None: no prefill line)
    pp_n: int              # tokens prefilled
    tg_tps: float | None   # decode tokens/s
    tg_n: int              # tokens decoded


class TimingTable:
    """Accumulates samples from server.log text fed in any chunking."""

    def __init__(self) -> None:
        self._pending: dict[int, dict] = {}
        self._carry = ""
        self.samples: list[Sample] = []

    def feed(self, text: str) -> int:
        text = self._carry + text
        lines = text.split("\n")
        self._carry = lines.pop()  # partial last line
        added = 0
        for line in lines:
            m = _TIMING.search(line)
            if m:
                task = int(m.group(1))
                ms, n = float(m.group(3)), int(m.group(4))
                tps = n / (ms / 1000.0) if ms > 0 else None
                d = self._pending.setdefault(task, {})
                if m.group(2).startswith("prompt"):
                    d["pp_tps"], d["pp_n"] = tps, n
                else:
                    d["tg_tps"], d["tg_n"] = tps, n
                continue
            m = _RELEASE.search(line)
            if m:
                task, n_tokens = int(m.group(1)), int(m.group(2))
                d = self._pending.pop(task, None)
                if d is None:
                    continue
                self.samples.append(Sample(n_tokens, d.get("pp_tps"), int(d.get("pp_n", 0)),
                                           d.get("tg_tps"), int(d.get("tg_n", 0))))
                added += 1
        if len(self._pending) > 512:  # a request whose release line never came
            for k in sorted(self._pending)[:-256]:
                self._pending.pop(k, None)
        return added

    def clear(self) -> None:
        self._pending.clear(); self._carry = ""; self.samples.clear()


def _near(samples: list[Sample], c: int, step: int, key: str, min_n: str, min_val: int) -> float | None:
    """Median speed measured near prompt length c. With nothing near c, the closest
    measurement below c decides (speed never improves with depth); with nothing
    below either, the shallowest measurement stands in (optimistic)."""
    usable = [s for s in samples if getattr(s, key) is not None and getattr(s, min_n) >= min_val]
    if not usable:
        return None
    radius = max(step, c // 4)
    near = [getattr(s, key) for s in usable if abs(s.n_tokens - c) <= radius]
    if near:
        return float(median(near))
    below = [s for s in usable if s.n_tokens < c]
    pool = below if below else usable
    anchor = max(pool, key=lambda s: s.n_tokens) if below else min(pool, key=lambda s: s.n_tokens)
    band = [getattr(s, key) for s in pool if abs(s.n_tokens - anchor.n_tokens) <= radius]
    return float(median(band))


def budget(n_ctx: int, reserve: int, samples: list[Sample], prior_pp_tps: float | None = None,
           *, reread_max_s: float = REREAD_MAX_S, decode_floor: float = DECODE_FLOOR) -> tuple[int, dict]:
    """Prompt tokens a session may reach before compacting, and how that was decided.

    Everything is relative to n_ctx, the context the server actually loaded
    (per slot). `reserve` is the headroom one step may add (output plus tool
    results); the budget never exceeds n_ctx - reserve and never drops below
    n_ctx / FLOOR_DIV."""
    n_ctx = max(1, int(n_ctx))
    step = max(1, n_ctx // GRID)
    cap = max(1, n_ctx - max(0, int(reserve)))
    # The floor is the larger of a quarter of the context and twice the fixed
    # prefix (system prompt, tool schemas, first message). Below the prefix a
    # compaction cannot bring the prompt under the budget, so the runtime would
    # compact on every step; measured live at 16k ctx before this rule existed.
    prefix = prefix_tokens(samples)
    floor = max(1, n_ctx // FLOOR_DIV, 2 * prefix if prefix else 0)
    if floor > cap:
        floor = cap
    info: dict = {"n_ctx": n_ctx, "floor": floor, "cap": cap, "prefix": prefix, "samples": 0, "basis": "cap"}

    measured = [s for s in samples if s.pp_tps is not None and s.pp_n >= PP_MIN_CHUNK]
    if not measured:
        if prior_pp_tps and prior_pp_tps > 0:
            c = int(PRIOR_DEPTH_FACTOR * prior_pp_tps * reread_max_s)
            info.update(basis="prior", pp_prior=round(prior_pp_tps, 1))
            return _snap(min(cap, max(floor, c)), step, floor, cap), info
        return cap, info

    info["samples"] = len(samples)
    short = [s.tg_tps for s in samples if s.tg_tps is not None and s.tg_n >= TG_MIN_TOKENS and s.n_tokens < n_ctx // 8]
    if not short:
        deep = sorted((s for s in samples if s.tg_tps is not None and s.tg_n >= TG_MIN_TOKENS), key=lambda s: s.n_tokens)
        short = [s.tg_tps for s in deep[: max(1, len(deep) // 4)]] if deep else []
    tg_short = float(median(short)) if short else None
    info["tg_short"] = round(tg_short, 1) if tg_short else None

    best = floor
    for k in range(1, GRID + 1):
        c = step * k
        if c > cap:
            break
        pp = _near(samples, c, step, "pp_tps", "pp_n", PP_MIN_CHUNK)
        reread = (c / pp) if pp else 0.0
        if reread > reread_max_s:
            info.update(limit="reread", at=c, reread_s=round(reread, 1), pp_tps=round(pp or 0, 1))
            break
        if tg_short:
            tg = _near(samples, c, step, "tg_tps", "tg_n", TG_MIN_TOKENS)
            if tg is not None and tg < decode_floor * tg_short:
                info.update(limit="decode", at=c, tg_tps=round(tg, 1))
                break
        best = c
    info["basis"] = "measured"
    return _snap(best, step, floor, cap), info


def prefix_tokens(samples: list[Sample]) -> int:
    """Size of the fixed prompt prefix: the cold prefill of a session's first
    request reads the whole prompt, so the smallest large prefill seen is the
    best estimate. 0 until a cold request has been observed."""
    cold = [s.pp_n for s in samples if s.pp_n >= PP_MIN_CHUNK and s.pp_tps is not None]
    return min(cold) if cold else 0


def _snap(c: int, step: int, floor: int, cap: int) -> int:
    c = (c // step) * step
    return min(cap, max(floor, c))


def step_reserve(n_ctx: int, output: int, tool_output_bytes: int) -> int:
    """Tokens one model step can add: its reply plus a couple of large tool
    results. Bounded to half the context so tiny contexts keep room to work."""
    return min(max(1, n_ctx // 2), max(0, output) + 2 * (max(0, tool_output_bytes) // 4))
