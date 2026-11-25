"""`localcode bench`: measure how fast the downloaded models run on this Mac.

For each model it starts the same llama-server command LocalCode itself would
launch, sends identical synthetic code-like prompts of about 1K, 8K and 32K
tokens, and reports load time, time to first token, prompt reading speed,
generation speed and approximate peak memory. Everything stays on this machine: only
already-downloaded models are used, nothing is downloaded or uploaded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SIZES = (1024, 8192, 32768)          # nominal prompt tokens
MAX_TOKENS = 128
COOLDOWN_S = 10
HEALTH_TIMEOUT_S = 600
THROTTLE_RATIO = 0.85
SCHEMA_VERSION = 1


# ---------------------------------------------------------------- pure maths

def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def spread(values: list[float]) -> tuple[float, float] | None:
    return (min(values), max(values)) if values else None


def synthetic_prompt(tokens: int, run_id: str) -> str:
    """Deterministic code-like text of roughly `tokens` tokens. The unique
    first line defeats prefix caching so the prefill is always real."""
    lines = [f"# benchmark run {run_id}", "# Review the module below and summarise what it does in one sentence."]
    i = 0
    chars = 0
    target = int(tokens * 3.0)
    while chars < target:
        line = (f"def handler_{i}(items, limit={i % 17 + 3}):\n"
                f"    total = sum(x * {i % 7 + 1} for x in items[:limit])\n"
                f"    return {{'id': {i}, 'total': total, 'ok': total > {i * 3}}}\n")
        lines.append(line)
        chars += len(line)
        i += 1
    return "\n".join(lines)


@dataclass
class RunResult:
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    prefill_tps: float
    gen_tps: float


def summarize_run(t_start: float, t_first: float | None, t_end: float, *,
                  timings: dict | None, usage: dict | None, chunks: int) -> RunResult:
    """Turn one streamed response into numbers. The server's own `timings`
    win when present; otherwise fall back to wall-clock arithmetic."""
    usage = usage or {}
    timings = timings or {}
    first = t_first if t_first is not None else t_end
    ttft = max(first - t_start, 1e-9)
    prompt_tokens = int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0)
    completion = int(timings.get("predicted_n") or usage.get("completion_tokens") or chunks)
    prompt_total = int(usage.get("prompt_tokens") or prompt_tokens)
    if timings.get("prompt_per_second"):
        prefill = float(timings["prompt_per_second"])
    else:
        prefill = prompt_total / ttft
    if timings.get("predicted_per_second"):
        gen = float(timings["predicted_per_second"])
    else:
        gen = max(completion - 1, 0) / max(t_end - first, 1e-9)
    return RunResult(prompt_total or prompt_tokens, completion, ttft, prefill, gen)


def throttle_flags(gen_series: list[float], limit_before: int, limit_after: int) -> list[str]:
    flags = []
    if len(gen_series) >= 2 and gen_series[-1] < THROTTLE_RATIO * gen_series[0]:
        flags.append(f"generation slowed to {gen_series[-1] / gen_series[0]:.0%} of the first run")
    if min(limit_before, limit_after) < 100:
        flags.append(f"macOS reported a CPU speed limit of {min(limit_before, limit_after)}%")
    return flags


# ---------------------------------------------------------------- environment

def _run_text(cmd: list[str], timeout: float = 3) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout or ""
    except Exception:  # noqa: BLE001
        return ""


def power_warning() -> str | None:
    text = _run_text(["pmset", "-g", "batt"])
    if "Battery Power" in text:
        return "on battery power; results are usually lower than when plugged in"
    if re.search(r"low power mode\s*=\s*1|lowpowermode\s+1", _run_text(["pmset", "-g"]), re.I):
        return "Low Power Mode is on; results will be lower"
    return None


def serving_reason() -> str | None:
    """A LocalCode session or any llama-server is running: the GPU is busy."""
    try:
        from localcode.ui.ports import find_running
        if find_running() is not None:
            return "a LocalCode session is serving a model"
    except Exception:  # noqa: BLE001, S110
        pass
    out = _run_text(["pgrep", "-f", "llama-server"])
    mine = {str(os.getpid()), str(os.getppid())}
    if any(p for p in out.split() if p not in mine):
        return "a llama-server process is running"
    return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def thermal_snapshot() -> int:
    from localcode.thermal import read_cpu_speed_limit
    return read_cpu_speed_limit()


class MemorySampler(threading.Thread):
    """Samples a process every 0.5 s: `footprint` when available, else RSS."""

    def __init__(self, pid: int, interval: float = 0.5):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.peak_mb = 0.0
        self.source = "footprint" if os.path.exists("/usr/bin/footprint") else "rss"
        self._halt = threading.Event()

    def sample(self) -> float | None:
        if self.source == "footprint":
            out = _run_text(["/usr/bin/footprint", "-p", str(self.pid)], timeout=5)
            m = re.search(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB)", out)
            if m:
                return float(m.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[m.group(2)]
            self.source = "rss"
        out = _run_text(["ps", "-o", "rss=", "-p", str(self.pid)]).strip()
        return int(out) / 1024 if out.isdigit() else None

    def run(self) -> None:
        while not self._halt.is_set():
            v = self.sample()
            if v is not None:
                self.peak_mb = max(self.peak_mb, v)
            self._halt.wait(self.interval)

    def finish(self) -> float:
        self._halt.set()
        self.join(timeout=2)
        return self.peak_mb


def buffers_from_log(text: str) -> float:
    """Sum the model/KV/compute buffer sizes llama-server prints at load (MiB)."""
    return sum(float(x) for x in re.findall(r"buffer size\s*=\s*([\d.]+)\s*MiB", text))


# ---------------------------------------------------------------- HTTP

def _headers(key: str) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def wait_healthy(base: str, proc, deadline_s: float = HEALTH_TIMEOUT_S) -> bool:
    from localcode.ui.http import local_urlopen
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if proc.poll() is not None:
            return False
        try:
            with local_urlopen(urllib.request.Request(base.removesuffix("/v1") + "/health"), timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.25)
    return False


class Cancelled(Exception):
    """The user cancelled a benchmark run."""


class CancelToken:
    """Lets another thread abort a benchmark: sets a flag and closes the request in flight."""

    def __init__(self) -> None:
        self.cancelled = False
        self.resp = None

    def cancel(self) -> None:
        self.cancelled = True
        resp = self.resp
        if resp is not None:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled()


def stream_request(base: str, key: str, model: str, prompt: str, timeout: float = 1800,
                   cancel: CancelToken | None = None) -> RunResult:
    from localcode.ui.http import local_urlopen
    body = {"model": model, "stream": True, "max_tokens": MAX_TOKENS, "temperature": 0,
            "cache_prompt": False, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers=_headers(key))
    t0 = time.monotonic()
    t_first = None
    chunks = 0
    timings = usage = None
    if cancel:
        cancel.check()
    with local_urlopen(req, timeout=timeout) as resp:
        if cancel:
            cancel.resp = resp
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if obj.get("timings"):
                    timings = obj["timings"]
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        chunks += 1
                        if t_first is None:
                            t_first = time.monotonic()
        except Exception:
            if cancel and cancel.cancelled:
                raise Cancelled() from None  # the close() from cancel() surfaces as a read error
            raise
        finally:
            if cancel:
                cancel.resp = None
    if cancel:
        cancel.check()
    return summarize_run(t0, t_first, time.monotonic(), timings=timings, usage=usage, chunks=chunks)


# ---------------------------------------------------------------- per model

@dataclass
class ModelReport:
    name: str
    status: str = "ok"                 # ok | skipped | failed | not downloaded
    reason: str = ""
    ctx: int = 0
    sizes: dict[int, dict] = field(default_factory=dict)
    skipped_sizes: list[int] = field(default_factory=list)
    peak_memory_mb: float | None = None
    memory_source: str = ""
    load_buffers_mb: float | None = None
    load_s: float | None = None          # server start until it answers /health
    first_reply_s: float | None = None   # first short reply after loading (one-off GPU setup)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict) -> "ModelReport":
        r = cls(str(d.get("model", "")), status=d.get("status", "ok"), reason=d.get("reason", ""),
                ctx=int(d.get("context") or 0))
        for k, v in (d.get("sizes") or {}).items():
            for key in ("ttft_range", "prefill_range", "gen_range"):
                if v.get(key) is not None:
                    v[key] = tuple(v[key])
            r.sizes[int(k)] = v
        r.skipped_sizes = [int(x) for x in d.get("skipped_sizes") or []]
        r.peak_memory_mb = d.get("peak_memory_mb")
        r.memory_source = d.get("memory_source") or ""
        r.load_buffers_mb = d.get("load_buffers_mb")
        r.load_s = d.get("load_s")
        r.first_reply_s = d.get("first_reply_s")
        r.warnings = list(d.get("warnings") or [])
        return r

    def to_json(self) -> dict:
        return {"model": self.name, "status": self.status, "reason": self.reason, "context": self.ctx,
                "sizes": {str(k): v for k, v in self.sizes.items()}, "skipped_sizes": self.skipped_sizes,
                "peak_memory_mb": self.peak_memory_mb, "memory_source": self.memory_source,
                "load_buffers_mb": self.load_buffers_mb, "load_s": self.load_s,
                "first_reply_s": self.first_reply_s, "warnings": self.warnings}


def ctx_from_command(cmd: list[str]) -> int:
    ctx = int(cmd[cmd.index("--ctx-size") + 1])
    for flag in ("--parallel", "-np"):
        if flag in cmd:
            try:
                ctx //= max(1, int(cmd[cmd.index(flag) + 1]))
            except (ValueError, IndexError):
                pass
    return ctx


def fits(size: int, ctx: int) -> bool:
    return int(size * 1.3) + MAX_TOKENS + 64 <= ctx


def bench_model(path: Path, runs: int, *, progress=print) -> ModelReport:
    from localcode.ui.server_cmd import server_command
    name = path.name.removesuffix(".gguf")
    report = ModelReport(name)
    port = free_port()
    try:
        cmd = server_command(str(path), port, name)
    except ValueError as exc:
        report.status, report.reason = "skipped", str(exc)
        return report
    report.ctx = ctx_from_command(cmd)
    sizes = [s for s in SIZES if fits(s, report.ctx)]
    report.skipped_sizes = [s for s in SIZES if s not in sizes]
    if not sizes:
        report.status, report.reason = "skipped", f"context {report.ctx} is too small for any benchmark size"
        return report
    key = cmd[cmd.index("--api-key") + 1] if "--api-key" in cmd else ""
    base = f"http://127.0.0.1:{port}/v1"
    log = tempfile.TemporaryFile()
    progress(f"  loading {name} (context {report.ctx})...")
    t_launch = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    sampler = MemorySampler(proc.pid)
    try:
        sampler.start()
        if not wait_healthy(base, proc):
            report.status, report.reason = "failed", "the model server did not become ready"
            return report
        report.load_s = round(time.monotonic() - t_launch, 1)
        report.first_reply_s = measure(report, base, key, name, sizes, runs, progress)
    except Exception as exc:  # noqa: BLE001
        report.status, report.reason = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        report.peak_memory_mb = round(sampler.finish(), 1) or None
        report.memory_source = sampler.source
        _terminate(proc)
        try:
            log.seek(0)
            report.load_buffers_mb = round(buffers_from_log(log.read().decode("utf-8", "replace")), 1) or None
        except Exception:  # noqa: BLE001, S110
            pass
        log.close()
    return report


def measure(report: ModelReport, base: str, key: str, name: str, sizes: list[int], runs: int, progress=print,
            cancel: CancelToken | None = None, on_step=None) -> float:
    """One warm-up request, then `runs` measured requests per size, against a
    server that is already answering at `base`. Fills report.sizes/warnings and
    returns the warm-up's wall time (on a fresh server: the first reply)."""
    before = thermal_snapshot()
    if on_step:
        on_step(0, len(sizes) * runs)
    t_warm = time.monotonic()
    stream_request(base, key, name, synthetic_prompt(256, f"warmup-{time.time_ns()}"), cancel=cancel)
    warm_s = round(time.monotonic() - t_warm, 1)
    series: list[float] = []
    for size in sizes:
        res = []
        for n in range(runs):
            progress(f"  {name}: ~{size // 1024}K prompt, run {n + 1}/{runs}")
            r = stream_request(base, key, name, synthetic_prompt(size, f"{size}-{n}-{time.time_ns()}"), cancel=cancel)
            res.append(r)
            if on_step:
                on_step(len(series) + 1, len(sizes) * runs)
            series.append(r.gen_tps)
        report.sizes[size] = {
            "runs": len(res),
            "prompt_tokens": int(median([r.prompt_tokens for r in res]) or 0),
            "ttft_s": median([r.ttft_s for r in res]),
            "ttft_range": spread([r.ttft_s for r in res]),
            "prefill_tps": median([r.prefill_tps for r in res]),
            "prefill_range": spread([r.prefill_tps for r in res]),
            "gen_tps": median([r.gen_tps for r in res]),
            "gen_range": spread([r.gen_tps for r in res]),
        }
    report.warnings += throttle_flags(series, before, thermal_snapshot())
    return warm_s


def server_pid(parent: int) -> int:
    """The llama-server under a supervisor-owned guard process, else the process itself."""
    out = _run_text(["pgrep", "-P", str(parent)]).split()
    return int(out[0]) if out and out[0].isdigit() else parent


def bench_running(name: str, base: str, key: str, ctx: int, pid: int | None, runs: int = 2,
                  sizes: tuple[int, ...] = (1024, 8192), progress=lambda m: None,
                  cancel: CancelToken | None = None, on_step=None,
                  load_s: float | None = None, first_reply_s: float | None = None) -> ModelReport:
    """Benchmark the model a LocalCode session already has loaded, over its own
    server. Nothing is launched or stopped; the caller owns the server and
    passes the load timings it recorded when it started it."""
    report = ModelReport(name, ctx=ctx, load_s=load_s, first_reply_s=first_reply_s)
    use = [s for s in sizes if fits(s, ctx)] if ctx else list(sizes)
    report.skipped_sizes = [s for s in sizes if s not in use]
    if not use:
        report.status, report.reason = "skipped", f"context {ctx} is too small for any benchmark size"
        return report
    sampler = MemorySampler(server_pid(pid)) if pid else None
    try:
        if sampler:
            sampler.start()
        measure(report, base, key, name, use, runs, progress, cancel=cancel, on_step=on_step)
    except Cancelled:
        report.status, report.reason = "cancelled", "cancelled"
    except Exception as exc:  # noqa: BLE001
        report.status, report.reason = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        if sampler:
            report.peak_memory_mb = round(sampler.finish(), 1) or None
            report.memory_source = sampler.source
    return report


def bench_via_session(control_port: int, token: str, runs: int, sizes: tuple[int, ...], progress=lambda m: None,
                      poll_s: float = 2.0, timeout_s: float = 3600) -> ModelReport:
    """Ask the running session's supervisor to benchmark its loaded model and wait."""
    from localcode.ui.http import local_urlopen
    from localcode.ui.auth import control_headers
    url = f"http://127.0.0.1:{control_port}/bench"
    hdr = {**control_headers(token), "Content-Type": "application/json"}
    body = json.dumps({"runs": runs, "sizes": list(sizes)}).encode()
    try:
        with local_urlopen(urllib.request.Request(url, data=body, headers=hdr, method="POST"), timeout=10) as r:
            json.load(r)
    except urllib.error.HTTPError as exc:
        try:
            msg = json.loads(exc.read()).get("error", str(exc))
        except (ValueError, OSError):
            msg = str(exc)
        raise RuntimeError(msg) from exc
    end = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < end:
        with local_urlopen(urllib.request.Request(url, headers=control_headers(token)), timeout=10) as r:
            st = json.load(r)
        if st.get("progress") and st["progress"] != last:
            last = st["progress"]
            progress(last)
        if st.get("state") == "done":
            return ModelReport.from_json(st["report"])
        if st.get("state") == "error":
            raise RuntimeError(st.get("error") or "benchmark failed")
        time.sleep(poll_s)
    raise RuntimeError("benchmark timed out")


def _terminate(proc) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001, S110
            pass


# ---------------------------------------------------------------- selection

def downloaded_models() -> dict[str, Path]:
    from localcode.model_files import model_files
    from localcode.models_catalog import model_dir
    from localcode.split_gguf import display_alias
    out: dict[str, Path] = {}
    for fname, path in model_files(model_dir()).items():
        out[display_alias(fname)] = path
    return out


def default_model(files: dict[str, Path]) -> str | None:
    """The model LocalCode recommends for this Mac when downloaded, else the
    most recently modified download."""
    if not files:
        return None
    try:
        from localcode.models_catalog import recommend
        from localcode.runtime import LocalCodeRuntimeGateway
        from localcode.config import load_config
        ram = LocalCodeRuntimeGateway(load_config().runtime)._system_ram_gb()
        stem = Path(recommend(ram).filename).name.removesuffix(".gguf")
        if stem in files:
            return stem
    except Exception:  # noqa: BLE001, S110
        pass
    return max(files, key=lambda k: files[k].stat().st_mtime)


def resolve_names(requested: list[str], files: dict[str, Path]) -> tuple[list[str], list[str]]:
    found, missing = [], []
    for want in requested:
        w = want.strip().removesuffix(".gguf")
        hits = [n for n in files if n == w] or [n for n in files if w.lower() in n.lower()]
        if len(hits) == 1:
            found.append(hits[0])
        elif len(hits) > 1:
            missing.append(f"{want} (ambiguous: {', '.join(sorted(hits)[:4])})")
        else:
            missing.append(want)
    return found, missing


def estimate_seconds(n_models: int, runs: int) -> int:
    """Deliberately pessimistic: 200 tok/s prefill, 15 tok/s generation."""
    per_run = lambda s: s / 200 + MAX_TOKENS / 15   # noqa: E731
    per_model = 60 + sum((runs) * per_run(s) for s in SIZES) + COOLDOWN_S
    return int(n_models * per_model)


# ---------------------------------------------------------------- output

def _fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def _pm(med, rng, nd=1):
    if med is None:
        return "-"
    if not rng or rng[0] is None:
        return f"{med:.{nd}f}"
    return f"{med:.{nd}f} ({rng[0]:.{nd}f}-{rng[1]:.{nd}f})"


def render_table(reports: list[ModelReport], compare: bool) -> str:
    ok = [r for r in reports if r.status == "ok"]
    base = ok[0] if ok else None
    head = ["Model", "Load s", "1st reply s", "Prompt", "TTFT s", "Prefill tok/s", "Gen tok/s", "Peak mem GB~"]
    if compare:
        head += ["TTFT vs 1st", "Prefill vs 1st", "Gen vs 1st"]
    rows = [head]
    for r in reports:
        if r.status != "ok":
            rows.append([r.name, r.status.upper(), r.reason] + [""] * (len(head) - 3))
            continue
        for i, (size, s) in enumerate(r.sizes.items()):
            mem = _fmt(r.peak_memory_mb / 1024, 1) if r.peak_memory_mb else "-"
            load = [_fmt(r.load_s), _fmt(r.first_reply_s)] if i == 0 else ["", ""]
            row = [r.name, *load, f"{size // 1024}K", _pm(s["ttft_s"], s["ttft_range"], 2),
                   _pm(s["prefill_tps"], s["prefill_range"], 0), _pm(s["gen_tps"], s["gen_range"], 1), mem]
            if compare:
                b = base.sizes.get(size) if base else None
                for key, inv in (("ttft_s", True), ("prefill_tps", False), ("gen_tps", False)):
                    if b and b.get(key) and s.get(key):
                        ratio = (b[key] / s[key]) if inv else (s[key] / b[key])
                        row.append("1.00x" if r is base else f"{ratio:.2f}x")
                    else:
                        row.append("-")
            rows.append(row)
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(head))]
    lines = ["  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    notes = ["", "Values are medians with (min-max). TTFT is the time to the first generated token.",
             "Load is from starting the model server until it answers; 1st reply is the first short answer",
             "after that (it includes any one-off GPU setup).",
             "Peak memory is approximate (process footprint, sampled every 0.5 s)."]
    if compare:
        notes.append("Compare columns are speedups relative to the first model; above 1.00x is faster.")
    for r in reports:
        for s in r.skipped_sizes:
            notes.append(f"{r.name}: {s // 1024}K prompt skipped, larger than its context ({r.ctx} tokens).")
        for w in r.warnings:
            notes.append(f"WARNING {r.name}: {w}; results may be understated.")
    return "\n".join(lines + notes)


def results_dir() -> Path:
    from localcode.paths import global_state_dir
    d = global_state_dir() / "bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def machine_label() -> str:
    """Chip and RAM as the machine profile reports them; nothing about the user."""
    try:
        from localcode.performance import detect_machine_profile
        m = detect_machine_profile()
        gpu = m.gpu_summary if m.gpu_summary != "none" else ""
        return f"{gpu}, {m.memory_gb} GB".lstrip(", ")
    except Exception:  # noqa: BLE001
        return ""


def build_json(reports: list[ModelReport], runs: int, notes: list[str]) -> dict:
    from localcode import __version__
    return {"schema": SCHEMA_VERSION, "localcode_version": __version__,
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "runs_per_size": runs, "max_tokens": MAX_TOKENS, "notes": notes,
            "machine": machine_label(),
            "on_battery": any("battery" in n for n in notes),
            "models": [r.to_json() for r in reports]}


def save_results(data: dict) -> Path:
    base = datetime.now().strftime("bench-%Y%m%d-%H%M%S")
    d = results_dir()
    path, n = d / f"{base}.json", 1
    while path.exists():
        n += 1
        path = d / f"{base}-{n}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


# ---------------------------------------------------------------- history

_QUANT = re.compile(r"((?:UD-)?(?:IQ|Q)\d[A-Za-z0-9_]*|BF16|F16|MXFP4\w*|NVFP4)$", re.I)


def split_quant(name: str) -> tuple[str, str]:
    m = _QUANT.search(name)
    if not m:
        return name, ""
    return name[:m.start()].rstrip("-_ ") or name, m.group(1)


def _size_stat(sizes: dict, size: int, key: str):
    s = sizes.get(str(size)) or sizes.get(size) or {}
    return s.get(key)


def load_history() -> list[dict]:
    """Every saved successful model run, newest first. Nothing is merged or deduplicated."""
    rows: list[dict] = []
    for f in sorted(results_dir().glob("bench-*.json")):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for i, m in enumerate(data.get("models") or []):
            if not isinstance(m, dict) or m.get("status", "ok") != "ok" or not m.get("sizes"):
                continue
            name = str(m.get("model", ""))
            display, quant = split_quant(name)
            sizes = m.get("sizes") or {}
            warnings = list(m.get("warnings") or []) + [n for n in (data.get("notes") or []) if n not in (m.get("warnings") or [])]
            rows.append({
                "id": f"{f.name}#{i}", "date": str(data.get("created", "")), "version": str(data.get("localcode_version", "")),
                "model": name, "display": display, "quant": quant, "machine": str(data.get("machine", "")),
                "context": m.get("context"),
                "ttft_s": _size_stat(sizes, 1024, "ttft_s"),
                "prefill_1k": _size_stat(sizes, 1024, "prefill_tps"), "prefill_8k": _size_stat(sizes, 8192, "prefill_tps"),
                "gen_1k": _size_stat(sizes, 1024, "gen_tps"), "gen_8k": _size_stat(sizes, 8192, "gen_tps"),
                "peak_memory_mb": m.get("peak_memory_mb"), "load_s": m.get("load_s"),
                "first_reply_s": m.get("first_reply_s"), "warnings": warnings,
                "throttled": any("throttl" in w.lower() or "slowed" in w.lower() for w in warnings),
                "on_battery": bool(data.get("on_battery")) or any("battery" in w for w in warnings),
            })
    rows.sort(key=lambda r: (r["date"], r["id"]), reverse=True)
    return rows


def delete_run(run_id: str) -> bool:
    fname, _, idx = run_id.partition("#")
    if not re.fullmatch(r"bench-[\w-]+\.json", fname) or not idx.isdigit():
        return False
    path = results_dir() / fname
    try:
        data = json.loads(path.read_text())
        models = data["models"]
        models.pop(int(idx))
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False
    if any(isinstance(m, dict) and m.get("status", "ok") == "ok" for m in models):
        path.write_text(json.dumps(data, indent=2))
    else:
        path.unlink()
    return True


def _pct(cur, prev, lower_is_better: bool = False):
    if not cur or not prev:
        return None
    ratio = (prev / cur) if lower_is_better else (cur / prev)
    return round((ratio - 1) * 100)


def mismatch_reasons(cur: dict, other: dict) -> list[str]:
    out = []
    if cur.get("version") != other.get("version"):
        out.append(f"LocalCode {other.get('version') or '?'}")
    for label, row in (("throttled", other), ("on battery", other)):
        key = "throttled" if label == "throttled" else "on_battery"
        if row.get(key) and not cur.get(key):
            out.append(label)
    if cur.get("on_battery") and not other.get("on_battery"):
        out.append("this run on battery")
    if cur.get("throttled") and not other.get("throttled"):
        out.append("this run throttled")
    if cur.get("machine") != other.get("machine"):
        out.append("other machine")
    return out


def compare_with_history(cur: dict, history: list[dict]) -> list[dict]:
    """This model's previous run plus the best previous run of every other model, with
    deltas against `cur` (positive = the current run is faster) and why a row may not compare."""
    prior = [h for h in history if h["id"] != cur.get("id")]
    picks: list[dict] = []
    same = [h for h in prior if h["model"] == cur["model"]]
    if same:
        picks.append(dict(same[0], relation="previous run"))
    best: dict[str, dict] = {}
    for h in prior:
        if h["model"] != cur["model"] and (h["model"] not in best or (h.get("gen_1k") or 0) > (best[h["model"]].get("gen_1k") or 0)):
            best[h["model"]] = h
    picks += [dict(h, relation="best run") for h in sorted(best.values(), key=lambda h: -(h.get("gen_1k") or 0))]
    for p in picks:
        p["delta"] = {"gen_pct": _pct(cur.get("gen_1k"), p.get("gen_1k")),
                      "prefill_pct": _pct(cur.get("prefill_1k"), p.get("prefill_1k")),
                      "ttft_pct": _pct(cur.get("ttft_s"), p.get("ttft_s"), True),
                      "load_pct": _pct(cur.get("load_s"), p.get("load_s"), True)}
        p["mismatch"] = mismatch_reasons(cur, p)
    return picks


def current_row(report: ModelReport, data: dict, run_id: str = "") -> dict:
    """The history-shaped row for a just-finished run (so it can be compared)."""
    name = report.name
    display, quant = split_quant(name)
    sizes = {str(k): v for k, v in report.sizes.items()}
    warnings = list(report.warnings) + list(data.get("notes") or [])
    return {"id": run_id, "date": data.get("created", ""), "version": data.get("localcode_version", ""), "model": name,
            "display": display, "quant": quant, "machine": data.get("machine", ""), "context": report.ctx,
            "ttft_s": _size_stat(sizes, 1024, "ttft_s"),
            "prefill_1k": _size_stat(sizes, 1024, "prefill_tps"), "prefill_8k": _size_stat(sizes, 8192, "prefill_tps"),
            "gen_1k": _size_stat(sizes, 1024, "gen_tps"), "gen_8k": _size_stat(sizes, 8192, "gen_tps"),
            "peak_memory_mb": report.peak_memory_mb, "load_s": report.load_s,
            "first_reply_s": report.first_reply_s, "warnings": warnings,
            "throttled": any("throttl" in w.lower() or "slowed" in w.lower() for w in warnings),
            "on_battery": bool(data.get("on_battery"))}


def render_saved_compare(reports: list[ModelReport]) -> str:
    """For --compare: the newest saved run of every model that was not measured now, with its date."""
    now = {r.name for r in reports}
    seen: dict[str, dict] = {}
    for h in load_history():
        if h["model"] not in now and h["model"] not in seen:
            seen[h["model"]] = h
    if not seen:
        return ""
    base = next((r for r in reports if r.status == "ok"), None)
    cur = current_row(base, {}) if base else None
    lines = ["", "Saved results for other models (not measured now):"]
    for h in seen.values():
        line = f"  {h['model']} ({h['date'][:10]}, LocalCode {h['version']}): gen {_f(h['gen_1k'])} tok/s at 1K, prefill {_f(h['prefill_1k'], 0)} tok/s, first token {_f(h['ttft_s'], 2)} s"
        if cur and cur["gen_1k"] and h["gen_1k"]:
            line += f" ({_pct(h['gen_1k'], cur['gen_1k']):+d}% gen vs {cur['model']})"
        lines.append(line)
    return "\n".join(lines)


def _f(v, nd=1) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def render_history(rows: list[dict]) -> str:
    if not rows:
        return "No saved benchmark results yet. Run /bench in LocalCode or `localcode bench`."
    head = ["Date", "LocalCode", "Model", "Machine", "TTFT s", "Prefill 1K", "Prefill 8K", "Gen 1K", "Gen 8K", "Mem GB", "Notes"]
    table = [head]
    for r in rows:
        notes = [n for n, on in (("throttled", r["throttled"]), ("battery", r["on_battery"])) if on]
        table.append([r["date"][:16].replace("T", " "), r["version"], r["model"], r["machine"], _f(r["ttft_s"], 2),
                      _f(r["prefill_1k"], 0), _f(r["prefill_8k"], 0), _f(r["gen_1k"]), _f(r["gen_8k"]),
                      _f(r["peak_memory_mb"] / 1024 if r["peak_memory_mb"] else None), ", ".join(notes)])
    widths = [max(len(str(row[i])) for row in table) for i in range(len(head))]
    lines = ["  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)).rstrip() for row in table]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def _bench_session(args, session, say, input_fn) -> int:
    """`localcode bench` while LocalCode is running: measure its loaded model over its own server."""
    from localcode.ui import run_dir
    from localcode.ui.auth import read_auth_file
    ctrl, status = session
    current = status.get("current")
    auth = read_auth_file(run_dir())
    if not auth:
        print("localcode bench: cannot authenticate to the running session.", file=sys.stderr)
        return 1
    print(f"Benchmarking {current} through the running LocalCode session (about 1 warm-up + {args.runs} runs at "
          "~1K, 8K and 32K tokens). It is busy meanwhile; wait to chat.", file=sys.stderr)
    warn = power_warning()
    if warn:
        print(f"Warning: {warn}.", file=sys.stderr)
    if not args.yes:
        try:
            answer = input_fn("Start now? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled; nothing was started.", file=sys.stderr)
            return 0
    try:
        report = bench_via_session(ctrl, auth[0], args.runs, SIZES, progress=say)
    except Exception as exc:  # noqa: BLE001
        print(f"localcode bench: {exc}", file=sys.stderr)
        return 1
    data = build_json([report], args.runs, [warn] if warn else [])
    saved = save_results(data)
    if args.json == "-":
        print(json.dumps(data, indent=2))
    else:
        print(render_table([report], args.compare))
        if args.compare:
            print(render_saved_compare([report]))
        if args.json:
            Path(args.json).expanduser().write_text(json.dumps(data, indent=2))
    return 0 if report.status == "ok" else 1


def main(argv: list[str] | None = None, *, input_fn=input, progress=None) -> int:
    parser = argparse.ArgumentParser(
        prog="localcode bench",
        description="Measure load time, time to first token, prompt speed, generation speed and memory of downloaded models.")
    parser.add_argument("--models", help="Comma separated model names (as shown in /models); default: the model LocalCode would pick")
    parser.add_argument("--all-downloaded", action="store_true", help="Benchmark every downloaded model")
    parser.add_argument("--compare", action="store_true", help="Add columns relative to the first model")
    parser.add_argument("--json", nargs="?", const="-", metavar="PATH", help="Print results as JSON, or write them to PATH")
    parser.add_argument("--history", action="store_true", help="Show saved benchmark results, newest first")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per prompt size (default 3)")
    parser.add_argument("--yes", "-y", action="store_true", help="Do not ask before starting")
    args = parser.parse_args(argv)
    say = progress or (lambda m: print(m, file=sys.stderr))
    if args.history:
        print(render_history(load_history()))
        return 0
    if args.runs < 1:
        print("localcode bench: --runs must be at least 1", file=sys.stderr)
        return 2

    session = None
    try:
        from localcode.ui.ports import find_running
        session = find_running()
    except Exception:  # noqa: BLE001, S110
        pass
    if session is None:
        reason = serving_reason()
        if reason:
            print(f"localcode bench: {reason}. Close LocalCode (and any other llama-server) first so the "
                  "measurements are not shared with another model.", file=sys.stderr)
            return 1

    files = downloaded_models()
    reports: list[ModelReport] = []
    if session is not None:
        current = session[1].get("current") or ""
        wanted = [current] if current else []
        if args.models:
            names, missing = resolve_names(args.models.split(","), files)
            wanted = names + missing
        if args.all_downloaded or not current or any(w != current for w in wanted):
            print("localcode bench: a LocalCode session is serving "
                  f"{current or 'no model'}, so only that model can be measured. Close LocalCode or run /bench "
                  "inside it; then run `localcode bench` again to compare other models.", file=sys.stderr)
            return 1
        return _bench_session(args, session, say, input_fn)
    if args.models:
        names, missing = resolve_names(args.models.split(","), files)
    elif args.all_downloaded:
        names, missing = sorted(files), []
    else:
        d = default_model(files)
        names, missing = ([d] if d else []), []
    for m in missing:
        reports.append(ModelReport(m, status="not downloaded", reason="not downloaded (bench never downloads)"))
    if not names:
        for r in reports:
            print(f"{r.name}: {r.reason}", file=sys.stderr)
        print("localcode bench: no downloaded model to benchmark. Download one from /models in LocalCode.", file=sys.stderr)
        return 1

    minutes = max(1, round(estimate_seconds(len(names), args.runs) / 60))
    print(f"Models to benchmark ({len(names)}):", file=sys.stderr)
    for n in names:
        print(f"  {n}", file=sys.stderr)
    for r in reports:
        print(f"  {r.name}: not downloaded", file=sys.stderr)
    print(f"Estimated time: up to about {minutes} min (1 warm-up + {args.runs} runs at ~1K, 8K and 32K tokens per model).",
          file=sys.stderr)
    warn = power_warning()
    if warn:
        print(f"Warning: {warn}.", file=sys.stderr)
    if not args.yes:
        try:
            answer = input_fn("Start now? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled; nothing was started.", file=sys.stderr)
            return 0

    # A per-run key so the throwaway server is not open to other local apps.
    from localcode.ui.auth import KEY_ENV, new_secret
    os.environ[KEY_ENV] = new_secret()
    notes = [warn] if warn else []
    for i, n in enumerate(names):
        say(f"[{i + 1}/{len(names)}] {n}")
        reports.insert(i, bench_model(files[n], args.runs, progress=say))
        if i < len(names) - 1:
            time.sleep(COOLDOWN_S)
    # keep order: benchmarked first, then not-downloaded
    data = build_json(reports, args.runs, notes)
    saved = save_results(data)
    if args.json == "-":
        print(json.dumps(data, indent=2))
    else:
        print(render_table(reports, args.compare))
        if args.compare:
            print(render_saved_compare(reports))
        if warn:
            print(f"\nNote: {warn}.")
        if args.json:
            Path(args.json).expanduser().write_text(json.dumps(data, indent=2))
    try:
        shown = "~/" + str(saved.relative_to(Path.home()))
    except ValueError:
        shown = saved.name
    print(f"\nSaved: {shown}", file=sys.stderr)
    return 0 if any(r.status == "ok" for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
