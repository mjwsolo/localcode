"""Local-only measurements. Missing telemetry stays unknown, never zero.

No prompts, tool arguments, output text or credentials enter the summary.
Repeated calls are observations, not proof that a call was unnecessary.
"""
import hashlib
import json
import re


def _median(xs):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    return None if not xs else (xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2)


def summarize(events, server_log):
    calls, seen, repeats, failures, durations = {}, set(), 0, 0, []
    tokens = {k: 0 for k in ('input', 'output', 'cache_read', 'cache_write')}
    usage_seen = False
    seen_ids = set()
    for event in events:
        part = event.get('part') or {}
        if event.get('type') == 'step_finish':
            usage = part.get('tokens')
            if isinstance(usage, dict):
                usage_seen = True
                for key in ('input', 'output'):
                    tokens[key] += usage.get(key, 0) or 0
                cache = usage.get('cache') or {}
                tokens['cache_read'] += cache.get('read', 0) or 0
                tokens['cache_write'] += cache.get('write', 0) or 0
        if event.get('type') != 'tool_use':
            continue
        state = part.get('state') or {}
        if state.get('status') not in ('completed', 'error'):
            continue
        identity = part.get('callID') or part.get('id')
        if identity and identity in seen_ids:
            continue
        if identity:
            seen_ids.add(identity)
        tool = part.get('tool', 'unknown')
        calls[tool] = calls.get(tool, 0) + 1
        # Compare in memory; never serialize potentially sensitive arguments.
        signature = hashlib.sha256(json.dumps([tool, state.get('input')], sort_keys=True).encode()).digest()
        repeats += signature in seen
        seen.add(signature)
        failures += state.get('status') == 'error'
        timing = state.get('time') or {}
        if isinstance(timing.get('start'), (int, float)) and isinstance(timing.get('end'), (int, float)):
            durations.append(max(0, timing['end'] - timing['start']) / 1000)
    requests = {}
    pattern = r'task\s+(\d+)\s*\|\s*(prompt eval time|eval time)\s*=\s*([\d.]+) ms /\s*(\d+) tokens'
    for task, kind, ms, count in re.findall(pattern, server_log):
        prefix = 'prefill' if kind.startswith('prompt') else 'decode'
        requests.setdefault(task, {})[prefix + '_ms'] = float(ms)
        requests[task][prefix + '_tokens'] = int(count)
    return {
        'usage_tokens': tokens if usage_seen else None,
        'tool_calls_by_name': calls, 'repeated_tool_calls': repeats,
        'tool_errors': failures, 'tool_duration_s': sum(durations) if durations else None,
        'server_requests': list(requests.values()),
        # Medians over this trial's requests, from the server's own timing lines.
        'decode_tps': _median([r['decode_tokens'] / (r['decode_ms'] / 1000) for r in requests.values()
                               if r.get('decode_ms') and r.get('decode_tokens', 0) >= 16]),
        'prefill_tps': _median([r['prefill_tokens'] / (r['prefill_ms'] / 1000) for r in requests.values()
                                if r.get('prefill_ms') and r.get('prefill_tokens', 0) >= 64]),
        # Timing lines do not include queue/transport time: do not call this TTFT.
        'time_to_first_token_s': None,
    }


def tree_rss(text, root):
    """Sum RSS for an owned supervisor and its descendants, not unrelated models."""
    rows = []
    for line in text.splitlines():
        try:
            pid, parent, rss = map(int, line.split())
            rows.append((pid, parent, rss))
        except ValueError:
            continue
    owned = {root}
    while True:
        more = {pid for pid, parent, _ in rows if parent in owned}
        if more <= owned:
            break
        owned |= more
    values = [rss * 1024 for pid, _, rss in rows if pid in owned]
    return sum(values) if values else None


class ResourceMeter:
    """Sample RSS and system-wide swap locally; unavailable metrics remain null."""
    def __init__(self, pid):
        import threading
        self.pid = pid
        self.peak = None
        self.swap_start = self.swap()
        self.swap_peak = self.swap_start
        self.limit_min, self.batt_start, self.on_battery = self.power()
        self.batt_end = self.batt_start
        self.ticks = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    @staticmethod
    def power():
        """(cpu_speed_limit_pct, battery_pct, on_battery) from pmset; None when unknown."""
        import subprocess
        limit = pct = None; on_batt = None
        try:
            raw = subprocess.check_output(['pmset', '-g', 'therm'], text=True, stderr=subprocess.DEVNULL, timeout=2)
            m = re.search(r'CPU_Speed_Limit\s*=\s*(\d+)', raw)
            limit = int(m[1]) if m else None  # no report is unknown, not proof of no throttling
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            raw = subprocess.check_output(['pmset', '-g', 'batt'], text=True, stderr=subprocess.DEVNULL, timeout=2)
            m = re.search(r'(\d+)%', raw)
            pct = int(m[1]) if m else None
            on_batt = "'Battery Power'" in raw
        except (OSError, subprocess.SubprocessError):
            pass
        return limit, pct, on_batt

    @staticmethod
    def swap():
        import subprocess
        try:
            raw = subprocess.check_output(['sysctl', '-n', 'vm.swapusage'], text=True, stderr=subprocess.DEVNULL, timeout=2)
            match = re.search(r'used\s*=\s*([\d.]+)([MG])', raw)
            return int(float(match[1]) * (1024**2 if match[2] == 'M' else 1024**3)) if match else None
        except (OSError, subprocess.SubprocessError):
            return None

    def sample(self):
        import subprocess
        while not self.stop.is_set():
            try:
                raw = subprocess.check_output(['ps','-axo','pid=,ppid=,rss='], text=True, timeout=2)
                value = tree_rss(raw, self.pid)
                if value is not None:
                    self.peak = max(self.peak or 0, value)
            except (OSError, subprocess.SubprocessError):
                pass
            swap = self.swap()
            if swap is not None:
                self.swap_peak = max(self.swap_peak or 0, swap)
            self.ticks += 1
            if self.ticks % 20 == 0:          # pmset every ~10 s; it is not free
                limit, pct, on_batt = self.power()
                if limit is not None:
                    self.limit_min = limit if self.limit_min is None else min(self.limit_min, limit)
                if pct is not None:
                    self.batt_end = pct
                if on_batt:
                    self.on_battery = True
            self.stop.wait(0.5)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=5)
        return {'peak_supervisor_tree_rss_bytes': self.peak,
                'system_swap_growth_bytes': None if self.swap_start is None or self.swap_peak is None else max(0,self.swap_peak-self.swap_start),
                # Thermal: the lowest CPU speed limit pmset reported during the trial
                # (100 = never throttled). Battery: percentage points lost, only
                # meaningful when on_battery is true.
                'cpu_speed_limit_min_pct': self.limit_min,
                'battery_drop_pct': None if self.batt_start is None or self.batt_end is None else max(0, self.batt_start - self.batt_end),
                'on_battery': self.on_battery}
