"""Local-only measurements. Missing telemetry stays unknown, never zero.

No prompts, tool arguments, output text or credentials enter the summary.
Repeated calls are observations, not proof that a call was unnecessary.
"""
import hashlib
import json
import re


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
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

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
            self.stop.wait(0.5)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=5)
        return {'peak_supervisor_tree_rss_bytes': self.peak,
                'system_swap_growth_bytes': None if self.swap_start is None or self.swap_peak is None else max(0,self.swap_peak-self.swap_start)}
