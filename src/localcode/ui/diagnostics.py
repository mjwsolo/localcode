"""Local performance observations. Never include prompts, paths or credentials."""
from __future__ import annotations

import re
import subprocess
from statistics import median


def timing_summary(samples) -> dict:
    def middle(values):
        return round(median(values), 2) if values else None
    recent = samples[-20:]
    reuse = []
    for sample in recent:
        prompt = sample.n_tokens - sample.tg_n
        if prompt > 0 and sample.pp_tps is not None:
            reuse.append(max(0.0, min(1.0, 1 - sample.pp_n / prompt)))
    return {
        'requests': len(recent),
        'decode_tps': middle([s.tg_tps for s in recent if s.tg_tps is not None and s.tg_n >= 16]),
        'prefill_tps': middle([s.pp_tps for s in recent if s.pp_tps is not None and s.pp_n >= 256]),
        'prompt_reuse_estimate_pct': middle([100 * n for n in reuse]),
    }


def machine_observations() -> dict:
    def read(args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=2)
        except (OSError, subprocess.SubprocessError):
            return ''
    processes = read(['ps', '-axo', 'comm='])
    servers = sum(line.strip().rsplit('/', 1)[-1].startswith('llama-server') for line in processes.splitlines()) if processes else None
    swap = re.search(r'used\s*=\s*([\d.]+)([MG])', read(['sysctl', '-n', 'vm.swapusage']))
    battery = read(['pmset', '-g', 'batt'])
    therm = re.search(r'CPU_Speed_Limit\s*=\s*(\d+)', read(['pmset', '-g', 'therm']))
    return {
        'llama_server_processes': servers,
        'swap_used_bytes': int(float(swap[1]) * (1024**2 if swap[2] == 'M' else 1024**3)) if swap else None,
        'on_battery': ("'Battery Power'" in battery) if battery else None,
        'cpu_speed_limit_pct': int(therm[1]) if therm else None,
        'warnings': ['Multiple llama-server processes may compete for memory and GPU time.'] if servers and servers > 1 else [],
    }
