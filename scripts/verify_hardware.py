#!/usr/bin/env python3
"""Local-only physical Mac gate: real supervisor, model switches, long prompts,
process-crash recovery, peak RSS and system swap. Never allocates fake pressure,
downloads models, registers a runner or uploads results.

Run on each representative Mac, with other heavy applications closed:
python scripts/verify_hardware.py --models gemma-12b --cycles 2 --prompt-tokens 2048
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import subprocess
import tempfile
import threading
import time

from verify_models import ROOT, Server, environment, verify
from localcode import diagnostics, models_catalog as catalog
from localcode.ui import auth
from localcode.ui.server_cmd import server_command
from localcode.ui import supervisor


def swap_bytes() -> int | None:
    value = diagnostics.sysctl('vm.swapusage')
    match = re.search(r'used\s*=\s*([\d.]+)([MG])', value)
    return int(float(match[1]) * (1024**2 if match[2] == 'M' else 1024**3)) if match else None


class Meter:
    def __init__(self, owner):
        self.owner = owner
        self.peak_rss_bytes = 0
        self.initial_swap = swap_bytes()
        self.peak_swap = self.initial_swap
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            proc = getattr(getattr(self.owner, "sup", None), "proc", None)
            if proc is not None and proc.poll() is None:
                try:
                    rss = int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(proc.pid)],
                              text=True, stderr=subprocess.DEVNULL, timeout=2).strip()) * 1024
                    self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            swap = swap_bytes()
            if swap is not None:
                self.peak_swap = max(self.peak_swap or 0, swap)
            self.stop.wait(0.25)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=5)
        return {'peak_server_rss_bytes': self.peak_rss_bytes,
                'system_swap_growth_bytes': None if self.initial_swap is None else
                max(0, (self.peak_swap or 0) - self.initial_swap)}


class ShippingServer(Server):
    """Reuse wire checks while lifecycle belongs to the actual supervisor."""
    def __enter__(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.base = f'http://127.0.0.1:{self.port}'
        self.command = server_command(str(self.choice.local_path), self.port, self.choice.key)
        self.sup = supervisor.Supervisor(str(ROOT / 'src/localcode/bin/llama-server'),
                                        self.port, self.choice.local_path.parent, 2048)
        self.alias = self.choice.local_path.stem
        started = time.monotonic()
        try:
            assert self.sup.start(self.alias, wait_s=420), 'model load failed'
            self.proc = self.sup.proc
            self.sup.state = {'state': 'ready', 'model': self.alias}
            self.load_seconds = time.monotonic() - started
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *exc):
        if hasattr(self, 'sup'):
            self.sup.stop()
            self.sup.log.close()

    def switch(self, choice):
        previous = self.sup.proc
        self.choice = choice
        self.alias = choice.local_path.stem
        self.sup.models_dir = choice.local_path.parent
        assert self.sup.start(self.alias, wait_s=420), 'model switch failed'
        self.proc = self.sup.proc
        self.sup.state = {'state': 'ready', 'model': self.alias}
        assert previous.poll() is not None, 'previous model process survived switch'
        assert self.request('/health')['status'] == 'ok'

    def recover(self):
        self.proc.kill()
        self.proc.wait(timeout=15)
        assert self.sup.server_status()['state'] != 'ready', 'crash incorrectly reported ready'
        assert self.sup.start(self.alias, wait_s=420), 'supervisor failed to reload after crash'
        self.proc = self.sup.proc
        self.sup.state = {'state': 'ready', 'model': self.alias}
        assert self.request('/health')['status'] == 'ok'


def long_turn(server, tokens):
    # Bound the stress prompt by the actual per-slot context, not total slots.
    assert tokens + 1024 <= server.sup.ctx, 'requested stress prompt exceeds per-slot context'
    text = 'This is synthetic hardware validation data. ' * tokens
    encoded = server.request('/tokenize', {'content': text})['tokens'][:tokens]
    text = server.request('/detokenize', {'tokens': encoded})['content']
    start = time.monotonic()
    response = server.request('/v1/chat/completions', {
        'messages': [{'role': 'user', 'content': text + '\nReply with exactly OK.'}],
        'temperature': 0, 'max_tokens': 512}, timeout=300)
    assert response['choices'][0]['message'].get('content', '').strip(), 'empty long-context response'
    return {'prompt_tokens_requested': tokens, 'elapsed_seconds': time.monotonic() - start,
            'usage': response.get('usage', {})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', required=True, help='downloaded catalog keys that fit this Mac')
    p.add_argument('--cycles', type=int, default=2)
    p.add_argument('--prompt-tokens', type=int, default=2048)
    p.add_argument('--max-rss-gb', type=float)
    p.add_argument('--max-swap-growth-gb', type=float)
    p.add_argument('--receipt', type=Path, default=ROOT / '.localcode-gate/hardware.json')
    a = p.parse_args()
    if diagnostics.platform_problem():
        p.error(diagnostics.platform_problem())
    if a.cycles < 1 or a.prompt_tokens < 1:
        p.error('cycles and prompt-tokens must be positive')
    if any(v is not None and v < 0 for v in (a.max_rss_gb, a.max_swap_growth_gb)):
        p.error('memory thresholds cannot be negative')
    choices = {c.key: c for c in catalog.CHOICES}
    for key in a.models:
        if key not in choices or not choices[key].local_path.is_file():
            p.error(f'{key}: model is not downloaded; this command never downloads models')
    result = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
              'chip': diagnostics.sysctl('machdep.cpu.brand_string'), 'macos': platform.mac_ver()[0],
              'ram_bytes': diagnostics.sysctl('hw.memsize'), 'runs': [], 'success': False}
    try:
        with tempfile.TemporaryDirectory(prefix='localcode-hardware-') as directory:
            work = Path(directory)
            old_here = supervisor.HERE
            supervisor.HERE = work
            try:
                with environment({auth.KEY_ENV: secrets.token_urlsafe(32), 'LOCALCODE_PROMPT_WARMUP': '0', 'LOCALCODE_INTERNAL_THINKING_MODE': 'off',
                                  'LOCALCODE_AGENT_RUN_DIR': str(work / 'run')}):
                    for cycle in range(a.cycles):
                        for key in a.models:
                            print(f'VERIFY hardware {key} cycle {cycle+1}', flush=True)
                            # Meter starts before load to include peak model allocation.
                            srv = ShippingServer(choices[key], work)
                            srv.key = os.environ[auth.KEY_ENV]
                            meter = Meter(srv)
                            # Track supervisor's child during initial load as well.
                            meter.thread.start()
                            try:
                                with srv:
                                    checks = verify(srv, runtime=True)
                                    stress = long_turn(srv, a.prompt_tokens)
                                    next_key = a.models[(a.models.index(key) + 1) % len(a.models)]
                                    srv.switch(choices[next_key])
                                    checks.append('supervisor-model-switch')
                                    srv.recover()
                                    checks.append('supervisor-crash-recovery')
                                    assert srv.request('/v1/chat/completions', {'messages': [{'role':'user','content':'Reply OK.'}], 'max_tokens':512})['choices'][0]['message'].get('content')
                            finally:
                                metrics = meter.finish()
                            result['runs'].append({'model': key, 'cycle': cycle+1, 'checks': checks,
                                                  'load_seconds': srv.load_seconds, **stress, **metrics})
                            if a.max_rss_gb is not None:
                                assert metrics['peak_server_rss_bytes'] <= a.max_rss_gb * 1024**3, 'RSS threshold exceeded'
                            if a.max_swap_growth_gb is not None:
                                assert metrics['system_swap_growth_bytes'] is not None, 'swap measurement unavailable'
                                assert metrics['system_swap_growth_bytes'] <= a.max_swap_growth_gb * 1024**3, 'swap threshold exceeded'
            finally:
                supervisor.HERE = old_here
        result['success'] = True
    except Exception as exc:
        result['error'] = str(exc)
        print(f'FAIL {exc}', flush=True)
    finally:
        a.receipt.parent.mkdir(parents=True, exist_ok=True)
        a.receipt.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
