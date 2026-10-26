"""Own a model child and reap it when the supervisor's pipe closes.

The guard survives supervisor SIGKILL. Only the owning supervisor holds the
pipe's write end; the model never inherits it. No PID lookup or process scan
is needed, so an unrelated server cannot be stopped accidentally.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

# Run as a standalone script, so it cannot import localcode.compat.
IS_WINDOWS = sys.platform == 'win32'
STOP_SIGNALS = tuple(
    s for s in (signal.SIGINT, signal.SIGTERM, getattr(signal, 'SIGHUP', None),
                getattr(signal, 'SIGBREAK', None)) if s is not None
)


def new_group_popen_kwargs() -> dict:
    if IS_WINDOWS:
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    return {'start_new_session': True}


def spawn_owned(command: list[str], *, stdout, lease_fd: int | None = None):
    args = [sys.executable, str(Path(__file__).resolve())]
    if lease_fd is not None:
        args += ['--lease-fd', str(lease_fd)]
    # Windows cannot hand an fd to a child (no pass_fds), so the lease stays with
    # the supervisor there; a Job Object will replace it in a later phase.
    inherit = {} if IS_WINDOWS else {'pass_fds': (lease_fd,) if lease_fd is not None else ()}
    return subprocess.Popen(
        [*args, '--', *command], stdin=subprocess.PIPE, stdout=stdout,
        stderr=subprocess.STDOUT, **new_group_popen_kwargs(), **inherit,
    )


def main(argv: list[str]) -> int:
    split = argv.index('--')
    lease_fd = int(argv[1]) if argv[:1] == ['--lease-fd'] else None
    command = argv[split + 1:]
    if not command:
        raise ValueError('model command is required')
    stop = threading.Event()
    for signum in STOP_SIGNALS:
        signal.signal(signum, lambda *_: stop.set())

    def watch_owner():
        try:
            while os.read(sys.stdin.fileno(), 1):
                pass
        finally:
            stop.set()

    threading.Thread(target=watch_owner, daemon=True).start()
    if stop.is_set():
        return 0
    inherit = {} if IS_WINDOWS else {'pass_fds': (lease_fd,) if lease_fd is not None else ()}
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, **inherit)
    sys.stderr.write(f'process guard: model pid {child.pid}\n')
    sys.stderr.flush()
    try:
        while child.poll() is None and not stop.wait(0.1):
            pass
        if child.poll() is not None:
            return child.wait()
        return 0
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        else:
            child.wait()


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
