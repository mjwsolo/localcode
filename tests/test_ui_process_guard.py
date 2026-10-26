"""Real process-death regressions; no sockets, model weights or GPU required."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from localcode.ui import process_guard
from localcode.ui.supervisor import Supervisor


def alive(pid):
    if sys.platform == 'win32':
        import psutil
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, 'process cleanup did not complete'
        time.sleep(0.05)


def model_command(pid_file, *, ignore_term=False):
    return [sys.executable, '-c',
            'import os,sys,time,signal,pathlib; '
            + ('signal.signal(signal.SIGTERM,signal.SIG_IGN); ' if ignore_term else '')
            + 'pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)', str(pid_file)]


@pytest.mark.parametrize('shutdown', [
    'pipe',
    pytest.param('signal', marks=pytest.mark.skipif(
        sys.platform == 'win32', reason='no SIGTERM on Windows; a Job Object replaces it')),
])
def test_guard_reaps_model_when_owner_closes_or_terminates(tmp_path, shutdown):
    pid_file = tmp_path / 'model.pid'
    with (tmp_path / 'log').open('wb') as log:
        guard = process_guard.spawn_owned(model_command(pid_file), stdout=log)
        try:
            wait_for(pid_file.exists)
            pid = int(pid_file.read_text())
            assert alive(pid)
            if shutdown == 'pipe':
                guard.stdin.close()
            else:
                guard.terminate()
            assert guard.wait(timeout=8) == 0
            assert not alive(pid), 'model survived guard shutdown'
        finally:
            if guard.poll() is None:
                guard.terminate()
                guard.wait(timeout=20)
            if guard.stdin and not guard.stdin.closed:
                guard.stdin.close()


@pytest.mark.parametrize('ignore_term', [False, True])
def test_force_killed_supervisor_cannot_orphan_model(tmp_path, ignore_term):
    # A real separate owner gets SIGKILL: no finally block or atexit can run.
    pid_file = tmp_path / 'model.pid'
    guard_pid_file = tmp_path / 'guard.pid'
    script = tmp_path / 'owner.py'
    script.write_text(
        'import importlib.util,pathlib,sys,time\n'
        f'spec=importlib.util.spec_from_file_location("guard", {str(Path(process_guard.__file__))!r})\n'
        'module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n'
        f'child=module.spawn_owned({model_command(pid_file, ignore_term=ignore_term)!r}, stdout=sys.stdout)\n'
        f'pathlib.Path({str(guard_pid_file)!r}).write_text(str(child.pid))\n'
        'time.sleep(60)\n'
    )
    with (tmp_path / 'log').open('wb') as log:
        owner = subprocess.Popen([sys.executable, str(script)], stdout=log, stderr=log)
        model = guard = None
        try:
            wait_for(lambda: pid_file.exists() and guard_pid_file.exists())
            model, guard = int(pid_file.read_text()), int(guard_pid_file.read_text())
            owner.kill()
            owner.wait(timeout=5)
            wait_for(lambda: not alive(model), seconds=18)
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=5)
            for pid in (model, guard):
                if pid and alive(pid):
                    os.kill(pid, signal.SIGKILL)


def test_shutdown_prevents_background_worker_from_spawning_a_new_model(tmp_path):
    supervisor = Supervisor.__new__(Supervisor)
    supervisor._process_lock = threading.RLock()
    supervisor._closing = threading.Event()
    supervisor.cancel = threading.Event()
    supervisor._warm_stop = threading.Event()
    supervisor._warm_replay = None
    supervisor.proc = None
    supervisor.models_dir = tmp_path
    supervisor.port = 8123
    supervisor.shutdown()
    # start must reject before any process spawn or model replacement.
    assert supervisor.start('missing-model') is False
    assert supervisor.proc is None
    assert supervisor.cancel.is_set()


def test_unconfirmed_stop_preserves_a_real_owned_process(tmp_path):
    pid_file = tmp_path / 'model.pid'
    with (tmp_path / 'log').open('wb') as log:
        guard = process_guard.spawn_owned(model_command(pid_file), stdout=log)
        try:
            wait_for(pid_file.exists)
            pid = int(pid_file.read_text())
            supervisor = Supervisor.__new__(Supervisor)
            supervisor.lock = threading.Lock()
            supervisor._process_lock = threading.RLock()
            supervisor._warm_stop = threading.Event()
            supervisor._warm_replay = None
            supervisor.state = {'state': 'ready'}
            supervisor.current = 'test-model'
            supervisor.proc = guard
            supervisor.progress = lambda: {'phase': 'unknown'}
            reply = supervisor.server_action('stop')
            assert reply['force_required'] is True
            time.sleep(0.2)
            # The Cancel/Escape UI paths send no second request. The first
            # request must not queue a delayed stop behind the confirmation.
            assert not supervisor.lock.locked()
            assert guard.poll() is None
            assert alive(pid)
            assert supervisor.state == {'state': 'ready'}
            assert supervisor.server_action('stop', force=True)['ok']
            wait_for(lambda: supervisor.state['state'] == 'stopped')
            assert not alive(pid)
            assert guard.wait(timeout=5) == 0
        finally:
            if guard.poll() is None:
                guard.stdin.close()
                guard.wait(timeout=20)
