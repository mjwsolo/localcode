"""Exercise real process ownership without loading model weights."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    control = port()
    worker = root / 'worker.py'
    worker.write_text('''import os, pathlib, subprocess, sys, time
from localcode.ui import supervisor as module
module.Path.home = staticmethod(lambda: pathlib.Path(os.environ['QA_ROOT']))
def start(self, alias, wait_s=240):
    self.proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                 start_new_session=True, pass_fds=(self.lease_fd,))
    pathlib.Path(os.environ['QA_ROOT'], 'model.pid').write_text(str(self.proc.pid))
    self.current = alias
    return True
module.Supervisor.start = start
raise SystemExit(module.main())
''')
    env = dict(os.environ, QA_ROOT=directory)
    args = [sys.executable, str(worker), '--server', sys.executable, '--model', 'test',
            '--port', str(port()), '--control-port', str(control), '--models-dir', directory]
    parent = subprocess.Popen([sys.executable, '-c',
        "import os,subprocess,sys,time; p=subprocess.Popen(sys.argv[1:]+['--parent-pid',str(os.getpid())]); print(p.pid,flush=True); time.sleep(60)",
        *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    model = None
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{control}/status', timeout=.2) as r:
                    if json.load(r)['state'] == 'ready': break
            except OSError: pass
            time.sleep(.05)
        else: raise AssertionError('first instance did not start')
        model = int((root / 'model.pid').read_text())
        second = subprocess.run(args, env=env, capture_output=True, text=True, timeout=5)
        assert second.returncode == 1, second
        assert second.stderr == '', second.stderr
        assert alive(model), 'second launch interfered with first model'
        parent.kill()
        parent.wait(timeout=3)
        for _ in range(100):
            if not alive(model): break
            time.sleep(.05)
        assert not alive(model), 'model survived launcher death'
        for _ in range(100):
            import fcntl
            with (root / '.local/share/localcode-agent/supervisor.lock').open('a+') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError: pass
            time.sleep(.05)
        else: raise AssertionError('lock not released after shutdown')
    finally:
        for pid in [parent.pid, child, model]:
            if pid and alive(pid):
                try: os.kill(pid, signal.SIGTERM)
                except ProcessLookupError: pass
        parent.wait(timeout=3)
print('second launch rejected; launcher death stops model and releases lock')
