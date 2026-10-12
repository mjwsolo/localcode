"""Explicit model lifecycle: busy guards, failure recovery and HTTP authorization."""
import threading
from types import SimpleNamespace

import pytest

from localcode.ui.supervisor import Supervisor


@pytest.fixture
def sup(tmp_path):
    s = Supervisor.__new__(Supervisor)
    s.lock = threading.Lock()
    s.current = 'test-model'
    s.models_dir = tmp_path
    (tmp_path / 'test-model.gguf').touch()
    s.state = {'state': 'ready', 'model': s.current}
    s.proc = SimpleNamespace(poll=lambda: None)
    s.progress = lambda: {'phase': 'idle'}
    s.healthy = lambda: True
    def stop():
        s.proc = None
    def start(alias):
        assert alias == 'test-model'
        s.proc = SimpleNamespace(poll=lambda: None)
        return True
    s.stop = stop
    s.start = start
    return s


def done(s):
    assert s.lock.acquire(timeout=3), 'lifecycle worker did not release lock'
    s.lock.release()


def test_stop_start_preserves_selected_model_and_reports_liveness(sup):
    assert sup.server_action('stop')['ok']
    done(sup)
    assert sup.current == 'test-model'
    assert sup.server_status()['state'] == 'stopped'
    assert not sup.server_status()['server_ready']
    assert sup.server_action('stop')['state'] == 'stopped'  # idempotent
    assert sup.server_action('start')['ok']
    done(sup)
    assert sup.server_status()['server_ready']


@pytest.mark.parametrize('phase', ['generating', 'prefill', 'unknown', None])
def test_stop_refuses_active_or_unverifiable_requests(sup, phase):
    sup.progress = lambda: {'phase': phase}
    assert 'error' in sup.server_action('stop')
    assert sup.proc is not None
    assert not sup.lock.locked()


def test_stop_rechecks_activity_before_termination(sup):
    phases = iter(['idle', 'generating'])
    sup.progress = lambda: {'phase': next(phases)}
    assert sup.server_action('stop')['ok']
    done(sup)
    assert sup.proc is not None
    assert 'cancelled' in sup.state['detail']


@pytest.mark.parametrize('action', ['start', 'stop'])
def test_operations_serialize_with_model_switch(sup, action):
    sup.lock.acquire()
    try:
        assert 'error' in sup.server_action(action)
    finally:
        sup.lock.release()


@pytest.mark.parametrize('reason', ['missing', 'no-model', 'running'])
def test_invalid_start_leaves_current_state_intact(sup, reason):
    if reason != 'running':
        sup.proc = None
    if reason == 'missing':
        (sup.models_dir / 'test-model.gguf').unlink()
    if reason == 'no-model':
        sup.current = None
    before = dict(sup.state)
    assert 'error' in sup.server_action('start')
    assert sup.state == before
    assert not sup.lock.locked()


@pytest.mark.parametrize('raises', [False, True])
def test_failed_load_cleans_up_and_can_retry(sup, raises):
    sup.proc = None
    def fail(_):
        sup.proc = SimpleNamespace(poll=lambda: None)
        if raises:
            raise OSError('load failed')
        return False
    good = sup.start
    sup.start = fail
    assert sup.server_action('start')['ok']
    done(sup)
    assert sup.state['state'] == 'error' and sup.proc is None
    sup.start = good
    assert sup.server_action('start')['ok']
    done(sup)
    assert sup.state['state'] == 'ready'


def test_thread_start_failure_restores_state(sup, monkeypatch):
    before = dict(sup.state)
    monkeypatch.setattr(threading.Thread, 'start', lambda _: (_ for _ in ()).throw(RuntimeError('no threads')))
    assert 'error' in sup.server_action('stop')
    assert sup.state == before and not sup.lock.locked()


def test_lifecycle_routes_require_auth_and_object_payload(monkeypatch):
    from tests.test_ui_server import control_request
    # control_request's fake supervisor needs the same entry point.
    import localcode.ui.supervisor as module
    original = module.make_handler
    calls = []
    def handler(s):
        s.server_action = lambda action: calls.append(action) or {'ok': True, 'state': 'loading'}
        return original(s)
    monkeypatch.setattr(module, 'make_handler', handler)
    for action in ('start', 'stop'):
        assert control_request('/server/'+action, method='POST', token='bad', body={})[0] == 403
        assert control_request('/server/'+action, method='POST', body=[])[0] == 400
        assert control_request('/server/'+action, method='POST', body={})[0] == 200
    assert calls == ['start', 'stop']


@pytest.mark.parametrize('code,name', [(-11, 'SIGSEGV'), (-6, 'SIGABRT'), (1, 'exit code 1'), (0, 'exit code 0')])
def test_unexpected_exit_reports_actual_cause_without_probing_dead_port(sup, monkeypatch, code, name):
    import urllib.request
    sup.proc = SimpleNamespace(poll=lambda: code, pid=2345)
    sup.progress = Supervisor.progress.__get__(sup)
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs: pytest.fail('dead port must not be probed'))
    status = sup.server_status()
    assert status['state'] == 'error' and not status['server_running']
    assert status['exit_code'] == code
    assert name in status['detail'] and '/server' in status['detail']
    progress = sup.progress()
    assert progress['phase'] == 'error' and progress['process_id'] == 2345
    assert progress['exit_code'] == code


def test_intentional_stop_and_new_live_process_do_not_report_previous_crash(sup):
    sup.proc = SimpleNamespace(poll=lambda: -11, pid=2345)
    assert sup.server_status()['exit_code'] == -11
    sup.state['state'] = 'stopped'
    assert sup._server_failure() == {}
    assert sup.server_status()['state'] == 'stopped'
    sup.state['state'] = 'ready'
    sup.proc = SimpleNamespace(poll=lambda: None, pid=2346)
    assert sup.server_status()['server_ready']
    assert 'exit_code' not in sup.server_status()
