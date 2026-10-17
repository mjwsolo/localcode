"""Explicit model lifecycle: busy guards, failure recovery and HTTP authorization."""
import threading
from types import SimpleNamespace

import pytest

from localcode.ui.supervisor import Supervisor


@pytest.fixture
def sup(tmp_path):
    s = Supervisor.__new__(Supervisor)
    s.lock = threading.Lock()
    s.active = {}
    s.cancel = threading.Event()
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
        s.server_action = lambda action, **kwargs: calls.append(action) or {'ok': True, 'state': 'loading'}
        return original(s)
    monkeypatch.setattr(module, 'make_handler', handler)
    for action in ('start', 'stop'):
        assert control_request('/server/'+action, method='POST', token='bad', body={})[0] == 403
        assert control_request('/server/'+action, method='POST', body=[])[0] == 400
        assert control_request('/server/'+action, method='POST', body={})[0] == 200
    assert calls == ['start', 'stop']


@pytest.mark.parametrize('code,name', [(-11, 'SIGSEGV'), (-6, 'SIGABRT'), (1, 'exit code 1'), (0, 'exit code 0')])
def test_unexpected_exit_reports_actual_cause_without_probing_dead_port(sup, monkeypatch, code, name):
    import localcode.ui.supervisor as module
    sup.proc = SimpleNamespace(poll=lambda: code, pid=2345)
    sup.progress = Supervisor.progress.__get__(sup)
    monkeypatch.setattr(module, 'local_urlopen', lambda *args, **kwargs: pytest.fail('dead port must not be probed'))
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


def test_select_loaded_model_leaves_server_and_warmup_untouched(sup, monkeypatch):
    from localcode.ui.supervisor import MODEL_GROUPS

    sup.cancel = threading.Event()
    sup.cancel.set()
    sup.active = {}
    sup._warm_status = 'warm'
    before = dict(sup.state)
    process = sup.proc
    monkeypatch.setattr(threading.Thread, 'start', lambda _: pytest.fail('must not reload'))
    result = sup.select(MODEL_GROUPS[0].key, 'test-model.gguf')
    assert result['already_loaded'] is True and result['detail'] == 'Model already loaded'
    assert result['state'] == 'ready' and result['ok'] is True
    assert sup.state == before and sup.proc is process
    assert sup._warm_status == 'warm' and sup.cancel.is_set()
    assert sup.active == {} and not sup.lock.locked()


@pytest.mark.parametrize('failure', ['stopped', 'exited', 'unhealthy', 'error-state'])
def test_select_previous_model_can_reload_when_server_is_unavailable(sup, monkeypatch, failure):
    from localcode.ui.supervisor import MODEL_GROUPS

    sup.cancel = threading.Event()
    if failure == 'stopped':
        sup.proc = None
        sup.state['state'] = 'stopped'
    elif failure == 'exited':
        sup.proc = SimpleNamespace(poll=lambda: 1)
    elif failure == 'unhealthy':
        sup.healthy = lambda: False
    else:
        sup.state['state'] = 'error'
        sup.healthy = lambda: False
    calls = []
    def worker(group, filename, alias):
        calls.append((group.key, filename, alias))
        sup.lock.release()
    sup._select_worker = worker
    assert sup.select(MODEL_GROUPS[0].key, 'test-model.gguf')['ok']
    done(sup)
    assert calls == [(MODEL_GROUPS[0].key, 'test-model.gguf', 'test-model')]


def test_select_loaded_model_during_another_operation_keeps_busy_guard(sup):
    from localcode.ui.supervisor import MODEL_GROUPS

    sup.lock.acquire()
    try:
        assert sup.select(MODEL_GROUPS[0].key, 'test-model.gguf') == {
            'error': 'a model switch is already in progress',
        }
        assert sup.lock.locked()
    finally:
        sup.lock.release()


def test_select_thread_failure_restores_state_and_allows_retry(sup, monkeypatch):
    from localcode.ui.supervisor import MODEL_GROUPS
    sup.current = None
    sup.cancel = threading.Event(); sup.cancel.set()
    sup.active = {}
    before = dict(sup.state)
    monkeypatch.setattr(threading.Thread, 'start', lambda _: (_ for _ in ()).throw(RuntimeError('no threads')))
    assert 'error' in sup.select(MODEL_GROUPS[0].key, 'test-model.gguf')
    assert sup.state == before and sup.active == {} and sup.cancel.is_set()
    assert not sup.lock.locked()


def test_reselect_loaded_model_does_not_touch_network_if_file_was_moved(sup, monkeypatch):
    import localcode.ui.supervisor as module
    (sup.models_dir / 'test-model.gguf').unlink()
    monkeypatch.setattr(module, 'fetch_quants', lambda _: pytest.fail('must not access HF'))
    monkeypatch.setattr(module, 'find_model', lambda *args: pytest.fail('must not scan model folders'))
    assert sup.select(module.MODEL_GROUPS[0].key, 'test-model.gguf')['already_loaded'] is True
    assert not sup.lock.locked()


def test_healthy_loaded_model_recovers_a_stale_error_without_restart(sup, monkeypatch):
    import localcode.ui.supervisor as module
    sup.state = {'state': 'error', 'model': 'failed-other-model'}
    monkeypatch.setattr(threading.Thread, 'start', lambda _: pytest.fail('healthy model must not restart'))
    assert sup.select(module.MODEL_GROUPS[0].key, 'test-model.gguf')['already_loaded']
    assert sup.state['state'] == 'ready' and sup.state['model'] == 'test-model'


@pytest.mark.parametrize('phase', ['generating', 'reading', 'unknown'])
def test_model_switch_refuses_active_or_unverifiable_requests_from_other_clients(sup, phase):
    from localcode.ui.supervisor import MODEL_GROUPS
    sup.progress = lambda: {'phase': phase}
    before = dict(sup.state)
    result = sup.select(MODEL_GROUPS[0].key, 'another-model.gguf')
    assert 'error' in result and 'active' in result['error']
    assert sup.state == before and sup.proc is not None and not sup.lock.locked()


@pytest.mark.parametrize('phase', ['generating', 'reading', 'unknown', None])
def test_explicit_stop_confirmation_stops_busy_or_unreachable_model(sup, phase):
    sup.progress = lambda: {'phase': phase}
    rejected = sup.server_action('stop')
    assert rejected['force_required'] is True
    assert sup.proc is not None
    assert sup.server_action('stop', force=True)['ok']
    done(sup)
    assert sup.proc is None
    assert sup.state['state'] == 'stopped'


def test_confirmed_stop_cannot_bypass_a_model_operation_lock(sup):
    sup.lock.acquire()
    try:
        assert 'error' in sup.server_action('stop', force=True)
        assert sup.proc is not None
    finally:
        sup.lock.release()


@pytest.mark.parametrize('force', ['true', 1, None, {}])
def test_stop_rejects_non_boolean_confirmation(force):
    from tests.test_ui_server import control_request
    response, _ = control_request('/server/stop', method='POST', body={'force': force})
    assert response == 400


def test_crash_watchdog_restarts_a_crashed_model_once(sup, monkeypatch):
    import localcode.ui.supervisor as module
    monkeypatch.setattr(module, "log", lambda message: None)
    sup._closing = threading.Event()
    sup.proc = SimpleNamespace(poll=lambda: -9, pid=1)
    starts = []

    def start(alias):
        starts.append(alias)
        sup.proc = SimpleNamespace(poll=lambda: -9, pid=2)  # crashes again at once
        if len(starts) == 1:
            sup.state = {'state': 'ready', 'model': alias}
        return True

    sup.start = start
    ticks = []
    real_sleep = module.time.sleep

    def sleep(_):
        ticks.append(1)
        if len(ticks) >= 5:
            sup._closing.set()
        real_sleep(0)

    monkeypatch.setattr(module.time, "sleep", sleep)
    sup.crash_watchdog(interval=0, cooldown=60)
    # One automatic restart; the repeat crash inside the cooldown is left for the user.
    assert starts == ['test-model']
    assert not sup.lock.locked()


def test_crash_watchdog_ignores_a_stopped_server(sup, monkeypatch):
    import localcode.ui.supervisor as module
    sup._closing = threading.Event()
    sup.state = {'state': 'stopped', 'model': 'test-model'}
    sup.proc = SimpleNamespace(poll=lambda: 0, pid=1)
    sup.start = lambda alias: pytest.fail('an intentional stop must not restart')
    calls = []
    monkeypatch.setattr(module.time, "sleep", lambda _: (calls.append(1), len(calls) >= 3 and sup._closing.set()))
    sup.crash_watchdog(interval=0)
