"""A model download runs in the background: the loaded model keeps serving, other
installed models can still be loaded, and a second download waits its turn."""
import threading
from types import SimpleNamespace

import pytest

from localcode.ui.supervisor import MODEL_GROUPS, Supervisor


@pytest.fixture
def sup(tmp_path, monkeypatch):
    import localcode.ui.supervisor as module
    s = Supervisor.__new__(Supervisor)
    s.lock = threading.Lock()
    s.active = {}
    s.cancel = threading.Event()
    s._closing = threading.Event()
    s.download_cancel = threading.Event()
    s.current = 'test-model'
    s.models_dir = tmp_path
    (tmp_path / 'test-model.gguf').touch()
    (tmp_path / 'other-model.gguf').touch()
    s.state = {'state': 'ready', 'model': s.current, 'detail': '', 'pct': None}
    s.proc = SimpleNamespace(poll=lambda: None)
    s.progress = lambda: {'phase': 'generating'}  # a reply is streaming the whole time
    s.healthy = lambda: True
    listed = [SimpleNamespace(filename=f, size_gb=1.0, shards=()) for f in ('big-a.gguf', 'big-b.gguf')]
    s._quant_rows = lambda g, refresh=False: listed
    s._quant_for = lambda g, f: next((q for q in listed if q.filename == f), None)
    s._model_ready = lambda f, q=None, size=None: (tmp_path / f).is_file()
    s._drop_retired_sidecars = lambda: None
    monkeypatch.setattr(module, 'choice_for_quant', lambda g, f, size, shards=(): SimpleNamespace(filename=f))
    return s


def test_download_runs_in_the_background_and_leaves_the_loaded_model_alone(sup, monkeypatch):
    import localcode.ui.supervisor as module
    gate, started = threading.Event(), threading.Event()

    def download_model(choice, on_progress, cancel_event):
        started.set()
        on_progress('big-a.gguf: 1.0/4.0 GB (25%)')
        gate.wait(5)
        (sup.models_dir / choice.filename).write_bytes(b'GGUF')
        return True, str(sup.models_dir / choice.filename)
    monkeypatch.setattr(module.bootstrap, 'download_model', download_model)
    g = MODEL_GROUPS[0]
    g_no_drafter = SimpleNamespace(**{**g.__dict__, 'drafter': None}) if hasattr(g, '__dict__') else g
    monkeypatch.setattr(module, 'MODEL_GROUPS', [g_no_drafter])

    res = sup.select(g.key, 'big-a.gguf')
    assert res == {'ok': True, 'model': 'big-a', 'state': 'downloading', 'background': True}
    assert started.wait(3)
    # The chat keeps its model: state, server and switch lock are untouched.
    assert sup.state['state'] == 'ready' and sup.current == 'test-model' and not sup.lock.locked()
    assert sup.download['state'] == 'downloading' and sup.download['pct'] == 25
    # Picking the same download again is a no-op; a second download waits its turn.
    assert sup.select(g.key, 'big-a.gguf')['background'] is True
    assert 'still downloading' in sup.select(g.key, 'big-b.gguf')['error']
    # The downloading model cannot be loaded yet.
    gate.set()
    for _ in range(100):
        if sup.download['state'] == 'done':
            break
        threading.Event().wait(0.02)
    assert sup.download['state'] == 'done'
    assert sup.current == 'test-model'  # a finished download never replaces the loaded model


def test_cancel_stops_only_the_download(sup, monkeypatch):
    import localcode.ui.supervisor as module

    def download_model(choice, on_progress, cancel_event):
        assert cancel_event.wait(3)
        return False, 'cancelled'
    monkeypatch.setattr(module.bootstrap, 'download_model', download_model)
    g = MODEL_GROUPS[0]
    monkeypatch.setattr(module, 'MODEL_GROUPS', [SimpleNamespace(**{**g.__dict__, 'drafter': None})])
    sup.select(g.key, 'big-a.gguf')
    assert sup.cancel_download() == {'ok': True}
    for _ in range(100):
        if sup.download['state'] != 'downloading':
            break
        threading.Event().wait(0.02)
    assert sup.download['state'] == 'cancelled'
    assert sup.state['state'] == 'ready' and sup.current == 'test-model'
