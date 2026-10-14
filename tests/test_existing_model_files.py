from pathlib import Path

from localcode import bootstrap, model_files
from localcode.ui.supervisor import Supervisor, MODEL_GROUPS


def test_nested_hf_snapshot_is_found_without_counting_sidecars_or_broken_links(tmp_path, monkeypatch):
    monkeypatch.setattr(model_files, 'search_roots', lambda _: [tmp_path])
    snapshot = tmp_path / 'models--test' / 'snapshots' / 'revision'
    snapshot.mkdir(parents=True)
    filename = MODEL_GROUPS[0].hf_repo.rsplit('/', 1)[-1].removesuffix('-GGUF') + '-Q4_K_M.gguf'
    weight = snapshot / filename
    weight.write_bytes(b'GGUF')
    (snapshot / 'mmproj-F16.gguf').write_bytes(b'GGUF')
    (snapshot / 'dflash-Q4_K_M.gguf').write_bytes(b'GGUF')
    (snapshot / 'broken.gguf').symlink_to(tmp_path / 'missing')
    assert model_files.model_files(tmp_path) == {filename: weight}
    assert model_files.find_model(filename, tmp_path) == weight


def test_configured_directory_wins_duplicate_names(tmp_path, monkeypatch):
    configured, cache = tmp_path / 'configured', tmp_path / 'cache'
    configured.mkdir(); cache.mkdir()
    for root in (configured, cache):
        (root / 'test.gguf').write_bytes(b'GGUF')
    monkeypatch.setattr(model_files, 'search_roots', lambda _: [configured, cache])
    assert model_files.find_model('test.gguf', configured) == configured / 'test.gguf'


def test_full_path_model_argument_honors_the_requested_file(tmp_path):
    weight = tmp_path / 'custom.gguf'
    with weight.open('wb') as stream:
        stream.write(b'GGUF')
        stream.truncate(1024 * 1024 + 1)
    assert bootstrap.resolve_model_arg(str(weight)) == weight


def test_picker_lists_local_only_quant_and_selects_it_offline(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    import localcode.ui.supervisor as module
    group = MODEL_GROUPS[0]
    filename = group.hf_repo.rsplit('/', 1)[-1].removesuffix('-GGUF') + '-Q4_K_M.gguf'
    cache, destination = tmp_path / 'cache', tmp_path / 'destination'
    cache.mkdir(); destination.mkdir()
    weight = cache / filename
    weight.write_bytes(b'GGUF')
    monkeypatch.setattr(model_files, 'search_roots', lambda _: [destination, cache])
    monkeypatch.setattr(module, 'fetch_quants', lambda _: [])
    sup = Supervisor.__new__(Supervisor)
    sup.models_dir = destination; sup.current = None
    sup.ram_gb = 64; sup.bandwidth = 150
    sup.active = {}; sup.lock = threading.Lock(); sup.cancel = threading.Event()
    sup.state = {'state': 'idle', 'pct': None}
    sup._rec_repo = lambda: None
    assert next(g for g in sup.catalog()['groups'] if g['key'] == group.key)['installed_count'] == 1
    row = next(q for q in sup.quants(group.key)['quants'] if q['filename'] == filename)
    assert row['downloaded'] is True
    selected = []
    def worker(g, fn, alias):
        selected.append(fn); sup.lock.release()
    sup._select_worker = worker
    assert sup.select(group.key, filename)['ok']
    assert sup.lock.acquire(timeout=3)
    sup.lock.release()
    assert selected == [filename]
    assert (destination / filename).is_symlink()
    assert (destination / filename).resolve() == weight.resolve()


def test_existing_weights_do_not_trigger_an_unrequested_drafter_download(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    import localcode.ui.supervisor as module
    monkeypatch.delenv('LOCALCODE_DOWNLOAD_DRAFTER', raising=False)
    filename = 'custom.gguf'; (tmp_path / filename).write_bytes(b'GGUF')
    group = SimpleNamespace(drafter=SimpleNamespace(local_path=tmp_path / 'missing-drafter.gguf'))
    monkeypatch.setattr(module.bootstrap, 'download_model', lambda *a, **k: (_ for _ in ()).throw(AssertionError('weights already exist')))
    monkeypatch.setattr(module.bootstrap, 'download_drafter', lambda *a, **k: (_ for _ in ()).throw(AssertionError('drafter was not requested')))
    sup = Supervisor.__new__(Supervisor)
    sup.models_dir = tmp_path; sup.lock = threading.Lock(); sup.lock.acquire()
    sup.cancel = threading.Event(); sup.active = {}; sup.current = None
    sup.start = lambda _: True
    sup._select_worker(group, filename, 'custom')
    assert sup.state['state'] == 'ready'
    assert not sup.lock.locked()
