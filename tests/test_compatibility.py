"""Platform/enterprise errors and the default shipping server command."""
from pathlib import Path

import pytest

from localcode import bootstrap, diagnostics, models_catalog
from localcode.config import AppConfig, RuntimeConfig, UIConfig
from localcode.runtime import LocalCodeRuntimeGateway
from localcode.ui import server_cmd


@pytest.mark.parametrize('system,arch,mac,translated,fragment', [
    ('Linux', 'x86_64', '', '', 'Apple silicon'),
    ('Darwin', 'x86_64', '14.0', '0', 'Intel'),
    ('Darwin', 'x86_64', '14.0', '1', 'Rosetta'),
    ('Darwin', 'arm64', '12.7', '', 'too old'),
    ('Darwin', 'arm64', '', '', 'determine'),
    ('Darwin', 'arm64', '13.0', '', None),
    ('Darwin', 'arm64', '26.0', '', None),
])
def test_platform_diagnosis(monkeypatch, system, arch, mac, translated, fragment):
    monkeypatch.setattr(diagnostics.platform, 'system', lambda: system)
    monkeypatch.setattr(diagnostics.platform, 'machine', lambda: arch)
    monkeypatch.setattr(diagnostics.platform, 'mac_ver', lambda: (mac, (), ''))
    monkeypatch.setattr(diagnostics, 'sysctl', lambda _: translated)
    result = diagnostics.platform_problem()
    assert (result is None) if fragment is None else fragment in result


def test_doctor_does_not_expose_proxy_credentials(monkeypatch):
    monkeypatch.setenv('HTTPS_PROXY', 'https://username:secret@example.com')
    assert 'secret' not in str(diagnostics.report())
    assert diagnostics.report()['network_configuration_present']['HTTPS_PROXY']


@pytest.mark.parametrize('error,category,hint', [
    ('connection failed: certificate verify failed', 'ssl', 'IT'),
    ('407 Proxy Authentication Required https://user:secret@proxy', 'proxy', 'IT'),
    ('No space left on device', 'disk_full', 'space'),
    ('403 Forbidden', 'auth', 'token'),
])
def test_download_diagnosis(error, category, hint):
    e = RuntimeError(error)
    assert bootstrap._classify_download_error(e) == category
    message = bootstrap._format_download_error(category, e)
    assert hint in message
    if category == 'proxy':
        assert 'secret' not in message


RAM = [8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192]


@pytest.mark.parametrize('ram', RAM)
@pytest.mark.parametrize('choice', models_catalog.CHOICES, ids=lambda c: c.key)
def test_shipping_command_model_ram_matrix(monkeypatch, tmp_path, ram, choice):
    config = AppConfig(runtime=RuntimeConfig(), ui=UIConfig())
    config.runtime.llama_cpp_binary = str(tmp_path / 'llama-server')
    Path(config.runtime.llama_cpp_binary).touch()
    config.runtime.quant_preset = 'fastest'
    config.runtime.kv_cache_type_v = 'turbo4'
    config.runtime.laptop_26b_runtime_mode = 'turbo'
    config.runtime.max_context_chars = 400000
    monkeypatch.setattr(server_cmd, 'load_config', lambda: config)
    monkeypatch.setattr(LocalCodeRuntimeGateway, '_system_ram_gb', lambda _: ram)
    monkeypatch.setenv('LOCALCODE_SERVER_KEY', 'test-only-key')
    monkeypatch.setenv('LOCALCODE_AGENT_RUN_DIR', str(tmp_path / 'run'))
    monkeypatch.setenv('LOCALCODE_PROMPT_WARMUP', '0')
    cmd = server_cmd.server_command(str(tmp_path / choice.filename), 8123, choice.key)
    assert cmd[cmd.index('--host') + 1] == '127.0.0.1'
    assert cmd[cmd.index('--port') + 1] == '8123'
    assert cmd[cmd.index('--alias') + 1] == choice.key
    assert 2048 <= int(cmd[cmd.index('--ctx-size') + 1]) <= LocalCodeRuntimeGateway._ram_ctx_ceiling(ram)
    assert '--api-key' in cmd


def test_oversize_model_rejected_before_spawn(tmp_path, monkeypatch):
    model = tmp_path / 'large.gguf'
    with model.open('wb') as f:
        f.truncate(7 * 1024 ** 3)  # sparse file, no 7 GB allocation
    monkeypatch.setattr(server_cmd, 'mmproj_for', lambda _: None)
    monkeypatch.setattr(server_cmd, 'catalog_choice', lambda _: None)
    with pytest.raises(ValueError, match='Insufficient RAM'):
        server_cmd.validate_model_memory(str(model), 8)
    server_cmd.validate_model_memory(str(model), 16)


def test_memory_guard_counts_sidecars(monkeypatch, tmp_path):
    from types import SimpleNamespace
    paths = [tmp_path / name for name in ('model.gguf', 'mmproj.gguf', 'draft.gguf')]
    for path in paths:
        with path.open('wb') as f:
            f.truncate(2 * 1024**3)
    monkeypatch.setattr(server_cmd, 'mmproj_for', lambda _: paths[1])
    monkeypatch.setattr(server_cmd, 'catalog_choice', lambda _: SimpleNamespace(drafter=SimpleNamespace(local_path=paths[2])))
    with pytest.raises(ValueError, match='Insufficient RAM'):
        server_cmd.validate_model_memory(str(paths[0]), 8)


def test_rejected_load_keeps_existing_server(monkeypatch, tmp_path):
    from localcode.ui.supervisor import Supervisor
    sup = object.__new__(Supervisor)
    sup.models_dir = tmp_path
    sup.port = 8000
    stopped = []
    sup.stop = lambda: stopped.append(True)
    def refuse(*args):
        raise ValueError('Insufficient RAM')
    monkeypatch.setattr(server_cmd, 'server_command', refuse)
    with pytest.raises(ValueError, match='Insufficient RAM'):
        sup.start('too-large')
    assert not stopped
