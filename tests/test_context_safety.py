"""Computed context ceilings and post-tuning validation use the actual machine."""
import pytest

from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway


@pytest.mark.parametrize('kv', ['q8_0', 'turbo4'])
def test_stale_balanced_context_cannot_exceed_24gib_tier(kv, monkeypatch):
    runtime = LocalCodeRuntimeGateway(RuntimeConfig(quant_preset='balanced', max_context_chars=393216, kv_cache_type_v=kv))
    monkeypatch.setattr(runtime, '_system_ram_gb', lambda: 24)
    monkeypatch.setattr(runtime, '_kv_bytes_per_token', lambda _: None)
    assert runtime._target_num_ctx(model_path='Qwen3.6-35B-A3B-UD-IQ2_M.gguf') <= runtime._ram_ctx_ceiling(24)


@pytest.mark.parametrize('source', ['environment', 'recommendation'])
def test_tuned_launcher_context_cannot_bypass_machine_ceiling(source, monkeypatch, tmp_path):
    import localcode.recommendations as recommendations
    from localcode.ui import server_cmd
    from localcode.config import AppConfig, UIConfig
    cfg = AppConfig(runtime=RuntimeConfig(), ui=UIConfig())
    cfg.runtime.kv_cache_type_v = 'turbo4'
    monkeypatch.setattr(server_cmd, 'load_config', lambda: cfg)
    monkeypatch.setattr(server_cmd, 'validate_model_memory', lambda *_: None)
    monkeypatch.setattr(LocalCodeRuntimeGateway, '_system_ram_gb', lambda _: 24)
    monkeypatch.setenv('LOCALCODE_AGENT_RUN_DIR', str(tmp_path / 'run'))
    monkeypatch.setenv('LOCALCODE_PROMPT_WARMUP', '0')
    monkeypatch.delenv('LOCALCODE_OVERRIDE_NCTX', raising=False)
    monkeypatch.setattr(recommendations, 'load_overrides', lambda _: {'LOCALCODE_OVERRIDE_NCTX':'98304'} if source=='recommendation' else {})
    if source == 'environment': monkeypatch.setenv('LOCALCODE_OVERRIDE_NCTX', '98304')
    command = server_cmd.server_command(str(tmp_path / 'Qwen3.6-35B-A3B-UD-IQ2_M.gguf'), 8123)
    assert command[command.index('--ctx-size') + 1] == '73728'
