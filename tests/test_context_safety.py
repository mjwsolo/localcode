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


@pytest.mark.parametrize("weights_gib,expect_rejection", [(12, True), (10, False)])
def test_final_memory_budget_includes_explicit_drafter(weights_gib, expect_rejection, monkeypatch, tmp_path):
    from localcode.ui import server_cmd
    from localcode.config import AppConfig, UIConfig
    model = tmp_path / "example-Q4_K_M.gguf"
    draft = tmp_path / "example-draft.gguf"
    for path, size in ((model, weights_gib), (draft, 1)):
        with path.open("wb") as f:
            f.truncate(size * 1024**3)
    cfg = AppConfig(runtime=RuntimeConfig(llama_cpp_draft_model=str(draft)), ui=UIConfig())
    monkeypatch.setattr(server_cmd, 'load_config', lambda: cfg)
    monkeypatch.setattr(LocalCodeRuntimeGateway, '_system_ram_gb', lambda _: 16)
    monkeypatch.setattr(LocalCodeRuntimeGateway, '_gguf_kv_meta', lambda *a: {"n_layers":48,"n_kv_heads":8,"head_dim":128})
    monkeypatch.setenv('LOCALCODE_PROMPT_WARMUP', '0')
    if expect_rejection:
        with pytest.raises(ValueError, match="minimum context"):
            server_cmd.server_command(str(model), 8123)
    else:
        cmd = server_cmd.server_command(str(model), 8123)
        ctx = int(cmd[cmd.index('--ctx-size')+1])
        # Target uses q8/q8; an ordinary drafter independently uses f16/f16.
        assert ctx * 48 * 8 * 128 * (2.125 + 4) + (weights_gib + 1 + 3) * 1024**3 <= 16 * 1024**3
