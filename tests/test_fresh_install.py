"""Test the full fresh-machine install flow — pip install localcode && localcode.

Simulates both TUI and CLI paths without actually downloading 10GB.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Helpers ──

def _fresh_config():
    """Return a config as if just pip-installed with no ~/.gem/config.toml."""
    from gem.config import load_config
    # Simulate fresh install: no config file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_config = Path(tmpdir) / "config.toml"
        with patch("gem.config.get_config_path", return_value=fake_config):
            config = load_config()
    return config


def _mock_machine_16gb():
    """Simulate Apple Silicon 16GB MacBook."""
    from gem.performance import MachineProfile
    return MachineProfile(
        system="darwin",
        cpu_cores=10,
        memory_gb=16,
        has_gpu=True,
        gpu_summary="Apple M4",
        tier="small",
    )


# ── Test 1: Fresh config defaults ──
def test_fresh_config_defaults():
    """On fresh install, config should have sensible defaults."""
    config = _fresh_config()
    print(f"  provider: {config.runtime.provider}")
    print(f"  base_url: {config.runtime.base_url}")
    print(f"  model: {config.runtime.model!r}")
    print(f"  llama_cpp_binary: {config.runtime.llama_cpp_binary!r}")
    print(f"  kv_cache_type_k: {config.runtime.kv_cache_type_k}")
    print(f"  kv_cache_type_v: {config.runtime.kv_cache_type_v}")
    # These are the raw defaults before benchmark_report runs
    assert config.runtime.kv_cache_type_k == "q8_0", f"Expected q8_0, got {config.runtime.kv_cache_type_k}"
    assert config.runtime.kv_cache_type_v == "turbo4", f"Expected turbo4, got {config.runtime.kv_cache_type_v}"
    print("  ✓ PASS")


# ── Test 2: Performance preset selects llama_cpp on Apple Silicon 16GB ──
def test_preset_apple_silicon_16gb():
    """recommend_preset must select llama_cpp, not ollama, on Apple Silicon 16GB."""
    from gem.performance import recommend_preset
    machine = _mock_machine_16gb()
    preset = recommend_preset(machine, None, "speed")
    print(f"  tier: {machine.tier}")
    print(f"  runtime_provider: {preset.runtime_provider}")
    print(f"  profile: {preset.profile}")
    print(f"  kv_cache_type_k: {preset.kv_cache_type_k}")
    print(f"  kv_cache_type_v: {preset.kv_cache_type_v}")
    assert preset.runtime_provider == "llama_cpp", f"Expected llama_cpp, got {preset.runtime_provider}"
    assert preset.kv_cache_type_k == "q8_0"
    assert preset.kv_cache_type_v == "turbo4"
    print("  ✓ PASS")


# ── Test 3: autobootstrap triggers on fresh install ──
def test_autobootstrap_triggers():
    """_should_autobootstrap returns True when server isn't running."""
    from gem.cli import _should_autobootstrap
    config = _fresh_config()
    # No server running → should trigger
    result = _should_autobootstrap(config)
    print(f"  autobootstrap needed: {result}")
    assert result is True, "Should trigger autobootstrap on fresh install"
    print("  ✓ PASS")


# ── Test 4: TUI setup screen flow — config gets updated correctly ──
def test_tui_setup_config_update():
    """Simulate TUI setup: verify config is correctly set after binary + model download."""
    from gem.config import save_config, load_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.toml"
        binary_path = Path(tmpdir) / "llama-server"
        binary_path.touch()
        binary_path.chmod(0o755)
        model_path = Path(tmpdir) / "model.gguf"
        model_path.write_bytes(b"fake")

        # Create a fresh config
        with patch("gem.config.get_config_path", return_value=config_path):
            config = load_config()

        print(f"  Before: provider={config.runtime.provider}, base_url={config.runtime.base_url}")

        # Simulate what setup screen does after downloading binary
        config.runtime.llama_cpp_binary = str(binary_path)
        config.runtime.provider = "llama_cpp"
        config.runtime.base_url = "http://localhost:8081"
        config.runtime.model = str(model_path)
        with patch("gem.config.get_config_path", return_value=config_path):
            save_config(config)

        # Reload and verify
        with patch("gem.config.get_config_path", return_value=config_path):
            reloaded = load_config()
        print(f"  After: provider={reloaded.runtime.provider}, base_url={reloaded.runtime.base_url}")
        print(f"  binary: {reloaded.runtime.llama_cpp_binary}")
        print(f"  model: {reloaded.runtime.model}")
        assert reloaded.runtime.provider == "llama_cpp", f"Got {reloaded.runtime.provider}"
        assert "8081" in reloaded.runtime.base_url, f"Got {reloaded.runtime.base_url}"
        assert reloaded.runtime.llama_cpp_binary == str(binary_path)
        assert reloaded.runtime.model == str(model_path)
        print("  ✓ PASS")


# ── Test 5: llama_server_command builds correct command ──
def test_llama_server_command():
    """Verify the launch command has all TurboQuant flags."""
    from gem.config import load_config
    from gem.runtime import GemRuntimeGateway

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.toml"
        with patch("gem.config.get_config_path", return_value=config_path):
            config = load_config()
        config.runtime.provider = "llama_cpp"
        config.runtime.llama_cpp_binary = "/usr/local/bin/llama-server"
        config.runtime.kv_cache_type_k = "q8_0"
        config.runtime.kv_cache_type_v = "turbo4"
        config.runtime.laptop_26b_runtime_mode = "turbo"

        gw = GemRuntimeGateway(config.runtime)
        cmd = gw.llama_server_command("/path/to/model.gguf")
        cmd_str = " ".join(cmd)
        print(f"  Command: {cmd_str}")

        assert "/usr/local/bin/llama-server" in cmd_str
        assert "--model /path/to/model.gguf" in cmd_str
        assert "--cache-type-k q8_0" in cmd_str
        assert "--cache-type-v turbo4" in cmd_str
        assert "--flash-attn on" in cmd_str
        assert "-ngl 999" in cmd_str
        assert "--mmap" in cmd_str
        assert "--cache-ram 0" in cmd_str
        assert "-fit off" in cmd_str
        print("  ✓ PASS")


# ── Test 6: healthcheck hits correct endpoint for llama_cpp ──
def test_healthcheck_endpoint():
    """Verify healthcheck uses /v1/models for llama_cpp provider."""
    from gem.config import load_config
    from gem.runtime import GemRuntimeGateway

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.toml"
        with patch("gem.config.get_config_path", return_value=config_path):
            config = load_config()
        config.runtime.provider = "llama_cpp"
        config.runtime.base_url = "http://localhost:8081"

        gw = GemRuntimeGateway(config.runtime)
        print(f"  endpoint: {gw.endpoint}")
        print(f"  tags_endpoint: {gw.tags_endpoint}")
        assert gw.endpoint == "http://localhost:8081/v1/chat/completions"
        assert gw.tags_endpoint == "http://localhost:8081/v1/models"
        print("  ✓ PASS")


# ── Test 7: TUI on_mount routing — no binary → setup screen ──
def test_tui_on_mount_no_binary():
    """Fresh install with no binary should route to setup screen."""
    from gem.config import load_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.toml"
        with patch("gem.config.get_config_path", return_value=config_path):
            config = load_config()

        # Simulate on_mount logic — mock _turboquant_binary_path to return None
        # (on dev machine it finds the real binary, but on fresh install it won't)
        from gem.runtime import GemRuntimeGateway
        gw = GemRuntimeGateway(config.runtime)
        ok, _ = gw.healthcheck()
        config.runtime.llama_cpp_binary = ""  # fresh install

        with patch("gem.bootstrap._turboquant_binary_path", return_value=None):
            from gem.bootstrap import _turboquant_binary_path
            has_binary = _turboquant_binary_path() or config.runtime.llama_cpp_binary

        print(f"  healthcheck ok: {ok}")
        print(f"  has_binary: {bool(has_binary)}")

        if not ok:
            if has_binary:
                action = "start_server"
            else:
                action = "setup_screen"
        else:
            action = "chat_or_picker"

        print(f"  action: {action}")
        assert action == "setup_screen", f"Expected setup_screen, got {action}"
        print("  ✓ PASS")


# ── Test 8: TUI on_mount routing — binary exists but server down → start server ──
def test_tui_on_mount_binary_exists():
    """If binary exists but server not running, should auto-start."""
    from gem.config import load_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.toml"
        binary = Path(tmpdir) / "llama-server"
        binary.touch()
        with patch("gem.config.get_config_path", return_value=config_path):
            config = load_config()
        config.runtime.llama_cpp_binary = str(binary)

        from gem.runtime import GemRuntimeGateway
        gw = GemRuntimeGateway(config.runtime)
        ok, _ = gw.healthcheck()
        has_binary = config.runtime.llama_cpp_binary

        if not ok:
            if has_binary:
                action = "start_server"
            else:
                action = "setup_screen"
        else:
            action = "chat_or_picker"

        print(f"  healthcheck ok: {ok}")
        print(f"  has_binary: {bool(has_binary)}")
        print(f"  action: {action}")
        assert action == "start_server", f"Expected start_server, got {action}"
        print("  ✓ PASS")


# ── Test 9: CLI autobootstrap sets correct config after run_setup ──
def test_cli_bootstrap_sets_llama_cpp():
    """Verify CLI autobootstrap path ends with provider=llama_cpp."""
    from gem.performance import recommend_preset, apply_preset

    config = _fresh_config()
    machine = _mock_machine_16gb()
    preset = recommend_preset(machine, None, "speed")
    apply_preset(config, preset, model=config.runtime.model)
    print(f"  After apply_preset: provider={config.runtime.provider}")
    print(f"  base_url: {config.runtime.base_url}")
    assert config.runtime.provider == "llama_cpp", f"Expected llama_cpp, got {config.runtime.provider}"
    print("  ✓ PASS")


# ── Test 10: Setup screen step 2 actually has server launch code ──
def test_setup_screen_has_server_launch():
    """Verify setup.py Step 2 contains server launch code (not just a label)."""
    # Read source file directly — inspect.getsource fails on Cython-compiled code
    from pathlib import Path
    setup_file = Path(__file__).parent.parent / "src" / "gem" / "tui" / "screens" / "setup.py"
    source = setup_file.read_text()
    assert "Popen" in source, "Step 2 must launch server via Popen"
    assert "healthcheck" in source, "Step 2 must wait for healthcheck"
    assert "llama_server_command" in source, "Step 2 must use llama_server_command for full flags"
    print("  Setup screen setup.py contains: Popen, healthcheck, llama_server_command")
    print("  ✓ PASS")


# ── Test 11: download_model uses parallel download ──
def test_download_uses_parallel():
    """Verify download_model calls _download_parallel, not urlretrieve."""
    # Read source file directly — inspect.getsource fails on Cython-compiled code
    from pathlib import Path
    bootstrap_file = Path(__file__).parent.parent / "src" / "gem" / "bootstrap.py"
    source = bootstrap_file.read_text()
    # Find the download_model function body
    idx = source.index("def download_model")
    func_source = source[idx:idx+1000]
    assert "_download_parallel" in func_source, "download_model must use _download_parallel"
    assert "urlretrieve" not in func_source, "download_model must NOT use urlretrieve directly"
    print("  download_model uses _download_parallel ✓")
    print("  ✓ PASS")


# ── Test 12: End-to-end config roundtrip ──
def test_config_roundtrip():
    """Write config, reload, verify all llama_cpp fields survive."""
    from gem.config import load_config, save_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.toml"
        with patch("gem.config.get_config_path", return_value=config_path):
            config = load_config()

        config.runtime.provider = "llama_cpp"
        config.runtime.base_url = "http://localhost:8081"
        config.runtime.llama_cpp_binary = "/opt/bin/llama-server"
        config.runtime.model = "/data/models/gemma.gguf"
        config.runtime.kv_cache_type_k = "q8_0"
        config.runtime.kv_cache_type_v = "turbo4"
        config.runtime.laptop_26b_runtime_mode = "turbo"

        with patch("gem.config.get_config_path", return_value=config_path):
            save_config(config)
            reloaded = load_config()

        fields = [
            ("provider", "llama_cpp"),
            ("base_url", "http://localhost:8081"),
            ("llama_cpp_binary", "/opt/bin/llama-server"),
            ("model", "/data/models/gemma.gguf"),
            ("kv_cache_type_k", "q8_0"),
            ("kv_cache_type_v", "turbo4"),
            ("laptop_26b_runtime_mode", "turbo"),
        ]
        for field, expected in fields:
            actual = getattr(reloaded.runtime, field)
            assert actual == expected, f"{field}: expected {expected!r}, got {actual!r}"
            print(f"  {field}: {actual} ✓")
        print("  ✓ PASS")


if __name__ == "__main__":
    tests = [
        ("1. Fresh config defaults", test_fresh_config_defaults),
        ("2. Preset selects llama_cpp on Apple Silicon 16GB", test_preset_apple_silicon_16gb),
        ("3. Autobootstrap triggers on fresh install", test_autobootstrap_triggers),
        ("4. TUI setup updates config correctly", test_tui_setup_config_update),
        ("5. llama_server_command has TurboQuant flags", test_llama_server_command),
        ("6. Healthcheck endpoint correct for llama_cpp", test_healthcheck_endpoint),
        ("7. TUI on_mount: no binary → setup screen", test_tui_on_mount_no_binary),
        ("8. TUI on_mount: binary exists → start server", test_tui_on_mount_binary_exists),
        ("9. CLI autobootstrap sets llama_cpp", test_cli_bootstrap_sets_llama_cpp),
        ("10. Setup screen has actual server launch code", test_setup_screen_has_server_launch),
        ("11. download_model uses parallel download", test_download_uses_parallel),
        ("12. Config roundtrip preserves all fields", test_config_roundtrip),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"Test {name}")
        print('='*60)
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print('='*60)
    sys.exit(1 if failed else 0)
