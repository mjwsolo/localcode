"""Exercise repository code loading in the shipped UI, with scratch files only."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pytest

from localcode.ui import plugin_path, ui_binary_path
from localcode.ui.launch import runtime_env, write_config


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="the bundled UI executable requires Apple Silicon macOS",
)


def _marker_plugin(path: Path, marker: Path, *, tui: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    export = '({id: "scratch-proof", tui: async () => {}})' if tui else "async () => ({})"
    path.write_text(
        'import {writeFileSync} from "node:fs";\n'
        f'writeFileSync({json.dumps(str(marker))}, "imported");\n'
        f"export default {export};\n"
    )


@pytest.fixture
def isolated_ui(tmp_path, monkeypatch):
    binary = ui_binary_path()
    assert binary is not None, "the release must include the UI binary"
    # Clear flags and redirect every persistent runtime location before invoking it.
    for key in tuple(os.environ):
        if key.startswith(("LOCALCODE_", "OPENCODE_")):
            monkeypatch.delenv(key)
    for name, folder in (
        ("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
        ("XDG_STATE_HOME", "state"), ("XDG_CACHE_HOME", "cache"),
        ("LOCALCODE_TEST_HOME", "home"),
    ):
        directory = tmp_path / folder
        directory.mkdir()
        monkeypatch.setenv(name, str(directory))
    monkeypatch.setenv("OPENCODE_DISABLE_MODELS_FETCH", "1")
    monkeypatch.setenv("LOCALCODE_DISABLE_PROJECT_CONFIG", "0")
    monkeypatch.setenv("OPENCODE_DISABLE_PROJECT_CONFIG", "0")
    project = tmp_path / "repository"
    project.mkdir()
    config = tmp_path / "launcher.json"
    write_config(config, port=1, ctx=8192, alias=None)
    explicit = tmp_path / "explicit.ts"
    explicit_marker = tmp_path / "explicit.marker"
    _marker_plugin(explicit, explicit_marker)
    cfg = json.loads(config.read_text())
    cfg["plugin"].append(str(explicit))
    config.write_text(json.dumps(cfg))
    global_dir = tmp_path / "config" / "localcode-agent"
    global_dir.mkdir()
    global_mcp = {"global-proof": {"type": "local", "command": ["false"], "enabled": False}}
    (global_dir / "localcode.json").write_text(json.dumps({"mcp": global_mcp}))
    project_mcp = {"project-proof": {"type": "local", "command": ["false"], "enabled": False}}
    configured = project / "configured.ts"
    configured_marker = tmp_path / "configured.marker"
    _marker_plugin(configured, configured_marker)
    (project / "opencode.json").write_text(json.dumps({"mcp": project_mcp, "plugin": ["./configured.ts"]}))
    discovered = []
    for directory in (".opencode", ".localcode-agent"):
        marker = tmp_path / f"{directory[1:]}.marker"
        _marker_plugin(project / directory / "plugins" / "proof.ts", marker)
        discovered.append(marker)
    return binary, project, config, explicit_marker, configured_marker, discovered


def _run(binary, project, env, *args):
    return subprocess.run(
        [str(binary), *args], cwd=project, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=20, check=False,
    )


@pytest.mark.parametrize("trusted", [False, True])
def test_project_plugins_and_mcp_require_explicit_trust(isolated_ui, trusted):
    binary, project, config, explicit, configured, discovered = isolated_ui
    env = runtime_env(config, 0, trust_project=trusted)
    resolved = _run(binary, project, env, "debug", "config")
    assert resolved.returncode == 0, resolved.stdout + resolved.stderr
    cfg = json.loads(resolved.stdout)
    assert "global-proof" in cfg["mcp"]
    assert ("project-proof" in cfg["mcp"]) is trusted
    assert any(str(plugin_path()) in str(spec) for spec in cfg["plugin"])
    assert any("/repository/.opencode/plugins/" in str(spec) for spec in cfg["plugin"]) is trusted
    # Startup imports plugins before resolving the deliberately absent model.
    # No inference, network service, or repository build command is required.
    _run(binary, project, env, "run", "--format", "json", "-m", "absent/absent", "scratch check")
    assert explicit.exists(), "explicit trusted plugins must still import"
    assert configured.exists() is trusted
    assert all(marker.exists() is trusted for marker in discovered)


@pytest.mark.parametrize("trusted", [False, True])
def test_project_tui_plugin_requires_explicit_trust(isolated_ui, trusted, tmp_path):
    import pty
    import select

    binary, project, config, _, _, _ = isolated_ui
    explicit_marker = tmp_path / "explicit-tui.marker"
    project_marker = tmp_path / "project-tui.marker"
    explicit_plugin = tmp_path / "explicit-tui.ts"
    project_plugin = project / "project-tui.ts"
    _marker_plugin(explicit_plugin, explicit_marker, tui=True)
    _marker_plugin(project_plugin, project_marker, tui=True)
    tui_config = config.with_suffix(".tui.json")
    cfg = json.loads(tui_config.read_text())
    cfg["plugin"].append(str(explicit_plugin))
    tui_config.write_text(json.dumps(cfg))
    (project / "tui.json").write_text(json.dumps({"plugin": ["./project-tui.ts"]}))
    env = runtime_env(config, 0, trust_project=trusted)
    env["TERM"] = "xterm-256color"
    master, slave = pty.openpty()
    proc = subprocess.Popen([str(binary)], cwd=project, env=env, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    output = bytearray()
    try:
        deadline = time.monotonic() + 15
        ready_at = None
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    output.extend(os.read(master, 65536))
                except OSError:
                    break
            if explicit_marker.exists():
                ready_at = ready_at or time.monotonic()
                if (trusted and project_marker.exists()) or (not trusted and time.monotonic() - ready_at > 0.5):
                    break
            if proc.poll() is not None:
                break
        assert explicit_marker.exists(), output.decode(errors="replace")[-3000:]
        assert project_marker.exists() is trusted
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        os.close(master)
