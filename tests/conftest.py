"""Shared pytest fixtures for the LocalCode test suite."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gem.config import (
    AppConfig,
    BrowserConfig,
    LoggingConfig,
    RuntimeConfig,
    SafetyConfig,
    SearchConfig,
    UIConfig,
    VoiceConfig,
)
from gem.runtime import GemRuntimeGateway


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with a few Python files."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True, check=True,
    )
    # Create some files
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    (repo / "utils.py").write_text("import os\n\ndef get_cwd():\n    return os.getcwd()\n")
    (repo / "sub").mkdir()
    (repo / "sub" / "module.py").write_text("class Foo:\n    pass\n")
    # Initial commit
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo, capture_output=True, check=True,
    )
    return repo


@pytest.fixture
def mock_config(tmp_path: Path) -> AppConfig:
    """Return an AppConfig with sensible test defaults, using a temp GEM_HOME."""
    gem_home = tmp_path / "gem_home"
    gem_home.mkdir()
    os.environ["GEM_HOME"] = str(gem_home)
    config = AppConfig(
        runtime=RuntimeConfig(
            provider="llama_cpp",
            base_url="http://localhost:9999",
            profile="e4b",
            model="test-model",
            mode="fast",
            temperature=0.1,
            max_context_chars=8000,
            llama_cpp_binary="/usr/local/bin/llama-server",
            kv_cache_type_k="q8_0",
            kv_cache_type_v="turbo4",
            laptop_26b_runtime_mode="speed",
        ),
        search=SearchConfig(),
        browser=BrowserConfig(),
        voice=VoiceConfig(),
        ui=UIConfig(),
        safety=SafetyConfig(),
        logging=LoggingConfig(),
    )
    yield config
    os.environ.pop("GEM_HOME", None)


@pytest.fixture
def mock_runtime(mock_config: AppConfig) -> GemRuntimeGateway:
    """Return a GemRuntimeGateway that does not need a real server.

    HTTP calls are replaced with a MagicMock so no network I/O occurs.
    """
    gw = GemRuntimeGateway(mock_config.runtime)
    gw._client = MagicMock()
    return gw
