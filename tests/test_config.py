"""Tests for localcode.config — loading, saving, env overrides, directory creation."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from localcode.config import (
    AppConfig,
    DEFAULT_CONFIG,
    LoggingConfig,
    RuntimeConfig,
    SafetyConfig,
    SearchConfig,
    UIConfig,
    ensure_home_dirs,
    get_config_path,
    get_home_dir,
    init_config_file,
    load_config,
    save_config,
)


class TestEnsureHomeDirs:
    """Verify that ensure_home_dirs creates the expected subdirectory tree."""

    def test_creates_expected_subdirs(self, tmp_path: Path) -> None:
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_test_home")
        try:
            home = ensure_home_dirs()
            for child in ("logs", "sessions", "jobs", "skills", "plugins"):
                assert (home / child).is_dir(), f"Missing subdir: {child}"
        finally:
            os.environ.pop("LOCALCODE_HOME", None)

    def test_idempotent(self, tmp_path: Path) -> None:
        """Calling ensure_home_dirs twice should not raise."""
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_test_home2")
        try:
            ensure_home_dirs()
            ensure_home_dirs()  # should not raise
        finally:
            os.environ.pop("LOCALCODE_HOME", None)


class TestGetHomeDir:
    """Verify LOCALCODE_HOME env override and default behaviour."""

    def test_override_via_env(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom_gem"
        os.environ["LOCALCODE_HOME"] = str(custom)
        try:
            assert get_home_dir() == custom
        finally:
            os.environ.pop("LOCALCODE_HOME", None)

    def test_default_is_dot_gem(self, monkeypatch) -> None:
        monkeypatch.delenv("LOCALCODE_HOME", raising=False)
        assert get_home_dir() == Path.home() / ".localcode"


class TestInitConfigFile:
    """Verify init_config_file writes the default TOML when absent."""

    def test_creates_default_config(self, tmp_path: Path) -> None:
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_init")
        try:
            path = init_config_file()
            assert path.exists()
            content = path.read_text()
            assert "[runtime]" in content
            assert "[search]" in content
            assert "[ui]" in content
        finally:
            os.environ.pop("LOCALCODE_HOME", None)

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_init2")
        try:
            path = init_config_file()
            path.write_text("# custom\n")
            init_config_file()
            assert path.read_text() == "# custom\n"
        finally:
            os.environ.pop("LOCALCODE_HOME", None)


class TestLoadConfig:
    """Verify load_config parses the TOML and builds AppConfig correctly."""

    def test_returns_appconfig(self, tmp_path: Path) -> None:
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_load")
        try:
            cfg = load_config()
            assert isinstance(cfg, AppConfig)
            assert isinstance(cfg.runtime, RuntimeConfig)
            assert isinstance(cfg.search, SearchConfig)
            assert isinstance(cfg.safety, SafetyConfig)
        finally:
            os.environ.pop("LOCALCODE_HOME", None)

    def test_default_provider(self, tmp_path: Path) -> None:
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_load2")
        try:
            cfg = load_config()
            assert cfg.runtime.provider in ("ollama", "llama_cpp")
        finally:
            os.environ.pop("LOCALCODE_HOME", None)

    def test_kv_cache_defaults(self, tmp_path: Path) -> None:
        """Default KV cache is q8_0 for both K and V.

        Measured on code text against an f16 reference: q8_0/q8_0 has 4x lower
        mean and 99.9%-tail KLD than q8_0/turbo4 for 5% less speed. The tail is
        the token that breaks a JSON key, so a coding agent takes the accuracy.
        """
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_load3")
        try:
            cfg = load_config()
            assert cfg.runtime.kv_cache_type_k == "q8_0"
            assert cfg.runtime.kv_cache_type_v == "q8_0"
        finally:
            os.environ.pop("LOCALCODE_HOME", None)


class TestSaveConfig:
    """Verify save_config writes a file that can be re-loaded."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """save_config -> load_config should preserve key values."""
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_rt")
        try:
            original = AppConfig(
                runtime=RuntimeConfig(
                    provider="llama_cpp",
                    model="my-model",
                    temperature=0.42,
                    llama_cpp_gpu_layers=12,
                    kv_cache_type_k="q8_0",
                    kv_cache_type_v="turbo4",
                ),
                search=SearchConfig(provider="brave", brave_api_key="abc123"),
                ui=UIConfig(show_debug=True, thinking_mode="full"),
                safety=SafetyConfig(confirm_destructive=False, max_fix_retries=5),
                logging=LoggingConfig(max_days=7),
            )
            save_config(original)
            reloaded = load_config()
            assert reloaded.runtime.provider == "llama_cpp"
            assert reloaded.runtime.model == "my-model"
            assert reloaded.runtime.temperature == pytest.approx(0.42)
            assert reloaded.runtime.llama_cpp_gpu_layers == 12
            assert reloaded.search.provider == "brave"
            assert reloaded.search.brave_api_key == "abc123"
            # browser / voice assertions dropped — sections removed in T0.9 purge
            assert reloaded.ui.show_debug is True
            assert reloaded.ui.thinking_mode == "full"
            assert reloaded.safety.confirm_destructive is False
            assert reloaded.safety.max_fix_retries == 5
            assert reloaded.logging.max_days == 7
        finally:
            os.environ.pop("LOCALCODE_HOME", None)


class TestEnvOverrides:
    """Verify that LOCALCODE_* environment variables override config file values."""

    def test_localcode_provider_env(self, tmp_path: Path) -> None:
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_env")
        os.environ["LOCALCODE_PROVIDER"] = "llama_cpp"
        try:
            cfg = load_config()
            assert cfg.runtime.provider == "llama_cpp"
        finally:
            os.environ.pop("LOCALCODE_HOME", None)
            os.environ.pop("LOCALCODE_PROVIDER", None)

    def test_localcode_model_env(self, tmp_path: Path) -> None:
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_env2")
        os.environ["LOCALCODE_MODEL"] = "custom-gemma"
        try:
            cfg = load_config()
            assert cfg.runtime.model == "custom-gemma"
        finally:
            os.environ.pop("LOCALCODE_HOME", None)
            os.environ.pop("LOCALCODE_MODEL", None)

    def test_localcode_llama_cpp_binary_env(self, tmp_path: Path) -> None:
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_env3")
        os.environ["LOCALCODE_LLAMA_CPP_BINARY"] = "/opt/llama/llama-server"
        try:
            cfg = load_config()
            assert cfg.runtime.llama_cpp_binary == "/opt/llama/llama-server"
        finally:
            os.environ.pop("LOCALCODE_HOME", None)
            os.environ.pop("LOCALCODE_LLAMA_CPP_BINARY", None)

    def test_localcode_temperature_env(self, tmp_path: Path) -> None:
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_env4")
        os.environ["LOCALCODE_TEMPERATURE"] = "0.99"
        try:
            cfg = load_config()
            assert cfg.runtime.temperature == pytest.approx(0.99)
        finally:
            os.environ.pop("LOCALCODE_HOME", None)
            os.environ.pop("LOCALCODE_TEMPERATURE", None)

    def test_localcode_base_url_env(self, tmp_path: Path) -> None:
        os.environ["LOCALCODE_HOME"] = str(tmp_path / "gem_env5")
        os.environ["LOCALCODE_BASE_URL"] = "http://192.168.1.100:8080"
        try:
            cfg = load_config()
            assert cfg.runtime.base_url == "http://192.168.1.100:8080"
        finally:
            os.environ.pop("LOCALCODE_HOME", None)
            os.environ.pop("LOCALCODE_BASE_URL", None)


class TestAppConfigPostInit:
    """Verify AppConfig.__post_init__ fills in defaults."""

    def test_safety_defaults_when_none(self) -> None:
        cfg = AppConfig(
            runtime=RuntimeConfig(),
            search=SearchConfig(),
            ui=UIConfig(),
            safety=None,
            logging=None,
        )
        assert isinstance(cfg.safety, SafetyConfig)
        assert isinstance(cfg.logging, LoggingConfig)
