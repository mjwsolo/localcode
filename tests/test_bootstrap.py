"""Tests for localcode.bootstrap — TurboQuant source detection and bundled binary paths."""
from __future__ import annotations

import platform
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from localcode.bootstrap import (
    _find_turboquant_source,
    _turboquant_binary_path,
    diffusion_cli_path,
)


class TestFindTurboquantSource:
    """Verify _find_turboquant_source looks in expected locations."""

    def test_finds_source_in_repo_root(self, tmp_path: Path) -> None:
        """If llama-cpp-turboquant/ is beside the localcode package, it should be found."""
        # Simulate: repo_root/llama-cpp-turboquant/CMakeLists.txt
        fake_repo = tmp_path / "repo"
        fake_repo.mkdir()
        tq = fake_repo / "llama-cpp-turboquant"
        tq.mkdir()
        (tq / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)")
        # Patch __file__ so the function thinks it's inside fake_repo/src/localcode/bootstrap.py
        fake_file = fake_repo / "src" / "localcode" / "bootstrap.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()
        with patch("localcode.bootstrap.__file__", str(fake_file)):
            result = _find_turboquant_source()
        assert result == tq

    def test_returns_none_when_not_present(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "src" / "localcode" / "bootstrap.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()
        with patch("localcode.bootstrap.__file__", str(fake_file)):
            with patch("pathlib.Path.home", return_value=tmp_path / "fakehome"):
                result = _find_turboquant_source()
        assert result is None

    def test_finds_source_in_home_dir(self, tmp_path: Path) -> None:
        """Fallback: check ~/llama-cpp-turboquant."""
        fake_home = tmp_path / "home"
        tq = fake_home / "llama-cpp-turboquant"
        tq.mkdir(parents=True)
        (tq / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)")
        fake_file = tmp_path / "elsewhere" / "src" / "localcode" / "bootstrap.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()
        with patch("localcode.bootstrap.__file__", str(fake_file)):
            with patch("pathlib.Path.home", return_value=fake_home):
                result = _find_turboquant_source()
        assert result == tq


class TestTurboquantBinaryPath:
    """Verify _turboquant_binary_path returns the built binary or None."""

    def test_returns_binary_when_built(self, tmp_path: Path) -> None:
        tq = tmp_path / "llama-cpp-turboquant"
        binary = tq / "build" / "bin" / "llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")
        (tq / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)")
        fake_bundled = tmp_path / "localcode" / "bin" / "llama-server"  # doesn't exist
        with patch("localcode.bootstrap.__file__", str(tmp_path / "localcode" / "bootstrap.py")):
            with patch("localcode.bootstrap._find_turboquant_source", return_value=tq):
                result = _turboquant_binary_path()
        assert result == binary

    def test_returns_none_when_not_built(self, tmp_path: Path) -> None:
        tq = tmp_path / "llama-cpp-turboquant"
        tq.mkdir()
        (tq / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)")
        with patch("localcode.bootstrap.__file__", str(tmp_path / "localcode" / "bootstrap.py")):
            with patch("pathlib.Path.home", return_value=tmp_path / "fakehome"):
                with patch("localcode.bootstrap._find_turboquant_source", return_value=tq):
                    result = _turboquant_binary_path()
        assert result is None

    def test_returns_none_when_no_source(self, tmp_path: Path) -> None:
        with patch("localcode.bootstrap.__file__", str(tmp_path / "localcode" / "bootstrap.py")):
            with patch("pathlib.Path.home", return_value=tmp_path / "fakehome"):
                with patch("localcode.bootstrap._find_turboquant_source", return_value=None):
                    result = _turboquant_binary_path()
        assert result is None


class TestDiffusionCliPath:
    """The diffusion runner resolves like llama-server: bundled first, never built."""

    def test_bundled_wins(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "pkg" / "localcode" / "bootstrap.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()
        bundled = fake_file.parent / "bin" / "llama-diffusion-cli"
        bundled.parent.mkdir()
        bundled.write_text("#!/bin/sh\n")
        with patch("localcode.bootstrap.__file__", str(fake_file)):
            assert diffusion_cli_path() == bundled

    def test_none_when_absent(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "pkg" / "localcode" / "bootstrap.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()
        with patch("localcode.bootstrap.__file__", str(fake_file)):
            with patch("pathlib.Path.home", return_value=tmp_path / "fakehome"):
                with patch("localcode.bootstrap._find_turboquant_source", return_value=None):
                    assert diffusion_cli_path() is None

    def test_no_build_or_network_helpers_exist(self) -> None:
        import localcode.bootstrap as b
        for name in ("ensure_diffusion_cli", "ensure_cohere_server", "ensure_muse_server",
                     "_ensure_cmake", "build_turboquant", "install_llama_cpp"):
            assert not hasattr(b, name), f"{name} must not come back: picking a model never builds"
