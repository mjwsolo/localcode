"""Tests for localcode.bootstrap — TurboQuant source detection, binary path, install plans."""
from __future__ import annotations

import platform
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from localcode.bootstrap import (
    InstallPlan,
    _find_turboquant_source,
    _turboquant_binary_path,
    build_turboquant,
    detect_install_plan,
    detect_llama_cpp_install_plan,
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


class TestBuildTurboquant:
    """Verify build_turboquant fails gracefully when prerequisites are missing."""

    def test_fails_when_source_missing(self) -> None:
        with patch("localcode.bootstrap._find_turboquant_source", return_value=None):
            ok, msg = build_turboquant()
        assert ok is False
        assert "not found" in msg.lower()

    def test_fails_without_cmake(self, tmp_path: Path) -> None:
        tq = tmp_path / "llama-cpp-turboquant"
        tq.mkdir()
        (tq / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)")
        with patch("localcode.bootstrap._find_turboquant_source", return_value=tq):
            with patch("localcode.bootstrap._ensure_cmake", return_value=False):
                ok, msg = build_turboquant()
        assert ok is False
        assert "cmake" in msg.lower()


class TestDetectInstallPlan:
    """Verify detect_install_plan returns correct platform-specific plans."""

    def test_darwin_with_brew(self) -> None:
        with patch("platform.system", return_value="Darwin"):
            with patch("shutil.which", return_value="/usr/local/bin/brew"):
                plan = detect_install_plan()
        assert plan is not None
        assert plan.label == "Homebrew"
        assert "brew" in plan.command[0]
        assert "ollama" in plan.command

    def test_linux_with_apt(self) -> None:
        with patch("platform.system", return_value="Linux"):
            with patch("shutil.which", side_effect=lambda x: "/usr/bin/apt-get" if x == "apt-get" else None):
                plan = detect_install_plan()
        assert plan is not None
        assert plan.label == "apt"

    def test_unknown_platform_returns_none(self) -> None:
        with patch("platform.system", return_value="Windows"):
            with patch("shutil.which", return_value=None):
                plan = detect_install_plan()
        assert plan is None

    def test_darwin_no_brew_returns_none(self) -> None:
        with patch("platform.system", return_value="Darwin"):
            with patch("shutil.which", return_value=None):
                plan = detect_install_plan()
        assert plan is None


class TestDetectLlamaCppInstallPlan:
    """Verify detect_llama_cpp_install_plan picks the right tool."""

    def test_darwin_with_brew(self) -> None:
        with patch("platform.system", return_value="Darwin"):
            with patch("shutil.which", return_value="/usr/local/bin/brew"):
                plan = detect_llama_cpp_install_plan()
        assert plan is not None
        assert "llama.cpp" in plan.command or "llama" in str(plan.command)

    def test_no_tools_returns_none(self) -> None:
        with patch("platform.system", return_value="Linux"):
            with patch("shutil.which", return_value=None):
                plan = detect_llama_cpp_install_plan()
        assert plan is None
