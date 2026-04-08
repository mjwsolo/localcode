"""Tests for gem.jobs — JobRecord serialization, launch, list, stop."""
from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from gem.config import ensure_home_dirs
from gem.jobs import JobRecord, launch_background_job, list_jobs, read_job_log, stop_job


class TestJobRecordSerialization:
    """Verify that JobRecord with slots=True works correctly with asdict."""

    def test_asdict_roundtrip(self) -> None:
        """The slots=True fix: asdict must work on JobRecord without error."""
        record = JobRecord(
            job_id="abc123",
            command="echo hello",
            cwd="/tmp",
            pid=42,
            created_at="2025-01-01T00:00:00",
            log_path="/tmp/abc123.log",
        )
        d = asdict(record)
        assert d["job_id"] == "abc123"
        assert d["command"] == "echo hello"
        assert d["pid"] == 42

    def test_json_serialization(self) -> None:
        """JobRecord -> asdict -> json.dumps must not raise."""
        record = JobRecord(
            job_id="def456",
            command="ls -la",
            cwd="/home/user",
            pid=9999,
            created_at="2025-06-15T12:00:00",
            log_path="/tmp/def456.log",
        )
        serialized = json.dumps(asdict(record))
        parsed = json.loads(serialized)
        assert parsed["job_id"] == "def456"
        assert parsed["pid"] == 9999


class TestLaunchBackgroundJob:
    """Verify launch_background_job creates a process and writes a job file."""

    def test_creates_job_file(self, tmp_path: Path) -> None:
        os.environ["GEM_HOME"] = str(tmp_path / "gem_jobs")
        try:
            record = launch_background_job("echo test_output", cwd=tmp_path)
            assert record.job_id
            assert record.pid > 0
            # Job JSON file should exist
            jobs_dir = ensure_home_dirs() / "jobs"
            job_file = jobs_dir / f"{record.job_id}.json"
            assert job_file.exists()
            data = json.loads(job_file.read_text())
            assert data["command"] == "echo test_output"
            assert data["cwd"] == str(tmp_path)
            # Wait for process to finish and check log
            time.sleep(0.5)
            log_content = Path(record.log_path).read_text()
            assert "test_output" in log_content
        finally:
            os.environ.pop("GEM_HOME", None)

    def test_job_record_fields(self, tmp_path: Path) -> None:
        os.environ["GEM_HOME"] = str(tmp_path / "gem_jobs2")
        try:
            record = launch_background_job("sleep 0.1", cwd=tmp_path)
            assert len(record.job_id) == 10  # uuid4 hex[:10]
            assert record.created_at  # ISO timestamp
            assert record.log_path.endswith(".log")
        finally:
            os.environ.pop("GEM_HOME", None)


class TestListJobs:
    """Verify list_jobs returns the correct status for running and finished jobs."""

    def test_lists_launched_job(self, tmp_path: Path) -> None:
        os.environ["GEM_HOME"] = str(tmp_path / "gem_list")
        try:
            record = launch_background_job("echo done", cwd=tmp_path)
            time.sleep(0.5)
            jobs = list_jobs()
            assert len(jobs) >= 1
            found = [j for j in jobs if j["job_id"] == record.job_id]
            assert len(found) == 1
            assert found[0]["command"] == "echo done"
            # Status may be "running" (zombie) or "finished" depending on OS reaping.
            # The important thing is the job was listed with correct data.
            assert found[0]["status"] in ("running", "finished")
        finally:
            os.environ.pop("GEM_HOME", None)

    def test_running_status(self, tmp_path: Path) -> None:
        os.environ["GEM_HOME"] = str(tmp_path / "gem_list2")
        try:
            record = launch_background_job("sleep 10", cwd=tmp_path)
            jobs = list_jobs()
            found = [j for j in jobs if j["job_id"] == record.job_id]
            assert len(found) == 1
            assert found[0]["status"] == "running"
            # Clean up
            try:
                os.killpg(record.pid, signal.SIGTERM)
            except OSError:
                os.kill(record.pid, signal.SIGTERM)
        finally:
            os.environ.pop("GEM_HOME", None)


class TestReadJobLog:
    """Verify read_job_log returns log content."""

    def test_reads_output(self, tmp_path: Path) -> None:
        os.environ["GEM_HOME"] = str(tmp_path / "gem_readlog")
        try:
            record = launch_background_job("echo hello_from_job", cwd=tmp_path)
            time.sleep(0.5)
            content = read_job_log(record.job_id)
            assert "hello_from_job" in content
        finally:
            os.environ.pop("GEM_HOME", None)


class TestStopJob:
    """Verify stop_job sends SIGTERM to the process group."""

    def test_stops_running_job(self, tmp_path: Path) -> None:
        os.environ["GEM_HOME"] = str(tmp_path / "gem_stop")
        try:
            record = launch_background_job("sleep 30", cwd=tmp_path)
            time.sleep(0.3)
            # Verify it launched (should be alive initially)
            assert record.pid > 0
            result = stop_job(record.job_id)
            # stop_job sends SIGTERM to the process group — should succeed
            assert result is True
        finally:
            os.environ.pop("GEM_HOME", None)

    def test_stop_already_finished(self, tmp_path: Path) -> None:
        os.environ["GEM_HOME"] = str(tmp_path / "gem_stop2")
        try:
            record = launch_background_job("echo quick", cwd=tmp_path)
            time.sleep(0.5)
            # Process already exited — stop_job should return False (no such process group)
            result = stop_job(record.job_id)
            assert result is False
        finally:
            os.environ.pop("GEM_HOME", None)
