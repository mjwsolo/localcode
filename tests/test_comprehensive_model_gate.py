"""Contract failures which the local promotion gate must reject."""
import json
import subprocess
from unittest.mock import Mock

import pytest

from scripts import verify_models as gate


def test_tool_loop_rejects_text_instead_of_a_call():
    with pytest.raises(AssertionError, match="no tool call"):
        gate.validate_tool({"content": "I would use get_weather"})


@pytest.mark.parametrize("arguments", ['{"city":"London"}', '{}', 'not JSON'])
def test_tool_loop_rejects_wrong_or_invalid_arguments(arguments):
    with pytest.raises((AssertionError, json.JSONDecodeError)):
        gate.validate_tool({"tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": arguments}}]})


def test_decision_gate_rejects_nonfinite_probabilities():
    data = {"answers": {"route": {"choice": "billing", "probabilities": {"billing": float("nan"), "shipping": 0, "technical": 0}}, "angry": {"noul": 0.5}, "urgency": {"score": 1}}}
    with pytest.raises(AssertionError):
        gate.validate_decisions(data)


def test_missing_required_models_fail_instead_of_skipping(monkeypatch):
    monkeypatch.setattr(gate.catalog, "CHOICES", [])
    with pytest.raises(SystemExit) as exc:
        gate.main(["--strict"])
    assert exc.value.code == 2


def test_environment_restores_credentials(monkeypatch):
    monkeypatch.setenv(gate.auth.KEY_ENV, "original")
    with gate.environment({gate.auth.KEY_ENV: "temporary"}):
        assert gate.auth.server_key() == "temporary"
    assert gate.auth.server_key() == "original"


def test_failed_startup_reaps_the_process(monkeypatch, tmp_path):
    choice = Mock(key="gemma", local_path=tmp_path / "model.gguf")
    proc = Mock()
    proc.poll.side_effect = [1, None]
    monkeypatch.setattr(gate, "server_command", lambda *a: ["llama-server", "--ctx-size", "4096"])
    monkeypatch.setattr(gate.subprocess, "Popen", lambda *a, **kw: proc)
    with pytest.raises(RuntimeError, match="exited during model load"):
        with gate.Server(choice, tmp_path):
            pass
    proc.terminate.assert_called_once()
    proc.wait.assert_called_once_with(timeout=15)


def test_shutdown_kills_then_waits_after_timeout(tmp_path):
    server = gate.Server(Mock(), tmp_path)
    server.proc = Mock()
    server.proc.poll.return_value = None
    server.proc.wait.side_effect = [subprocess.TimeoutExpired("server", 15), 0]
    server.__exit__(None, None, None)
    server.proc.kill.assert_called_once()
    assert server.proc.wait.call_count == 2


@pytest.mark.parametrize("mutation", ["commit", "binaries", "coverage", "runtime", "decisions", "turbo"])
def test_invalid_receipts_cannot_publish_a_promotion(mutation):
    rows = [{"key": key, "turbo4": False, "checks": ["auth", "template", "chat", "tool-loop", "systemone", "decision-control", "runtime"]} for key in gate.REQUIRED_KEYS]
    rows.append({"key": "qwen", "turbo4": True, "checks": ["tool-loop"]})
    receipt = {"commit": "candidate", "dirty": False, "strict": True, "success": True, "binaries": {"server": "hash"}, "models": rows}
    gate.validate_receipt(receipt, "candidate", {"server": "hash"}, gate.REQUIRED_KEYS)
    if mutation == "commit": receipt["commit"] = "old"
    elif mutation == "binaries": receipt["binaries"] = {"server": "old"}
    elif mutation == "coverage": receipt["models"] = rows[1:]
    elif mutation == "runtime": next(r for r in rows if r["key"] == "openjev")["checks"].remove("runtime")
    elif mutation == "decisions": next(r for r in rows if r["key"] == "openjev")["checks"].remove("decision-control")
    else: receipt["models"] = rows[:-1]
    with pytest.raises(AssertionError):
        gate.validate_receipt(receipt, "candidate", {"server": "hash"}, gate.REQUIRED_KEYS)
