"""A release gate must not be steered by the developer's own LocalCode state."""
import os
from pathlib import Path

from scripts import verify_models as gate

HOSTILE = """[runtime]
max_context_chars = 42000
llama_cpp_threads = 3
llama_cpp_batch_size = 64
llama_cpp_binary = "/nonexistent/llama-server"
internal_thinking_mode = "on"
"""


def command(gguf: Path) -> list[str]:
    return gate.server_command(str(gguf), 8123, "m")


def test_hostile_real_home_config_does_not_change_the_gate_command(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    gguf = models / "tiny.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 4096)
    monkeypatch.setenv("LOCALCODE_SERVER_KEY", "k")
    for k in [k for k in os.environ if k.startswith("LOCALCODE_") and k != "LOCALCODE_SERVER_KEY"]:
        monkeypatch.delenv(k)

    # Reference: a pristine home.
    clean = tmp_path / "clean"
    (clean / "state").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(clean))
    monkeypatch.setenv("LOCALCODE_HOME", str(clean / "state"))
    wanted = command(gguf)

    # The developer's home, with a config written to bend the launch.
    real = tmp_path / "real"
    (real / ".localcode").mkdir(parents=True)
    (real / ".localcode/config.toml").write_text(HOSTILE)
    monkeypatch.setenv("HOME", str(real))
    monkeypatch.setenv("LOCALCODE_HOME", str(real / ".localcode"))
    monkeypatch.setenv("LOCALCODE_MAX_CONTEXT_CHARS", "42000")
    monkeypatch.setenv("LOCALCODE_LLAMA_CPP_THREADS", "3")
    assert command(gguf) != wanted, "the hostile config must be able to change the command, or this test proves nothing"

    with gate.fresh_home(models):
        assert os.environ["HOME"] != str(real)
        assert "LOCALCODE_LLAMA_CPP_THREADS" not in os.environ
        assert Path(os.environ["LOCALCODE_MODEL_DIR"]) == models.resolve()
        with gate.environment({"LOCALCODE_SERVER_KEY": "k"}):
            isolated = command(gguf)
    assert isolated == wanted
    assert os.environ["HOME"] == str(real) and os.environ["LOCALCODE_LLAMA_CPP_THREADS"] == "3"
    assert (real / ".localcode/config.toml").read_text() == HOSTILE


def test_isolated_home_still_finds_models_in_the_real_directory(tmp_path):
    models = tmp_path / "elsewhere"
    models.mkdir()
    with gate.fresh_home(models):
        assert gate.catalog.model_dir() == models.resolve()
