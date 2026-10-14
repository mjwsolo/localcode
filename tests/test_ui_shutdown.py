import subprocess
from unittest.mock import Mock, call

from localcode.ui.launch import _model_alias, _shutdown_supervisor


def test_shutdown_waits_longer_than_the_model_grace_period():
    process = Mock()
    process.poll.return_value = None
    _shutdown_supervisor(process)
    assert process.method_calls == [call.poll(), call.terminate(), call.wait(timeout=35)]


def test_forced_supervisor_shutdown_is_reaped():
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired('supervisor', 35), 0]
    _shutdown_supervisor(process)
    assert process.method_calls == [call.poll(), call.terminate(), call.wait(timeout=35), call.kill(), call.wait(timeout=5)]


def test_launch_honors_full_path_without_copying_weights(tmp_path):
    cache, models = tmp_path / 'cache', tmp_path / 'models'
    cache.mkdir()
    weight = cache / 'custom.gguf'; weight.write_bytes(b'GGUF')
    assert _model_alias(str(weight), models) == 'custom'
    assert (models / weight.name).is_symlink()
    assert (models / weight.name).resolve() == weight.resolve()
