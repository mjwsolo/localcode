from typer.testing import CliRunner
from cli import app

runner = CliRunner()


def test_greet():
    result = runner.invoke(app, ["greet", "--name", "alice"])
    assert result.exit_code == 0
    assert "hello, alice" in result.stdout
