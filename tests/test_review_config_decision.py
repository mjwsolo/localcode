"""A mistyped config section and a non-UTF-8 state file fail cleanly."""
from localcode import decision
from localcode.config import get_config_path, load_config


def test_mistyped_section_falls_back_and_names_the_section(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path))
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('runtime = "foo"\nui = 3\n')
    cfg = load_config()
    assert cfg.runtime.provider
    err = capsys.readouterr().err
    assert "[runtime]" in err and str(path) in err


def test_decide_reports_a_non_utf8_state_file(tmp_path, capsys):
    bad = tmp_path / "state.bin"
    bad.write_bytes(b"\xff\xfe\x00bad")
    assert decision.main(["--question", "ok?", "--state-file", str(bad)]) == 1
    err = capsys.readouterr().err
    assert str(bad) in err and "UTF-8" in err
