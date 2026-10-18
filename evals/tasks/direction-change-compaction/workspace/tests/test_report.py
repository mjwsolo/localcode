import json

from app.report import load


def test_load(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([{"week": 1}]))
    assert load(p) == [{"week": 1}]
