"""LOCALCODE_DISABLE_TOOLS removes named tools from the per-round surface."""
from localcode.tools import schemas_for_goal


def _names(schemas):
    return {s["function"]["name"] for s in schemas}


def test_default_surface_has_web_search():
    assert "web_search" in _names(schemas_for_goal("build_app"))


def test_env_removes_listed_tools(monkeypatch):
    monkeypatch.setenv("LOCALCODE_DISABLE_TOOLS", "web_search, agent")
    names = _names(schemas_for_goal("build_app"))
    assert "web_search" not in names and "agent" not in names
    assert "web_fetch" in names and "bash" in names
