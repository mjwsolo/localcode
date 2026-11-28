import json
import os
from pathlib import Path

import pytest

from localcode import importer

SECRET = "sk-SECRETVALUE123"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    h = tmp_path / "home"
    h.mkdir()
    return h


def cfg_path(h):
    return h / ".config" / "localcode-agent" / "localcode.json"


def write_claude(h, servers, extra=None):
    data = dict(extra or {})
    data["mcpServers"] = servers
    (h / ".claude.json").write_text(json.dumps(data))


def run(h, argv, answers=()):
    lines, ans = [], list(answers)
    code = importer.cli_main(argv, home=h, input_fn=lambda _p: ans.pop(0), out=lines.append)
    return code, "\n".join(lines)


def test_claude_mapping_local_and_remote(home):
    write_claude(home, {
        "fs": {"type": "stdio", "command": "npx", "args": ["-y", "srv", "."], "env": {"API_KEY": SECRET}},
        "gh": {"type": "http", "url": "https://x.example/mcp", "headers": {"Authorization": f"Bearer {SECRET}"}},
        "old": {"type": "sse", "url": "https://y.example/sse"},
    })
    det = importer.detect(home)
    by = {i.name: i.spec for i in det.items}
    assert by["fs"] == {"type": "local", "command": ["npx", "-y", "srv", "."], "environment": {"API_KEY": SECRET}, "enabled": False}
    assert by["gh"]["type"] == "remote" and by["gh"]["headers"]["Authorization"].endswith(SECRET)
    assert by["old"] == {"type": "remote", "url": "https://y.example/sse", "enabled": False}


def test_codex_toml(home):
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text(
        '[mcp_servers.ctx]\ncommand = "uvx"\nargs = ["ctx"]\n[mcp_servers.ctx.env]\nTOKEN = "abc"\n'
        '[mcp_servers.web]\nurl = "https://w.example/mcp"\n[mcp_servers.web.http_headers]\nX-Key = "k"\n')
    by = {i.name: i.spec for i in importer.detect(home).items}
    assert by["ctx"]["command"] == ["uvx", "ctx"] and by["ctx"]["environment"] == {"TOKEN": "abc"}
    assert by["web"] == {"type": "remote", "url": "https://w.example/mcp", "headers": {"X-Key": "k"}, "enabled": False}


def test_cursor(home):
    (home / ".cursor").mkdir()
    (home / ".cursor" / "mcp.json").write_text(json.dumps({"mcpServers": {"c": {"command": "node", "args": ["a.js"]}}}))
    items = importer.detect(home).items
    assert [(i.source, i.name) for i in items] == [("cursor", "c")]


def test_malformed_and_empty_and_missing(home):
    (home / ".claude.json").write_text("{not json")
    (home / ".cursor").mkdir()
    (home / ".cursor" / "mcp.json").write_text("")
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text("= = =")
    det = importer.detect(home)
    assert det.items == [] and len(det.notes) == 3
    code, text = run(home, ["--yes"])
    assert code == 0 and "Nothing to import" in text


def test_unsupported_entries_noted(home):
    write_claude(home, {"bad": {"type": "weird"}, "str": "x", "ok": {"command": "a"}})
    det = importer.detect(home)
    assert [i.name for i in det.items] == ["ok"] and len(det.notes) == 2


def test_huge_claude_json_only_mcpservers_used(home):
    extra = {"projects": {f"/p{i}": {"history": ["x" * 100] * 20, "mcpServers": {"proj": {"command": "evil"}}} for i in range(2000)},
             "oauthAccount": {"token": "TOPSECRET"}}
    write_claude(home, {"one": {"command": "a"}}, extra)
    assert (home / ".claude.json").stat().st_size > 3_000_000
    det = importer.detect(home)
    assert [i.name for i in det.items] == ["one"]


def test_no_project_mcp_json(home, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".mcp.json").write_text(json.dumps({"mcpServers": {"p": {"command": "x"}}}))
    monkeypatch.chdir(proj)
    assert importer.detect(home).items == []


def make_skill(home, name, desc="Does a thing", script=False):
    d = home / ".claude" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\nbody\n")
    if script:
        (d / "scripts").mkdir()
        (d / "scripts" / "run.sh").write_text("echo hi\n")
    return d


def test_skill_detect_and_copy(home):
    make_skill(home, "plain")
    make_skill(home, "risky", script=True)
    (home / ".claude" / "skills" / "notaskill").mkdir()
    items = {i.name: i for i in importer.detect(home).items}
    assert set(items) == {"plain", "risky"}
    assert items["risky"].has_scripts and not items["plain"].has_scripts
    assert items["plain"].size > 0 and items["plain"].description == "Does a thing"
    code, text = run(home, ["--yes"])
    assert code == 0 and "CONTAINS SCRIPTS" in text
    assert (importer.skills_dir(home) / "risky" / "scripts" / "run.sh").is_file()
    assert (home / ".claude" / "skills" / "risky").is_dir()  # source untouched


def test_dry_run_writes_nothing(home):
    write_claude(home, {"a": {"command": "x"}})
    make_skill(home, "s")
    code, text = run(home, ["--dry-run"])
    assert code == 0 and "dry run" in text
    assert not cfg_path(home).exists() and not importer.skills_dir(home).exists()


def test_n_writes_nothing(home):
    write_claude(home, {"a": {"command": "x"}})
    code, text = run(home, [], ["all", "n"])
    assert code == 0 and "No changes made" in text
    assert not cfg_path(home).exists()
    code, _ = run(home, [], [""])
    assert not cfg_path(home).exists()


def test_import_disabled_backup_and_preserves(home):
    write_claude(home, {"a": {"command": "x", "args": ["1"]}, "b": {"url": "https://b.example"}})
    p = cfg_path(home)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"model": "keep", "mcp": {"mine": {"type": "local", "command": ["m"], "enabled": True}}}))
    code, text = run(home, [], ["1", "y"])
    assert code == 0
    data = json.loads(p.read_text())
    assert data["model"] == "keep" and data["mcp"]["mine"]["enabled"] is True
    assert data["mcp"]["a"]["enabled"] is False and "b" not in data["mcp"]
    baks = list(p.parent.glob("localcode.json.bak-*"))
    assert len(baks) == 1 and json.loads(baks[0].read_text())["mcp"].keys() == {"mine"}
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    assert not list(p.parent.glob("*.tmp"))


def test_clash_skipped_with_suggestion(home):
    write_claude(home, {"a": {"command": "new"}})
    p = cfg_path(home)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"mcp": {"a": {"type": "local", "command": ["old"]}}}))
    code, text = run(home, ["--yes"])
    assert "try 'a-claude'" in text and "Nothing new" in text
    assert json.loads(p.read_text())["mcp"]["a"]["command"] == ["old"]


def test_idempotent(home):
    write_claude(home, {"a": {"command": "x"}})
    make_skill(home, "s")
    run(home, ["--yes"])
    first = cfg_path(home).read_text()
    # user enabled it meanwhile; re-run must not touch or clash
    d = json.loads(first)
    d["mcp"]["a"]["enabled"] = True
    cfg_path(home).write_text(json.dumps(d))
    code, text = run(home, ["--yes"])
    assert "already imported" in text and "Nothing new" in text
    assert json.loads(cfg_path(home).read_text())["mcp"]["a"]["enabled"] is True


def test_mcp_main_dispatches_import(home, monkeypatch, capsys):
    from localcode.ui import launch
    monkeypatch.setenv("HOME", str(home))
    assert launch.mcp_main(["import", "--help"]) == 0
    assert "--dry-run" in capsys.readouterr().out


def test_secrets_never_printed(home):
    write_claude(home, {
        "a": {"command": "srv", "args": ["--api-key", SECRET, f"--token={SECRET}", "ok"], "env": {"TOK": SECRET}},
        "b": {"url": f"https://u:{SECRET}@h.example/mcp?api_key={SECRET}&x=1", "headers": {"Authorization": SECRET}},
    })
    code, text = run(home, [], ["all", "n"])
    assert SECRET not in text
    assert "TOK" in text and "Authorization" in text and "srv --api-key *** --token=*** ok" in text
    assert "two places" in text
    code, text = run(home, ["--dry-run"])
    assert SECRET not in text
    assert "will run on your Mac when you enable it" in text


def test_invalid_user_config_untouched(home):
    write_claude(home, {"a": {"command": "x"}})
    p = cfg_path(home)
    p.parent.mkdir(parents=True)
    p.write_text("{broken")
    code, text = run(home, ["--yes"])
    assert code == 1 and p.read_text() == "{broken"


def test_from_filter_and_bad_args(home):
    write_claude(home, {"a": {"command": "x"}})
    (home / ".cursor").mkdir()
    (home / ".cursor" / "mcp.json").write_text(json.dumps({"mcpServers": {"c": {"command": "y"}}}))
    _, text = run(home, ["--from", "cursor", "--dry-run"])
    assert "MCP server c" in text and "MCP server a" not in text
    code, _ = run(home, ["--from", "bogus"])
    assert code == 2


def test_symlinked_skill_dir_and_inner_links_not_followed(home, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "SKILL.md").write_text("---\ndescription: d\n---\n")
    (tmp_path / "outside.txt").write_text("private")
    os.symlink(tmp_path / "outside.txt", real / "leak.txt")
    sk = home / ".claude" / "skills"
    sk.mkdir(parents=True)
    os.symlink(real, sk / "linked")
    run(home, ["--yes"])
    dest = importer.skills_dir(home) / "linked"
    assert (dest / "SKILL.md").is_file() and not (dest / "leak.txt").exists()


# ---------------------------------------------------------------- supervisor endpoints

@pytest.fixture
def sup(home, monkeypatch, tmp_path):
    from localcode.ui import supervisor
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr("localcode.paths.GLOBAL_STATE_DIR", tmp_path / "state", raising=False)
    return supervisor.Supervisor.__new__(supervisor.Supervisor)


def test_detect_endpoint_masks_secrets(sup, home):
    write_claude(home, {"fs": {"command": "npx", "args": ["srv", "--api-key", SECRET], "env": {"TOKEN": SECRET}},
                        "gh": {"type": "http", "url": "https://x.example/mcp?token=" + SECRET, "headers": {"Authorization": SECRET}}})
    skill = home / ".claude" / "skills" / "tidy"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\ndescription: tidy things\n---\nbody")
    (skill / "run.sh").write_text("echo hi")
    out = sup.import_detect()
    assert SECRET not in json.dumps(out)
    by = {i["name"]: i for i in out["items"]}
    assert by["fs"]["command"] == "npx srv --api-key ***" and by["fs"]["env_names"] == ["TOKEN"]
    assert by["fs"]["label"] == "Claude Code" and by["gh"]["header_names"] == ["Authorization"]
    assert by["tidy"]["has_scripts"] and by["tidy"]["tokens"] > 0 and by["tidy"]["size"]
    assert by["fs"]["status"] == "new"


def test_apply_endpoint_writes_disabled_and_reports_clash(sup, home):
    write_claude(home, {"fs": {"command": "npx", "args": ["srv"]}})
    ids = [i["id"] for i in sup.import_detect()["items"]]
    res = sup.import_apply(ids)
    assert res["ok"] and res["servers"] == ["fs"] and res["backup"] is None
    assert json.loads(cfg_path(home).read_text())["mcp"]["fs"]["enabled"] is False
    again = sup.import_apply(ids)
    assert again["servers"] == []                       # already there, nothing rewritten
    assert "error" in sup.import_apply(["99"]) and "error" in sup.import_apply([])
    write_claude(home, {"fs": {"command": "other"}})
    res = sup.import_apply([sup.import_detect()["items"][0]["id"]])
    assert res["servers"] == ["fs-claude"]
    assert res["backup"].startswith("~/") and str(home) not in res["backup"]


def test_import_hint_checks_existence_only_and_shows_once(sup, home, monkeypatch):
    assert sup.import_hint() is None                    # nothing to find
    (home / ".claude.json").write_text("{}")
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text("")
    os.chmod(home / ".claude.json", 0)
    os.chmod(home / ".codex" / "config.toml", 0)
    opened = []
    real_open = open
    monkeypatch.setattr("builtins.open", lambda *a, **k: (opened.append(a[0]), real_open(*a, **k))[1])
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: opened.append("read_text") or "")
    try:
        msg = sup.import_hint()
    finally:
        os.chmod(home / ".claude.json", 0o600)
        os.chmod(home / ".codex" / "config.toml", 0o600)
    assert msg == "Found settings from Claude Code (and Codex). Type /import to bring over your MCP servers and skills."
    assert not [o for o in opened if "claude.json" in str(o) or "config.toml" in str(o)]
    assert sup.import_hint() is None                    # shown once


def test_reserved_server_names_preserve_mcp_settings(home):
    write_claude(home, {name: {"command": "new"} for name in ("servers", "timeout", "ordinary")})
    path = cfg_path(home)
    path.parent.mkdir(parents=True)
    nested = {"existing": {"type": "local", "command": ["original"]}}
    path.write_text(json.dumps({"mcp": {"servers": nested, "timeout": 42}}))
    assert run(home, ["--yes"])[0] == 0
    config = json.loads(path.read_text())["mcp"]
    assert config["servers"] == nested and config["timeout"] == 42
    assert config["ordinary"]["enabled"] is False


def test_skill_import_preserves_existing_sibling(home):
    make_skill(home, "demo")
    sibling = importer.skills_dir(home) / ".demo.importing"
    sibling.mkdir(parents=True)
    (sibling / "keep").write_text("original")
    assert run(home, ["--yes"])[0] == 0
    assert (sibling / "keep").read_text() == "original"
    assert (importer.skills_dir(home) / "demo" / "SKILL.md").is_file()
