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
