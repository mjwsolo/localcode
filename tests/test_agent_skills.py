"""The bundled skills: well-formed, cheap to list, wired into every session, and
their helper scripts produce real files with nothing installed."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from localcode.ui.launch import bundled_skills_dir, write_config

SKILLS = sorted(p for p in bundled_skills_dir().iterdir() if p.is_dir() and not p.name.startswith("__"))


def _front(path: Path) -> dict:
    m = re.match(r"---\n(.*?)\n---\n", path.read_text(), re.S)
    assert m, f"{path} has no frontmatter"
    return dict(line.split(": ", 1) for line in m.group(1).splitlines())


def test_each_skill_is_named_after_its_folder_with_a_short_description():
    assert {p.name for p in SKILLS} >= {"word-documents", "pdf-documents", "spreadsheets", "images", "audio-video"}
    total = 0
    for p in SKILLS:
        front = _front(p / "SKILL.md")
        assert front["name"] == p.name
        # Descriptions are in every request: keep each to a sentence or two.
        assert 40 <= len(front["description"]) <= 260, (p.name, len(front["description"]))
        total += len(front["description"])
        assert len((p / "SKILL.md").read_text()) < 2500, p.name
    assert total <= 1200, total


def test_sessions_load_the_bundled_skills_and_may_run_their_scripts(tmp_path):
    cfg = tmp_path / "s.json"
    write_config(cfg, port=1, ctx=32768, alias="m")
    data = json.loads(cfg.read_text())
    assert data["skills"]["paths"] == [str(bundled_skills_dir())]
    allowed = [r["resource"] for r in data["permissions"] if r["action"] == "external_directory" and r["effect"] == "allow"]
    assert str(bundled_skills_dir() / "*") in allowed


def test_xlsx_script_writes_a_workbook(tmp_path):
    (tmp_path / "Scores.csv").write_text('Name,Score\nAda,91.5\n"Bo, Jr",77\n')
    out = tmp_path / "out.xlsx"
    subprocess.run([sys.executable, str(bundled_skills_dir() / "spreadsheets/make_xlsx.py"), str(out), str(tmp_path / "Scores.csv")], check=True)
    z = zipfile.ZipFile(out)
    assert 'name="Scores"' in z.read("xl/workbook.xml").decode()
    sheet = z.read("xl/worksheets/sheet1.xml").decode()
    assert "<v>91.5</v>" in sheet and "Bo, Jr" in sheet


def test_pdf_script_writes_a_pdf_with_pages(tmp_path):
    md = tmp_path / "in.md"
    md.write_text("# Title\n\nSome (text).\n\n" + "\n".join(f"- item {i}" for i in range(120)))
    out = tmp_path / "out.pdf"
    subprocess.run([sys.executable, str(bundled_skills_dir() / "pdf-documents/make_pdf.py"), str(md), str(out)], check=True)
    data = out.read_bytes()
    assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
    assert data.count(b"/Type /Page ") >= 2


def test_package_installs_always_ask(tmp_path):
    import fnmatch
    cfg = tmp_path / "s.json"
    write_config(cfg, port=1, ctx=32768, alias="m")
    rules = json.loads(cfg.read_text())["permissions"]
    asks = [r["resource"] for r in rules if r["action"] == "shell" and r["effect"] == "ask"]
    # The install rules come last, so they win over any earlier shell rule.
    assert rules[-len(asks):] == [r for r in rules if r["action"] == "shell" and r["effect"] == "ask"]
    for cmd in ["pip3 install python-docx", "/usr/bin/python3 -m pip install python-docx", "brew install ffmpeg", "npm i -g serve"]:
        assert any(fnmatch.fnmatchcase(cmd, a) for a in asks), cmd
    for cmd in ["npm install", "python3 make.py", "uv sync"]:
        assert not any(fnmatch.fnmatchcase(cmd, a) for a in asks), cmd


def test_settings_folder_is_not_preallowed_and_formatters_are_off(tmp_path):
    cfg = tmp_path / "s.json"
    write_config(cfg, port=1, ctx=32768, alias="m")
    data = json.loads(cfg.read_text())
    allowed = [r["resource"] for r in data["permissions"] if r["action"] == "external_directory" and r["effect"] == "allow"]
    assert not any("localcode-agent" in a and ".config" in a for a in allowed), allowed
    assert data["formatter"] is False
