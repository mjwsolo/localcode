"""Import MCP servers and skills from other coding agents.

Pure logic, no network and no subprocess. Other apps' config files are read
only when the user runs `localcode mcp import`, and only their user-level
files (never a project's .mcp.json). Imported servers are always written
disabled; skills are copied.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

SOURCES = ("claude", "codex", "cursor")
SCRIPT_SUFFIXES = {".sh", ".bash", ".zsh", ".py", ".js", ".mjs", ".ts", ".rb", ".pl", ".ps1", ".command"}
_SECRET_NAME = re.compile(r"(token|secret|password|passwd|api[-_]?key|apikey|auth|credential|bearer|private[-_]?key)", re.I)


@dataclass
class Item:
    kind: str                      # "mcp" | "skill"
    source: str                    # claude | codex | cursor
    name: str
    spec: dict = field(default_factory=dict)     # mcp: opencode-shaped server config
    path: Path | None = None                     # skill: source folder
    description: str = ""
    size: int = 0
    has_scripts: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class Detection:
    items: list[Item] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # unreadable files, skipped entries


@dataclass
class Step:
    item: Item
    status: str                    # new | exists | clash
    final_name: str
    suggestion: str = ""


def user_config_dir(home: Path | None = None) -> Path:
    home = home or Path.home()
    return Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config") / "localcode-agent"


def user_config_file(home: Path | None = None) -> Path:
    d = user_config_dir(home)
    for name in ("localcode.json", "localcode.jsonc"):
        if (d / name).is_file():
            return d / name
    return d / "localcode.json"


def skills_dir(home: Path | None = None) -> Path:
    return user_config_dir(home) / "skills"


# ---------------------------------------------------------------- mapping

def _str_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if isinstance(v, (str, int, float, bool))}


def map_server(raw: object) -> tuple[dict | None, str]:
    """One other-agent server entry -> opencode shape (disabled), or (None, reason)."""
    if not isinstance(raw, dict):
        return None, "not an object"
    url = raw.get("url") or raw.get("httpUrl") or raw.get("serverUrl")
    kind = str(raw.get("type") or "").lower()
    if isinstance(url, str) and url and kind in ("", "http", "sse", "streamable-http", "streamable_http", "remote"):
        spec: dict = {"type": "remote", "url": url, "enabled": False}
        headers = _str_map(raw.get("headers") or raw.get("http_headers"))
        if headers:
            spec["headers"] = headers
        return spec, ""
    cmd = raw.get("command")
    if kind in ("", "stdio", "local") and cmd:
        if isinstance(cmd, list):
            argv = [str(c) for c in cmd]
        elif isinstance(cmd, str):
            argv = [cmd]
        else:
            return None, "unsupported command"
        args = raw.get("args")
        if isinstance(args, list):
            argv += [str(a) for a in args]
        spec = {"type": "local", "command": argv, "enabled": False}
        env = _str_map(raw.get("env") or raw.get("environment"))
        if env:
            spec["environment"] = env
        return spec, ""
    return None, f"unsupported server type {kind or 'unknown'}"


def _read_json(path: Path, notes: list[str]) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeDecodeError):
        notes.append(f"{path.name}: could not be read as JSON, skipped")
        return None
    return data if isinstance(data, dict) else None


def _servers_to_items(source: str, servers: object, notes: list[str], label: str) -> list[Item]:
    out: list[Item] = []
    if not isinstance(servers, dict):
        return out
    for name, raw in servers.items():
        spec, why = map_server(raw)
        if spec is None:
            notes.append(f"{label}: server '{name}' skipped ({why})")
            continue
        item = Item("mcp", source, str(name), spec=spec)
        if isinstance(raw, dict) and raw.get("bearer_token_env_var"):
            item.warnings.append("uses a bearer token from an environment variable; set the header yourself after importing")
        out.append(item)
    return out


# ---------------------------------------------------------------- skills

def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    out: dict[str, str] = {}
    for line in text[3:end].splitlines():
        m = re.match(r"^([A-Za-z_-]+):\s*(.*)$", line)
        if m:
            out[m.group(1).lower()] = m.group(2).strip().strip("\"'")
    return out


def _skill_item(folder: Path) -> Item | None:
    md = folder / "SKILL.md"
    if not md.is_file():
        return None
    try:
        text = md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fm = _frontmatter(text)
    size, scripts = 0, False
    for p in folder.rglob("*"):
        if p.is_symlink() or not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        size += st.st_size
        if p.suffix.lower() in SCRIPT_SUFFIXES or (st.st_mode & 0o111) or "scripts" in p.relative_to(folder).parts[:-1]:
            scripts = True
    return Item("skill", "claude", folder.name, path=folder, description=fm.get("description", ""),
                size=size, has_scripts=scripts)


def description_tokens(desc: str, name: str = "") -> int:
    """Rough cost of the skill's listing line in every request (~4 chars per token)."""
    return max(1, (len(desc) + len(name) + 40) // 4)


# ---------------------------------------------------------------- detect

def detect(home: Path | None = None, sources: tuple[str, ...] = SOURCES) -> Detection:
    home = home or Path.home()
    det = Detection()
    if "claude" in sources:
        # Only the top-level mcpServers key is used; the rest of the file
        # (history, projects, credentials) is parsed but never read.
        data = _read_json(home / ".claude.json", det.notes)
        if data is not None:
            det.items += _servers_to_items("claude", data.get("mcpServers"), det.notes, ".claude.json")
            del data
        sk = home / ".claude" / "skills"
        if sk.is_dir():
            for d in sorted(sk.iterdir()):
                if d.is_dir():
                    it = _skill_item(d)
                    if it:
                        det.items.append(it)
    if "codex" in sources:
        p = home / ".codex" / "config.toml"
        if p.is_file():
            try:
                data = tomllib.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeDecodeError):
                det.notes.append("config.toml: could not be read as TOML, skipped")
            else:
                det.items += _servers_to_items("codex", data.get("mcp_servers"), det.notes, "codex config.toml")
    if "cursor" in sources:
        data = _read_json(home / ".cursor" / "mcp.json", det.notes)
        if data is not None:
            det.items += _servers_to_items("cursor", data.get("mcpServers"), det.notes, "cursor mcp.json")
    return det


# ---------------------------------------------------------------- config io

def load_user_config(home: Path | None = None) -> tuple[dict, str]:
    """(config, error). An absent file is an empty config."""
    f = user_config_file(home)
    if not f.is_file():
        return {}, ""
    try:
        text = f.read_text(encoding="utf-8")
    except OSError as e:
        return {}, f"could not read {f.name}: {e.strerror or e}"
    try:
        data = json.loads(text)
    except ValueError:
        try:
            from .ui.launch import _strip_jsonc_comments
            data = json.loads(_strip_jsonc_comments(text))
        except Exception:
            return {}, f"{f.name} is not valid JSON; fix it first, nothing was changed"
    if not isinstance(data, dict):
        return {}, f"{f.name} is not a JSON object; nothing was changed"
    return data, ""


def _existing_servers(cfg: dict) -> dict[str, dict]:
    mcp = cfg.get("mcp")
    out: dict[str, dict] = {}
    if isinstance(mcp, dict):
        if isinstance(mcp.get("servers"), dict):
            out.update({k: v for k, v in mcp["servers"].items() if isinstance(v, dict)})
        out.update({k: v for k, v in mcp.items() if k not in ("servers", "timeout") and isinstance(v, dict)})
    return out


def _same_server(a: dict, b: dict) -> bool:
    def strip(d: dict) -> dict:
        return {k: v for k, v in d.items() if k != "enabled"}
    return strip(a) == strip(b)


def plan(items: list[Item], home: Path | None = None) -> list[Step]:
    cfg, _ = load_user_config(home)
    existing = _existing_servers(cfg)
    taken = set(existing)
    sdir = skills_dir(home)
    steps: list[Step] = []
    for it in items:
        if it.kind == "mcp":
            cur = existing.get(it.name)
            if cur is not None and _same_server(cur, it.spec):
                steps.append(Step(it, "exists", it.name))
            elif it.name not in taken:
                steps.append(Step(it, "new", it.name))
                taken.add(it.name)
            else:
                n = f"{it.name}-{it.source}"
                k = 2
                while n in taken:
                    n = f"{it.name}-{it.source}{k}"
                    k += 1
                steps.append(Step(it, "clash", it.name, suggestion=n))
        elif (sdir / it.name).exists():
            steps.append(Step(it, "exists", it.name))
        else:
            steps.append(Step(it, "new", it.name))
    return steps


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.chmod(tmp, 0o600)  # imported servers can carry tokens
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _ignore_links(directory: str, names: list[str]) -> list[str]:
    return [n for n in names if os.path.islink(os.path.join(directory, n))]


def apply(steps: list[Step], home: Path | None = None) -> dict:
    """Write the `new` steps. Returns {"servers", "skills", "backup", "error"}."""
    result: dict = {"servers": [], "skills": [], "backup": None, "error": ""}
    todo = [s for s in steps if s.status == "new"]
    servers = [s for s in todo if s.item.kind == "mcp"]
    if servers:
        cfg, err = load_user_config(home)
        if err:
            result["error"] = err
            return result
        f = user_config_file(home)
        if f.is_file():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            bak = f.with_name(f"{f.name}.bak-{stamp}")
            n = 1
            while bak.exists():
                n += 1
                bak = f.with_name(f"{f.name}.bak-{stamp}-{n}")
            shutil.copy2(f, bak)
            os.chmod(bak, 0o600)
            result["backup"] = str(bak)
        mcp = cfg.get("mcp")
        if not isinstance(mcp, dict):
            mcp = {}
        for s in servers:
            spec = copy.deepcopy(s.item.spec)
            spec["enabled"] = False
            mcp[s.final_name] = spec
            result["servers"].append(s.final_name)
        cfg["mcp"] = mcp
        _atomic_write_json(f, cfg)
    for s in todo:
        if s.item.kind != "skill" or s.item.path is None:
            continue
        dest = skills_dir(home) / s.final_name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.importing")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(s.item.path.resolve(), tmp, ignore=_ignore_links)
        os.replace(tmp, dest)
        result["skills"].append(s.final_name)
    return result


# ---------------------------------------------------------------- display

def mask_args(argv: list[str]) -> list[str]:
    """The command line as it will run, with values of secret-looking flags hidden."""
    out: list[str] = []
    hide_next = False
    for a in argv:
        if hide_next:
            out.append("***")
            hide_next = False
        elif a.startswith("-") and "=" in a:
            k = a.partition("=")[0]
            out.append(f"{k}=***" if _SECRET_NAME.search(k) else a)
        elif a.startswith("-") and _SECRET_NAME.search(a):
            out.append(a)
            hide_next = True
        elif "=" in a and not a.startswith("/") and _SECRET_NAME.search(a.partition("=")[0]):
            out.append(f"{a.partition('=')[0]}=***")
        else:
            out.append(a)
    return out


def mask_url(url: str) -> str:
    url = re.sub(r"//[^/@\s]+@", "//***@", url)
    return re.sub(r"([?&][^=&#]*)=([^&#]*)",
                  lambda m: f"{m.group(1)}=***" if _SECRET_NAME.search(m.group(1)) else m.group(0), url)


def _quote(a: str) -> str:
    return a if re.fullmatch(r"[\w@%+=:,./*-]+", a) else json.dumps(a)


def describe_server(spec: dict) -> list[str]:
    lines: list[str] = []
    if spec.get("type") == "local":
        cmd = " ".join(_quote(a) for a in mask_args([str(c) for c in spec.get("command", [])]))
        lines.append("will run on your Mac when you enable it: " + cmd)
        if spec.get("environment"):
            lines.append("environment variables (names only): " + ", ".join(sorted(spec["environment"])))
    else:
        lines.append("connects to: " + mask_url(str(spec.get("url", ""))))
        if spec.get("headers"):
            lines.append("headers (names only): " + ", ".join(sorted(spec["headers"])))
    return lines


def _human(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n >= 1024 else f"{n} B"


# ---------------------------------------------------------------- cli

IMPORT_HELP = """usage: localcode mcp import [--from claude|codex|cursor] [--dry-run] [--yes]

Copy MCP servers and skills you already use in other coding agents into
localcode. Reads only user-level files, only now, because you ran this:
  Claude Code   ~/.claude.json (mcpServers) and ~/.claude/skills
  Codex         ~/.codex/config.toml ([mcp_servers.*])
  Cursor        ~/.cursor/mcp.json
Imported servers start disabled; nothing runs until you enable one in /mcps.
Skills are copied into ~/.config/localcode-agent/skills.
  --dry-run   show what would be imported and write nothing
  --yes       import everything new without asking"""


def _parse_selection(text: str, n: int) -> list[int] | None:
    text = text.strip().lower()
    if text in ("", "none", "n"):
        return []
    if text in ("a", "all"):
        return list(range(n))
    picked: set[int] = set()
    for part in text.replace(" ", ",").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            return None
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if lo < 1 or hi > n or lo > hi:
            return None
        picked.update(range(lo - 1, hi))
    return sorted(picked)


def cli_main(argv: list[str], home: Path | None = None,
             input_fn: Callable[[str], str] = input, out: Callable[[str], None] = print) -> int:
    args = list(argv)
    if any(a in ("-h", "--help", "help") for a in args):
        out(IMPORT_HELP)
        return 0
    dry = "--dry-run" in args
    yes = "--yes" in args
    args = [a for a in args if a not in ("--dry-run", "--yes")]
    sources = SOURCES
    if args:
        if args[0] == "--from" and len(args) == 2 and args[1] in SOURCES:
            sources = (args[1],)
        elif len(args) == 1 and args[0].startswith("--from=") and args[0][7:] in SOURCES:
            sources = (args[0][7:],)
        else:
            out(IMPORT_HELP)
            return 2
    det = detect(home, sources)
    for note in det.notes:
        out(f"note: {note}")
    steps = plan(det.items, home)
    if not steps:
        out("Nothing to import: no MCP servers or skills found in the user config of " + ", ".join(sources) + ".")
        return 0

    out("Found in other agents' user-level config:")
    for i, s in enumerate(steps, 1):
        it = s.item
        tag = {"new": "", "exists": "  (already imported)",
               "clash": f"  (name taken by a different server; try '{s.suggestion}')"}[s.status]
        if it.kind == "mcp":
            out(f"  {i}. [{it.source}] MCP server {it.name} ({it.spec['type']}){tag}")
        else:
            extra = f"{_human(it.size)}, about {description_tokens(it.description, it.name)} tokens in every request"
            if it.has_scripts:
                extra += ", CONTAINS SCRIPTS"
            out(f"  {i}. [{it.source}] skill {it.name} ({extra}){tag}")
    importable = [i for i, s in enumerate(steps) if s.status == "new"]
    if not importable:
        out("Nothing new to import.")
        return 0
    if dry or yes:
        chosen = importable
    else:
        while True:
            sel = _parse_selection(input_fn("Import which? (numbers like 1,3 or 2-4, 'all', Enter for none): "), len(steps))
            if sel is not None:
                break
            out("Could not read that; try again.")
        chosen = [i for i in sel if i in importable]
        skipped = [i + 1 for i in sel if i not in importable]
        if skipped:
            out(f"Skipping already imported or name-clashing items: {', '.join(map(str, skipped))}")
    if not chosen:
        out("Nothing selected. No changes made.")
        return 0

    picked = [steps[i] for i in chosen]
    out("")
    out("Summary" + (" (dry run, nothing will be written)" if dry else "") + ":")
    for s in picked:
        it = s.item
        if it.kind == "mcp":
            out(f"  MCP server {s.final_name} from {it.source}, imported disabled")
            for line in describe_server(it.spec):
                out(f"    {line}")
            for w in it.warnings:
                out(f"    warning: {w}")
        else:
            out(f"  skill {s.final_name} from {it.source}: copied to {skills_dir(home) / s.final_name}"
                + (" (contains scripts the model may run)" if it.has_scripts else ""))
    if any(s.item.spec.get("environment") or s.item.spec.get("headers") for s in picked if s.item.kind == "mcp"):
        out("  Note: the values for the names above are copied into your localcode config, "
            "so those secrets will then exist in two places.")
    out("  Nothing runs at import. Enable a server in /mcps once you have checked its command.")
    if dry:
        return 0
    if not yes and input_fn("Import these? [y/N]: ").strip().lower() not in ("y", "yes"):
        out("No changes made.")
        return 0
    res = apply(picked, home)
    if res["error"]:
        out(f"localcode: {res['error']}")
        return 1
    if res["backup"]:
        out(f"Backed up your config to {res['backup']}")
    if res["servers"]:
        out("Imported (disabled): " + ", ".join(res["servers"]))
    if res["skills"]:
        out("Copied skills: " + ", ".join(res["skills"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli_main(sys.argv[1:]))
