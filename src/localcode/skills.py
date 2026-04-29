"""LocalCode skills — markdown-based prompt templates with a local registry.

Two invocation paths exist in parallel, both reading from the same `*.md`
files on disk:

1. **User-invoked** (`@skillname` / `#skillname` in chat) — the original
   path. `resolve_referenced_skills()` scans the user's text and
   expands matching names into the conversation. Good for "please do
   a review" when the user explicitly asks.

2. **Model-invoked** (skill tool) — added for Qwen3.6/Gemma 4 agents.
   The model sees a one-line listing of each skill in its system prompt
   and calls `skill(name=...)` when it judges a pattern-match. The tool
   returns the full skill body; the next model turn acts on it.
   Research basis: Voyager (arXiv 2305.16291), terminal coding tools bundled
   skills, OpenAI agent `SkillTool` architecture. See ARCHITECTURE.md
   §1.11 for why this specifically helps small quantized models.

Skills declare optional YAML-lite frontmatter:

```markdown
---
name: run-tests
description: Detect the test framework and run tests, parsing results.
when_to_use: User asks to run tests / verify a change / after you've
  edited code that has tests.
---

<body — instructions the model follows on invocation>
```

The parser accepts skills WITHOUT frontmatter (treats the whole file
as the body and derives a minimal description). This keeps the
existing `@skill` path working while letting new skills opt into
richer metadata.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import ensure_home_dirs

SKILL_PATTERN = re.compile(r"(?:@|#)([a-zA-Z0-9_-]+)")
REGISTRY_FILE = ensure_home_dirs() / "skills" / "registry.json"

# Cap on the one-liner the model sees in the prompt listing. Mirrors the
# pattern agent uses to prevent one skill from dominating the budget.
DESCRIPTION_CAP = 200

# Frontmatter validation caps (borrowed from minimal-agent/coding-agent spec).
# Names kept short so the model's system-prompt listing stays compact; a
# long description is truncated rather than rejected so legacy skills
# keep working.
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024


# ── Skill directories ────────────────────────────────────────────────────

def skill_dirs(repo_root: Path) -> list[Path]:
    return [
        ensure_home_dirs() / "skills",
        repo_root / ".localcode" / "skills",
    ]


def _global_skill_dir() -> Path:
    d = ensure_home_dirs() / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Core API (unchanged interface) ───────────────────────────────────────

def list_skills(repo_root: Path) -> list[str]:
    names: set[str] = set()
    for directory in skill_dirs(repo_root):
        if not directory.exists():
            continue
        for path in directory.glob("*.md"):
            names.add(path.stem)
    return sorted(names)


def load_skill(repo_root: Path, name: str) -> str | None:
    for directory in skill_dirs(repo_root):
        path = directory / f"{name}.md"
        if path.exists():
            return path.read_text(errors="replace")
    return None


def resolve_referenced_skills(repo_root: Path, text: str) -> list[tuple[str, str]]:
    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in SKILL_PATTERN.findall(text):
        if name in seen:
            continue
        content = load_skill(repo_root, name)
        if content:
            resolved.append((name, content))
            seen.add(name)
    return resolved


# ── Registry ─────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {"skills": {}, "sources": []}
    try:
        return json.loads(REGISTRY_FILE.read_text())
    except Exception:
        return {"skills": {}, "sources": []}


def _save_registry(data: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(data, indent=2))


# ── Install / Remove ────────────────────────────────────────────────────

def install_skill(source: str) -> tuple[bool, str]:
    """Install a skill from a local path or URL.

    Supports:
      - Local .md file path
      - URL to a raw .md file
      - Directory containing .md files
    """
    source_path = Path(source).expanduser()

    # Local file
    if source_path.is_file() and source_path.suffix == ".md":
        return _install_local_file(source_path)

    # Local directory
    if source_path.is_dir():
        installed = []
        for md_file in source_path.glob("*.md"):
            ok, msg = _install_local_file(md_file)
            if ok:
                installed.append(md_file.stem)
        if installed:
            return True, f"Installed {len(installed)} skills: {', '.join(installed)}"
        return False, f"No .md skill files found in {source}"

    # URL
    if source.startswith("http://") or source.startswith("https://"):
        return _install_from_url(source)

    return False, f"Cannot install from '{source}'. Provide a .md file, directory, or URL."


def _install_local_file(path: Path) -> tuple[bool, str]:
    dest = _global_skill_dir() / path.name
    shutil.copy2(path, dest)
    name = path.stem
    reg = _load_registry()
    reg["skills"][name] = {
        "source": str(path),
        "path": str(dest),
    }
    _save_registry(reg)
    return True, f"Installed skill '{name}' -> {dest}"


def _install_from_url(url: str) -> tuple[bool, str]:
    try:
        response = httpx.get(url, timeout=20.0, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        return False, f"Failed to fetch {url}: {exc}"

    # Derive name from URL
    url_path = url.rstrip("/").split("/")[-1]
    if not url_path.endswith(".md"):
        url_path = url_path + ".md"
    name = url_path.removesuffix(".md")

    dest = _global_skill_dir() / url_path
    dest.write_text(response.text)

    reg = _load_registry()
    reg["skills"][name] = {
        "source": url,
        "path": str(dest),
    }
    _save_registry(reg)
    return True, f"Installed skill '{name}' from {url} -> {dest}"


def remove_skill(name: str) -> tuple[bool, str]:
    """Remove an installed skill."""
    dest = _global_skill_dir() / f"{name}.md"
    if dest.exists():
        dest.unlink()
    reg = _load_registry()
    if name in reg.get("skills", {}):
        del reg["skills"][name]
        _save_registry(reg)
    return True, f"Removed skill '{name}'"


def search_skills(query: str, repo_root: Path) -> list[dict[str, str]]:
    """Search installed skills by name and content."""
    query_lower = query.lower()
    results: list[dict[str, str]] = []
    for directory in skill_dirs(repo_root):
        if not directory.exists():
            continue
        for path in directory.glob("*.md"):
            name = path.stem
            content = path.read_text(errors="replace")
            if query_lower in name.lower() or query_lower in content[:500].lower():
                preview = content[:200].replace("\n", " ").strip()
                results.append({
                    "name": name,
                    "path": str(path),
                    "preview": preview,
                })
    return results


def skill_info(name: str, repo_root: Path) -> dict[str, str] | None:
    """Get info about a specific skill."""
    content = load_skill(repo_root, name)
    if content is None:
        return None
    reg = _load_registry()
    entry = reg.get("skills", {}).get(name, {})
    return {
        "name": name,
        "source": entry.get("source", "local"),
        "path": entry.get("path", ""),
        "content_preview": content[:500],
        "size": str(len(content)),
    }



def _bundled_skill_dir() -> Path:
    """Directory where the package's built-in starter skills live.

    These ship as `.md` files alongside the Python code (one file per
    skill) so users can inspect, copy, and customize them the same way
    they would their own skills — no dict-of-strings opacity.
    """
    return Path(__file__).parent / "skills"


def ensure_builtin_skills() -> int:
    """Copy bundled `.md` skills into `~/.localcode/skills/` if not already there.

    Returns the count newly created. We copy rather than symlink so
    user edits to `~/.localcode/skills/foo.md` stay local and survive package
    upgrades. The package-bundled originals remain the fallback
    registry source — see `load_registry`.
    """
    bundled = _bundled_skill_dir()
    if not bundled.is_dir():
        return 0
    target = _global_skill_dir()
    created = 0
    for path in sorted(bundled.glob("*.md")):
        dst = target / path.name
        if not dst.exists():
            dst.write_text(path.read_text(encoding="utf-8"))
            created += 1
    return created


# ── Model-invoked skill tool (Voyager / agent pattern) ──────────────────
#
# The body-per-invocation design keeps the system prompt tiny even as the
# skill count grows. Each skill contributes ~1 line to the prompt; the
# full body (which may be 500-2000 tokens of detailed instructions)
# only loads when the model calls the tool.

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def _truthy(val: str) -> bool:
    """Parse YAML-lite boolean: true/yes/1 → True, anything else → False."""
    return val.strip().lower() in ("true", "yes", "1", "on")


def _clip(s: str, cap: int) -> str:
    """Truncate s to cap chars with an ellipsis; preserves leading strip."""
    s = s.strip()
    if len(s) <= cap:
        return s
    return s[: cap - 1].rstrip() + "…"


@dataclass
class Skill:
    """Parsed skill — frontmatter metadata + markdown body."""
    name: str
    description: str
    when_to_use: str
    body: str
    source_path: Path
    origin: str  # "user" | "project" | "bundled"
    # When True, skill body is still reachable via @-reference or direct
    # invocation, but it's hidden from the model's skill listing so the
    # model won't call it autonomously. Useful for skills that require
    # human judgment (e.g. destructive-ops recipes).
    disable_model_invocation: bool = False

    def one_line(self) -> str:
        """Short listing entry injected into the system prompt."""
        desc = self.description.strip().replace("\n", " ")
        if len(desc) > DESCRIPTION_CAP:
            desc = desc[: DESCRIPTION_CAP - 1] + "…"
        return f"- `{self.name}` — {desc}"


@dataclass
class SkillRegistry:
    """Resolved set of skills available this session. Rebuilt per agent-loop."""
    skills: dict[str, Skill] = field(default_factory=dict)

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def ordered(self) -> list[Skill]:
        return sorted(self.skills.values(), key=lambda s: s.name)

    def listing(self) -> str:
        """Block for the system prompt — empty string if no skills registered.

        Skips any skill with `disable_model_invocation=True` — the model
        doesn't need to see recipes it can't/shouldn't auto-invoke.
        """
        visible = [s for s in self.ordered() if not s.disable_model_invocation]
        if not visible:
            return ""
        return "\n".join(s.one_line() for s in visible)


def select_dynamic_skills(
    user_text: str,
    registry: SkillRegistry,
    *,
    recent_tools: list[str] | None = None,
    last_failed_tool: str | None = None,
    target_chars: int = 1400,
) -> list[Skill]:
    """Select a small per-turn skill set.

    Priority is intentionally narrow: explicit recovery from the last
    failed tool first, then current-turn intent. Recent tools are
    telemetry-only because they are stale across follow-up turns.
    `target_chars` is a soft prompt-budget target, not a turn cap.
    """
    selected: list[Skill] = []
    used = 0

    def add(name: str) -> None:
        nonlocal used
        skill = registry.get(name)
        if skill is None or skill.disable_model_invocation:
            return
        if any(s.name == skill.name for s in selected):
            return
        cost = len(skill.body) + len(skill.description)
        if selected and used + cost > target_chars:
            return
        selected.append(skill)
        used += cost

    if last_failed_tool:
        add(_tool_to_skill_name(last_failed_tool))
    # Do not auto-inject from stale recent tools. Telemetry still records
    # recent-tool candidates, but selecting from them polluted follow-up
    # turns: a prior bash launch made "refactor this to Rust" inject the
    # run-tests skill, adding irrelevant prompt weight and bad guidance.
    for name in _predict_skill_names(user_text):
        add(name)
    return selected


def dynamic_skill_candidates(
    user_text: str,
    *,
    recent_tools: list[str] | None = None,
    last_failed_tool: str | None = None,
) -> list[dict[str, str]]:
    """Explain which skill triggers were considered for telemetry."""
    candidates: list[dict[str, str]] = []
    if last_failed_tool:
        candidates.append({
            "name": _tool_to_skill_name(last_failed_tool),
            "reason": f"last_failed_tool:{last_failed_tool}",
        })
    for tool in (recent_tools or [])[:3]:
        mapped = _tool_to_skill_name(tool)
        if mapped:
            candidates.append({"name": mapped, "reason": f"recent_tool:{tool}"})
    for name in _predict_skill_names(user_text):
        candidates.append({"name": name, "reason": "user_intent"})
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for item in candidates:
        key = (item.get("name", ""), item.get("reason", ""))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def dynamic_skill_block(skills: list[Skill]) -> str:
    if not skills:
        return ""
    parts = ["\n\n## Selected Skills\n"]
    for skill in skills:
        parts.append(f"\n### {skill.name}\n{skill.body.strip()}\n")
    return "".join(parts)


def _tool_to_skill_name(tool_name: str) -> str:
    mapping = {
        "bash": "run-tests",
        "grep": "locate",
        "glob": "locate",
        "list_files": "locate",
        "read_file": "locate",
        "edit_file": "edit-verified",
        "multi_edit": "edit-verified",
        "write_file": "edit-verified",
    }
    return mapping.get((tool_name or "").strip(), "")


def _predict_skill_names(user_text: str) -> list[str]:
    predicted: list[str] = []
    lower = (user_text or "").lower()
    for name, pattern in _AUTO_TRIGGERS:
        if re.search(pattern, lower, re.IGNORECASE):
            predicted.append(name)
    return predicted


def _parse_skill_file(path: Path, origin: str) -> Skill | None:
    """Parse a `.md` skill file. Accepts files with or without frontmatter.

    Without frontmatter, derives a minimal description from the first
    non-empty line so older `@skill` files still register as invocable.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    # Strip BOM if present
    if raw.startswith("﻿"):
        raw = raw[1:]

    name = path.stem
    m = _FRONTMATTER_RE.match(raw)
    if m:
        front_text, body = m.group(1), m.group(2).strip()
        fields: dict[str, str] = {}
        cur_key: str | None = None
        for line in front_text.splitlines():
            if not line.strip():
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:", line):
                key, _, val = line.partition(":")
                cur_key = key.strip()
                fields[cur_key] = val.strip()
            elif cur_key is not None and (line.startswith(" ") or line.startswith("\t")):
                fields[cur_key] = (fields[cur_key] + " " + line.strip()).strip()
        name = _clip(fields.get("name", name) or path.stem, MAX_NAME_LENGTH)
        description = _clip(fields.get("description", ""), MAX_DESCRIPTION_LENGTH)
        when_to_use = fields.get("when_to_use", "").strip()
        disable_flag = _truthy(fields.get("disable-model-invocation", "")
                               or fields.get("disable_model_invocation", ""))
    else:
        body = raw.strip()
        # Derive a one-line description from the first non-empty line of
        # the body — keeps legacy no-frontmatter skills usable from the tool.
        first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        description = _clip(first_line, MAX_DESCRIPTION_LENGTH)
        when_to_use = ""
        disable_flag = False

    if not description:
        description = f"Skill '{name}' (no description provided)."

    return Skill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        body=body,
        source_path=path,
        origin=origin,
        disable_model_invocation=disable_flag,
    )


def _parse_skill_from_string(name: str, body_with_frontmatter: str) -> Skill | None:
    """Parse a skill whose full markdown (frontmatter + body) lives in
    memory rather than on disk. Kept for callers that generate skill
    content dynamically (rare) — the primary on-disk path is
    `_parse_skill_file`.
    """
    # Reuse the file-parser by pointing source_path at a sentinel path
    # derived from the name. This path is never read; it's just a label.
    sentinel = Path("<builtin>") / f"{name}.md"
    raw = body_with_frontmatter
    if raw.startswith("﻿"):
        raw = raw[1:]
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return None
    front_text, body = m.group(1), m.group(2).strip()
    fields: dict[str, str] = {}
    cur_key: str | None = None
    for line in front_text.splitlines():
        if not line.strip():
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:", line):
            key, _, val = line.partition(":")
            cur_key = key.strip()
            fields[cur_key] = val.strip()
        elif cur_key is not None and (line.startswith(" ") or line.startswith("\t")):
            fields[cur_key] = (fields[cur_key] + " " + line.strip()).strip()
    skill_name = _clip(fields.get("name", name) or name, MAX_NAME_LENGTH)
    description = _clip(fields.get("description", ""), MAX_DESCRIPTION_LENGTH)
    when_to_use = fields.get("when_to_use", "").strip()
    disable_flag = _truthy(fields.get("disable-model-invocation", "")
                           or fields.get("disable_model_invocation", ""))
    if not description:
        return None
    return Skill(
        name=skill_name,
        description=description,
        when_to_use=when_to_use,
        body=body,
        source_path=sentinel,
        origin="bundled",
        disable_model_invocation=disable_flag,
    )


def load_registry(repo_root: Path) -> SkillRegistry:
    """Build a fresh registry. Resolution order: user → project → bundled.

    Bundled skills are loaded directly from the package's
    `src/localcode/skills/*.md` directory (see `_bundled_skill_dir`). Users
    can override any bundled skill by dropping a same-name `.md` file
    in `~/.localcode/skills/` or `{repo}/.localcode/skills/` — first-found wins.
    """
    registry: dict[str, Skill] = {}

    # User + project scopes win — scanned first so first-found-wins keeps them.
    for directory in skill_dirs(repo_root):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            skill = _parse_skill_file(path, origin=_origin_for(directory))
            if skill is None:
                continue
            if skill.name not in registry:
                registry[skill.name] = skill

    # Bundled skills as the lowest-priority tier. Any name already
    # claimed by user/project is left alone.
    bundled = _bundled_skill_dir()
    if bundled.is_dir():
        for path in sorted(bundled.glob("*.md")):
            skill = _parse_skill_file(path, origin="bundled")
            if skill is None or skill.name in registry:
                continue
            registry[skill.name] = skill

    return SkillRegistry(skills=registry)


def _origin_for(directory: Path) -> str:
    s = str(directory)
    if ".localcode/skills" in s and s.startswith(str(Path.home())):
        return "user"
    return "project"


def skill_tool_schema() -> dict:
    """Tool schema for the model-invoked `skill` tool. Drop into TOOL_SCHEMAS."""
    return {
        "type": "function",
        "function": {
            "name": "skill",
            "description": (
                "Invoke a named skill (recipe) to load its detailed "
                "instructions into the conversation. Available skill "
                "names are listed under 'Available skills' in your "
                "system prompt. Use a skill when the user's request "
                "matches its purpose — the skill body will guide your "
                "next actions. Call with {\"name\": \"<skill-name>\"}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name, e.g. 'review' or 'run-tests'.",
                    },
                },
                "required": ["name"],
            },
        },
    }


# ── Keyword auto-activation ─────────────────────────────────────────
#
# C-6 measured the model invoking `skill` voluntarily only 1/7 of the
# time — IQ2_M doesn't reliably pattern-match skill listings into tool
# calls. Rather than relying on model agency, we detect clear-intent
# phrases in the user's turn and auto-activate the matching skill: its
# body gets prepended to the model's context for that turn so it sees
# the recipe natively without needing to invoke the tool.
#
# Conservative patterns only — false positives cost the model ~1 KB of
# extra system-prompt context (cheap); false negatives leave the model
# on its own (we're no worse than before).

_AUTO_TRIGGERS: tuple[tuple[str, str], ...] = (
    # (skill_name, regex pattern — case-insensitive)
    ("run-tests",         r"\b(run|pass(?:ing)?)\s+(?:the\s+)?tests?\b|\bpytest\b|\bjest\b|\bcargo\s+test\b|\bgo\s+test\b|\bare\s+(?:the\s+)?tests\s+passing\b"),
    ("git-commit-safely", r"\b(git\s+)?commit(?:\s+(?:this|my|the|changes?|now))?\b|\bgit\s+add\b"),
    ("debug",             r"\btraceback\b|\bfailing\s+test\b|\bexception\b|\berror:\s|\bwhy\s+(?:is|does).*(?:fail|error|break)"),
    ("review",            r"\breview\s+(?:this|my|the|src|code)\b|\bcode\s+review\b|\baudit\s+(?:this|my|the)\b"),
    ("explain",           r"\bexplain\s+(?:this|that|the)\b|\bwhat\s+does\s+.+\s+do\b|\bhow\s+does\s+.+\s+work\b|\bwalk\s+me\s+through\b"),
    ("locate",            r"\bwhere\s+(?:is|does)\b|\bfind\s+(?:the\s+)?file\b|\bwhich\s+file\b"),
)


def auto_activate(user_text: str, registry: SkillRegistry) -> Skill | None:
    """Return the first skill whose trigger pattern matches `user_text`,
    or None. Case-insensitive regex scan — no NLP, no embeddings.

    Called once per user turn in the agent loop. On match the returned
    skill's body should be injected as a system-role note before the
    model's first response so it sees the recipe in-context.
    """
    import re
    if not user_text:
        return None
    for name, pattern in _AUTO_TRIGGERS:
        skill = registry.get(name)
        if skill is None:
            continue
        if skill.disable_model_invocation:
            continue
        if re.search(pattern, user_text, re.IGNORECASE):
            return skill
    return None


def invoke_skill(registry: SkillRegistry, name: str) -> str:
    """Return the skill body for tool-result injection, or an error message."""
    skill = registry.get(name)
    if skill is None:
        # Only list model-invocable skills in the error — mirrors listing().
        visible = sorted(n for n, s in registry.skills.items()
                         if not s.disable_model_invocation)
        available = ", ".join(visible) or "(none registered)"
        return (
            f"Error: no skill named '{name}'. "
            f"Available skills: {available}. "
            f"Use the exact name from the system prompt's skill listing."
        )
    if skill.disable_model_invocation:
        return (
            f"Error: skill '{name}' is marked disable-model-invocation "
            f"and can only be invoked by the user via @{name}."
        )
    header = f"# Skill: {skill.name}"
    if skill.when_to_use:
        header += f"\n(When to use: {skill.when_to_use})"
    return (
        f"{header}\n\n{skill.body}\n\n"
        f"---\nFollow the instructions above to complete the user's request. "
        f"Use your other tools (read_file, write_file, bash, etc.) as needed."
    )
