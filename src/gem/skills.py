"""Gem skills — markdown-based prompt templates with a local registry."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import httpx

from .config import ensure_home_dirs

SKILL_PATTERN = re.compile(r"(?:@|#)([a-zA-Z0-9_-]+)")
REGISTRY_FILE = ensure_home_dirs() / "skills" / "registry.json"


# ── Skill directories ────────────────────────────────────────────────────

def skill_dirs(repo_root: Path) -> list[Path]:
    return [
        ensure_home_dirs() / "skills",
        repo_root / ".gem" / "skills",
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


# ── Built-in starter skills ─────────────────────────────────────────────

BUILTIN_SKILLS = {
    "review": """Review the code changes in the current git diff.
Focus on:
- Correctness: Are there bugs or logic errors?
- Security: Any injection, auth, or data exposure issues?
- Performance: Obvious bottlenecks or N+1 patterns?
- Style: Does it match the surrounding code conventions?

Be specific. Reference file paths and line numbers. Suggest fixes as edit_file calls.""",

    "test": """Write tests for the code I point you to.
- Use the project's existing test framework (detect from package.json, pyproject.toml, Cargo.toml).
- Match the style of existing tests if any exist.
- Cover happy path, edge cases, and error cases.
- Run the tests to verify they pass.""",

    "explain": """Explain the code I point you to.
- Start with the high-level purpose (one sentence).
- Walk through the key logic flow.
- Note any non-obvious patterns or gotchas.
- Keep it concise — I can ask follow-ups.""",

    "refactor": """Refactor the code I point you to.
- Preserve all existing behavior (no feature changes).
- Focus on readability, reducing duplication, better naming.
- Run tests before and after to verify nothing broke.
- Show a clear diff of changes.""",

    "debug": """Help me debug an issue.
- First, reproduce the problem by reading the relevant code and error output.
- Use grep/glob to find related code.
- Form a hypothesis about the root cause.
- Suggest a fix. Apply it if you're confident.
- Run tests to verify the fix.""",
}


def ensure_builtin_skills() -> int:
    """Write built-in starter skills if they don't exist. Returns count created."""
    skill_dir = _global_skill_dir()
    created = 0
    for name, content in BUILTIN_SKILLS.items():
        path = skill_dir / f"{name}.md"
        if not path.exists():
            path.write_text(content)
            created += 1
    return created
