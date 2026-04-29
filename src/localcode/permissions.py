from __future__ import annotations

import json
from pathlib import Path

from .config import ensure_home_dirs


def _permission_path() -> Path:
    return ensure_home_dirs() / "permissions.json"


class PermissionStore:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = str(repo_root.resolve())
        self.path = _permission_path()
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"repos": {}}
        return json.loads(self.path.read_text())

    def _repo_entry(self) -> dict:
        repos = self.data.setdefault("repos", {})
        return repos.setdefault(
            self.repo_root,
            {"allow": [], "deny": []},
        )

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    def decision_for(self, tool_name: str) -> str | None:
        entry = self._repo_entry()
        if tool_name in entry["deny"]:
            return "deny"
        if tool_name in entry["allow"]:
            return "allow"
        # Auto-approve if approved 3+ times this session
        approve_count = entry.get("approve_counts", {}).get(tool_name, 0)
        if approve_count >= 3:
            self.allow(tool_name)
            return "allow"
        return None

    def record_approval(self, tool_name: str) -> None:
        """Record that user approved this tool. After 3, auto-approve."""
        entry = self._repo_entry()
        counts = entry.setdefault("approve_counts", {})
        counts[tool_name] = counts.get(tool_name, 0) + 1
        self.save()
        if counts[tool_name] >= 3:
            self.allow(tool_name)

    def allow(self, tool_name: str) -> None:
        entry = self._repo_entry()
        if tool_name not in entry["allow"]:
            entry["allow"].append(tool_name)
        if tool_name in entry["deny"]:
            entry["deny"].remove(tool_name)
        self.save()

    def deny(self, tool_name: str) -> None:
        entry = self._repo_entry()
        if tool_name not in entry["deny"]:
            entry["deny"].append(tool_name)
        if tool_name in entry["allow"]:
            entry["allow"].remove(tool_name)
        self.save()

    def status_rows(self) -> list[tuple[str, str]]:
        entry = self._repo_entry()
        rows: list[tuple[str, str]] = []
        for name in sorted(entry["allow"]):
            rows.append((name, "allow"))
        for name in sorted(entry["deny"]):
            rows.append((name, "deny"))
        return rows
