"""Checkpoint-and-branch context control — milestone snapshots with a revert/branch tree.

What this is
------------
A pure-Python mechanism that lets the agent SNAPSHOT conversation + work state at a
milestone, then REVERT to that milestone (or BRANCH off it to try a different approach)
when a path turns out to be a dead end. Each checkpoint records the parent it was created
from, so the set of checkpoints forms a tree you can walk and explore.

How it differs from what already exists
---------------------------------------
- `undo.py` (ChangeLog): a flat, linear file-only "undo last edit" log. No conversation
  state, no named milestones, no branching.
- `snapshots.py` (GhostSnapshot): flat conversation snapshots saved on interrupt. No
  explicit parent/branch relationship between snapshots, so you cannot model "try
  approach B from milestone X, keep A around".

This module adds the missing piece: explicit, labelled MILESTONES linked by parent id
(a branchable tree) plus a clean, side-effect-light revert API that hands the caller the
messages + files to restore and lets the CALLER decide when to touch the filesystem.

Design / semantics
------------------
- `Checkpoint`: id, label, deep-copied messages, the changed files at that point (path ->
  FileState with existed/content/hash), a round/token marker, parent id, timestamp.
- `CheckpointStore`: create / get / list / children / revert, optional JSON persistence.
- `revert(id)` is PURE: it returns `RevertPlan(messages, files_to_restore)`. It does NOT
  write the filesystem or mutate the live conversation. The caller applies the messages
  and (opt-in) calls `restore_files(plan)` to write file content back.
- Branching: pass `parent` to `create()`. Two checkpoints sharing a parent are siblings /
  alternative branches explored from the same milestone.

Scope / limitations (be honest)
-------------------------------
- In-memory PoC by default; JSON persistence is opt-in (`persist_dir` / save / load).
- Filesystem restore only covers files that were SNAPSHOTTED into the checkpoint. It is
  not a full VCS: it will not detect/restore files changed outside the snapshot set, and
  it overwrites current content wholesale (no 3-way merge).
- Message restore is whatever was deep-copied at create() time; messages must be
  JSON-serialisable for persistence to round-trip.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

Message = dict[str, Any]


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()


@dataclass
class FileState:
    """Snapshot of a single file at checkpoint time.

    `existed` is False when the file did not exist yet — reverting then means the
    file should be deleted to match the milestone.
    """
    path: str            # relative to repo root
    existed: bool
    content: str         # original content ("" if it did not exist)
    content_hash: str    # sha256 of content ("" if it did not exist)

    @classmethod
    def capture(cls, repo_root: Path, rel_path: str) -> "FileState":
        full = (repo_root / rel_path)
        existed = full.is_file()
        content = ""
        if existed:
            try:
                content = full.read_text(errors="replace")
            except Exception:
                existed = False
        return cls(
            path=rel_path,
            existed=existed,
            content=content,
            content_hash=_hash(content) if existed else "",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "existed": self.existed,
            "content": self.content,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileState":
        return cls(
            path=d["path"],
            existed=bool(d.get("existed", True)),
            content=d.get("content", ""),
            content_hash=d.get("content_hash", ""),
        )


@dataclass
class Checkpoint:
    """A milestone snapshot of conversation + work state."""
    id: str
    label: str
    messages: list[Message]                 # deep copy of the conversation
    files: dict[str, FileState]             # rel_path -> FileState
    round_marker: int = 0                   # round/turn index at checkpoint time
    token_marker: int = 0                   # token count marker at checkpoint time
    parent_id: Optional[str] = None         # checkpoint this branched from
    created_at: float = field(default_factory=time.time)

    @property
    def changed_files(self) -> list[str]:
        return sorted(self.files.keys())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "messages": self.messages,
            "files": {p: fs.to_dict() for p, fs in self.files.items()},
            "round_marker": self.round_marker,
            "token_marker": self.token_marker,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Checkpoint":
        return cls(
            id=d["id"],
            label=d.get("label", ""),
            messages=list(d.get("messages", [])),
            files={p: FileState.from_dict(fs) for p, fs in d.get("files", {}).items()},
            round_marker=int(d.get("round_marker", 0)),
            token_marker=int(d.get("token_marker", 0)),
            parent_id=d.get("parent_id"),
            created_at=float(d.get("created_at", time.time())),
        )


@dataclass
class RevertPlan:
    """The result of `revert()` — what the CALLER should apply.

    Pure data: applying it (restoring messages, writing files) is the caller's job.
    """
    checkpoint_id: str
    messages: list[Message]          # messages to set the conversation back to
    files_to_restore: list[FileState]  # files to write back / delete

    @property
    def files_to_write(self) -> list[FileState]:
        return [f for f in self.files_to_restore if f.existed]

    @property
    def files_to_delete(self) -> list[FileState]:
        return [f for f in self.files_to_restore if not f.existed]


class CheckpointStore:
    """In-memory store of checkpoints forming a parent/child tree.

    Optionally persistable to JSON under a session/project directory. The store
    itself performs NO filesystem restore — use the module-level `restore_files`
    helper (opt-in) to apply a RevertPlan's file content.
    """

    def __init__(self, repo_root: Path | str | None = None,
                 persist_dir: Path | str | None = None) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.persist_dir = Path(persist_dir) if persist_dir is not None else None
        self._checkpoints: dict[str, Checkpoint] = {}

    # ---- creation -------------------------------------------------------
    def create(self, label: str, messages: list[Message],
               changed_files: list[str] | None = None, *,
               parent: str | None = None,
               round_marker: int = 0,
               token_marker: int = 0,
               capture_content: bool = True) -> str:
        """Create a checkpoint and return its id.

        `messages` is DEEP-COPIED so later mutation of the live conversation does
        not corrupt the checkpoint. `changed_files` are relative paths whose state
        is captured (content + hash) when `capture_content` is True; set False to
        record only that they changed (path + hash of current content, no copy of
        body kept smaller — content still captured for restore unless absent).

        `parent` links this checkpoint to the milestone it branched from. Pass an
        unknown parent id and you get a ValueError so trees stay consistent.
        """
        if parent is not None and parent not in self._checkpoints:
            raise ValueError(f"unknown parent checkpoint: {parent}")

        files: dict[str, FileState] = {}
        for rel in (changed_files or []):
            fs = FileState.capture(self.repo_root, rel)
            if not capture_content:
                # keep metadata only; drop the body to save memory
                fs = FileState(path=fs.path, existed=fs.existed,
                               content="", content_hash=fs.content_hash)
            files[rel] = fs

        cp = Checkpoint(
            id=uuid.uuid4().hex[:10],
            label=label,
            messages=copy.deepcopy(list(messages)),
            files=files,
            round_marker=round_marker,
            token_marker=token_marker,
            parent_id=parent,
        )
        self._checkpoints[cp.id] = cp
        if self.persist_dir is not None:
            self.save()
        return cp.id

    # ---- lookups --------------------------------------------------------
    def get(self, checkpoint_id: str) -> Checkpoint | None:
        cp = self._checkpoints.get(checkpoint_id)
        if cp is not None:
            return cp
        # prefix match convenience
        matches = [c for cid, c in self._checkpoints.items()
                   if cid.startswith(checkpoint_id)]
        return matches[0] if len(matches) == 1 else None

    def list(self) -> list[Checkpoint]:
        """All checkpoints, oldest first."""
        return sorted(self._checkpoints.values(), key=lambda c: c.created_at)

    def children(self, checkpoint_id: str) -> list[Checkpoint]:
        """Checkpoints branched directly off `checkpoint_id`, oldest first."""
        return sorted(
            (c for c in self._checkpoints.values() if c.parent_id == checkpoint_id),
            key=lambda c: c.created_at,
        )

    def lineage(self, checkpoint_id: str) -> list[Checkpoint]:
        """Path from the root down to `checkpoint_id` (inclusive)."""
        chain: list[Checkpoint] = []
        cur = self.get(checkpoint_id)
        seen: set[str] = set()
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            chain.append(cur)
            cur = self.get(cur.parent_id) if cur.parent_id else None
        chain.reverse()
        return chain

    # ---- revert / branch ------------------------------------------------
    def revert(self, checkpoint_id: str) -> RevertPlan:
        """Return a PURE plan to restore the conversation/files to a checkpoint.

        Does NOT mutate anything. The caller sets its conversation to
        `plan.messages` and (opt-in) calls `restore_files(plan, repo_root)`.
        """
        cp = self.get(checkpoint_id)
        if cp is None:
            raise KeyError(f"unknown checkpoint: {checkpoint_id}")
        return RevertPlan(
            checkpoint_id=cp.id,
            messages=copy.deepcopy(cp.messages),
            files_to_restore=[copy.deepcopy(fs) for fs in cp.files.values()],
        )

    def branch(self, checkpoint_id: str, label: str) -> tuple[RevertPlan, str]:
        """Branch off a milestone: create a child and return how to revert to it.

        Convenience for "go back to milestone X and try a different approach":
        creates a new checkpoint whose parent is X (sharing X's messages/files as a
        starting point) and returns (revert_plan_for_X, new_child_id). The caller
        applies the plan, explores, then later checkpoints again under new_child_id.
        """
        parent = self.get(checkpoint_id)
        if parent is None:
            raise KeyError(f"unknown checkpoint: {checkpoint_id}")
        plan = self.revert(parent.id)
        child = Checkpoint(
            id=uuid.uuid4().hex[:10],
            label=label,
            messages=copy.deepcopy(parent.messages),
            files={p: copy.deepcopy(fs) for p, fs in parent.files.items()},
            round_marker=parent.round_marker,
            token_marker=parent.token_marker,
            parent_id=parent.id,
        )
        self._checkpoints[child.id] = child
        if self.persist_dir is not None:
            self.save()
        return plan, child.id

    # ---- persistence (opt-in) ------------------------------------------
    def _persist_path(self) -> Path:
        assert self.persist_dir is not None
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return self.persist_dir / "checkpoints.json"

    def save(self, path: Path | str | None = None) -> Path:
        """Persist all checkpoints to JSON. Uses `persist_dir` if no path given."""
        target = Path(path) if path is not None else self._persist_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "repo_root": str(self.repo_root),
            "checkpoints": [c.to_dict() for c in self.list()],
        }
        target.write_text(json.dumps(payload, indent=2))
        return target

    @classmethod
    def load(cls, path: Path | str, *,
             persist_dir: Path | str | None = None) -> "CheckpointStore":
        """Load a store from a JSON file written by `save`."""
        p = Path(path)
        data = json.loads(p.read_text())
        store = cls(
            repo_root=data.get("repo_root"),
            persist_dir=persist_dir if persist_dir is not None else p.parent,
        )
        for cd in data.get("checkpoints", []):
            cp = Checkpoint.from_dict(cd)
            store._checkpoints[cp.id] = cp
        return store


def restore_files(plan: RevertPlan, repo_root: Path | str) -> dict[str, str]:
    """OPT-IN side effect: write a RevertPlan's files back to disk.

    Files that existed at checkpoint time are overwritten with their snapshotted
    content; files that did NOT exist are deleted (to match the milestone). Returns
    a `{rel_path: status}` report. This is the ONLY function in the module that
    touches the filesystem — callers invoke it explicitly.
    """
    root = Path(repo_root).resolve()
    report: dict[str, str] = {}
    for fs in plan.files_to_restore:
        # Path safety: fs.path comes from a persisted JSON file. An absolute
        # path or a `../..` component would write or UNLINK outside the repo
        # (`Path("/repo") / "/etc/x"` resolves to /etc/x). Refuse anything
        # that escapes the repo root.
        try:
            full = (root / fs.path).resolve()
            full.relative_to(root)
        except ValueError:
            report[fs.path] = "skipped: path escapes repo root"
            continue
        try:
            if fs.existed:
                # Metadata-only snapshots (capture_content=False) store
                # content="" but keep the REAL file's hash. Writing that ""
                # back would zero the user's file — detect the mismatch
                # and skip instead of destroying data.
                if fs.content_hash and fs.content_hash != _hash(fs.content):
                    report[fs.path] = "skipped: content not captured"
                    continue
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(fs.content)
                report[fs.path] = "restored"
            else:
                if full.exists():
                    full.unlink()
                    report[fs.path] = "deleted"
                else:
                    report[fs.path] = "absent"
        except Exception as exc:  # pragma: no cover - defensive
            report[fs.path] = f"error: {exc}"
    return report
