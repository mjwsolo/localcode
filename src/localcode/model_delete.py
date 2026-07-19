"""`/delete` — remove downloaded models from disk, safely.

All the logic behind the TUI's `/delete` slash command lives here so it can
be unit-tested without a running Textual app. The chat screen's handler is a
thin wrapper that calls :func:`run_delete_command` and prints the returned
lines.

Safety model (deleting a 40 GB model means a 40 GB re-download, so the
command is deliberately protective):

* ``/delete``                 — list what's on disk with sizes; deletes nothing.
* ``/delete <model>``         — show exactly what would be removed; deletes nothing.
* ``/delete <model> confirm`` — the ONLY form that deletes anything.

Hard refusals (even with ``confirm``):

* the model the server is currently serving (``config.runtime.model``);
* a model with an in-flight background download (deleting under the
  downloader would just corrupt the resume state).

Multi-file awareness: vision models ship an ``mmproj-*.gguf`` sidecar that is
SHARED between quants of the same family (e.g. the Gemma 26B Q3 and Q8 quants
use one projector file). The sidecar is only slated for deletion when the last
family member that references it is being removed — and never while the
currently-configured model still references it. Partial downloads
(``*.gguf.part`` files, undersized finals, and the hub's ``.incomplete``
sidecars) are listed and cleaned up too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models_catalog import ModelChoice, by_filename, model_dir

# A returned output line: ("info" | "error", text). The TUI maps these to
# ChatLog.append_info / append_error.
OutputLine = tuple[str, str]


def human_size(num_bytes: int) -> str:
    """Human-readable size in DECIMAL units (matches the catalog's `size_gb`
    convention and what HuggingFace shows — 11.2 GB == 11.2e9 bytes)."""
    if num_bytes >= 1000 ** 3:
        return f"{num_bytes / 1000 ** 3:.1f} GB"
    if num_bytes >= 1000 ** 2:
        return f"{num_bytes / 1000 ** 2:.0f} MB"
    if num_bytes >= 1000:
        return f"{num_bytes / 1000:.0f} KB"
    return f"{num_bytes} B"


@dataclass
class InstalledModel:
    """One deletable thing in the models dir: a model, a partial download,
    or an orphaned vision sidecar."""

    name: str                    # display name (catalog name or bare filename)
    key: str | None              # catalog key when the file maps to a ModelChoice
    path: Path                   # the primary file (main GGUF or its .part)
    partial: bool                # True when only a partial download exists
    is_sidecar: bool             # True for an orphaned mmproj entry
    files: list[Path] = field(default_factory=list)  # everything deleted with it

    @property
    def size_bytes(self) -> int:
        total = 0
        for f in self.files:
            try:
                total += f.stat().st_size
            except OSError:
                pass
        return total

    @property
    def filename(self) -> str:
        """The main GGUF filename (without any .part suffix)."""
        n = self.path.name
        return n[:-5] if n.endswith(".part") else n


def _is_mmproj(filename: str) -> bool:
    return filename.startswith("mmproj-")


def _hub_leftovers(models_dir: Path, filename: str) -> list[Path]:
    """Best-effort: the hub downloader keeps resume state under
    ``<models_dir>/.cache/huggingface/...`` — collect any ``.incomplete`` /
    ``.metadata`` files that belong to `filename` so deleting a partial
    really frees the space."""
    cache = models_dir / ".cache" / "huggingface"
    if not cache.is_dir():
        return []
    out: list[Path] = []
    try:
        for f in cache.rglob(f"{filename}*"):
            if f.is_file():
                out.append(f)
    except OSError:
        pass
    return out


def _complete_on_disk(path: Path, choice: ModelChoice | None) -> bool:
    """Same completeness contract as bootstrap._is_complete_download (kept
    call-compatible via a local import to avoid a circular import at module
    load)."""
    from .bootstrap import _is_complete_download
    return path.is_file() and _is_complete_download(path, choice)


def list_installed(models_dir: Path | None = None,
                   current_filename: str | None = None) -> list[InstalledModel]:
    """Scan the models dir and return everything `/delete` can act on.

    `current_filename` is the filename of the currently-configured model —
    used only to decide whether a shared mmproj sidecar may be slated for
    deletion (never take the projector out from under the serving model,
    even if its main GGUF was somehow removed by hand).
    """
    d = models_dir if models_dir is not None else model_dir()
    if not d.is_dir():
        return []

    # Case-insensitive sort so listing indices are stable and intuitive
    # ("gemma…" and "Qwen…" order by name, not by ASCII case).
    ggufs = sorted((p for p in d.glob("*.gguf") if p.is_file()),
                   key=lambda p: p.name.lower())
    parts = sorted((p for p in d.glob("*.gguf.part") if p.is_file()),
                   key=lambda p: p.name.lower())

    main_files = [p for p in ggufs if not _is_mmproj(p.name)]
    main_parts = [p for p in parts if not _is_mmproj(p.name[:-5])]
    mmproj_files = [p for p in ggufs if _is_mmproj(p.name)]

    # filenames (sans .part) of every main model present in any form — used
    # to decide whether an mmproj is still referenced by a surviving model.
    present_main: set[str] = {p.name for p in main_files} | {
        p.name[:-5] for p in main_parts
    }

    def _referenced_by(mmproj_name: str, fnames: set[str], excluding: str) -> bool:
        """True if any model in `fnames` OTHER than `excluding` needs this mmproj."""
        for fname in fnames:
            if fname == excluding:
                continue
            c = by_filename(fname)
            if c is not None and c.mmproj_filename == mmproj_name:
                return True
        return False

    # For sidecar-deletion decisions the current-config model counts as a
    # referencer even if its GGUF is missing (never yank the serving model's
    # projector); for orphan detection only files actually on disk count.
    with_current = present_main | ({current_filename} if current_filename else set())

    def _mmproj_referenced_by_others(mmproj_name: str, excluding: str) -> bool:
        return _referenced_by(mmproj_name, with_current, excluding)

    entries: list[InstalledModel] = []
    claimed_mmproj: set[str] = set()

    # Complete main models first, then partials — stable, predictable indices.
    for p in main_files:
        choice = by_filename(p.name)
        complete = _complete_on_disk(p, choice)
        files: list[Path] = [p]
        # A stale .part next to a complete file is leftover — clean it too.
        part = p.with_name(p.name + ".part")
        if part.is_file():
            files.append(part)
        files.extend(_hub_leftovers(d, p.name))
        if choice is not None and choice.mmproj_filename:
            mp = d / choice.mmproj_filename
            if mp.is_file() and not _mmproj_referenced_by_others(
                choice.mmproj_filename, excluding=p.name
            ):
                files.append(mp)
                claimed_mmproj.add(choice.mmproj_filename)
        entries.append(InstalledModel(
            name=choice.name if choice is not None else p.name,
            key=choice.key if choice is not None else None,
            path=p,
            partial=not complete,
            is_sidecar=False,
            files=files,
        ))

    # Bare partials (no complete file at the final name).
    for p in main_parts:
        final = p.name[:-5]
        if any(e.path.name == final for e in entries):
            continue  # already bundled with the complete/undersized file above
        choice = by_filename(final)
        files = [p]
        files.extend(_hub_leftovers(d, final))
        entries.append(InstalledModel(
            name=choice.name if choice is not None else final,
            key=choice.key if choice is not None else None,
            path=p,
            partial=True,
            is_sidecar=False,
            files=files,
        ))

    # Orphaned vision sidecars — mmproj files no ON-DISK model references
    # and no entry above already claims. Listed so users can reclaim the
    # space. (A sidecar referenced only by the current-config model is still
    # listed — `in_use_reason` refuses its deletion with a clear message.)
    for p in mmproj_files:
        if p.name in claimed_mmproj:
            continue
        if _referenced_by(p.name, present_main, excluding=""):
            continue
        entries.append(InstalledModel(
            name=f"{p.name} (vision sidecar, unused)",
            key=None,
            path=p,
            partial=False,
            is_sidecar=True,
            files=[p],
        ))

    return entries


def resolve_target(target: str, entries: list[InstalledModel]) -> InstalledModel | list[InstalledModel]:
    """Find the entry `target` refers to.

    Accepts a 1-based index from the `/delete` listing, a catalog key
    (``gemma-q8``), a filename (with or without ``.gguf``/``.part``), a
    display name, or a unique case-insensitive substring. Returns the match,
    or a (possibly empty) list of candidates when zero/many match.
    """
    t = target.strip()
    if t.isdigit():
        idx = int(t) - 1
        if 0 <= idx < len(entries):
            return entries[idx]
        return []
    tl = t.lower()
    # Exact identifiers first.
    for e in entries:
        candidates = {
            (e.key or "").lower(),
            e.name.lower(),
            e.path.name.lower(),
            e.filename.lower(),
            e.filename.lower().removesuffix(".gguf"),
        }
        if tl in candidates:
            return e
    # Fall back to substring matching (unique match only).
    subs = [
        e for e in entries
        if tl in e.name.lower()
        or tl in e.filename.lower()
        or (e.key is not None and tl in e.key.lower())
    ]
    if len(subs) == 1:
        return subs[0]
    return subs


def in_use_reason(entry: InstalledModel, config) -> str | None:
    """Why `entry` must NOT be deleted right now — or None when it's safe.

    Two hard blocks:
      * it's the model the server is configured to serve right now;
      * it has an in-flight (queued/downloading) background download.
    """
    current = ""
    try:
        current = Path(getattr(config.runtime, "model", "") or "").name
    except Exception:
        current = ""
    if current and current == entry.filename:
        return (
            f"{entry.name} is the model currently loaded by the server. "
            f"Switch to another model with /model first, then delete it."
        )
    # Never yank the vision projector the serving model references.
    if current:
        cur_choice = by_filename(current)
        if cur_choice is not None and cur_choice.mmproj_filename:
            for f in entry.files:
                if f.name == cur_choice.mmproj_filename:
                    return (
                        f"{f.name} is the vision sidecar of the currently "
                        f"loaded model ({cur_choice.name}). Switch models "
                        f"with /model first, then delete it."
                    )
    try:
        from . import bootstrap
        for dl in bootstrap.list_active_downloads():
            if dl.get("model_key") == entry.filename:
                return (
                    f"{entry.name} is still downloading "
                    f"({dl.get('progress_pct', 0)}%). Let it finish — or "
                    f"restart LocalCode to stop the download — before deleting."
                )
    except Exception:
        pass
    return None


def delete_entry(entry: InstalledModel) -> tuple[bool, list[OutputLine], int]:
    """Remove every file belonging to `entry`. Returns (ok, lines, freed_bytes)."""
    freed = 0
    errors: list[OutputLine] = []
    for f in entry.files:
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        try:
            f.unlink(missing_ok=True)
            freed += size
        except OSError as e:
            errors.append(("error", f"Couldn't delete {f.name}: {e}"))
    return (not errors), errors, freed


# ── The command itself ───────────────────────────────────────────────


def _listing_lines(entries: list[InstalledModel], d: Path,
                   current_filename: str | None = None) -> list[OutputLine]:
    if not entries:
        return [(
            "info",
            f"No downloaded models found in {d}.\n"
            f"Use /model to browse and download one.",
        )]
    lines = [f"DOWNLOADED MODELS ({d})"]
    name_w = max(len(e.name) for e in entries)
    total = 0
    for i, e in enumerate(entries, start=1):
        size = e.size_bytes
        total += size
        tag = ""
        if current_filename and e.filename == current_filename:
            tag = "  (in use — can't be deleted)"
        elif e.partial:
            tag = "  (partial download)"
        lines.append(f"  {i}. {e.name.ljust(name_w)}  {human_size(size):>9}{tag}")
    lines.append("")
    lines.append(f"Total on disk: {human_size(total)}")
    lines.append("Delete one with /delete <number or name> — you'll be asked to confirm.")
    return [("info", "\n".join(lines))]


def _not_found_lines(target: str, entries: list[InstalledModel],
                     candidates: list[InstalledModel]) -> list[OutputLine]:
    if candidates:
        opts = ", ".join(c.key or c.filename for c in candidates)
        return [(
            "error",
            f"'{target}' matches more than one model: {opts}. "
            f"Use the exact name, or its number from /delete.",
        )]
    if not entries:
        return [(
            "error",
            f"No downloaded model matches '{target}' — nothing is downloaded yet.",
        )]
    valid = ", ".join(e.key or e.filename for e in entries)
    return [(
        "error",
        f"No downloaded model matches '{target}'. Downloaded: {valid}. "
        f"Run /delete to see the full list.",
    )]


def _confirm_lines(entry: InstalledModel) -> list[OutputLine]:
    lines = [f"Delete {entry.name}?"]
    for f in entry.files:
        try:
            size = human_size(f.stat().st_size)
        except OSError:
            size = "?"
        note = ""
        if _is_mmproj(f.name):
            note = "  (vision sidecar, no longer needed by other models)"
        elif f.name.endswith(".part") or f.suffix in (".incomplete", ".metadata"):
            note = "  (partial download)"
        lines.append(f"  {f.name}  {size}{note}")
    total = human_size(entry.size_bytes)
    lines.append("")
    if entry.partial:
        lines.append(f"This frees {total}. The download is incomplete — deleting it "
                     f"means starting over from 0% if you want this model later.")
    else:
        lines.append(f"This frees {total} — but getting {entry.name} back later "
                     f"means re-downloading all {total}.")
    handle = entry.key or entry.filename
    lines.append(f"To proceed, run: /delete {handle} confirm")
    lines.append("Nothing has been deleted.")
    return [("info", "\n".join(lines))]


def run_delete_command(arg: str, config,
                       models_dir: Path | None = None) -> list[OutputLine]:
    """Execute `/delete <arg>` and return the lines to print.

    `arg` is everything after `/delete` (may be empty). Deletion happens
    ONLY when the last token is the literal word `confirm` AND the target
    resolves AND the model isn't in use.
    """
    d = models_dir if models_dir is not None else model_dir()
    current = ""
    try:
        current = Path(getattr(config.runtime, "model", "") or "").name
    except Exception:
        current = ""
    entries = list_installed(d, current_filename=current or None)

    tokens = arg.split()
    confirmed = len(tokens) >= 2 and tokens[-1].lower() == "confirm"
    target = " ".join(tokens[:-1] if confirmed else tokens).strip()

    # `/delete` (or a stray bare `/delete confirm`) → list, delete nothing.
    if not target:
        return _listing_lines(entries, d, current_filename=current or None)

    found = resolve_target(target, entries)
    if not isinstance(found, InstalledModel):
        return _not_found_lines(target, entries, found)

    reason = in_use_reason(found, config)
    if reason is not None:
        return [("error", f"Can't delete: {reason}")]

    if not confirmed:
        return _confirm_lines(found)

    ok, errors, freed = delete_entry(found)
    if ok:
        return [("info", f"Deleted {found.name} — freed {human_size(freed)}.")]
    out: list[OutputLine] = list(errors)
    if freed:
        out.append(("info", f"Partially deleted {found.name} — freed {human_size(freed)} "
                            f"but some files remain (see above)."))
    return out
