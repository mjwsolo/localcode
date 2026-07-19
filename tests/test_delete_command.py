"""Tests for the `/delete` slash command (model_delete.py).

Covers the full safety contract:
  * bare `/delete` lists downloaded models with sizes — deletes nothing;
  * `/delete <target>` shows a confirmation preview — deletes nothing;
  * only `/delete <target> confirm` deletes;
  * the in-use (currently served) model and in-flight downloads are refused;
  * partial downloads and shared vision sidecars are handled correctly;
  * unknown targets get a helpful message listing valid names.

Model files are created SPARSE (os.truncate) so an "11.2 GB" GGUF costs no
real disk — st_size is what the completeness check and the size display read.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from localcode.model_delete import (
    human_size,
    in_use_reason,
    list_installed,
    resolve_target,
    run_delete_command,
)
from localcode.models_catalog import by_key

GEMMA = by_key("gemma")            # gemma-4-26B-A4B-it-UD-IQ3_S.gguf, 11.2 GB
GEMMA_Q8 = by_key("gemma-q8")      # same family — SHARES the mmproj sidecar
QWEN = by_key("qwen")              # Qwen3.6-35B-A3B-UD-IQ2_M.gguf, 10.7 GB


def _config(model: str = "") -> SimpleNamespace:
    return SimpleNamespace(runtime=SimpleNamespace(model=model))


def _make_file(d: Path, name: str, size_bytes: int) -> Path:
    p = d / name
    p.touch()
    os.truncate(p, size_bytes)
    return p


def _make_model(d: Path, choice, complete: bool = True) -> Path:
    """Create a sparse GGUF for a catalog choice — full-size when complete,
    ~40% of the declared size otherwise (fails the completeness check)."""
    full = int(choice.size_gb * 1000 ** 3)
    return _make_file(d, choice.filename, full if complete else int(full * 0.4))


def _run(arg: str, d: Path, config=None) -> str:
    """Run the command and flatten the output lines for assertions."""
    lines = run_delete_command(arg, config or _config(), models_dir=d)
    return "\n".join(text for _kind, text in lines)


# ── formatting ───────────────────────────────────────────────────────


def test_human_size_is_decimal_and_readable() -> None:
    assert human_size(int(11.2 * 1000 ** 3)) == "11.2 GB"
    assert human_size(870 * 1000 ** 2) == "870 MB"
    assert human_size(0) == "0 B"


# ── listing (`/delete` with no args) ─────────────────────────────────


def test_bare_delete_lists_models_with_sizes_and_indices(tmp_path: Path) -> None:
    _make_model(tmp_path, GEMMA)
    _make_model(tmp_path, QWEN)
    out = _run("", tmp_path)
    assert "1." in out and "2." in out
    assert GEMMA.name in out and QWEN.name in out
    assert "11.2 GB" in out and "10.7 GB" in out
    assert "Total on disk" in out
    assert "/delete <number or name>" in out


def test_bare_delete_marks_the_in_use_model(tmp_path: Path) -> None:
    p = _make_model(tmp_path, GEMMA)
    out = _run("", tmp_path, _config(model=str(p)))
    assert "in use" in out


def test_bare_delete_with_nothing_downloaded(tmp_path: Path) -> None:
    out = _run("", tmp_path)
    assert "No downloaded models" in out


def test_partial_download_is_listed_as_partial(tmp_path: Path) -> None:
    _make_file(tmp_path, GEMMA.filename + ".part", 3 * 1000 ** 3)
    out = _run("", tmp_path)
    assert GEMMA.name in out
    assert "partial download" in out


def test_undersized_final_file_is_listed_as_partial(tmp_path: Path) -> None:
    _make_model(tmp_path, GEMMA, complete=False)
    out = _run("", tmp_path)
    assert "partial download" in out


def test_listing_never_deletes_anything(tmp_path: Path) -> None:
    p = _make_model(tmp_path, GEMMA)
    _run("", tmp_path)
    assert p.exists()


# ── target resolution ────────────────────────────────────────────────


def test_resolve_by_index_key_and_filename(tmp_path: Path) -> None:
    _make_model(tmp_path, GEMMA)
    _make_model(tmp_path, QWEN)
    entries = list_installed(tmp_path)
    assert resolve_target("1", entries).filename == GEMMA.filename
    assert resolve_target("qwen", entries).filename == QWEN.filename
    assert resolve_target(GEMMA.filename, entries).filename == GEMMA.filename
    assert resolve_target(GEMMA.filename[:-5], entries).filename == GEMMA.filename  # sans .gguf


def test_unknown_target_lists_valid_names(tmp_path: Path) -> None:
    _make_model(tmp_path, GEMMA)
    out = _run("nonexistent-model", tmp_path)
    assert "No downloaded model matches 'nonexistent-model'" in out
    assert "gemma" in out
    assert "Run /delete" in out


def test_ambiguous_target_is_refused_with_candidates(tmp_path: Path) -> None:
    _make_model(tmp_path, GEMMA)
    _make_model(tmp_path, GEMMA_Q8)
    out = _run("Gemma 4 26B", tmp_path)  # substring of both display names
    assert "more than one model" in out
    assert (tmp_path / GEMMA.filename).exists()
    assert (tmp_path / GEMMA_Q8.filename).exists()


def test_out_of_range_index_is_not_found(tmp_path: Path) -> None:
    _make_model(tmp_path, GEMMA)
    out = _run("7", tmp_path)
    assert "No downloaded model matches '7'" in out


# ── confirmation gating ──────────────────────────────────────────────


def test_delete_without_confirm_previews_and_deletes_nothing(tmp_path: Path) -> None:
    p = _make_model(tmp_path, GEMMA)
    out = _run("gemma", tmp_path)
    assert p.exists(), "no deletion may happen without the confirm token"
    assert f"Delete {GEMMA.name}?" in out
    assert GEMMA.filename in out          # exact file path shown
    assert "11.2 GB" in out               # size to be freed shown
    assert "/delete gemma confirm" in out  # exact command to proceed
    assert "Nothing has been deleted" in out


def test_preview_warns_about_the_re_download_cost(tmp_path: Path) -> None:
    _make_model(tmp_path, GEMMA)
    out = _run("gemma", tmp_path)
    assert "re-download" in out.lower()


def test_bare_confirm_token_deletes_nothing(tmp_path: Path) -> None:
    p = _make_model(tmp_path, GEMMA)
    _run("confirm", tmp_path)
    assert p.exists()


# ── deletion (`/delete <target> confirm`) ────────────────────────────


def test_delete_with_confirm_removes_the_file(tmp_path: Path) -> None:
    p = _make_model(tmp_path, GEMMA)
    out = _run("gemma confirm", tmp_path)
    assert not p.exists()
    assert "Deleted" in out and GEMMA.name in out
    assert "11.2 GB" in out  # freed amount reported


def test_delete_by_index_with_confirm(tmp_path: Path) -> None:
    p = _make_model(tmp_path, GEMMA)
    out = _run("1 confirm", tmp_path)
    assert not p.exists()
    assert "freed" in out.lower()


def test_delete_partial_download_cleans_part_and_hub_leftovers(tmp_path: Path) -> None:
    part = _make_file(tmp_path, GEMMA.filename + ".part", 2 * 1000 ** 3)
    hub_dir = tmp_path / ".cache" / "huggingface" / "download"
    hub_dir.mkdir(parents=True)
    leftover = _make_file(hub_dir, GEMMA.filename + ".incomplete", 1000 ** 3)
    out = _run("gemma confirm", tmp_path)
    assert not part.exists()
    assert not leftover.exists()
    assert "Deleted" in out


def test_shared_mmproj_survives_while_a_family_member_remains(tmp_path: Path) -> None:
    # Two Gemma quants share one vision sidecar. Deleting one quant must
    # keep the projector; deleting the last one removes it.
    _make_model(tmp_path, GEMMA)
    q8 = _make_model(tmp_path, GEMMA_Q8)
    mmproj = _make_file(tmp_path, GEMMA.mmproj_filename, int(1.2 * 1000 ** 3))

    _run("gemma-q8 confirm", tmp_path)
    assert not q8.exists()
    assert mmproj.exists(), "shared sidecar must survive while gemma remains"

    _run("gemma confirm", tmp_path)
    assert not (tmp_path / GEMMA.filename).exists()
    assert not mmproj.exists(), "last family member takes the sidecar with it"


def test_orphaned_mmproj_is_listed_and_deletable(tmp_path: Path) -> None:
    mmproj = _make_file(tmp_path, QWEN.mmproj_filename, 900 * 1000 ** 2)
    out = _run("", tmp_path)
    assert "vision sidecar" in out
    _run(f"{QWEN.mmproj_filename} confirm", tmp_path)
    assert not mmproj.exists()


# ── in-use refusal ───────────────────────────────────────────────────


def test_currently_loaded_model_is_refused_even_with_confirm(tmp_path: Path) -> None:
    p = _make_model(tmp_path, GEMMA)
    cfg = _config(model=str(p))
    out = _run("gemma confirm", tmp_path, cfg)
    assert p.exists()
    assert "Can't delete" in out
    assert "/model" in out  # tells the user how to unblock


def test_in_use_refusal_happens_before_the_confirm_step_too(tmp_path: Path) -> None:
    p = _make_model(tmp_path, GEMMA)
    out = _run("gemma", tmp_path, _config(model=str(p)))
    assert "Can't delete" in out
    assert "Nothing has been deleted" not in out  # refusal, not a preview
    assert p.exists()


def test_current_models_vision_sidecar_is_refused(tmp_path: Path) -> None:
    # Only the sidecar is on disk (main GGUF missing), but config still
    # points at the Gemma model — the projector must not be deletable.
    mmproj = _make_file(tmp_path, GEMMA.mmproj_filename, int(1.2 * 1000 ** 3))
    cfg = _config(model=str(tmp_path / GEMMA.filename))
    out = _run(f"{GEMMA.mmproj_filename} confirm", tmp_path, cfg)
    assert mmproj.exists()
    assert "Can't delete" in out


def test_in_flight_download_is_refused(tmp_path: Path, monkeypatch) -> None:
    import localcode.bootstrap as bootstrap_mod

    p = _make_model(tmp_path, GEMMA, complete=False)  # partial: mid-download
    monkeypatch.setattr(
        bootstrap_mod,
        "list_active_downloads",
        lambda: [{"model_key": GEMMA.filename, "name": GEMMA.name,
                  "progress_pct": 42, "status": "downloading"}],
    )
    out = _run("gemma confirm", tmp_path)
    assert p.exists()
    assert "Can't delete" in out
    assert "still downloading" in out


def test_in_use_reason_is_none_for_a_safe_model(tmp_path: Path) -> None:
    _make_model(tmp_path, QWEN)
    entries = list_installed(tmp_path)
    assert in_use_reason(entries[0], _config(model="")) is None


# ── TUI registration ─────────────────────────────────────────────────


def test_delete_is_a_registered_slash_command() -> None:
    from localcode.tui.screens.chat import _SLASH_COMMANDS, _is_known_command

    names = [name for name, _desc in _SLASH_COMMANDS]
    assert "/delete" in names
    desc = dict(_SLASH_COMMANDS)["/delete"]
    assert desc, "palette entry needs help text"
    assert _is_known_command("/delete")
    assert _is_known_command("/delete gemma confirm")
