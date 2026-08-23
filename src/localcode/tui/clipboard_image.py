"""Read an image off the OS clipboard as PNG bytes.

Terminals never hand an application image data through a Cmd+V / Ctrl+V
paste — the OS only pastes text, so an image paste arrives as an EMPTY
paste event. When that happens the TUI calls :func:`read_clipboard_png`
to pull the image directly out of the OS clipboard and attach it.

macOS-only. On every other platform this returns ``None`` (no
dependency, no crash) so callers can attempt it unconditionally.

Implementation notes
--------------------
We shell out to ``osascript`` (always present on macOS, no new Python
dependency). Two cases are handled:

1. Raw image data on the clipboard (e.g. a screenshot from Cmd+Shift+4
   or "Copy Image" in a browser). We coerce the clipboard to
   ``«class PNGf»`` and write the bytes to a temp file, then read them
   back in Python.
2. A copied image FILE (e.g. Finder "Copy" on a .png). The clipboard
   exposes a file URL (``«class furl»``); if it points at a common
   image extension we read that file directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def read_clipboard_png() -> bytes | None:
    """Return image bytes from the OS clipboard, or ``None``.

    Prefers a PNG coercion of raw clipboard image data; falls back to a
    copied image file. Returns ``None`` on non-macOS platforms and
    whenever the clipboard holds no image.
    """
    if sys.platform != "darwin":
        return None
    # Fast path: `pngpaste` (Homebrew) when installed — noticeably quicker
    # than round-tripping AppleScript. NEVER required: the osascript path
    # below is the dependency-free guarantee.
    data = _read_png_via_pngpaste()
    if data:
        return data
    data = _read_png_via_osascript()
    if data:
        return data
    return _read_image_from_file_url()


def _read_png_via_pngpaste() -> bytes | None:
    """Optional fast path via the `pngpaste` binary, when present."""
    import shutil
    exe = shutil.which("pngpaste")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "-"], capture_output=True, timeout=10
        )
    except Exception:
        return None
    data = result.stdout if result.returncode == 0 else None
    return data if data and _looks_like_png(data) else None


def _read_png_via_osascript() -> bytes | None:
    """Coerce raw clipboard image data to PNG and read the bytes."""
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        # AppleScript: try to grab the clipboard as PNG; bail with a
        # sentinel if there's no image. Write to a temp file we can read
        # back. `«class PNGf»` is the PNG pasteboard type; `«class furl»`
        # (used below) is a file URL.
        script = (
            "try\n"
            "\tset pngData to (the clipboard as «class PNGf»)\n"
            "on error\n"
            "\treturn \"NO_IMAGE\"\n"
            "end try\n"
            f"set outFile to (POSIX file \"{tmp_path}\")\n"
            "try\n"
            "\tset fh to open for access outFile with write permission\n"
            "\tset eof fh to 0\n"
            "\twrite pngData to fh\n"
            "\tclose access fh\n"
            "on error\n"
            "\ttry\n"
            "\t\tclose access fh\n"
            "\tend try\n"
            "\treturn \"WRITE_ERROR\"\n"
            "end try\n"
            "return \"OK\"\n"
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if (result.stdout or "").strip() != "OK":
            return None
        data = Path(tmp_path).read_bytes()
        return data if _looks_like_png(data) else None
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _read_image_from_file_url() -> bytes | None:
    """Read a copied image FILE (Finder copy) from its clipboard URL."""
    script = (
        "try\n"
        "\treturn POSIX path of (the clipboard as «class furl»)\n"
        "on error\n"
        "\treturn \"\"\n"
        "end try\n"
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    path_str = (result.stdout or "").strip()
    if not path_str:
        return None
    p = Path(path_str)
    if p.suffix.lower() not in _IMAGE_EXTS or not p.is_file():
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None


def _looks_like_png(data: bytes) -> bool:
    return bool(data) and data[:8] == _PNG_MAGIC
