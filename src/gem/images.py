"""Gem image support — clipboard capture, file loading, base64 encoding for vision."""
from __future__ import annotations

import base64
import mimetypes
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}

# Max image size to send to model (5MB base64 ≈ 3.75MB raw)
MAX_IMAGE_BYTES = 5 * 1024 * 1024


@dataclass(slots=True)
class ImageData:
    """An image ready to send to the model."""
    base64_data: str
    mime_type: str
    source: str  # "clipboard", "file:/path", "screenshot"
    width: int = 0
    height: int = 0

    @property
    def size_kb(self) -> int:
        return len(self.base64_data) * 3 // 4 // 1024


# ── Clipboard image capture ──────────────────────────────────────────────

def clipboard_has_image() -> bool:
    """Check if the system clipboard contains an image."""
    system = platform.system()
    if system == "Darwin":
        # macOS: use osascript to check clipboard type
        try:
            result = subprocess.run(
                ["osascript", "-e", 'clipboard info'],
                capture_output=True, text=True, timeout=3,
            )
            return "«class PNGf»" in result.stdout or "«class TIFF»" in result.stdout
        except Exception:
            return False
    elif system == "Linux":
        # Linux: check xclip for image targets
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
                capture_output=True, text=True, timeout=3,
            )
            return "image/png" in result.stdout or "image/jpeg" in result.stdout
        except Exception:
            return False
    return False


def read_clipboard_image() -> ImageData | None:
    """Read an image from the system clipboard. Returns None if no image."""
    system = platform.system()

    if system == "Darwin":
        return _read_clipboard_macos()
    elif system == "Linux":
        return _read_clipboard_linux()
    elif system == "Windows":
        return _read_clipboard_windows()
    return None


def _read_clipboard_macos() -> ImageData | None:
    """Read clipboard image on macOS using osascript + pngpaste or pbpaste."""
    # Try pngpaste first (brew install pngpaste)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            ["pngpaste", tmp.name],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0 and os.path.getsize(tmp.name) > 0:
            return _load_image_file(tmp.name, "clipboard")
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Fallback: use osascript to save clipboard image
    try:
        script = '''
        set theFile to POSIX file "%s"
        try
            set imageData to the clipboard as «class PNGf»
            set fileRef to open for access theFile with write permission
            write imageData to fileRef
            close access fileRef
            return "ok"
        on error
            return "no image"
        end try
        ''' % tmp.name
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=5,
        )
        if "ok" in result.stdout and os.path.getsize(tmp.name) > 0:
            return _load_image_file(tmp.name, "clipboard")
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return None


def _read_clipboard_linux() -> ImageData | None:
    """Read clipboard image on Linux using xclip."""
    try:
        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            data = result.stdout
            if len(data) > MAX_IMAGE_BYTES:
                return None
            b64 = base64.b64encode(data).decode("ascii")
            return ImageData(base64_data=b64, mime_type="image/png", source="clipboard")
    except Exception:
        pass
    return None


def _read_clipboard_windows() -> ImageData | None:
    """Read clipboard image on Windows using PowerShell."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        ps_script = (
            f'Add-Type -AssemblyName System.Windows.Forms; '
            f'$img = [System.Windows.Forms.Clipboard]::GetImage(); '
            f'if ($img) {{ $img.Save("{tmp.name}"); echo "ok" }} '
            f'else {{ echo "no" }}'
        )
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True, text=True, timeout=5,
        )
        if "ok" in result.stdout and os.path.getsize(tmp.name) > 0:
            return _load_image_file(tmp.name, "clipboard")
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return None


# ── File-based image loading ─────────────────────────────────────────────

def load_image_from_path(path: str | Path) -> ImageData | None:
    """Load an image from a file path. Returns None if not a valid image."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return None
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return _load_image_file(str(p), f"file:{p}")


def _load_image_file(path: str, source: str) -> ImageData | None:
    """Read a file and return ImageData."""
    try:
        data = Path(path).read_bytes()
        if len(data) > MAX_IMAGE_BYTES:
            return None
        if len(data) < 100:
            return None
        b64 = base64.b64encode(data).decode("ascii")
        mime, _ = mimetypes.guess_type(path)
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        return ImageData(base64_data=b64, mime_type=mime, source=source)
    except Exception:
        return None


# ── Screenshot capture ───────────────────────────────────────────────────

def take_screenshot(region: str = "full") -> ImageData | None:
    """Capture a screenshot. region: 'full' or 'selection'."""
    system = platform.system()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()

    try:
        if system == "Darwin":
            cmd = ["screencapture"]
            if region == "selection":
                cmd.append("-i")  # interactive selection
            else:
                cmd.append("-x")  # silent
            cmd.append(tmp.name)
            subprocess.run(cmd, timeout=30)
        elif system == "Linux":
            if region == "selection":
                subprocess.run(["gnome-screenshot", "-a", "-f", tmp.name], timeout=30)
            else:
                subprocess.run(["gnome-screenshot", "-f", tmp.name], timeout=30)
        else:
            return None

        if os.path.getsize(tmp.name) > 0:
            return _load_image_file(tmp.name, "screenshot")
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return None


# ── Message formatting for Ollama ────────────────────────────────────────

def build_image_message(
    text: str,
    images: list[ImageData],
) -> dict:
    """Build an Ollama-compatible message with images.

    Ollama format:
    {
        "role": "user",
        "content": "What's in this image?",
        "images": ["base64data1", "base64data2"]
    }
    """
    return {
        "role": "user",
        "content": text,
        "images": [img.base64_data for img in images],
    }


def detect_image_paths_in_text(text: str) -> list[str]:
    """Find file paths in text that look like images."""
    import re
    # Match common path patterns
    paths = re.findall(r'(?:^|\s)([~/.]?[\w./-]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff))', text, re.IGNORECASE)
    return paths
