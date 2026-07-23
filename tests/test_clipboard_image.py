"""Tests for clipboard image paste plumbing.

Covers:
- compose_messages(..., images=[...]) builds OpenAI-compatible image parts
- read_clipboard_png() round-trips a real PNG through the macOS clipboard
  (guarded to darwin — skipped elsewhere since it shells to osascript)
"""

from __future__ import annotations

import base64
import subprocess
import sys

import pytest

from localcode.composer import compose_messages, _build_user_message
from localcode.models import GEMMA_PROFILES
from localcode.tui.clipboard_image import read_clipboard_png


# A minimal valid 1x1 transparent PNG.
_ONE_PX_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhV"
    "AAAAAElFTkSuQmCC"
)


def _vision_profile():
    return [p for p in GEMMA_PROFILES.values() if p.supports_vision][0]


def _text_only_profile():
    # Fabricate a non-vision profile by copying a vision one with the flag off.
    from dataclasses import replace
    return replace(_vision_profile(), supports_vision=False)


def test_compose_messages_with_images_builds_image_url_parts():
    profile = _vision_profile()
    img = _ONE_PX_PNG_B64
    messages = compose_messages(
        profile,
        system_prompt="sys",
        context_block="",
        conversation=[],
        user_text="what is in this image?",
        images=[img],
    )
    user_msg = messages[-1]
    assert user_msg["role"] == "user"
    content = user_msg["content"]
    assert isinstance(content, list), "image message content must be a parts list"
    image_parts = [p for p in content if p.get("type") == "image_url"]
    text_parts = [p for p in content if p.get("type") == "text"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == f"data:image/png;base64,{img}"
    assert text_parts and text_parts[0]["text"] == "what is in this image?"


def test_compose_messages_without_images_is_plain_text():
    profile = _vision_profile()
    messages = compose_messages(
        profile, "sys", "", [], "hello", images=None,
    )
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_build_user_message_text_only_profile_ignores_images():
    # A non-vision model must never get image parts even if images are passed.
    profile = _text_only_profile()
    msg = _build_user_message("hi", [_ONE_PX_PNG_B64], profile, "llama_cpp")
    assert msg == {"role": "user", "content": "hi"}


@pytest.mark.skipif(
    sys.platform != "darwin" or __import__("os").environ.get("LOCALCODE_RUN_CLIPBOARD_TESTS") != "1",
    reason="set LOCALCODE_RUN_CLIPBOARD_TESTS=1 on macOS to mutate the system clipboard",
)
def test_read_clipboard_png_round_trip(tmp_path):
    # Write a real PNG to disk, load it onto the clipboard as PNG image
    # DATA (not a file URL), then assert the reader returns valid PNG bytes.
    png_path = tmp_path / "one.png"
    png_path.write_bytes(base64.b64decode(_ONE_PX_PNG_B64))

    subprocess.run(
        [
            "osascript",
            "-e",
            f'set the clipboard to (read (POSIX file "{png_path}") as «class PNGf»)',
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    data = read_clipboard_png()
    assert data is not None, "expected PNG bytes from the clipboard"
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "returned bytes are not a PNG"


def test_read_clipboard_png_returns_none_off_darwin(monkeypatch):
    # On a non-darwin platform the reader must be a safe no-op.
    monkeypatch.setattr(sys, "platform", "linux")
    assert read_clipboard_png() is None


def test_vision_enabled_config_round_trip(tmp_path, monkeypatch):
    # The vision toggle must persist across save/load — this is the fix
    # that stops OFF→ON from re-downloading the projector (state now lives
    # in config, not in file presence).
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALCODE_VISION_ENABLED", raising=False)
    from localcode.config import load_config, save_config

    cfg = load_config()
    assert cfg.runtime.vision_enabled is False  # default off
    cfg.runtime.vision_enabled = True
    save_config(cfg)

    reloaded = load_config()
    assert reloaded.runtime.vision_enabled is True
