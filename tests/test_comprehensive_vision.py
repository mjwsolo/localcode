"""Image (vision) input coverage.

The user-facing feature: attach an image, the model sees it. Under the hood
that's `composer._build_user_message`, which must emit the right wire format
per provider (Ollama vs llama_cpp vs HF/MLX). These tests pin every branch
plus the capability flags on the vision-capable profiles (including the new
Gemma 4 12B).
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode.composer import compose_messages
from localcode.models import GEMMA_PROFILES

# A 1x1 PNG, base64 — stands in for a real screenshot.
TINY_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nFAKEIMAGEDATA").decode()

VISION_PROFILE = GEMMA_PROFILES["gemma4-12b"]  # dense 12B, vision + audio


def _user_msg(provider: str, images=None):
    msgs = compose_messages(
        profile=VISION_PROFILE,
        system_prompt="sys",
        context_block="",
        conversation=[],
        user_text="what is in this image?",
        images=images,
        provider=provider,
    )
    return msgs[-1]  # the composed user message is always last


# ── Per-provider wire format ────────────────────────────────────────


def test_ollama_uses_images_field():
    msg = _user_msg("ollama", images=[TINY_PNG_B64])
    assert msg["role"] == "user"
    assert msg["images"] == [TINY_PNG_B64]
    assert msg["content"] == "what is in this image?"


def test_llama_cpp_uses_openai_image_url_parts():
    msg = _user_msg("llama_cpp", images=[TINY_PNG_B64])
    parts = msg["content"]
    assert isinstance(parts, list)
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    text_parts = [p for p in parts if p.get("type") == "text"]
    assert len(image_parts) == 1 and len(text_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert TINY_PNG_B64 in image_parts[0]["image_url"]["url"]


@pytest.mark.parametrize("provider", ["huggingface-local", "mlx-local"])
def test_hf_mlx_use_multipart_image_parts(provider):
    msg = _user_msg(provider, images=[TINY_PNG_B64])
    parts = msg["content"]
    assert isinstance(parts, list)
    image_parts = [p for p in parts if p.get("type") == "image"]
    assert len(image_parts) == 1
    assert image_parts[0]["image"].startswith("data:image/png;base64,")


def test_multiple_images_all_included():
    msg = _user_msg("llama_cpp", images=[TINY_PNG_B64, TINY_PNG_B64, TINY_PNG_B64])
    image_parts = [p for p in msg["content"] if p.get("type") == "image_url"]
    assert len(image_parts) == 3


# ── Negative / capability cases ─────────────────────────────────────


def test_no_images_yields_plain_text_message():
    msg = _user_msg("llama_cpp", images=None)
    assert msg["content"] == "what is in this image?"
    assert "images" not in msg


def test_non_vision_profile_drops_images():
    """A text-only profile must not smuggle images into the payload."""
    text_only = GEMMA_PROFILES["gemma4-e2b"]
    assert text_only.supports_vision  # sanity: e2b is actually vision-capable
    # Build a genuinely non-vision profile by copying with the flag off.
    from dataclasses import replace
    no_vision = replace(text_only, supports_vision=False)
    msg = compose_messages(
        profile=no_vision, system_prompt="s", context_block="",
        conversation=[], user_text="hi", images=[TINY_PNG_B64], provider="llama_cpp",
    )[-1]
    assert msg["content"] == "hi"
    assert "images" not in msg


def test_12b_profile_declares_vision_and_audio():
    assert VISION_PROFILE.supports_vision is True
    assert VISION_PROFILE.supports_audio is True
