from __future__ import annotations

from .models import ModelProfile


def compose_messages(
    profile: ModelProfile,
    system_prompt: str,
    context_block: str,
    conversation: list[dict[str, str]],
    user_text: str,
    images: list[str] | None = None,
    provider: str = "llama_cpp",
) -> list[dict]:
    """Compose the message list for the model.

    Args:
        images: Optional list of base64-encoded image strings, sent as
            llama-server's OpenAI-compatible `image_url` multipart content.
    """
    # Build the user message with appropriate image format
    user_msg = _build_user_message(user_text, images, profile, provider)

    if profile.supports_native_system:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        # Only inject repo context for substantive tasks
        if context_block.strip() and len(user_text) > 20:
            messages.append({"role": "user", "content": f"[Context]\n{context_block}"})
            messages.append({"role": "assistant", "content": "Got it."})
        messages.extend(conversation)
        messages.append(user_msg)
        return messages

    # No native system role — inject instructions into first user message
    injected_user_prefix = (
        "LocalCode operating instructions:\n"
        f"{system_prompt}\n\n"
        "Repository context:\n"
        f"{context_block}\n\n"
        "Follow the instructions above while answering the next user message."
    )
    if conversation:
        first = conversation[0]
        if first.get("role") == "user" and str(first.get("content", "")).startswith("LocalCode operating instructions:\n"):
            adjusted = conversation.copy()
        else:
            adjusted = [{"role": "user", "content": injected_user_prefix}, *conversation]
    else:
        adjusted = [{"role": "user", "content": injected_user_prefix}]
    return [*adjusted, user_msg]


def _build_user_message(
    text: str,
    images: list[str] | None,
    profile: ModelProfile,
    provider: str,
) -> dict:
    """Build a user message with llama-server's OpenAI-compatible image parts."""
    if not images or not profile.supports_vision:
        return {"role": "user", "content": text}

    content_parts: list[dict] = []
    for img_b64 in images:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })
    content_parts.append({"type": "text", "text": text})
    return {"role": "user", "content": content_parts}
