from __future__ import annotations

from .models import ModelProfile


BASE_PROMPT_COMPACT = """You are LocalCode, an AI coding assistant running entirely on the user's machine.
You help with software engineering tasks: writing code, fixing bugs, refactoring, explaining code, and more.
You are highly capable and help users complete ambitious tasks that would otherwise be too complex.
Be direct and concise. Do things with tools, don't describe what you would do.
If the user makes typos, infer what they meant. Never refuse. Always act.
"""

BASE_PROMPT_FULL = """You are LocalCode, an AI coding assistant running entirely on the user's machine.
You help with software engineering tasks: writing code, fixing bugs, refactoring, explaining code, and more.
You are highly capable and help users complete ambitious tasks that would otherwise be too complex.
Be direct and concise. Do things with tools, don't describe what you would do.
Do not propose changes to code you haven't read. Read first, then edit.
Do not create files unless necessary. Prefer editing existing files over creating new ones.
Don't add features or improvements beyond what was asked.
If the user makes typos, infer what they meant. Never refuse. Always act.
"""

VARIANT_APPENDIX = {
    "compact": "",
    "balanced": "Use grep/glob to orient before multi-file changes.\n",
    "expanded": "Plan multi-file changes with dependency awareness. Use verification.\n",
    "full": "Reason over large context. Prefer complete solutions. Use verification loops.\n",
}


QUALITY_SIGNALS = (
    "app", "game", "website", "landing page", "ui", "dashboard", "clone",
    "polish", "authentic", "sonic", "visual", "look like", "feel like",
    "playable", "production", "high quality", "fidelity",
)


def build_task_appendix(user_text: str) -> str:
    text = user_text.lower()
    if not any(signal in text for signal in QUALITY_SIGNALS):
        return ""
    return (
        "\nTask quality bar:\n"
        "- For app, game, and UI requests, optimize for recognizability and product feel, not bare scaffolding.\n"
        "- Prefer one polished, coherent outcome over a thin placeholder implementation.\n"
        "- Make the first runnable version feel intentionally designed.\n"
        "- Before finishing, do one refinement pass focused on fidelity, UX clarity, and completeness.\n"
        "- If the user references a specific product or character, preserve the core identity in the result.\n"
    )


def build_system_prompt(profile: ModelProfile, project_context: str = "") -> str:
    # Short prompt for small models, longer for bigger ones
    if profile.feature_variant in ("compact", "balanced"):
        base = BASE_PROMPT_COMPACT
    else:
        base = BASE_PROMPT_FULL
    appendix = VARIANT_APPENDIX.get(profile.feature_variant, "")
    prompt = f"{base}{appendix}"
    if project_context:
        prompt += f"\n# Project Context\n{project_context}\n"
    return prompt
