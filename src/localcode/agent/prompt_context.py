"""Prompt assembly helpers for the agent loop."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import (
    REASONING_RULES,
    SYSTEM_PROMPT,
    _load_project_instructions,
    model_identity_line,
    project_stack_line,
)
from .sections import Section, SectionContext, compose_system_prompt, default_sections
__all__ = [
    "PromptBuildResult",
    "build_agent_system_prompt",
    "build_task_goal_block",
    "build_dynamic_skills_block",
    "build_target_grounding_block",
    "build_incremental_milestones_block",
]


@dataclass
class PromptBuildResult:
    system_prompt: str
    selected_skills: list[str]
    selected_skill_origins: list[str]
    selected_skill_chars: int
    skill_candidates: list[dict[str, str]]


def build_agent_system_prompt(
    *,
    app: Any,
    user_text: str,
    goal_state: Any,
    task_state: Any,
    base_system_prompt: str,
    network_status: str,
    use_thinking: bool,
) -> PromptBuildResult:
    project_instructions = _load_project_instructions(app.repo_root)
    task_goal_block = build_task_goal_block(user_text, goal_state, task_state)
    target_grounding_block = build_target_grounding_block(app.repo_root, goal_state)
    milestones_block = build_incremental_milestones_block(goal_state)
    skills_block, skill_names, skill_origins, skill_chars, skill_candidates = (
        build_dynamic_skills_block(app, user_text)
    )
    # Tell the model which local model/quant it's actually running on, so
    # "which model are you using?" gets a correct answer instead of a guess.
    # Derived per-session from the active model (config.runtime.model →
    # catalog friendly name + quant). Folded into the front of the
    # network_status slot so it renders right after the "Working directory"
    # line, with no change to the section registry. Empty when no model is
    # configured (keeps the prompt prefix byte-identical for that case).
    identity_line = ""
    try:
        identity_line = model_identity_line(str(app.config.runtime.model or ""))
    except Exception:
        identity_line = ""
    # Name the detected project stack (one line) so the model writes code
    # in the project's actual language/conventions instead of falling back
    # to Python idioms. Cheap, marker-file based, and best-effort: any
    # failure leaves the slot empty so the cached prefix is unaffected.
    stack_line = ""
    try:
        stack_line = project_stack_line(app.repo_root)
    except Exception:
        stack_line = ""
    network_status_with_identity = identity_line + stack_line + network_status

    def _render_caller_template(ctx: SectionContext) -> str:
        return base_system_prompt.format(
            cwd=ctx.cwd,
            project_instructions=ctx.project_instructions,
            network_status=ctx.network_status,
            skills_block=ctx.skills_block,
            reasoning_rules=ctx.reasoning_rules,
        )

    section_ctx = SectionContext(
        cwd=app.repo_root,
        project_instructions=project_instructions,
        network_status=network_status_with_identity,
        skills_block=skills_block,
        reasoning_rules=REASONING_RULES if use_thinking else "",
    )
    using_default_prompt = base_system_prompt == SYSTEM_PROMPT
    system = compose_system_prompt(
        section_ctx,
        sections=default_sections()
        if using_default_prompt
        else [Section(id="caller_template", render=_render_caller_template, cacheable=False)],
        emit_cache_marker=using_default_prompt,
    )
    return PromptBuildResult(
        system_prompt=system + task_goal_block + target_grounding_block + milestones_block,
        selected_skills=skill_names,
        selected_skill_origins=skill_origins,
        selected_skill_chars=skill_chars,
        skill_candidates=skill_candidates,
    )


def build_target_grounding_block(repo_root: Any, goal_state: Any) -> str:
    """One short block that pins the target directory for build-type turns.

    Grounds the model so it stops thrashing between the repo root and a subdir.
    Returned only for build_app-shaped goals; empty otherwise so the cached
    prompt prefix is unaffected for ordinary turns.
    """
    goal_type = str(getattr(goal_state, "goal_type", "") or "")
    if goal_type != "build_app":
        return ""
    try:
        root = str(Path(repo_root).resolve())
    except Exception:
        root = str(repo_root or "")
    if not root:
        return ""
    return (
        "\n\nTarget location (work here, do not drift):\n"
        f"- The project root is: {root}\n"
        "- Create and edit all files under that root. Do not switch between the "
        "root and an unrelated subdirectory between steps; decide once and stay there.\n"
    )


def build_incremental_milestones_block(goal_state: Any) -> str:
    """Lightweight guidance to build a large app in verifiable milestones.

    For a build_app goal, instruct the model to make incremental, verified
    progress — scaffold first, then one feature at a time, building/running
    between steps — instead of emitting an entire app in a single giant
    write (which is what hits the per-round token wall and lands unverified
    code). This is intentionally a prompt-context guidance block rather than
    a mechanism: cheap, no per-round bookkeeping, and the build-verification
    nudge in the loop already enforces the "build before finishing" half.

    Empty for non-build_app goals so the cached prompt prefix is unaffected.
    """
    goal_type = str(getattr(goal_state, "goal_type", "") or "")
    if goal_type != "build_app":
        return ""
    return (
        "\n\nBuild in verifiable milestones (do not emit the whole app at once):\n"
        "1. SCAFFOLD: create the minimal runnable skeleton first — entrypoint, "
        "project/config file, and a placeholder that starts. Run it to confirm "
        "it launches before adding features.\n"
        "2. ONE FEATURE AT A TIME: add a single feature per step, in small files. "
        "After each feature, build/typecheck or run the relevant check and FIX "
        "any error before moving to the next.\n"
        "3. KEEP IT RUNNABLE: never leave the project in a broken state between "
        "steps. If a step is too large for one tool call, split it across calls; "
        "don't drop scope.\n"
        "4. FINISH: only report done after a final build/run passes with no "
        "errors. Don't claim it works without having run it.\n"
    )


def build_task_goal_block(user_text: str, goal_state: Any, task_state: Any) -> str:
    # The general agentic goal block ("Current goal: X / Continue until
    # complete and verified" plus stage guidance, port hints, build_app
    # phase ladder, user-requested-features checklist) was removed
    # 2026-04-29. None of it measurably prevented the failures it was
    # added to mitigate (loops, over-verification, premature stops on
    # Qwen3.6 IQ2_M); mainstream agentic CLIs don't do it; and the
    # "Continue until complete" framing actively pushed the model into
    # eager exploration on trivial inputs ("hi" → list_files + 3× read
    # README). Only one piece survived: the question-type protective
    # wording, which has a regression test and prevents auto-resume of
    # a prior failed/blocked task on a clarifying question.
    if goal_state.goal_type == "question":
        block = (
            "\n\nCurrent goal:\n"
            f"- Answer this question directly: {user_text.strip()}\n"
            "- This is a question/diagnostic turn, not permission to resume prior coding work.\n"
            "- Do not continue, rebuild, edit, launch, or verify a previous task unless the user explicitly asks you to do that.\n"
            "- If local evidence is needed, inspect only the smallest relevant files/logs, then answer.\n"
        )
        if task_state is not None:
            status = getattr(task_state, "status", "") or ""
            slug = getattr(task_state, "task_slug", "") or ""
            if status or slug:
                block += f"- Previous task context: {slug or 'task'} status={status or 'unknown'}.\n"
        return block
    return ""


def build_dynamic_skills_block(
    app: Any,
    user_text: str,
) -> tuple[str, list[str], list[str], int, list[dict[str, str]]]:
    try:
        from ..skills import (
            dynamic_skill_block,
            dynamic_skill_candidates,
            load_registry,
            select_dynamic_skills,
        )

        registry = load_registry(app.repo_root)
        catalog = registry.listing()
        catalog_block = (
            "\n\n## Available Skills\n"
            "Load a matching recipe with the skill tool; full bodies are deferred.\n"
            + catalog
            if catalog else ""
        )
        recent_tools = list(getattr(app, "_recent_tool_names", []) or [])
        last_failed_tool = str(getattr(app, "_last_failed_tool_name", "") or "")
        candidates = dynamic_skill_candidates(
            user_text,
            recent_tools=recent_tools,
            last_failed_tool=last_failed_tool,
        )
        selected = select_dynamic_skills(
            user_text,
            registry,
            recent_tools=recent_tools,
            last_failed_tool=last_failed_tool,
        )
        # Gate on selected count rather than letting dynamic_skill_block
        # return its own placeholder/header for an empty list. With no
        # selected skills (the steady-state today: events show
        # selected_count=0 every turn), the prompt template's
        # {skills_block} slot becomes an empty string — saves a few
        # hundred bytes of header/scaffolding per round and, more
        # importantly, keeps the prompt prefix byte-identical to other
        # turns so llama.cpp's prefix-cache can hit.
        if not selected:
            return (
                catalog_block,
                [],
                [],
                0,
                candidates,
            )
        block = catalog_block + dynamic_skill_block(selected)
        return (
            block,
            [skill.name for skill in selected],
            [skill.origin for skill in selected],
            len(block),
            candidates,
        )
    except Exception:
        return "", [], [], 0, []
