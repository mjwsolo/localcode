"""Prompt assembly helpers for the agent loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompts import REASONING_RULES, SYSTEM_PROMPT, _load_project_instructions
from .sections import Section, SectionContext, compose_system_prompt, default_sections


@dataclass(slots=True)
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
    skills_block, skill_names, skill_origins, skill_chars, skill_candidates = (
        build_dynamic_skills_block(app, user_text)
    )
    notebook_block = ""

    def _render_caller_template(ctx: SectionContext) -> str:
        return base_system_prompt.format(
            cwd=ctx.cwd,
            project_instructions=ctx.project_instructions,
            network_status=ctx.network_status,
            skills_block=ctx.skills_block,
            reasoning_rules=ctx.reasoning_rules,
            notebook_block=ctx.notebook_block,
        )

    section_ctx = SectionContext(
        cwd=app.repo_root,
        project_instructions=project_instructions,
        network_status=network_status,
        skills_block=skills_block,
        reasoning_rules=REASONING_RULES if use_thinking else "",
        notebook_block=notebook_block,
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
        system_prompt=system + task_goal_block,
        selected_skills=skill_names,
        selected_skill_origins=skill_origins,
        selected_skill_chars=skill_chars,
        skill_candidates=skill_candidates,
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
                "",
                [],
                [],
                0,
                candidates,
            )
        block = dynamic_skill_block(selected)
        return (
            block,
            [skill.name for skill in selected],
            [skill.origin for skill in selected],
            len(block),
            candidates,
        )
    except Exception:
        return "", [], [], 0, []
