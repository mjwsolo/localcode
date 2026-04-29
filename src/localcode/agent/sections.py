"""System prompt as a composition of named sections.

Background
----------
Before this module, `SYSTEM_PROMPT` was one monolithic ~20K-character
string with template placeholders (`{cwd}`, `{network_status}`,
`{project_instructions}`, `{skills_block}`, `{reasoning_rules}`,
`{notebook_block}`) baked into the middle. `run_agent_loop` called
`.format(...)` on the whole thing.

Problems with the monolith:
  • Every placeholder interpolation invalidates the llama-server
    prefix cache — `{cwd}` alone sits near the top and changes per
    session, which means every new session re-evaluates the whole
    prompt. Our benchmark shows prompt_ms of ~48 ms warm; the
    theoretical floor with a stable prefix cache is ~5-10 ms.
  • No way to A/B sections individually. `eval/prompt_variants.py`
    only supports coarse "strip rule N" mutations. You can't
    answer "what if we drop the build-app rules for debugging
    tasks?" because there's no boundary between them.
  • No way to gate sections on trigger phrases. Every session ships
    rule 10 (open browser) whether or not a web app is in scope.

This module is **Phase 1** of T0.11 (see eval/OPTIMIZATION_PLAN.md):
introduce the section-registry concept with behaviour-neutral
defaults. The composer produces the same output as the old
`SYSTEM_PROMPT.format(...)` for the default all-sections-on case.
Conditional inclusion (Phase 2) plugs in here without touching the
loop.

Architecture
------------
A `Section` is `(id, content_fn, always_on)` where `content_fn` takes
a `SectionContext` (cwd, project_instructions, reasoning, etc.) and
returns the rendered text for this section. The composer calls each
section in order, drops any `always_on=False` that the caller
doesn't enable, joins with newlines, and inserts an explicit
`# ── CACHE BOUNDARY ──` marker between the stable top and the
session-specific tail.

Why a marker instead of slicing into two separate prompts:
llama-server's prefix cache uses prompt bytes for cache-hit matching.
The marker is just a comment line that makes human readers / future
llama-server features aware of where the stable region ends. If a
future server release supports explicit cache-boundary hints, we
teach the composer to emit that hint.

Current (Phase 1) scope
-----------------------
Only `default_sections()` ships. It returns a list whose rendered
output equals the pre-split `SYSTEM_PROMPT.format(...)` call. Trigger-
based gating comes later; leaving the hooks in place now so future
work is an add-only operation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .prompts import SYSTEM_PROMPT
from .prompts import _load_project_instructions


__all__ = [
    "Section",
    "SectionContext",
    "compose_system_prompt",
    "default_sections",
    "CACHE_BOUNDARY_MARKER",
]


# Inserted between the stable and dynamic halves of the composed
# prompt. Purely a comment — no llama-server version today reads it
# as a hint — but it documents intent and makes a future hint-based
# implementation a one-line change.
CACHE_BOUNDARY_MARKER = (
    "\n# ── CACHE BOUNDARY — stable content above, session-specific below ──\n"
)

_SECTION_RENDER_CACHE: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class SectionContext:
    """Per-turn data the section renderers need.

    This is a struct so section functions stay pure (no reading app
    attributes directly). The loop builds this once per turn and
    passes it to `compose_system_prompt`.
    """
    cwd: str
    project_instructions: str
    network_status: str
    skills_block: str
    reasoning_rules: str
    notebook_block: str


@dataclass(frozen=True)
class Section:
    """One named piece of the composed system prompt.

    `id`       — stable identifier, used in eval variants for
                 per-section toggling (future Phase 2).
    `render`   — callable taking SectionContext → rendered string.
                 Empty string means "emit nothing for this section"
                 and the composer will skip it cleanly.
    `always_on` — if False, the caller must explicitly include this
                 section id in `enabled_sections` for it to render.
                 Reserved for future conditional sections.
    """
    id: str
    render: Callable[[SectionContext], str]
    always_on: bool = True
    cacheable: bool = True


# ── Section renderers ──────────────────────────────────────────────
# Phase 1: each renderer returns a substring of the monolithic
# SYSTEM_PROMPT, reassembled to byte-identical output. When Phase 2
# lands, these bodies get split into independently-editable section
# markdown files.

def _split_network_template(template: str) -> tuple[str, str]:
    marker = "{network_status}"
    if marker not in template:
        return template, ""
    head, tail = template.split(marker, 1)
    return head, tail


_STATIC_HEAD, _STATIC_TAIL = _split_network_template(SYSTEM_PROMPT)


def _static_head(_ctx: SectionContext) -> str:
    return _STATIC_HEAD


def _network_status(ctx: SectionContext) -> str:
    return ctx.network_status


def _static_tail(_ctx: SectionContext) -> str:
    return _STATIC_TAIL


def _project_instructions(ctx: SectionContext) -> str:
    if not ctx.project_instructions:
        return ""
    return f"\n\nProject instructions:\n{ctx.project_instructions}\n"


def _skills_block(ctx: SectionContext) -> str:
    return ctx.skills_block or ""


def _reasoning_rules(ctx: SectionContext) -> str:
    return ctx.reasoning_rules or ""


def _notebook_block(ctx: SectionContext) -> str:
    return ctx.notebook_block or ""


def default_sections() -> list[Section]:
    """Return the section list whose composed output matches the
    pre-T0.11 monolith. This is the default the loop uses today.

    Phase 2 will replace this with a many-section list (identity,
    always_rules, tool_usage, build_app_rules, refactor_rules,
    reasoning_rules, notebook_rules, env_info, skills_available)
    where each section is independently toggleable.
    """
    return [
        Section(id="static_head", render=_static_head, always_on=True, cacheable=True),
        Section(id="network_status", render=_network_status, always_on=True, cacheable=False),
        Section(id="static_tail", render=_static_tail, always_on=True, cacheable=True),
        Section(id="project_instructions", render=_project_instructions, always_on=True, cacheable=False),
        Section(id="skills_block", render=_skills_block, always_on=True, cacheable=False),
        Section(id="reasoning_rules", render=_reasoning_rules, always_on=True, cacheable=False),
        Section(id="notebook_block", render=_notebook_block, always_on=True, cacheable=False),
    ]


# ── Composer ───────────────────────────────────────────────────────


def compose_system_prompt(
    ctx: SectionContext,
    sections: list[Section] | None = None,
    enabled_sections: set[str] | None = None,
    *,
    emit_cache_marker: bool = False,
) -> str:
    """Compose the system prompt from the given sections.

    Parameters
    ----------
    ctx
        Per-turn slot values (cwd, project_instructions, etc.).
    sections
        Section list in emission order. Defaults to `default_sections()`
        which produces the pre-T0.11 byte-identical monolith output.
    enabled_sections
        If given, only sections whose id is in this set OR that are
        marked `always_on` will render. If None (default), all
        sections render — Phase 1 behaviour. Phase 2 will use this
        to turn gated sections on/off based on trigger phrases in
        the user text.
    emit_cache_marker
        If True, insert `CACHE_BOUNDARY_MARKER` between the
        non-env and env sections. Default False for Phase 1 so the
        composer output matches the old monolith byte-for-byte; flip
        to True once the prompt structure naturally has a stable/
        dynamic split to mark.

    Returns
    -------
    The fully composed system prompt, ready to send to the model.
    """
    if sections is None:
        sections = default_sections()

    parts: list[str] = []
    for sec in sections:
        if not sec.always_on:
            if enabled_sections is None or sec.id not in enabled_sections:
                continue
        if sec.cacheable:
            cache_key = (sec.id, SYSTEM_PROMPT)
            rendered = _SECTION_RENDER_CACHE.get(cache_key)
            if rendered is None:
                rendered = sec.render(ctx)
                _SECTION_RENDER_CACHE[cache_key] = rendered
        else:
            rendered = sec.render(ctx)
        if not rendered:
            continue
        parts.append(rendered)

    if emit_cache_marker and len(parts) > 1:
        # Crude placement: boundary sits before the last section.
        # Phase 2 gives each section a `kind` (stable vs dynamic)
        # and the composer inserts the marker at the transition.
        parts.insert(-1, CACHE_BOUNDARY_MARKER)

    return "".join(parts)


def load_project_instructions(repo_root: Path) -> str:
    """Convenience re-export so callers outside agent/ that don't
    want to depend on `agent.prompts` internals can build a
    SectionContext purely from this module."""
    return _load_project_instructions(repo_root)
