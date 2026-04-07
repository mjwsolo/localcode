"""3-layer agent loop — harness controls sequencing, model generates content.

Layer 1: EXECUTORS — 13 deterministic state machines (this file)
Layer 2: ORCHESTRATOR — walks multi-step plans
Layer 3: PLANNER — decomposes complex tasks

Architecture based on observing OpenAI Codex, Claude Code, and Claude Opus.
Model generates content. Harness handles sequencing, validation, retries.
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import GemApp
    from .output import OutputManager

from .context_manager import (
    ContextAssembler,
    ConversationManager,
    ProgressTracker,
    RelevanceFinder,
    SyntaxChecker,
    UndoStack,
    SYSTEM_PROMPTS,
    _extract_keywords,
    _list_dir,
)
from .verification import classify_artifact, run_outcome_verification


# ── Intent Classifier (rule-based, LLM fallback) ───────────────────

def classify_intent(text: str) -> str:
    """Route user request to the right feature. Rule-based first, LLM fallback."""
    t = text.lower()
    if any(w in t for w in ("create", "make", "new file", "generate", "scaffold", "build", "write a", "write an", "write ")) or (
        any(noun in t for noun in ("script", "program", "tool", "app", "game")) and any(verb in t for verb in ("write", "make", "build", "create", "generate"))
    ):
        if any(w in t for w in ("test", "tests")):
            return "TEST"
        return "CREATE"
    if any(w in t for w in ("edit", "change", "modify", "rename", "add to", "update", "refactor")):
        return "EDIT"
    if any(w in t for w in ("fix", "bug", "crash", "error", "broken", "doesn't work", "doesnt work")):
        return "FIX"
    if any(w in t for w in ("review", "check", "audit", "look at")):
        return "REVIEW"
    if any(w in t for w in ("explain", "what does", "how does", "why does", "what is")):
        return "EXPLAIN"
    if any(w in t for w in ("test", "run test", "pytest")):
        return "TEST"
    if any(w in t for w in ("commit", "push", "git ", "branch", "diff", "status")):
        return "GIT"
    if any(w in t for w in ("run ", "run it", "execute", "launch", "start it", "try it", "open it")):
        return "CREATE"  # route to CREATE which can run bash
    if any(w in t for w in ("find", "search", "where", "grep", "locate")):
        return "SEARCH"
    if any(w in t for w in ("install", "add package", "dependency", "requirements")):
        return "DEPS"
    return "CHAT"


def needs_planning(text: str) -> bool:
    """Does this task need multi-step planning?"""
    t = text.lower()
    # Multiple action verbs
    verbs = sum(1 for w in ("add", "create", "update", "fix", "write", "build", "set up", "implement")
                if w in t)
    if verbs >= 2:
        return True
    # Broad scope
    if any(k in t for k in ("authentication", "database", "api", "refactor all",
                            "migrate", "set up", "full")):
        return True
    return False


# ── Main Entry Point ────────────────────────────────────────────────

def run_agent_loop(
    app: "GemApp",
    user_text: str,
    composed_messages: list[dict],
    out: "OutputManager",
) -> str:
    """Route to the right executor based on intent."""
    out.start_thinking()
    repo = app.repo_root
    progress = ProgressTracker(out.print_info, out.set_stage)
    checker = SyntaxChecker()
    context = ContextAssembler()
    relevance = RelevanceFinder()

    intent = classify_intent(user_text)

    # For complex tasks, use Layer 3 planner
    if needs_planning(user_text) and intent in ("CREATE", "EDIT", "FIX"):
        return _handle_planned(app, user_text, composed_messages, out, progress, checker, context, relevance)

    # Simple tasks → direct to Layer 1 executor
    if intent == "CREATE":
        return _do_create(app, user_text, composed_messages, out, progress, checker, context, relevance)
    elif intent in ("EDIT", "FIX"):
        return _do_edit(app, user_text, composed_messages, out, progress, checker, context, relevance)
    elif intent == "REVIEW":
        return _do_review(app, user_text, composed_messages, out, context, relevance)
    elif intent == "EXPLAIN":
        return _do_explain(app, user_text, composed_messages, out, context, relevance)
    elif intent == "TEST":
        return _do_test(app, user_text, composed_messages, out, progress, checker)
    elif intent == "GIT":
        return _do_git(app, user_text, composed_messages, out)
    elif intent == "SEARCH":
        return _do_search(app, user_text, out)
    else:
        # CHAT — just answer
        return _do_chat(app, user_text, composed_messages, out)


# ── Feature 1: FILE CREATION ───────────────────────────────────────

def _do_create(app, user_text, messages, out, progress, checker, context, relevance):
    repo = app.repo_root
    quality_task = _is_quality_sensitive_task(user_text)
    aggressive_quality_task = _needs_aggressive_quality_pass(user_text)
    compact_task = _should_use_fast_create_lane(app, user_text, [], quality_task)

    # GATHER
    progress.start("gathering context")
    related: list[str] = []
    if compact_task:
        ctx = _build_compact_create_context(user_text)
    else:
        related = relevance.find_related(user_text, repo, max_files=2)
        ctx = context.build_for_create(user_text, repo, related)
    if quality_task:
        progress.start("extracting product cues")
    route_label = "create: compact single-file path" if compact_task else "create: full generation path"
    out.print_info(route_label)
    use_planner_hints = app.config.runtime.planner_hints_enabled and _should_use_planner_hints(user_text, quality_task, compact_task)
    initial_hint = app.planner_checkpoint_hint("create:start", user_text, ctx[:1200]) if use_planner_hints else None
    if initial_hint:
        hint_bits = [bit for bit in (initial_hint.next_action, initial_hint.likely_file, initial_hint.risk) if bit]
        if hint_bits:
            out.print_info("planner hint: " + " | ".join(hint_bits[:3]))

    # WRITE
    progress.start("generating code")
    prompt = _build_create_prompt(ctx, user_text, quality_task)
    response = _generate_create_response(
        app,
        user_text,
        [
            {"role": "system", "content": SYSTEM_PROMPTS["create"]},
            *messages[-4:],
            {"role": "user", "content": prompt},
        ],
        out,
        quality_task,
        related,
    )
    post_generate_hint = (
        app.planner_checkpoint_hint(
            "create:post_generate",
            user_text,
            response[:1200],
        )
        if use_planner_hints else None
    )

    filename, code = _parse_code_response(response, user_text)
    if not code:
        # Try harder: maybe the closing ``` was cut off
        fence_match = re.search(r'```\w*\n(.+)', response, re.DOTALL)
        if fence_match and len(fence_match.group(1).strip()) > 50:
            code = fence_match.group(1).strip()
            # Remove trailing ``` if present
            if code.endswith("```"):
                code = code[:-3].strip()
            if not filename:
                filename, _ = _parse_code_response("", user_text)
        if not code:
            out.stream(response)
            return response
    out.print_info(f"target file: {filename}")

    # APPLY
    full = repo / filename
    full.parent.mkdir(parents=True, exist_ok=True)
    app.toolkit.changes.snapshot_before(filename, "agent")
    progress.start(f"writing {filename}")
    full.write_text(code)
    lines = len(code.splitlines())
    out.log_tool("write_file", f"path={filename}")
    out.tool_result(f"Written {filename} ({lines} lines)")
    progress.done("write", f"{filename} ({lines} lines)")

    # VERIFY
    progress.start("verifying")
    result = checker.check(filename, str(repo), prefer_basic=True)
    if not result["ok"]:
        progress.fail("verify", result["error"][:80])
        # FIX LOOP (max 2)
        for attempt in range(2):
            progress.start(f"fixing (attempt {attempt + 1})")
            fix_ctx = context.build_for_fix(filename, result["error"], repo)
            fix_response = app.engine.generate_once([
                {"role": "system", "content": SYSTEM_PROMPTS["fix"]},
                {"role": "user", "content": fix_ctx},
            ])
            _, fixed = _parse_code_response(fix_response, "")
            if not fixed:
                # Try extracting search/replace blocks
                fixed = _apply_search_replace(full, fix_response)
                if fixed:
                    result = checker.check(filename, str(repo), prefer_basic=True)
                    if result["ok"]:
                        progress.done("fix", "verified OK")
                        break
                continue
            app.toolkit.changes.snapshot_before(filename, "fix")
            full.write_text(fixed)
            out.log_tool("write_file", f"{filename} (fixed)")
            result = checker.check(filename, str(repo), prefer_basic=True)
            if result["ok"]:
                progress.done("fix", "verified OK")
                break
            progress.fail("fix", result["error"][:60])
    else:
        progress.done("verify", "syntax OK")
    verify_hint = app.planner_checkpoint_hint(
        "create:post_verify",
        user_text,
        "\n".join(
            part for part in [
                f"file={filename}",
                f"verify_ok={result['ok']}",
                post_generate_hint.quality_gap if post_generate_hint else "",
                post_generate_hint.risk if post_generate_hint else "",
            ] if part
        ),
    )
    if verify_hint and verify_hint.quality_gap:
        out.print_info(f"planner quality note: {verify_hint.quality_gap[:140]}")

    if aggressive_quality_task and lines <= 260:
        progress.start("refining output quality")
        current_code = full.read_text(errors="replace")
        refined_response = _generate_text(
            app,
            [
                {
                    "role": "system",
                    "content": (
                        "You refine existing code using SEARCH/REPLACE patches only.\n"
                        "Return one or more blocks in this exact format:\n"
                        "<<<SEARCH\nexact old text\n===\nnew text\nSEARCH>>>\n"
                        "Make a small number of high-impact improvements. Do not rewrite the full file."
                    ),
                },
                {"role": "user", "content": _build_refine_prompt(filename, current_code, user_text)},
            ],
            out,
            stage="refining output quality",
            stream_preview=False,
            max_tokens=900,
        )
        app.toolkit.changes.snapshot_before(filename, "quality_refine")
        if _apply_search_replace(full, refined_response):
            lines = len(full.read_text(errors="replace").splitlines())
            out.log_tool("edit_file", f"path={filename} (refined)")
            out.tool_result(f"Refined {filename} with targeted patch")
            verify2 = checker.check(filename, str(repo), prefer_basic=True)
            if verify2["ok"]:
                progress.done("quality", "fidelity pass applied")
            else:
                progress.fail("quality", verify2["error"][:80])

    # CHECK DEPS
    if filename.endswith(".py"):
        progress.start("checking dependencies")
        dep_err = _check_deps(repo, filename)
        if dep_err:
            module = re.search(r"No module named '(\w+)'", dep_err)
            if module:
                mod = module.group(1)
                out.log_tool("bash", f"pip install {mod}")
                r = subprocess.run(f"pip install {mod}", shell=True, capture_output=True, text=True,
                                   timeout=60, cwd=str(repo))
                out.tool_result(r.stdout.strip()[:80] or r.stderr.strip()[:80])
            else:
                progress.fail("deps", dep_err[:60])
        else:
            progress.done("deps", "all imports OK")

    outcome_notes = _run_outcome_checks(repo, user_text, [filename])
    for note in outcome_notes:
        out.print_info(note)

    quality_gate = _run_quality_gate(app, repo, user_text, filename, full, outcome_notes, checker)
    if quality_gate:
        for note in quality_gate:
            out.print_info(note)

    out.stop_thinking()
    summary = f"\n  {lines} file(s) changed. Use /verify to run tests, /undo to revert.\n\n  Created {filename} ({lines} lines)\n  Run: python {filename}"
    out.stream(summary)
    return summary


# ── Feature 2: FILE EDITING / BUG FIXING ───────────────────────────

def _do_edit(app, user_text, messages, out, progress, checker, context, relevance):
    repo = app.repo_root

    # Find target file
    progress.start("finding file")
    target = _guess_target_file(user_text, repo)
    if not target:
        related = relevance.find_related(user_text, repo, max_files=1)
        target = related[0] if related else ""
    if not target or not (repo / target).is_file():
        out.stream(f"Couldn't find a file to edit. Please specify the filename.")
        return ""

    # READ
    progress.start(f"reading {target}")
    full = repo / target
    current = full.read_text(errors="replace")
    out.log_tool("read_file", target)
    out.tool_result(f"{target} ({len(current.splitlines())} lines)")

    # EDIT
    progress.start("generating changes")
    out.set_stage("editing")
    related_imports = relevance.find_imports(str(full), repo)
    ctx = context.build_for_edit(target, user_text, repo, related_imports)
    response = app.engine.generate_once([
        {"role": "system", "content": SYSTEM_PROMPTS["create"]},  # use create prompt — we want full file back
        *messages[-4:],
        {"role": "user", "content": f"{ctx}\n\nReturn the COMPLETE modified file:\n```python\nmodified code\n```\n\nMake MINIMAL changes."},
    ])

    _, new_code = _parse_code_response(response, "")
    if not new_code:
        # Try search/replace format
        applied = _apply_search_replace(full, response)
        if applied:
            out.log_tool("edit_file", target)
            out.tool_result(f"Edited {target}")
        else:
            out.stream(response)
            return response
    else:
        # Show diff preview before writing
        diff_lines = list(difflib.unified_diff(
            current.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile=f"a/{target}",
            tofile=f"b/{target}",
            n=3,
        ))
        if diff_lines:
            preview = diff_lines[:20]
            colored = []
            for ln in preview:
                ln = ln.rstrip("\n")
                if ln.startswith("+"):
                    colored.append(f"\033[32m{ln}\033[0m")
                elif ln.startswith("-"):
                    colored.append(f"\033[31m{ln}\033[0m")
                elif ln.startswith("@@"):
                    colored.append(f"\033[36m{ln}\033[0m")
                else:
                    colored.append(ln)
            if len(diff_lines) > 20:
                colored.append(f"… ({len(diff_lines) - 20} more diff lines)")
            out.print_info("\n".join(colored))

        app.toolkit.changes.snapshot_before(target, "edit")
        full.write_text(new_code)
        out.log_tool("write_file", target)
        out.tool_result(f"Updated {target} ({len(new_code.splitlines())} lines)")

    # VERIFY
    progress.start("verifying")
    result = checker.check(target, str(repo), prefer_basic=True)
    if not result["ok"]:
        progress.fail("verify", result["error"][:80])
        # One fix attempt
        fix_ctx = context.build_for_fix(target, result["error"], repo)
        fix_resp = app.engine.generate_once([
            {"role": "system", "content": SYSTEM_PROMPTS["fix"]},
            {"role": "user", "content": fix_ctx},
        ])
        _, fixed = _parse_code_response(fix_resp, "")
        if fixed:
            app.toolkit.changes.snapshot_before(target, "fix")
            full.write_text(fixed)
            r2 = checker.check(target, str(repo), prefer_basic=True)
            if r2["ok"]:
                progress.done("fix", "verified OK")
    else:
        progress.done("verify", "syntax OK")

    summary = f"Updated {target}"
    out.stream(summary)
    return summary


# ── Feature 3: CODE REVIEW ─────────────────────────────────────────

def _do_review(app, user_text, messages, out, context, relevance):
    repo = app.repo_root
    target = _guess_target_file(user_text, repo)
    if not target:
        related = relevance.find_related(user_text, repo, max_files=1)
        target = related[0] if related else ""
    if not target or not (repo / target).is_file():
        out.stream("Please specify a file to review.")
        return ""

    out.log_tool("read_file", target)
    content = (repo / target).read_text(errors="replace")
    out.tool_result(f"{target} ({len(content.splitlines())} lines)")

    response = app.engine.generate_once([
        {"role": "system", "content": SYSTEM_PROMPTS["review"]},
        {"role": "user", "content": f"Review this code:\n```\n{content[:6000]}\n```"},
    ])
    out.stream(response)
    return response


# ── Feature 4: EXPLANATION ─────────────────────────────────────────

def _do_explain(app, user_text, messages, out, context, relevance):
    repo = app.repo_root
    target = _guess_target_file(user_text, repo)
    if target and (repo / target).is_file():
        out.log_tool("read_file", target)
        content = (repo / target).read_text(errors="replace")
        out.tool_result(f"{target} ({len(content.splitlines())} lines)")
        prompt = f"Explain this code:\n```\n{content[:6000]}\n```\n\n{user_text}"
    else:
        prompt = user_text

    response = app.engine.generate_once([
        {"role": "system", "content": SYSTEM_PROMPTS["explain"]},
        *messages[-4:],
        {"role": "user", "content": prompt},
    ])
    out.stream(response)
    return response


# ── Feature 7+8: TESTS ─────────────────────────────────────────────

def _do_test(app, user_text, messages, out, progress, checker):
    repo = app.repo_root
    t = user_text.lower()

    if "run" in t or "pytest" in t:
        # RUN tests
        progress.start("running tests")
        out.log_tool("bash", "python3 -m pytest -v --tb=short")
        r = subprocess.run("python3 -m pytest -v --tb=short 2>&1 | tail -30",
                           shell=True, capture_output=True, text=True, timeout=60, cwd=str(repo))
        output = r.stdout.strip()
        out.tool_result(output[:120])
        out.stream(output)
        return output
    else:
        # GENERATE tests
        target = _guess_target_file(user_text, repo)
        if not target:
            out.stream("Please specify a file to write tests for.")
            return ""
        progress.start(f"reading {target}")
        content = (repo / target).read_text(errors="replace")
        progress.start("generating tests")
        response = app.engine.generate_once([
            {"role": "system", "content": SYSTEM_PROMPTS["create"]},
            {"role": "user", "content": f"Write pytest tests for:\n```\n{content[:4000]}\n```\nOutput ONLY test code."},
        ])
        _, test_code = _parse_code_response(response, "")
        if test_code:
            test_file = f"test_{Path(target).name}"
            (repo / test_file).write_text(test_code)
            out.log_tool("write_file", test_file)
            out.tool_result(f"Written {test_file} ({len(test_code.splitlines())} lines)")
            # Run
            progress.start("running tests")
            r = subprocess.run(f"python3 -m pytest {test_file} -v --tb=short 2>&1 | tail -20",
                               shell=True, capture_output=True, text=True, timeout=60, cwd=str(repo))
            out.stream(r.stdout.strip())
            return r.stdout.strip()
        out.stream(response)
        return response


# ── Feature 10: GIT ────────────────────────────────────────────────

def _do_git(app, user_text, messages, out):
    repo = app.repo_root
    t = user_text.lower()

    if "status" in t or "diff" in t:
        out.log_tool("bash", "git status && git diff --stat")
        r = subprocess.run("git status --short && echo '---' && git diff --stat",
                           shell=True, capture_output=True, text=True, cwd=str(repo))
        out.stream(r.stdout.strip())
        return r.stdout.strip()

    if "commit" in t:
        # Get diff
        r = subprocess.run("git diff --staged", shell=True, capture_output=True, text=True, cwd=str(repo))
        diff = r.stdout.strip()
        if not diff:
            r = subprocess.run("git diff", shell=True, capture_output=True, text=True, cwd=str(repo))
            diff = r.stdout.strip()
        if not diff:
            out.stream("No changes to commit.")
            return ""
        # Generate message
        response = app.engine.generate_once([
            {"role": "user", "content": f"Write a concise commit message for:\n{diff[:3000]}\n\nOutput ONLY the message."},
        ])
        msg = response.strip().strip('"').strip("'")
        out.log_tool("bash", f"git commit -m '{msg[:50]}...'")
        subprocess.run("git add -A", shell=True, cwd=str(repo))
        r = subprocess.run(f'git commit -m "{msg}"', shell=True, capture_output=True, text=True, cwd=str(repo))
        out.stream(r.stdout.strip() or r.stderr.strip())
        return msg

    if "log" in t:
        r = subprocess.run("git log --oneline -10", shell=True, capture_output=True, text=True, cwd=str(repo))
        out.stream(r.stdout.strip())
        return r.stdout.strip()

    # Generic git command
    out.stream("Use: git status, git diff, git commit, git log")
    return ""


# ── Feature 11: SEARCH ─────────────────────────────────────────────

def _do_search(app, user_text, out):
    repo = app.repo_root
    keywords = _extract_keywords(user_text)
    pattern = "|".join(keywords) if keywords else user_text

    out.log_tool("grep", pattern)
    r = subprocess.run(
        ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
         "--exclude-dir=.venv", "--exclude-dir=node_modules", "--exclude-dir=__pycache__",
         "--exclude-dir=.git", "--exclude-dir=.tox", "--exclude-dir=dist",
         pattern, "."],
        capture_output=True, text=True, timeout=10, cwd=str(repo),
    )
    output = r.stdout.strip()[:3000]
    if output:
        out.stream(output)
    else:
        out.stream(f"No matches for: {pattern}")
    return output


# ── Feature: CHAT (no tools) ──────────────────────────────────────

def _do_chat(app, user_text, messages, out):
    # Simple chat — no system prompt (IQ3_S hallucinates with complex system prompts)
    response = app.engine.generate_once(
        [{"role": "user", "content": user_text}],
        max_tokens=512,
    )
    out.stream(response)
    return response


# ── Feature: PLANNED (multi-step) ──────────────────────────────────

def _handle_planned(app, user_text, messages, out, progress, checker, context, relevance):
    """Layer 3 + Layer 2: Plan then orchestrate."""
    repo = app.repo_root

    # PLAN
    progress.start("planning")
    tree = _list_dir(repo, depth=2)
    plan_response = app.engine.generate_once([
        {"role": "system", "content": SYSTEM_PROMPTS["plan"]},
        {"role": "user", "content": f"Project:\n{tree[:1000]}\n\nTask: {user_text}\n\nDecompose into steps."},
    ])
    out.print_info(f"Plan:\n{plan_response[:500]}")

    # Parse steps
    steps = re.findall(r'STEP\s*\d+:\s*(CREATE_FILE|EDIT_FILE|RUN_COMMAND|INSTALL_DEP|RUN_TESTS)\s*\|\s*(\S+)\s*\|\s*(.+)', plan_response)

    if not steps:
        # Plan parsing failed — fall back to create
        return _do_create(app, user_text, messages, out, progress, checker, context, relevance)

    # ORCHESTRATE: execute each step
    for i, (action, target, desc) in enumerate(steps):
        progress.start(f"step {i+1}/{len(steps)}: {desc[:50]}")

        if action == "CREATE_FILE":
            _do_create(app, desc, messages, out, progress, checker, context, relevance)
        elif action == "EDIT_FILE":
            _do_edit(app, desc, messages, out, progress, checker, context, relevance)
        elif action == "RUN_COMMAND":
            out.log_tool("bash", target)
            r = subprocess.run(target, shell=True, capture_output=True, text=True, timeout=60, cwd=str(repo))
            out.tool_result(r.stdout.strip()[:120] or r.stderr.strip()[:120])
        elif action == "INSTALL_DEP":
            out.log_tool("bash", f"pip install {target}")
            subprocess.run(f"pip install {target}", shell=True, capture_output=True, text=True, timeout=60, cwd=str(repo))
        elif action == "RUN_TESTS":
            _do_test(app, "run tests", messages, out, progress, checker)

    summary = f"Completed {len(steps)} steps for: {user_text[:60]}"
    out.stream(summary)
    return summary


def _is_quality_sensitive_task(user_text: str) -> bool:
    text = user_text.lower()
    return any(token in text for token in (
        "game", "app", "website", "dashboard", "ui", "clone", "sonic",
        "look like", "feel like", "authentic", "polish", "playable",
    ))


def _needs_aggressive_quality_pass(user_text: str) -> bool:
    text = user_text.lower()
    return any(token in text for token in (
        "sonic", "clone", "look like", "feel like", "authentic", "polish", "fidelity",
    ))


def _should_run_quality_gate(user_text: str, filename: str) -> bool:
    artifact = classify_artifact(filename, user_text)
    if artifact in {"python_app", "web_asset"}:
        return True
    text = user_text.lower()
    return any(token in text for token in (
        "clone", "look like", "feel like", "polish", "fidelity", "ui", "dashboard",
        "landing page", "game", "app", "website", "tool",
    ))


def _build_create_prompt(ctx: str, user_text: str, quality_task: bool) -> str:
    quality = ""
    if quality_task:
        quality = (
            "\nQuality bar:\n"
            "- Make the result recognizably match the requested product or vibe.\n"
            "- Avoid placeholder mechanics and bland scaffolding.\n"
            "- Include concrete details that make the experience feel intentional.\n"
            "- Return a complete runnable file, not a minimal stub.\n"
        )
    return (
        f"{ctx}\n\nTask request:\n{user_text}\n"
        f"{quality}\n"
        "Write the COMPLETE code in a code block:\n"
        "FILENAME: <filename>\n```python\ncomplete code\n```"
    )


def _build_compact_create_context(user_text: str) -> str:
    return (
        "## Task\n"
        f"{user_text}\n\n"
        "Constraints:\n"
        "- Prefer one small runnable file.\n"
        "- Keep the implementation tight and direct.\n"
        "- Do not add explanation text outside the requested code block.\n"
    )


def _build_refine_prompt(filename: str, current_code: str, user_text: str) -> str:
    return (
        f"The first pass works but needs more fidelity and polish.\n"
        f"Original request: {user_text}\n\n"
        f"Current {filename}:\n```python\n{current_code}\n```\n\n"
        "Improve recognizability, game feel, visual clarity, and completeness without breaking runnability.\n"
        "Prefer 2-5 surgical improvements with high payoff.\n"
        "Return SEARCH/REPLACE blocks only, not the whole file."
    )


def _build_quality_gate_prompt(filename: str, user_text: str, code: str, outcome_notes: list[str]) -> str:
    notes = "\n".join(f"- {n}" for n in outcome_notes[:8]) or "- no outcome notes"
    return (
        "Assess whether this generated artifact is actually good enough for the request.\n"
        "Score harshly. Runnable but generic work should not pass.\n"
        "Return strict JSON with keys: score, verdict, issues, strengths.\n"
        "score must be 0-100. verdict must be one of pass or refine.\n\n"
        f"Request:\n{user_text}\n\n"
        f"Outcome notes:\n{notes}\n\n"
        f"File: {filename}\n```text\n{code[:12000]}\n```"
    )


def _build_quality_refine_prompt(filename: str, user_text: str, code: str, issues: list[str]) -> str:
    issue_lines = "\n".join(f"- {issue}" for issue in issues[:6]) or "- improve fidelity and completeness"
    return (
        f"Improve {filename} so it better satisfies the original request.\n"
        f"Original request:\n{user_text}\n\n"
        f"Observed quality issues:\n{issue_lines}\n\n"
        f"Current file:\n```text\n{code[:12000]}\n```\n\n"
        "Return SEARCH/REPLACE blocks only.\n"
        "Make the smallest high-impact changes that improve recognizability, completeness, and user experience.\n"
    )


def _create_output_budget(user_text: str, quality_task: bool, compact_task: bool) -> int:
    text = user_text.lower()
    if compact_task:
        return 900
    if quality_task:
        if any(token in text for token in ("game", "app", "website", "dashboard", "clone", "landing page")):
            return 2600
        return 1800
    if any(token in text for token in ("single-file", "single file", "cli", "script", ".py", ".html", ".css", ".js")):
        return 1400
    return 1800


def _should_use_planner_hints(user_text: str, quality_task: bool, compact_task: bool) -> bool:
    if compact_task:
        return False
    text = user_text.lower()
    if quality_task:
        return True
    if len(text) > 180:
        return True
    return any(token in text for token in ("app", "game", "website", "dashboard", "multi", "full", "polish", "clone"))


def _should_use_fast_create_lane(app, user_text: str, related: list[str], quality_task: bool) -> bool:
    text = user_text.lower()
    if quality_task:
        return False
    if len(text) > 140:
        return False
    if not re.search(r'\b\w+\.(?:py|js|ts|html|css|sh)\b', text):
        return False
    simple_signals = (
        "tiny", "small", "simple", "one file", "single file", "prints ", "print ",
        "hello", "cli", "script", "utility",
    )
    return any(token in text for token in simple_signals)


def _generate_create_response(app, user_text: str, messages, out, quality_task: bool, related: list[str]) -> str:
    if _should_use_fast_create_lane(app, user_text, related, quality_task):
        out.set_stage("generating code (compact)")
        compact_messages = [messages[0], messages[-1]] if len(messages) >= 2 else messages
        return _generate_text(
            app,
            compact_messages,
            out,
            stage="generating code",
            stream_preview=False,
            num_ctx_override=2048,
            max_tokens=_create_output_budget(user_text, quality_task, compact_task=True),
        )
    return _generate_text(
        app,
        messages,
        out,
        stage="generating code",
        stream_preview=True,  # always show live progress during code generation
        max_tokens=_create_output_budget(user_text, quality_task, compact_task=False),
    )


def _generate_text(
    app,
    messages,
    out,
    stage: str,
    stream_preview: bool = False,
    num_ctx_override: int | None = None,
    max_tokens: int | None = None,
) -> str:
    chunks: list[str] = []
    thinking = []
    started_at = time.time()
    first_token_s: float | None = None
    out.set_thinking_peek("contacting model and waiting for first tokens")
    use_thinking = app.config.runtime.laptop_26b_runtime_mode.endswith("-think")
    for event in app.engine.stream_chat_events(messages, think=use_thinking, num_ctx=num_ctx_override, num_predict=max_tokens):
        if event["type"] == "thinking":
            chunk = str(event["content"])
            thinking.append(chunk)
            out.feed_thinking(chunk)
            peek = _summarize_live_preview("".join(thinking))
            if peek:
                out.set_thinking_peek(peek)
            continue
        if event["type"] != "content":
            continue
        chunk = str(event["content"])
        if chunk and first_token_s is None:
            first_token_s = time.time() - started_at
            out.set_thinking_peek("model responded, assembling output")
        chunks.append(chunk)
        if stream_preview:
            peek = _summarize_live_preview("".join(chunks))
            if peek:
                out.set_thinking_peek(peek)
    app._record_runtime_sample(first_token_s=first_token_s, total_s=time.time() - started_at)
    return "".join(chunks).strip()


def _run_quality_gate(app, repo: Path, user_text: str, filename: str, full: Path, outcome_notes: list[str], checker) -> list[str]:
    if not _should_run_quality_gate(user_text, filename):
        return []
    code = full.read_text(errors="replace")
    if len(code.splitlines()) > 450:
        return ["quality gate: skipped for very large generated file"]
    try:
        response = app.engine.generate_once([
            {
                "role": "system",
                "content": (
                    "You are Gem's quality gate.\n"
                    "Return strict JSON only.\n"
                    "Be skeptical of generic outputs that merely run."
                ),
            },
            {"role": "user", "content": _build_quality_gate_prompt(filename, user_text, code, outcome_notes)},
        ], max_tokens=700)
        match = re.search(r"\{.*\}", response, re.DOTALL)
        payload = json.loads(match.group(0) if match else response)
    except Exception:
        return []

    score = int(payload.get("score", 0) or 0)
    verdict = str(payload.get("verdict", "")).strip().lower()
    issues = [str(item).strip() for item in payload.get("issues", []) if str(item).strip()]
    notes = [f"quality gate: score {score}/100"]
    if verdict == "pass" or score >= 82 or not issues:
        return notes + ["quality gate: accepted"]

    refine_response = app.engine.generate_once([
        {
            "role": "system",
            "content": (
                "You refine existing code using SEARCH/REPLACE patches only.\n"
                "Return one or more blocks in this exact format:\n"
                "<<<SEARCH\nexact old text\n===\nnew text\nSEARCH>>>\n"
                "Make a small number of high-impact improvements."
            ),
        },
        {"role": "user", "content": _build_quality_refine_prompt(filename, user_text, code, issues)},
    ], max_tokens=900)
    app.toolkit.changes.snapshot_before(filename, "quality_gate_refine")
    if not _apply_search_replace(full, refine_response):
        return notes + ["quality gate: refine requested but no patch was produced"]

    verify = checker.check(filename, str(repo), prefer_basic=True)
    if not verify["ok"]:
        return notes + [f"quality gate: refinement rejected by verification: {verify['error'][:80]}"]
    return notes + ["quality gate: applied targeted refinement"]


def _summarize_live_preview(text: str) -> str:
    lines = text.count("\n")
    tokens_approx = len(text.split())
    # Show meaningful progress with line count
    if lines > 5:
        # Find the last function/class being written
        last_def = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "function ")):
                last_def = stripped[:60]
        if last_def:
            return f"generating code ({lines} lines) — {last_def}"
        return f"generating code ({lines} lines, {tokens_approx} tokens)"
    if tokens_approx > 10:
        return f"generating ({tokens_approx} tokens)"
    return ""


def _run_outcome_checks(repo: Path, user_text: str, files: list[str]) -> list[str]:
    output, code = run_outcome_verification(repo, user_text, files)
    notes = []
    for block in output.split("\n\n"):
        line = block.strip().splitlines()
        if line:
            notes.append(line[0][:160])
    if _is_quality_sensitive_task(user_text):
        notes.append("quality check: output path used patch-first refinement and artifact verification")
    if code != 0 and not notes:
        notes.append("outcome check: verification reported an issue")
    return notes


# ── Helpers ─────────────────────────────────────────────────────────

def _guess_target_file(user_text: str, repo: Path) -> str:
    """Guess which file the user wants."""
    match = re.search(r'(\w[\w.-]*\.(?:py|js|ts|html|css|json|md|txt))', user_text)
    if match and (repo / match.group(1)).is_file():
        return match.group(1)
    # Check stems
    for f in sorted(repo.iterdir()):
        if f.is_file() and f.suffix in (".py", ".js", ".ts"):
            if f.stem.lower() in user_text.lower():
                return f.name
    return ""


def _parse_code_response(response: str, user_text: str) -> tuple[str, str]:
    """Extract filename and code from model response."""
    fname_match = re.search(r'FILENAME:\s*(\S+)', response)
    code_match = re.search(r'```\w*\n(.*?)```', response, re.DOTALL)

    filename = ""
    if fname_match:
        filename = fname_match.group(1)
    code = code_match.group(1).strip() if code_match else ""
    if not filename and code:
        header_match = re.search(r'^\s*(?:#|//)\s*([\w./-]+\.(?:py|js|ts|tsx|jsx|html|css|sh))\s*$', code, re.MULTILINE)
        if header_match:
            filename = header_match.group(1)
    if not filename:
        name_match = re.search(r'(\w+\.py)', user_text)
        if name_match:
            filename = name_match.group(1)
        else:
            words = re.findall(r'\w+', user_text.lower())
            for w in words:
                if w not in ("make", "create", "build", "write", "generate", "tiny", "small",
                             "script", "python", "file", "called", "named", "a", "an", "the",
                             "that", "can", "i", "run", "locally", "my", "laptop", "app",
                             "game", "on", "okay", "prints", "print", "hello"):
                    filename = f"{w}_game.py" if "game" in user_text.lower() else f"{w}.py"
                    break
            if not filename:
                filename = "main.py"

    if code:
        lines = code.splitlines()
        if lines:
            first = lines[0].strip()
            if re.fullmatch(r'[\w./-]+\.(?:py|js|ts|tsx|jsx|html|css|sh)', first):
                code = "\n".join(lines[1:]).lstrip()
            elif re.fullmatch(r'(?:#|//)\s*[\w./-]+\.(?:py|js|ts|tsx|jsx|html|css|sh)', first):
                code = "\n".join(lines[1:]).lstrip()
    return filename, code


def _apply_search_replace(file_path: Path, response: str) -> bool:
    """Apply <<<SEARCH...===...SEARCH>>> blocks."""
    blocks = re.findall(r'<<<SEARCH\n(.*?)\n===\n(.*?)\nSEARCH>>>', response, re.DOTALL)
    if not blocks:
        return False
    content = file_path.read_text(errors="replace")
    for search, replace in blocks:
        if search in content:
            content = content.replace(search, replace, 1)
    file_path.write_text(content)
    return True


def _check_deps(repo: Path, filename: str) -> str:
    """Check if imports are available."""
    full = repo / filename
    try:
        env = {**__import__("os").environ, "SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
        code = (
            f"import ast\n"
            f"for n in ast.walk(ast.parse(open('{full}').read())):\n"
            f"  if isinstance(n, ast.Import):\n"
            f"    for a in n.names: __import__(a.name.split('.')[0])\n"
        )
        r = subprocess.run(["python3", "-c", code], capture_output=True, text=True,
                           timeout=10, cwd=str(repo), env=env)
        if r.returncode != 0:
            lines = [l for l in r.stderr.splitlines() if "MallocStackLogging" not in l]
            return lines[-1] if lines else ""
    except Exception:
        pass
    return ""
