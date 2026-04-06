"""3-layer agent loop — harness controls sequencing, model generates content.

Layer 1: EXECUTORS — 13 deterministic state machines (this file)
Layer 2: ORCHESTRATOR — walks multi-step plans
Layer 3: PLANNER — decomposes complex tasks

Architecture based on observing OpenAI Codex, Claude Code, and Claude Opus.
Model generates content. Harness handles sequencing, validation, retries.
"""
from __future__ import annotations

import json
import re
import subprocess
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


# ── Intent Classifier (rule-based, LLM fallback) ───────────────────

def classify_intent(text: str) -> str:
    """Route user request to the right feature. Rule-based first, LLM fallback."""
    t = text.lower()
    if any(w in t for w in ("create", "make", "new file", "generate", "scaffold", "build")):
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
    repo = app.repo_root
    progress = ProgressTracker(out.print_info)
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

    # GATHER
    progress.start("gathering context")
    related = relevance.find_related(user_text, repo, max_files=2)
    ctx = context.build_for_create(user_text, repo, related)

    # WRITE
    progress.start("generating code")
    out.set_stage("generating code")
    prompt = f"{ctx}\n\nWrite the COMPLETE code in a code block:\nFILENAME: <filename>\n```python\ncomplete code\n```"
    response = app.engine.generate_once([
        {"role": "system", "content": SYSTEM_PROMPTS["create"]},
        *messages[-4:],
        {"role": "user", "content": prompt},
    ])

    filename, code = _parse_code_response(response, user_text)
    if not code:
        out.stream(response)
        return response

    # APPLY
    full = repo / filename
    full.parent.mkdir(parents=True, exist_ok=True)
    app.toolkit.changes.snapshot_before(filename, "agent")
    full.write_text(code)
    lines = len(code.splitlines())
    out.log_tool("write_file", filename)
    out.tool_result(f"Written {filename} ({lines} lines)")
    progress.done("write", f"{filename} ({lines} lines)")

    # VERIFY
    progress.start("verifying")
    result = checker.check(filename, str(repo))
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
                    result = checker.check(filename, str(repo))
                    if result["ok"]:
                        progress.done("fix", "verified OK")
                        break
                continue
            app.toolkit.changes.snapshot_before(filename, "fix")
            full.write_text(fixed)
            out.log_tool("write_file", f"{filename} (fixed)")
            result = checker.check(filename, str(repo))
            if result["ok"]:
                progress.done("fix", "verified OK")
                break
            progress.fail("fix", result["error"][:60])
    else:
        progress.done("verify", "syntax OK")

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

    summary = f"Created {filename} ({lines} lines). Run: python {filename}"
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
        app.toolkit.changes.snapshot_before(target, "edit")
        full.write_text(new_code)
        out.log_tool("write_file", target)
        out.tool_result(f"Updated {target} ({len(new_code.splitlines())} lines)")

    # VERIFY
    progress.start("verifying")
    result = checker.check(target, str(repo))
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
            r2 = checker.check(target, str(repo))
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
        ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts", pattern, "."],
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
    response = app.engine.generate_once([*messages[-6:], {"role": "user", "content": user_text}])
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
    if not filename:
        name_match = re.search(r'(\w+\.py)', user_text)
        if name_match:
            filename = name_match.group(1)
        else:
            words = re.findall(r'\w+', user_text.lower())
            for w in words:
                if w not in ("make", "create", "build", "a", "an", "the", "that", "can",
                             "i", "run", "locally", "my", "laptop", "app", "game", "on", "okay"):
                    filename = f"{w}_game.py" if "game" in user_text.lower() else f"{w}.py"
                    break
            if not filename:
                filename = "main.py"

    code = code_match.group(1).strip() if code_match else ""
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
