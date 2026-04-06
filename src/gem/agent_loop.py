"""State machine agent loop — harness controls sequencing, model generates content.

Based on observing OpenAI Codex, Claude Code, and Claude Opus:
- Model generates content (code, fixes, answers)
- Harness handles sequencing, validation, and tool execution
- Predictable state machine, not autonomous tool selection

States: GATHER → WRITE → VERIFY → FIX → DONE

This is more reliable with local quantized models than giving
them full autonomy over tool ordering.
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


def run_agent_loop(
    app: "GemApp",
    user_text: str,
    composed_messages: list[dict],
    out: "OutputManager",
) -> str:
    """State machine: GATHER → WRITE → VERIFY → FIX → DONE.

    The harness controls the sequence. The model only generates content.
    """
    repo = app.repo_root

    # ── GATHER: understand what exists ──
    out.print_info("▶ gathering context")
    existing_files = _list_files(repo)
    relevant_content = _read_relevant_files(repo, user_text, existing_files)

    # ── DECIDE: is this a new file or an edit? ──
    target_file = _guess_target_file(user_text, existing_files)
    is_edit = target_file and (repo / target_file).is_file()

    if is_edit:
        return _handle_edit(app, user_text, target_file, composed_messages, out)
    else:
        return _handle_create(app, user_text, existing_files, relevant_content, composed_messages, out)


def _handle_create(
    app: "GemApp",
    user_text: str,
    existing_files: list[str],
    relevant_content: str,
    composed_messages: list[dict],
    out: "OutputManager",
) -> str:
    """Create a new file: ask model for code → write → verify → fix if needed."""
    repo = app.repo_root

    # ── WRITE: ask model to generate complete code ──
    out.print_info("▶ generating code")
    out.set_stage("generating code")

    file_list = ", ".join(existing_files[:20]) if existing_files else "(empty directory)"
    prompt = (
        f"Existing files: {file_list}\n"
        f"{relevant_content}\n\n"
        f"Task: {user_text}\n\n"
        f"Write the COMPLETE code. Output it in a code block:\n"
        f"FILENAME: <the filename>\n"
        f"```python\n"
        f"complete code here\n"
        f"```\n\n"
        f"Write ALL the code needed — not a scaffold. One file, complete and working."
    )

    response = app.engine.generate_once([
        *composed_messages,
        {"role": "user", "content": prompt},
    ])

    # Parse filename and code from response
    filename, code = _parse_code_response(response, user_text)
    if not code:
        out.stream(response)
        return response

    # Write the file
    out.log_tool("write_file", filename)
    full_path = repo / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    app.toolkit.changes.snapshot_before(filename, "agent")
    full_path.write_text(code)
    lines = len(code.splitlines())
    out.tool_result(f"Written {filename} ({lines} lines)")

    # ── VERIFY: syntax check ──
    out.print_info("▶ verifying")
    out.set_stage("verifying")
    error = _verify_python(repo, filename)

    if error:
        out.tool_result(f"⚠ {error}", error=True)
        # ── FIX: ask model to fix the error (max 2 attempts) ──
        for fix_attempt in range(2):
            out.print_info(f"▶ fixing (attempt {fix_attempt + 1})")
            out.set_stage("fixing")
            current_code = full_path.read_text(errors="replace")
            fix_prompt = (
                f"This code has an error:\n"
                f"```python\n{current_code}\n```\n\n"
                f"Error: {error}\n\n"
                f"Fix the error and return the COMPLETE corrected file in a code block:\n"
                f"```python\nfixed code\n```"
            )
            fix_response = app.engine.generate_once([
                {"role": "user", "content": fix_prompt},
            ])
            _, fixed_code = _parse_code_response(fix_response, "")
            if fixed_code:
                app.toolkit.changes.snapshot_before(filename, "agent_fix")
                full_path.write_text(fixed_code)
                out.log_tool("write_file", f"{filename} (fixed)")
                out.tool_result(f"Written {filename} ({len(fixed_code.splitlines())} lines)")

                error = _verify_python(repo, filename)
                if not error:
                    out.print_info("▶ fix successful")
                    break
                out.tool_result(f"⚠ {error}", error=True)
            else:
                break

    # ── CHECK DEPS: verify imports work ──
    if filename.endswith(".py"):
        out.print_info("▶ checking dependencies")
        out.set_stage("checking deps")
        dep_error = _check_deps(repo, filename)
        if dep_error:
            out.tool_result(f"⚠ {dep_error}", error=True)
            # Try to install the missing module
            module = _extract_module_name(dep_error)
            if module:
                out.log_tool("bash", f"pip install {module}")
                result = subprocess.run(
                    f"pip install {module}", shell=True,
                    capture_output=True, text=True, timeout=60,
                    cwd=str(repo),
                )
                output = (result.stdout + result.stderr).strip()
                out.tool_result(output[:120])

    # ── DONE ──
    out.print_info("▶ done")
    summary = f"Created {filename} ({lines} lines). Run: python {filename}"
    out.stream(summary)
    return summary


def _handle_edit(
    app: "GemApp",
    user_text: str,
    target_file: str,
    composed_messages: list[dict],
    out: "OutputManager",
) -> str:
    """Edit an existing file: read → ask model for changes → write → verify."""
    repo = app.repo_root

    # ── READ: get current file contents ──
    out.print_info("▶ reading file")
    out.set_stage("reading")
    full_path = repo / target_file
    current_code = full_path.read_text(errors="replace")
    out.log_tool("read_file", target_file)
    out.tool_result(f"{target_file} ({len(current_code.splitlines())} lines)")

    # ── EDIT: ask model for modified version ──
    out.print_info("▶ generating changes")
    out.set_stage("editing")
    prompt = (
        f"Here is {target_file}:\n"
        f"```python\n{current_code}\n```\n\n"
        f"Task: {user_text}\n\n"
        f"Return the COMPLETE modified file with your changes applied.\n"
        f"```python\nmodified code\n```\n\n"
        f"Make MINIMAL changes — only modify what's needed for the task."
    )

    response = app.engine.generate_once([
        *composed_messages,
        {"role": "user", "content": prompt},
    ])

    _, new_code = _parse_code_response(response, "")
    if not new_code:
        out.stream(response)
        return response

    # Write the modified file
    app.toolkit.changes.snapshot_before(target_file, "agent_edit")
    full_path.write_text(new_code)
    out.log_tool("write_file", target_file)
    out.tool_result(f"Updated {target_file} ({len(new_code.splitlines())} lines)")

    # ── VERIFY ──
    out.print_info("▶ verifying")
    out.set_stage("verifying")
    error = _verify_python(repo, target_file)
    if error:
        out.tool_result(f"⚠ {error}", error=True)
        # One fix attempt
        out.print_info("▶ fixing")
        fix_prompt = (
            f"This code has an error:\n```python\n{new_code}\n```\n\n"
            f"Error: {error}\n\nReturn the COMPLETE fixed file:\n```python\nfixed\n```"
        )
        fix_response = app.engine.generate_once([{"role": "user", "content": fix_prompt}])
        _, fixed = _parse_code_response(fix_response, "")
        if fixed:
            app.toolkit.changes.snapshot_before(target_file, "agent_fix")
            full_path.write_text(fixed)
            out.log_tool("write_file", f"{target_file} (fixed)")
            out.tool_result(f"Updated {target_file} ({len(fixed.splitlines())} lines)")
    else:
        out.print_info("▶ verified OK")

    # ── DONE ──
    summary = f"Updated {target_file}. Run: python {target_file}"
    out.stream(summary)
    return summary


# ── Helper functions ──────────────────────────────────────────────

def _list_files(repo: Path) -> list[str]:
    """List files in repo (fast, no hidden/cache dirs)."""
    files = []
    for p in repo.rglob("*"):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts:
            try:
                files.append(str(p.relative_to(repo)))
            except ValueError:
                pass
    return sorted(files)[:50]


def _read_relevant_files(repo: Path, user_text: str, files: list[str]) -> str:
    """Read files that seem relevant to the task."""
    text_lower = user_text.lower()
    relevant = []
    for f in files:
        fname = f.lower()
        # Check if any word from user text appears in filename
        if any(w in fname for w in text_lower.split() if len(w) > 3):
            try:
                content = (repo / f).read_text(errors="replace")[:3000]
                relevant.append(f"--- {f} ---\n{content}")
            except Exception:
                pass
    if relevant:
        return "Relevant existing files:\n" + "\n".join(relevant[:3])
    return ""


def _guess_target_file(user_text: str, existing_files: list[str]) -> str:
    """Guess which file the user wants to edit/create."""
    # Check for explicit filename in user text
    match = re.search(r'(\w[\w.-]*\.(?:py|js|ts|html|css|json|md|txt))', user_text)
    if match:
        return match.group(1)

    # Check if user is referring to an existing file by content
    text_lower = user_text.lower()
    for f in existing_files:
        fname = Path(f).stem.lower()
        if fname in text_lower and len(fname) > 2:
            return f

    return ""


def _parse_code_response(response: str, user_text: str) -> tuple[str, str]:
    """Extract filename and code from model response."""
    # Look for FILENAME: header
    fname_match = re.search(r'FILENAME:\s*(\S+)', response)

    # Look for code block
    code_match = re.search(r'```\w*\n(.*?)```', response, re.DOTALL)

    filename = ""
    if fname_match:
        filename = fname_match.group(1)
    elif not filename:
        # Guess from user text
        name_match = re.search(r'(\w+\.py)', user_text)
        if name_match:
            filename = name_match.group(1)
        else:
            # Generate from task description
            words = re.findall(r'\w+', user_text.lower())
            for w in words:
                if w not in ("make", "create", "build", "a", "an", "the", "that", "can",
                            "i", "run", "locally", "my", "laptop", "app", "game", "on"):
                    filename = f"{w}_game.py" if "game" in user_text.lower() else f"{w}.py"
                    break
            if not filename:
                filename = "main.py"

    code = code_match.group(1).strip() if code_match else ""
    return filename, code


def _verify_python(repo: Path, filename: str) -> str:
    """Syntax check a Python file. Returns error string or empty."""
    full = repo / filename
    if not full.is_file() or not filename.endswith(".py"):
        return ""
    try:
        code = full.read_text(errors="replace")
        compile(code, filename, "exec")
    except SyntaxError as e:
        return f"SyntaxError in {filename} line {e.lineno}: {e.msg}"
    return ""


def _check_deps(repo: Path, filename: str) -> str:
    """Check if imports in a Python file are available."""
    full = repo / filename
    try:
        env = {**__import__("os").environ, "SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
        result = subprocess.run(
            ["python3", "-c", f"import ast\nfor n in ast.walk(ast.parse(open('{full}').read())):\n"
             f"  if isinstance(n, ast.Import):\n"
             f"    for a in n.names:\n"
             f"      __import__(a.name.split('.')[0])"],
            capture_output=True, text=True, timeout=10, cwd=str(repo), env=env,
        )
        if result.returncode != 0:
            stderr = [l for l in result.stderr.splitlines() if "MallocStackLogging" not in l]
            return stderr[-1] if stderr else ""
    except Exception:
        pass
    return ""


def _extract_module_name(error: str) -> str:
    """Extract module name from an ImportError message."""
    match = re.search(r"No module named '(\w+)'", error)
    return match.group(1) if match else ""
