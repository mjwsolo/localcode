"""Context management — the #1 make-or-break for small local models.

Good context selection makes a 26B perform like a 70B.
6 components, all zero LLM cost:

1. TokenBudget — enforce token limits per call
2. FileSummarizer — AST-based compression (20x reduction)
3. ContextAssembler — pick what goes into each LLM call
4. RelevanceFinder — find related files by keyword + recency
5. ConversationManager — compress multi-turn history
6. SyntaxChecker — language-aware validation
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


# ── 1. Token Budget ─────────────────────────────────────────────────

class TokenBudget:
    """Track and enforce token limits per LLM call."""

    def __init__(self, model_max_context: int = 8192):
        self.max_context = model_max_context
        self.reserved_system = 500
        self.reserved_tools = 400
        self.reserved_output = 2048
        self.available = (self.max_context
                          - self.reserved_system
                          - self.reserved_tools
                          - self.reserved_output)

    def count(self, text: str) -> int:
        """Fast approximate token count. 1 token ≈ 4 chars for code."""
        return len(text) // 4

    def fits(self, text: str) -> bool:
        return self.count(text) <= self.available

    def truncate(self, text: str, budget: int) -> str:
        """Hard truncate to token budget, break at line boundary."""
        max_chars = budget * 4
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_nl = truncated.rfind("\n")
        if last_nl > 0:
            truncated = truncated[:last_nl]
        return truncated + "\n... (truncated)"


# ── 2. File Summarizer ──────────────────────────────────────────────

class FileSummarizer:
    """Extract structural summary from code files. Zero LLM cost.

    A 400-line Flask app becomes ~150 tokens:
      from flask import Flask
      class PostService:
          def create(self, user_id, title, body): ...
          def get(self, post_id): ...
      def create_app(config): ...
    """

    def summarize(self, file_path: str) -> str:
        try:
            content = Path(file_path).read_text(errors="replace")
        except Exception:
            return ""

        ext = Path(file_path).suffix
        if ext == ".py":
            return self._summarize_python(content)
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            return self._summarize_js(content)
        else:
            # Fallback: first 20 + last 10 lines
            lines = content.splitlines()
            if len(lines) <= 40:
                return content
            return "\n".join(lines[:20] + ["  ..."] + lines[-10:])

    def _summarize_python(self, content: str) -> str:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content[:500]

        lines = []

        # Imports
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                names = ", ".join(a.name for a in node.names)
                lines.append(f"import {names}")
            elif isinstance(node, ast.ImportFrom):
                names = ", ".join(a.name for a in node.names)
                lines.append(f"from {node.module} import {names}")

        # Constants (top-level assignments with UPPER_CASE)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        try:
                            val = ast.get_source_segment(content, node)
                            if val and len(val) < 80:
                                lines.append(val)
                        except Exception:
                            pass

        # Class and function signatures
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(
                    getattr(b, "id", getattr(b, "attr", "?"))
                    for b in node.bases
                )
                lines.append(f"\nclass {node.name}({bases}):" if bases else f"\nclass {node.name}:")
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        args = self._format_args(item.args)
                        lines.append(f"    def {item.name}({args}): ...")

            elif isinstance(node, ast.FunctionDef):
                args = self._format_args(node.args)
                doc = ast.get_docstring(node)
                lines.append(f"\ndef {node.name}({args}):")
                if doc:
                    lines.append(f'    """{doc.splitlines()[0]}"""')
                lines.append("    ...")

        return "\n".join(lines) if lines else content[:500]

    def _summarize_js(self, content: str) -> str:
        """Regex-based JS/TS summary."""
        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if (stripped.startswith("import ") or
                stripped.startswith("export ") or
                stripped.startswith("function ") or
                stripped.startswith("class ") or
                stripped.startswith("const ") and "=>" in stripped or
                stripped.startswith("export default")):
                lines.append(stripped)
        return "\n".join(lines) if lines else content[:500]

    @staticmethod
    def _format_args(args) -> str:
        return ", ".join(a.arg for a in args.args)


# ── 3. Context Assembler ────────────────────────────────────────────

class ContextAssembler:
    """Builds the optimal context window for each LLM call.

    Treats context like RAM in an embedded system — every token counts.
    """

    def __init__(self, max_context: int = 8192):
        self.budget = TokenBudget(max_context)
        self.summarizer = FileSummarizer()

    def build_for_create(self, task: str, project_root: Path,
                         reference_files: list[str] | None = None) -> str:
        parts = []
        remaining = self.budget.available

        # Reference files (match style)
        if reference_files:
            for rf in reference_files[:2]:
                if remaining <= 500:
                    break
                summary = self.summarizer.summarize(str(project_root / rf))
                summary = self.budget.truncate(summary, min(remaining // 3, 800))
                parts.append(f"## {rf} (reference)\n```\n{summary}\n```")
                remaining -= self.budget.count(summary)

        # Directory structure
        tree = _list_dir(project_root, depth=2)
        tree = self.budget.truncate(tree, 200)
        parts.append(f"## Project structure\n```\n{tree}\n```")

        parts.append(f"## Task\n{task}")
        return "\n\n".join(parts)

    def build_for_edit(self, target_file: str, task: str,
                       project_root: Path,
                       related_files: list[str] | None = None) -> str:
        parts = []
        remaining = self.budget.available

        # Priority 1: target file (60% of budget)
        full = project_root / target_file
        if full.is_file():
            content = full.read_text(errors="replace")
            tokens = self.budget.count(content)
            if tokens <= remaining * 0.6:
                parts.append(f"## {target_file}\n```\n{content}\n```")
                remaining -= tokens
            else:
                relevant = self._extract_relevant_section(content, task, int(remaining * 0.6))
                parts.append(f"## {target_file} (relevant section)\n```\n{relevant}\n```")
                remaining -= self.budget.count(relevant)

        # Priority 2: related file signatures (30%)
        if related_files and remaining > 300:
            for rf in related_files[:3]:
                if remaining <= 200:
                    break
                summary = self.summarizer.summarize(str(project_root / rf))
                summary = self.budget.truncate(summary, min(remaining // 2, 500))
                parts.append(f"## {rf} (signatures)\n```\n{summary}\n```")
                remaining -= self.budget.count(summary)

        parts.append(f"## Task\n{task}")
        return "\n\n".join(parts)

    def build_for_fix(self, target_file: str, error: str,
                      project_root: Path,
                      previous_diff: str | None = None) -> str:
        parts = []
        remaining = self.budget.available

        # Priority 1: error output
        error_trunc = self.budget.truncate(error, 500)
        parts.append(f"## Error\n```\n{error_trunc}\n```")
        remaining -= self.budget.count(error_trunc)

        # Priority 2: the file
        full = project_root / target_file
        if full.is_file():
            content = full.read_text(errors="replace")
            content = self.budget.truncate(content, int(remaining * 0.7))
            parts.append(f"## {target_file}\n```\n{content}\n```")
            remaining -= self.budget.count(content)

        # Priority 3: previous attempt
        if previous_diff and remaining > 200:
            diff_trunc = self.budget.truncate(previous_diff, remaining)
            parts.append(f"## Previous fix (didn't work)\n```\n{diff_trunc}\n```")

        parts.append("## Task\nFix the error above.")
        return "\n\n".join(parts)

    def _extract_relevant_section(self, content: str, task: str, budget: int) -> str:
        """Extract the section most relevant to the task. No LLM needed."""
        lines = content.splitlines()
        keywords = _extract_keywords(task)

        # Score each line
        scored = []
        for i, line in enumerate(lines):
            score = sum(1 for kw in keywords if kw.lower() in line.lower())
            scored.append((i, score))

        best_line = max(scored, key=lambda x: x[1])[0] if scored else 0
        max_lines = (budget * 4) // 80
        half = max_lines // 2
        start = max(0, best_line - half)
        end = min(len(lines), best_line + half)

        # Expand to function/class boundary
        while start > 0 and not _is_boundary(lines[start]):
            start -= 1
        while end < len(lines) - 1 and not _is_boundary(lines[end]):
            end += 1

        section = lines[start:end]
        result = "\n".join(f"{i + start + 1}\t{l}" for i, l in enumerate(section))
        if start > 0:
            result = f"... (lines 1-{start} omitted)\n" + result
        if end < len(lines):
            result += f"\n... (lines {end + 1}-{len(lines)} omitted)"
        return result


# ── 4. Relevance Finder ─────────────────────────────────────────────

class RelevanceFinder:
    """Find files related to a task. Zero LLM cost."""

    def find_related(self, task: str, project_root: Path,
                     max_files: int = 5) -> list[str]:
        keywords = _extract_keywords(task)
        all_files = _list_code_files(project_root)

        scored = []
        for fpath in all_files:
            score = 0
            fname = Path(fpath).stem.lower()

            # Filename match
            for kw in keywords:
                if kw.lower() in fname:
                    score += 10

            # Content match (first 4KB only)
            full = project_root / fpath
            try:
                content = full.read_text(errors="replace")[:4000]
                for kw in keywords:
                    score += content.lower().count(kw.lower())
            except Exception:
                continue

            # Recency bonus
            try:
                mtime = full.stat().st_mtime
                days_old = (time.time() - mtime) / 86400
                score += max(0, 5 - days_old)
            except Exception:
                pass

            if score > 0:
                scored.append((fpath, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scored[:max_files]]

    def find_imports(self, file_path: str, project_root: Path) -> list[str]:
        """Find local files imported by a given file."""
        try:
            content = Path(file_path).read_text(errors="replace")
            tree = ast.parse(content)
        except Exception:
            return []

        local_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                rel_path = node.module.replace(".", "/") + ".py"
                full = project_root / rel_path
                if full.is_file():
                    local_imports.append(rel_path)
        return local_imports


# ── 5. Conversation Manager ─────────────────────────────────────────

class ConversationManager:
    """Compress multi-turn history without blowing token budget."""

    def __init__(self, max_history_tokens: int = 1000):
        self.max_history = max_history_tokens
        self.turns: list[dict] = []
        self.budget = TokenBudget()

    def add_turn(self, role: str, content: str):
        tokens = self.budget.count(content)
        summary = self._auto_summarize(content) if role == "assistant" and tokens > 200 else content
        self.turns.append({
            "role": role, "content": content,
            "tokens": tokens, "summary": summary,
        })

    def get_history(self) -> list[dict]:
        """Return conversation history that fits in budget."""
        result = []
        remaining = self.max_history

        # Last turn always full
        if self.turns:
            last = self.turns[-1]
            result.append({"role": last["role"], "content": last["content"]})
            remaining -= last["tokens"]

        # Older turns use summaries
        for turn in reversed(self.turns[:-1]):
            summary_tokens = self.budget.count(turn["summary"])
            if summary_tokens <= remaining:
                result.append({"role": turn["role"], "content": turn["summary"]})
                remaining -= summary_tokens
            else:
                break

        result.reverse()
        return result

    @staticmethod
    def _auto_summarize(content: str) -> str:
        """Rule-based summary. No LLM needed."""
        parts = []
        for line in content.splitlines():
            low = line.lower().strip()
            if any(op in low for op in ("created", "wrote", "edited", "deleted", "installed", "error", "failed")):
                parts.append(line.strip())
        if not parts:
            parts = [l.strip() for l in content.splitlines()[:2]]
        return " | ".join(parts[:5])


# ── 6. Syntax Checker ───────────────────────────────────────────────

class SyntaxChecker:
    """Language-aware syntax validation. No LLM needed."""

    CHECKS = {
        ".py": 'python3 -c "import ast; ast.parse(open(\'{f}\').read())"',
        ".js": "node --check '{f}'",
        ".ts": "npx tsc --noEmit '{f}' 2>&1 | head -5",
        ".sh": "bash -n '{f}'",
        ".rb": "ruby -c '{f}'",
        ".go": "go vet '{f}'",
        ".rs": "cargo check 2>&1 | head -10",
    }

    def check(self, file_path: str, cwd: str = ".") -> dict:
        ext = Path(file_path).suffix
        cmd_template = self.CHECKS.get(ext)
        if not cmd_template:
            return {"ok": True, "skip": f"no checker for {ext}"}

        cmd = cmd_template.format(f=file_path)
        try:
            env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=15, cwd=cwd, env=env,
            )
            stderr = "\n".join(
                l for l in result.stderr.splitlines()
                if "MallocStackLogging" not in l
            )
            return {
                "ok": result.returncode == 0,
                "error": stderr.strip() if result.returncode != 0 else None,
            }
        except subprocess.TimeoutExpired:
            return {"ok": True, "skip": "check timed out"}
        except Exception as e:
            return {"ok": True, "skip": str(e)}


# ── Undo Stack ──────────────────────────────────────────────────────

class UndoStack:
    """Track file changes for rollback."""

    def __init__(self):
        self.stack: list[tuple[str, str | None, float]] = []

    def snapshot(self, file_path: str):
        """Call BEFORE every write/edit."""
        full = Path(file_path)
        content = full.read_text(errors="replace") if full.is_file() else None
        self.stack.append((str(file_path), content, time.time()))

    def undo(self) -> str:
        if not self.stack:
            return "Nothing to undo"
        file_path, prev_content, _ = self.stack.pop()
        if prev_content is None:
            Path(file_path).unlink(missing_ok=True)
            return f"Deleted {file_path} (was newly created)"
        else:
            Path(file_path).write_text(prev_content)
            return f"Reverted {file_path}"

    @property
    def can_undo(self) -> bool:
        return bool(self.stack)


# ── Progress Tracker ────────────────────────────────────────────────

class ProgressTracker:
    """Show user what's happening during multi-step tasks."""

    def __init__(self, callback=None):
        self._cb = callback or (lambda x: None)

    def start(self, step: str):
        self._cb(f"▶ {step}")

    def done(self, step: str, result: str = "ok"):
        self._cb(f"  ✓ {step} — {result}")

    def fail(self, step: str, error: str):
        self._cb(f"  ✗ {step} — {error}")


# ── Helpers ─────────────────────────────────────────────────────────

def _list_dir(root: Path, depth: int = 3) -> str:
    """Fast directory listing, skipping hidden/cache dirs."""
    lines = []
    for p in sorted(root.rglob("*")):
        if any(part.startswith(".") or part in ("__pycache__", "node_modules", "venv", ".venv")
               for part in p.relative_to(root).parts):
            continue
        rel = p.relative_to(root)
        if len(rel.parts) > depth:
            continue
        indent = "  " * (len(rel.parts) - 1)
        name = rel.name + ("/" if p.is_dir() else "")
        lines.append(f"{indent}{name}")
    return "\n".join(lines[:100])


def _list_code_files(root: Path) -> list[str]:
    """List code files in project."""
    extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".rb", ".sh", ".java", ".c", ".cpp", ".h"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in extensions:
            if not any(part.startswith(".") or part in ("__pycache__", "node_modules", "venv")
                       for part in p.relative_to(root).parts):
                try:
                    files.append(str(p.relative_to(root)))
                except ValueError:
                    pass
    return sorted(files)[:200]


def _extract_keywords(task: str) -> list[str]:
    """Pull likely-relevant identifiers from a task description."""
    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]+', task)
    stopwords = {"the", "a", "an", "to", "in", "for", "and", "or", "this", "that",
                 "with", "from", "add", "remove", "fix", "change", "make", "update",
                 "create", "write", "build", "run", "locally", "can", "my", "app"}
    return [w for w in words if w.lower() not in stopwords and len(w) > 2]


def _is_boundary(line: str) -> bool:
    """Is this line a class/function definition?"""
    stripped = line.lstrip()
    return (stripped.startswith("def ") or stripped.startswith("class ")
            or stripped.startswith("async def ") or stripped.startswith("function ")
            or stripped.startswith("export "))


# ── System Prompts (TINY) ───────────────────────────────────────────

SYSTEM_PROMPTS = {
    "create": "You generate code files. Output ONLY valid code in a code block. No explanations.",
    "edit": (
        "You edit code. Output SEARCH/REPLACE blocks:\n"
        "<<<SEARCH\nexact lines to find\n===\nreplacement lines\nSEARCH>>>\n"
        "Multiple blocks allowed. Match whitespace exactly."
    ),
    "review": "Review code for bugs and issues. Format: LINE {n}: {severity} — {issue}",
    "explain": "Explain the code concisely. Max 5 sentences. Focus on what and why.",
    "plan": (
        "Decompose the task into steps. Format:\n"
        "STEP {n}: {ACTION} | {target_file} | {description}\n"
        "DEPENDS_ON: {step_numbers or 'none'}\n"
        "Actions: CREATE_FILE, EDIT_FILE, RUN_COMMAND, INSTALL_DEP, RUN_TESTS"
    ),
    "fix": "Fix the bug. Output ONLY search/replace blocks:\n<<<SEARCH\nexact lines\n===\nfixed lines\nSEARCH>>>",
    "classify": "Classify intent. ONE word: CREATE, EDIT, FIX, REVIEW, EXPLAIN, TEST, REFACTOR, SEARCH, GIT, CHAT",
}
