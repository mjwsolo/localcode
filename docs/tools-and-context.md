# Tools and Context Management — Complete Specification

## Final Tool List: 12

### CORE (model can request via structured output): 8 tools
1. `read(file_path, offset?, limit?)` — Read files with line numbers
2. `write(file_path, content)` — Create/overwrite files
3. `edit(file_path, old_string, new_string, replace_all?)` — Surgical search/replace
4. `glob(pattern, path?)` — Find files by pattern
5. `grep(pattern, path?, glob_filter?, context?)` — Regex search file contents
6. `bash(command, timeout?)` — Run shell commands
7. `list_dir(path?, depth?)` — Tree-style directory listing
8. `ask_user(question)` — Prompt user for input (use sparingly)

### HARNESS-ONLY (model never calls these directly): 4 tools
9. `diff(file_path, new_content)` — Preview changes before applying
10. `syntax_check(file_path)` — Auto-run after every write/edit
11. `undo()` — Revert last file change
12. `progress(step, status)` — UI updates

**The split matters:** Tools 1-8 are what the model can request. Tools 9-12 are what the harness calls automatically. The model doesn't choose to syntax-check — the harness does it every time, unconditionally.

---

## Context Management Components

### Component 1: Token Budget

```python
class TokenBudget:
    def __init__(self, model_max_context=8192):
        self.reserved_system = 500   # system prompt
        self.reserved_tools = 400    # tool definitions
        self.reserved_output = 2048  # response budget
        self.available = model_max_context - 500 - 400 - 2048  # ~5344 for file context

    def count(self, text): return len(text) // 4  # 1 token ≈ 4 chars for code
    def fits(self, text): return self.count(text) <= self.available
    def truncate_to_budget(self, text, budget):
        max_chars = budget * 4
        if len(text) <= max_chars: return text
        return text[:max_chars].rsplit("\n", 1)[0] + "\n... (truncated)"
```

### Component 2: File Summarizer (AST-based, zero LLM cost)

Compresses a 400-line Flask app to ~150 tokens:
```
from flask import Flask, request, jsonify
from models import db, User, Post

class PostService:
    def create(self, user_id, title, body): ...
    def get(self, post_id): ...
    def list_by_user(self, user_id, page, per_page): ...

def create_app(config): ...
def register_routes(app): ...
```

20x compression. Extracts: imports, class/function signatures, constants.

### Component 3: Context Assembler

Decides what goes into each LLM call based on task type:

**For edits:**
- Priority 1 (60% budget): Target file (or relevant section if >100 lines)
- Priority 2 (30% budget): Related file signatures (imports, function names)
- Priority 3 (10% budget): Task description

**For creates:**
- Priority 1: Reference files (to match style)
- Priority 2: Directory structure
- Priority 3: Task description

**For fixes:**
- Priority 1: Error output (most important!)
- Priority 2: The file
- Priority 3: Previous fix attempt diff

### Component 4: Relevance Finder (zero LLM cost)

Ranks files by relevance using:
- Filename keyword match (+10 per keyword)
- Content keyword match (+1 per occurrence)
- Recency bonus (+5 if modified today)

### Component 5: Conversation History Manager

Multi-turn conversations blow up context. Aggressive compression:
- Last turn: always full content
- Older turns: auto-summarized to key facts (files changed, errors)
- Rule-based summary, no LLM cost

### Component 6: System Prompts (TINY, task-specific)

```python
SYSTEM_PROMPTS = {
    "create": "You generate code. Output ONLY valid code. No fences. No explanations.",
    "edit": "Output SEARCH/REPLACE blocks:\n<<<SEARCH\nexact lines\n===\nreplacement\nSEARCH>>>",
    "review": "List issues: LINE {n}: {severity} — {issue}",
    "explain": "Explain concisely. Max 5 sentences.",
    "plan": "Decompose into steps: STEP {n}: {ACTION} | {target} | {desc}\nDEPENDS_ON: ...",
    "fix": "Fix the bug. Output ONLY search/replace blocks.",
    "classify": "Respond ONE word: CREATE, EDIT, FIX, REVIEW, EXPLAIN, TEST, REFACTOR, SEARCH, GIT"
}
```

Each prompt ~50-80 tokens. NOT 500.

---

## Summary

| Component | Purpose | Uses LLM? |
|-----------|---------|-----------|
| Token counter | Enforce budget | No |
| File summarizer | Compress to signatures | No (AST) |
| Context assembler | Pick what goes in each call | No |
| Relevance finder | Find related files | No (keyword+recency) |
| Conversation manager | Compress multi-turn history | No |
| System prompts | Tiny task-specific prompts | — |
| Intent classifier | Route requests | Fallback only |
| 8 core tools | File/shell operations | No |
| 4 harness tools | Validation/UI | No |
| Orchestrator + Planner | Multi-step coordination | Yes (1 call each) |

**6 out of 9 components use zero LLM calls.** Offload everything to deterministic code. Use model only for: generating code, diagnosing bugs, decomposing plans.
