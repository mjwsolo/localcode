# Codex Tool Call Sequences — Reference

Captured from OpenAI Codex CLI running real tasks. These sequences should be codified into LocalCode's prompts and behavior.

## Task: "make a pong game I can run locally using pygame"

### Sequence (6 steps):

1. **bash** `rg --files .` — list all files in directory (understand project structure)
2. **bash** `sed -n '1,220p' pong.py` — read existing pong.py if any (check what's already there)
3. **bash** `sed -n '1,220p' requirements.txt` — check if pygame is already in requirements
4. **write_file** `pong.py` (189 lines) — write COMPLETE game in one shot (not scaffold + edit)
5. **bash** `python -m py_compile pong.py` — verify file parses correctly (syntax check)
6. **bash** `python -c "import pygame; print(pygame.__version__)"` — verify pygame is installed

### Key patterns:
- **Explore FIRST**: reads existing files before writing (2 reads before 1 write)
- **Complete code**: writes 189 lines in ONE write_file, not incremental scaffold + edit
- **Verify AFTER**: syntax check + import check after writing
- **Does NOT install deps**: checks if pygame is installed, doesn't run pip install
- **Does NOT run the app**: just verifies it compiles and deps exist
- **Plain text summary at end**: describes what was done, offers next steps

### Order: explore → write → verify → summarize
### NOT: install → scaffold → edit → edit → edit → run

## Task: "add an AI opponent to pong.py"

### Sequence (3 steps):

1. **bash** `sed -n '1,240p' pong.py` — read the full current code first
2. **write_file** `pong.py` — surgical edit: +12 lines, -7 lines (not a full rewrite)
   - Added `AI_SPEED = 6` constant
   - Added `track_ball()` method to Paddle class (6 lines)
   - Replaced keyboard controls for right paddle with `right_paddle.track_ball(ball)` (1 line)
   - Updated UI text: "W/S vs UP/DOWN" → "W/S vs CPU"
   - Updated message text
3. **bash** `python -m py_compile pong.py` — verify it still parses

### Key patterns:
- **Read FIRST**: reads entire file before making changes
- **Minimal edit**: only +12/-7 lines changed, not a full rewrite
- **Codex uses write_file for edits too**: it doesn't use edit_file/old_string/new_string — it replaces the whole file but with minimal diff
- **Verify AFTER**: py_compile check
- **3 steps total**: read → write → verify. That's it.

### Order: read → write (minimal diff) → verify → summarize

---

## Codified Rules (from observations):

### For NEW files:
1. Explore: read existing files to understand project
2. Write complete code in ONE write_file (not scaffolds)
3. Verify: `python -m py_compile file.py`
4. Check deps: `python -c "import module"`
5. Summarize in plain text

### For EDITING existing files:
1. Read the entire file first
2. Write the modified file (whole file, minimal diff)
3. Verify: `python -m py_compile file.py`
4. Summarize in plain text

## Task: "make a pong game" — Claude Code (Anthropic)

### Sequence (2 steps):
1. **bash** `pip install pygame` — install deps first
2. **write_file** `pong.py` (119 lines) — write complete game

### Key differences from OpenAI Codex:
- No exploration (doesn't read existing files)
- Installs deps BEFORE writing (opposite of Codex)
- No verification step (no py_compile, no import check)
- Only 2 tool calls vs Codex's 6
- Much faster but less careful

### Best of both for LocalCode:
- Skip exploration for NEW files (Claude's speed)
- Read existing files before EDITING (Codex's carefulness)
- Always verify after writing (Codex's reliability)
- Don't install deps blindly — check first (Codex's approach)

## Task: "make a pong game" — Claude Opus (detailed analysis)

### Sequence (6 phases):

**Phase 1: Context gathering (PARALLEL)**
1. `glob *.py` — check existing python files
2. `glob **/requirements*.txt` — check existing deps
3. `bash python3 --version` — confirm python available
4. `bash python3 -c 'import pygame'` — check pygame installed
All 4 run in parallel.

**Phase 2: Conditional install**
5. `bash pip3 install pygame` — ONLY if step 4 failed

**Phase 3: Write**
6. `write_file pong.py` — complete game, 150-200 lines, one shot

**Phase 4: Verify**
7. `bash python3 -c 'import ast; ast.parse(...); print("syntax ok")'`

**Phase 5: Fix loop (if needed, max 2-3 iterations)**
8. `read_file pong.py`
9. `edit_file pong.py` — targeted fix
10. Re-verify

**Phase 6: Done** — plain text summary

### Key rules to codify:

| Rule | Why |
|------|-----|
| Parallel-first | Independent reads/checks batch together |
| Glob before Write | Never create without checking what exists |
| Write > Edit for new files | Edit is for surgical changes to existing files |
| Single Write for small files | Don't split 200-line file across multiple writes |
| Syntax-check, don't run | GUI apps can't be verified in CLI |
| Read before Edit | Hard rule — reject edits to unread files |
| Fix loop is bounded | Max 2-3 iterations, then surface error to user |
| No gratuitous verification | Don't re-read what you just wrote unless something failed |

### What NOT to do:
- No planning step for single files
- No mkdir for current directory
- No tests/README/git unless asked
- No incremental "skeleton then fill in" — wasted round trips

## Gemma 4 Specific Recommendations (from Claude Opus)

**Don't replicate autonomous sequencing. Build a state machine:**

```
GATHER_CONTEXT → PLAN → WRITE → VERIFY → FIX_LOOP → DONE
```

Each state has:
- Fixed tool calls the HARNESS makes (glob, read, bash)
- ONE LLM call to Gemma 4 for the creative part (generate code, diagnose error)
- Transition rules the HARNESS controls

**Key difference: Model generates content. Harness handles sequencing.**

| Cloud model approach | Local Gemma 4 approach |
|---------------------|----------------------|
| Model decides tool call order | Harness orchestrates, model fills in content |
| Model emits full file in one call | Harness may need to prompt section-by-section |
| Model picks which tools to parallelize | Harness defines the DAG statically |
| Model self-corrects on errors | Harness parses errors, re-prompts with context |

---

### Universal:
- NEVER run GUI apps via bash
- NEVER install deps without checking first
- ALWAYS verify after writing
- ALWAYS read before editing
- Complete code, not scaffolds/TODOs
