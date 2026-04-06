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

### Universal:
- NEVER run GUI apps via bash
- NEVER install deps without checking first
- ALWAYS verify after writing
- ALWAYS read before editing
- Complete code, not scaffolds/TODOs
