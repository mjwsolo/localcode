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

## Codified Rules (from this observation):

1. Read existing files first to understand what's there
2. Write complete working code in ONE write_file (not scaffolds)
3. After writing Python: run `python -m py_compile` to verify
4. After verifying: check if dependencies are importable
5. Don't run GUI apps (they block) — just tell user how to run
6. End with plain text summary + next steps
