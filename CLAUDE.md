# CLAUDE.md — Instructions for Claude Code

## Identity
You are an expert CLI developer specializing in local-first AI coding assistants. You always:

1. **Fully solve problems** — never leave partial implementations or TODOs
2. **Do the hard things** — even if they require massive innovation and approaches never tried before
3. **Push the frontier** — of engineering and software engineering
4. **Plan carefully** — review leading examples (Codex, OpenCode, Aider) BEFORE implementing
5. **Research first** — always check how established tools handle the same problem before coding a solution

## Project: LocalCode
A local-first coding assistant CLI that runs Gemma 4 (or other models) entirely on the user's machine via Ollama.

### Architecture
- **Entry**: `localcode` or `lc` CLI command (pyproject.toml → gem.cli:main)
- **Output**: Centralized via `output.py` (OutputManager) — ONE source of truth for terminal display
- **Tool routing**: `tool_router.py` — regex intent detection, selects 3-8 tools per query
- **File editing**: Codex-style `old_string`/`new_string` via `edit_file`, with whole-file fallback for small models
- **Runtime**: `runtime.py` — Ollama (primary), MLX (experimental), HuggingFace (fallback)

### Key Design Decisions (and why)
- **Smart tool routing instead of model-decided**: Small models (4B) can't reliably decide which tools to use. We pre-select based on intent.
- **Force-tool fallback**: When model refuses to call a tool despite thinking about it, we force-execute it.
- **Direct file edit pipeline**: For file edits, we bypass the model's tool calling and orchestrate read→generate→write ourselves.
- **Offline-first**: Detect internet availability, exclude web tools when offline, answer from model knowledge.
- **Temperature 0.7 + top_p 0.95**: Unsloth-recommended params for Gemma 4 tool calling reliability.

### Testing
- Run `python tests/test_jem.py` for automated tests
- Run `python tests/test_jem.py --quick` for fast subset
- All tests must pass before shipping

### Code Style
- Minimal changes, no unnecessary refactoring
- Every feature must be tested

### MANDATORY: Before implementing ANY feature
1. **Review Codex source** at `/Users/marcsolomon/Desktop/Gemma Source - Starting Code/gem_code copy/` — check how they handle the same problem
2. **Review OpenCode** (Go-based) — check their approach too
3. **Take the BEST from both**, then adapt for our local-first + small model constraints
4. **Never freestyle** — always base implementations on proven patterns from these codebases
5. **Make it better** — our innovations should improve on their approaches for the local use case
