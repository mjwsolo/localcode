# Patterns from Claude Code (Production Codebase Study)

## Priority Implementation List for LocalCode

### Tier 1: Build NOW
1. **Agent Tool Loop** — model-driven, proven 100% success rate
2. **7 Core Tools** — read_file, write_file, edit_file, bash, grep, glob, ls
3. **System Prompt** — identity + rules + tool instructions (~300 tokens)
4. **Context Management** — tool result truncation, auto-compact at 85%
5. **Performance Metrics** — tokens/sec, context usage, timing on exit
6. **`/doctor` Command** — check server, model, config, resources

### Tier 2: Build NEXT
7. **Memory System** — persistent memories in ~/.localcode/projects/<path>/memory/
8. **Auto-Extract Memories** — learn user preferences from conversations
9. **`/init` Command** — analyze codebase, generate project config
10. **Session Resume** — `/resume` with search and replay
11. **Startup Profiling** — measure and optimize boot time

### Tier 3: Build LATER  
12. **Coordinator Mode** — multi-agent orchestration
13. **Skills System** — markdown skill definitions, user-extensible
14. **File Indexing** — background git-aware file suggestions
15. **Config Migrations** — named migration functions
16. **Keybinding Customization**

### Skip (Cloud/Enterprise Only)
- Bridge/remote, OAuth, analytics, sync, voice, plugins, buddy pet

## Key Architecture Patterns to Adopt

### Bootstrap State (from bootstrapCore)
Single state module with ALL global state. Currently scattered across 5+ files.

### JSONL History (from historyCore)
Append-only JSONL with file locking, project-scoped, session-ordered.

### Compaction (from compactCore)
9-section summary prompt with analysis scratchpad. Keep last 6 messages.
Post-compact: re-read top 5 recently read files.

### Permissions (from hooksCore)
3-source approval: hooks → user → classifier. Bash confirm for destructive.

### Tool Result Management (from microCompactCore)
- Per-tool limits: grep 20K, bash 30K, read_file 2000 lines
- Per-turn aggregate: 100K chars max
- Old results cleared with "[Old tool result cleared]"
- File-unchanged optimization for re-reads

### Commands (from commandsCore)  
Each command: name, description, prompt, allowed tools, execute function.
Key ones: /init, /compact, /doctor, /resume, /diff, /commit, /help
