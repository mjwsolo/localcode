# Agent Loop Blueprint

## Based on: Codex CLI (2381 lines), OpenCode (~300 lines), our proven 5-turn test

## Core Tools (7)
1. `read_file` — path, offset, limit. MUST read before edit.
2. `write_file` — path, content. MUST read first if file exists.
3. `edit_file` — path, old_string, new_string, replace_all. 
4. `bash` — command. Confirm destructive commands.
5. `grep` — pattern, path, include, max_results, context.
6. `glob` — pattern, path.
7. `ls` — path, ignore patterns.

## Agent Loop (~450 lines, 15 functions)

### Main Loop
```
for round in range(MAX_ROUNDS=20):
    response = call_model(messages, tools)
    
    if no tool_calls:
        display(response.text)
        break
    
    for tool in tool_calls:
        if destructive(tool): confirm()
        snapshot_if_file_change(tool)
        result = execute(tool)
        result = truncate(result)  # per-tool limits
        messages.append(tool_result)
    
    if context > 85%: compact(messages)
```

### Context Management
- Tool result limits: grep 20K, bash 30K, read_file 2000 lines, others 50K
- Per-turn aggregate: 100K chars max
- Auto-compact at 85% of 32K budget
- Keep last 6 messages on compact
- File-unchanged optimization: stub for re-reads

### System Prompt (~300 tokens)
Identity + rules + tool usage instructions + environment info.

### Display
- Stream text live during generation
- Tool calls shown as progress: `● read_file path=main.py`
- Tool results shown truncated: `  └ 42 lines`
- Errors shown in red
- Thinking shown as dim indicator

### Error Handling
- Connection/timeout: retry 2x with backoff
- Malformed tool call: feed error back to model
- Max output tokens: inject "continue" message, retry 2x
- Ctrl+C: clean abort, save partial state

### Safety
- bash: confirm rm, git push, sudo, pip install
- write_file: snapshot before overwrite
- edit_file: show diff
- Max 20 rounds per task
