# Tools

## Native Gemma 4 Tool Calling

LocalCode uses Gemma 4's native tool calling system. The model decides when to call tools and which tool to use — no regex routing needed.

## Available Tools

| Tool | Description | Example |
|------|-------------|---------|
| `read_file` | Read file contents | `read_file({"path": "main.py"})` |
| `write_file` | Create/overwrite a file | `write_file({"path": "app.py", "content": "..."})` |
| `edit_file` | Surgical find-and-replace | `edit_file({"path": "f.py", "old_string": "...", "new_string": "..."})` |
| `bash` | Run shell commands | `bash({"command": "pip install pygame"})` |
| `grep` | Search code by pattern | `grep({"pattern": "def main"})` |
| `glob` | Find files by pattern | `glob({"pattern": "*.py"})` |
| `web_search` | Search the web (DuckDuckGo) | `web_search({"query": "pygame docs"})` |

## How Tool Calling Works

1. User asks: "Read the file pyproject.toml"
2. Model outputs: `<|tool_call>call:read_file{path:<|"|>pyproject.toml<|"|>}<tool_call|>`
3. Server parses the native Gemma 4 tool call format
4. App executes `read_file("pyproject.toml")`
5. Result sent back to model as `role: tool` message
6. Model continues with the file contents in context

## Tool Call Format

Gemma 4 uses special tokens for tool calls (not JSON):

```
Declaration:  <|tool>declaration:NAME{...}<tool|>
Call:         <|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>
Response:     <|turn>tool\n[result]<turn|>
```

The llama-server's chat parser handles this automatically when tools are passed via the OpenAI-compatible API.

## Reliability

Gemma 4 26B-A4B scores **85.5%** on tau2-bench (tool calling benchmark). At IQ3_S quantization, tool calling works reliably for direct requests.
