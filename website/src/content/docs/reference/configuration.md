---
title: Configuration
description: Environment variables, the run directory, localcode.json, LOCALCODE.md, and the classic interface.
---

There is nothing to configure before the first run. Most day-to-day settings live in the interface: `/models`, `/settings`, `/themes`, `/permissions`, `/mcps`. This page covers what lives outside it.

## Environment variables

| Variable | Effect |
| --- | --- |
| `LOCALCODE_FRONTEND` | `ui` (default) or `classic` for the previous 0.3 interface |
| `LOCALCODE_MODEL_DIR` | Where GGUFs live. Default `~/.local/share/localcode/models`. The picker's **Models folder** entry changes the same setting |
| `LOCALCODE_MODELS_DIR` | Older override name; `LOCALCODE_MODEL_DIR` takes precedence |
| `LOCALCODE_AGENT_RUN_DIR` | Supervisor logs, the per-session runtime config, and the voice runtime. Default `~/.local/share/localcode-agent/run` |
| `LOCALCODE_MIC` | Microphone device for voice input |
| `LOCALCODE_ENABLE_EXA=1` + `EXA_API_KEY` | Web search through Exa instead of the keyless DuckDuckGo backend |
| `LOCALCODE_ENABLE_PARALLEL=1` + `PARALLEL_API_KEY` | Web search through Parallel instead of the keyless DuckDuckGo backend |
| `LOCALCODE_WEBFETCH_MAX_CHARS` | Cap on text returned by a page fetch. Default `20000` |
| `OPENCODE_DISABLE_LSP_DOWNLOAD` | Default `1`. Language servers are installed only through `/lsp` after you confirm |
| `LOCALCODE_UI_BIN` | Developer override: path to the interface binary |
| `LOCALCODE_LLAMA_SERVER` | Developer override: path to a `llama-server` binary |

## Where state lives

| Path | What is there |
| --- | --- |
| `~/.local/share/localcode/models/` | Downloaded GGUFs and image projectors |
| `~/.local/share/localcode-agent/run/` | `server.log` and supervisor logs, the session config, the voice runtime |
| `<project>/.localcode-agent/` | Per-project runtime state: sessions and the plugin log |

The launcher writes nothing into your project. The runtime keeps its per-project state in `.localcode-agent/`; add it to `.gitignore` if you do not want it tracked.

## `LOCALCODE.md`

The runtime reads project instructions from `LOCALCODE.md` in the project root: the checks to run, conventions, files to leave alone. It is included in every session in that project. `/project-context` opens it for editing.

## `localcode.json`

An optional `localcode.json` in the project root follows the opencode config schema. Use it for MCP servers, permission rules, keybinds, and the theme:

```json
{
  "$schema": "https://localcode.dev/schema/config.json",
  "theme": "localcode",
  "permission": {
    "bash": { "pytest *": "allow" }
  },
  "mcp": {
    "filesystem": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

See [MCP](/localcode/guides/mcp) for the server shapes. The model provider section is not yours to set: each session gets a generated config that points the runtime at the local `llama-server`, and the project file is layered on top of it.

## Skills

Skill folders are read from `<project>/.localcode-agent/skills/` and `~/.config/localcode-agent/skills/`. See [Skills](/localcode/guides/skills-and-hooks).

## The classic interface

`localcode --classic` (or `LOCALCODE_FRONTEND=classic`) opens the 0.3 interface. It keeps its own configuration in `~/.localcode/config.toml` and `<project>/.localcode/`, and its own variables such as `LOCALCODE_AUTONOMY` and `LOCALCODE_HOME`. The headless `localcode run` uses the same files. Those are documented in the 0.3 docs, reachable from the version switcher in the header.
