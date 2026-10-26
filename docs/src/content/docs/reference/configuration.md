---
title: Configuration
description: Environment variables, the run directory, localcode.json, and LOCALCODE.md.
---

There is nothing to configure before the first run. Most day-to-day settings live in the interface: `/models`, `/settings`, `/themes`, `/permissions`, `/mcps`. This page covers what lives outside it.

## Context

Sessions start with minimal context: assistant instructions, the working directory,
tool definitions and the current conversation. They do not automatically attach
workspace listings, project instruction files or skill catalogs. Files can still
be read when needed for the work you request. Older tool arguments and results
are abbreviated in requests; the complete conversation stays saved on disk.

Use `localcode --context full` to include project instructions, skill and MCP
instructions, automatic workspace listings and unabridged tool history in requests.
Normal context-window limits and compaction still apply. Combine
it with `--trust-project` when using repository configuration. Saved warmup
captures are not replayed in minimal mode.

MCP servers without an explicit `enabled: true` setting stay disconnected at
minimal startup. Enable the server you need with `/mcps`, or set `enabled: true`
in its configuration. Full context mode retains the previous connection defaults.

Switching models in an active chat offers **Start a new chat** first. The previous
conversation remains saved, and unsent text moves to the new chat. Choose **Continue this chat** to have the new model
process that conversation. Selecting an already running model keeps it loaded.

After an upgrade, exit running LocalCode windows before launching the updated
version. The launcher refuses to attach a new interface to an outdated model
supervisor, so an old process cannot silently keep the previous behavior active.

## Where state lives

| Path | What is there |
| --- | --- |
| `~/.local/share/localcode/models/` | Downloaded GGUFs and image projectors |
| `~/.local/share/localcode-agent/opencode.db` | Every session: messages, tool calls and the event history that `--resume` replays. A SQLite file, and the largest thing here after the models |
| `~/.local/share/localcode-agent/log/` | The interface's own log |
| `~/.local/share/localcode-agent/run/` | `server.log` (every model request, token counts and speeds), `supervisor.log` (model loading, context and slots), the session config, the voice runtime |
| `~/.config/localcode-agent/` | Your `localcode.json` and user-level skills |
| `~/.local/state/localcode-agent/` | Prompt history and the last model you chose |
| `~/.cache/localcode-agent/` | The interface's cache; safe to delete |
| `~/.localcode/` | `config.toml` for the launcher, and its own small cache |
| `<project>/.localcode-agent/` | Per-project state: `localcode-plugin.log` (why the agent was nudged, stopped, or had a path corrected) and project skills |

Sessions are not stored in the project. The launcher writes nothing into your project; the runtime keeps a little per-project state in `.localcode-agent/`, which you can add to `.gitignore`.

To watch a session as it runs:

```
tail -f ~/.local/share/localcode-agent/run/server.log
tail -f .localcode-agent/localcode-plugin.log
```

## `LOCALCODE.md`

`LOCALCODE.md` stores project instructions: checks to run, conventions and files
to leave alone. `/project-context` opens it for editing. Full context mode
includes it automatically; in minimal mode, ask the agent to read it when relevant.

## `localcode.json`

An optional `localcode.json` in the project root follows the opencode config schema. Use it for MCP servers, permission rules, keybinds, and the theme:

Project configuration and discovered project plugins are disabled by default.
Launch with `localcode --trust-project` after reviewing them. User-global
configuration remains enabled. The flag is required on each fresh or attached
launch that should use repository settings.

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

Their catalog is included in full context mode. Minimal mode does not list skills
automatically; a named skill can still be loaded when explicitly requested.

## Local service diagnostics

`localcode doctor` reports the selected context mode, detected GGUF files,
model process IDs and memory usage, and the running service state. It omits
API keys, prompt contents and full process arguments. Use it when a local API
cannot connect or a model switch appears to leave a previous server running.

The picker detects existing GGUFs in its configured folder and subfolders,
the previous default folder, `~/models`, Hugging Face Hub caches, and the standard
LM Studio folders. Selecting one links to the existing file rather than copying
or downloading its weights. Models stored elsewhere can be loaded by full path
with `localcode --model /path/to/model.gguf`.

Selecting existing weights does not download a missing speculative drafter. The
model runs without it, only slower. New model downloads from the picker include the
drafter the model ships with, and its size is counted in the size the picker shows.

Available quant listings refresh in the background. Local and cached models
remain selectable while the remote listing is slow or unavailable. A failed
model load releases its new process; validation failures preserve a healthy
previous process. Model switches also refuse to interrupt requests from other clients.
