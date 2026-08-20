---
title: Configuration
description: Config file locations, layering, and the sections you are most likely to touch.
---

## Where the config is stored

| Path | Scope |
| --- | --- |
| `~/.localcode/config.toml` | Global, for the whole machine |
| `<project>/.localcode/config.toml` | Project — added on top of the global config |

Set `LOCALCODE_HOME` to replace `~/.localcode` with another location.

You can change most settings in the TUI (`/model`, `/thinking`, `/permissions`, `/sounds`, …). Changes are saved to the global file automatically. Edit the TOML by hand only for settings that are not available in the UI.

## Sections

### `[runtime]`

Settings for the model server and text generation.

| Key | Default | Notes |
| --- | --- | --- |
| `provider` | `llama_cpp` | Inference backend |
| `base_url` | `http://localhost:8081` | The URL that receives chat completions. **Any URL is allowed. It is not checked or limited to localhost.** You can override it with `LOCALCODE_BASE_URL` |
| `model` | *(per-Mac recommendation)* | Model tag |
| `mode` | `fast` | `fast` or focused more on reasoning |
| `internal_thinking_mode` | `off` | Hidden reasoning: `off` or `auto`. Off by default |
| `thinking_budget_tokens` | `0` | `0` = catalogue default; a negative value disables it |
| `max_rounds` | `0` | `0` = unlimited interactive loop |
| `kv_cache_type_k` | `q8_0` | K cache quantisation |
| `kv_cache_type_v` | `turbo4` | V cache quantisation (TurboQuant) |
| `model_dir` | *(empty)* | Where GGUFs are downloaded; blank → `~/.local/share/localcode/models` |
| `vision_enabled` | `false` | Allows image input |
| `request_timeout_seconds` | `600` | Timeout for each request |

### `[safety]`

| Key | Default |
| --- | --- |
| `confirm_destructive` | `true` |
| `confirm_installs` | `true` |
| `show_diff_before_apply` | `true` |
| `jail_to_project` | `true` |
| `auto_approve_agent` | `false` |

### `[search]`

This section has `provider` (default `duckduckgo`), plus `google_api_key`, `google_cx`, `brave_api_key`, and `serpapi_api_key`.

**None of these settings change the `web_search` tool used by the model.** That tool always searches DuckDuckGo through the `ddgs` package. The other providers are part of an older code path that the agent does not use. Adding a key does not change where searches go.

### `[ui]` and `[logging]`

The settings are `ui.show_debug`, `ui.sounds_enabled`, `logging.enabled`, `logging.log_prompts`, `logging.log_responses`, and `logging.max_days` (30).

## Other files in `~/.localcode/`

| File | Purpose |
| --- | --- |
| `mcp.json` | MCP server definitions (key: `mcpServers`) |
| `hooks.toml` | Global lifecycle hooks |
| `skills/` | Global skills + `registry.json` |

UI turn traces are **no longer** written to `~/.localcode/telemetry/turns.jsonl`. They are now stored in each project's `<project>/.localcode/events.jsonl` file as `ui_turn_end` records. Set `LOCALCODE_TELEMETRY=0` to stop creating them. In either case, localcode does not upload the records.
