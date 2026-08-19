---
title: Configuration
description: Config file locations, layering, and the sections you are most likely to touch.
---

## Where config lives

| Path | Scope |
| --- | --- |
| `~/.localcode/config.toml` | Global, machine-wide |
| `<project>/.localcode/config.toml` | Project — layered on top of global |

`LOCALCODE_HOME` overrides `~/.localcode` if you need to relocate it.

Most settings are changed from inside the TUI (`/model`, `/thinking`,
`/permissions`, `/sounds`, …) and written back to the global file
automatically. Editing the TOML by hand is for the settings the UI doesn't
expose.

## Sections

### `[runtime]`

The model server and generation settings.

| Key | Default | Notes |
| --- | --- | --- |
| `provider` | `llama_cpp` | Inference backend |
| `base_url` | `http://localhost:8081` | Local server address |
| `model` | *(per-Mac recommendation)* | Model tag |
| `mode` | `fast` | `fast` or reasoning-biased |
| `internal_thinking_mode` | `off` | Hidden reasoning: `off` or `auto`. Off by default |
| `thinking_budget_tokens` | `0` | `0` = catalogue default, negative disables |
| `max_rounds` | `0` | `0` = unlimited interactive loop |
| `kv_cache_type_k` | `q8_0` | K cache quantisation |
| `kv_cache_type_v` | `turbo4` | V cache quantisation (TurboQuant) |
| `model_dir` | *(empty)* | Where GGUFs download; blank → `~/.local/share/localcode/models` |
| `vision_enabled` | `false` | Image input |
| `request_timeout_seconds` | `600` | Per-request timeout |

### `[safety]`

| Key | Default |
| --- | --- |
| `confirm_destructive` | `true` |
| `confirm_installs` | `true` |
| `show_diff_before_apply` | `true` |
| `jail_to_project` | `true` |
| `auto_approve_agent` | `false` |

### `[search]`

`provider` defaults to `duckduckgo`. Setting `google_api_key` + `google_cx`,
`brave_api_key` or `serpapi_api_key` switches `web_search` to that provider —
which changes who sees your queries. See
[Network Boundary](/localcode/concepts/network-boundary).

### `[ui]` and `[logging]`

`ui.show_debug`, `ui.sounds_enabled`, and `logging.enabled` /
`logging.log_prompts` / `logging.log_responses` / `logging.max_days` (30).

## Other files under `~/.localcode/`

| File | Purpose |
| --- | --- |
| `mcp.json` | MCP server definitions (key: `mcpServers`) |
| `hooks.toml` | Global lifecycle hooks |
| `skills/` | Global skills + `registry.json` |
| `telemetry/turns.jsonl` | Local-only turn traces |
