---
title: Configuration
description: Where settings live and what the common ones do.
---

## Files

| Path | Scope |
| --- | --- |
| `~/.localcode/config.toml` | Every project |
| `<project>/.localcode/config.toml` | This project, layered on top of the global file |

Set `LOCALCODE_HOME` to move `~/.localcode` somewhere else.

Most settings can be changed inside the app (`/model`, `/thinking`, `/permissions`, `/sounds`, `/vision`) and are saved for you. Edit the file by hand for anything else.

## `[runtime]`

| Key | Default | What it does |
| --- | --- | --- |
| `base_url` | `http://localhost:8081` | Where prompts are sent. Any URL is accepted. `LOCALCODE_BASE_URL` overrides it. See [Network Boundary](/localcode/concepts/network-boundary) |
| `model` | recommended for your Mac | Model tag |
| `internal_thinking_mode` | `off` | Hidden reasoning: `off` or `auto` |
| `thinking_budget_tokens` | `0` | Reasoning-token limit. `0` uses the model's default; a negative value turns reasoning off |
| `max_rounds` | `0` | Maximum model/tool rounds per turn. `0` is unlimited |
| `kv_cache_type_k` | `q8_0` | How the K half of the KV cache is stored |
| `kv_cache_type_v` | `turbo4` | How the V half of the KV cache is stored |
| `model_dir` | `~/.local/share/localcode/models` | Where models are downloaded |
| `vision_enabled` | `false` | Let the model see images |
| `request_timeout_seconds` | `600` | Timeout for one request to the model server |

## `[safety]`

| Key | Default |
| --- | --- |
| `confirm_destructive` | `true` |
| `confirm_installs` | `true` |
| `show_diff_before_apply` | `true` |
| `jail_to_project` | `true` |
| `auto_approve_agent` | `false` |

## `[search]`

Has `provider`, `google_api_key`, `google_cx`, `brave_api_key` and `serpapi_api_key`. The `web_search` tool the model uses ignores these and always searches DuckDuckGo.

## `[ui]` and `[logging]`

`ui.show_debug`, `ui.sounds_enabled`, `logging.enabled`, `logging.log_prompts`, `logging.log_responses`, `logging.max_days` (default 30).

## Other files in `~/.localcode/`

| File | Purpose |
| --- | --- |
| `mcp.json` | MCP servers |
| `hooks.toml` | Hooks for every project |
| `skills/` | Skills |

Each project also gets `<project>/.localcode/events.jsonl`, a local log of tool calls and turns. Set `LOCALCODE_TELEMETRY=0` to leave the UI turn-summary records out of it. Nothing in it is uploaded.
