---
title: Network Boundary
description: What stays on your Mac, and which features use the network.
---

By default, inference runs on your Mac. localcode needs no API key and has no cloud fallback. This page lists every feature that uses the network.

## What stays on your Mac

- **Inference.** Prompts go to the model server localcode starts at `http://localhost:8081`.
- **Your files, prompts and edits.**
- **Logs.** `<project>/.localcode/events.jsonl` records tool calls and turns. It is never uploaded.

There is no analytics, usage reporting or version check.

## What uses the network

| What | Where | When |
| --- | --- | --- |
| Connectivity check | TCP connect to `1.1.1.1:443` | At the start of a turn, at most once every 30 seconds. Opens and closes the connection; sends no data |
| Model download | `huggingface.co` | First launch, and when you choose a model you do not have |
| Quant browsing | Hugging Face API | When you browse other quantisations in the model picker |
| Voice models (optional) | `huggingface.co` | The first time you turn on voice input or output |
| `web_search` tool | DuckDuckGo | When the model calls it. Never prompts |
| `web_fetch` tool | The URL in the call | When the model calls it. Never prompts |
| Skill install from URL | That URL | When you install one that way |
| MCP servers | Wherever you pointed them | When the model calls their tools. Never prompts |
| Shell commands | Wherever the command goes | Subject to [approvals](/localcode/start-here/permissions) |
| Code-search embeddings | `huggingface.co` | Only if you have installed `sentence-transformers`, the first time the search index is built. Without it, the index is built locally and downloads nothing |
| Server binary fallback | GitHub Releases | Only if the server binary shipped in the package is missing |

The `search` section of the config file has keys for Google, Brave and SerpAPI. The `web_search` tool does not use them; it always searches DuckDuckGo.

## Moving inference off your Mac

`runtime.base_url` in `~/.localcode/config.toml`, or the `LOCALCODE_BASE_URL` environment variable, sets where prompts are sent. Any URL is accepted. Point it at a `llama-server` on another machine and every prompt, every file localcode reads for context and every model reply travels to that machine. The app does not warn you. Check the current value with `/status`.

## Things to think about

- `web_search` and `web_fetch` send text from your conversation to a third party. The model decides when to call them, and they never ask first.
- An MCP server is a program you run. A remote one sees every argument the model sends it.
- `localcode run` (headless) never prompts for anything.

## Running with no network

After the model is downloaded, everything in "What stays on your Mac" keeps working. See [Offline](/localcode/guides/offline).

## Checking for yourself

localcode is Apache-2.0. You can watch what it records:

```sh
tail -f .localcode/events.jsonl | jq .
```
