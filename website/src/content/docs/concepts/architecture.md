---
title: Architecture
description: The pieces localcode is made of, from the launcher down to the inference server.
---

## The stack

`localcode` is a small Python launcher. It starts three things, all on `127.0.0.1`:

<div style="display:flex;flex-direction:column;align-items:center;gap:0.55rem;margin:1.75rem 0;font-family:var(--lc-font-code, ui-monospace, monospace);font-size:0.82rem;line-height:1.35;">
  <div style="display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:0.5rem;">
    <span style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.85rem;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);white-space:nowrap;">Launcher</span>
    <span style="color:var(--lc-accent);font-weight:700;">&rarr;</span>
    <span style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.85rem;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);white-space:nowrap;">Supervisor + llama-server</span>
    <span style="color:var(--lc-accent);font-weight:700;">&rarr;</span>
    <span style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.85rem;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);">Runtime + plugin</span>
  </div>
  <span style="color:var(--lc-accent);font-size:1.25rem;line-height:1;">&darr;</span>
  <div style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.9rem;text-align:center;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);">Tools (read / edit / bash / grep / websearch / MCP)</div>
  <span style="color:var(--lc-accent);font-size:1.25rem;line-height:1;">&darr;</span>
  <div style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.9rem;text-align:center;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);"><code style="background:none;padding:0;">http://127.0.0.1:PORT/v1</code><br /><span style="opacity:0.72;font-size:0.85em;">llama-server, llama.cpp fork + TurboQuant KV compression</span></div>
  <span style="color:var(--lc-accent);font-size:1.25rem;line-height:1;">&darr;</span>
  <div style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.9rem;text-align:center;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);">GGUF weights on disk</div>
</div>

1. **Launcher** - `localcode` picks free ports, starts the supervisor, writes a per-session config in the run directory, and runs the interface binary in your project. It writes nothing into the project. See [Configuration](/localcode/reference/configuration).
2. **Supervisor** - a Python process that owns `llama-server`. The model server listens on `127.0.0.1:8123` or the next free port. The supervisor serves the model picker and status API on a control port from `8323`. It downloads a quant when you pick one, reloads the server on the same port when you switch models, and shuts the server down when you exit.
3. **Runtime + plugin** - the interface, `localcode-ui`, is a fork of opencode branded localcode. Every cloud feature is removed. It talks only to `http://127.0.0.1:PORT/v1`. The config loads localcode's discipline plugin from the package.
4. **Tools** - file reading and editing, glob and grep, shell commands, todo list, web search and fetch, language servers you install through `/lsp`, and any MCP servers you configure.

One model server runs per user. Opening `localcode` in another terminal starts a separate interface session attached to that server. Close either window without ending the other session; use `/models` to switch the shared model.

The previous interface (0.3) is still in the package as `localcode --classic`. It runs its own Python agent loop against the same `llama-server`. `localcode run`, the headless mode, uses that loop too.

## The discipline plugin

Small local models finish a task when the loop makes them. The plugin adds localcode's completion discipline to the runtime:

- **Plan first** - the model lays out the steps in the todo list before editing.
- **Keep going** - a turn does not end while todo items are still open.
- **Prove it** - after edits, the plugin runs the project's own typecheck and tests and feeds failures back to the model.
- **Audit stubs** - placeholder code and `TODO` bodies are flagged before the task counts as done.
- **No foreground servers** - a command that would block the session, such as a dev server, is refused with a hint to run it in the background.
- **Stop cleanly** - when no progress is being made across rounds, the plugin asks the model to wrap up and then stops, instead of looping.

## Built specifically for small models

localcode is designed to enable agentic coding with local models on consumer hardware. The prompts, the loop, and the model server are tuned for small quantised models rather than a frontier model:

- **Long context on 16 GB** - the llama.cpp fork compresses the KV cache with TurboQuant (about 3.8x smaller than f16), so long contexts fit on small machines. See [Unified Memory](/localcode/concepts/unified-memory).
- **Fast multi-turn** - the server keeps the prompt prefix between turns, so the next turn does not re-read it.
- **Compact tool definitions** - the runtime shortens built-in tool descriptions and large MCP schemas before sending them to the model. This leaves more room for the task and project context, especially on a 16 GB Mac.
- **Hidden reasoning is off by default** - the server starts with `--reasoning off`. Small models finish faster and follow tool calls better without it.
- **Vision on demand** - the image projector for a vision-capable model is a separate download through `/vision`. Nothing is fetched until you ask.

## Everything is loopback

The launcher, supervisor, model server, and interface all bind to `127.0.0.1`. There is no telemetry, no update check, and no account. The network is used only for downloads you request in the picker, `/vision`, `/lsp`, or voice, and for the web tools and MCP servers when the model calls them. See [Network Boundary](/localcode/concepts/network-boundary).
