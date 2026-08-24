---
title: Architecture
description: The pieces localcode is made of, from the TUI down to the inference server.
---

## The stack

This is the default setup. The rest of this page describes it:

<div style="display:flex;flex-direction:column;align-items:center;gap:0.55rem;margin:1.75rem 0;font-family:var(--lc-font-code, ui-monospace, monospace);font-size:0.82rem;line-height:1.35;">
  <div style="display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:0.5rem;">
    <span style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.85rem;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);white-space:nowrap;">Textual TUI</span>
    <span style="color:var(--lc-accent);font-weight:700;">&rarr;</span>
    <span style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.85rem;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);white-space:nowrap;">Agent loop</span>
    <span style="color:var(--lc-accent);font-weight:700;">&rarr;</span>
    <span style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.85rem;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);">Tools (read / edit / bash / search / MCP)</span>
  </div>
  <span style="color:var(--lc-accent);font-size:1.25rem;line-height:1;">&darr;</span>
  <div style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.9rem;text-align:center;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);"><code style="background:none;padding:0;">runtime.base_url</code> (default <code style="background:none;padding:0;">http://localhost:8081</code>)</div>
  <span style="color:var(--lc-accent);font-size:1.25rem;line-height:1;">&darr;</span>
  <div style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.9rem;text-align:center;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);"><strong>llama-server</strong> started by localcode<br><span style="opacity:0.72;font-size:0.85em;">llama.cpp fork + TurboQuant KV compression</span></div>
  <span style="color:var(--lc-accent);font-size:1.25rem;line-height:1;">&darr;</span>
  <div style="border:1px solid var(--lc-accent);border-radius:var(--lc-radius,4px);padding:0.5rem 0.9rem;text-align:center;background:color-mix(in srgb, var(--lc-accent) 9%, transparent);">GGUF weights on disk</div>
</div>

- **TUI** - the main product interface. Setup, mode choice, the model picker, and chat are all screens in one Textual app.
- **Agent loop** - the model creates tool calls, the tools run, and the results go back to the model. Turn state, todos, and goal context continue across user messages.
- **Tools** - file reading and editing, glob/grep, shell commands, project checks, syntax checks, code navigation and symbol inspection, notebook editing, app launching, the two network tools, and any MCP tools you have configured.
- **Inference server** - by default, localcode starts its own `llama-server` (the binary included in the wheel) at `localhost:8081`.

## Built specifically for small models

localcode is designed specifically to enable high-performance agentic coding with local models on consumer hardware. The prompts, the agent loop, and the model server are all tuned for small quantised models rather than a frontier model:

- **Prompts tuned for small models** - the system prompt runs a plan-then-execute loop: lay out the steps, keep exactly one in progress, and require evidence before a task counts as done, instead of assuming the model self-organises.
- **Finishes the whole task** - the loop keeps working until the goal is actually done, so the model does not stop mid-task and call it finished.
- **Tool-call repair** - malformed JSON arguments and extra spaces in tool names are fixed instead of failing the round.
- **Recovery modes** - separate paths handle cut-off tool calls and reasoning loops, each with its own exit reason in the event stream.
- **Long context on 16 GB** - the llama.cpp fork compresses the KV cache with TurboQuant (about 3.8x smaller than f16), so long contexts fit on small machines.
- **Fast multi-turn** - the server snapshots its state at turn boundaries, so the next turn reuses the prefix instead of re-reading it.
- **Speculative decoding** - an optional draft model speeds up generation without changing the output.
- **Hidden reasoning is off by default** - turn it on per model with `/thinking`; models without a reasoning channel say so instead of silently ignoring it.
- **Syntax checks before shell runs** - tree-sitter catches broken edits before they run.
