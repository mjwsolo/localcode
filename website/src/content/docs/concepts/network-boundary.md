---
title: Network Boundary
description: Every outbound path localcode has, what triggers it, and what stays on your Mac.
---

**Inference runs on your Mac. Your code and prompts are never sent to a model
provider — there is no API key and no remote inference.** A handful of other
things do reach the network, and this page lists all of them rather than
rounding down to "offline".

## What never leaves your Mac

- **Inference.** Generation runs against a `llama-server` process on
  `http://localhost:8081`, started from the binary shipped inside the wheel.
  No hosted model, no API key, no remote fallback.
- **Your files, prompts and edits.** The agent reads and writes your working
  tree directly.
- **Session and event logs.** `<project>/.localcode/events.jsonl` is an
  append-only local audit trail of tool calls, turn boundaries, server
  lifecycle and nudges. It is never uploaded. Set `LOCALCODE_TELEMETRY=0` to
  turn off the UI turn-trace records inside it.

There is no analytics endpoint, no usage reporting, and no version check.

## Every outbound path

| # | What | Where it goes | When |
| --- | --- | --- | --- |
| 1 | **Connectivity probe** | TCP connect to `1.1.1.1:443` | Automatic. Once per turn at most — the verdict is cached for 30 s. No data is sent; the handshake alone tells localcode whether to let the model attempt downloads |
| 2 | **Model download** | `huggingface.co` | First launch, and whenever you pick a model you don't have |
| 3 | **Quant browsing** | Hugging Face repo tree API | Only while you browse other quantisations in the model picker; results are cached |
| 4 | **`llama-server` fallback binary** | `github.com/mjwsolo/localcode` Releases | Only if the bundled binary can't be used and a prebuilt one is fetched instead. Refused outright if TLS certificate verification fails — localcode will not download an executable over an unverified connection |
| 5 | **Experimental runner sources** | `git clone --depth 1 https://github.com/ggml-org/llama.cpp`, plus a specific PR ref | Only if you pick an experimental model whose architecture the bundled server lacks. localcode then builds that runner from source on your machine |
| 6 | **Voice model** (optional `voice` extra) | `huggingface.co/ggerganov/whisper.cpp` | First time you enable voice input |
| 7 | **Voice output voices** (optional `voice` extra) | `huggingface.co/rhasspy/piper-voices` | First time a speech voice is used |
| 8 | **`web_search` tool** | DuckDuckGo by default | Only when the model calls the tool |
| 9 | **`web_fetch` tool** | The URL named in the call | Only when the model calls the tool |
| 10 | **Skill install from a URL** | That URL | Only when you install one that way |
| 11 | **MCP servers** | Wherever you pointed them | Only for servers you configured in `~/.localcode/mcp.json` |
| 12 | **Shell commands** | Wherever the command goes | Whenever a `bash` call runs — see the approval note below |

`web_search` can be pointed at Google, Brave or SerpAPI instead of DuckDuckGo
by putting a key in your config; that sends the query to that provider instead.

## Approvals: read this before trusting the prompt

Whether a network-capable shell command stops to ask you depends on the
autonomy level:

- **`suggest`** — every `bash` call needs approval.
- **`auto_edit`** (the interactive default) — file edits are automatic; `bash`
  calls and installs still ask.
- **`full_auto`** — `bash` runs without asking. The safety layer still blocks
  the operations it always blocks, but "you approve each command" is **not**
  true at this level.

**`localcode run` (headless) forces `full_auto`**, because there is no human to
answer a prompt. A headless run can therefore make network requests — `curl`,
`pip install`, `web_fetch` — without any per-command confirmation. Run headless
against repositories and networks where that is acceptable.

The same applies to MCP tools and the two web tools: at `full_auto` they are
invoked without a prompt.

## The parts to be deliberate about

**`web_search` and `web_fetch` send content off the machine.** A search query is
written by the model from your conversation, so treat any use of these tools as
sending that text to a third party.

**MCP servers are arbitrary programs.** A remote MCP server sees whatever
arguments the model passes to its tools. Only configure servers you trust — see
[MCP](/localcode/guides/mcp).

**Building an experimental runner** clones and compiles third-party source on
your machine. That is a bigger trust decision than downloading a model file.

## Running with no network at all

Once a model is downloaded, the agent loop, file tools, shell tools and
inference all work without a connection. What breaks is exactly the table
above. The connectivity probe simply fails, and the model is told the machine
is offline so it stops attempting downloads. See
[Offline](/localcode/guides/offline).

## Verifying it yourself

localcode is Apache-2.0, so the boundary is auditable. The outbound call sites
are `app.py` (`is_online`, the probe), `bootstrap.py` (model downloads, the
release binary, the `git clone` for experimental runners), `hf_quants.py` (the
repo tree API), `voice.py` (speech assets), `tools/web_search.py` and
`tools/web_fetch.py`, `skills.py` (install from URL), and the MCP client.
`runtime.py`, `launcher.py` and `server_manager.py` talk to `localhost` only.

You can also watch what happened after the fact:

```sh
tail -f .localcode/events.jsonl | jq .
```
