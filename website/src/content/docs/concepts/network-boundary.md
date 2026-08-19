---
title: Network Boundary
description: The outbound paths known in the current implementation, what triggers them, and what stays on your Mac by default.
---

**In the default, supported configuration, inference runs on your Mac: no API
key, no model provider, no remote fallback.** Several other things do reach the
network, and one configuration change moves inference itself off the machine.
This page names the ones known in the current implementation, rather than
rounding down to "offline".

## Scope: the default configuration

Everything below assumes `runtime.base_url` is left at its default,
`http://localhost:8081`. That value is freely configurable — see
[Inference endpoint](#inference-endpoint-the-one-that-moves-the-boundary) — and
pointing it elsewhere changes the privacy story completely.

## What stays on your Mac by default

- **Inference.** Generation posts to `<base_url>/v1/chat/completions`, which by
  default is a `llama-server` process on `http://localhost:8081`, started from
  the binary shipped inside the wheel.
- **Your files, prompts and edits.** The agent reads and writes your working
  tree directly.
- **Session and event logs.** `<project>/.localcode/events.jsonl` is an
  append-only local audit trail of tool calls, turn boundaries, server
  lifecycle and nudges. It is never uploaded. Set `LOCALCODE_TELEMETRY=0` to
  turn off the UI turn-trace records inside it.

There is no analytics endpoint, no usage reporting, and no version check.

## Inference endpoint: the one that moves the boundary

`runtime.base_url` in `~/.localcode/config.toml`, or the `LOCALCODE_BASE_URL`
environment variable, sets where chat completions are sent. **It accepts any
URL and is not validated or restricted to localhost.**

If you point it at a remote server, then every prompt localcode builds — your
message, the file contents it gathered as context, tool results, and the
model's replies — is sent to that server over the network. Nothing in
localcode's UI marks this as a different mode; the only signal is the value you
set.

This is a supported thing to do (it is how you drive a llama-server on another
machine on your LAN), but it is outside the local-only claim. If local-only is
the reason you are here, leave `base_url` alone and check it with `/status`.

## Known outbound paths in the current implementation

This inventory was compiled by reading the source, and it is the set the
maintainers of this page know about. It is **not proven complete**: an optional
dependency, a newly added tool, or a library making its own request can
introduce a path that is not listed here. If local-only matters to you, verify
at the network layer as well.

| # | What | Where it goes | When |
| --- | --- | --- | --- |
| 1 | **Connectivity probe** | TCP connect to `1.1.1.1:443` | Automatic. Once per turn at most — the verdict is cached for 30 s. No application payload is sent; the connection is opened and closed, and reaching it tells localcode whether to let the model attempt downloads |
| 2 | **Model download** | `huggingface.co` | First launch, and whenever you pick a model you don't have |
| 3 | **Quant browsing** | Hugging Face repo tree API | Only while you browse other quantisations in the model picker; results are cached |
| 4 | **`llama-server` fallback binary** | `github.com/mjwsolo/localcode` Releases | Only if the bundled binary can't be used and a prebuilt one is fetched instead. Refused outright if TLS certificate verification fails — localcode will not download an executable over an unverified connection |
| 5 | **Experimental runner sources** | `git clone --depth 1 https://github.com/ggml-org/llama.cpp`, plus a specific PR ref | Only if you pick an experimental model whose architecture the bundled server lacks. localcode then builds that runner from source on your machine |
| 6 | **Voice model** (optional `voice` extra) | `huggingface.co/ggerganov/whisper.cpp` | First time you enable voice input |
| 7 | **Voice output voices** (optional `voice` extra) | `huggingface.co/rhasspy/piper-voices` | First time a speech voice is used |
| 8 | **`web_search` tool** | DuckDuckGo, via the `ddgs` package | Whenever the model calls it — **never prompts** |
| 9 | **`web_fetch` tool** | The URL named in the call | Whenever the model calls it — **never prompts** |
| 10 | **Skill install from a URL** | That URL | Only when you install one that way |
| 11 | **MCP servers** | Wherever you pointed them | Whenever the model calls one of their tools — **never prompts** |
| 12 | **Shell commands** | Wherever the command goes | Whenever a `bash` or `background_process` call runs — see the approval note below |
| 13 | **Custom inference endpoint** | Whatever `base_url` names | Every turn, if you changed it. Carries your prompts and code context |
| 14 | **Semantic-index embedding model** | Hugging Face | The first time the code-search index is built, if `sentence-transformers` is installed. See below |

The `search` config section has keys for Google, Brave and SerpAPI. The
`web_search` tool the model actually calls does **not** read them — it always
queries DuckDuckGo through the `ddgs` package. Setting a key changes nothing
about where a search goes.

## The semantic index can pull a model in the background

localcode's code search has two legs: a lexical one, and a semantic one backed
by an embedding index. When a turn needs search context and no index exists
yet, localcode kicks off a **one-time background build** — on a daemon thread,
with no prompt and no progress in the chat.

If `sentence-transformers` is importable, that build loads an embedding model,
downloading it from Hugging Face on first use:

- **`all-MiniLM-L6-v2`** (~22M params) is the default.
- **`nomic-ai/nomic-embed-text-v1.5`** (~137M params) is used *only* when you
  set `LOCALCODE_TRUST_REMOTE_CODE=1`.

That environment variable is not a formality. `nomic-embed-text-v1.5` requires
`trust_remote_code=True`, which means sentence-transformers **downloads and
executes Python from the model repository** on your machine. localcode defaults
away from it for exactly that reason. Do not set it unless you have decided to
trust that repository as executable code.

`sentence-transformers` is not one of localcode's declared dependencies. If it
is not installed, the index falls back to a local scikit-learn TF-IDF build and
nothing is downloaded.

## Approvals: the network tools never ask

This is the part most likely to surprise you.

**`web_search`, `web_fetch` and every MCP tool are dispatched without a
confirmation prompt, at every autonomy level — including `suggest`.** The
confirmation gate only considers shell-executing tools (`bash`,
`background_process`) and file-write tools; anything else is passed straight
through. So "read-only mode" does not mean "no network".

For shell commands, which is where confirmation does apply:

- **`suggest`** — every shell command is confirmed, and so is every file write.
- **`auto_edit`** (the interactive default) — file writes go through without
  asking. A `bash` command is confirmed when it matches the risky-shell pattern
  list (piping a download into a shell, a force-push, `sudo rm`,
  `git reset --hard origin`) **or** the destructive substring list, which is
  broad: it includes `pip install`, `npm install`, `brew install`, `npm run`,
  `python `, `node `, `git push`, `sudo `, `rm -r`, `docker rm` and
  `kubectl delete`. A bare `curl https://…` is on neither list and runs
  unprompted; `curl … | sh` is caught.
- **`background_process`** is confirmed at every level except `full_auto`,
  whatever the command — unless its leading token was already approved for the
  session, which is checked before the always-confirm rule and bypasses it.
- **`full_auto`** — nothing is confirmed. The autonomy-independent hard blocks
  still apply.

**`localcode run` (headless) forces `full_auto`.** A headless run can make
network requests without any confirmation at all. Run it against repositories
and networks where that is acceptable.

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
inference all work without a connection. What breaks is the set of paths listed
above — as far as that list goes. The connectivity probe simply fails, and the model is told the machine
is offline so it stops attempting downloads. See
[Offline](/localcode/guides/offline).

## Verifying it yourself

localcode is Apache-2.0, so the boundary is auditable. The outbound call sites
are `app.py` (`is_online`, the probe), `bootstrap.py` (model downloads, the
release binary, the `git clone` for experimental runners), `hf_quants.py` (the
repo tree API), `voice.py` (speech assets), `tools/web_search.py` and
`tools/web_fetch.py`, `skills.py` (install from URL), and the MCP client.

`runtime.py`, `launcher.py` and `server_manager.py` address whatever
`config.runtime.base_url` holds. They are not pinned to loopback, so they are
local exactly as long as that setting is.

`embeddings.py` is the semantic-index path described above.

You can also watch what happened after the fact:

```sh
tail -f .localcode/events.jsonl | jq .
```
