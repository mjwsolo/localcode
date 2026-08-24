---
title: Network Boundary
description: The outbound paths known in the current implementation, what triggers them, and what stays on your Mac by default.
---

**With the default supported setup, inference runs on your Mac. It needs no API
key or model provider, and it has no remote fallback.** Some other features do
use the network. One setting can also move inference off your Mac. This page
lists the network access known in the current code. It does not simply call
localcode "offline."

## Scope: the default setup

This page assumes that `runtime.base_url` uses its default value:
`http://localhost:8081`. You can change this value. See
[Inference endpoint](#inference-endpoint-the-one-that-moves-the-boundary).
Pointing it somewhere else completely changes the privacy boundary.

## What stays on your Mac by default

- **Inference.** Generation sends requests to
  `<base_url>/v1/chat/completions`. By default, this is a `llama-server`
  process at `http://localhost:8081`. It uses the binary included in the wheel.
- **Your files, prompts and edits.** The agent reads and writes your working
  tree directly.
- **Session and event logs.** `<project>/.localcode/events.jsonl` is a local,
  append-only record. It logs tool calls, turn boundaries, server lifecycle
  events and nudges. It is never uploaded. Set `LOCALCODE_TELEMETRY=0` to
  disable the UI turn-trace records inside it.

There is no analytics endpoint, usage reporting or version check.

## Inference endpoint: the one that moves the boundary

`runtime.base_url` in `~/.localcode/config.toml`, or the `LOCALCODE_BASE_URL`
environment variable, controls where chat completions are sent. **It accepts
any URL. It is not checked or limited to localhost.**

If you use a remote server, localcode sends every prompt it builds to that
server over the network. This includes your message, file contents collected
for context, tool results and the model's replies. The localcode UI does not
show that you are using a different mode. The only sign is the value you set.

This setup is supported. For example, you can use it to run a llama-server on
another machine on your LAN. But it is not local-only. If you need local-only
operation, do not change `base_url`. Check its value with `/status`.

## Known outbound paths in the current implementation

This list comes from reading the source. It includes every path known to the
maintainers of this page. It is **not proven complete**. An optional dependency,
a new tool or a library could make a request that is not listed here. If
local-only operation matters to you, also check traffic at the network layer.

| # | What | Where it goes | When |
| --- | --- | --- | --- |
| 1 | **Connectivity probe** | TCP connect to `1.1.1.1:443` | Automatic. At most once per turn because the result is cached for 30 s. It sends no application payload. It opens and closes the connection. If the connection works, localcode lets the model try downloads |
| 2 | **Model download** | `huggingface.co` | On first launch and whenever you choose a model you do not have |
| 3 | **Quant browsing** | Hugging Face repo tree API | Only when you browse other quantisations in the model picker. Results are cached |
| 4 | **`llama-server` fallback binary** | `github.com/mjwsolo/localcode` Releases | Only when the included binary cannot be used and localcode downloads a prebuilt one. If TLS certificate verification fails, localcode refuses the download. It will not download an executable over an unverified connection |
| 5 | **Experimental runner sources** | `git clone --depth 1 https://github.com/ggml-org/llama.cpp`, plus a specific PR ref | Only when you choose an experimental model with an architecture that the included server does not support. localcode then builds that runner from source on your machine |
| 6 | **Voice model** (optional `voice` extra) | `huggingface.co/ggerganov/whisper.cpp` | The first time you enable voice input |
| 7 | **Voice output voices** (optional `voice` extra) | `huggingface.co/rhasspy/piper-voices` | The first time a speech voice is used |
| 8 | **`web_search` tool** | DuckDuckGo, via the `ddgs` package | Whenever the model calls it - **never prompts** |
| 9 | **`web_fetch` tool** | The URL named in the call | Whenever the model calls it - **never prompts** |
| 10 | **Skill install from a URL** | That URL | Only when you install one that way |
| 11 | **MCP servers** | Wherever you pointed them | Whenever the model calls one of their tools - **never prompts** |
| 12 | **Shell commands** | Wherever the command goes | Whenever a `bash` or `background_process` call runs - see the approval note below |
| 13 | **Custom inference endpoint** | Whatever `base_url` names | Every turn if you changed it. It carries your prompts and code context |
| 14 | **Semantic-index embedding model** | Hugging Face | The first time the code-search index is built, if `sentence-transformers` is installed. See below |

The `search` config section has keys for Google, Brave and SerpAPI. The
`web_search` tool does **not** use these keys. It always searches DuckDuckGo
through the `ddgs` package. Setting a key does not change where searches go.

## The semantic index can download a model in the background

localcode code search has two parts. One uses lexical search. The other uses a
semantic embedding index. When a turn needs search context and no index exists,
localcode starts a **one-time background build**. It runs on a daemon thread,
without a prompt or chat progress message.

If `sentence-transformers` can be imported, the build loads an embedding model.
It downloads the model from Hugging Face the first time it is used:

- **`all-MiniLM-L6-v2`** (~22M params) is the default.
- **`nomic-ai/nomic-embed-text-v1.5`** (~137M params) is used *only* when you
  set `LOCALCODE_TRUST_REMOTE_CODE=1`.

This environment variable has a serious effect.
`nomic-embed-text-v1.5` needs `trust_remote_code=True`. This means
sentence-transformers **downloads and runs Python from the model repository**
on your machine. localcode does not use it by default for this reason. Set the
variable only if you trust that repository as executable code.

`sentence-transformers` is not a declared localcode dependency. If it is not
installed, the index uses a local scikit-learn TF-IDF build instead. It
downloads nothing.

## Approvals: the network tools never ask

This behavior may surprise you.

**`web_search`, `web_fetch` and all MCP tools run without a confirmation
prompt at every autonomy level, including `suggest`.** The confirmation check
only covers shell tools (`bash`, `background_process`) and file-writing tools.
All other tools run directly. So "read-only mode" does not mean "no network."

For shell commands, confirmation works as follows:

- **`suggest`** - every shell command and every file write needs confirmation.
- **`auto_edit`** (the interactive default) - file writes do not ask for
  confirmation. A `bash` command asks for confirmation when it matches the
  risky-shell pattern list, such as piping a download into a shell, a
  force-push, `sudo rm` or `git reset --hard origin`. It also asks when the
  command matches the broad destructive substring list. This list includes
  `pip install`, `npm install`, `brew install`, `npm run`, `python `, `node `,
  `git push`, `sudo `, `rm -r`, `docker rm` and `kubectl delete`. A plain
  `curl https://…` matches neither list and runs without asking. `curl … | sh`
  is caught.
- **`background_process`** asks for confirmation at every level except
  `full_auto`, regardless of the command. There is one exception: if its first
  token was already approved for the session, that check happens before the
  always-confirm rule and bypasses it.
- **`full_auto`** - nothing asks for confirmation. The hard blocks that apply
  at every autonomy level still apply.

**`localcode run` (headless) forces `full_auto`.** A headless run can make
network requests without any confirmation. Use it only with repositories and
networks where this is acceptable.

## The parts to think about carefully

**`web_search` and `web_fetch` send content off your machine.** The model writes
search queries from your conversation. Treat every use of these tools as
sending that text to a third party.

**MCP servers are arbitrary programs.** A remote MCP server sees every argument
the model sends to its tools. Only use servers you trust. See
[MCP](/localcode/guides/mcp).

**Building an experimental runner** downloads and compiles third-party source
on your machine. This requires more trust than downloading a model file.

## Running with no network at all

After downloading a model, the agent loop, file tools, shell tools and
inference all work without a connection. The paths listed above stop working,
as far as this list covers them. The connectivity probe simply fails. The model
is then told that the machine is offline, so it stops trying to download files.
See [Offline](/localcode/guides/offline).

## Verifying it yourself

localcode uses Apache-2.0, so you can inspect its boundaries. The outbound call
sites are `app.py` (`is_online`, the probe), `bootstrap.py` (model downloads,
the release binary and the `git clone` for experimental runners),
`hf_quants.py` (the repo tree API), `voice.py` (speech assets),
`tools/web_search.py`, `tools/web_fetch.py`, `skills.py` (install from URL) and
the MCP client.

`runtime.py`, `launcher.py` and `server_manager.py` use the address stored in
`config.runtime.base_url`. They are not limited to loopback. They stay local
only while that setting points to a local address.

`embeddings.py` contains the semantic-index path described above.

You can also watch recorded activity afterward:

```sh
tail -f .localcode/events.jsonl | jq .
```
