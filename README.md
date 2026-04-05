# Gem

Gem is a local-first AI coding assistant for developers who want an open-source, terminal-native workflow with Gemma 4 running on their own machine.

This repo currently contains two things:

- `src/gem/`: the new Python implementation for a pragmatic v1.
- `gem_code/`: the legacy TypeScript/Ink codebase snapshot that was inspected during the migration.

## Why The Refactor

The legacy app is a large TypeScript terminal product with deep coupling to hosted APIs, remote control, analytics, MCP plumbing, and product-specific services. The parts worth preserving conceptually are:

- interactive terminal workflow
- streaming assistant output
- background-safe shell/task execution
- session persistence
- repo-aware context gathering
- diff-oriented coding flow

The new Python version keeps those ideas and drops the product-specific sprawl.

## Interface Choice

A TUI is the right v1 interface for Gem because the target user is already living in a terminal while coding. It keeps file paths, git, diffs, shell output, and model interaction in one place without introducing browser or hosted-service requirements.

Practical implication: this version is intentionally terminal-first, keyboard-driven, and local. It does not attempt to recreate every feature from the legacy app.

## Runtime Choice

Gem targets **Ollama** first for local Gemma 4 because it is the simplest practical setup:

- one local HTTP endpoint
- streaming responses
- simple install story
- no hosted dependency

Gem now also has:

- an experimental `llama_cpp` provider path for local server setups that expose an OpenAI-style `/v1/chat/completions` endpoint. This is the preferred direction for aggressive low-latency tuning on constrained hardware.
- an `mlx-local` provider path for Apple Silicon users running MLX quantized Gemma models locally.
- an advanced `huggingface-local` provider path for users who want to run Gemma checkpoints directly through local `transformers` + `torch`.
- a local browser preset through Playwright MCP.
- a local voice stack with `whisper.cpp` or `faster-whisper` for STT and `kokoro` or `piper` for TTS.

Recommended stack:

- default onboarding: `ollama`
- Mac performance path: `mlx-local`
- cross-platform performance path: `llama_cpp`
- advanced custom local backend: `huggingface-local`

## Gemma 4 Design

Gem is now explicitly centered on Gemma 4.

Official Gemma docs currently state:

- Gemma 4 released on **March 31, 2026**
- Gemma 4 ships in **E2B, E4B, 31B, and 26B A4B / MoE-style** variants
- Gemma 4 supports **text, image, and audio input**
- Gemma 4 supports context windows up to **256K**

Practical implications for Gem:

- setup starts with Gemma 4 profile selection
- prompts and context budgets vary by selected Gemma 4 tier
- tool use is exposed as a local-first coding workflow, not as cloud orchestration
- the assistant stays useful on both small and large local hardware

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Fast Start

Target flow:

```bash
gem setup --install --benchmark
gem
```

`gem setup --install` is now provider-aware:

- `ollama` installs or checks Ollama and can pull the selected model
- `mlx-local` installs `mlx-lm` and then attempts to resolve/download the selected MLX model
- `huggingface-local` installs `transformers`, `torch`, and `accelerate`, then attempts to resolve/download the selected model
- `llama_cpp` now attempts a best-effort local install where supported and then tells the user the exact server command to run

`gem doctor` is also provider-aware and now checks:

- selected backend dependencies
- selected model id fields like `mlx_model_id` or `huggingface_model_id`
- `llama_cpp` server readiness hints

Gem can also print or start supported local runtimes directly:

```bash
gem runtime-cmd
gem runtime-up
```

During `gem setup`, Gem now shows the Gemma profile options with short explanations, for example:

- `e2b` `[faster for small machines]`
- `e4b` `[balanced default for most laptops]`
- `26b-moe` `[stronger coding on bigger local rigs]`
- `31b` `[best quality for large local workstations]`

If you do not want Gem to attempt local installation:

```bash
gem setup
gem
```

## Local Gemma 4 Setup

Google announced Gemma 4 on March 31, 2026 with E2B, E4B, 26B MoE, and 31B variants. Gem treats those as first-class profiles so prompts and context strategy can adapt to the selected model tier.

Install and run Ollama, then pull or create your Gemma 4 model locally. Exact tags vary by local setup, so Gem exposes both a profile and a model tag.

Default config path:

```text
~/.gem/config.toml
```

Create a starter config:

```bash
gem config-init
```

Example:

```toml
[runtime]
provider = "ollama"
base_url = "http://localhost:11434"
profile = "e4b"
model = "gemma4:e4b"
mode = "balanced"
planner_model = "gemma4:e2b"
draft_model = "gemma4:e2b"
planner_enabled = true
adaptive_execution = true
escalation_enabled = true
huggingface_model_id = ""
huggingface_device = "auto"
huggingface_dtype = "auto"
mlx_model_id = ""
temperature = 0.2
max_context_chars = 40000
request_timeout_seconds = 120
max_retries = 2

[search]
provider = "duckduckgo"
google_api_key = ""
google_cx = ""
brave_api_key = ""
serpapi_api_key = ""

[browser]
enabled = true
mcp_server_name = "browser"
launch_command = "npx"
launch_args = ["-y", "@playwright/mcp@latest"]

[voice]
stt_provider = "whisper.cpp"
tts_provider = "kokoro"
whisper_model_path = ""
faster_whisper_model = "small"
kokoro_voice = "af_heart"
piper_model_path = ""

[ui]
show_debug = false
thinking_mode = "summary"
```

Advanced Hugging Face local example:

```toml
[runtime]
provider = "huggingface-local"
profile = "e2b"
model = ""
mode = "fast"
huggingface_model_id = "google/gemma-2-2b-it"
huggingface_device = "auto"
huggingface_dtype = "bfloat16"
```

For the Hugging Face local backend, install the additional local runtime yourself:

```bash
pip install transformers torch accelerate
```

Advanced MLX local example for Apple Silicon:

```toml
[runtime]
provider = "mlx-local"
profile = "31b"
model = ""
mode = "fast"
mlx_model_id = "mlx-community/gemma-4-31b-it-4bit"
quant_preset = "fastest"
```

Install the MLX local runtime:

```bash
pip install -U mlx-lm
```

## Usage

Start a new session in a repo:

```bash
gem
```

Start with a bigger profile:

```bash
gem --profile 31b
```

Start with an explicit local tag:

```bash
gem --model gemma4:26b-moe
```

Resume a prior session:

```bash
gem resume <session-id>
```

Resume the latest session for the current repo:

```bash
gem resume --latest
```

Run health checks:

```bash
gem doctor
```

List profiles and installed local models:

```bash
gem models
```

Run a one-shot non-interactive prompt:

```bash
gem exec "summarize the current diff"
```

Show or update settings:

```bash
gem settings show
gem mode fast
gem quant fastest
gem benchmark
gem recommend-model "mlx-community/gemma-4-31b-it-4bit"
gem recommend-model "gemma4-27b-it-Q4_K_M.gguf"
gem settings set search.provider google
gem settings set search.google_api_key YOUR_KEY
gem settings set search.google_cx YOUR_CX
gem browser-setup
gem voice-status
gem export-traces gem_traces.jsonl
```

Build and query the local code index:

```bash
gem index build
gem index search "auth token refresh"
```

Run task-completion benchmarks:

```bash
gem benchmark --tasks benchmarks/tasks.json
```

Register an MCP stdio server:

```bash
gem mcp-add filesystem npx -y @modelcontextprotocol/server-filesystem .
gem mcp-list
```

## TUI Commands

Inside Gem:

- `/help` shows commands
- `/status` shows runtime, repo, and session state
- `/model` shows profiles and lets you switch profile or model tag
- `/skills` lists disk-loaded skills
- `/tools` lists available local, plugin, web, and MCP tools
- `/mcp` lists configured MCP servers
- `/thinking [hidden|summary|full]` controls reasoning display
- `/search <query>` runs the configured web search provider
- `/browser status` shows Playwright MCP browser readiness
- `/browser setup` writes the Playwright MCP preset into Gem config
- `/browser open <url>` opens a page through the browser MCP server
- `/browser snapshot` requests a browser accessibility snapshot
- `/voice status` shows local voice readiness
- `/voice say <text>` writes local speech audio with the configured TTS backend
- `/voice transcribe <file>` transcribes a local audio file
- `/ignite` points the user to guided setup
- `/permissions` shows repo-scoped allow and deny rules
- `/timeline` shows recent agent, tool, verify, and model events
- `/verify [cmd]` runs the current verification plan or an explicit command
- `/agent <task>` runs the multi-step agent loop in the foreground
- `/agentbg <task>` runs the multi-step agent loop in the background
- `/files [pattern]` lists repo files
- `/find <query>` searches the local code index
- `/index` builds the local code index
- `/read <path>` previews a file
- `/add <path>` pins a file into model context
- `/drop <path>` removes a pinned file
- `/context` shows pinned files
- `/shell <command>` runs a shell command with visible output
- `/bg <command>` starts a visible background job
- `/jobs` lists background jobs
- `/log <jobid>` shows a background job log
- `/diff` shows current git diff
- `/apply` applies the last assistant diff block with `git apply`
- `/clear` clears conversation history for the current session
- `/quit` exits

## Current Features

Gem currently includes:

- interactive terminal chat
- streaming local model output
- Gemma 4 profile-aware prompting and context budgets
- session persistence and repo-scoped resume
- repo file browsing and pinned file context
- auto-compact of long conversations
- one-shot non-interactive `gem exec`
- first-class settings management with `gem settings`
- local code indexing and search for large repos
- visible shell execution
- background jobs with logs
- safer file/hunk-aware patch review before `git apply`
- iterative agent loop with explicit step, verify, and repair phases
- benchmark-driven performance presets for `fast`, `balanced`, and `deep` modes
- quant preset UX with `smallest`, `fastest`, `balanced`, and `best`
- cache/session policies with rolling message windows
- planner lane, progressive model escalation, and adaptive context policy
- draft-model assist path for faster first-pass local reasoning
- repo cartridges that compress repo state into a smaller reusable context block
- repo-scoped and session-scoped approval memory
- provider-based web search via DuckDuckGo, Google Custom Search, Brave, or SerpAPI
- local browser automation via Playwright MCP
- local voice support with `whisper.cpp` or `faster-whisper` plus `kokoro` or `piper`
- model-driven tool calling for loaded tools
- skills loaded from `~/.gem/skills` and `.gem/skills`
- plugin loading from `~/.gem/plugins` and `.gem/plugins`
- stdio MCP server registration and tool discovery
- runtime retries and request timeouts for flaky local inference
- experimental multi-runtime support: `ollama`, `mlx-local`, `llama_cpp`, and `huggingface-local`
- local trace export for future Gem-specific drafter training

## Performance Direction

Gem is now moving toward a layered local-performance architecture:

- `fast` mode keeps context tight, uses planner-first behavior, and prefers smaller Gemma variants
- `balanced` mode is the default for general coding work
- `deep` mode allows broader context and stronger verification on larger machines
- `gem benchmark` inspects CPU, RAM, and available GPU signals and recommends a preset
- `gem setup --benchmark` applies that preset during onboarding

The next performance layers are intentionally staged:

1. benchmark + mode presets
2. tighter KV/cache policy and aggressive compaction
3. planner/router lane with smaller Gemma variants
4. `llama.cpp` performance backend usage
5. speculative decoding / draft-model support
6. Gem-specific drafter distillation and tuning

## Skills, Plugins, And MCP

Skills:

- Add markdown files under `~/.gem/skills/` or `.gem/skills/`
- Reference them in chat with `@skillname` or `#skillname`

Plugins:

- Add Python files under `~/.gem/plugins/` or `.gem/plugins/`
- Each plugin can expose a `register(registry)` function and add tool builders

MCP:

- Gem currently supports **stdio MCP servers**
- Add them with `gem mcp-add`
- Once configured, their tools are exposed inside Gem as `mcp__server__tool`

## Design Notes

Gem v1 is intentionally simple:

- no remote web control
- no hosted auth
- no analytics
- no opaque framework stack

That keeps the app contributor-friendly and realistic for an open-source project serving a large user base.

## Verification

After install:

```bash
gem doctor
python -m compileall src
```

## Tradeoffs

- The current TUI is prompt-driven rather than a multi-pane fullscreen interface.
- Tool calling depends on the local runtime and model behaving well with Ollama tool schemas.
- MCP support is intentionally narrow and currently limited to stdio servers.
- Plugin and skill APIs are intentionally small and may change.
- Ollama is the only implemented runtime today.
- The legacy `gem_code/` tree is preserved as reference material, not an active runtime.
- `gem setup --install` can automate first-run prep on supported systems, but cross-platform installation still depends on local package manager availability.

## Sources

- Google Gemma releases: https://ai.google.dev/gemma/docs/releases
- Google Gemma personal code assistant guide: https://ai.google.dev/gemma/docs/personal-code-assistant
- Google Gemma docs overview: https://ai.google.dev/gemma/docs
- MCP transports spec: https://modelcontextprotocol.io/specification/2025-03-26/basic/transports
- MCP tools concept: https://modelcontextprotocol.io/legacy/concepts/tools
- Ollama model library: https://ollama.com/library
