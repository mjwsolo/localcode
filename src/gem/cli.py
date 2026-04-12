from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import run_agent_loop
from .app import GemApp
from .browser import browser_status, ensure_browser_mcp
from .bootstrap import pull_model, run_setup
from .exec_mode import run_exec
from .config import get_config_path, init_config_file, load_config
from .indexer import build_index, search_index
from .logging_utils import configure_logging
from .mcp import add_mcp_config, load_mcp_configs
from .models import GEMMA_PROFILES, get_runtime_model, resolve_profile
from .performance import apply_preset, benchmark_report
from .runtime import GemRuntimeGateway
from .session import SessionStore
from .settings import set_setting, show_settings
from .skills import (
    ensure_builtin_skills,
    install_skill,
    list_skills,
    remove_skill,
    search_skills,
    skill_info,
)
from .toolkit import GemToolkit
from .verification import run_verification


def is_online() -> bool:
    """Stub: assume online after network module removal."""
    return True


def provider_readiness(runtime_config) -> tuple[bool, list[str]]:
    """Stub: assume provider is ready after provider_checks module removal."""
    return True, []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localcode", description="LocalCode — AI coding assistant running entirely on your machine")
    parser.add_argument("--profile", help="Gemma 4 profile: e2b, e4b, 26b-laptop, 26b-moe, 31b")
    parser.add_argument("--model", help="Explicit local runtime model tag")
    parser.add_argument("--cli", action="store_true", help="Use the classic CLI instead of the TUI")
    parser.add_argument("-c", "--cwd", type=str, default=None,
                        help="Working directory for the project (defaults to current directory)")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("config-init", help="create a default config file")
    setup = subparsers.add_parser("setup", help="prepare local runtime and Ollama")
    setup.add_argument("--install", action="store_true", help="attempt to install Ollama and pull the selected model")
    setup.add_argument("--benchmark", action="store_true", help="benchmark the local machine and apply a recommended preset")
    benchmark = subparsers.add_parser("benchmark", help="inspect the local machine and recommend a Gem performance preset")
    benchmark.add_argument("--mode", choices=["fast", "balanced", "deep"])
    benchmark.add_argument("--tasks", help="Run task-completion benchmarks from a JSON task file")
    benchmark.add_argument("--compare-26b-laptop", action="store_true", help="Compare 26B laptop runtime modes (speed vs fit) on benchmark tasks")
    benchmark.add_argument("--compare-models", nargs="+", help="Compare explicit local model tags on benchmark tasks")
    benchmark.add_argument("--compare-26b-quants", action="store_true", help="Compare the installed 26B quant candidate set on benchmark tasks")
    benchmark.add_argument("--compare-gguf-dir", help="Compare all candidate GGUF files present in this directory via llama.cpp")
    benchmark.add_argument("--install-missing", action="store_true", help="When comparing 26B quants, install any missing candidate tags before running")
    benchmark.add_argument("--provider", choices=["ollama", "llama_cpp", "mlx-local", "huggingface-local"], help="Provider to use for explicit model comparison")
    benchmark.add_argument("--repeats", type=int, default=1, help="Repeat each benchmark task this many times")
    subparsers.add_parser("status", help="show runtime status and configuration")
    subparsers.add_parser("doctor", help="(alias for status)")
    subparsers.add_parser("browser-setup", help="install the Playwright MCP browser preset into LocalCode config")
    subparsers.add_parser("voice-status", help="show local voice subsystem readiness")
    subparsers.add_parser("runtime-cmd", help="print the local runtime launch command for the selected provider")
    subparsers.add_parser("runtime-up", help="start the selected local runtime in the background when supported")
    subparsers.add_parser("models", help="list Gemma profiles and installed local models")
    mode = subparsers.add_parser("mode", help="show or change the active Gem performance mode")
    mode.add_argument("value", nargs="?", choices=["fast", "balanced", "deep"])
    quant = subparsers.add_parser("quant", help="show or change the active quant preset")
    quant.add_argument("value", nargs="?", choices=["smallest", "fastest", "balanced", "best"])
    recommend = subparsers.add_parser("recommend-model", help="Recommend backend and quant policy for an exact local model tag")
    recommend.add_argument("model_tag")
    exec_parser = subparsers.add_parser("exec", help="run a one-shot non-interactive prompt")
    exec_parser.add_argument("prompt")
    agent_parser = subparsers.add_parser("agent", help="run an agentic coding task with verify/retry")
    agent_parser.add_argument("prompt")
    agent_bg = subparsers.add_parser("agent-bg", help="run an agentic coding task in the background")
    agent_bg.add_argument("prompt")
    hidden_agent = subparsers.add_parser("agent-runner", help=argparse.SUPPRESS)
    hidden_agent.add_argument("prompt")
    orch_parser = subparsers.add_parser("orch", help="multi-agent orchestrated task (planner→workers→reviewer)")
    orch_parser.add_argument("prompt")

    resume = subparsers.add_parser("resume", help="resume a saved session")
    resume.add_argument("session_id", nargs="?")
    resume.add_argument("--latest", action="store_true", help="resume the latest session for the current repo")

    subparsers.add_parser("sessions", help="list saved sessions")
    verify_parser = subparsers.add_parser("verify", help="run the repo verification command")
    verify_parser.add_argument("verify_command", nargs="?")
    export = subparsers.add_parser("export-traces", help="export local sessions as jsonl traces")
    export.add_argument("output", nargs="?", default="gem_traces.jsonl")
    settings = subparsers.add_parser("settings", help="show or update Gem settings")
    settings_sub = settings.add_subparsers(dest="settings_command")
    settings_sub.add_parser("show", help="show current settings")
    settings_set = settings_sub.add_parser("set", help="set a setting value")
    settings_set.add_argument("key")
    settings_set.add_argument("value")
    index = subparsers.add_parser("index", help="build or query the local code index")
    index_sub = index.add_subparsers(dest="index_command")
    index_sub.add_parser("build", help="build the local code index")
    index_search = index_sub.add_parser("search", help="search the local code index")
    index_search.add_argument("query")
    mcp_add = subparsers.add_parser("mcp-add", help="add a stdio MCP server")
    mcp_add.add_argument("name")
    mcp_add.add_argument("server_command")
    mcp_add.add_argument("args", nargs="*")
    subparsers.add_parser("mcp-list", help="list configured MCP servers")

    # -- Speed --
    speed = subparsers.add_parser("speed", help="speed optimization tools")
    speed_sub = speed.add_subparsers(dest="speed_command")
    speed_sub.add_parser("status", help="show current speed optimizations")
    speed_launch = speed_sub.add_parser("launch", help="print optimal llama-server launch command")
    speed_launch.add_argument("model_path", help="path to GGUF model file")
    speed_launch.add_argument("--draft", help="path to draft model GGUF for speculative decoding")
    speed_sub.add_parser("benchmark", help="benchmark current model speed")

    # -- Daemon --
    daemon = subparsers.add_parser("daemon", help="manage the LocalCode background daemon")
    daemon_sub = daemon.add_subparsers(dest="daemon_command")
    daemon_sub.add_parser("start", help="start the daemon (keeps model hot)")
    daemon_sub.add_parser("stop", help="stop the daemon")
    daemon_sub.add_parser("status", help="show daemon status")
    daemon_sub.add_parser("log", help="show daemon log")

    # -- Claw mode --
    claw = subparsers.add_parser("claw", help="submit a task to the daemon (OpenClaw-style)")
    claw.add_argument("prompt", nargs="?")
    claw.add_argument("--status", action="store_true", help="show status of recent claw tasks")
    claw.add_argument("--result", help="get the result of a specific task by ID")
    claw.add_argument("--list", action="store_true", help="list recent claw tasks")

    # -- Skills --
    skill = subparsers.add_parser("skill", help="manage Gem skills")
    skill_sub = skill.add_subparsers(dest="skill_command")
    skill_sub.add_parser("list", help="list installed skills")
    skill_install = skill_sub.add_parser("install", help="install a skill from file, directory, or URL")
    skill_install.add_argument("source")
    skill_remove = skill_sub.add_parser("remove", help="remove an installed skill")
    skill_remove.add_argument("name")
    skill_search = skill_sub.add_parser("search", help="search installed skills")
    skill_search.add_argument("query")
    skill_info_parser = skill_sub.add_parser("info", help="show info about a skill")
    skill_info_parser.add_argument("name")
    skill_sub.add_parser("init", help="install built-in starter skills")

    return parser


def _should_autobootstrap(config) -> bool:
    """Check if first-time setup is needed (server not reachable)."""
    runtime_ok, _ = GemRuntimeGateway(config.runtime).healthcheck()
    return not runtime_ok


def run_status() -> int:
    config = load_config()
    profile = resolve_profile(config.runtime.profile, config.runtime.model)
    config.runtime.model = get_runtime_model(profile, config.runtime.model)
    gateway = GemRuntimeGateway(config.runtime)
    toolkit = GemToolkit(Path.cwd(), config)
    ok, details = gateway.healthcheck()
    console = Console()

    # ── Runtime ──
    runtime_table = Table(show_header=False, title="Runtime", title_style="bold green")
    online = is_online()
    runtime_table.add_row("status", "[green]ok[/green]" if ok else "[red]unreachable[/red]")
    runtime_table.add_row("network", "[green]online[/green]" if online else "[yellow]offline[/yellow]")
    runtime_table.add_row("provider", config.runtime.provider)
    runtime_table.add_row("model", config.runtime.model)
    runtime_table.add_row("profile", profile.key)
    runtime_table.add_row("mode", config.runtime.mode)
    runtime_table.add_row("thinking", config.ui.thinking_mode)
    runtime_table.add_row("server", config.runtime.base_url)
    if not ok:
        runtime_table.add_row("error", details)
    console.print(runtime_table)

    # ── Performance ──
    perf_table = Table(show_header=False, title="Performance", title_style="bold green")
    if config.runtime.provider == "llama_cpp":
        perf_table.add_row("gpu_layers", str(config.runtime.llama_cpp_gpu_layers))
        perf_table.add_row("threads", str(config.runtime.llama_cpp_threads))
        perf_table.add_row("batch_size", str(config.runtime.llama_cpp_batch_size))
    perf_table.add_row("kv_cache", f"{config.runtime.kv_cache_type_k} / {config.runtime.kv_cache_type_v}")
    perf_table.add_row("context", config.runtime.cache_policy)
    perf_table.add_row("timeout", f"{config.runtime.request_timeout_seconds}s")
    console.print(perf_table)

    # ── Config ──
    config_table = Table(show_header=False, title="Config", title_style="bold green")
    config_table.add_row("file", str(get_config_path()))
    search_provider, search_status = toolkit.search_status()
    config_table.add_row("search", f"{search_provider} ({search_status})")
    mcp_count = len(load_mcp_configs())
    if mcp_count:
        config_table.add_row("mcp_servers", str(mcp_count))
    console.print(config_table)

    # ── Diagnostics (only if problems) ──
    provider_ok, provider_messages = provider_readiness(config.runtime)
    if provider_messages:
        console.print(Panel("\n".join(provider_messages), title="Issues", border_style="yellow"))
    diagnostics = toolkit.diagnostics()
    if diagnostics:
        console.print(Panel("\n".join(diagnostics), title="Diagnostics", border_style="yellow"))

    toolkit.close()
    return 0 if ok and provider_ok else 1


def main(argv: list[str] | None = None) -> None:
    import os
    import warnings
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MallocStackLogging"] = "0"
    os.environ["MallocStackLoggingNoCompact"] = "0"
    # Ollama performance optimizations (from mmap article)
    os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    # Keep one hot A4B model resident on small laptops unless the user overrides it.
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    # Set KV cache type from config (after config loads)
    os.environ["OLLAMA_KV_CACHE_TYPE"] = config.runtime.kv_cache_type_k
    configure_logging(config.ui.show_debug)
    console = Console()

    # Resolve project directory from -c/--cwd flag or current directory
    if args.cwd:
        project_dir = Path(args.cwd).resolve()
        if not project_dir.is_dir():
            console.print(f"[red]Error:[/] directory not found: {args.cwd}")
            sys.exit(1)
        os.chdir(project_dir)
    else:
        project_dir = Path.cwd()

    # TUI is the default UI. Use --cli for the classic terminal interface.
    if not getattr(args, 'cli', False) and args.command is None and not os.environ.get("LOCALCODE_CLI"):
        from .tui.app import LocalCodeTUI
        app = LocalCodeTUI(show_mode_picker=True)
        app.run()
        return

    if args.command == "config-init":
        path = init_config_file()
        console.print(f"Config ready at {path}")
        return
    if args.command == "setup":
        raise SystemExit(run_setup(config, args.profile, args.model, args.install, args.benchmark))
    if args.command == "benchmark":
        if args.tasks:
            console.print("Task benchmarks have been removed.")
            return
        machine, preset = benchmark_report(config, args.mode)
        table = Table(show_header=False)
        table.add_row("system", machine.system)
        table.add_row("cpu_cores", str(machine.cpu_cores))
        table.add_row("memory_gb", str(machine.memory_gb))
        table.add_row("gpu", machine.gpu_summary)
        table.add_row("tier", machine.tier)
        table.add_row("recommended_mode", preset.mode)
        table.add_row("runtime_provider", preset.runtime_provider)
        table.add_row("profile", preset.profile)
        table.add_row("max_context_chars", str(preset.max_context_chars))
        table.add_row("quant_preset", preset.quant_preset)
        table.add_row("cache_policy", preset.cache_policy)
        table.add_row("rolling_window_messages", str(preset.rolling_window_messages))
        table.add_row("llama_cpp_gpu_layers", str(preset.llama_cpp_gpu_layers))
        table.add_row("llama_cpp_threads", str(preset.llama_cpp_threads))
        table.add_row("llama_cpp_batch_size", str(preset.llama_cpp_batch_size))
        console.print(table)
        if preset.notes:
            console.print(Panel("\n".join(preset.notes), title="Recommendations"))
        return
    if args.command == "recommend-model":
        console.print("Model recommendation has been removed.")
        return
    if args.command in ("status", "doctor"):
        raise SystemExit(run_status())
    if args.command == "browser-setup":
        path = ensure_browser_mcp(config)
        console.print(f"Browser MCP preset saved at {path}")
        console.print("\n".join(browser_status(config)))
        return
    if args.command == "voice-status":
        return
    if args.command == "speed":
        if args.speed_command == "launch":
            gw = GemRuntimeGateway(config.runtime)
            cmd = gw.llama_server_command(args.model_path)
            if args.draft:
                # Override draft model
                config.runtime.llama_cpp_draft_model = args.draft
                gw2 = GemRuntimeGateway(config.runtime)
                cmd = gw2.llama_server_command(args.model_path)
            console.print(" ".join(cmd))
            return
        if args.speed_command == "benchmark":
            import time
            console.print("[bold]Speed Benchmark[/bold]\n")
            # Test Ollama
            try:
                import httpx
                start = time.time()
                resp = httpx.post(
                    f"{config.runtime.base_url}/api/generate",
                    json={"model": config.runtime.model or "gemma4:e4b", "prompt": "Write fizzbuzz in Python", "stream": False, "think": False, "options": {"num_ctx": 4096}},
                    timeout=120,
                )
                data = resp.json()
                elapsed = data.get("eval_duration", 0) / 1e9
                tokens = data.get("eval_count", 0)
                speed = tokens / elapsed if elapsed > 0 else 0
                console.print(f"  Ollama ({config.runtime.model or 'gemma4:e4b'}): {speed:.1f} tok/s ({tokens} tokens in {elapsed:.1f}s)")
            except Exception as e:
                console.print(f"  Ollama: not available ({e})")
            return
        # Default: show status
        console.print("[bold]Speed Optimizations[/bold]\n")
        table = Table("Setting", "Value", "Effect")
        mode = config.runtime.laptop_26b_runtime_mode
        mode_desc = {
            "turbo":   "GPU Turbo — 27 tok/s decode, 32K ctx, 85 tok/s prompt (sysctl required)",
            "speed":   "CPU mmap — 18.5 tok/s decode, 10K ctx",
            "context": "GPU Safe — 13 tok/s decode, 32K ctx (no sysctl needed)",
        }.get(mode, mode)
        table.add_row("Runtime mode", f"{mode} ({mode_desc})", "speed or context")
        table.add_row("KV cache K", config.runtime.kv_cache_type_k, "q8_0 = preserve key quality")
        table.add_row("KV cache V", config.runtime.kv_cache_type_v, "turbo4 = 3.8x compression, +0.23% PPL")
        table.add_row("Spec type", config.runtime.llama_cpp_spec_type or "(none)", "ngram-mod = 1.5-2x speedup")
        table.add_row("Draft max", str(config.runtime.llama_cpp_draft_max), "Max speculative tokens")
        table.add_row("Draft model", config.runtime.llama_cpp_draft_model or "(none)", "Small model for speculation")
        table.add_row("Lookup cache", str(config.runtime.llama_cpp_lookup_cache), "Prompt lookup = 2-4x on edits")
        table.add_row("Escalation", str(config.runtime.escalation_enabled), "Auto-switch fast↔smart model")
        console.print(table)
        return
    if args.command == "runtime-cmd":
        console.print("Runtime launch command has been removed. Use llama-server directly.")
        return
    if args.command == "runtime-up":
        console.print("Runtime launch has been removed. Start llama-server directly.")
        return
    if args.command == "mode":
        if not args.value:
            console.print(f"Current mode: {config.runtime.mode}")
            return
        _, preset = benchmark_report(config, args.value)
        apply_preset(config, preset, model=config.runtime.model)
        console.print(f"Updated mode to {args.value}")
        return
    if args.command == "quant":
        if not args.value:
            console.print(f"Current quant preset: {config.runtime.quant_preset}")
            return
        config.runtime.quant_preset = args.value
        from .config import save_config
        save_config(config)
        console.print(f"Updated quant preset to {args.value}")
        return
    if args.command == "models":
        table = Table("profile", "default_model", "variant", "context_window", "summary")
        for profile in GEMMA_PROFILES.values():
            table.add_row(
                profile.key.replace("gemma4-", ""),
                profile.default_model,
                profile.feature_variant,
                str(profile.advertised_context_window),
                profile.summary,
            )
        console.print(table)
        gateway = GemRuntimeGateway(config.runtime)
        try:
            installed = gateway.list_models()
            console.print("Installed Ollama models: " + (", ".join(installed) or "(none)"))
        except Exception as exc:
            console.print(f"Installed Ollama models unavailable: {exc}")
        return
    if args.command == "sessions":
        rows = SessionStore().list_sessions()
        table = Table("session_id", "created_at", "repo_root")
        for row in rows:
            table.add_row(*row)
        console.print(table)
        return
    if args.command == "settings":
        if args.settings_command in {None, "show"}:
            show_settings(config)
            return
        if args.settings_command == "set":
            console.print(set_setting(config, args.key, args.value))
            return
    if args.command == "index":
        if args.index_command == "build":
            count, path = build_index(Path.cwd())
            console.print(f"Built code index for {count} files at {path}")
            return
        if args.index_command == "search":
            results = search_index(Path.cwd(), args.query)
            table = Table("path", "chunk", "preview")
            for item in results:
                table.add_row(item["path"], item["chunk_id"], item["preview"])
            console.print(table)
            return
        console.print("Use `localcode index build` or `localcode index search <query>`.")
        return
    if args.command == "mcp-add":
        path = add_mcp_config(args.name, args.server_command, args.args)
        console.print(f"MCP config saved to {path}")
        return
    if args.command == "mcp-list":
        table = Table("name", "command", "args")
        for cfg in load_mcp_configs():
            table.add_row(cfg.name, cfg.command, " ".join(cfg.args))
        console.print(table)
        return
    # -- Daemon commands (removed) --
    if args.command == "daemon":
        console.print("Daemon has been removed.")
        return

    # -- Claw commands (removed — daemon dependency) --
    if args.command == "claw":
        console.print("Claw/daemon has been removed. Use `localcode agent <prompt>` instead.")
        return

    # -- Skill commands --
    if args.command == "skill":
        if args.skill_command in {None, "list"}:
            names = list_skills(Path.cwd())
            if not names:
                console.print("No skills installed. Run `localcode skill init` for starter skills.")
            else:
                for name in names:
                    console.print(f"  {name}")
            return
        if args.skill_command == "install":
            ok, msg = install_skill(args.source)
            console.print(msg)
            return
        if args.skill_command == "remove":
            ok, msg = remove_skill(args.name)
            console.print(msg)
            return
        if args.skill_command == "search":
            results = search_skills(args.query, Path.cwd())
            if not results:
                console.print("No matching skills.")
                return
            table = Table("name", "preview")
            for r in results:
                table.add_row(r["name"], r["preview"][:80])
            console.print(table)
            return
        if args.skill_command == "info":
            info = skill_info(args.name, Path.cwd())
            if not info:
                console.print(f"Skill not found: {args.name}")
                return
            table = Table(show_header=False)
            for k, v in info.items():
                table.add_row(k, str(v)[:200])
            console.print(table)
            return
        if args.skill_command == "init":
            count = ensure_builtin_skills()
            console.print(f"Installed {count} built-in starter skills." if count else "Built-in skills already installed.")
            return
        console.print("Usage: localcode skill [list|install|remove|search|info|init]")
        return

    if args.command == "verify":
        output, code = run_verification(Path.cwd(), args.verify_command)
        console.print(output)
        raise SystemExit(code)
    if args.command == "export-traces":
        console.print("Trace export has been removed.")
        return
    if args.command == "exec":
        raise SystemExit(run_exec(config, args.prompt, Path.cwd(), args.profile, args.model))
    if args.command == "agent-bg":
        console.print("Background agent has been removed. Use `localcode agent <prompt>` instead.")
        return
    if args.command == "orch":
        console.print("Orchestrator has been removed. Use `localcode agent <prompt>` instead.")
        return
    if args.command in {"agent", "agent-runner"}:
        app = GemApp(config=config, cwd=Path.cwd(), profile_name=args.profile, model_name=args.model)
        try:
            composed = [{"role": "user", "content": args.prompt}]
            result = run_agent_loop(app, args.prompt, composed, app.out)
            if result:
                console.print(result)
        finally:
            app.close()
        return
    if args.command == "resume":
        if args.latest:
            latest = SessionStore().latest_for_repo(Path.cwd())
            if latest is None:
                console.print("No saved session for this repo.")
                raise SystemExit(1)
            session_id = latest.session_id
        elif args.session_id:
            session_id = args.session_id
        else:
            console.print("Provide a session id or use --latest.")
            raise SystemExit(1)
        app = GemApp(
            config=config,
            session_id=session_id,
            cwd=Path.cwd(),
            profile_name=args.profile,
            model_name=args.model,
        )
        try:
            app.run()
        finally:
            app.close()
        return

    if args.command is None and _should_autobootstrap(config):
        console.print("[green]Setting up LocalCode for first launch...[/]")
        code = run_setup(
            config,
            args.profile,
            args.model,
            auto_install=True,
            benchmark=True,
            assume_defaults=True,
        )
        if code != 0:
            raise SystemExit(code)
        config = load_config()
        # Install built-in skills on first run
        count = ensure_builtin_skills()
        if count:
            console.print(f"  Installed {count} built-in skills (review, test, explain, refactor, debug)")

    # Auto-select best model: always prefer 26B if installed, even if config says e4b
    if not args.model:
        best = _auto_select_model(config)
        if best and best != config.runtime.model:
            config.runtime.model = best

    show_mode_picker = (
        config.runtime.provider == "llama_cpp"
        and config.runtime.kv_cache_type_v.startswith("turbo")
        and not getattr(args, "prompt", None)
    )

    app = GemApp(config=config, cwd=Path.cwd(), profile_name=args.profile, model_name=args.model)
    app._show_mode_picker = show_mode_picker
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n  Exiting.")
    except Exception as exc:
        import traceback
        console.print(f"[red]  Error: {exc}[/]")
        traceback.print_exc()
    finally:
        try:
            app.close()
        except Exception:
            pass


def _gpu_memory_unlocked() -> bool:
    """Check if GPU memory is sufficient for turbo mode."""
    from .performance import metal_gpu_available
    return metal_gpu_available()




def _auto_select_model(config) -> str:
    """Pick the best available model. 26B is the target — it's what we optimized for.

    The 26B is the only model that can reliably do tool calling via JSON,
    handle multi-step tasks, and produce quality code. e4b/e2b are fallbacks
    only if 26B isn't installed yet.
    """
    # For llama_cpp, the model is whatever the server is running — no selection needed.
    # The server is already loaded with the right GGUF.
    if config.runtime.provider == "llama_cpp":
        return config.runtime.model or ""

    try:
        from .runtime import GemRuntimeGateway
        gw = GemRuntimeGateway(config.runtime)
        installed = gw.list_models()
    except Exception:
        return ""

    # 26B is the primary model — everything was optimized for it
    for m in installed:
        if "26b" in m.lower() or "gemma26b" in m.lower():
            config.runtime.profile = "gemma4-26b-moe"
            return m

    # 26B not installed — warn and fall back
    import sys
    sys.stderr.write(
        "\033[33m  ! 26B model not found. Install it for best results:\n"
        "    ollama pull gemma4:26b\n"
        "    Falling back to smaller model.\033[0m\n"
    )

    # Fallback order
    for model_name, profile_key in [("gemma4:e4b", "gemma4-e4b"), ("gemma4:e2b", "gemma4-e2b")]:
        if any(model_name in m for m in installed):
            config.runtime.profile = profile_key
            return model_name

    for m in installed:
        if "gemma" in m.lower():
            return m

    return ""


if __name__ == "__main__":
    main(sys.argv[1:])
