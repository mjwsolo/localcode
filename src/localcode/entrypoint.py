"""LocalCode launcher — TUI + essential subcommands."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .bootstrap import run_setup
from .config import get_config_path, init_config_file, load_config
from .logging_utils import configure_logging
from .models import GEMMA_PROFILES, get_runtime_model, resolve_profile
from .performance import benchmark_report
from .runtime import LocalCodeRuntimeGateway


_TERMINAL_RESTORE = (
    "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1004l"
    "\x1b[?1005l\x1b[?1006l\x1b[?1015l\x1b[?2004l"
    "\x1b[?25h\x1b[?1049l\x1b[0m"
)


def _reset_terminal_state() -> None:
    payload = _TERMINAL_RESTORE.encode()
    try:
        with open("/dev/tty", "wb", buffering=0) as tty:
            tty.write(payload)
        return
    except Exception:
        pass
    try:
        if sys.__stdout__.isatty():
            sys.__stdout__.write(_TERMINAL_RESTORE)
            sys.__stdout__.flush()
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localcode", description="LocalCode — AI coding assistant running entirely on your machine")
    parser.add_argument("--profile", help="Gemma 4 profile: e2b, e4b, 26b-laptop, 26b-moe, 31b")
    parser.add_argument("--model", help="Explicit local runtime model tag")
    parser.add_argument("-c", "--cwd", type=str, default=None,
                        help="Working directory for the project (defaults to current directory)")
    parser.add_argument(
        "--preview-screen",
        choices=["setup", "mode-picker", "model-picker", "chat"],
        help="Visual-test a single screen with mock state. No real "
             "server / model is started — useful for iterating on UI "
             "tweaks without going through the full new-user flow.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("config-init", help="create a default config file")
    setup = subparsers.add_parser("setup", help="prepare local runtime and Ollama")
    setup.add_argument("--install", action="store_true", help="attempt to install Ollama and pull the selected model")
    setup.add_argument("--benchmark", action="store_true", help="benchmark the local machine and apply a recommended preset")
    benchmark = subparsers.add_parser("benchmark", help="inspect the local machine and recommend a LocalCode performance preset")
    benchmark.add_argument("--mode", choices=["fast", "balanced", "deep"])
    subparsers.add_parser("status", help="show runtime status and configuration")
    subparsers.add_parser("doctor", help="(alias for status)")
    subparsers.add_parser("models", help="list Gemma profiles and installed local models")
    subparsers.add_parser(
        "unstick",
        help="recover from a stuck llama-server without rebooting "
             "(runs memory_pressure + purge, requires admin)",
    )

    return parser


def run_status() -> int:
    config = load_config()
    profile = resolve_profile(config.runtime.profile, config.runtime.model)
    config.runtime.model = get_runtime_model(profile, config.runtime.model)
    gateway = LocalCodeRuntimeGateway(config.runtime)
    ok, details = gateway.healthcheck()
    console = Console()

    runtime_table = Table(show_header=False, title="Runtime", title_style="bold green")
    runtime_table.add_row("status", "[green]ok[/green]" if ok else "[red]unreachable[/red]")
    runtime_table.add_row("provider", config.runtime.provider)
    runtime_table.add_row("model", config.runtime.model)
    runtime_table.add_row("profile", profile.key)
    runtime_table.add_row("mode", config.runtime.mode)
    runtime_table.add_row("server", config.runtime.base_url)
    if not ok:
        runtime_table.add_row("error", details)
    console.print(runtime_table)

    perf_table = Table(show_header=False, title="Performance", title_style="bold green")
    if config.runtime.provider == "llama_cpp":
        perf_table.add_row("gpu_layers", str(config.runtime.llama_cpp_gpu_layers))
        perf_table.add_row("threads", str(config.runtime.llama_cpp_threads))
    perf_table.add_row("kv_cache", f"{config.runtime.kv_cache_type_k} / {config.runtime.kv_cache_type_v}")
    perf_table.add_row("context", config.runtime.cache_policy)
    console.print(perf_table)

    config_table = Table(show_header=False, title="Config", title_style="bold green")
    config_table.add_row("file", str(get_config_path()))
    console.print(config_table)

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> None:
    import os
    import signal
    import warnings
    _reset_terminal_state()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # macOS malloc-warning silencer. Setting `MallocStackLogging=0`
    # triggers "can't turn off malloc stack logging because it was
    # not enabled" into every subprocess's stderr — leaks into the
    # chat UI. We pop both *StackLogging* vars; ALSO pop
    # `MallocNanoZone` (typically set to 0 by Xcode / IDE dev
    # shells), which on some configurations triggers the same
    # warning indirectly via libsystem allocator setup paths.
    # Empirically still leaks on heavy / long-running subprocess
    # workloads, but stripping these from the env is a no-cost
    # mitigation. Subprocesses spawned via bash.py / server_manager
    # also re-filter these out of their own env dicts.
    os.environ.pop("MallocStackLogging", None)
    os.environ.pop("MallocStackLoggingNoCompact", None)
    os.environ.pop("MallocNanoZone", None)
    os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    os.environ["OLLAMA_KV_CACHE_TYPE"] = config.runtime.kv_cache_type_k
    configure_logging(config.ui.show_debug)
    console = Console()

    if args.cwd:
        project_dir = Path(args.cwd).resolve()
        if not project_dir.is_dir():
            console.print(f"[red]Error:[/] directory not found: {args.cwd}")
            sys.exit(1)
        os.chdir(project_dir)

    # Subcommands
    if args.command == "config-init":
        path = init_config_file()
        console.print(f"Config ready at {path}")
        return
    if args.command == "unstick":
        # Recovery path for the "llama-server is stuck in a kernel wait
        # that SIGKILL can't reach" scenario. Not a reboot — but still
        # requires admin for memory_pressure + purge.
        from .recovery import attempt_recovery
        ok, msg = attempt_recovery(verbose=True)
        console.print(f"\n{'[green]✓[/]' if ok else '[yellow]⚠[/]'} {msg}")
        sys.exit(0 if ok else 1)
    if args.command == "setup":
        raise SystemExit(run_setup(config, args.profile, args.model, args.install, args.benchmark))
    if args.command == "benchmark":
        machine, preset = benchmark_report(config, args.mode)
        table = Table(show_header=False)
        table.add_row("system", machine.system)
        table.add_row("cpu_cores", str(machine.cpu_cores))
        table.add_row("memory_gb", str(machine.memory_gb))
        table.add_row("gpu", machine.gpu_summary)
        table.add_row("tier", machine.tier)
        table.add_row("recommended_mode", preset.mode)
        table.add_row("profile", preset.profile)
        console.print(table)
        if preset.notes:
            console.print(Panel("\n".join(preset.notes), title="Recommendations"))
        return
    if args.command in ("status", "doctor"):
        raise SystemExit(run_status())
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
        return

    # --preview-screen: visual-test a screen in isolation. Mocks the
    # minimal app state each screen needs and pushes only that one.
    if getattr(args, "preview_screen", None):
        from .tui.app import LocalCodeTUI
        app = LocalCodeTUI(show_mode_picker=False)
        app._preview_screen = args.preview_screen.replace("-", "_")
        app.run()
        return

    # Default: launch TUI
    from .tui.app import LocalCodeTUI
    from .server_manager import ServerManager

    app = LocalCodeTUI(show_mode_picker=False)
    try:
        app.run()
    finally:
        try:
            _reset_terminal_state()
        except Exception:
            pass
        try:
            ServerManager.get().shutdown()
        except Exception:
            pass
        try:
            _reset_terminal_state()
        except Exception:
            pass
    # Handle GPU unlock: TUI exits with code 42 when sudo is needed
    if getattr(app, 'return_code', None) == 42:
        console.print("\n[bold]GPU memory unlock required (one-time setup)[/]\n")
        import subprocess as sp
        result = sp.run(["sudo", "sysctl", "iogpu.wired_limit_mb=14336"])
        if result.returncode == 0:
            console.print("\n[green]Done![/] Restarting LocalCode...\n")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            console.print("\n[red]Failed.[/] Run manually: sudo sysctl iogpu.wired_limit_mb=14336")


if __name__ == "__main__":
    main(sys.argv[1:])
