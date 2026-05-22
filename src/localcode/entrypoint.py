"""LocalCode launcher — TUI + essential subcommands."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Lightweight imports only at module level — these are needed for
# `--help` and argument parsing.  The heavy modules (`bootstrap`,
# `models`, `runtime`, `performance`, and `rich.{Panel,Table}`) are
# only imported inside the subcommand branches that actually use
# them, shaving ~300–500 ms off `localcode --help` and the no-op
# TUI launch path on cold start.
from rich.console import Console
from .config import get_config_path, init_config_file, load_config
from .logging_utils import configure_logging


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
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        default=None,
        help="Resume a previous session by ID (use `--resume last` for the most recent). "
             "Session IDs are printed when LocalCode exits.",
    )
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
    # `status` / `doctor` were CLI subcommands; replaced with the
    # in-TUI `/status` slash command. Run localcode and type `/status`
    # in the chat input.
    subparsers.add_parser("models", help="list Gemma profiles and installed local models")
    subparsers.add_parser(
        "unstick",
        help="recover from a stuck llama-server without rebooting "
             "(runs memory_pressure + purge, requires admin)",
    )

    return parser


def _harden_against_debugger_attach() -> None:
    """Refuse all debugger attachments at the kernel level.

    macOS ptrace(PT_DENY_ATTACH=31, 0, 0, 0) tells the kernel that this
    process will NEVER be a debug target. Any subsequent
    `lldb -p <our-pid>`, `dtrace -p <our-pid>`, etc. fails with
    "Operation not permitted" — they can't reach in to SIGSTOP us, can't
    dump backtraces, can't trigger the Touch ID prompt.

    This is the bulletproof complement to the bash-tool regex block.
    The bash regex stops the AGENT from spawning lldb in the first place;
    PT_DENY_ATTACH stops ANY OTHER process (a stray terminal, a misclick,
    Activity Monitor's "Sample Process", a wrapper script) from doing it
    either. Both layers — defense in depth.

    Side effect: legitimate `lldb -p $(pgrep localcode)` from a developer
    also fails. Acceptable trade-off given the user-impact of accidental
    SIGSTOPs killing the TUI mid-session. To debug, use logs in
    ~/.localcode/ or set LOCALCODE_ALLOW_DEBUGGER=1 to skip this.
    """
    import os, platform
    if platform.system() != "Darwin":
        return
    if os.environ.get("LOCALCODE_ALLOW_DEBUGGER") == "1":
        return
    try:
        import ctypes
        libc = ctypes.CDLL("/usr/lib/libc.dylib")
        PT_DENY_ATTACH = 31
        # Annotate the call signature — ptrace returns int. Without an
        # explicit restype/argtypes ctypes can corrupt the return on
        # arm64, hide a non-zero failure code, and silently no-op
        # without raising. Setting restype catches that.
        libc.ptrace.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        libc.ptrace.restype = ctypes.c_int
        rc = libc.ptrace(PT_DENY_ATTACH, 0, None, 0)
        # Log to a known file so user can verify it ran on their box.
        import os as _os
        log_dir = _os.path.expanduser("~/.localcode")
        try:
            _os.makedirs(log_dir, exist_ok=True)
            with open(_os.path.join(log_dir, "logs", "startup.log"), "a") as _f:
                _f.write(f"[startup] PT_DENY_ATTACH rc={rc} pid={_os.getpid()}\n")
        except Exception:
            pass
    except Exception as e:
        try:
            import os as _os
            log_dir = _os.path.expanduser("~/.localcode/logs")
            _os.makedirs(log_dir, exist_ok=True)
            with open(_os.path.join(log_dir, "startup.log"), "a") as _f:
                _f.write(f"[startup] PT_DENY_ATTACH FAILED: {e}\n")
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:
    _harden_against_debugger_attach()
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
        from .bootstrap import run_setup
        raise SystemExit(run_setup(config, args.profile, args.model, args.install, args.benchmark))
    if args.command == "benchmark":
        from .performance import benchmark_report
        from rich.panel import Panel
        from rich.table import Table
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
    if args.command == "models":
        from .models import GEMMA_PROFILES
        from rich.table import Table
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
    # Pass through --resume so the TUI can seed the chat log with the
    # prior session's messages before showing the chat screen.
    if getattr(args, "resume", None):
        app._resume_session_id = args.resume
    try:
        app.run()
    finally:
        try:
            _reset_terminal_state()
        except Exception:
            pass
        try:
            # force=True → SIGKILL straight away. The graceful path
            # waits up to 5 s for llama-server to dealloc 38 GB of
            # Metal memory, which the kernel reclaims for free on
            # process death anyway. Skipping it = exit feels instant.
            ServerManager.get().shutdown(force=True)
        except Exception:
            pass
        try:
            _reset_terminal_state()
        except Exception:
            pass
        # Print exit summary AFTER textual has restored the terminal so
        # the lines survive in scrollback — same UX as Claude Code's
        # "Resume with: ..." footer. Pull session ID + last assistant
        # message from app state; silent no-op if we can't find them.
        try:
            _print_exit_summary(app)
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


def _print_exit_summary(app) -> None:
    """Print a Claude-Code-style exit footer to stdout so the user sees:

      - the last assistant exchange (briefly), and
      - a `localcode --resume <id>` hint they can copy-paste

    This runs AFTER textual restores the terminal, so the lines persist
    in the user's terminal scrollback — they can scroll back through the
    whole conversation since textual was in inline-output mode for the
    last bit and the chat log itself was rendered into a normal scroll
    buffer (which we plan to add separately if textual ate it).
    """
    session_id = None
    last_user = ""
    last_assistant = ""
    # Pull from the engine's session if available
    try:
        engine = getattr(app, "engine", None)
        if engine is not None and getattr(engine, "session", None) is not None:
            session_id = engine.session.session_id
            for m in reversed(engine.session.messages):
                role = m.get("role", "")
                if not last_assistant and role == "assistant":
                    last_assistant = m.get("content", "")[:240]
                if not last_user and role == "user":
                    last_user = m.get("content", "")[:120]
                if last_user and last_assistant:
                    break
    except Exception:
        pass
    # Fallback: latest session file on disk
    if session_id is None:
        try:
            from .session import SessionStore
            store = SessionStore()
            sess = sorted(
                store.sessions_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if sess:
                session_id = sess[0].stem
        except Exception:
            pass
    # Build the footer. Plain ANSI — Rich/Textual is gone by now.
    # ALWAYS print something so the user has visible confirmation
    # localcode shut down cleanly and knows how to come back.
    BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
    GREEN = "\033[32m"
    out = ["", f"{DIM}─── LocalCode session ended ───{RESET}"]
    if last_user:
        out.append(f"  {DIM}you:{RESET}       {last_user}")
    if last_assistant:
        clipped = last_assistant.replace("\n", " ")
        out.append(f"  {DIM}assistant:{RESET}  {clipped}")
    out.append("")
    if session_id:
        out.append(f"  {BOLD}Resume:{RESET}  {GREEN}localcode --resume {session_id}{RESET}")
        out.append(f"  {DIM}Or just `localcode --resume last` to grab the most recent.{RESET}")
    else:
        out.append(f"  {BOLD}Restart:{RESET}  {GREEN}localcode{RESET}")
    out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
