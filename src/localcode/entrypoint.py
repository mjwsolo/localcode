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

# Snapshot of the controlling tty's termios settings, captured BEFORE the
# TUI puts the terminal into raw mode. The escape sequence above only undoes
# escape-driven state (alt-screen, mouse tracking, hidden cursor); it does
# NOT undo the kernel-level raw mode (ECHO/ICANON/ISIG off). If the TUI
# crashes or hangs and its own teardown is skipped, the terminal is left in
# raw mode — no echo, and Ctrl+C generates no SIGINT — which looks like a
# dead terminal. Restoring this snapshot (or `stty sane`) brings it back.
_ORIG_TERMIOS = None  # tuple[int fd, list attrs] | None


def _snapshot_terminal_state() -> None:
    """Capture the tty's termios attrs before the TUI mutates them."""
    global _ORIG_TERMIOS
    if _ORIG_TERMIOS is not None:
        return
    try:
        import termios
        for stream in (sys.__stdin__, sys.__stdout__):
            try:
                if stream is not None and stream.isatty():
                    fd = stream.fileno()
                    _ORIG_TERMIOS = (fd, termios.tcgetattr(fd))
                    return
            except Exception:
                continue
    except Exception:
        pass


def _reset_terminal_state() -> None:
    payload = _TERMINAL_RESTORE.encode()
    wrote = False
    try:
        with open("/dev/tty", "wb", buffering=0) as tty:
            tty.write(payload)
        wrote = True
    except Exception:
        pass
    if not wrote:
        try:
            if sys.__stdout__.isatty():
                sys.__stdout__.write(_TERMINAL_RESTORE)
                sys.__stdout__.flush()
        except Exception:
            pass
    # Undo raw mode at the kernel level — escape codes can't do this. Restore
    # the snapshot if we have one; otherwise fall back to `stty sane`. Without
    # this, a crashed/hung TUI leaves the shell with no echo and a dead
    # Ctrl+C until the user blindly types `reset`.
    try:
        import termios
        if _ORIG_TERMIOS is not None:
            fd, attrs = _ORIG_TERMIOS
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        else:
            raise RuntimeError("no snapshot")
    except Exception:
        try:
            import subprocess
            subprocess.run(
                ["stty", "sane"],
                stdin=open("/dev/tty", "rb"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
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

    # The TUI (bare `localcode`) is the product — first-run setup, config,
    # benchmarking, and model management all live inside it. Only two non-TUI
    # entry points remain: `run` (headless) and `unstick` (recovery).
    run = subparsers.add_parser(
        "run",
        help="run a single coding goal headlessly (no TUI) and exit — "
             "for scripting, CI, and the benchmark harness",
    )
    run.add_argument("--goal", required=True, help="the task for the agent to perform")
    run.add_argument("--binary", default=None,
                     help="path to a llama-server binary (e.g. stock llama.cpp on Linux CI; "
                          "pair with LOCALCODE_SERVER_FLAVOR=vanilla)")
    run.add_argument("--timeout", type=int, default=0,
                     help="abort after N seconds (0 = no limit)")
    run.add_argument("--max-rounds", type=int, default=None,
                     help="maximum model/tool rounds (0 = unlimited)")
    run.add_argument("--thinking", choices=["off", "auto", "on"], default=None,
                     help="hidden-reasoning policy for this run")
    run.add_argument("--thinking-budget", type=int, default=None,
                     help="reasoning-token budget (0 = model default, negative = disable)")
    run.add_argument("--quiet", action="store_true",
                     help="suppress streamed agent output; print only the final answer")
    run.add_argument("--json", action="store_true",
                     help="emit the agent's event stream as JSON Lines (one JSON object "
                          "per line) on stdout instead of human-readable output — for "
                          "editors / CI driving LocalCode programmatically")

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


def _run_headless(config, args, console) -> int:
    """Run one goal through the agent loop with no TUI, then exit.

    Same backend the TUI drives (LocalCodeApp + .ask), minus the Textual
    screen: events stream to stdout and we return a process exit code.
    This is what the eval harness and CI benchmark invoke. Approvals are
    forced to full-auto — there's no human to answer prompts.

    Exit codes: 0 ok · 1 error · 124 timeout · 130 interrupted.

    With ``--json`` the same backend runs, but instead of Rich/human output
    we emit the agent's event stream as JSON Lines on stdout (one object per
    line) and a final ``result`` event with status + token counts + exit
    reason. See ``headless_json`` for the schema.
    """
    import os
    from pathlib import Path as _Path
    os.environ["LOCALCODE_AUTONOMY"] = "full_auto"  # no human to approve tools

    if getattr(args, "json", False):
        from .headless_json import run_headless_json
        return run_headless_json(config, args)

    from .app import LocalCodeApp
    from .server_manager import _probe_health
    from .bootstrap import resolve_model_arg
    from .models_catalog import CHOICES

    # Resolve the model to a concrete downloaded GGUF. `--model` and the config
    # may hold a TAG (e.g. "qwen38", "gemma26b-iq3"), a display name, a bare
    # filename, or a path - resolve_model_arg handles them all, so an explicit
    # --model is honored instead of being ignored for the config default. Order:
    # explicit --model → configured model → smallest downloaded catalog model.
    resolved: _Path | None = None
    for candidate in (args.model, config.runtime.model):
        resolved = resolve_model_arg(candidate)
        if resolved:
            break
    if resolved is None:
        downloaded = [c for c in CHOICES if c.local_path.exists()]
        if downloaded:
            resolved = min(downloaded, key=lambda c: c.size_gb).local_path
    if resolved is None:
        console.print(
            "[red]error:[/] no model found on disk. Launch `localcode` and "
            "download one in the TUI, or pass --model <downloaded.gguf>."
        )
        return 1
    config.runtime.model = str(resolved)
    console.print(f"[dim]model: {resolved.name}[/]")
    if args.binary:
        config.runtime.llama_cpp_binary = args.binary
        console.print(f"[dim]server binary: {args.binary}[/]")

    app = LocalCodeApp(config, profile_name=args.profile)

    # Ensure the inference server is up — same path the agent uses on a
    # cold turn: probe the usual port range, restart if nothing answers.
    if not any(_probe_health(p, timeout=1.0) for p in range(8081, 8100)):
        console.print("[dim]starting model server…[/]")
        if not app.engine._restart_server():
            console.print(
                "[red]error:[/] could not start the model server "
                "(model downloaded? llama-server binary present?)"
            )
            return 1

    # Optional hard timeout so a stuck run can't hang CI forever.
    if args.timeout and args.timeout > 0:
        import signal

        def _on_timeout(_sig, _frame):
            raise TimeoutError(f"run exceeded {args.timeout}s")

        signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(args.timeout)

    try:
        result = app.ask(args.goal, stream=not args.quiet)
    except TimeoutError as e:
        console.print(f"[red]timeout:[/] {e}")
        return 124
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/]")
        return 130
    except Exception as e:  # noqa: BLE001 — headless: surface any failure as exit 1
        console.print(f"[red]run failed:[/] {type(e).__name__}: {e}")
        return 1
    finally:
        if args.timeout and args.timeout > 0:
            import signal
            signal.alarm(0)

    if args.quiet:
        console.print(result or "")
    return 0


def main(argv: list[str] | None = None) -> None:
    # Opt-in agent-plane front end. Default behaviour is untouched.
    import os as _os
    import sys as _sys
    if _os.environ.get("LOCALCODE_FRONTEND") == "agent":
        from .frontend_agent import run as _run_agent
        # The console script calls main() with no argv; fall back to sys.argv.
        _sys.exit(_run_agent(argv if argv is not None else _sys.argv[1:]))
    _harden_against_debugger_attach()
    # Snapshot the terminal while it's still sane — before the TUI enters
    # raw mode — so any exit path (crash, hang, kill) can restore it.
    _snapshot_terminal_state()
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
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    configure_logging(config.ui.show_debug)
    console = Console()

    if args.cwd:
        project_dir = Path(args.cwd).resolve()
        if not project_dir.is_dir():
            console.print(f"[red]Error:[/] directory not found: {args.cwd}")
            sys.exit(1)
        os.chdir(project_dir)

    # Subcommands. The TUI (bare `localcode`) is the product; setup, config,
    # benchmark, and model listing all happen inside it. The only non-TUI
    # entry points kept are `run` (headless JSON API for CI/scripting) and
    # `unstick` (stuck-server recovery).
    if args.command == "unstick":
        # Recovery path for the "llama-server is stuck in a kernel wait
        # that SIGKILL can't reach" scenario. Not a reboot — but still
        # requires admin for memory_pressure + purge.
        from .recovery import attempt_recovery
        ok, msg = attempt_recovery(verbose=True)
        console.print(f"\n{'[green]✓[/]' if ok else '[yellow]⚠[/]'} {msg}")
        sys.exit(0 if ok else 1)
    if args.command == "run":
        if args.max_rounds is not None:
            if args.max_rounds < 0:
                parser.error("--max-rounds must be 0 or greater")
            config.runtime.max_rounds = args.max_rounds
        if args.thinking is not None:
            config.runtime.internal_thinking_mode = args.thinking
        if args.thinking_budget is not None:
            config.runtime.thinking_budget_tokens = args.thinking_budget
        sys.exit(_run_headless(config, args, console))

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
    # Mouse capture OFF by default (same as tui.app.main). Capturing the mouse
    # disables the terminal's NATIVE text selection, so Cmd+C on an empty
    # native selection makes the terminal ring its bell — the persistent copy
    # "beep". This is the entry point `localcode` actually uses; it previously
    # called app.run() with Textual's mouse=True default, so the beep fix in
    # tui.app.main never reached it. Opt back in with LOCALCODE_MOUSE=1.
    import os as _os
    _mouse = _os.environ.get("LOCALCODE_MOUSE", "0") == "1"
    try:
        app.run(mouse=_mouse)
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
        # the lines survive in scrollback — a familiar "Resume with: ..."
        # footer pattern. Pull session ID + last assistant
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
    """Print an exit footer to stdout so the user sees:

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
