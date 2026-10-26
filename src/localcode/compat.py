"""Small platform shims so POSIX-only calls do not crash import or use on Windows.

Everything here keeps POSIX behaviour byte-for-byte; the Windows branches are the
only new code. Features that are not ported yet should fail with a clear message,
not an ImportError or AttributeError.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys

IS_WINDOWS = sys.platform == "win32"

# O_NOFOLLOW does not exist on Windows; O_BINARY must be set there so os.open
# does not translate newlines in binary data.
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)

SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
# Signals a terminal or a parent can send to ask a process to stop.
STOP_SIGNALS = tuple(
    s for s in (
        signal.SIGINT, signal.SIGTERM,
        getattr(signal, "SIGHUP", None), getattr(signal, "SIGBREAK", None),
    ) if s is not None
)


def new_group_popen_kwargs() -> dict:
    """Popen kwargs that put the child in its own process group / session."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def lock_exclusive_nb(fd_or_file) -> None:
    """Take an exclusive non-blocking lock; raise BlockingIOError if held."""
    fd = fd_or_file if isinstance(fd_or_file, int) else fd_or_file.fileno()
    if IS_WINDOWS:
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError(str(exc)) from None
        return
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def kill_tree(pid: int, *, force: bool = False) -> None:
    """Stop `pid` and its children. POSIX: its process group. Windows: taskkill /T."""
    if IS_WINDOWS:
        cmd = ["taskkill", "/PID", str(pid), "/T"] + (["/F"] if force else [])
        r = subprocess.run(cmd, capture_output=True, timeout=10)
        if r.returncode == 128:  # no such process
            raise ProcessLookupError(pid)
        return
    os.killpg(os.getpgid(pid), SIGKILL if force else signal.SIGTERM)


def is_group_leader(pid: int) -> bool:
    """True when `pid` leads its own process group (how we spawn servers)."""
    if IS_WINDOWS:
        return True  # no process groups; the command-line check still applies
    return os.getpgid(pid) == pid


def process_command_line(pid: int) -> str:
    """Command line of `pid`, or '' when unknown."""
    if IS_WINDOWS:
        try:
            import psutil
            return " ".join(psutil.Process(pid).cmdline())
        except Exception:
            return ""
    try:
        return subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
    except Exception:
        return ""


def listening_pids(port: int) -> list[int]:
    """PIDs holding a listening socket on `port`."""
    if IS_WINDOWS:
        try:
            import psutil
            return sorted({c.pid for c in psutil.net_connections(kind="inet")
                           if c.status == "LISTEN" and c.laddr and c.laddr.port == port and c.pid})
        except Exception:
            return []
    try:
        r = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=3)
    except Exception:
        return []
    return [int(p) for p in r.stdout.split() if p.isdigit()]
