from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Callable

from ._subproc_env import clean_env

DEFAULT_TIMEOUT = 120  # seconds
MAX_OUTPUT_CHARS = 100_000


@dataclass
class ShellResult:
    command: str
    returncode: int
    output: str
    timed_out: bool = False


def run_shell(
    command: str,
    cwd: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_output: int = MAX_OUTPUT_CHARS,
    on_output: Callable[[str], None] | None = None,
) -> ShellResult:
    env = clean_env()
    env["LOCALCODE_SHELL"] = "1"  # let child processes know they're inside localcode
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    except Exception as exc:
        return ShellResult(command=command, returncode=1, output=str(exc))

    output_parts: list[str] = []
    total_chars = 0
    timed_out = False
    assert process.stdout is not None

    try:
        for line in process.stdout:
            output_parts.append(line)
            total_chars += len(line)
            if on_output is not None:
                on_output(line.rstrip("\n"))
            if total_chars >= max_output:
                output_parts.append(f"\n...[output truncated at {max_output} chars]")
                break
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # kill the entire process group
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except Exception:
            pass
        process.wait(timeout=5)
        output_parts.append(f"\n...[command timed out after {timeout}s]")
        returncode = 124
    except Exception as exc:
        output_parts.append(f"\n...[error: {exc}]")
        returncode = process.wait()

    return ShellResult(
        command=command,
        returncode=returncode,
        output="".join(output_parts),
        timed_out=timed_out,
    )
