"""Bound a UI invocation and clean up its owned process group on timeout."""
import os
import signal
import subprocess


def run_bounded(args, *, timeout, **kwargs):
    if timeout <= 0:
        raise subprocess.TimeoutExpired(args, timeout)
    process = subprocess.Popen(args, start_new_session=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        if isinstance(error, subprocess.TimeoutExpired):
            raise subprocess.TimeoutExpired(args, timeout, output=stdout, stderr=stderr) from error
        raise
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
