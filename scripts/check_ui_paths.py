"""Reject developer paths, allowing known upstream build paths and example text."""
import re
import subprocess
import sys


def unexpected_paths(lines):
    allowed = ("/Users/runner/", "/Users/name/", "/Users/administrator/Library/Services/buildkite-agent/builds/")
    return {match for line in lines for match in re.findall(r'/Users/[^\s\"\x27`<>]+', line)
            if not match.startswith(allowed)}


if __name__ == "__main__":
    with subprocess.Popen(["strings", sys.argv[1]], stdout=subprocess.PIPE, text=True, errors="replace") as proc:
        paths = unexpected_paths(proc.stdout)
        code = proc.wait()
    if code or paths:
        raise SystemExit("Developer path embedded in binary; build from a neutral directory")
