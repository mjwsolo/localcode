import { expect, test } from "bun:test"
import path from "node:path"

test("updating the launcher while it waits cannot corrupt its remaining shell code", () => {
  const result = Bun.spawnSync([process.env.LOCALCODE_PY ?? "python3", "-c", `
from pathlib import Path
import subprocess, tempfile, time
source = Path('frontend_opencode.sh').read_text()
header = source[source.index('set -euo'):source.index('HERE=')]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    script, ready, stop = [root / name for name in ['launch.sh', 'ready', 'stop']]
    script.write_text('#!/bin/bash\\n' + header + 'touch "$1"\\nwhile [ ! -f "$2" ]; do sleep 0.02; done\\nprintf "clean exit\\\\n"\\n')
    child = subprocess.Popen(['bash', str(script), str(ready), str(stop)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):
            if ready.exists(): break
            time.sleep(.02)
        assert ready.exists()
        # Valid new shell code, but old file offsets land inside its quoted string.
        script.write_text('printf "%s\\\\n" "' + 'x' * 20000 + '"\\n')
        stop.touch()
        out, err = child.communicate(timeout=5)
        assert (child.returncode, out, err) == (0, 'clean exit\\n', ''), (child.returncode, out, err)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
`], { cwd: path.resolve(import.meta.dir, "..") })
  expect(result.stderr.toString()).toBe("")
  expect(result.exitCode).toBe(0)
})
