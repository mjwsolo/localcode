import { expect, test } from "bun:test"
import path from "node:path"

test("read aloud toggles its owned process without stopping unrelated processes", () => {
  const python = process.env.LOCALCODE_PY ?? "python3"
  const result = Bun.spawnSync([python, "-c", `
import os, pathlib, subprocess, tempfile, threading
from localcode_supervisor import Supervisor
with tempfile.TemporaryDirectory() as directory:
    executable = pathlib.Path(directory) / 'say'
    executable.write_text('#!/bin/sh\\nexec /bin/sleep 30\\n')
    executable.chmod(0o755)
    os.environ['PATH'] = directory + os.pathsep + os.environ['PATH']
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.speech_lock = threading.Lock()
    supervisor.speech_proc = None
    other = subprocess.Popen([str(executable)])
    try:
        assert supervisor.voice_speak('hello') == {'ok': True, 'speaking': True}
        owned = supervisor.speech_proc
        assert owned.poll() is None
        assert supervisor.voice_speak('hello') == {'ok': True, 'speaking': False}
        assert owned.poll() is not None
        assert other.poll() is None
        assert supervisor.voice_speak('') == {'error': 'nothing to say'}
    finally:
        other.terminate()
        other.wait()
`], { cwd: path.resolve(import.meta.dir, "..") })
  expect(result.stderr.toString()).toBe("")
  expect(result.exitCode).toBe(0)
})
