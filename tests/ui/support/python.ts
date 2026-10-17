import { existsSync, mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import path from "node:path"

/** The interpreter that runs localcode in these tests and the env it needs.
 *  CI sets LOCALCODE_PY to a python with the package installed; a developer
 *  checkout has neither, so fall back to the repo venv or the system python
 *  with `src/` on PYTHONPATH. Without this the four python-backed suites fail
 *  on every local run and silently rot. */
const root = path.resolve(import.meta.dir, "../../..")
const venv = path.join(root, ".venv/bin/python")
export const python = process.env.LOCALCODE_PY ?? (existsSync(venv) ? venv : "python3")
// Supervisors started by these tests log and keep state in a throwaway run
// directory, never the user's real one.
const runDir = process.env.LOCALCODE_AGENT_RUN_DIR ?? mkdtempSync(path.join(tmpdir(), "localcode-test-run-"))
export const pythonEnv = {
  ...process.env,
  LOCALCODE_AGENT_RUN_DIR: runDir,
  PYTHONPATH: [path.join(root, "src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
}


/** Only skip a real integration test after an explicit capability denial. */
const listener = Bun.spawnSync([python, "-c", `
import errno, socket
try:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        sock.listen(1)
except OSError as error:
    if error.errno in (errno.EPERM, errno.EACCES):
        raise SystemExit(77)
    raise
`], { env: pythonEnv })
if (listener.exitCode !== 0 && listener.exitCode !== 77) {
  throw new Error(`Loopback capability probe failed: ${listener.stderr.toString()}`)
}
if (listener.exitCode === 77 && process.env.LOCALCODE_REQUIRE_INTEGRATION === "1") {
  throw new Error("Runner denies loopback listeners; integration coverage is required")
}
export const loopbackAvailable = listener.exitCode === 0
