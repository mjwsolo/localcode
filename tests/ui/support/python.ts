import { existsSync } from "node:fs"
import path from "node:path"

/** The interpreter that runs localcode in these tests and the env it needs.
 *  CI sets LOCALCODE_PY to a python with the package installed; a developer
 *  checkout has neither, so fall back to the repo venv or the system python
 *  with `src/` on PYTHONPATH. Without this the four python-backed suites fail
 *  on every local run and silently rot. */
const root = path.resolve(import.meta.dir, "../../..")
const venv = path.join(root, ".venv/bin/python")
export const python = process.env.LOCALCODE_PY ?? (existsSync(venv) ? venv : "python3")
export const pythonEnv = { ...process.env, PYTHONPATH: [path.join(root, "src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) }
