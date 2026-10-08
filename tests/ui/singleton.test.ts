import { expect, test } from "bun:test"
import path from "node:path"
import { python, pythonEnv } from "./support/python"

test("only one supervisor runs and a killed launcher cannot leave its model behind", () => {
  const result = Bun.spawnSync([python, "support/singleton.py"], {
    cwd: import.meta.dir,
    env: pythonEnv,
  })
  expect(result.stderr.toString()).toBe("")
  expect(result.exitCode).toBe(0)
}, 15000)
