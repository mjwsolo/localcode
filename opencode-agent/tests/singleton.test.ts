import { expect, test } from "bun:test"
import path from "node:path"

test("only one supervisor runs and a killed launcher cannot leave its model behind", () => {
  const result = Bun.spawnSync([process.env.LOCALCODE_PY ?? "python3", "tests/support/singleton.py"], {
    cwd: path.resolve(import.meta.dir, ".."),
  })
  expect(result.stderr.toString()).toBe("")
  expect(result.exitCode).toBe(0)
}, 15000)
