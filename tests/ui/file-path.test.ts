import { expect, test } from "bun:test"
import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import LocalcodePlugin from "../../src/localcode/ui/plugin/localcode"

test("newline paths are rejected before a file tool can create an invisible filename", async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "localcode-path-"))
  try {
    const hooks = await LocalcodePlugin({ client: {}, directory } as any)
    for (const tool of ["read", "write", "edit"]) {
      await expect(hooks["tool.execute.before"]!({ tool } as any, { args: { filePath: "app.py\n" } })).rejects.toThrow("single-line file path")
      await hooks["tool.execute.before"]!({ tool } as any, { args: { filePath: "my app.py" } })
    }
    expect(await fs.readdir(directory)).not.toContain("app.py\n")
  } finally {
    await fs.rm(directory, { recursive: true, force: true })
  }
})
