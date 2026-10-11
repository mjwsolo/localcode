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

test("a miscopied project path is corrected, a real outside path is left alone, and garbage gets a retryable error", async () => {
  // a long random directory name, the kind a small model drops a character from
  const parent = await fs.mkdtemp(path.join(os.tmpdir(), "localcode-path-"))
  const directory = path.join(parent, "cd6dvv2519l0ts64fdxrrq180000gn", "T", "project")
  const outside = path.join(parent, "elsewhere")
  try {
    await fs.mkdir(path.join(directory, "tests"), { recursive: true })
    await fs.mkdir(path.join(directory, "app"))
    await fs.mkdir(outside)
    await fs.writeFile(path.join(directory, "tests", "test_calc.py"), "")
    await fs.writeFile(path.join(directory, "README.md"), "")
    await fs.writeFile(path.join(outside, "notes.txt"), "")
    const hooks = await LocalcodePlugin({ client: {}, directory } as any)
    const before = async (tool: string, filePath: string) => {
      const output = { args: { filePath } }
      await hooks["tool.execute.before"]!({ tool } as any, output)
      return output.args.filePath
    }
    const real = await fs.realpath(directory)
    const expected = (rel: string) => [path.join(directory, rel), path.join(real, rel)]

    // the three shapes seen in real runs: a changed character, a truncated name, dropped directories
    for (const garbled of ["cd6dvv2519-l0ts64fdxrrq180000gn/T/project", "cd6dvv2519/T/project", "cd6dvv2519-ljne6t9f"]) {
      expect(expected("tests/test_calc.py")).toContain(await before("read", path.join(parent, garbled, "tests", "test_calc.py")))
      expect(expected("README.md")).toContain(await before("edit", path.join(parent, garbled, "README.md")))
    }
    // a new file in an existing project directory is placed there
    expect(expected("app/new.py")).toContain(await before("write", path.join(parent, "cd6dvv2519-x", "T", "project", "app", "new.py")))

    // untouched: relative paths, correct absolute paths, and real places outside the project
    expect(await before("read", "tests/test_calc.py")).toBe("tests/test_calc.py")
    expect(await before("read", path.join(directory, "tests", "test_calc.py"))).toBe(path.join(directory, "tests", "test_calc.py"))
    expect(await before("read", path.join(outside, "notes.txt"))).toBe(path.join(outside, "notes.txt"))
    expect(await before("read", path.join(outside, "missing.txt"))).toBe(path.join(outside, "missing.txt"))
    // a missing directory inside some other real tree is not this project's path gone wrong
    expect(await before("read", path.join(outside, "ghost", "tests", "test_calc.py"))).toBe(path.join(outside, "ghost", "tests", "test_calc.py"))

    // an invented root (models often write /home/user/project/...) cannot exist, and the rest names a project file
    expect(expected("tests/test_calc.py")).toContain(await before("read", "/nonexistent-root-dir/project/tests/test_calc.py"))

    // nothing in the project matches: the model is told the root and can retry, instead of a permission prompt
    await expect(before("read", path.join(parent, "cd6dvv2519-x", "T", "project", "nope.py"))).rejects.toThrow("relative to the project root")
    await expect(before("write", path.join(parent, "cd6dvv2519-x", "brand", "new.py"))).rejects.toThrow("is not inside the project")
  } finally {
    await fs.rm(parent, { recursive: true, force: true })
  }
})
