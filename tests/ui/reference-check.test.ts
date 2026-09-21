import { expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Plugin from "../../src/localcode/ui/plugin/localcode";

test("blocks a nested solution-only typecheck and accepts build mode or a child project", async () => {
  const dir = mkdtempSync(join(tmpdir(), "reference-check-"));
  try {
    const app = join(dir, "app");
    mkdirSync(app);
    writeFileSync(join(app, "tsconfig.json"), JSON.stringify({ files: [], references: [{ path: "./tsconfig.app.json" }] }));
    writeFileSync(join(app, "package.json"), JSON.stringify({ scripts: { build: "tsc && vite build" } }));
    const hooks = await (Plugin as any)({ directory: dir, client: {} });
    const before = (command: string) => hooks["tool.execute.before"]({ tool: "bash" }, { args: { command } });
    await expect(before("cd app && npm run build")).rejects.toThrow("zero source files");
    await expect(hooks["tool.execute.before"]({ tool: "bash" }, { args: { command: "npm run build", workdir: app } })).rejects.toThrow("zero source files");
    await expect(before("cd app && npx tsc --noEmit")).rejects.toThrow("zero source files");
    for (const cmd of ["npx tsc -b", "npx tsc --build", "npx tsc -p tsconfig.app.json", "npx tsc --project=tsconfig.app.json", "npx tsc --showConfig", "npx tsc --listFiles"])
      await expect(before(`cd app && ${cmd}`)).resolves.toBeUndefined();
    writeFileSync(join(app, "package.json"), JSON.stringify({ scripts: { build: "tsc -b && vite build" } }));
    await expect(before("cd app && npm run build")).resolves.toBeUndefined();
    writeFileSync(join(app, "tsconfig.json"), JSON.stringify({ files: ["src/main.ts"] }));
    await expect(before("cd app && npx tsc --noEmit")).resolves.toBeUndefined();
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
