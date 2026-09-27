import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Plugin from "../../src/localcode/ui/plugin/localcode";

// The layout block is part of the cached prefix: identical on every request of a
// session, two levels deep, capped, and blind to build/tool directories.
test("workspace layout is constant per session, two levels, capped, and skips noise dirs", async () => {
  const dir = mkdtempSync(join(tmpdir(), "layout-"));
  try {
    mkdirSync(join(dir, "app")); writeFileSync(join(dir, "app", "calc.py"), "");
    mkdirSync(join(dir, "tests")); writeFileSync(join(dir, "tests", "test_calc.py"), "");
    mkdirSync(join(dir, "node_modules", "x"), { recursive: true }); writeFileSync(join(dir, "node_modules", "x", "i.js"), "");
    mkdirSync(join(dir, "app", "deep", "deeper"), { recursive: true }); writeFileSync(join(dir, "app", "deep", "deeper", "z.py"), "");
    for (let i = 0; i < 80; i++) writeFileSync(join(dir, `f${String(i).padStart(2, "0")}.txt`), "");
    const hooks = await (Plugin as any)({ directory: dir, client: {} });
    const a = { system: [] as string[] }; await hooks["experimental.chat.system.transform"]({ agent: "build" }, a);
    writeFileSync(join(dir, "late.txt"), "");            // files created mid-session must NOT change the prefix
    const b = { system: [] as string[] }; await hooks["experimental.chat.system.transform"]({ agent: "build" }, b);
    expect(b.system).toEqual(a.system);
    const layout = a.system.find((s) => s.startsWith("WORKSPACE LAYOUT"))!;
    expect(layout).toBeDefined();
    expect(layout).toContain("app/"); expect(layout).toContain("  calc.py"); expect(layout).toContain("tests/");
    expect(layout).not.toContain("node_modules"); expect(layout).not.toContain("deeper"); expect(layout).not.toContain("late.txt");
    expect(layout).toContain("truncated");
    expect(layout.split("\n").length - 1).toBeLessThanOrEqual(60);
    expect(layout.length).toBeLessThanOrEqual(2600);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
