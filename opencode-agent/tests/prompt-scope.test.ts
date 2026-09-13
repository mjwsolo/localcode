import { expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Plugin from "../plugins/localcode";

test("completion rules stay out of compaction and other helper prompts", async () => {
  const directory = mkdtempSync(join(tmpdir(), "prompt-scope-"));
  try {
    const hooks = await (Plugin as any)({ directory, client: {} });
    for (const agent of ["compaction", "title", "summary"]) {
      const output = { system: ["Summarize only."] };
      await hooks["experimental.chat.system.transform"]({ agent }, output);
      expect(output.system).toEqual(["Summarize only."]);
    }
    const initial = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ agent: "build" }, initial);
    expect(initial.system).toEqual([]);
    await hooks["tool.execute.before"]({ tool: "websearch" }, { args: { query: "general research" } });
    await hooks["experimental.chat.system.transform"]({ agent: "build" }, initial);
    expect(initial.system).toEqual([]);
    await hooks["tool.execute.before"]({ tool: "read" }, { args: { filePath: "." } });
    const output = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ agent: "build" }, output);
    expect(output.system.join("\n")).toContain("requested multi-step workspace changes");
    await hooks["chat.message"]({ sessionID: "test", agent: "build" }, { parts: [{ type: "text", text: "a different topic" }] });
    const next = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ agent: "build" }, next);
    expect(next.system).toEqual([]);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
