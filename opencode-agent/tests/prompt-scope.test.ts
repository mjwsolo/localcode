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
    const output = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ agent: "build" }, output);
    expect(output.system.join("\n")).toContain("requested multi-step workspace changes");
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
