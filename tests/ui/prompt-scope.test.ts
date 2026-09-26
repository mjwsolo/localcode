import { expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Plugin from "../../src/localcode/ui/plugin/localcode";

// The system prompt is llama-server's cached prefix: it must be identical for
// every request of a session, so the completion rules are present for the build
// agent from the first request on (never toggled by workspace activity) and the
// open-todo list travels on the user turn, not in the system prompt.
test("completion rules stay out of helper prompts and are constant for the build agent", async () => {
  const directory = mkdtempSync(join(tmpdir(), "prompt-scope-"));
  try {
    const hooks = await (Plugin as any)({ directory, client: {} });
    for (const agent of ["compaction", "title", "summary"]) {
      const output = { system: ["Summarize only."] };
      await hooks["experimental.chat.system.transform"]({ agent }, output);
      expect(output.system).toEqual(["Summarize only."]);
    }
    const first = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ agent: "build" }, first);
    expect(first.system.join("\n")).toContain("requested multi-step workspace changes");
    await hooks["tool.execute.before"]({ tool: "read" }, { args: { filePath: "." } });
    const afterTool = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ agent: "build" }, afterTool);
    expect(afterTool.system).toEqual(first.system);
    await hooks["chat.message"]({ sessionID: "test", agent: "build" }, { parts: [{ type: "text", text: "a different topic" }] });
    const nextTurn = { system: [] as string[] };
    await hooks["experimental.chat.system.transform"]({ agent: "build" }, nextTurn);
    expect(nextTurn.system).toEqual(first.system);
    expect(nextTurn.system.join("\n")).not.toContain("YOUR OPEN TODOS");
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
