import { expect, test } from "bun:test";
import { applyMasked, argsSummary, checkpointIndex, leanMessages, maskCandidates, maskLimits, maskPath, maskStub, withCheckpointTodos } from "../../src/localcode/ui/plugin/context";

// Model-request messages as opencode v2 hands them to session.hook("context"):
// assistant tool calls, with their results in the following role "tool" message.
const user = (text: string, extra: any[] = []) => ({ role: "user", content: [{ type: "text", text }, ...extra] });

test("older tool data is bounded without mutating saved messages or breaking tool pairing", () => {
  const messages = [
    user("edit this file"),
    { role: "assistant", content: [
      { type: "reasoning", text: "private old reasoning" },
      { type: "tool-call", id: "call-1", name: "read", input: { filePath: "app.py", content: "a".repeat(9000) } },
      { type: "text", text: "I updated the file." },
    ] },
    { role: "tool", content: [
      { type: "tool-result", id: "call-1", name: "read", result: { type: "content", value: [
        { type: "text", text: "b".repeat(24000) },
        { type: "media", mediaType: "image/png", data: "large" },
      ] } },
    ] },
    user("now explain it"),
    { role: "assistant", content: [{ type: "tool-call", id: "call-2", name: "read", input: {} }] },
    { role: "tool", content: [{ type: "tool-result", id: "call-2", name: "read", result: { type: "text", value: "c".repeat(9000) } }] },
  ];
  const original = JSON.stringify(messages);
  const result = leanMessages(messages);
  expect(JSON.stringify(messages)).toBe(original);
  const call = result[1].content.find((part: any) => part.type === "tool-call");
  expect(call.id).toBe("call-1");
  expect(call.input.filePath).toBe("app.py");
  expect(call.input.content.length).toBeLessThanOrEqual(600);
  const output = result[2].content[0];
  expect(output.id).toBe("call-1");
  expect(output.result.value).toHaveLength(1);
  expect(output.result.value[0].text.length).toBeLessThanOrEqual(800);
  expect(result[1].content.some((part: any) => part.type === "reasoning")).toBe(false);
  expect(result[1].content.at(-1).text).toBe("I updated the file.");
  expect(result[5].content[0].result.value.length).toBe(9000);
});

test("only automatic workspace context added by LocalCode is removed", () => {
  const parts = [
    { type: "text", text: "WORKSPACE SNAPSHOT AT SESSION START is what I want to discuss" },
    { type: "text", text: "WORKSPACE SNAPSHOT AT SESSION START: files", metadata: { localcode: true } },
  ];
  const messages = [{ role: "user", content: parts }];
  expect(leanMessages(messages)[0].content).toEqual([parts[0]]);
  expect(leanMessages(messages, true)[0].content).toEqual(parts);
});

test("interrupted tool output is also abbreviated and file identifiers remain exact", () => {
  const filePath = "a/".repeat(400) + "file.py";
  const failed = { type: "tool-result", id: "interrupted", name: "bash", resultType: "error", result: {
    error: { message: "cancelled " + "x".repeat(2000) },
    content: [{ type: "text", text: "log".repeat(9000) }],
  } };
  const result = leanMessages([
    { role: "assistant", content: [{ type: "tool-call", id: "interrupted", name: "bash", input: { filePath } }] },
    { role: "tool", content: [failed] },
    user("continue"),
  ]);
  expect(result[0].content[0].input.filePath).toBe(filePath);
  expect(result[1].content[0].result.content[0].text.length).toBeLessThanOrEqual(800);
  expect(result[1].content[0].result.error.message.length).toBeLessThanOrEqual(400);
  expect(failed.result.content[0].text.length).toBe(27000);
});

test("requested skills and user decisions remain complete, while old pending calls shrink", () => {
  const calls = [
    { type: "tool-call", id: "s", name: "skill", input: { name: "requested" } },
    { type: "tool-call", id: "q", name: "question", input: { text: "pick".repeat(400) } },
    { type: "tool-call", id: "pending", name: "write", input: { content: "x".repeat(9000) } },
  ];
  const results = [
    { type: "tool-result", id: "s", name: "skill", result: { type: "text", value: "rules".repeat(2000) } },
    { type: "tool-result", id: "q", name: "question", result: { type: "text", value: "decision".repeat(200) } },
  ];
  const result = leanMessages([
    { role: "assistant", content: calls },
    { role: "tool", content: results },
    user("continue"),
  ]);
  expect(result[0].content[0]).toEqual(calls[0]);
  expect(result[0].content[1]).toEqual(calls[1]);
  expect(result[1].content).toEqual(results);
  expect(result[0].content[2].input.content.length).toBeLessThanOrEqual(600);
  expect(result[0].content[2].id).toBe("pending");
});


test("masking thresholds are shares of the measured budget, never fixed token counts", () => {
  expect(maskLimits(8192)).toEqual({ trigger: 5734, protect: 2048, minimum: 1024 });
  expect(maskLimits(262144)).toEqual({ trigger: 183500, protect: 65536, minimum: 32768 });
  for (const value of [0, -1, NaN, Infinity]) expect(maskLimits(value)).toBeUndefined();
});

test("old tool output is masked only near the budget, in one batch, with a stub that points at the saved output", () => {
  const turn = (n: number, size: number, name = "read") => [
    user("turn " + n),
    { role: "assistant", content: [{ type: "tool-call", id: "c" + n, name, input: { filePath: `src/f${n}.py` } }] },
    { role: "tool", content: [{ type: "tool-result", id: "c" + n, name, result: { type: "text", value: "x".repeat(size) } }] },
  ];
  const limits = { trigger: 9000, protect: 1000, minimum: 1500 };
  // 4,000 chars = 1,000 tokens each.
  const messages = [...turn(1, 4000), ...turn(2, 4000), ...turn(3, 4000, "skill"), ...turn(4, 4000), ...turn(5, 4000)];
  // Below the trigger nothing is masked, however much is reclaimable.
  expect(maskCandidates(messages, limits, new Set(), 8999)).toEqual([]);
  expect(maskCandidates(messages, undefined, new Set(), 1e9)).toEqual([]);
  // At the trigger: c5 fills the protected window; c3 is a skill; c4, c2, c1 reclaim 3,000 tokens,
  // including output from the current turn (no turn-count exemption).
  const masked = new Set(maskCandidates(messages, limits, new Set(), 9000));
  expect([...masked].sort()).toEqual(["c1", "c2", "c4"]);
  const result = applyMasked(messages, masked);
  const stub = maskStub("read", { filePath: "src/f1.py" }, "c1");
  expect(result[2].content[0].result).toEqual({ type: "text", value: stub });
  expect(stub).toBe("[Old tool output masked to save context: read(filePath=src/f1.py). Full output saved in .localcode-agent/tool-out/c1.txt; read that file if you need it again.]");
  expect(result[8].content[0].result.value.length).toBe(4000);
  expect(result[14].content[0].result.value.length).toBe(4000);
  expect(messages[2].content[0].result.value.length).toBe(4000);
  // Too little to reclaim: nothing changes. A masked output stops the scan, so a later batch
  // never edits history older than the previous one.
  expect(maskCandidates(messages, { ...limits, minimum: 5000 }, new Set(), 9000)).toEqual([]);
  expect(maskCandidates(messages, limits, masked, 1e9)).toEqual([]);
  // Failed tools are never masked.
  const failed = [...turn(1, 4000), ...turn(2, 0)];
  failed[2] = { role: "tool", content: [{ type: "tool-result", id: "c1", name: "bash", resultType: "error", result: { error: { message: "x" }, content: [] } }] };
  expect(maskCandidates(failed, { trigger: 0, protect: 0, minimum: 0 }, new Set(), 1)).toEqual([]);
});

test("argument summaries are short and deterministic, and saved-output paths are safe", () => {
  expect(argsSummary({ command: "pytest -q\n  tests", description: "run" })).toBe("command=pytest -q tests, description=run");
  expect(argsSummary({ content: "x".repeat(500), filePath: "a.py" })).toBe("filePath=a.py, content=" + "x".repeat(77) + "...");
  expect(argsSummary(undefined)).toBe("");
  expect(maskPath("call/../1:2")).toBe(".localcode-agent/tool-out/call____1_2.txt");
});

test("open todos ride at the tail of the compaction checkpoint", () => {
  const messages = [{ role: "user", content: [{ type: "text", text: "<conversation-checkpoint>\nsummary" }] }, user("next")];
  expect(checkpointIndex(messages)).toBe(0);
  expect(checkpointIndex([user("hello")])).toBe(-1);
  const out = withCheckpointTodos(messages, 0, "Open todos: [ ] b");
  expect(out[0].content.map((part: any) => part.text)).toEqual(["<conversation-checkpoint>\nsummary", "Open todos: [ ] b"]);
  expect(messages[0].content).toHaveLength(1);
  expect(withCheckpointTodos(messages, 0, "")).toBe(messages);
});
