import { expect, test } from "bun:test";
import { leanHistory } from "../../src/localcode/ui/plugin/context";

test("older tool data is bounded without mutating saved messages or breaking tool pairing", () => {
  const messages = [
    { info: { role: "user" }, parts: [{ type: "text", text: "edit this file" }] },
    { info: { role: "assistant" }, parts: [
      { type: "reasoning", text: "private old reasoning" },
      { type: "tool", callID: "call-1", tool: "read", state: { status: "completed", input: { filePath: "app.py", content: "a".repeat(9000) }, output: "b".repeat(24000), attachments: [{ mime: "image/png", url: "data:large" }] } },
      { type: "text", text: "I updated the file." },
    ] },
    { info: { role: "user" }, parts: [{ type: "text", text: "now explain it" }] },
    { info: { role: "assistant" }, parts: [{ type: "tool", callID: "call-2", state: { status: "completed", input: {}, output: "c".repeat(9000) } }] },
  ];
  const original = JSON.stringify(messages);
  const result = leanHistory(messages);
  expect(JSON.stringify(messages)).toBe(original);
  const tool = result[1].parts.find((part: any) => part.type === "tool");
  expect(tool.callID).toBe("call-1");
  expect(tool.state.input.filePath).toBe("app.py");
  expect(tool.state.output.length).toBeLessThanOrEqual(800);
  expect(tool.state.input.content.length).toBeLessThanOrEqual(600);
  expect(tool.state.attachments).toEqual([]);
  expect(result[1].parts.some((part: any) => part.type === "reasoning")).toBe(false);
  expect(result[1].parts.at(-1).text).toBe("I updated the file.");
  expect(result[3].parts[0].state.output.length).toBe(9000);
});

test("only synthetic automatic workspace context is removed", () => {
  const parts = [
    { type: "text", text: "WORKSPACE SNAPSHOT AT SESSION START is what I want to discuss" },
    { type: "text", text: "WORKSPACE SNAPSHOT AT SESSION START: files", synthetic: true },
  ];
  const messages = [{ info: { role: "user" }, parts }];
  expect(leanHistory(messages)[0].parts).toEqual([parts[0]]);
  expect(leanHistory(messages, true)[0].parts).toEqual(parts);
});

test("interrupted tool output is also abbreviated and file identifiers remain exact", () => {
  const filePath = "a/".repeat(400) + "file.py";
  const original = { type: "tool", callID: "interrupted", state: {
    status: "error", input: { filePath }, error: "cancelled",
    metadata: { interrupted: true, output: "log".repeat(9000) },
  } };
  const result = leanHistory([
    { info: { role: "assistant" }, parts: [original] },
    { info: { role: "user" }, parts: [{ type: "text", text: "continue" }] },
  ]);
  expect(result[0].parts[0].state.input.filePath).toBe(filePath);
  expect(result[0].parts[0].state.metadata.output.length).toBeLessThanOrEqual(800);
  expect(original.state.metadata.output.length).toBe(27000);
});

test("requested skills and user decisions remain complete, while old pending calls shrink", () => {
  const parts = [
    { type: "tool", tool: "skill", state: { status: "completed", input: { name: "requested" }, output: "rules".repeat(2000) } },
    { type: "tool", tool: "question", state: { status: "completed", input: {}, output: "decision".repeat(200) } },
    { type: "tool", tool: "write", callID: "pending", state: { status: "pending", input: { content: "x".repeat(9000) } } },
  ];
  const result = leanHistory([
    { info: { role: "assistant" }, parts },
    { info: { role: "user" }, parts: [{ type: "text", text: "continue" }] },
  ]);
  expect(result[0].parts[0]).toEqual(parts[0]);
  expect(result[0].parts[1]).toEqual(parts[1]);
  expect(result[0].parts[2].state.input.content.length).toBeLessThanOrEqual(600);
  expect(result[0].parts[2].callID).toBe("pending");
  expect(result[0].parts[2].state.status).toBe("pending");
});
