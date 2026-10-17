import { expect, test } from "bun:test";
import { leanMessages } from "../../src/localcode/ui/plugin/context";

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
