import { expect, spyOn, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { needsVision } from "../../src/localcode/ui/plugin/vision";
import Plugin from "../../src/localcode/ui/plugin/localcode";

const img = { type: "file", mime: "image/png", url: "data:image/png;base64,AA==" };
const txt = { type: "text", text: "what is this" };

test("needsVision only for an image part with vision off", () => {
  expect(needsVision([txt, img], { vision: false })).toBe(true);
  expect(needsVision([txt, img], { vision: true })).toBe(false);
  expect(needsVision([txt], { vision: false })).toBe(false);
  expect(needsVision([{ type: "file", mime: "text/plain" }], { vision: false })).toBe(false);
  expect(needsVision([img], undefined)).toBe(false);
});

test("plugin strips the image and toasts when the model has no vision", async () => {
  const dir = mkdtempSync(join(tmpdir(), "vision-"));
  const prev = process.env.LOCALCODE_CONTROL_URL;
  const control = "http://control.invalid";
  process.env.LOCALCODE_CONTROL_URL = control;
  const request = spyOn(globalThis, "fetch").mockImplementation(async (url) => {
    if (String(url) !== control + "/status") throw new Error("unexpected control endpoint");
    return Response.json({ vision: false });
  });
  const toasts: any[] = [];
  try {
    const hooks = await (Plugin as any)({ directory: dir, client: { tui: { showToast: (i: any) => toasts.push(i) } } });
    const output = { parts: [{ ...txt }, { ...img }] as any[] };
    await hooks["chat.message"]({ sessionID: "v", agent: "build", messageID: "m" }, output);
    expect(output.parts.some((p) => p.type === "file")).toBe(false);
    expect(output.parts[0].text).toBe("what is this");
    expect(toasts.length).toBe(1);
  } finally {
    if (prev === undefined) delete process.env.LOCALCODE_CONTROL_URL; else process.env.LOCALCODE_CONTROL_URL = prev;
    request.mockRestore();
    rmSync(dir, { recursive: true, force: true });
  }
});
