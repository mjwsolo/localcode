import { expect, test } from "bun:test";
import { createServer } from "node:http";
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
  const srv = createServer((_q, res) => { res.setHeader("content-type", "application/json"); res.end(JSON.stringify({ vision: false })); });
  await new Promise<void>((r) => srv.listen(0, "127.0.0.1", () => r()));
  const port = (srv.address() as any).port;
  const prev = process.env.LOCALCODE_CONTROL_URL;
  process.env.LOCALCODE_CONTROL_URL = `http://127.0.0.1:${port}`;
  const dir = mkdtempSync(join(tmpdir(), "vision-"));
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
    srv.close(); rmSync(dir, { recursive: true, force: true });
  }
});
