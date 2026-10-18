import { expect, test } from "bun:test";
import { createVisionLoader, hasImage } from "../../src/localcode/ui/plugin/vision";

const user = (...content: any[]) => ({ role: "user", content });

test("finds images attached by the user and returned by tools", () => {
  expect(hasImage([user({ type: "text", text: "hi" })])).toBe(false);
  expect(hasImage([user({ type: "media", mediaType: "image/png", data: "x" })])).toBe(true);
  expect(hasImage([user({ type: "media", mediaType: "application/pdf", data: "x" })])).toBe(false);
  expect(hasImage([{ role: "tool", content: [{ type: "tool-result", id: "1", name: "read", result: { type: "content", value: [
    { type: "text", text: "Image read" }, { type: "file", mime: "image/jpeg", uri: "data:" },
  ] } }] }])).toBe(true);
  expect(hasImage([{ role: "tool", content: [{ type: "tool-result", id: "1", name: "read", result: { type: "text", value: "plain" } }] }])).toBe(false);
});

function fakeSupervisor(initial: Record<string, any>) {
  let status: Record<string, any> = { state: "ready", current: "qwen", vision: true, vision_loaded: false, vision_available: true, vision_size_gb: 0.9, ...initial };
  const calls: string[] = [];
  const fetch = (async (url: string, init?: any) => {
    const path = new URL(url).pathname;
    calls.push(`${init?.method ?? "GET"} ${path}`);
    if (path === "/vision/load") {
      if (!status.vision) return new Response(JSON.stringify({ error: "Run /vision to download it." }));
      status = { ...status, state: "loading" };
      setTimeout(() => { status = { ...status, state: "ready", vision_loaded: true }; }, 5);
      return new Response(JSON.stringify({ ok: true, loading: true }));
    }
    return new Response(JSON.stringify(status));
  }) as unknown as typeof fetch;
  return { fetch, calls, status: () => status };
}

const image = [user({ type: "media", mediaType: "image/png", data: "x" })];

test("the first image loads the projector after one notice, then never again", async () => {
  const sup = fakeSupervisor({});
  const toasts: string[] = [];
  const ensure = createVisionLoader({
    control: "http://127.0.0.1:1", headers: () => ({}), fetch: sup.fetch,
    toast: (t) => { toasts.push(t.body.message); }, sleep: (ms) => new Promise((r) => setTimeout(r, Math.min(ms, 2))),
  });
  await ensure([user({ type: "text", text: "no image" })]);
  expect(sup.calls).toEqual([]);  // text-only requests never touch the supervisor
  await ensure(image);
  expect(sup.status().vision_loaded).toBe(true);
  expect(sup.calls.filter((c) => c === "POST /vision/load")).toHaveLength(1);
  expect(toasts).toHaveLength(1);
  expect(toasts[0]).toContain("0.9 GB image projector");
  await ensure(image);
  expect(sup.calls.filter((c) => c === "POST /vision/load")).toHaveLength(1);
  expect(toasts).toHaveLength(1);
});

test("a projector that is not downloaded is never fetched; the user is pointed to /vision", async () => {
  const sup = fakeSupervisor({ vision: false });
  const toasts: string[] = [];
  const ensure = createVisionLoader({ control: "http://127.0.0.1:1", headers: () => ({}), fetch: sup.fetch, toast: (t) => { toasts.push(t.body.message); } });
  await ensure(image);
  expect(sup.calls).not.toContain("POST /vision/load");
  expect(toasts[0]).toContain("/vision");
});

test("without a supervisor nothing happens", async () => {
  const ensure = createVisionLoader({ control: undefined, headers: () => ({}), toast: () => { throw new Error("no toast"); } });
  await ensure(image);
});
