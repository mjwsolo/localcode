import { expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Plugin from "../../src/localcode/ui/plugin/localcode";

test("confirmed backend crash explains the cause and stops retries once per process", async () => {
  const directory = mkdtempSync(join(tmpdir(), "server-crash-"));
  const fetchBefore = globalThis.fetch;
  const controlBefore = process.env.LOCALCODE_CONTROL_URL;
  const tokenBefore = process.env.LOCALCODE_CONTROL_TOKEN;
  const messages: any[] = [];
  const aborts: any[] = [];
  let state: any = { state: "error", exit_code: -11, process_id: 100, detail: "Model server crashed (SIGSEGV). Open /server to restart." };
  let requests = 0;
  try {
    process.env.LOCALCODE_CONTROL_URL = "http://127.0.0.1:8323";
    process.env.LOCALCODE_CONTROL_TOKEN = "test-secret";
    globalThis.fetch = (async (url: any, options: any) => {
      expect(String(url)).toBe("http://127.0.0.1:8323/status");
      expect(options.headers["x-localcode-token"]).toBe("test-secret");
      requests++;
      return Response.json(state);
    }) as any;
    const hooks = await (Plugin as any)({ directory, client: {
      tui: { showToast: async (value: any) => messages.push(value) },
      session: { abort: async (value: any) => aborts.push(value) },
    } });
    const retry = { event: { type: "session.status", properties: { sessionID: "s", status: { type: "retry" } } } };
    await hooks.event(retry);
    await hooks.event({ event: { type: "session.error", properties: { sessionID: "s", error: { name: "APIError" } } } });
    expect(messages).toHaveLength(1);
    expect(messages[0].body.message).toContain("SIGSEGV");
    expect(aborts).toEqual([{ path: { id: "s" } }]);
    state = { ...state, process_id: 101 };
    await hooks.event(retry);
    expect(messages).toHaveLength(2);
    state = { state: "ready" };
    await hooks.event(retry);
    state = { state: "error", detail: "Connection temporarily unavailable" };
    await hooks.event(retry);
    expect(messages).toHaveLength(2);
    expect(aborts).toHaveLength(2);
    const before = requests;
    await hooks.event({ event: { type: "session.error", properties: { sessionID: "s", error: { name: "MessageAbortedError" } } } });
    expect(requests).toBe(before);
  } finally {
    globalThis.fetch = fetchBefore;
    if (controlBefore === undefined) delete process.env.LOCALCODE_CONTROL_URL; else process.env.LOCALCODE_CONTROL_URL = controlBefore;
    if (tokenBefore === undefined) delete process.env.LOCALCODE_CONTROL_TOKEN; else process.env.LOCALCODE_CONTROL_TOKEN = tokenBefore;
    rmSync(directory, { recursive: true, force: true });
  }
});
