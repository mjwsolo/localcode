/**
 * Contract tests for the pi APIs localcode's extensions depend on.
 *
 * These run on every weekly pi bump. They don't need a model or a server:
 * each extension's factory is invoked against a recording mock of
 * ExtensionAPI, and we assert that everything we rely on still registers and
 * that the pure logic still behaves. If pi renames a hook or changes a
 * registration signature, the typecheck (tsc against the installed package)
 * or these assertions go red at bump time instead of in a user's terminal.
 */
import { describe, expect, it } from "vitest";
import { scrub } from "../extensions/localcode-redact.ts";

function mockPi() {
  const handlers = new Map<string, Function[]>();
  const commands = new Map<string, any>();
  const tools = new Map<string, any>();
  const providers = new Map<string, any>();
  return {
    api: {
      on: (ev: string, fn: Function) => {
        handlers.set(ev, [...(handlers.get(ev) ?? []), fn]);
      },
      registerCommand: (name: string, opts: any) => commands.set(name, opts),
      registerTool: (tool: any) => tools.set(tool.name, tool),
      registerProvider: (name: string, cfg: any) => providers.set(name, cfg),
      setModel: async () => true,
      unregisterProvider: () => {},
      registerShortcut: () => {},
      registerFlag: () => {},
    } as any,
    handlers, commands, tools, providers,
  };
}

describe("localcode.ts (provider + picker + first-run)", () => {
  it("registers the provider, the /model command, and the hooks it needs", async () => {
    const m = mockPi();
    const mod = await import("../extensions/localcode.ts");
    await mod.default(m.api);
    expect(m.providers.has("localcode")).toBe(true);
    expect(m.commands.has("model")).toBe(true);
    for (const hook of ["before_provider_request", "model_select", "session_start"]) {
      expect(m.handlers.has(hook), `hook ${hook}`).toBe(true);
    }
  });

  it("disables the thinking channel on every provider request", async () => {
    const m = mockPi();
    const mod = await import("../extensions/localcode.ts");
    await mod.default(m.api);
    const h = m.handlers.get("before_provider_request")![0];
    const out = h({ payload: { model: "x" } }, {});
    expect(out.chat_template_kwargs).toEqual({ enable_thinking: false });
  });
});

describe("localcode-safety.ts (approval gate)", () => {
  it("blocks dangerous commands outright, headless", async () => {
    const m = mockPi();
    const mod = await import("../extensions/localcode-safety.ts");
    mod.default(m.api);
    const h = m.handlers.get("tool_call")![0];
    const res = await h({ toolName: "bash", input: { command: "sudo rm -rf /var" } }, { hasUI: false });
    expect(res?.block).toBe(true);
  });
  it("lets ordinary commands through", async () => {
    const m = mockPi();
    const mod = await import("../extensions/localcode-safety.ts");
    mod.default(m.api);
    const h = m.handlers.get("tool_call")![0];
    const res = await h({ toolName: "bash", input: { command: "ls -la" } }, { hasUI: false });
    expect(res?.block).not.toBe(true);
  });
});

describe("localcode-redact.ts", () => {
  it("scrubs vendor tokens and leaves ordinary text alone", () => {
    expect(scrub("key=sk-ant-abcdefghijklmnopqrstuv123")).toContain("[redacted:anthropic-key]");
    expect(scrub("token ghp_ABCDEFGHIJKLMNOPQRSTUV")).toContain("[redacted:github-token]");
    const code = "const sha = 'a94a8fe5ccb19ba61c4c0873d391e987982fbbd3';";
    expect(scrub(code)).toBe(code);
  });
  it("wires tool_result and input hooks", async () => {
    const m = mockPi();
    const mod = await import("../extensions/localcode-redact.ts");
    mod.default(m.api);
    expect(m.handlers.has("tool_result")).toBe(true);
    expect(m.handlers.has("input")).toBe(true);
  });
});

describe("web + app tools", () => {
  it("register web_search, web_fetch and launch_app", async () => {
    const m = mockPi();
    (await import("../extensions/localcode-web.ts")).default(m.api);
    (await import("../extensions/localcode-app.ts")).default(m.api);
    for (const t of ["web_search", "web_fetch", "launch_app"]) {
      expect(m.tools.has(t), `tool ${t}`).toBe(true);
    }
  });
});
