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
    for (const hook of ["model_select", "session_start"]) {
      expect(m.handlers.has(hook), `hook ${hook}`).toBe(true);
    }
    // thinking is a SERVER switch now (scripts/server_cmd.py); no per-request hook
    expect(m.handlers.has("before_provider_request")).toBe(false);
  });

  it("server command switches hidden thinking off for every wire and model", async () => {
    const { execFileSync } = await import("node:child_process");
    const py = process.env.LOCALCODE_PY ?? `${process.env.HOME}/Desktop/Github/localcode/localcodevenv/bin/python`;
    const { readdirSync } = await import("node:fs");
    const dir = `${process.env.HOME}/.local/share/localcode/models`;
    const gguf = (() => { try { return readdirSync(dir).find((f) => f.endsWith(".gguf") && /^(gemma|Qwen)/.test(f)); } catch { return undefined; } })();
    if (!gguf) return; // no local models on this machine: nothing to assert
    const out = execFileSync(py, ["scripts/server_cmd.py", `${dir}/${gguf}`, "8123", "x"], { encoding: "utf8" });
    expect(out).toContain("--reasoning\noff");
    expect(out).toContain("--reasoning-budget\n0");
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

describe("localcode-app.ts bash guard (servers never hang the agent)", () => {
  it("blocks foreground dev servers, allows backgrounded ones, adds a default timeout", async () => {
    const m = mockPi();
    (await import("../extensions/localcode-app.ts")).default(m.api);
    const h = m.handlers.get("tool_call")![0];
    for (const command of ["npm run dev", "npm install && npm start", "npx vite --port 5173", "python3 -m http.server 8000"]) {
      const res = await h({ toolName: "bash", input: { command } }, {});
      expect(res?.block, command).toBe(true);
      expect(String(res?.reason)).toContain("launch_app");
    }
    for (const command of ["npm run build", "nohup npm run dev > app.log 2>&1 &", "npm test", "ls -la"]) {
      const input: Record<string, unknown> = { command };
      const res = await h({ toolName: "bash", input }, {});
      expect(res?.block, command).not.toBe(true);
      expect(input.timeout).toBe(600);
    }
    const kept: Record<string, unknown> = { command: "sleep 1", timeout: 5 };
    await h({ toolName: "bash", input: kept }, {});
    expect(kept.timeout).toBe(5);
  });
});

describe("localcode-todo.ts (plan + completion gates, ported from localcode's loop)", () => {
  it("registers todo_write, injects the planning rule, and hooks agent_end", async () => {
    const m = mockPi();
    (await import("../extensions/localcode-todo.ts")).default(m.api);
    expect(m.tools.has("todo_write")).toBe(true);
    for (const hook of ["before_agent_start", "tool_call", "agent_end"]) expect(m.handlers.has(hook), hook).toBe(true);
    const start = m.handlers.get("before_agent_start")![0];
    const out = start({ prompt: "build me an app", systemPrompt: "BASE" }, {});
    expect(out.systemPrompt).toContain("BASE");
    expect(out.systemPrompt).toContain("todo_write FIRST");
    expect(out.systemPrompt).toContain("no TODOs, stubs, placeholders");
  });
  it("pure gates: verify note, stub scan, project check detection", async () => {
    const { verifyNote, stubLines, projectCheck } = await import("../extensions/localcode-todo.ts");
    const done = (c: string) => ({ content: c, status: "completed" as const });
    expect(verifyNote([done("a"), done("b"), done("c")])).toContain("verification step");
    expect(verifyNote([done("a"), done("b"), done("run the tests")])).toBe("");
    expect(verifyNote([done("a"), done("b")])).toBe("");
    const { writeFileSync, mkdtempSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const { join } = await import("node:path");
    const d = mkdtempSync(join(tmpdir(), "lc-todo-"));
    writeFileSync(join(d, "a.ts"), "export const x = 1; // TODO: implement for real\n");
    writeFileSync(join(d, "b.ts"), "export const y = 2;\n");
    const hits = stubLines([join(d, "a.ts"), join(d, "b.ts")]);
    expect(hits.length).toBe(1);
    expect(hits[0]).toContain("a.ts");
    expect(projectCheck(d)).toBe(null);
    writeFileSync(join(d, "package.json"), JSON.stringify({ scripts: { build: "vite build" } }));
    expect(projectCheck(d)).toBe(null); // no node_modules yet: nothing runnable
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

describe("localcode-nav.ts (structural search)", () => {
  it("registers code_navigation and finds real symbols", async () => {
    const m = (await import("vitest")).expect && (() => {
      const handlers = new Map(); const tools = new Map<string, any>();
      return { api: { on(){}, registerTool: (t: any) => tools.set(t.name, t) } as any, tools };
    })();
    (await import("../extensions/localcode-nav.ts")).default(m.api);
    const tool = m.tools.get("code_navigation");
    expect(tool).toBeTruthy();
    const res = await tool.execute("t", { action: "definition", symbol: "scrub", path: "extensions" },
      undefined, undefined, { cwd: process.cwd() } as any);
    expect(res.content[0].text).toMatch(/localcode-redact\.ts:\d+: function scrub/);
  });
});
