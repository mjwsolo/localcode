import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Plugin from "../../src/localcode/ui/plugin/localcode";

// The snapshot rides on the session's FIRST user turn (history, cached, labelled as a
// snapshot); later turns get a change report; the system prompt carries neither.
test("snapshot on the first turn, change report later, system prompt untouched", async () => {
  const dir = mkdtempSync(join(tmpdir(), "layout-"));
  try {
    mkdirSync(join(dir, "app")); writeFileSync(join(dir, "app", "calc.py"), "");
    mkdirSync(join(dir, "tests")); writeFileSync(join(dir, "tests", "test_calc.py"), "");
    mkdirSync(join(dir, "node_modules", "x"), { recursive: true }); writeFileSync(join(dir, "node_modules", "x", "i.js"), "");
    const hooks = await (Plugin as any)({ directory: dir, client: {} });
    const sys = { system: [] as string[] }; await hooks["experimental.chat.system.transform"]({ agent: "build" }, sys);
    expect(sys.system.join("\n")).not.toContain("WORKSPACE SNAPSHOT");
    const t1 = { parts: [{ type: "text", text: "hello" }] as any[] };
    await hooks["chat.message"]({ sessionID: "s1", agent: "build", messageID: "m1" }, t1);
    const snap = t1.parts.find((p) => String(p.text).startsWith("WORKSPACE SNAPSHOT"));
    expect(snap).toBeDefined();
    expect(snap.text).toContain("app/"); expect(snap.text).toContain("  calc.py"); expect(snap.text).not.toContain("node_modules");
    // the user edits a file outside the session; the agent edits another via a tool
    writeFileSync(join(dir, "app", "calc.py"), "changed", { flush: true });
    writeFileSync(join(dir, "NEW.md"), "x");
    await hooks["tool.execute.after"]({ tool: "write", args: { filePath: join(dir, "tests", "agent.py") }, sessionID: "s1" }, { output: "ok", metadata: {} });
    writeFileSync(join(dir, "tests", "agent.py"), "by agent");
    const t2 = { parts: [{ type: "text", text: "next" }] as any[] };
    await hooks["chat.message"]({ sessionID: "s1", agent: "build", messageID: "m2" }, t2);
    expect(t2.parts.some((p) => String(p.text).startsWith("WORKSPACE SNAPSHOT"))).toBe(false);
    const rep = t2.parts.find((p) => String(p.text).startsWith("WORKSPACE CHANGED"));
    expect(rep).toBeDefined();
    expect(rep.text).toContain("added: NEW.md"); expect(rep.text).toContain("modified: app/calc.py"); expect(rep.text).not.toContain("agent.py");
    // nothing changed since -> no report
    const t3 = { parts: [{ type: "text", text: "again" }] as any[] };
    await hooks["chat.message"]({ sessionID: "s1", agent: "build", messageID: "m3" }, t3);
    expect(t3.parts.length).toBe(1);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});


// A second plugin instance (new process, same session) must not re-send the snapshot,
// must see the user's external change, and must not report the first instance's own edit.
test("snapshot state survives a process restart", async () => {
  const dir = mkdtempSync(join(tmpdir(), "layout2-"));
  try {
    mkdirSync(join(dir, "app")); writeFileSync(join(dir, "app", "calc.py"), "");
    const h1 = await (Plugin as any)({ directory: dir, client: {} });
    const t1 = { parts: [{ type: "text", text: "hello" }] as any[] };
    await h1["chat.message"]({ sessionID: "s9", agent: "build", messageID: "m1" }, t1);
    expect(t1.parts.some((p) => String(p.text).startsWith("WORKSPACE SNAPSHOT"))).toBe(true);
    await h1["tool.execute.after"]({ tool: "edit", args: { filePath: join(dir, "app", "calc.py") }, sessionID: "s9" }, { output: "ok", metadata: {} });
    writeFileSync(join(dir, "app", "calc.py"), "edited by agent", { flush: true });
    writeFileSync(join(dir, "USER.md"), "by user");
    const h2 = await (Plugin as any)({ directory: dir, client: {} });   // new process
    const t2 = { parts: [{ type: "text", text: "next" }] as any[] };
    await h2["chat.message"]({ sessionID: "s9", agent: "build", messageID: "m2" }, t2);
    expect(t2.parts.some((p) => String(p.text).startsWith("WORKSPACE SNAPSHOT"))).toBe(false);
    const rep = t2.parts.find((p) => String(p.text).startsWith("WORKSPACE CHANGED"));
    expect(rep).toBeDefined();
    expect(rep.text).toContain("added: USER.md"); expect(rep.text).not.toContain("calc.py");
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
