import { describe, expect, test } from "bun:test";
import {
  PLATEAU_MIN_ROUND, PLATEAU_NUDGE_AFTER, PLATEAU_STOP_AFTER, PLATEAU_PASS_GRACE,
  PlateauTracker, checkPassed, editedPaths, failureSignatures, isCheckCommand, isTempPath,
  newProgressMemory, plateauNudgeText, progressOf, type ToolEvent,
} from "../plugins/localcode";

const edit = (filePath: string): ToolEvent => ({ tool: "edit", args: { filePath, oldString: "a", newString: "b" } });
const write = (filePath: string): ToolEvent => ({ tool: "write", args: { filePath, content: "x" } });
const read = (filePath: string): ToolEvent => ({ tool: "read", args: { filePath } });
const grep = (): ToolEvent => ({ tool: "grep", args: { pattern: "x" } });
const check = (output: string, exit: number, command = "npm test"): ToolEvent => ({ tool: "bash", args: { command }, output, metadata: { exit } });
const todo = (done: number, open = 1): ToolEvent => ({
  tool: "todowrite",
  args: { todos: [...Array(done).fill({ content: "d", status: "completed" }), ...Array(open).fill({ content: "o", status: "pending" })] },
});

const FAIL_A = "\x1b[31m✕ adds numbers\x1b[0m\n  ● Calculator › adds numbers\n    AssertionError: expected 3 to equal 4\n      at Object.<anonymous> (/home/u/proj/src/calc.test.ts:12:5)";
const FAIL_B = "✕ divides\n  ● Calculator › divides\n    Error: division by zero\n";

describe("isCheckCommand", () => {
  test("recognises the project's own checks in program position", () => {
    for (const c of ["npm test", "npm run build", "pnpm typecheck", "yarn lint", "bun test", "npx tsc --noEmit", "tsc -p .", "vitest run",
      "jest src", "pytest -q", "python3 -m pytest tests", "pyright", "mypy .", "cargo test", "cargo check", "go test ./...", "cd app && npm run build"]) {
      expect(isCheckCommand(c)).toBe(true);
    }
  });
  test("ignores reads, installs and mentions", () => {
    for (const c of ["cat package.json", "npm install jest", "grep -r tsc src", "echo pytest", "ls tests", "git status", "npm run dev"]) {
      expect(isCheckCommand(c)).toBe(false);
    }
  });
});

describe("isTempPath / editedPaths", () => {
  test("temp dirs are excluded", () => {
    expect(isTempPath("/tmp/scratch.ts")).toBe(true);
    expect(isTempPath("/private/tmp/x/y.ts")).toBe(true);
    expect(isTempPath("/var/folders/ab/T/z.py")).toBe(true);
    expect(isTempPath("/proj/node_modules/x/index.js")).toBe(true);
    expect(isTempPath("/proj/.localcode-agent/plugins/localcode.ts")).toBe(true);
    expect(isTempPath("/proj/src/tmpfile.ts")).toBe(false);
    expect(isTempPath("/proj/src/app.ts")).toBe(false);
  });
  test("edit/write/apply_patch paths are extracted; reads are not edits", () => {
    expect(editedPaths(edit("/p/a.ts"))).toEqual(["/p/a.ts"]);
    expect(editedPaths(write("/p/b.ts"))).toEqual(["/p/b.ts"]);
    expect(editedPaths({ tool: "apply_patch", args: {}, metadata: { files: [{ filePath: "/p/c.ts" }, { filePath: "/p/d.ts" }] } })).toEqual(["/p/c.ts", "/p/d.ts"]);
    expect(editedPaths(read("/p/a.ts"))).toEqual([]);
  });
});

describe("failureSignatures", () => {
  test("strips ANSI, directories, numbers and stack frames; is order independent", () => {
    const a = failureSignatures(FAIL_A);
    expect(a.length).toBeGreaterThan(0);
    for (const s of a) { expect(s).not.toMatch(/\x1b/); expect(s).not.toMatch(/\/home\//); expect(s).not.toMatch(/\d/); }
    expect(a.some((s) => s.includes("AssertionError: expected N to equal N"))).toBe(true);
    const shuffled = FAIL_A.split("\n").reverse().join("\n");
    expect(failureSignatures(shuffled)).toEqual(a);
  });
  test("different line numbers give the same signature", () => {
    expect(failureSignatures("FAILED tests/test_x.py::test_a - AssertionError: 1 != 2"))
      .toEqual(failureSignatures("FAILED tests/test_x.py::test_a - AssertionError: 7 != 9"));
  });
  test("clean output has no signatures", () => {
    expect(failureSignatures("✓ 12 tests passed\nDone in 1.2s")).toEqual([]);
  });
});

describe("checkPassed", () => {
  test("uses the exit code when present, output otherwise", () => {
    expect(checkPassed(check("ok", 0))).toBe(true);
    expect(checkPassed(check("✓ all good", 1))).toBe(false);
    expect(checkPassed({ tool: "bash", args: { command: "npm test" }, output: "✓ all good" })).toBe(true);
    expect(checkPassed({ tool: "bash", args: { command: "npm test" }, output: FAIL_A })).toBe(false);
  });
});

describe("progressOf", () => {
  test("a new path is progress, re-editing it is not, temp paths never are", () => {
    const mem = newProgressMemory();
    expect(progressOf(edit("/p/a.ts"), mem)).toBeTruthy();
    expect(progressOf(edit("/p/a.ts"), mem)).toBeNull();
    expect(progressOf(write("/p/a.ts"), mem)).toBeNull();
    expect(progressOf(write("/p/b.ts"), mem)).toBeTruthy();
    expect(progressOf(write("/tmp/x.ts"), mem)).toBeNull();
  });
  test("reads and greps are never progress", () => {
    const mem = newProgressMemory();
    expect(progressOf(read("/p/a.ts"), mem)).toBeNull();
    expect(progressOf(grep(), mem)).toBeNull();
    expect(progressOf({ tool: "bash", args: { command: "ls -la" }, output: "x", metadata: { exit: 0 } }, mem)).toBeNull();
  });
  test("checks: first pass, pass-after-fail, shrinking or new failures are progress; identical failures and repeat passes are not", () => {
    const mem = newProgressMemory();
    expect(progressOf(check(FAIL_A, 1), mem)).toBeTruthy();          // never-seen failure
    expect(progressOf(check(FAIL_A, 1), mem)).toBeNull();            // identical failure
    expect(progressOf(check(FAIL_A + "\n" + FAIL_B, 1), mem)).toBeTruthy(); // new signature (grew, but new)
    expect(progressOf(check(FAIL_B, 1), mem)).toBeTruthy();          // failure set shrank
    expect(progressOf(check(FAIL_B, 1), mem)).toBeNull();
    expect(progressOf(check("all passed", 0), mem)).toBeTruthy();    // pass after failing
    expect(progressOf(check("all passed", 0), mem)).toBeNull();      // passing again
    expect(progressOf(check(FAIL_A, 1), mem)).toBeNull();            // old failure again after a pass: seen before, not smaller than 0? count rose, sig known
    expect(progressOf(check("ok", 0), mem)).toBeTruthy();            // pass after failing again
  });
  test("first-ever passing check is progress", () => {
    const mem = newProgressMemory();
    expect(progressOf(check("ok", 0), mem)).toBeTruthy();
    expect(progressOf(check("ok", 0), mem)).toBeNull();
  });
  test("todowrite is progress only when the completed count rises", () => {
    const mem = newProgressMemory();
    expect(progressOf(todo(0, 3), mem)).toBeNull();
    expect(progressOf(todo(1, 2), mem)).toBeTruthy();
    expect(progressOf(todo(1, 2), mem)).toBeNull();
    expect(progressOf(todo(1, 5), mem)).toBeNull();
    expect(progressOf(todo(2, 4), mem)).toBeTruthy();
  });
});

/** Run `n` no-progress tool rounds (reads, or re-edits of a known path), returning decisions. */
function stall(t: PlateauTracker, n: number, path = "/p/a.ts") {
  const out: string[] = [];
  for (let i = 0; i < n; i++) {
    t.observe(i % 2 ? read(path) : t.mem.changedPaths.has(path) ? edit(path) : grep());
    out.push(t.endRound());
  }
  return out;
}

describe("PlateauTracker thresholds", () => {
  test("steps without tools are not rounds", () => {
    const t = new PlateauTracker();
    expect(t.endRound()).toBe("none");
    expect(t.round).toBe(0);
  });
  test("progress rounds reset the counter", () => {
    const t = new PlateauTracker();
    t.observe(edit("/p/a.ts"));
    expect(t.endRound()).toBe("none");
    stall(t, 5);
    t.observe(write("/p/new.ts")); expect(t.endRound()).toBe("none");
    expect(t.noProgress).toBe(0);
    expect(stall(t, PLATEAU_NUDGE_AFTER - 1).every((d) => d === "none")).toBe(true);
  });
  test("never acts before round 4; nudges once at 6 consecutive no-progress rounds", () => {
    const t = new PlateauTracker();
    t.observe(edit("/p/a.ts")); t.endRound(); // round 1, progress
    const d = stall(t, PLATEAU_NUDGE_AFTER);
    expect(d.slice(0, -1).every((x) => x === "none")).toBe(true);
    expect(d.at(-1)).toBe("nudge");
    expect(t.round).toBe(1 + PLATEAU_NUDGE_AFTER);
    expect(t.round).toBeGreaterThanOrEqual(PLATEAU_MIN_ROUND);
  });
  test("a fresh tracker that stalls from round 1 still waits for the 6th no-progress round (>= round 4)", () => {
    const t = new PlateauTracker();
    const d = stall(t, PLATEAU_NUDGE_AFTER);
    expect(d.indexOf("nudge")).toBe(PLATEAU_NUDGE_AFTER - 1);
  });
  test("after the nudge, 8 further no-progress rounds stop; progress in between resets", () => {
    const t = new PlateauTracker();
    stall(t, PLATEAU_NUDGE_AFTER);
    expect(t.nudged).toBe(true);
    stall(t, 3);
    t.observe(write("/p/fresh.ts")); expect(t.endRound()).toBe("none");
    const d = stall(t, PLATEAU_STOP_AFTER);
    expect(d.slice(0, -1).every((x) => x === "none")).toBe(true);
    expect(d.at(-1)).toBe("stop");
    expect(t.stopped).toBe(true);
    // once stopped the tracker is inert
    t.observe(edit("/p/a.ts")); expect(t.endRound()).toBe("none");
  });
  test("no second nudge: after the first nudge the only outcome is stop", () => {
    const t = new PlateauTracker();
    stall(t, PLATEAU_NUDGE_AFTER);
    const d = stall(t, PLATEAU_STOP_AFTER);
    expect(d.filter((x) => x === "nudge")).toHaveLength(0);
    expect(d.at(-1)).toBe("stop");
  });
  test("a check that passed within the last 2 rounds converts the stop into a finish-now nudge and 3 more rounds", () => {
    const t = new PlateauTracker();
    stall(t, PLATEAU_NUDGE_AFTER);
    stall(t, PLATEAU_STOP_AFTER - 2);
    // round with a passing check: first pass this session -> progress, resets the counter
    t.observe(check("ok", 0)); expect(t.endRound()).toBe("none");
    expect(t.noProgress).toBe(0);
    // now stall again: 8 rounds, but the pass is old by then -> stop
    const t2 = new PlateauTracker();
    stall(t2, PLATEAU_NUDGE_AFTER);
    stall(t2, PLATEAU_STOP_AFTER - 1);
    // a repeat pass (not progress) in the 8th round: passed within 2 rounds -> pass-nudge instead of stop
    t2.mem.lastCheckPassed = true; // simulate an earlier pass so this one is not "progress"
    t2.observe(check("ok", 0));
    expect(t2.endRound()).toBe("pass-nudge");
    expect(t2.stopped).toBe(false);
    const d = stall(t2, PLATEAU_PASS_GRACE);
    expect(d.slice(0, -1).every((x) => x === "none")).toBe(true);
    expect(d.at(-1)).toBe("stop");
  });
  test("the pass-nudge is granted only once", () => {
    const t = new PlateauTracker();
    stall(t, PLATEAU_NUDGE_AFTER);
    stall(t, PLATEAU_STOP_AFTER - 1);
    t.mem.lastCheckPassed = true;
    t.observe(check("ok", 0)); expect(t.endRound()).toBe("pass-nudge");
    stall(t, PLATEAU_PASS_GRACE - 1);
    t.observe(check("ok", 0)); expect(t.endRound()).toBe("stop");
  });
});

describe("nudge wording", () => {
  test("includes the open items and the SYSTEM prefix", () => {
    const text = plateauNudgeText([{ content: "Write README", status: "pending" }, { content: "Add tests", status: "in_progress" }]);
    expect(text.startsWith("SYSTEM:")).toBe(true);
    expect(text).toContain(`No new progress for ${PLATEAU_NUDGE_AFTER} steps`);
    expect(text).toContain("- Write README");
    expect(text).toContain("- Add tests");
    expect(text).toContain("run the project's own check ONCE");
  });
  test("omits the items block when there are none", () => {
    expect(plateauNudgeText([])).not.toContain("Open plan items");
  });
});

// ---------------------------------------------------------------------------
// Hook wiring: drive the plugin with a stub client and OpenCode-shaped events.
// ---------------------------------------------------------------------------
import { LocalcodePlugin } from "../plugins/localcode";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function boot() {
  const prompts: string[] = [];
  const aborts: string[] = [];
  const client: any = {
    session: {
      prompt: async (o: any) => { prompts.push(o.body.parts[0].text); return {}; },
      abort: async (o: any) => { aborts.push(o.path.id); return {}; },
    },
  };
  const dir = mkdtempSync(join(tmpdir(), "plateau-"));
  const hooks: any = await LocalcodePlugin({ client, directory: dir } as any);
  const sid = "ses_1";
  let n = 0;
  const toolRound = async (ev: ToolEvent) => {
    await hooks["tool.execute.after"]({ tool: ev.tool, sessionID: sid, callID: `c${++n}`, args: ev.args }, { title: "", output: ev.output ?? "", metadata: ev.metadata ?? {} });
    await hooks.event({ event: { type: "message.part.updated", properties: { part: { id: `p${n}`, type: "step-finish", sessionID: sid, messageID: "m1", reason: "tool-calls" } } } });
  };
  const userMessage = async (text: string) => hooks["chat.message"]({ sessionID: sid }, { message: {}, parts: [{ type: "text", text }] });
  const idle = async () => hooks.event({ event: { type: "session.idle", properties: { sessionID: sid } } });
  const setTodos = async (todos: any[]) => hooks.event({ event: { type: "todo.updated", properties: { todos } } });
  return { hooks, prompts, aborts, sid, toolRound, userMessage, idle, setTodos };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

describe("plugin wiring", () => {
  test("nudges once mid-loop after 6 stalled rounds, with open items; then aborts after 8 more and silences the todo gate", async () => {
    const h = await boot();
    await h.userMessage("build me a thing");
    await h.setTodos([{ content: "Ship README", status: "pending" }, { content: "Done part", status: "completed" }]);
    await h.toolRound(write("/p/app.ts"));            // progress
    for (let i = 0; i < PLATEAU_NUDGE_AFTER; i++) await h.toolRound(edit("/p/app.ts"));
    await tick();
    expect(h.prompts).toHaveLength(1);
    expect(h.prompts[0]).toContain("No new progress for 6 steps");
    expect(h.prompts[0]).toContain("- Ship README");
    expect(h.prompts[0]).not.toContain("Done part");
    for (let i = 0; i < PLATEAU_STOP_AFTER - 1; i++) await h.toolRound(edit("/p/app.ts"));
    expect(h.aborts).toHaveLength(0);
    await h.toolRound(edit("/p/app.ts"));
    expect(h.aborts).toEqual([h.sid]);
    // the todo gate would normally re-prompt here (one todo still open) - it must not
    await h.idle();
    await tick();
    expect(h.prompts).toHaveLength(1);
    // a genuine new user message re-arms everything
    await h.userMessage("thanks, now do X");
    await h.idle();
    await tick();
    expect(h.prompts).toHaveLength(2); // todo gate active again
    expect(h.prompts[1]).toContain("unfinished todo");
  });
  test("duplicate step-finish events for the same part are one round; the SYSTEM nudge does not reset the tracker", async () => {
    const h = await boot();
    await h.userMessage("go");
    await h.hooks["tool.execute.after"]({ tool: "read", sessionID: h.sid, callID: "c", args: { filePath: "/p/a" } }, { title: "", output: "", metadata: {} });
    const part = { id: "dup", type: "step-finish", sessionID: h.sid, messageID: "m", reason: "tool-calls" };
    await h.hooks.event({ event: { type: "message.part.updated", properties: { part } } });
    await h.hooks.event({ event: { type: "message.part.updated", properties: { part } } });
    for (let i = 0; i < PLATEAU_NUDGE_AFTER - 2; i++) await h.toolRound(read("/p/a"));
    await tick();
    expect(h.prompts).toHaveLength(0);
    await h.userMessage("SYSTEM: some nudge");       // not a genuine user message
    await h.toolRound(read("/p/a"));
    await tick();
    expect(h.prompts).toHaveLength(1);
  });
  test("a recent passing check turns the stop into a finish-now nudge", async () => {
    const h = await boot();
    await h.userMessage("go");
    await h.toolRound(check("ok", 0));                 // first pass: progress
    for (let i = 0; i < PLATEAU_NUDGE_AFTER; i++) await h.toolRound(read("/p/a"));
    for (let i = 0; i < PLATEAU_STOP_AFTER - 1; i++) await h.toolRound(read("/p/a"));
    await h.toolRound(check("ok", 0));                 // repeat pass, no progress, 8th round
    await tick();
    expect(h.aborts).toHaveLength(0);
    expect(h.prompts).toHaveLength(2);
    expect(h.prompts[1]).toContain("Your check passed — finish now");
    for (let i = 0; i < PLATEAU_PASS_GRACE; i++) await h.toolRound(read("/p/a"));
    expect(h.aborts).toEqual([h.sid]);
  });
});
