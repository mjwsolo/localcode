/**
 * localcode's completion discipline for pi, ported from localcode's own loop
 * (agent/loop.py, agent/recovery.py, agent/prompts.py):
 *
 *  1. `todo_write` — the model plans multi-step work as a checklist and keeps
 *     exactly one item in_progress. Open items are shown to the model on every
 *     turn (they survive compaction because they live here, not in history).
 *  2. Open-todo gate — when the agent stops while items are still open, it is
 *     sent back to the next item. Bounded: 15 continuations, and it gives up
 *     after 3 rounds where the open count stops falling.
 *  3. Verify-before-finish — closing a 3+ item list with no verification item
 *     gets a note in the tool result.
 *  4. Build-verification gate — when the run changed code, the project's own
 *     typecheck/build/tests are run FOR the model; errors are sent back (3x).
 *  5. Stub audit — files the run wrote are scanned once for placeholders.
 *
 * pi has no plan state of its own: on the Anki task it stopped with a broken
 * build every time until this existed.
 */
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

type Todo = { content: string; status: "pending" | "in_progress" | "completed" };

const MAX_TODO_CONTINUATIONS = 15;
const MAX_TODO_STUCK = 3;
const MAX_BUILD_VERIFY = 3;
const NUDGE_PREFIX = "SYSTEM:";

const PLANNING_RULE = `
FINISH THE WHOLE TASK (most important):
- Keep going until the user's request is COMPLETELY done. Do not end your turn while any part of the work remains. A dev server that starts, a scaffold that installs, a single file written — none of these is "done" unless that was the entire request.
- PLAN, THEN EXECUTE THE PLAN. For any real multi-step task, call todo_write FIRST to lay out every concrete step (one per requirement) — skip the plan for one/two-step tasks, never write a single-step plan. Keep exactly ONE item in_progress. The last item is always verification: run the project's build/typecheck/tests and exercise what you built.
- Write complete, runnable code — no TODOs, stubs, placeholders, "demo only" or "you could add…". If a piece is too big for one call, split it across calls; never drop it.
- Only stop for one of two reasons: (a) every todo is completed and verified, or (b) you have ONE specific blocking question you cannot answer yourself. The harness sends you back to the next open item if you stop early.
- Never run a foreground server (npm run dev, vite, http.server) through bash: use launch_app.`;

const VERIFY_RE = /verif|test|build|typecheck|smoke|run the app|check/i;
const STUB_RE = /(?:^|\W)(TODO:?|FIXME|placeholder|stub|not implemented|demo only|demo-only|coming soon)(?:\W|$)/i;

export function projectCheck(cwd: string): string[] | null {
  const pkg = join(cwd, "package.json");
  if (existsSync(pkg)) {
    if (!existsSync(join(cwd, "node_modules"))) return null;
    if (existsSync(join(cwd, "tsconfig.json"))) return ["npx", "tsc", "--noEmit", "-p", "tsconfig.json"];
    try {
      const scripts = JSON.parse(readFileSync(pkg, "utf8")).scripts ?? {};
      if (scripts.build) return ["npm", "run", "build"];
    } catch {}
    return null;
  }
  if (existsSync(join(cwd, "tests"))) return ["python3", "-m", "pytest", "-q", "-x"];
  if (existsSync(join(cwd, "pyproject.toml")) || existsSync(join(cwd, "setup.py"))) return ["python3", "-m", "compileall", "-q", "."];
  return null;
}

/** Run the project's check; returns the error tail, or null when it passes / cannot run. */
export function runCheck(cwd: string, cmd: string[]): string | null {
  try {
    execFileSync(cmd[0], cmd.slice(1), { cwd, stdio: ["ignore", "pipe", "pipe"], timeout: 300_000, env: { ...process.env, CI: "1" } });
    return null;
  } catch (e: any) {
    if (e?.code === "ENOENT") return null;
    const out = `${e?.stdout ?? ""}${e?.stderr ?? ""}`;
    const tail = out.split("\n").filter(Boolean).slice(-40).join("\n");
    return tail.slice(0, 4000) || `exit ${e?.status ?? "?"}`;
  }
}

export function stubLines(files: Iterable<string>): string[] {
  const hits: string[] = [];
  for (const f of files) {
    let text = "";
    try { text = readFileSync(f, "utf8"); } catch { continue; }
    if (!/\.(ts|tsx|js|jsx|py|rs|go|java|kt|swift|rb|php|vue|svelte)$/.test(f)) continue;
    for (const line of text.split("\n")) {
      if (STUB_RE.test(line)) hits.push(`${f}: ${line.trim().slice(0, 160)}`);
      if (hits.length >= 12) return hits;
    }
  }
  return hits;
}

const SKIP_DIRS = new Set(["node_modules", ".git", "dist", "build", "target", ".venv", "venv", "__pycache__"]);
const SRC_RE = /\.(ts|tsx|js|jsx|py|rs|go|java|kt|swift|rb|php|vue|svelte)$/;

/** Source files under `root` modified at/after `since` (ms). Bounded walk; catches files written via bash too. */
export function filesChangedSince(root: string, since: number): string[] {
  const out: string[] = [];
  const stack = [root];
  let seen = 0;
  while (stack.length && out.length < 200) {
    const dir = stack.pop()!;
    let entries: string[] = [];
    try { entries = readdirSync(dir); } catch { continue; }
    for (const name of entries) {
      if (++seen > 20000) return out;
      const p = join(dir, name);
      let st; try { st = statSync(p); } catch { continue; }
      if (st.isDirectory()) { if (!SKIP_DIRS.has(name) && !name.startsWith(".")) stack.push(p); continue; }
      if (SRC_RE.test(name) && st.mtimeMs >= since) out.push(p);
    }
  }
  return out;
}

export function verifyNote(todos: Todo[]): string {
  if (todos.length < 3 || !todos.every((t) => t.status === "completed")) return "";
  if (VERIFY_RE.test(todos.map((t) => t.content).join(" "))) return "";
  return "\n\nNOTE: you just marked a 3+ item task list fully done and none of the items was a verification step. Before you write your final summary, run the project's build/typecheck/tests (or a focused smoke check) to confirm the work actually runs, and re-read the user's original request: every requirement must be implemented for real — no stubs, placeholders or 'demo only' pieces. Reopen anything partial as a todo. Do not claim it works without verifying.";
}

export function renderTodos(todos: Todo[]): string {
  const open = todos.filter((t) => t.status !== "completed");
  if (!open.length) return "";
  return "\n\nYOUR OPEN TODOS (from todo_write; the task is not complete until every one is completed):\n" +
    open.map((t) => `- [${t.status === "in_progress" ? ">" : " "}] ${t.content}`).join("\n");
}

export default function (pi: ExtensionAPI) {
  let todos: Todo[] = [];
  const changedFiles = new Set<string>();
  let continueCount = 0;
  let stuckCount = 0;
  let lastRemaining = Number.MAX_SAFE_INTEGER;
  let buildVerifyNudges = 0;
  let stubNudgeDone = false;
  let cwd = process.cwd();
  let turnStartedAt = Date.now();
  const log = (msg: string) => { try { process.stderr.write(`[localcode gate] ${msg}\n`); } catch {} };

  const todoWrite = defineTool({
    name: "todo_write",
    label: "Todo list",
    description:
      "Plan multi-step work as a checklist and track it. Send the FULL list each time. " +
      "Statuses: pending, in_progress (exactly one), completed. The task is not done until every item is completed.",
    promptSnippet: "todo_write: plan multi-step work as a checklist; keep exactly one item in_progress",
    parameters: Type.Object({
      todos: Type.Array(Type.Object({
        content: Type.String({ description: "One concrete step" }),
        status: Type.Union([Type.Literal("pending"), Type.Literal("in_progress"), Type.Literal("completed")]),
      })),
    }),
    async execute(_id, params) {
      todos = (params.todos as Todo[]).filter((t) => t && typeof t.content === "string");
      const open = todos.filter((t) => t.status !== "completed").length;
      const text = `Todo list updated: ${todos.length} item(s), ${open} open.` + verifyNote(todos);
      if (open === 0) todos = [];  // classic semantics: cleared when everything is completed
      return { content: [{ type: "text", text }], details: { open } };
    },
  });
  pi.registerTool(todoWrite);

  pi.on("before_agent_start", (event) => {
    if (!event.prompt.startsWith(NUDGE_PREFIX)) {
      // a genuine new user turn: reset the per-turn bounds
      continueCount = 0; stuckCount = 0; lastRemaining = Number.MAX_SAFE_INTEGER;
      buildVerifyNudges = 0; stubNudgeDone = false; changedFiles.clear();
      turnStartedAt = Date.now() - 1000;
    }
    return { systemPrompt: event.systemPrompt + "\n" + PLANNING_RULE + renderTodos(todos) };
  });

  pi.on("tool_call", (event, ctx) => {
    if (ctx && (ctx as any).cwd) cwd = (ctx as any).cwd;
    if (event.toolName === "write" || event.toolName === "edit") {
      const input = event.input as Record<string, unknown>;
      const p = String(input.path ?? input.file ?? "");
      if (p) changedFiles.add(resolve(cwd, p));
    }
  });

  pi.on("agent_end", async (event) => {
    const last = event.messages[event.messages.length - 1] as any;
    const lastText = typeof last?.content === "string" ? last.content
      : Array.isArray(last?.content) ? last.content.map((c: any) => c.text ?? "").join(" ") : "";
    // a focused blocking question stands: the user gets to answer it
    const blockingQuestion = /\?\s*$/.test(lastText.trim()) && lastText.length < 600;

    // 2. open-todo gate
    const open = todos.filter((t) => t.status !== "completed");
    if (open.length && !blockingQuestion && continueCount < MAX_TODO_CONTINUATIONS && stuckCount < MAX_TODO_STUCK) {
      if (open.length >= lastRemaining) stuckCount += 1; else stuckCount = 0;
      lastRemaining = open.length;
      if (stuckCount < MAX_TODO_STUCK) {
        continueCount += 1;
        const next = open.find((t) => t.status === "in_progress") ?? open[0];
        log(`${open.length} todo(s) still open — continuing with: ${next.content}`);
        await pi.sendUserMessage(
          `${NUDGE_PREFIX} You still have ${open.length} unfinished todo(s). The task is NOT complete — do not stop. Continue now with: ${next.content}. Mark a todo completed via todo_write only when it is genuinely done, and keep going until every item is completed.`,
          { deliverAs: "followUp" },
        );
        return;
      }
    }

    for (const f of filesChangedSince(cwd, turnStartedAt)) changedFiles.add(f);
    if (!changedFiles.size) return;

    // 5. stub audit (once per turn)
    if (!stubNudgeDone) {
      const stubs = stubLines(changedFiles);
      if (stubs.length) {
        stubNudgeDone = true;
        log(`placeholders found in ${stubs.length} line(s) — sending back`);
        await pi.sendUserMessage(
          `${NUDGE_PREFIX} your changes still contain placeholders — the user asked for complete, working features, not stubs:\n${stubs.join("\n")}\nImplement each one for real (reopen it as a todo if needed), or tell the user explicitly which requirement you cannot meet and why.`,
          { deliverAs: "followUp" },
        );
        return;
      }
    }

    // 4. build-verification gate (bounded)
    if (buildVerifyNudges < MAX_BUILD_VERIFY) {
      const cmd = projectCheck(cwd);
      if (cmd) {
        const errors = runCheck(cwd, cmd);
        if (errors) {
          buildVerifyNudges += 1;
          log(`${cmd.join(" ")} failed — sending errors back`);
          await pi.sendUserMessage(
            `${NUDGE_PREFIX} the project's typecheck/build (\`${cmd.join(" ")}\`) was run for you and reported errors. FIX each one with targeted edits, then finish. Do not claim it works until these are gone:\n\n${errors}`,
            { deliverAs: "followUp" },
          );
        }
      }
    }
  });
}
