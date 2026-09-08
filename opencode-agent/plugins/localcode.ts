/**
 * localcode's completion discipline for OpenCode, as a plugin (no fork work).
 * Same rules as localcode's own loop and the pi/codex ports:
 *
 *  1. Plan first: the planning rule goes into the system prompt; OpenCode's
 *     built-in todowrite is the checklist, and open items are re-shown every turn.
 *  2. Open-todo gate: on session.idle with items still open, the session is
 *     prompted to continue with the next item (15 max, stops after 3 rounds
 *     with no progress).
 *  3. Build-verification gate: when files changed, the project's own
 *     typecheck/build/tests are run and errors are sent back (3x).
 *  4. Stub audit: files changed this turn (any tool) are scanned once for
 *     placeholders.
 *  5. Bash guard: foreground servers (npm run dev, vite, http.server...) are
 *     blocked with a reason; they hang the agent.
 *
 * Loaded from `.opencode/plugins/localcode.ts` in the project (the launcher and
 * the benchmark copy it there). Hooks only, no custom tools, so it needs no
 * node_modules in the project.
 */
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import type { Plugin } from "@opencode-ai/plugin";

type Todo = { content: string; status: string };

const MAX_TODO_CONTINUATIONS = 15;
const MAX_TODO_STUCK = 3;
const MAX_BUILD_VERIFY = 3;
const NUDGE_PREFIX = "SYSTEM:";

const PLANNING_RULE = `FINISH THE WHOLE TASK (most important):
- Keep going until the user's request is COMPLETELY done. Do not end your turn while any part of the work remains. A dev server that starts, a scaffold that installs, a single file written — none of these is "done" unless that was the entire request.
- PLAN, THEN EXECUTE THE PLAN. For any real multi-step task, call todowrite FIRST to lay out every concrete step (one per requirement, and every deliverable the user named — a README is a plan item like any feature, not a closing flourish). Skip the plan for one/two-step tasks. Keep exactly ONE item in_progress.
- The last plan item is verification, and it is BOUNDED: run the project's own build/typecheck once and one smoke check of the main flow. Do not install browsers, write test frameworks or build test rigs unless the task asks for tests.
- Write complete, runnable code — no TODOs, stubs, placeholders, "demo only" or "you could add…". If a piece is too big for one call, split it across calls; never drop it.
- Only stop for one of two reasons: (a) every todo is completed and verified, or (b) you have ONE specific blocking question you cannot answer yourself. The harness sends you back to the next open item if you stop early.
- Never run a foreground server (npm run dev, vite, http.server) through bash: start it in the background with nohup ... & and a log file, then curl it.`;

const SERVER_CMD = /\b(npm|pnpm|yarn|bun)\s+(run\s+)?(dev|start|serve|preview)\b|\bvite\b(?!\s+build)|\bnext\s+(dev|start)\b|python3?\s+-m\s+http\.server|\buvicorn\b|\bflask\s+run\b|\bgunicorn\b|\bnodemon\b|\bng\s+serve\b/i;
const BACKGROUNDED = /&\s*$|\bnohup\b|\bsetsid\b|\bdisown\b|\btimeout\s+\d/;
const STUB_RE = /(?:^|\W)(TODO:?|FIXME|placeholder|stub|not implemented|demo only|demo-only|coming soon)(?:\W|$)/i;
const SKIP_DIRS = new Set(["node_modules", ".git", "dist", "build", "target", ".venv", "venv", "__pycache__", ".opencode"]);
const SRC_RE = /\.(ts|tsx|js|jsx|py|rs|go|java|kt|swift|rb|php|vue|svelte)$/;

function projectCheck(cwd: string): string[] | null {
  const pkg = join(cwd, "package.json");
  if (existsSync(pkg)) {
    if (!existsSync(join(cwd, "node_modules"))) return null;
    if (existsSync(join(cwd, "tsconfig.json"))) return ["npx", "tsc", "--noEmit", "-p", "tsconfig.json"];
    try { if (JSON.parse(readFileSync(pkg, "utf8")).scripts?.build) return ["npm", "run", "build"]; } catch {}
    return null;
  }
  if (existsSync(join(cwd, "tests"))) return ["python3", "-m", "pytest", "-q", "-x"];
  if (existsSync(join(cwd, "pyproject.toml")) || existsSync(join(cwd, "setup.py"))) return ["python3", "-m", "compileall", "-q", "."];
  return null;
}

function runCheck(cwd: string, cmd: string[]): string | null {
  try {
    execFileSync(cmd[0], cmd.slice(1), { cwd, stdio: ["ignore", "pipe", "pipe"], timeout: 300_000, env: { ...process.env, CI: "1" } });
    return null;
  } catch (e: any) {
    if (e?.code === "ENOENT") return null;
    const out = `${e?.stdout ?? ""}${e?.stderr ?? ""}`;
    return out.split("\n").filter(Boolean).slice(-40).join("\n").slice(0, 4000) || `exit ${e?.status ?? "?"}`;
  }
}

function filesChangedSince(root: string, since: number): string[] {
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

function stubLines(files: string[]): string[] {
  const hits: string[] = [];
  for (const f of files) {
    let text = ""; try { text = readFileSync(f, "utf8"); } catch { continue; }
    for (const line of text.split("\n")) {
      if (STUB_RE.test(line)) hits.push(`${f}: ${line.trim().slice(0, 160)}`);
      if (hits.length >= 12) return hits;
    }
  }
  return hits;
}

function renderTodos(todos: Todo[]): string {
  const open = todos.filter((t) => t.status !== "completed" && t.status !== "cancelled");
  if (!open.length) return "";
  return "YOUR OPEN TODOS (from todowrite; the task is not complete until every one is completed):\n" +
    open.map((t) => `- [${t.status === "in_progress" ? ">" : " "}] ${t.content}`).join("\n");
}

export const LocalcodePlugin: Plugin = async ({ client, directory }) => {
  let todos: Todo[] = [];
  let continueCount = 0;
  let stuckCount = 0;
  let lastRemaining = Number.MAX_SAFE_INTEGER;
  let buildVerifyNudges = 0;
  let stubNudgeDone = false;
  let turnStartedAt = Date.now() - 1000;
  const log = (msg: string) => { try { process.stderr.write(`[localcode gate] ${msg}\n`); } catch {} };

  async function nudge(sessionID: string, text: string) {
    await client.session.prompt({ path: { id: sessionID }, body: { parts: [{ type: "text", text }] } });
  }

  return {
    "experimental.chat.system.transform": async (_input, output) => {
      output.system.push(PLANNING_RULE);
      const open = renderTodos(todos);
      if (open) output.system.push(open);
    },

    "chat.message": async (_input, output) => {
      const text = output.parts.map((p: any) => (p.type === "text" ? p.text : "")).join(" ");
      if (!text.startsWith(NUDGE_PREFIX)) {
        continueCount = 0; stuckCount = 0; lastRemaining = Number.MAX_SAFE_INTEGER;
        buildVerifyNudges = 0; stubNudgeDone = false;
        turnStartedAt = Date.now() - 1000;
      }
    },

    "tool.execute.before": async (input, output) => {
      if (input.tool !== "bash") return;
      const cmd = String(output.args?.command ?? "");
      if (SERVER_CMD.test(cmd) && !BACKGROUNDED.test(cmd)) {
        throw new Error(
          `That command starts a server that never exits, so the bash tool would hang. ` +
          `Run it in the background instead: nohup ${cmd} > app.log 2>&1 &  then curl the port.`,
        );
      }
    },

    event: async ({ event }) => {
      if (event.type === "todo.updated") {
        todos = (event.properties as any).todos ?? [];
        return;
      }
      if (event.type !== "session.idle") return;
      const sessionID = (event.properties as any).sessionID as string;

      const open = todos.filter((t) => t.status !== "completed" && t.status !== "cancelled");
      if (open.length && continueCount < MAX_TODO_CONTINUATIONS && stuckCount < MAX_TODO_STUCK) {
        if (open.length >= lastRemaining) stuckCount += 1; else stuckCount = 0;
        lastRemaining = open.length;
        if (stuckCount < MAX_TODO_STUCK) {
          continueCount += 1;
          const next = open.find((t) => t.status === "in_progress") ?? open[0];
          log(`${open.length} todo(s) still open — continuing with: ${next.content}`);
          await nudge(sessionID, `${NUDGE_PREFIX} You still have ${open.length} unfinished todo(s). The task is NOT complete — do not stop. Continue now with: ${next.content}. Mark a todo completed via todowrite only when it is genuinely done, and keep going until every item is completed.`);
          return;
        }
      }

      const changed = filesChangedSince(directory, turnStartedAt);
      if (!changed.length) return;

      if (!stubNudgeDone) {
        const stubs = stubLines(changed);
        if (stubs.length) {
          stubNudgeDone = true;
          log(`placeholders found in ${stubs.length} line(s) — sending back`);
          await nudge(sessionID, `${NUDGE_PREFIX} your changes still contain placeholders — the user asked for complete, working features, not stubs:\n${stubs.join("\n")}\nImplement each one for real (reopen it as a todo if needed), or tell the user explicitly which requirement you cannot meet and why.`);
          return;
        }
      }

      if (buildVerifyNudges < MAX_BUILD_VERIFY) {
        const cmd = projectCheck(directory);
        if (cmd) {
          const errors = runCheck(directory, cmd);
          if (errors) {
            buildVerifyNudges += 1;
            log(`${cmd.join(" ")} failed — sending errors back`);
            await nudge(sessionID, `${NUDGE_PREFIX} the project's typecheck/build (\`${cmd.join(" ")}\`) was run for you and reported errors. FIX each one with targeted edits, then finish. Do not claim it works until these are gone:\n\n${errors}`);
          }
        }
      }
    },
  };
};

export default LocalcodePlugin;
