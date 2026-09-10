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
 *  6. Plateau breaker: a "round" is one LLM step that executed tools (closed by
 *     the `step-finish` part OpenCode publishes on message.part.updated; each
 *     tool.execute.after is attributed to the open round; session.idle flushes
 *     a trailing round). Progress in a round = a write/edit to a path never
 *     changed this session, a project check that passes for the first time or
 *     after failing, a check whose failure set shrinks or shows a new failure
 *     signature, or a todowrite raising the completed count. From round 4 on,
 *     6 consecutive no-progress rounds nudge once ("stop experimenting, deliver
 *     the open items, run the check once, finish"); 8 more stop the session via
 *     client.session.abort and print a stderr partial summary. If the last check
 *     passed within 2 rounds the stop is replaced once by a "finish now" nudge
 *     and 3 more rounds. A stopped session also disables the idle gates.
 *
 * Loaded from `.opencode/plugins/localcode.ts` in the project (the launcher and
 * the benchmark copy it there). Hooks only, no custom tools, so it needs no
 * node_modules in the project.
 */
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
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

// A dev server is a PROGRAM being run, so only match in program position (start of
// command or after ; && || | ( or npx/bunx). Round 5 of the Anki bench died in 2 min
// because "\bvite\b" matched inside `npm create vite@latest`, `vite-plugin-pwa` and a
// grep pattern, and the guard kept blocking installs.
const PROG = String.raw`(?:^|[;&|(]\s*|\b(?:npx|bunx|pnpm\s+exec|yarn\s+exec)\s+)`;
const SERVER_CMD = new RegExp(
  PROG + String.raw`(?:(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|preview)\b` +
  String.raw`|vite(?:\s+(?:dev|serve|preview))?(?:\s|$)(?![\s\S]*\bbuild\b)` +
  String.raw`|next\s+(?:dev|start)\b|python3?\s+-m\s+http\.server|uvicorn\b|flask\s+run\b|gunicorn\b|nodemon\b|ng\s+serve\b)`,
  "i",
);
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

// ---------------------------------------------------------------------------
// Plateau breaker: pure parts (exported for tests).
// ---------------------------------------------------------------------------

export const PLATEAU_MIN_ROUND = 4;        // never act before this many tool rounds
export const PLATEAU_NUDGE_AFTER = 6;      // consecutive no-progress rounds -> nudge once
export const PLATEAU_STOP_AFTER = 8;       // further no-progress rounds after the nudge -> stop
export const PLATEAU_RECENT_PASS = 2;      // a check that passed within this many rounds blocks the stop...
export const PLATEAU_PASS_GRACE = 3;       // ...and grants this many more rounds after a "finish now" nudge

export type ToolEvent = { tool: string; args: any; output?: string; metadata?: any };
export type ProgressMemory = {
  changedPaths: Set<string>;
  failureSigs: Set<string>;
  lastFailureCount: number | null;
  lastCheckPassed: boolean | null; // null = no check run yet this session
  completedTodos: number;
};
export type PlateauDecision = "none" | "nudge" | "pass-nudge" | "stop";

export function newProgressMemory(): ProgressMemory {
  return { changedPaths: new Set(), failureSigs: new Set(), lastFailureCount: null, lastCheckPassed: null, completedTodos: 0 };
}

// Same program-position idea as SERVER_CMD: a check is a PROGRAM being run.
export const CHECK_CMD = new RegExp(
  PROG + String.raw`(?:(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|build|typecheck|type-check|lint|check)\b` +
  String.raw`|(?:npx\s+|bunx\s+)?(?:tsc|vitest|jest|pytest|pyright|mypy|eslint)\b` +
  String.raw`|python3?\s+-m\s+(?:pytest|mypy|pyright|unittest)\b` +
  String.raw`|cargo\s+(?:test|build|check|clippy)\b|go\s+(?:test|build|vet)\b)`,
  "i",
);
export function isCheckCommand(cmd: string): boolean { return CHECK_CMD.test(cmd); }

const TEMP_RE = /(?:^|\/)(?:tmp|private\/tmp|var\/folders|node_modules|\.localcode-agent|\.opencode|\.git)(?:\/|$)|^\/dev\//;
export function isTempPath(p: string): boolean {
  const norm = p.replace(/\\/g, "/");
  if (TEMP_RE.test(norm)) return true;
  try { const t = tmpdir().replace(/\\/g, "/"); if (t && norm.startsWith(t)) return true; } catch {}
  return false;
}

const EDIT_TOOLS = new Set(["edit", "write", "multiedit", "apply_patch", "patch"]);
export function editedPaths(ev: ToolEvent): string[] {
  if (!EDIT_TOOLS.has(ev.tool)) return [];
  const out: string[] = [];
  const a = ev.args ?? {};
  for (const p of [a.filePath, a.file_path, a.path]) if (typeof p === "string" && p) out.push(p);
  const files = ev.metadata?.files;
  if (Array.isArray(files)) for (const f of files) {
    const p = typeof f === "string" ? f : f?.filePath ?? f?.relativePath ?? f?.path;
    if (typeof p === "string" && p) out.push(p);
  }
  return out;
}

const ANSI_RE = /\[[0-9;]*[A-Za-z]/g;
const FAIL_LINE_RE = /FAILED|--- FAIL|\bFAIL\b|AssertionError|error TS\d+|\bError\b|\berror\b|✕|✗|●|panicked|Traceback/;
/** Order-independent, path/number-free failure signatures from a check's output. */
export function failureSignatures(output: string): string[] {
  const sigs = new Set<string>();
  for (const raw of String(output ?? "").replace(ANSI_RE, "").split("\n")) {
    const line = raw.trim();
    if (!line || !FAIL_LINE_RE.test(line)) continue;
    if (/^\s*at\s|^\s*File\s"/.test(line)) continue;
    const norm = line
      .replace(/(?:[A-Za-z]:)?(?:\.{0,2}\/)?(?:[\w.@-]+\/)+/g, "")  // strip directories, keep basename
      .replace(/0x[0-9a-f]+/gi, "N")
      .replace(/\d+(?:\.\d+)?/g, "N")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 200);
    if (norm) sigs.add(norm);
  }
  return [...sigs].sort();
}

/** Pass/fail of a check from the bash tool result: metadata.exit when present, else output heuristics. */
export function checkPassed(ev: ToolEvent): boolean {
  const exit = ev.metadata?.exit;
  if (typeof exit === "number") return exit === 0;
  const out = String(ev.output ?? "");
  const m = out.match(/<shell_metadata>[\s\S]*?exit(?:\s*code)?[:=\s]+(\d+)/i);
  if (m) return m[1] === "0";
  return failureSignatures(out).length === 0;
}

/**
 * Classify one tool call. Mutates `mem`. Returns a short reason when the call is
 * progress, null otherwise. Reads, greps, re-edits of known paths, checks that
 * pass again or fail identically are not progress.
 */
export function progressOf(ev: ToolEvent, mem: ProgressMemory): string | null {
  const paths = editedPaths(ev).filter((p) => !isTempPath(p));
  if (paths.length) {
    const fresh = paths.filter((p) => !mem.changedPaths.has(p));
    for (const p of paths) mem.changedPaths.add(p);
    return fresh.length ? `new file(s): ${fresh.join(", ")}` : null;
  }
  if (ev.tool === "todowrite") {
    const todos: Todo[] = Array.isArray(ev.args?.todos) ? ev.args.todos : [];
    const done = todos.filter((t) => t?.status === "completed").length;
    if (done > mem.completedTodos) { mem.completedTodos = done; return `todos completed: ${done}`; }
    return null;
  }
  if (ev.tool === "bash" && isCheckCommand(String(ev.args?.command ?? ""))) {
    if (checkPassed(ev)) {
      const first = mem.lastCheckPassed !== true;
      mem.lastCheckPassed = true;
      mem.lastFailureCount = 0;
      return first ? "check passed" : null;
    }
    mem.lastCheckPassed = false;
    const sigs = failureSignatures(ev.output ?? "");
    const unseen = sigs.filter((s) => !mem.failureSigs.has(s));
    const shrank = mem.lastFailureCount !== null && sigs.length < mem.lastFailureCount;
    for (const s of sigs) mem.failureSigs.add(s);
    mem.lastFailureCount = sigs.length;
    if (unseen.length) return `new failure signature(s): ${unseen.length}`;
    if (shrank) return `failure set shrank to ${sigs.length}`;
    return null;
  }
  return null;
}

/** Round bookkeeping and thresholds. One instance per session; reset on a genuine user message. */
export class PlateauTracker {
  mem = newProgressMemory();
  round = 0;                 // completed tool rounds
  noProgress = 0;            // consecutive no-progress rounds since the last reset
  nudged = false;
  passNudged = false;
  stopped = false;
  lastPassRound = -Infinity; // round number (1-based, the round it ran in) of the last passing check
  private open = false;      // a round is open once a tool ran in the current step
  private reasons: string[] = [];

  observe(ev: ToolEvent): void {
    if (this.stopped) return;
    this.open = true;
    const r = progressOf(ev, this.mem);
    if (r) this.reasons.push(r);
    if (ev.tool === "bash" && isCheckCommand(String(ev.args?.command ?? "")) && this.mem.lastCheckPassed) this.lastPassRound = this.round + 1;
  }

  /** Close the current round (a step-finish, or session.idle as a flush). Steps without tools are ignored. */
  endRound(): PlateauDecision {
    if (!this.open || this.stopped) return "none";
    this.open = false;
    this.round += 1;
    const progressed = this.reasons.length > 0;
    this.reasons = [];
    if (progressed) { this.noProgress = 0; return "none"; }
    this.noProgress += 1;
    if (this.round < PLATEAU_MIN_ROUND) return "none";
    if (!this.nudged) {
      if (this.noProgress >= PLATEAU_NUDGE_AFTER) { this.nudged = true; this.noProgress = 0; return "nudge"; }
      return "none";
    }
    if (this.noProgress >= PLATEAU_STOP_AFTER) {
      if (!this.passNudged && this.round - this.lastPassRound <= PLATEAU_RECENT_PASS) {
        this.passNudged = true;
        this.noProgress = PLATEAU_STOP_AFTER - PLATEAU_PASS_GRACE;
        return "pass-nudge";
      }
      this.stopped = true;
      return "stop";
    }
    return "none";
  }
}

export function plateauNudgeText(open: Todo[]): string {
  const items = open.length ? `\nOpen plan items you must deliver now:\n${open.map((t) => `- ${t.content}`).join("\n")}` : "";
  return `${NUDGE_PREFIX} No new progress for ${PLATEAU_NUDGE_AFTER} steps — you have reached the limit of this approach. Stop experimenting.${items}\nWhere a requirement cannot be met, say so explicitly in the deliverable instead of retrying; run the project's own check ONCE; then finish.`;
}
export const PLATEAU_PASS_TEXT = `${NUDGE_PREFIX} Your check passed — finish now. Do not start another experiment: write up what is delivered and what is not, then stop.`;

export const LocalcodePlugin: Plugin = async ({ client, directory }) => {
  let todos: Todo[] = [];
  let continueCount = 0;
  let stuckCount = 0;
  let lastRemaining = Number.MAX_SAFE_INTEGER;
  let buildVerifyNudges = 0;
  let stubNudgeDone = false;
  let turnStartedAt = Date.now() - 1000;
  let plateau = new PlateauTracker();
  let plateauStopped = false;
  const seenSteps = new Set<string>();
  const log = (msg: string) => { try { process.stderr.write(`[localcode gate] ${msg}\n`); } catch {} };
  const plog = (msg: string) => { try { process.stderr.write(`[localcode plateau] ${msg}\n`); } catch {} };

  async function nudge(sessionID: string, text: string) {
    await client.session.prompt({ path: { id: sessionID }, body: { parts: [{ type: "text", text }] } });
  }
  // Mid-loop nudge: the prompt is inserted into history and picked up at the next
  // step, but the call resolves only when the whole loop ends, so never await it.
  function nudgeAsync(sessionID: string, text: string) {
    nudge(sessionID, text).catch((e) => plog(`nudge failed: ${e?.message ?? e}`));
  }

  const openTodos = () => todos.filter((t) => t.status !== "completed" && t.status !== "cancelled");

  async function onRoundEnd(sessionID: string): Promise<PlateauDecision> {
    const decision = plateau.endRound();
    if (decision === "none") return decision;
    const open = openTodos();
    if (decision === "nudge") {
      plog(`round ${plateau.round}: ${PLATEAU_NUDGE_AFTER} no-progress rounds — nudging to deliver`);
      nudgeAsync(sessionID, plateauNudgeText(open));
      return decision;
    }
    if (decision === "pass-nudge") {
      plog(`round ${plateau.round}: check passed recently — asking to finish, ${PLATEAU_PASS_GRACE} more rounds`);
      nudgeAsync(sessionID, PLATEAU_PASS_TEXT);
      return decision;
    }
    // stop
    plateauStopped = true;
    const files = [...plateau.mem.changedPaths];
    plog(`stopping session after ${plateau.round} tool rounds with no progress since the nudge; ` +
      `files delivered ${files.length}${files.length ? ` (${files.slice(0, 12).join(", ")}${files.length > 12 ? ", ..." : ""})` : ""}, ` +
      `open items ${open.length}${open.length ? `: ${open.map((t) => t.content).join("; ")}` : ""}, ` +
      `last check ${plateau.mem.lastCheckPassed === null ? "never run" : plateau.mem.lastCheckPassed ? "passed" : "failed"}; treat as partial`);
    try { await client.session.abort({ path: { id: sessionID } }); } catch (e: any) { plog(`abort failed: ${e?.message ?? e}`); }
    return decision;
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
        plateau = new PlateauTracker(); plateauStopped = false; seenSteps.clear();
      }
    },

    "tool.execute.after": async (input, output) => {
      plateau.observe({ tool: input.tool, args: input.args, output: output?.output, metadata: output?.metadata });
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
      if (event.type === "message.part.updated") {
        // Round boundary: OpenCode publishes one step-finish part per LLM step.
        const part = (event.properties as any).part;
        if (part?.type === "step-finish" && part.sessionID && !seenSteps.has(part.id)) {
          seenSteps.add(part.id);
          await onRoundEnd(part.sessionID);
        }
        return;
      }
      if (event.type !== "session.idle") return;
      const sessionID = (event.properties as any).sessionID as string;

      // Flush a trailing round if a step-finish never arrived; then, if the
      // plateau breaker stopped this session, none of the gates may restart it.
      if ((await onRoundEnd(sessionID)) !== "none" || plateauStopped) return;

      const open = openTodos();
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
