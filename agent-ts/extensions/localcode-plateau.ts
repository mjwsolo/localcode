/**
 * localcode's plateau detector for pi.
 *
 * HarnessBench 2026-09-08 (17 tasks, 27B model, 20 min cap): pi hit the cap on
 * 3 tasks. In each the model had reached its ceiling and kept re-editing the
 * same file, re-running the same failing check, or issuing empty web searches
 * until the clock ran out; its score at the cap equalled what a harness that
 * stopped early got. This extension ends that loop:
 *
 *  1. Progress accounting per tool round (one assistant turn that called tools).
 *     Progress = a write/edit to a path not yet touched this task, a
 *     build/test command that passes when the previous one failed (or on its
 *     first pass), a failing check whose FAILURE SET shrank or changed (see
 *     below), or a todo item becoming completed. Reads, greps, re-edits of
 *     already-edited files, checks that fail the same way again, web tools
 *     and experiments under the temp dir are not progress.
 *  2. After 6 consecutive no-progress rounds: ONE steer message telling the
 *     model to stop experimenting and deliver, NAMING the open plan items
 *     (from todo_write) when there is a plan. Counter reset.
 *  3. After 8 more: the run is aborted with an honest summary on stderr (and
 *     in the TUI). In `-p` print mode an aborted run exits 1. Exception: if
 *     the most recent check PASSED within the last 2 rounds the run is not
 *     aborted; the model is steered once to finish now and gets 3 more rounds.
 *  4. Identical-call breaker: the same tool with the same arguments, failing
 *     or empty twice in a row, is rejected on the 3rd+ call before it runs.
 *
 * Failure-set progress (HarnessBench 2026-09-09, task 087): a model in a
 * GENUINE debugging loop re-edits the same files while the test failures
 * change each run; that used to look identical to a stuck loop and was
 * aborted a round or two before it passed. Each failing check's output is
 * normalised (ANSI, directory parts of paths and bare numbers such as
 * durations, timestamps and line numbers dropped), its failure lines
 * extracted (pytest FAILED/ERROR, tsc "error TS", jest ✕/●, cargo/gcc
 * "error:", generic FAIL, Python *Error) and compared with the previous
 * check this task: fewer failures is progress; the same count with a failure
 * signature never seen this task is progress; a signature seen before is the
 * real loop and is not.
 *
 * Plus a short BOUNDED WORK rule in the system prompt (complements the plan
 * rule from localcode-todo.ts; deduplicated when both are loaded).
 */
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const NO_PROGRESS_NUDGE_AT = 6;
/** no-progress rounds allowed AFTER the wrap-up steer before the run is stopped */
export const NO_PROGRESS_HALT_AFTER_NUDGE = 8;
export const NO_PROGRESS_HALT_AT = NO_PROGRESS_NUDGE_AT + NO_PROGRESS_HALT_AFTER_NUDGE;
/** a check that passed this many rounds ago (or fewer) blocks the abort */
export const PASS_GRACE_ROUNDS = 2;
/** rounds granted after the "your check passed — finish now" steer */
export const FINISH_GRACE_ROUNDS = 3;
export const MIN_ROUNDS = 4;
export const REPEAT_REJECT_AT = 3;
const NUDGE_PREFIX = "SYSTEM:";

const NUDGE_HEAD = `${NUDGE_PREFIX} No new progress for ${NO_PROGRESS_NUDGE_AT} steps — you have reached the limit of this approach. Stop experimenting. `;

/** Generic wrap-up steer, used when the model recorded no plan. */
export const PLATEAU_NUDGE =
  NUDGE_HEAD + "Now: " +
  "(1) write every deliverable the user named that does not exist yet, using the best approach you have; " +
  "(2) where a requirement cannot be met, say so explicitly in the deliverable or README instead of retrying; " +
  "(3) run the project's own check ONCE; (4) finish.";

/** Wrap-up steer naming the open plan items (task 043: the generic wording was ignored). */
export function plateauNudge(openItems: string[]): string {
  if (!openItems.length) return PLATEAU_NUDGE;
  return NUDGE_HEAD +
    `Open plan items you must deliver now:\n${openItems.map((t) => `- ${t}`).join("\n")}\n` +
    "For each: write it now using the best approach you have; where a requirement cannot be met, say so explicitly in the deliverable or README instead of retrying. " +
    "Then run the project's own check ONCE and finish.";
}

export const FINISH_NUDGE = `${NUDGE_PREFIX} Your check passed — finish now: write any remaining deliverable and stop.`;

export const REPEAT_REJECTED = "REJECTED: repeated identical call; use a materially different tool or arguments";

/** Full rule (when localcode-todo.ts is not loaded) and the complement added when it is. */
export const BOUNDED_RULE = `
BOUNDED WORK:
- Every deliverable the user named is its own plan item (a README is a plan item, not a closing flourish).
- Verification is BOUNDED: run the project's own build/typecheck/tests once and one smoke check. Do not build test rigs, install browsers, or write throwaway experiments unless the task asks for tests.
- If a requirement cannot be met after two attempts, state that plainly in the deliverable and move to the next item.`;
export const BOUNDED_RULE_COMPLEMENT = `
- Do not write throwaway experiments. If a requirement cannot be met after two attempts, state that plainly in the deliverable and move to the next item.`;

/** Build / test / typecheck commands whose pass-after-fail counts as progress. */
export const CHECK_CMD_RE =
  /\b(npm|pnpm|yarn|bun)\s+(run\s+)?(test|build|typecheck|lint|check)\b|\b(npx\s+)?(tsc|vitest|jest|pyright|mypy|pytest|eslint)\b|\bcargo\s+(build|test|check|clippy)\b|\bgo\s+(build|test|vet)\b|\bpython3?\s+-m\s+(pytest|unittest|compileall|mypy)\b|\bmake\s*(test|build|check)?\s*$/i;

const EMPTY_RESULT_RE = /^\s*(no results|no matches|nothing found|not found|\[\]|\{\})\b/i;

export function stableKey(toolName: string, input: unknown): string {
  const sort = (v: any): any =>
    Array.isArray(v) ? v.map(sort)
      : v && typeof v === "object" ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sort(v[k])]))
      : v;
  return `${toolName}:${JSON.stringify(sort(input ?? {}))}`;
}

export function resultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((c: any) => (c?.type === "text" ? c.text ?? "" : "")).join("\n");
}

/** A tool result that gave the model nothing to work with. */
export function isEmptyOrFailed(isError: boolean, content: unknown): boolean {
  if (isError) return true;
  const text = resultText(content).trim();
  return text.length === 0 || EMPTY_RESULT_RE.test(text);
}

/** Drop ANSI codes, directory parts of paths and bare numbers (durations, timestamps, line numbers, counts). */
export function normaliseCheckOutput(text: string): string {
  return text
    .replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "")
    .replace(/(?:[A-Za-z]:)?(?:\.{1,2}\/|\/)?(?:[\w.@+~-]+\/)+(?=[\w.-])/g, "")
    .replace(/(?<![A-Za-z_\d])\d+(?:\.\d+)?/g, "N");
}

const PRIMARY_FAILURE_RE = /^\s*(?:FAIL(?:ED)?\b|ERROR\b|error(?:\[E\d+\])?:|[✕✖×]|●)|\berror TS\d+\b/;
const SECONDARY_FAILURE_RE = /\b\w*Error\b/;

/**
 * Failure signatures of one check run: normalised, deduplicated failure lines.
 * Summary-style lines (pytest FAILED/ERROR, jest FAIL/✕/●, tsc "error TS", cargo/gcc
 * "error:") are preferred; bare *Error mentions (Python tracebacks) are used only when
 * there are none, so traceback verbosity does not change the count. A failing run
 * with no recognisable failure line yields one signature: its normalised tail.
 */
export function extractFailures(text: string, failed = true): string[] {
  const lines = normaliseCheckOutput(text).split("\n").map((l) => l.trimEnd()).filter((l) => l.trim());
  let hits = lines.filter((l) => PRIMARY_FAILURE_RE.test(l));
  if (!hits.length) hits = lines.filter((l) => SECONDARY_FAILURE_RE.test(l));
  if (!hits.length && failed) hits = [lines.slice(-5).join(" | ") || "(no output)"];
  return [...new Set(hits.map((l) => l.trim()))].sort();
}

export function isScratchPath(p: string): boolean {
  const tmp = tmpdir();
  return p.startsWith("/tmp/") || p.startsWith("/private/tmp/") || (tmp.length > 1 && p.startsWith(tmp + "/"));
}

type Todo = { content: string; status: string };

export default function (pi: ExtensionAPI) {
  const log = (msg: string) => { try { process.stderr.write(`[localcode plateau] ${msg}\n`); } catch {} };
  let cwd = process.cwd();

  // per-task state (a task = one genuine user turn plus the harness' own SYSTEM: continuations)
  let changedPaths = new Set<string>();
  let rounds = 0;
  let noProgress = 0;
  let roundHadProgress = false;
  let nudged = false;
  let halted = false;
  let noProgressTotal = 0;                        // rounds since the last progress, never reset by a steer
  let haltAt = NO_PROGRESS_NUDGE_AT;              // no-progress rounds until the next intervention
  let finishNudged = false;
  let lastCheckPassed: boolean | null = null;   // null: no build/test run yet this task
  let checkPassedThisRound = false;
  let lastPassRound: number | null = null;      // round index of the most recent passing check
  let lastFailureCount: number | null = null;   // failures in the previous check (0 for a pass)
  let seenFailures = new Set<string>();          // every failure signature seen this task
  let pendingChecks = new Set<string>();         // toolCallIds of in-flight build/test commands
  let maxCompletedTodos = 0;
  let lastTodos: Todo[] = [];
  // identical-call breaker
  let lastKey = "";
  let lastKeyFailures = 0;

  const reset = () => {
    changedPaths = new Set(); rounds = 0; noProgress = 0; roundHadProgress = false;
    noProgressTotal = 0; haltAt = NO_PROGRESS_NUDGE_AT; finishNudged = false;
    nudged = false; halted = false; lastCheckPassed = null; checkPassedThisRound = false;
    lastPassRound = null; lastFailureCount = null; seenFailures = new Set(); pendingChecks = new Set();
    maxCompletedTodos = 0; lastTodos = []; lastKey = ""; lastKeyFailures = 0;
  };
  const openItems = () => lastTodos.filter((t) => t.status !== "completed").map((t) => t.content);

  pi.on("before_agent_start", (event) => {
    if (!event.prompt.startsWith(NUDGE_PREFIX)) reset();
    const rule = /it is BOUNDED/.test(event.systemPrompt) ? BOUNDED_RULE_COMPLEMENT : BOUNDED_RULE;
    return { systemPrompt: event.systemPrompt + "\n" + rule };
  });

  pi.on("tool_call", (event, ctx) => {
    if (ctx && (ctx as any).cwd) cwd = (ctx as any).cwd;
    if (halted) {
      return { block: true, terminate: true, reason: "localcode stopped this run: no progress for too long. Summarise what was delivered and what was not." };
    }
    const input = (event.input ?? {}) as Record<string, unknown>;

    // identical-call breaker
    const key = stableKey(event.toolName, input);
    if (key === lastKey && lastKeyFailures >= REPEAT_REJECT_AT - 1) {
      log(`rejected repeated identical ${event.toolName} call (${lastKeyFailures} empty/failed before it)`);
      return { block: true, reason: REPEAT_REJECTED };
    }
    if (key !== lastKey) { lastKey = key; lastKeyFailures = 0; }

    if (event.toolName === "write" || event.toolName === "edit") {
      const p = resolve(cwd, String(input.path ?? input.file ?? ""));
      if (!isScratchPath(p) && !changedPaths.has(p)) { changedPaths.add(p); roundHadProgress = true; }
    } else if (event.toolName === "todo_write") {
      const todos = Array.isArray(input.todos) ? (input.todos as Todo[]) : [];
      lastTodos = todos;
      const done = todos.filter((t) => t?.status === "completed").length;
      if (done > maxCompletedTodos) { maxCompletedTodos = done; roundHadProgress = true; }
    } else if (event.toolName === "bash") {
      if (CHECK_CMD_RE.test(String(input.command ?? ""))) pendingChecks.add(event.toolCallId);
    }
  });

  pi.on("tool_result", (event) => {
    const failedOrEmpty = isEmptyOrFailed(event.isError, event.content);
    if (stableKey(event.toolName, event.input) === lastKey) lastKeyFailures = failedOrEmpty ? lastKeyFailures + 1 : 0;

    if (pendingChecks.has(event.toolCallId)) {
      pendingChecks.delete(event.toolCallId);
      const passed = !event.isError;
      if (passed) {
        if (lastCheckPassed !== true) roundHadProgress = true;   // first pass, or pass after failure
        checkPassedThisRound = true;
        lastFailureCount = 0;
      } else {
        // a failing check is progress when its failure set shrank, or is the same size but
        // contains a failure never seen this task; the same failures again are the real loop
        const sigs = extractFailures(resultText(event.content), true);
        const unseen = sigs.some((s) => !seenFailures.has(s));
        if (lastFailureCount !== null && lastFailureCount > 0) {
          if (sigs.length < lastFailureCount) roundHadProgress = true;
          else if (sigs.length === lastFailureCount && unseen) roundHadProgress = true;
        }
        for (const s of sigs) seenFailures.add(s);
        lastFailureCount = sigs.length;
      }
      lastCheckPassed = passed;
    }
  });

  pi.on("turn_end", (event, ctx) => {
    if (!event.toolResults?.length) return;   // a text-only turn is not a tool round
    rounds += 1;
    if (roundHadProgress) { noProgress = 0; noProgressTotal = 0; } else { noProgress += 1; noProgressTotal += 1; }
    if (checkPassedThisRound) lastPassRound = rounds;
    roundHadProgress = false;
    checkPassedThisRound = false;
    if (halted || rounds < MIN_ROUNDS || noProgress < haltAt) return;

    if (!nudged) {
      nudged = true;
      noProgress = 0;
      haltAt = NO_PROGRESS_HALT_AFTER_NUDGE;
      const open = openItems();
      log(`no progress for ${NO_PROGRESS_NUDGE_AT} rounds — telling the model to deliver and finish${open.length ? ` (${open.length} open plan items)` : ""}`);
      pi.sendUserMessage(plateauNudge(open), { deliverAs: "steer" });
      return;
    }

    // the check passed a moment ago: the model is finishing, not looping — never abort here
    const recentPass = lastCheckPassed === true && lastPassRound !== null && rounds - lastPassRound < PASS_GRACE_ROUNDS;
    if (recentPass && !finishNudged) {
      finishNudged = true;
      noProgress = 0;
      haltAt = FINISH_GRACE_ROUNDS;
      log(`check passed ${rounds - lastPassRound!} round(s) ago — telling the model to finish now (${FINISH_GRACE_ROUNDS} more rounds)`);
      pi.sendUserMessage(FINISH_NUDGE, { deliverAs: "steer" });
      return;
    }

    halted = true;
    const rel = (p: string) => (p.startsWith(cwd + "/") ? p.slice(cwd.length + 1) : p);
    const delivered = [...changedPaths].map(rel);
    const open = openItems();
    const summary = [
      `localcode stopped this run: no new progress for ${noProgressTotal} tool rounds (${rounds} total), even after being told to deliver and finish.`,
      delivered.length ? `Files written or edited (${delivered.length}): ${delivered.join(", ")}` : "No files were written or edited.",
      open.length ? `Plan items still open (${open.length}): ${open.join("; ")}` : (lastTodos.length ? "Every plan item was marked completed." : "No plan was recorded."),
      lastCheckPassed === null ? "The project's own build/test was never run." : `Last build/test run: ${lastCheckPassed ? "passed" : "FAILED"}.`,
      "Treat the result as partial: check the delivered files against the request before relying on them.",
    ].join("\n");
    log(summary);
    try { if (ctx?.hasUI) ctx.ui.notify(summary, "warning"); } catch {}
    try { ctx.abort(); } catch (e) { log(`abort failed: ${e}`); }
  });
}
