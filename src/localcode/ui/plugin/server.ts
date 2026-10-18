/**
 * The opencode v2 server entry of LocalCode's plugin directory. It wires the
 * completion-discipline engine (`localcode.ts`) to the runtime:
 *
 *   engine handler                          v2 runtime
 *   ─────────────────────────────────────   ─────────────────────────────────────────────
 *   experimental.chat.system.transform  ->  session.hook("context" | "compaction" | "generate"): event.system
 *   experimental.chat.messages.transform ->  session.hook("context"): event.messages
 *   (v1 runtime) old tool output pruning ->  session.hook("context"): observation masking near the
 *                                            budget, see maskCandidates (context.ts)
 *   tool.definition                     ->  session.hook("context"): event.tools[name].description
 *   experimental.session.compacting     ->  session.hook("compaction"): appended to the summary request
 *   chat.message                        ->  session.hook("prompt"); added parts ride in the user
 *                                            message's metadata and are added back to that message
 *                                            in every request (constant prefix, survives restarts)
 *   tool.execute.before / after         ->  tool.hook("execute.before" | "execute.after")
 *   event                               ->  event.subscribe(): step ended, execution ended/failed/
 *                                            interrupted, retry scheduled
 *   client.session.prompt (nudge)       ->  session.synthetic (steer delivery, wakes the session)
 *   client.session.abort                ->  session.interrupt
 *   client.tui.showToast                ->  an RPC event the terminal UI entry (`tui.ts`) shows
 *   todowrite (v1 built-in)             ->  a plugin tool with the same name and schema
 */
import LocalcodePlugin from "./localcode";
import { execFile } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { applyMasked, checkpointIndex, estimateRequest, leanMessages, maskCandidates, maskedOutputs, maskLimits, maskPath, withCheckpointTodos } from "./context";
import { LocalcodeRpc } from "./rpc";

/** Tool names the engine was written against (opencode v1) for the v2 tools that replaced them. */
const ENGINE_TOOL: Record<string, string> = { shell: "bash", patch: "apply_patch" };
export const engineTool = (name: string) => ENGINE_TOOL[name] ?? name;

/** Key under which turn context (workspace snapshot, change report, open todos) is stored on a user message. */
export const CONTEXT_KEY = "localcode";

export type Todo = { content: string; status: "pending" | "in_progress" | "completed" | "cancelled"; priority?: string; id?: string };

const STATUSES = ["pending", "in_progress", "completed", "cancelled"] as const;

/** The checklist tool's input, as JSON Schema (the same fields opencode v1's todowrite took). */
export const TodoInput = {
  type: "object",
  properties: {
    todos: {
      type: "array",
      description: "The complete, updated list",
      items: {
        type: "object",
        properties: {
          content: { type: "string", description: "Brief description of the task" },
          status: { type: "string", enum: [...STATUSES] },
          priority: { type: "string", enum: ["high", "medium", "low"] },
          id: { type: "string" },
        },
        required: ["content", "status"],
      },
    },
  },
  required: ["todos"],
} as const;

/** Validate the model's checklist; a malformed call fails with a message the model can act on. */
export function parseTodos(input: unknown): Todo[] {
  const todos = (input as { todos?: unknown } | null)?.todos;
  if (!Array.isArray(todos)) throw new Error("todowrite needs a todos array: [{content, status}]");
  return todos.map((item, index) => {
    const todo = item as Partial<Todo> | null;
    if (!todo || typeof todo.content !== "string" || !todo.content.trim()) throw new Error(`todos[${index}].content must be a non-empty string`);
    if (!STATUSES.includes(todo.status as Todo["status"])) throw new Error(`todos[${index}].status must be one of ${STATUSES.join(", ")}`);
    return {
      content: todo.content,
      status: todo.status as Todo["status"],
      ...(typeof todo.priority === "string" ? { priority: todo.priority } : {}),
      ...(typeof todo.id === "string" ? { id: todo.id } : {}),
    };
  });
}

/** The compact description LocalCode's runtime gave this tool under opencode v1. */
export const TODO_DESCRIPTION = [
  "Create and update the task list for this session.",
  "",
  "Use it for work with 3 or more distinct steps, or when the user gives several tasks. Skip it for a single edit, a single command, or a question.",
  "",
  "States: pending, in_progress (exactly one at a time), completed, cancelled.",
  "",
  "Rules:",
  "- Add one item per requirement, including every deliverable the user named.",
  "- Mark an item in_progress before starting it and completed only after the work and its verification are actually done.",
  "- Update as you go; do not batch completions.",
  "- If blocked, keep the item in_progress and add a follow-up describing the blocker.",
  "- Items are specific and actionable; keep the user's commands verbatim.",
  "- Send the complete list every time; it replaces the previous one.",
].join("\n");

export function renderChecklist(todos: readonly Todo[]): string {
  if (!todos.length) return "The todo list is empty.";
  const mark = (status: string) => status === "completed" ? "[x]" : status === "in_progress" ? "[>]" : status === "cancelled" ? "[-]" : "[ ]";
  return todos.map((todo) => `${mark(todo.status)} ${todo.content}`).join("\n");
}

/** Text of a tool result the way the engine saw v1 output. */
export function resultText(result: any): string {
  if (!result) return "";
  const content = result.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const text = content.filter((item: any) => item?.type === "text").map((item: any) => item.text).join("\n");
    if (text) return text;
  }
  if (result.output === undefined) return "";
  return typeof result.output === "string" ? result.output : JSON.stringify(result.output);
}

/** Turn context stored on user messages, added back as text parts the model reads with that message. */
export function withTurnContext(messages: any[]): any[] {
  return messages.map((message) => {
    const extra: unknown = message?.metadata?.[CONTEXT_KEY]?.context;
    if (message?.role !== "user" || !Array.isArray(extra) || !extra.length) return message;
    const parts = extra.filter((text): text is string => typeof text === "string")
      .map((text) => ({ type: "text", text, metadata: { localcode: true } }));
    const content = typeof message.content === "string" ? [{ type: "text", text: message.content }] : [...(message.content ?? [])];
    return Object.assign(Object.create(Object.getPrototypeOf(message)), message, { content: [...content, ...parts] });
  });
}

/** Apply the engine's system rules to the runtime's system parts: one merged system message. */
export async function transformSystem(hooks: any, agent: string | undefined, system: any[]): Promise<any[]> {
  const output = { system: system.map((part) => typeof part === "string" ? part : part?.text ?? "") };
  await hooks["experimental.chat.system.transform"]({ agent }, output);
  const first = system.find((part) => typeof part === "object" && part) ?? { type: "text" };
  return output.system.map((text: string) => ({ ...first, type: "text", text }));
}

type Ctx = any;

/** `pmset -g` lists `lowpowermode 1` while macOS Low Power Mode is on. */
export function lowPowerModeOn(pmset: string): boolean {
  return /^\s*lowpowermode\s+1\b/m.test(pmset);
}

export function createAdapter(ctx: Ctx) {
  const directory: string = ctx.location?.directory ?? process.cwd();
  let currentAgent: string | undefined;
  let rpc: { events: { emit: (name: "toast", data: any) => Promise<void> } } | undefined;
  const client = {
    session: {
      async prompt(input: { path: { id: string }; body: { parts: { text: string }[] } }) {
        const text = input.body.parts.map((part) => part.text).join("\n");
        await ctx.session.synthetic({
          sessionID: input.path.id,
          text,
          description: "LocalCode",
          metadata: { [CONTEXT_KEY]: { nudge: true } },
          delivery: "steer",
        });
      },
      async abort(input: { path: { id: string } }) {
        await ctx.session.interrupt({ sessionID: input.path.id });
      },
    },
    tui: {
      async showToast(input: { body: { title?: string; message: string; variant: string; duration?: number } }) {
        await rpc?.events.emit("toast", {
          message: input.body.message,
          variant: input.body.variant,
          ...(input.body.title ? { title: input.body.title } : {}),
          ...(input.body.duration ? { duration: input.body.duration } : {}),
        });
      },
    },
  };
  return {
    client,
    directory,
    setRpc(value: typeof rpc) { rpc = value; },
    agent: () => currentAgent,
    setAgent(value: string | undefined) { if (value) currentAgent = value; },
  };
}

/** Translate a v2 runtime event into the event shapes the engine handles. */
export function engineEvents(event: any, agent: string | undefined): { type: string; properties: any }[] {
  const data = event?.data ?? {};
  switch (event?.type) {
    case "session.step.ended":
      return [
        { type: "message.updated", properties: { info: { role: "assistant", sessionID: data.sessionID, agent: agent ?? "build", finish: data.finish } } },
        { type: "message.part.updated", properties: { part: { type: "step-finish", id: event.id ?? `${data.assistantMessageID}:${data.sessionID}`, sessionID: data.sessionID } } },
      ];
    case "session.execution.succeeded":
      return [{ type: "session.idle", properties: { sessionID: data.sessionID } }];
    case "session.execution.interrupted":
      return data.reason === "user" || data.reason === "superseded"
        ? [{ type: "session.error", properties: { sessionID: data.sessionID, error: { name: "MessageAbortedError" } } }]
        : [];
    case "session.execution.failed":
      return [{ type: "session.error", properties: { sessionID: data.sessionID, error: data.error } }];
    case "session.retry.scheduled":
      return [{ type: "session.status", properties: { sessionID: data.sessionID, status: { type: "retry" }, error: data.error } }];
    default:
      return [];
  }
}

export default {
  id: "localcode",
  async setup(ctx: Ctx) {
    const adapter = createAdapter(ctx);
    const hooks = await LocalcodePlugin({ client: adapter.client, directory: adapter.directory });
    const todos = new Map<string, Todo[]>();

    try {
      adapter.setRpc(await ctx.rpc.register(LocalcodeRpc.Definition, {}));
    } catch { /* A runtime without RPC still runs every gate; only notifications are lost. */ }

    // The checklist tool the planning rule and the open-todo gate rely on (v2 has none built in).
    await ctx.tool.transform((tools: any) => {
      tools.add({
        name: "todowrite",
        description: TODO_DESCRIPTION,
        input: TodoInput,
        options: { codemode: false },
        execute: async (input: unknown, context: { sessionID: string }) => {
          const list = parseTodos(input);
          todos.set(context.sessionID, list);
          await hooks.event({ event: { type: "todo.updated", properties: { sessionID: context.sessionID, todos: list } } });
          const open = list.filter((todo) => todo.status !== "completed" && todo.status !== "cancelled").length;
          return { content: `${renderChecklist(list)}\n\n${open} open, ${list.length - open} done.`, metadata: { todos: list } };
        },
      });
    });

    // Observation masking (context.ts): history is edited only when the prompt nears the budget,
    // in one large batch. Decisions and saved outputs live on disk per session, so a restarted
    // process (a headless run is one process per turn) sends the same prefix.
    const stateDir = join(adapter.directory, ".localcode-agent");
    const statePath = (sessionID: string) => join(stateDir, `context-${sessionID.replace(/[^A-Za-z0-9_-]/g, "_")}.json`);
    type ContextState = { masked: string[]; checkpoints: Record<string, string> };
    const states = new Map<string, { masked: Set<string>; checkpoints: Record<string, string> }>();
    const contextState = (sessionID: string) => {
      let state = states.get(sessionID);
      if (!state) {
        let saved: Partial<ContextState> = {};
        try { saved = JSON.parse(readFileSync(statePath(sessionID), "utf8")); } catch { /* new session */ }
        state = { masked: new Set(Array.isArray(saved.masked) ? saved.masked : []), checkpoints: saved.checkpoints && typeof saved.checkpoints === "object" ? saved.checkpoints : {} };
        states.set(sessionID, state);
      }
      return state;
    };
    const saveState = (sessionID: string) => {
      const state = states.get(sessionID);
      if (!state) return;
      try { mkdirSync(stateDir, { recursive: true }); writeFileSync(statePath(sessionID), JSON.stringify({ masked: [...state.masked], checkpoints: state.checkpoints })); } catch { /* best effort */ }
    };
    const logFile = join(stateDir, "localcode-plugin.log");
    const log = (line: string) => { try { mkdirSync(stateDir, { recursive: true }); appendFileSync(logFile, `${new Date().toISOString()} ${line}\n`); } catch { /* best effort */ } };

    // Estimates are calibrated by the server's own prompt counts: the last request's estimate
    // against the prompt tokens the step reported.
    const calibration = new Map<string, { estimate: number; ratio: number }>();
    const inputBudget = async (model: { providerID?: string; id?: string } | undefined) => {
      if (model?.providerID !== "localcode") return 0;
      try {
        const list = await ctx.model.list();
        const info = (list?.data ?? list ?? []).find((item: any) => item?.providerID === model.providerID && item?.id === model.id);
        const limit = info?.limit;
        return limit?.input || (limit?.context ? limit.context - (limit.output || 0) : 0);
      } catch {
        return 0;
      }
    };
    const mask = async (event: any, messages: any[], kind: string) => {
      const sessionID: string | undefined = event.sessionID;
      if (!sessionID) return undefined;
      const state = contextState(sessionID);
      if (kind !== "context") return state;
      const current = applyMasked(messages, state.masked);
      const raw = estimateRequest(current, event.system ?? [], event.tools ?? {});
      const cal = calibration.get(sessionID);
      const prompt = Math.round(raw * (cal?.ratio ?? 1));
      calibration.set(sessionID, { estimate: raw, ratio: cal?.ratio ?? 1 });
      const budget = await inputBudget(event.model);
      const ids = maskCandidates(messages, maskLimits(budget), state.masked, prompt);
      if (!ids.length) return state;
      // Save the full outputs first: the stub points the model at these files.
      for (const [id, text] of maskedOutputs(messages, ids)) {
        const file = join(adapter.directory, maskPath(id));
        try { if (!existsSync(file)) { mkdirSync(dirname(file), { recursive: true }); writeFileSync(file, text); } } catch { /* the stub still names the tool and arguments */ }
      }
      for (const id of ids) state.masked.add(id);
      saveState(sessionID);
      calibration.set(sessionID, { estimate: estimateRequest(applyMasked(messages, state.masked), event.system ?? [], event.tools ?? {}), ratio: cal?.ratio ?? 1 });
      log(`[localcode context] masked ${ids.length} old tool output(s): prompt ~${prompt} of budget ${budget} tokens`);
      return state;
    };

    /** Open todos at a compaction, frozen per checkpoint so the prefix after it stays the same. */
    const checkpointTodos = (sessionID: string | undefined, messages: any[]) => {
      if (!sessionID) return messages;
      const index = checkpointIndex(messages);
      if (index < 0) return messages;
      const id = String(messages[index]?.id ?? "checkpoint");
      const state = contextState(sessionID);
      if (!(id in state.checkpoints)) {
        const open = (todos.get(sessionID) ?? []).filter((todo) => todo.status !== "completed" && todo.status !== "cancelled");
        state.checkpoints[id] = open.length
          ? `Open todos when the conversation was compacted (keep them current with todowrite):\n${renderChecklist(open)}`
          : "";
        saveState(sessionID);
      }
      return withCheckpointTodos(messages, index, state.checkpoints[id]);
    };

    const shape = async (event: any, kind: "context" | "compaction" | "generate") => {
      adapter.setAgent(event.agent);
      event.system = await transformSystem(hooks, event.agent, event.system ?? []);
      let messages = checkpointTodos(event.sessionID, withTurnContext(event.messages ?? []));
      const state = await mask(event, messages, kind);
      const lean = { messages };
      await hooks["experimental.chat.messages.transform"]({}, lean);
      messages = state ? applyMasked(lean.messages, state.masked) : lean.messages;
      if (kind === "compaction") {
        const output = { context: [] as string[] };
        await hooks["experimental.session.compacting"]({}, output);
        if (output.context.length) {
          // Summarizer guidance rides as its own trailing message, so the cached
          // conversation prefix is reused. Appended to the user's last turn it read as
          // something the user said: summaries recorded it as user instructions, and a
          // weaker model summarized the checklist instead of the work.
          const text = output.context.join("\n\n");
          messages = [...messages, { role: "user", content: [{ type: "text", text: `[Summary guidance from LocalCode, not from the user] ${text}` }] }];
        }
      }
      event.messages = messages;
      for (const [name, tool] of Object.entries<any>(event.tools ?? {})) {
        const output = { description: tool.description };
        await hooks["tool.definition"]({ toolID: engineTool(name) }, output);
        tool.description = output.description;
      }
    };
    await ctx.session.hook("context", (event: any) => shape(event, "context"));
    await ctx.session.hook("compaction", (event: any) => shape(event, "compaction"));
    await ctx.session.hook("generate", (event: any) => shape(event, "generate"));

    // Per-request prefix-cache measurement: the server reports how much of each prompt it
    // served from cache (llama.cpp's cached_tokens); the rest it had to process again.
    const onStepUsage = (data: any) => {
      const tokens = data?.tokens;
      if (!tokens || typeof tokens.input !== "number") return;
      const cached = Number(tokens.cache?.read ?? 0) || 0;
      const processed = Number(tokens.input) || 0;
      const prompt = cached + processed + (Number(tokens.cache?.write ?? 0) || 0);
      if (prompt <= 0) return;
      const sessionID = data.sessionID;
      const cal = sessionID ? calibration.get(sessionID) : undefined;
      if (sessionID && cal && cal.estimate > 0) {
        calibration.set(sessionID, { estimate: cal.estimate, ratio: Math.min(3, Math.max(0.5, prompt / cal.estimate)) });
      }
      log(`[localcode cache] prompt ${prompt} tokens: cached ${cached}, processed ${processed} (hit ${Math.round((100 * cached) / prompt)}%)`);
    };

    // Low Power Mode roughly halves local generation speed; say so once, the first time a
    // prompt is sent while it is on.
    let lowPowerShown = false;
    const lowPowerCheck = () => {
      if (lowPowerShown || process.platform !== "darwin") return;
      execFile("pmset", ["-g"], { timeout: 1500 }, (error, stdout) => {
        if (error || lowPowerShown || !lowPowerModeOn(String(stdout))) return;
        lowPowerShown = true;
        void adapter.client.tui.showToast({ body: {
          title: "Low Power Mode is on",
          message: "Local models can run up to about 2x slower in Low Power Mode. Turn it off in System Settings > Battery for full speed.",
          variant: "warning",
          duration: 12000,
        } }).catch(() => {});
      });
    };

    await ctx.session.hook("prompt", async (event: any) => {
      lowPowerCheck();
      const session = await ctx.session.get({ sessionID: event.sessionID }).catch(() => undefined);
      const agent = session?.agent ?? session?.data?.agent;
      adapter.setAgent(agent);
      const parts = [{ type: "text", text: String(event.prompt?.text ?? "") }];
      await hooks["chat.message"]({ sessionID: event.sessionID, agent, messageID: event.messageID }, { parts });
      const added = parts.slice(1).map((part: any) => String(part.text ?? "")).filter(Boolean);
      if (added.length) event.metadata = { ...event.metadata, [CONTEXT_KEY]: { context: added } };
    });

    await ctx.tool.hook("execute.before", async (event: any) => {
      adapter.setAgent(event.agent);
      const output = { args: event.input && typeof event.input === "object" ? event.input : {} };
      await hooks["tool.execute.before"]({ tool: engineTool(event.tool), sessionID: event.sessionID }, output);
      event.input = output.args;
    });
    await ctx.tool.hook("execute.after", async (event: any) => {
      const result = event.status === "completed" ? event.result : { content: event.error?.message ?? "", metadata: { error: true } };
      await hooks["tool.execute.after"](
        { tool: engineTool(event.tool), args: event.input, sessionID: event.sessionID },
        { output: resultText(result), metadata: result?.metadata ?? {} },
      );
    });

    const controller = new AbortController();
    void (async () => {
      for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
        if (event?.type === "session.step.ended") onStepUsage(event.data ?? {});
        for (const translated of engineEvents(event, adapter.agent())) await hooks.event({ event: translated }).catch(() => {});
      }
    })().catch(() => {});
    return () => controller.abort();
  },
};

export { leanMessages };
