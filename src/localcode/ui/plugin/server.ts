/**
 * The opencode v2 server entry of LocalCode's plugin directory. It wires the
 * completion-discipline engine (`localcode.ts`) to the runtime:
 *
 *   engine handler                          v2 runtime
 *   ─────────────────────────────────────   ─────────────────────────────────────────────
 *   experimental.chat.system.transform  ->  session.hook("context" | "compaction" | "generate"): event.system
 *   experimental.chat.messages.transform ->  session.hook("context"): event.messages
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
import { leanMessages } from "./context";
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
      description: "The updated todo list",
      items: {
        type: "object",
        properties: {
          content: { type: "string", description: "Brief description of the task" },
          status: { type: "string", enum: [...STATUSES], description: "Current status of the task: pending, in_progress, completed, cancelled" },
          priority: { type: "string", enum: ["high", "medium", "low"], description: "Priority level of the task" },
          id: { type: "string", description: "Unique identifier for the todo item" },
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

export const TODO_DESCRIPTION =
  "Create and update the task checklist for the current session. Use it for work with three or more steps: " +
  "write the whole list once, keep exactly one item in_progress, and mark items completed as soon as they are done. " +
  "Send the complete list every time; it replaces the previous one.";

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

    const shape = async (event: any, kind: "context" | "compaction" | "generate") => {
      adapter.setAgent(event.agent);
      event.system = await transformSystem(hooks, event.agent, event.system ?? []);
      let messages = withTurnContext(event.messages ?? []);
      const lean = { messages };
      await hooks["experimental.chat.messages.transform"]({}, lean);
      messages = lean.messages;
      if (kind === "compaction") {
        const output = { context: [] as string[] };
        await hooks["experimental.session.compacting"]({}, output);
        if (output.context.length) {
          const text = output.context.join("\n\n");
          const last = messages.at(-1);
          messages = last?.role === "user" && Array.isArray(last.content)
            ? [...messages.slice(0, -1), Object.assign(Object.create(Object.getPrototypeOf(last)), last, { content: [...last.content, { type: "text", text }] })]
            : [...messages, { role: "user", content: [{ type: "text", text }] }];
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

    await ctx.session.hook("prompt", async (event: any) => {
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
        for (const translated of engineEvents(event, adapter.agent())) await hooks.event({ event: translated }).catch(() => {});
      }
    })().catch(() => {});
    return () => controller.abort();
  },
};

export { leanMessages };
