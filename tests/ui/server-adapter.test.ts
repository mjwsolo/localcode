import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import plugin, { engineEvents, engineTool, lowPowerModeOn, parseTodos, renderChecklist, transformSystem, withTurnContext } from "../../src/localcode/ui/plugin/server";
import LocalcodePluginDefault from "../../src/localcode/ui/plugin/localcode";

const { editedPaths } = LocalcodePluginDefault as any;
const directories: string[] = [];
afterAll(() => { for (const directory of directories) rmSync(directory, { recursive: true, force: true }); });

/** A fake opencode v2 plugin context recording what LocalCode registers and calls. */
function fakeRuntime() {
  const directory = mkdtempSync(join(tmpdir(), "localcode-adapter-"));
  directories.push(directory);
  writeFileSync(join(directory, "app.py"), "print('hi')\n");
  const sessionHooks = new Map<string, (event: any) => Promise<void> | void>();
  const toolHooks = new Map<string, (event: any) => Promise<void> | void>();
  const tools: any[] = [];
  const synthetic: any[] = [];
  const interrupts: any[] = [];
  const toasts: any[] = [];
  const queue: any[] = [];
  let wake: (() => void) | undefined;
  const ctx = {
    location: { directory },
    rpc: { async register(definition: any) {
      expect(definition.id).toBe("localcode");
      return { events: { async emit(name: string, data: any) { toasts.push({ name, data }); } } };
    } },
    tool: {
      async transform(fn: (editor: any) => void) { fn({ add: (tool: any) => tools.push(tool) }); },
      async hook(name: string, fn: any) { toolHooks.set(name, fn); },
    },
    session: {
      async hook(name: string, fn: any) { sessionHooks.set(name, fn); },
      async get() { return { agent: "build" }; },
      async synthetic(input: any) { synthetic.push(input); },
      async interrupt(input: any) { interrupts.push(input); },
    },
    event: {
      subscribe({ signal }: { signal: AbortSignal }) {
        return { async *[Symbol.asyncIterator]() {
          while (!signal.aborted) {
            if (queue.length) { yield queue.shift(); continue; }
            await new Promise<void>((resolve) => { wake = resolve; signal.addEventListener("abort", () => resolve(), { once: true }); });
          }
        } };
      },
    },
  };
  return {
    ctx, directory, tools, synthetic, interrupts, toasts, sessionHooks, toolHooks,
    publish(event: any) { queue.push(event); wake?.(); },
  };
}

const settle = async (until: () => boolean) => { for (let i = 0; i < 100 && !until(); i++) await Bun.sleep(5); };

test("v2 tool names map to the names the engine's gates were written for", () => {
  expect(engineTool("shell")).toBe("bash");
  expect(engineTool("patch")).toBe("apply_patch");
  expect(engineTool("read")).toBe("read");
  // v2 edit results list changed files under `file`.
  expect(editedPaths({ tool: "apply_patch", args: {}, metadata: { files: [{ file: "/p/a.ts" }] } })).toEqual(["/p/a.ts"]);
  expect(editedPaths({ tool: "write", args: { path: "/p/b.ts", content: "x" } })).toEqual(["/p/b.ts"]);
});

test("the todo tool validates its input and renders a checklist", () => {
  expect(parseTodos({ todos: [{ content: "plan", status: "completed", priority: "high" }, { content: "build", status: "in_progress", extra: 1 }] }))
    .toEqual([{ content: "plan", status: "completed", priority: "high" }, { content: "build", status: "in_progress" }]);
  expect(() => parseTodos({})).toThrow("todos array");
  expect(() => parseTodos({ todos: [{ content: "x", status: "done" }] })).toThrow("status must be one of");
  expect(() => parseTodos({ todos: [{ content: "", status: "pending" }] })).toThrow("content");
  expect(renderChecklist([{ content: "a", status: "completed" }, { content: "b", status: "in_progress" }, { content: "c", status: "pending" }, { content: "d", status: "cancelled" }]))
    .toBe("[x] a\n[>] b\n[ ] c\n[-] d");
});

test("turn context stored on a user message is added back as LocalCode text parts", () => {
  class Message { constructor(public role: string, public content: any[], public metadata?: any) {} }
  const messages = [new Message("user", [{ type: "text", text: "hi" }], { localcode: { context: ["WORKSPACE SNAPSHOT AT SESSION START\napp.py"] } }), new Message("assistant", [{ type: "text", text: "ok" }])];
  const result = withTurnContext(messages);
  expect(result[0]).toBeInstanceOf(Message);
  expect(result[0].content).toEqual([{ type: "text", text: "hi" }, { type: "text", text: "WORKSPACE SNAPSHOT AT SESSION START\napp.py", metadata: { localcode: true } }]);
  expect(result[1]).toBe(messages[1]);
  expect(messages[0].content).toHaveLength(1);
});

test("system parts become one merged system message", async () => {
  const hooks = { async "experimental.chat.system.transform"(_: any, output: { system: string[] }) { output.system.splice(0, output.system.length, output.system.join("\n") + "\nRULE"); } };
  expect(await transformSystem(hooks, "build", [{ type: "text", text: "a" }, { type: "text", text: "b" }])).toEqual([{ type: "text", text: "a\nb\nRULE" }]);
});

test("runtime events translate to the events the engine's gates read", () => {
  expect(engineEvents({ type: "session.step.ended", id: "e1", data: { sessionID: "s", finish: "length" } }, "build").map((e) => e.type)).toEqual(["message.updated", "message.part.updated"]);
  expect(engineEvents({ type: "session.step.ended", id: "e1", data: { sessionID: "s", finish: "length" } }, "build")[0].properties.info.finish).toBe("length");
  expect(engineEvents({ type: "session.execution.succeeded", data: { sessionID: "s" } }, "build")).toEqual([{ type: "session.idle", properties: { sessionID: "s" } }]);
  expect(engineEvents({ type: "session.execution.interrupted", data: { sessionID: "s", reason: "user" } }, "build")[0].properties.error.name).toBe("MessageAbortedError");
  expect(engineEvents({ type: "session.execution.interrupted", data: { sessionID: "s", reason: "shutdown" } }, "build")).toEqual([]);
  expect(engineEvents({ type: "session.updated", data: {} }, "build")).toEqual([]);
});

test("the server plugin registers the todo tool, carries turn context, guards tools and nudges on open todos", async () => {
  const runtime = fakeRuntime();
  const mode = process.env.LOCALCODE_CONTEXT;
  process.env.LOCALCODE_CONTEXT = "full"; // workspace snapshots are opt-in
  const dispose = await plugin.setup(runtime.ctx);
  try {
    expect(runtime.tools.map((tool) => tool.name)).toEqual(["todowrite"]);
    expect(runtime.tools[0].options).toEqual({ codemode: false });
    expect(runtime.tools[0].input.properties.todos.items.required).toEqual(["content", "status"]);
    for (const name of ["context", "compaction", "generate", "prompt"]) expect(runtime.sessionHooks.has(name)).toBe(true);

    // A user prompt: the workspace snapshot rides in the message metadata, not the system prompt.
    const prompt: any = { sessionID: "ses", messageID: "msg", prompt: { text: "build it" }, metadata: {} };
    await runtime.sessionHooks.get("prompt")!(prompt);
    const context: string[] = prompt.metadata.localcode?.context ?? [];
    expect(context.some((text) => text.startsWith("WORKSPACE SNAPSHOT AT SESSION START") && text.includes("app.py"))).toBe(true);

    // The request for that turn: system merged, snapshot added back to the user message.
    const request: any = {
      agent: "build",
      system: [{ type: "text", text: "You are a coding agent." }, { type: "text", text: "Environment" }],
      messages: [{ role: "user", content: [{ type: "text", text: "build it" }], metadata: prompt.metadata }],
      tools: { shell: { description: "Run a command" }, todowrite: { description: "todo" } },
    };
    await runtime.sessionHooks.get("context")!(request);
    expect(request.system).toHaveLength(1);
    expect(request.messages[0].content.some((part: any) => part.metadata?.localcode === true && part.text.startsWith("WORKSPACE SNAPSHOT"))).toBe(true);

    // Compaction requests carry LocalCode's checklist as its own labelled message, never
    // inside the user's turn (summaries recorded it as the user's words).
    const compaction: any = { agent: "build", system: [{ type: "text", text: "Summarize" }], messages: [{ role: "user", content: [{ type: "text", text: "build it" }] }], tools: {} };
    await runtime.sessionHooks.get("compaction")!(compaction);
    expect(compaction.messages).toHaveLength(2);
    expect(compaction.messages[0].content).toHaveLength(1);
    expect(compaction.messages[1].content[0].text).toStartWith("[Summary guidance from LocalCode, not from the user]");

    // A newline in a file path is refused before the tool runs.
    await expect(runtime.toolHooks.get("execute.before")!({ tool: "read", sessionID: "ses", input: { path: "app.py\nx" } })).rejects.toThrow("newline");

    // Two open todos, then a shell call and an idle session: the open-todo gate continues the turn.
    const result = await runtime.tools[0].execute({ todos: [{ content: "write app", status: "in_progress" }, { content: "test app", status: "pending" }] }, { sessionID: "ses" });
    expect(result.content).toContain("[>] write app");
    expect(result.content).toContain("2 open, 0 done.");
    expect(result.metadata.todos).toHaveLength(2);
    const shell = { tool: "shell", sessionID: "ses", input: { command: "ls" } };
    await runtime.toolHooks.get("execute.before")!(shell);
    await runtime.toolHooks.get("execute.after")!({ ...shell, status: "completed", result: { content: [{ type: "text", text: "app.py" }], metadata: { exit: 0 } } });
    runtime.publish({ type: "session.step.ended", id: "step-1", data: { sessionID: "ses", finish: "tool-calls" } });
    runtime.publish({ type: "session.execution.succeeded", data: { sessionID: "ses" } });
    await settle(() => runtime.synthetic.length > 0);
    expect(runtime.synthetic).toHaveLength(1);
    expect(runtime.synthetic[0]).toMatchObject({ sessionID: "ses", delivery: "steer", metadata: { localcode: { nudge: true } } });
    expect(runtime.synthetic[0].text).toContain("2 unfinished todo(s)");
    expect(runtime.synthetic[0].text).toContain("write app");
    await settle(() => runtime.toasts.length > 0);
    expect(runtime.toasts[0]).toMatchObject({ name: "toast", data: { title: "Continuing", variant: "info" } });
  } finally {
    await dispose();
    if (mode === undefined) delete process.env.LOCALCODE_CONTEXT;
    else process.env.LOCALCODE_CONTEXT = mode;
  }
});

test("old tool output is masked near the budget in one batch, saved to disk, and the decision survives a restart", async () => {
  const runtime = fakeRuntime();
  (runtime.ctx as any).model = { async list() { return { data: [{ providerID: "localcode", id: "m", limit: { context: 16384, input: 8192, output: 4096 } }] }; } };
  let dispose = await plugin.setup(runtime.ctx);
  const turn = (n: number, size = 8000) => [
    { role: "user", content: [{ type: "text", text: "turn " + n }] },
    { role: "assistant", content: [{ type: "tool-call", id: "c" + n, name: "read", input: { filePath: `f${n}.py` } }] },
    { role: "tool", content: [{ type: "tool-result", id: "c" + n, name: "read", result: { type: "text", value: `out${n} ` + "x".repeat(size) } }] },
  ];
  const request = (messages: any[]) => ({ sessionID: "ses", agent: "build", model: { providerID: "localcode", id: "m" }, system: [], tools: {}, messages });
  const values = (event: any) => event.messages.filter((m: any) => m.role === "tool").map((m: any) => m.content[0].result.value);
  const isStub = (v: string) => v.startsWith("[Old tool output masked");
  try {
    // Budget 8192: masking starts at ~5,734 tokens. Three outputs of ~2,000 tokens stay below it.
    const small: any = request([...turn(1), ...turn(2)]);
    await runtime.sessionHooks.get("context")!(small);
    expect(values(small).some(isStub)).toBe(false);
    // Past the trigger every output outside the protected window (2,048 tokens) is masked at once,
    // including ones from the current turn.
    const history = [...turn(1), ...turn(2), ...turn(3), ...turn(4)];
    const first: any = request(history);
    await runtime.sessionHooks.get("context")!(first);
    expect(values(first).map(isStub)).toEqual([true, true, true, false]);
    expect(values(first)[0]).toContain("read(filePath=f1.py)");
    expect(readFileSync(join(runtime.directory, ".localcode-agent/tool-out/c1.txt"), "utf8")).toStartWith("out1 ");
    // Growing a little does not edit history again.
    const more: any = request([...history, { role: "assistant", content: [{ type: "text", text: "ok" }] }]);
    await runtime.sessionHooks.get("context")!(more);
    expect(values(more).map(isStub)).toEqual([true, true, true, false]);
    // A new process (a headless run is one per turn) sends the identical prefix.
    await dispose();
    dispose = await plugin.setup(runtime.ctx);
    const again: any = request(history);
    await runtime.sessionHooks.get("context")!(again);
    expect(values(again)).toEqual(values(first));
    // Another provider has no measured budget: nothing is masked.
    const other: any = { ...request([...turn(1), ...turn(2), ...turn(3), ...turn(4)]), sessionID: "other", model: { providerID: "else", id: "x" } };
    await runtime.sessionHooks.get("context")!(other);
    expect(values(other).some(isStub)).toBe(false);
  } finally {
    await dispose();
  }
});

test("the step's cached and processed prompt tokens are logged per request", async () => {
  const runtime = fakeRuntime();
  const dispose = await plugin.setup(runtime.ctx);
  try {
    runtime.publish({ type: "session.step.ended", id: "s1", data: { sessionID: "ses", finish: "stop", tokens: { input: 250, output: 10, reasoning: 0, cache: { read: 750, write: 0 } } } });
    const file = join(runtime.directory, ".localcode-agent/localcode-plugin.log");
    await settle(() => { try { return readFileSync(file, "utf8").includes("[localcode cache]"); } catch { return false; } });
    expect(readFileSync(file, "utf8")).toContain("[localcode cache] prompt 1000 tokens: cached 750, processed 250 (hit 75%)");
  } finally {
    await dispose();
  }
});

test("Low Power Mode is read from pmset", () => {
  expect(lowPowerModeOn("System-wide power settings:\nCurrently in use:\n standby              1\n lowpowermode         1\n")).toBe(true);
  expect(lowPowerModeOn(" lowpowermode         0\n")).toBe(false);
  expect(lowPowerModeOn("Currently in use:\n sleep 1\n")).toBe(false);
});

test("open todos at a compaction ride at the checkpoint's tail, frozen for that checkpoint", async () => {
  const runtime = fakeRuntime();
  const dispose = await plugin.setup(runtime.ctx);
  try {
    const todo = runtime.tools.find((tool) => tool.name === "todowrite");
    await todo.execute({ todos: [{ content: "write app", status: "in_progress" }, { content: "plan", status: "completed" }] }, { sessionID: "ses" });
    const checkpoint = () => ({ id: "cmp1", role: "user", content: [{ type: "text", text: "<conversation-checkpoint>\nsummary\n</conversation-checkpoint>" }] });
    const request = () => ({ sessionID: "ses", agent: "build", system: [], tools: {}, messages: [checkpoint()] });
    const first: any = request();
    await runtime.sessionHooks.get("context")!(first);
    const tail = first.messages[0].content.at(-1).text;
    expect(tail).toContain("[>] write app");
    expect(tail).not.toContain("plan");
    // Later todo changes do not rewrite this checkpoint (the cached prefix stays the same).
    await todo.execute({ todos: [{ content: "write app", status: "completed" }] }, { sessionID: "ses" });
    const later: any = request();
    await runtime.sessionHooks.get("context")!(later);
    expect(later.messages[0].content.at(-1).text).toBe(tail);
  } finally {
    await dispose();
  }
});
