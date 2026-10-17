import { expect, spyOn, test } from "bun:test";
import { answerRows, requestFor, setupModes, summary, type ModesContext } from "../../src/localcode/ui/plugin/modes";
import tui from "../../src/localcode/ui/plugin/tui";

const form = { question: "Which team?", state: "Charged twice", kind: "choice" as const, options: "billing\nshipping", image: "" };

test("decision UI sends explicit labels through the typed protocol", () => {
  const request = requestFor(form);
  expect(request.questions.answer.type).toBe("choice");
  expect(request.questions.answer.criteria).toEqual({billing: null, shipping: null});
  expect(request).not.toHaveProperty("messages");
});

test("duplicate and missing choices cannot run", () => {
  expect(() => requestFor({...form, options: "billing\nbilling"})).toThrow();
  expect(() => requestFor({...form, options: "billing"})).toThrow();
  expect(() => requestFor({...form, question: ""})).toThrow();
});

test("probability rows have unique identities so the native list keeps every answer", () => {
  const rows=answerRows({model:"OpenJev",answers:{answer:{noul:0.8}}});
  expect(rows.map(r=>r.value)).toEqual(["result:yes","result:no"]);
  expect(rows.map(r=>r.footer)).toEqual(["80.0%","20.0%"]);
});

test("the Run decision row summarises the top answer", () => {
  expect(summary({model:"OpenJev",answers:{answer:{noul:0.014}}})).toBe("No 98.6%");
  expect(summary({model:"OpenJev",answers:{answer:{score:0.8,probabilities:{"0":0.2,"1":0.8},legend:{"0":"low","1":"high"}}}})).toBe("high 80.0%");
});

test("score results show the supplied level names", () => {
  const rows=answerRows({model:"OpenJev",answers:{answer:{score:0.8,probabilities:{"0":0.2,"1":0.8},legend:{"0":"low","1":"high"}}}});
  expect(rows.map(r=>r.title)).toEqual(["high","low"]);
});

// Run the plugin's registered commands against a fake opencode v2 TUI context with
// the real dialog semantics: showing a dialog replaces (and so closes) the previous
// one, closing settles its promise with undefined, choosing settles it with a value.
async function withTui(run: (host: ReturnType<typeof fakeTui>) => Promise<void> | void, controlUrl = "") {
  const host = fakeTui();
  const startup = process.env.LOCALCODE_MODE;
  const control = process.env.LOCALCODE_CONTROL_URL;
  delete process.env.LOCALCODE_MODE;
  process.env.LOCALCODE_CONTROL_URL = controlUrl;
  const dispose = setupModes(host.context);
  try {
    await run(host);
  } finally {
    dispose();
    if (startup === undefined) delete process.env.LOCALCODE_MODE;
    else process.env.LOCALCODE_MODE = startup;
    if (control === undefined) delete process.env.LOCALCODE_CONTROL_URL;
    else process.env.LOCALCODE_CONTROL_URL = control;
  }
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function fakeTui() {
  type Dialog =
    | { kind: "select"; title: string; options: readonly { value: unknown; title: string }[]; settle: (value: unknown) => void }
    | { kind: "prompt"; title: string; value?: string; settle: (value: unknown) => void }
    | { kind: "confirm"; title: string; message: string; settle: (value: unknown) => void };
  let dialog: Dialog | undefined;
  const commands = new Map<string, () => void | false | Promise<void>>();
  const slashes = new Map<string, string>();
  const dispatched: string[] = [];
  const toasts: { message: string; variant?: string }[] = [];
  const sizes: string[] = [];
  const show = <T,>(next: (settle: (value: T | undefined) => void) => Dialog) => new Promise<T | undefined>((resolve) => {
    const previous = dialog;
    dialog = undefined;
    previous?.settle(undefined);
    let settled = false;
    dialog = next((value) => { if (!settled) { settled = true; resolve(value); } });
  });
  const close = (value?: unknown) => {
    const current = dialog;
    dialog = undefined;
    current?.settle(value);
  };
  const context: ModesContext = {
    keymap: {
      layer(factory) {
        for (const command of factory().commands ?? []) {
          if (!command.id) continue;
          commands.set(command.id, command.run);
          // As the v2 keymap does it: only commands with a slash entry appear in the slash menu.
          if (command.slash) slashes.set(command.slash.name, command.id);
        }
      },
      dispatch(id) { dispatched.push(id); },
    },
    ui: {
      dialog: {
        set(options) { sizes.push(options.size ?? "medium"); },
        clear() { close(undefined); },
        confirm: (options) => show<boolean>((settle) => ({ kind: "confirm", title: options.title, message: options.message, settle: settle as (value: unknown) => void })),
        prompt: (options) => show<string>((settle) => ({ kind: "prompt", title: options.title, value: options.value, settle: settle as (value: unknown) => void })),
        select: (options) => show((settle) => ({ kind: "select", title: options.title, options: options.options, settle: settle as (value: unknown) => void })),
      },
      toast: { show(options) { toasts.push(options); } },
    },
  };
  const expectDialog = <K extends Dialog["kind"]>(kind: K) => {
    if (dialog?.kind !== kind) throw new Error(`expected a ${kind} dialog, found ${dialog?.kind ?? "none"} (${dialog?.title ?? ""})`);
    return dialog as Extract<Dialog, { kind: K }>;
  };
  return {
    context, toasts, slashes, dispatched, sizes,
    dialog: () => dialog,
    command: async (id: string) => { void commands.get(id)!(); await tick(); },
    async choose(value: string) {
      const select = expectDialog("select");
      if (!select.options.some((option) => option.value === value)) throw new Error(`no option ${value} in ${select.title}`);
      close(value);
      await tick();
    },
    rows: () => expectDialog("select").options.map((option) => option.value),
    async confirm(value: string) { expectDialog("prompt"); close(value); await tick(); },
    confirmation: () => expectDialog("confirm"),
    async confirmDialog() { expectDialog("confirm"); close(true); await tick(); },
    async cancelDialog() { close(undefined); await tick(); },
  };
}

test("selecting Chat closes the chooser without opening anything else", async () => {
  await withTui(async (host) => {
    await host.command("localcode.mode");
    expect(host.dialog()?.title).toBe("LocalCode mode");
    await host.choose("chat");
    expect(host.dialog()).toBeUndefined();
    expect(host.dispatched).toEqual([]);
  });
});

test("Decisions opens over the chat, keeps the form across visits and Back closes it", async () => {
  await withTui(async (host) => {
    await host.command("localcode.mode");
    await host.choose("decisions");
    expect(host.dialog()?.title).toBe("Decisions · System One");
    expect(host.sizes.at(-1)).toBe("large");
    await host.choose("question");
    await host.confirm("Is it relevant?");
    expect(host.dialog()?.title).toBe("Decisions · System One");
    await host.choose("chat");
    expect(host.dialog()).toBeUndefined();

    await host.command("localcode.decide");
    const question = (host.dialog() as any).options.find((option: any) => option.value === "question");
    expect(question).toBeDefined();
    await host.cancelDialog();
    expect(host.dialog()).toBeUndefined();

    // Opening the mode chooser over the form closes the form; Chat then leaves nothing open.
    await host.command("localcode.decide");
    await host.command("localcode.mode");
    await host.choose("chat");
    expect(host.dialog()).toBeUndefined();
  });
});

test("the answer type controls which rows the form shows and the Model row opens the model picker", async () => {
  await withTui(async (host) => {
    await host.command("localcode.decide");
    expect(host.rows()).not.toContain("options");
    await host.choose("kind");
    await host.choose("choice");
    expect(host.rows()).toContain("options");
    await host.choose("model");
    expect(host.dispatched).toEqual(["model.list"]);
  });
});

test("Back cancels image preparation before inference and repeated Run cannot duplicate it", async () => {
  let finishRead!: (data: ArrayBuffer) => void;
  const read = new Promise<ArrayBuffer>((resolve) => { finishRead = resolve; });
  let reads = 0;
  const file = spyOn(Bun, "file").mockImplementation(() => ({
    size: 1,
    arrayBuffer() { reads++; return read; },
  }) as ReturnType<typeof Bun.file>);
  const fetch = spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({
    state: "ready", current: "OpenJev-Q4_K_M", supports_systemone: true,
  }), { headers: { "content-type": "application/json" } }));
  try {
    await withTui(async (host) => {
      await host.command("localcode.decide");
      await tick();
      await host.choose("question"); await host.confirm("Is it relevant?");
      await host.choose("image"); await host.confirm("example.png");
      await host.choose("run");
      expect(host.dialog()?.title).toBe("Decisions · System One");
      await host.choose("run");
      expect(reads).toBe(1);
      await host.choose("chat");
      finishRead(new Uint8Array([1]).buffer);
      await tick();
      expect(host.dialog()).toBeUndefined();
      expect(fetch.mock.calls.map(([url]) => String(url))).toEqual(["http://127.0.0.1:8124/status"]);
    }, "http://127.0.0.1:8124");
  } finally {
    file.mockRestore();
    fetch.mockRestore();
  }
});

test("a decision result is shown as rows on the redrawn form", async () => {
  const network = spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    if (String(input).endsWith("/decision")) return Response.json({ model: "OpenJev", answers: { answer: { noul: 0.75 } } });
    return Response.json({ state: "ready", current: "OpenJev-Q4_K_M", supports_systemone: true });
  });
  try {
    await withTui(async (host) => {
      await host.command("localcode.decide");
      await tick();
      await host.choose("question"); await host.confirm("Is it relevant?");
      await host.choose("run");
      for (let i = 0; i < 50 && !host.rows().includes("result:yes"); i++) await Bun.sleep(10);
      expect(host.rows()).toEqual(expect.arrayContaining(["result:model", "result:yes", "result:no"]));
    }, "http://127.0.0.1:8125");
  } finally { network.mockRestore(); }
});

test("Run without a decision model explains how to pick one", async () => {
  const network = spyOn(globalThis, "fetch").mockImplementation(async () => Response.json({ state: "ready", current: "Qwen", supports_systemone: false }));
  try {
    await withTui(async (host) => {
      await host.command("localcode.decide");
      await tick();
      await host.choose("question"); await host.confirm("Is it relevant?");
      await host.choose("run");
      expect(host.toasts.some((t) => t.variant === "error" && t.message.includes("OpenJev"))).toBe(true);
      expect(network.mock.calls.some(([url]) => String(url).endsWith("/decision"))).toBe(false);
    }, "http://127.0.0.1:8126");
  } finally { network.mockRestore(); }
});

test("/server fetches authenticated connection info and changes the live API port", async () => {
  const token = process.env.LOCALCODE_CONTROL_TOKEN;
  process.env.LOCALCODE_CONTROL_TOKEN = "server-test-token";
  const changes: unknown[] = [];
  const requests: (string | null)[] = [];
  const network = spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const request = new Request(input, init);
    requests.push(request.headers.get("x-localcode-token"));
    if (new URL(request.url).pathname === "/server/port") return request.json().then((body) => {
      changes.push(body);
      return Response.json({ok: true, restart_required: false, base_url: "http://127.0.0.1:9234/v1"});
    });
    return Response.json({state: "ready", port: 8123, base_url: "http://127.0.0.1:8123/v1", model: "localcode/test", preferred_port: null, server_ready: true});
  });
  try {
    await withTui(async (host) => {
      await host.command("localcode.server");
      await host.choose("port");
      await host.confirm("99");
      expect(changes).toHaveLength(0);
      expect(host.toasts.some(t => t.variant === "error" && t.message.includes("1024"))).toBe(true);
      await host.confirm("9234");
      for (let i = 0; i < 50 && changes.length === 0; i++) await Bun.sleep(10);
      expect(changes).toEqual([{port: 9234}]);
      for (let i = 0; i < 50 && !host.toasts.some(t => t.message.includes("API moved")); i++) await Bun.sleep(10);
      expect(requests.every(token => token === "server-test-token")).toBe(true);
      expect(host.toasts.some(t => t.message.includes("API moved to http://127.0.0.1:9234/v1"))).toBe(true);
    }, "http://127.0.0.1:8323");
  } finally {
    network.mockRestore();
    if (token === undefined) delete process.env.LOCALCODE_CONTROL_TOKEN;
    else process.env.LOCALCODE_CONTROL_TOKEN = token;
  }
});

test("/server surfaces port errors and keeps the prompt for a retry", async () => {
  let changes = 0;
  const network = spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    if (String(input).endsWith("/server/port")) {
      changes++;
      return Response.json({error: "Port occupied; current port unchanged"}, {status: 409});
    }
    return Response.json({state: "ready", port: 8123, base_url: "http://127.0.0.1:8123/v1", model: "localcode/test", preferred_port: null});
  });
  try {
    await withTui(async (host) => {
      await host.command("localcode.server");
      await host.choose("port");
      await host.confirm("9234");
      for (let i = 0; i < 50 && host.toasts.length === 0; i++) await Bun.sleep(10);
      expect(changes).toBe(1);
      expect(host.toasts.some(t => t.variant === "error" && t.message.includes("Port occupied"))).toBe(true);
      expect(host.toasts.some(t => t.message.includes("API moved"))).toBe(false);
      expect(host.dialog()?.kind).toBe("prompt");
      expect((host.dialog() as any).value).toBe("9234");
      await host.confirm("9234"); // failure released the guard, allowing a retry
      for (let i = 0; i < 50 && changes < 2; i++) await Bun.sleep(10);
      expect(changes).toBe(2);
    }, "http://127.0.0.1:8323");
  } finally { network.mockRestore(); }
});

test("/server stops and starts and reports rejected operations", async () => {
  let running = true;
  let reject = false;
  const actions: string[] = [];
  const fetcher = spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = new URL(String(input)).pathname;
    if (path === "/server/stop" || path === "/server/start") {
      expect(init?.method).toBe("POST");
      expect(init?.body).toBe("{}");
      actions.push(path);
      if (reject) return Response.json({error:"A request is active"},{status:409});
      running = path.endsWith("start");
      return Response.json({ok:true,state:running ? "loading" : "stopped"});
    }
    return Response.json({state:running ? "ready" : "stopped", server_running:running, server_ready:running, port:8123, base_url:"http://127.0.0.1:8123/v1",model:"localcode/test"});
  });
  try {
    await withTui(async host => {
      await host.command("localcode.server");
      await host.choose("stop");
      await Bun.sleep(20);
      expect(actions).toEqual(["/server/stop"]);
      await host.choose("start");
      await Bun.sleep(20);
      expect(actions).toEqual(["/server/stop","/server/start"]);
      reject = true;
      await host.choose("stop");
      await Bun.sleep(20);
      expect(host.toasts.some(t=>t.variant === "error" && t.message.includes("A request is active"))).toBe(true);
    }, "http://127.0.0.1:8323");
  } finally { fetcher.mockRestore(); }
});

test("native keymap discovers server, mode and decide slash commands", async () => {
  await withTui((host) => {
    expect(Object.fromEntries(host.slashes)).toEqual({server: "localcode.server", mode: "localcode.mode", decide: "localcode.decide"});
  });
});

test("busy or unverified server stop offers explicit confirmation and retries only after acceptance", async () => {
  const posted: unknown[] = [];
  const network = spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    if (new URL(String(input)).pathname === "/server/stop") {
      const body = JSON.parse(String(init?.body)); posted.push(body);
      if (body.force !== true) return Response.json({error: "Server activity could not be checked.", force_required: true}, {status: 409});
      return Response.json({ok: true, state: "stopping"});
    }
    return Response.json({state:"ready",server_running:true,port:8123,base_url:"http://127.0.0.1:8123/v1"});
  });
  try {
    await withTui(async host => {
      await host.command("localcode.server"); await host.choose("stop");
      await Bun.sleep(20);
      expect(host.confirmation().title).toBe("Stop shared model server?");
      expect(host.confirmation().message).toContain("every LocalCode window and connected app");
      expect(posted).toEqual([{}]);
      await host.cancelDialog(); await Bun.sleep(20);
      expect(posted).toEqual([{}]);
      await host.command("localcode.server"); await host.choose("stop"); await Bun.sleep(20);
      await host.confirmDialog(); await Bun.sleep(20);
      expect(posted).toEqual([{}, {}, {force:true}]);
    }, "http://127.0.0.1:8323");
  } finally { network.mockRestore(); }
});

test("the terminal entry shows server notifications as toasts and adds the modes", async () => {
  const host = fakeTui();
  let handler: ((event: { data: any }) => void) | undefined;
  let stopped = false;
  const context = {
    ...host.context,
    client: { rpc: (definition: { id: string }) => ({ events: { on(name: string, callback: typeof handler) {
      expect(definition.id).toBe("localcode");
      expect(name).toBe("toast");
      handler = callback;
      return () => { stopped = true; };
    } } }) },
  };
  const dispose = await tui.setup(context as any);
  handler!({ data: { message: "Model server stopped", variant: "warning", title: "LocalCode", duration: 5000 } });
  expect(host.toasts).toEqual([{ message: "Model server stopped", variant: "warning", title: "LocalCode", duration: 5000 }]);
  expect(host.slashes.size).toBe(3);
  dispose();
  expect(stopped).toBe(true);
});
