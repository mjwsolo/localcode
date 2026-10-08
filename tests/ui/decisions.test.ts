import { expect, spyOn, test } from "bun:test";
import modes, { answerRows, requestFor } from "../../src/localcode/ui/plugin/modes";

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

test("score results show the supplied level names", () => {
  const rows=answerRows({model:"OpenJev",answers:{answer:{score:0.8,probabilities:{"0":0.2,"1":0.8},legend:{"0":"low","1":"high"}}}});
  expect(rows.map(r=>r.title)).toEqual(["high","low"]);
});

// Run the plugin's registered commands and routes, including the actual mode
// chooser callbacks. Pure request helpers cannot detect session navigation bugs.
async function withTui(run: (host: ReturnType<typeof fakeTui>) => Promise<void> | void, controlUrl = "") {
  const host = fakeTui();
  const startup = process.env.LOCALCODE_MODE;
  const control = process.env.LOCALCODE_CONTROL_URL;
  delete process.env.LOCALCODE_MODE;
  process.env.LOCALCODE_CONTROL_URL = controlUrl;
  try {
    await modes.tui(host.api as unknown as Parameters<typeof modes.tui>[0]);
    await run(host);
  } finally {
    host.dispose();
    if (startup === undefined) delete process.env.LOCALCODE_MODE;
    else process.env.LOCALCODE_MODE = startup;
    if (control === undefined) delete process.env.LOCALCODE_CONTROL_URL;
    else process.env.LOCALCODE_CONTROL_URL = control;
  }
}

function fakeTui() {
  type Current = { name: string; params?: Record<string, unknown> };
  type Option = { value: string };
  type Select = { options: Option[]; onSelect: (option: Option) => void };
  type Prompt = { onConfirm: (value: string) => void };
  let current: Current = { name: "session", params: { sessionID: "ses_first" } };
  let dialog: Select | Prompt | undefined;
  const commands = new Map<string, () => void>();
  const routes = new Map<string, (input: { params?: Record<string, unknown> }) => unknown>();
  const disposals: (() => void)[] = [];
  const navigations: Current[] = [];
  const toasts: { message: string; variant: string }[] = [];
  const api = {
    state: { ready: true },
    route: {
      get current() { return current; },
      navigate(name: string, params?: Record<string, unknown>) {
        current = { name, ...(params ? { params } : {}) };
        navigations.push(current);
      },
      register(items: { name: string; render: (input: { params?: Record<string, unknown> }) => unknown }[]) {
        for (const item of items) routes.set(item.name, item.render);
      },
    },
    keymap: {
      registerLayer(layer: { commands: { name: string; run: () => void }[] }) {
        for (const command of layer.commands) commands.set(command.name, command.run);
      },
    },
    ui: {
      Dialog(props: unknown) { return props; },
      DialogSelect(props: Select) { return props; },
      DialogPrompt(props: Prompt) { return props; },
      toast(value: { message: string; variant: string }) { toasts.push(value); },
      dialog: {
        get open() { return dialog !== undefined; },
        clear() { dialog = undefined; },
        replace(render: () => Select | Prompt) { dialog = render(); },
      },
    },
    lifecycle: { onDispose(callback: () => void) { disposals.push(callback); } },
  };
  return {
    api, navigations, toasts,
    current: () => current,
    dispose: () => disposals.forEach((callback) => callback()),
    command: (name: string) => commands.get(name)!(),
    choose(value: string) {
      const select = dialog as Select;
      select.onSelect(select.options.find((option) => option.value === value)!);
    },
    confirm: (value: string) => (dialog as Prompt).onConfirm(value),
    selectRow(value: string) {
      const page = routes.get(current.name)!({ params: current.params }) as { children: Select };
      page.children.onSelect(page.children.options.find((option) => option.value === value)!);
    },
  };
}

test("selecting Chat in an active session preserves that session without navigating", async () => {
  await withTui((host) => {
    host.command("localcode.mode");
    host.choose("chat");
    expect(host.current()).toEqual({ name: "session", params: { sessionID: "ses_first" } });
    expect(host.navigations).toEqual([]);
    expect(host.api.ui.dialog.open).toBe(false);
  });
});

test("Decisions returns to the exact session and Chat never restores a stale session", async () => {
  await withTui((host) => {
    const first = host.current();
    host.command("localcode.mode");
    host.choose("decisions");
    expect(host.current().name).toBe("localcode.decisions");
    host.selectRow("chat");
    expect(host.current()).toEqual(first);

    const second = { name: "session", params: { sessionID: "ses_second", prompt: { input: "keep this" } } };
    host.api.route.navigate(second.name, second.params);
    const count = host.navigations.length;
    host.command("localcode.mode");
    host.choose("chat");
    expect(host.current()).toEqual(second);
    expect(host.navigations).toHaveLength(count);

    host.command("localcode.decide");
    host.command("localcode.mode");
    host.choose("chat");
    expect(host.current()).toEqual(second);
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
      host.command("localcode.decide");
      await new Promise((resolve) => setTimeout(resolve, 0));
      host.selectRow("question"); host.confirm("Is it relevant?");
      host.selectRow("image"); host.confirm("example.png");
      host.selectRow("run");
      host.selectRow("run");
      expect(reads).toBe(1);
      host.selectRow("chat");
      finishRead(new Uint8Array([1]).buffer);
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(host.current()).toEqual({ name: "session", params: { sessionID: "ses_first" } });
      expect(fetch.mock.calls.map(([url]) => String(url))).toEqual(["http://127.0.0.1:8124/status"]);
    }, "http://127.0.0.1:8124");
  } finally {
    file.mockRestore();
    fetch.mockRestore();
  }
});

test("/server fetches authenticated connection info and changes the live API port without navigating", async () => {
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
      host.choose("port");
      host.confirm("99");
      expect(changes).toHaveLength(0);
      host.confirm("9234");
      for (let i = 0; i < 50 && changes.length === 0; i++) await Bun.sleep(10);
      expect(changes).toEqual([{port: 9234}]);
      expect(requests.every(token => token === "server-test-token")).toBe(true);
      expect(host.navigations).toEqual([]);
    }, "http://127.0.0.1:8323");
  } finally {
    network.mockRestore();
    if (token === undefined) delete process.env.LOCALCODE_CONTROL_TOKEN;
    else process.env.LOCALCODE_CONTROL_TOKEN = token;
  }
});

test("/server rejects duplicate submissions and surfaces port errors without losing the session", async () => {
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
      host.choose("port");
      host.confirm("9234");
      host.confirm("9234");
      for (let i = 0; i < 50 && host.toasts.length === 0; i++) await Bun.sleep(10);
      expect(changes).toBe(1);
      expect(host.toasts.some(t => t.variant === "error" && t.message.includes("Port occupied"))).toBe(true);
      expect(host.toasts.some(t => t.message.includes("API moved"))).toBe(false);
      expect(host.current()).toEqual({name: "session", params: {sessionID: "ses_first"}});
      host.confirm("9234"); // failure released the UI guard, allowing a retry
      for (let i = 0; i < 50 && changes < 2; i++) await Bun.sleep(10);
      expect(changes).toBe(2);
    }, "http://127.0.0.1:8323");
  } finally { network.mockRestore(); }
});
