/**
 * Terminal-only LocalCode modes for the opencode v2 TUI plugin API: /mode (Chat or
 * Decisions), /decide (the System One decision form) and /server (the shared model
 * server's connection, port and start/stop). Everything is a dialog over the current
 * chat, so leaving a mode always returns to the exact session that was open.
 */
type Kind = "yes-no" | "choice" | "score";
type Status = { state: string; current?: string; supports_systemone?: boolean };
type Answer = { choice?: string; noul?: number; score?: number; probabilities?: Record<string, number>; legend?: Record<string, string> | string[] };
type Result = { model: string; answers: Record<string, Answer>; usage?: { input_tokens?: number; output_tokens?: number } };
type Form = { question: string; state: string; kind: Kind; options: string; image: string };
type Option = { title: string; value: string; description?: string; footer?: string; category?: string };

/** The parts of the v2 TUI plugin context this module uses (see @opencode/plugin/tui). */
export type ModesContext = {
  readonly keymap: {
    layer(factory: () => { mode?: string; commands?: readonly { id?: string; title?: string; group?: string; palette?: true; slash?: { name: string }; run: () => void | false | Promise<void> }[] }): void;
    dispatch(id: string, input?: string): void;
  };
  readonly ui: {
    readonly dialog: {
      set(options: { size?: "medium" | "large" | "xlarge" }): void;
      clear(): void;
      confirm(options: { title: string; message: string; label?: { confirm?: string; cancel?: string } }): Promise<boolean | undefined>;
      prompt(options: { title: string; description?: string; placeholder?: string; value?: string }): Promise<string | undefined>;
      select<Value>(options: { title: string; placeholder?: string; options: readonly { title: string; value: Value; description?: string; footer?: string; category?: string }[]; current?: Value }): Promise<Value | undefined>;
    };
    readonly toast: { show(options: { title?: string; message: string; variant?: "info" | "success" | "warning" | "error"; duration?: number }): void };
  };
};

export function requestFor(form: Form) {
  if (!form.question.trim()) throw new Error("Enter a question first");
  const options = form.options.split("\n").map((s) => s.trim()).filter(Boolean);
  if (form.kind === "choice" && (options.length < 2 || options.length > 52 || new Set(options).size !== options.length)) {
    throw new Error("Enter 2 to 52 different choices, one per line");
  }
  if (form.kind === "score" && (options.length < 2 || options.length > 10)) {
    throw new Error("Enter 2 to 10 levels, one per line, lowest to highest");
  }
  const question = {
    type: form.kind === "yes-no" ? "noul" : form.kind,
    instructions: form.question.trim(),
    ...(form.kind === "choice" ? { criteria: Object.fromEntries(options.map((label) => [label, null])) } : {}),
    ...(form.kind === "score" ? { criteria: options } : {}),
  };
  return { state: form.state, questions: { answer: question } };
}

export function answerRows(result: Result): Option[] {
  const answer = result.answers.answer;
  if (!answer) throw new Error("Decision response has no answer");
  if (answer.noul !== undefined) return [
    { title: "Yes", value: "result:yes", footer: `${(answer.noul * 100).toFixed(1)}%`, category: "Result" },
    { title: "No", value: "result:no", footer: `${((1 - answer.noul) * 100).toFixed(1)}%`, category: "Result" },
  ];
  return Object.entries(answer.probabilities ?? {}).sort((a, b) => b[1] - a[1]).map(([label, value]) => ({
    title: (Array.isArray(answer.legend) ? answer.legend[Number(label)] : answer.legend?.[label]) ?? (label + (label === answer.choice ? " ✓" : "")), value: "result:" + label,
    footer: `${(value * 100).toFixed(1)}%`, category: "Result",
  }));
}

export function setupModes(context: ModesContext): () => void {
  let form: Form = { question: "", state: "", kind: "yes-no", options: "", image: "" };
  let status: Status | undefined;
  let result: Result | undefined;
  let busy = false;
  let pending: AbortController | undefined;
  let pageOpen = false;
  let showing = false;
  let rerender = false;
  let disposed = false;
  const control = (process.env.LOCALCODE_CONTROL_URL ?? "").replace(/\/$/, "");
  const headers = { "content-type": "application/json", "x-localcode-token": process.env.LOCALCODE_CONTROL_TOKEN ?? "" };
  const error = (message: string) => context.ui.toast.show({ variant: "error", message, duration: 8000 });
  const info = (message: string, duration?: number) => context.ui.toast.show({ variant: "info", message, ...(duration ? { duration } : {}) });

  /**
   * Redraw the decision form: the select dialog is closed and reopened with fresh rows.
   * Only while the form itself is showing; an open edit prompt is never interrupted, and
   * the form is rebuilt with current rows when the prompt closes.
   */
  const refresh = () => {
    if (!pageOpen || !showing) return;
    rerender = true;
    context.ui.dialog.clear();
  };
  const back = () => {
    pending?.abort();
    pageOpen = false;
  };

  async function poll() {
    if (!pageOpen || !control) return;
    try {
      const response = await fetch(control + "/status", { headers, signal: AbortSignal.timeout(2000) });
      if (!response.ok) return;
      const data = await response.json() as Status;
      const next = { state: data.state, current: data.current, supports_systemone: data.supports_systemone };
      if (JSON.stringify(next) === JSON.stringify(status)) return;
      status = next;
      refresh();
    } catch { /* Status is retried; explicit evaluation reports failures. */ }
  }

  function rows(): Option[] {
    return [
      { title: "Model: " + (status?.current ?? "none loaded"), value: "model", description: status?.supports_systemone ? "Ready for decisions" : "Select OpenJev with /models", category: "Decisions" },
      { title: "Question", value: "question", description: form.question || "What should the model decide?", category: "Input" },
      { title: "Content (optional)", value: "state", description: form.state.slice(0, 60) || "Text or JSON to evaluate", category: "Input" },
      { title: "Answer type: " + form.kind, value: "kind", category: "Input" },
      ...(form.kind === "yes-no" ? [] : [{ title: form.kind === "score" ? "Levels (lowest to highest)" : "Choices", value: "options", description: form.options.replaceAll("\n", " · ") || "Enter one per line", category: "Input" }]),
      { title: "Image (optional)", value: "image", description: form.image || "PNG, JPEG or WebP file path", category: "Input" },
      { title: busy ? "Evaluating…" : "Run decision", value: "run", category: "Actions" },
      ...(result ? [{ title: "Answered by " + result.model, value: "result:model", description: "Prediction only; no tools or code actions executed", category: "Result" }, ...answerRows(result), ...(result.answers.answer.score === undefined ? [] : [{ title: "Score", value: "result:score", footer: result.answers.answer.score.toFixed(3), category: "Result" }])] : []),
      { title: "Back to chat", value: "chat", category: "Actions" },
    ];
  }

  async function edit(key: "question" | "state" | "options" | "image", title: string) {
    const value = await context.ui.dialog.prompt({ title, value: form[key] });
    if (value === undefined) return;
    form = { ...form, [key]: value };
    result = undefined;
  }

  async function kind() {
    const value = await context.ui.dialog.select<Kind>({ title: "Answer type", current: form.kind, options: [
      { title: "Yes / no", value: "yes-no", description: "Probability of yes and no" },
      { title: "Choice", value: "choice", description: "Pick from your own labels" },
      { title: "Score", value: "score", description: "Ordered levels, lowest to highest" },
    ] });
    if (value === undefined) return;
    form = { ...form, kind: value };
    result = undefined;
  }

  async function evaluate() {
    if (busy) return;
    busy = true;
    result = undefined;
    const controller = new AbortController();
    pending = controller;
    refresh();
    try {
      const body: ReturnType<typeof requestFor> & { images?: string[] } = requestFor(form);
      if (form.image.trim()) {
        const file = Bun.file(form.image.trim().replace(/^~\//, (process.env.HOME ?? "") + "/"));
        if (file.size > 5 * 1024 * 1024) throw new Error("Image exceeds 5 MB");
        const ext = form.image.trim().split(".").at(-1)?.toLowerCase();
        const mime = ext === "png" ? "image/png" : ext === "webp" ? "image/webp" : ["jpg", "jpeg"].includes(ext ?? "") ? "image/jpeg" : "";
        if (!mime) throw new Error("Image must be PNG, JPEG or WebP");
        body.images = [`data:${mime};base64,${Buffer.from(await file.arrayBuffer()).toString("base64")}`];
      }
      if (controller.signal.aborted) return;
      const response = await fetch(control + "/decision", { method: "POST", headers, body: JSON.stringify(body), signal: AbortSignal.any([controller.signal, AbortSignal.timeout(65000)]) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error ?? "Decision request failed");
      if (!pageOpen) return;
      result = data as Result;
    } catch (cause) {
      if (pageOpen && !controller.signal.aborted) error(cause instanceof Error ? cause.message : "Decision request failed");
    } finally {
      busy = false;
      pending = undefined;
      refresh();
    }
  }

  /** The decision form: a select dialog over the chat, reopened after every edit and status change. */
  async function open() {
    if (pageOpen) return;
    pageOpen = true;
    void poll();
    const timer = setInterval(() => void poll(), 2000);
    try {
      while (pageOpen && !disposed) {
        rerender = false;
        const selection = context.ui.dialog.select<string>({ title: "Decisions · System One", placeholder: "Choose a field or action", options: rows() });
        context.ui.dialog.set({ size: "large" });
        showing = true;
        const value = await selection;
        showing = false;
        if (value === undefined) {
          if (rerender) continue;
          back();
          break;
        }
        if (value === "chat") { back(); break; }
        if (busy) continue;
        // The model picker is a dialog of its own; the form keeps its fields for the next /decide.
        if (value === "model") { back(); context.keymap.dispatch("model.list"); break; }
        if (value === "kind") await kind();
        else if (value === "question") await edit("question", "Decision question");
        else if (value === "state") await edit("state", "Content to evaluate (optional)");
        else if (value === "options") await edit("options", form.kind === "score" ? "Levels: lowest to highest, one per line" : "Choices: one per line");
        else if (value === "image") await edit("image", "Image file path (optional)");
        else if (value === "run") {
          if (status?.state !== "ready" || status?.supports_systemone !== true) error("Select a decision model such as OpenJev using the Model row first");
          else void evaluate();
        }
      }
    } finally {
      clearInterval(timer);
      showing = false;
      pageOpen = false;
    }
  }

  async function choose() {
    const value = await context.ui.dialog.select<string>({ title: "LocalCode mode", current: pageOpen ? "decisions" : "chat", options: [
      { title: "Chat", value: "chat", description: "Ask questions and work on code" },
      { title: "Decisions", value: "decisions", description: "Typed choices, yes/no probabilities and scores" },
    ] });
    if (value === "decisions") await open();
  }

  let changingPort = false;
  let changingServer = false;
  async function changeServer(action: "start" | "stop", force = false) {
    if (changingServer) return;
    changingServer = true;
    let confirmStop = false;
    let detail = "";
    try {
      const response = await fetch(control + "/server/" + action, {
        method: "POST", headers, body: JSON.stringify(force ? { force: true } : {}), signal: AbortSignal.timeout(5000),
      });
      const outcome = await response.json() as { error?: string; state?: string; force_required?: boolean };
      if (!response.ok) {
        if (action !== "stop" || outcome.force_required !== true) throw new Error(outcome.error || `HTTP ${response.status}`);
        confirmStop = true;
        detail = outcome.error || "The server may be busy";
      } else info(action === "stop" ? "Stopping shared model server. Chat history stays available." : "Loading model. Refresh /server to check progress.", 8000);
    } catch (cause) {
      error(`Could not confirm server operation: ${cause instanceof Error ? cause.message : String(cause)}. Refresh /server before retrying.`);
      return;
    } finally {
      changingServer = false;
    }
    if (confirmStop) {
      const accepted = await context.ui.dialog.confirm({
        title: "Stop shared model server?",
        message: `${detail} Stop anyway? This interrupts requests in every LocalCode window and connected app, then frees model memory. Saved chat history stays available.`,
      });
      if (accepted === true) await changeServer("stop", true);
      return;
    }
    await server();
  }

  /** The port prompt stays available after a rejected value or a failed change, as a retry. */
  async function changePort(current: number | string | null, entered?: string) {
    const value = await context.ui.dialog.prompt({ title: "New API port (1024–65535 or auto)", value: entered ?? String(current ?? "auto") });
    if (value === undefined || changingPort) return;
    const input = value.trim();
    if (input !== "auto" && (!/^\d+$/.test(input) || Number(input) < 1024 || Number(input) > 65535)) {
      error("Enter a port from 1024 to 65535, or auto");
      return changePort(current, value);
    }
    changingPort = true;
    let moved: { base_url: string };
    try {
      const r = await fetch(control + "/server/port", { method: "POST", headers, body: JSON.stringify({ port: input === "auto" ? null : Number(input) }), signal: AbortSignal.timeout(3000) });
      if (!r.ok) throw new Error((await r.json() as { error: string }).error);
      moved = await r.json() as { base_url: string };
    } catch (e) {
      error(`Could not confirm port change: ${String(e)}. Refresh /server to check the current address.`);
      return changePort(current, value);
    } finally {
      changingPort = false;
    }
    info(`API moved to ${moved.base_url}. Update other apps to use this address; this chat stays connected.`, 8000);
    await server();
  }

  async function server() {
    if (!control) return error("Start LocalCode to manage its model server");
    let info_: { state: string; base_url: string; model?: string; port: number; preferred_port: number | string | null; server_ready: boolean; server_running?: boolean; warmup?: string; detail?: string; budget?: number; ctx?: number; performance?: { decode_tps?: number; prefill_tps?: number; prompt_reuse_estimate_pct?: number }; machine?: { warnings?: string[]; swap_used_bytes?: number; on_battery?: boolean } };
    try {
      const response = await fetch(control + "/server", { headers, signal: AbortSignal.timeout(12000) });
      if (!response.ok) throw new Error(`Server status: HTTP ${response.status}`);
      info_ = await response.json();
    } catch (e) { return error(String(e)); }
    const s = info_;
    const copy = async (value: string, secret = false) => {
      const proc = Bun.spawn(["pbcopy"], { stdin: new Blob([value]), stdout: "ignore", stderr: "ignore" });
      if (await proc.exited !== 0) throw new Error("Could not copy to clipboard");
      info(secret ? "Copied with API key. Keep the clipboard contents private." : "Copied to clipboard");
    };
    const value = await context.ui.dialog.select<string>({ title: "Model server · " + s.state, options: [
      { title: "Status", value: "status", description: s.detail || s.state },
      { title: "Generation", value: "status", description: s.performance?.decode_tps != null ? `${s.performance.decode_tps} tokens/s · recent requests` : "Not enough generation measured yet" },
      { title: "Prompt processing", value: "status", description: s.performance?.prefill_tps != null ? `${s.performance.prefill_tps} tokens/s · recent requests` : "Not enough prompt processing measured yet" },
      { title: "Prompt reuse (estimate)", value: "status", description: s.performance?.prompt_reuse_estimate_pct != null ? `${s.performance.prompt_reuse_estimate_pct}% · recent requests` : "Unknown" },
      { title: "Context budget", value: "status", description: `${s.budget ?? "automatic"} / ${s.ctx ?? "unknown"} tokens` },
      { title: "System swap", value: "status", description: s.machine?.swap_used_bytes != null ? `${(s.machine.swap_used_bytes / 1024 ** 3).toFixed(2)} GiB · system-wide` : "Unknown" },
      { title: "Power", value: "status", description: s.machine?.on_battery == null ? "Unknown" : s.machine.on_battery ? "Battery" : "External power" },
      ...(s.machine?.warnings ?? []).map((message) => ({ title: "Performance warning", value: "status", description: message })),
      { title: "Prompt cache", value: "status", description: s.warmup === "warming" ? "Warming up in background · sending a message interrupts warm-up" : s.warmup === "warm" ? "Warm prefix available · changed context still needs processing" : s.warmup === "cold" ? "First use: no saved prefix yet" : s.warmup ?? "Unknown" },
      { title: "Base URL", value: "url", description: s.base_url },
      { title: "Model", value: "model", description: s.model ?? "No model loaded" },
      { title: "Copy API key", value: "key", description: "Private · only for clients you trust" },
      { title: "Copy curl example", value: "curl", description: "Includes your API key; lists available models" },
      { title: "Change API port now", value: "port", description: `Current: ${s.port} · saved: ${s.preferred_port ?? "automatic"}` },
      ...(!["loading", "downloading", "stopping"].includes(s.state) ? [{
        title: s.server_running ? "Stop model server" : "Start model server",
        value: s.server_running ? "stop" : "start",
        description: s.server_running ? "Frees model memory · affects all windows and API clients · chat stays open" : "Reloads the previous model · loading and context processing take time",
      }] : []),
      { title: "Refresh", value: "refresh", description: "Loopback only · available to tools on this Mac" },
    ] });
    if (value === undefined) return;
    if (value === "refresh" || value === "status") return server();
    if (value === "start" || value === "stop") return changeServer(value);
    if (value === "port") return changePort(s.preferred_port);
    const key = process.env.LOCALCODE_SERVER_KEY;
    if ((value === "key" || value === "curl") && !key) return error("API key unavailable; use localcode api --show-key in a terminal");
    const text = value === "url" ? s.base_url : value === "model" ? s.model : value === "key" ? key : `curl '${s.base_url}/models' -H 'Authorization: Bearer ${key}'`;
    if (text) await copy(text, value === "key" || value === "curl").catch((e) => error(String(e)));
  }

  context.keymap.layer(() => ({ mode: "global", commands: [
    { id: "localcode.server", title: "Model server connection and port", group: "LocalCode", palette: true, slash: { name: "server" }, run: () => void server() },
    { id: "localcode.mode", title: "Choose Chat or Decisions", group: "LocalCode", palette: true, slash: { name: "mode" }, run: () => void choose() },
    { id: "localcode.decide", title: "Open decision mode", group: "LocalCode", palette: true, slash: { name: "decide" }, run: () => void open() },
  ] }));
  const startup = process.env.LOCALCODE_MODE === "decisions" ? setTimeout(() => void open(), 100) : undefined;
  return () => {
    disposed = true;
    clearTimeout(startup);
    pending?.abort();
    if (pageOpen) { pageOpen = false; context.ui.dialog.clear(); }
  };
}
