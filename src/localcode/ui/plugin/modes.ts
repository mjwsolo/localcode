/** Terminal-only mode chooser using the pinned OpenCode TUI extension API. */
import type { TuiPluginApi, TuiPluginModule } from "@opencode-ai/plugin/tui";

type Kind = "yes-no" | "choice" | "score";
type Status = { state: string; current?: string; supports_systemone?: boolean };
type Answer = { choice?: string; noul?: number; score?: number; probabilities?: Record<string, number>; legend?: Record<string, string> | string[] };
type Result = { model: string; answers: Record<string, Answer>; usage?: { input_tokens?: number; output_tokens?: number } };
type Form = { question: string; state: string; kind: Kind; options: string; image: string };
const route = "localcode.decisions";

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

export function answerRows(result: Result) {
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

async function tui(api: TuiPluginApi) {
  let previous: { name: string; params?: Record<string, unknown> } = { name: "home" };
  let form: Form = { question: "", state: "", kind: "yes-no", options: "", image: "" };
  let status: Status | undefined;
  let result: Result | undefined;
  let busy = false;
  let revision = 0;
  let pending: AbortController | undefined;
  const control = (process.env.LOCALCODE_CONTROL_URL ?? "").replace(/\/$/, "");
  const headers = { "content-type": "application/json", "x-localcode-token": process.env.LOCALCODE_CONTROL_TOKEN ?? "" };
  const error = (message: string) => api.ui.toast({ variant: "error", message, duration: 8000 });
  const refresh = () => {
    if (api.route.current.name === route) api.route.navigate(route, { revision: ++revision });
  };
  const back = () => {
    pending?.abort();
    api.ui.dialog.clear();
    api.route.navigate(previous.name, previous.params);
  };
  async function poll() {
    if (api.route.current.name !== route || !control) return;
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
  function open() {
    if (api.route.current.name !== route) {
      const current = api.route.current;
      previous = current.name === "session" ? { name: "session", params: { ...current.params } } : { name: "home" };
    }
    api.ui.dialog.clear();
    api.route.navigate(route, { revision: ++revision });
    void poll();
  }
  function edit(key: "question" | "state" | "options" | "image", title: string) {
    api.ui.dialog.replace(() => api.ui.DialogPrompt({ title, value: form[key], onConfirm(value) {
      form = { ...form, [key]: value };
      result = undefined;
      api.ui.dialog.clear();
      refresh();
    } }));
  }
  function kind() {
    api.ui.dialog.replace(() => api.ui.DialogSelect<Kind>({ title: "Answer type", current: form.kind, options: [
      { title: "Yes / no", value: "yes-no", description: "Probability of yes and no" },
      { title: "Choice", value: "choice", description: "Pick from your own labels" },
      { title: "Score", value: "score", description: "Ordered levels, lowest to highest" },
    ], onSelect(option) {
      form = { ...form, kind: option.value };
      result = undefined;
      api.ui.dialog.clear();
      refresh();
    } }));
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
      if (api.route.current.name !== route) return;
      result = data as Result;
    } catch (cause) {
      if (api.route.current.name === route) error(cause instanceof Error ? cause.message : "Decision request failed");
    } finally {
      busy = false;
      pending = undefined;
      refresh();
    }
  }
  api.route.register([{ name: route, render(input) {
    // The host uses a reactive route store; reading the revision redraws fields
    // without persisting user content in the global settings database.
    void input.params?.revision;
    const rows = [
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
    return api.ui.Dialog({ size: "large", onClose: back, children: api.ui.DialogSelect({ title: "Decisions · System One", placeholder: "Choose a field or action", options: rows, skipFilter: true, onSelect(item) {
      if (item.value === "chat") return back();
      if (busy) return;
      if (item.value === "model") return api.keymap.dispatchCommand("model.list");
      if (item.value === "kind") return kind();
      if (item.value === "question") return edit("question", "Decision question");
      if (item.value === "state") return edit("state", "Content to evaluate (optional)");
      if (item.value === "options") return edit("options", form.kind === "score" ? "Levels: lowest to highest, one per line" : "Choices: one per line");
      if (item.value === "image") return edit("image", "Image file path (optional)");
      if (item.value === "run") {
        if (status?.state !== "ready" || status?.supports_systemone !== true) return error("Select a decision model such as OpenJev using the Model row first");
        void evaluate();
      }
    } }) });
  } }]);
  const choose = () => api.ui.dialog.replace(() => api.ui.DialogSelect({ title: "LocalCode mode", current: api.route.current.name === route ? "decisions" : "chat", options: [
    { title: "Chat", value: "chat", description: "Ask questions and work on code" },
    { title: "Decisions", value: "decisions", description: "Typed choices, yes/no probabilities and scores" },
  ], onSelect(item) {
    if (item.value === "decisions") return open();
    if (api.route.current.name === route) return back();
    api.ui.dialog.clear();
  } }));
  api.keymap.registerLayer({ commands: [
    { name: "localcode.mode", title: "Choose Chat or Decisions", category: "LocalCode", slash: { name: "mode" }, run: choose },
    { name: "localcode.decide", title: "Open decision mode", category: "LocalCode", slash: { name: "decide" }, run: open },
  ] });
  api.keymap.registerLayer({ enabled: () => api.route.current.name === route && !api.ui.dialog.open, priority: 5, commands: [{ name: "localcode.mode.back", run: back }], bindings: [{ key: "escape", command: "localcode.mode.back" }] });
  const timer = setInterval(() => void poll(), 2000);
  let startup: ReturnType<typeof setInterval> | undefined;
  if (process.env.LOCALCODE_MODE === "decisions") startup = setInterval(() => {
    if (!api.state.ready) return;
    clearInterval(startup);
    open();
  }, 100);
  api.lifecycle.onDispose(() => { clearInterval(timer); clearInterval(startup); pending?.abort(); });
}

export default { id: "localcode.modes", tui } satisfies TuiPluginModule;
