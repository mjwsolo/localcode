/**
 * localcode — the whole model experience in one extension.
 *
 *   first run  : no model on disk → the picker opens by itself
 *   browse     : model family → quant, with ✓ downloaded / ↓ downloadable / ★ recommended
 *   download   : llama-server fetches it from Hugging Face, live progress
 *   switch     : loads it and selects it, no restart
 *
 * Everything is built from native APIs: registerProvider, registerCommand,
 * ctx.ui.select / confirm / notify / setStatus. No UI code of our own.
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir, totalmem } from "node:os";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";

type Quant = {
  key: string; label: string; filename: string; hf_repo: string; size_gb: number;
  revision: string; humaneval: number | null; recommended_at_ram_gb: number | null;
  mmproj_filename: string | null; mmproj_hf_filename: string | null;
};
type Group = { key: string; name: string; maker: string; architecture: string; license: string; notes: string; quants: Quant[] };

const BASE = (process.env.LLAMA_BASE_URL ?? "http://127.0.0.1:8080").replace(/\/+$/, "");
const MODELS_DIR = process.env.LOCALCODE_MODELS_DIR ?? join(homedir(), ".local/share/localcode/models");
const { groups } = JSON.parse(readFileSync(new URL("./catalog.json", import.meta.url), "utf8")) as { groups: Group[] };
const RAM_GB = Math.round(totalmem() / 1024 ** 3);

const onDisk = (q: Quant) => existsSync(join(MODELS_DIR, q.filename));
const allQuants = groups.flatMap((g) => g.quants);
const RECOMMENDED = allQuants
  .filter((q) => q.recommended_at_ram_gb !== null && q.recommended_at_ram_gb <= RAM_GB)
  .sort((a, b) => (b.recommended_at_ram_gb ?? 0) - (a.recommended_at_ram_gb ?? 0))[0];

const modelId = (q: Quant) => q.filename.replace(/\.gguf$/, "");
const quantName = (q: Quant) => modelId(q).match(/((UD-)?(IQ|Q|BF)[0-9][^-]*(_[A-Z0-9]+)*)$/)?.[1] ?? q.label;
/** what llama-server wants for a Hugging Face pull: owner/repo:QUANT */
const hfRef = (q: Quant) => `${q.hf_repo}:${quantName(q).replace(/^UD-/, "")}`;
const groupOf = (q: Quant) => groups.find((g) => g.quants.some((x) => x.key === q.key))!;
const pretty = (q: Quant) => `${groupOf(q).name} · ${quantName(q)}`;

async function serverModels(): Promise<{ id: string; status: string; ctx: number | null; vision: boolean }[]> {
  try {
    const r = await fetch(`${BASE}/models`, { signal: AbortSignal.timeout(10_000) });
    const d = (await r.json()) as any;
    return (d.data ?? []).map((m: any) => ({
      id: m.id, status: m.status?.value ?? "unknown",
      ctx: m.meta?.n_ctx ?? m.meta?.n_ctx_train ?? null,
      vision: !!m.architecture?.input_modalities?.includes("image"),
    }));
  } catch { return []; }
}

/** (Re)register every model the server currently exposes, under friendly names. */
async function syncProvider(pi: ExtensionAPI) {
  const live = await serverModels();
  const known = new Map<string, Quant>();
  for (const q of allQuants) {
    known.set(modelId(q), q);                       // Qwen3.8-27B-UD-Q4_K_XL
    known.set(hfRef(q), q);                         // unsloth/...-GGUF:Q4_K_XL
    known.set(hfRef(q).toLowerCase(), q);
  }
  pi.registerProvider("localcode", {
    baseUrl: `${BASE}/v1`, apiKey: "local", api: "openai-completions",
    models: live.map((m) => {
      const q =
        known.get(m.id) ??
        known.get(m.id.toLowerCase()) ??
        known.get(m.id.split("/").pop() ?? "") ??
        allQuants.find((x) => m.id.includes(quantName(x)) && m.id.toLowerCase().includes(groupOf(x).name.split(" ")[0].toLowerCase()));
      return {
        id: m.id, name: q ? pretty(q) : m.id, reasoning: false,
        input: m.vision ? ["text", "image"] : ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: m.ctx ?? 32768, maxTokens: 8192,
      };
    }),
  });
  return live;
}

async function post(path: string, body: unknown) {
  return fetch(`${BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
}

/** Ask llama-server to pull a model from HF, reporting progress in the status line. */
async function download(ctx: ExtensionCommandContext, q: Quant): Promise<boolean> {
  ctx.ui.setStatus("localcode", `downloading ${pretty(q)} …`);
  const started = await post("/models", { model: hfRef(q) });
  if (!started.ok) {
    ctx.ui.setStatus("localcode", "");
    ctx.ui.notify(`Download could not start: ${await started.text()}`, "error");
    return false;
  }
  for (;;) {
    await new Promise((r) => setTimeout(r, 1500));
    if (onDisk(q)) break;
    const entry = (await serverModels()).find((m) => m.id === modelId(q) || m.id === hfRef(q));
    if (entry && entry.status !== "downloading") break;
    ctx.ui.setStatus("localcode", `downloading ${pretty(q)} … ${q.size_gb.toFixed(1)} GB`);
  }
  ctx.ui.setStatus("localcode", "");
  return true;
}

async function useModel(pi: ExtensionAPI, ctx: ExtensionCommandContext, q: Quant) {
  ctx.ui.setStatus("localcode", `loading ${pretty(q)} …`);
  await post("/models/load", { model: modelId(q) }).catch(() => {});
  const live = await syncProvider(pi);
  ctx.ui.setStatus("localcode", "");
  const id = live.find((m) => m.id === modelId(q))?.id ?? modelId(q);
  const model = ctx.modelRegistry.find("localcode", id);
  if (!model) { ctx.ui.notify(`${pretty(q)} is ready. Select it with /model.`, "warning"); return; }
  await pi.setModel(model);
  ctx.ui.notify(`Model: ${pretty(q)}`, "info");
}

const groupLine = (g: Group) => {
  const n = g.quants.filter(onDisk).length;
  const star = RECOMMENDED && g.quants.some((q) => q.key === RECOMMENDED.key) ? " ★" : "";
  return `${g.name}${star} — ${g.maker} · ${n > 0 ? `${n} of ${g.quants.length} downloaded` : `${g.quants.length} to download`}`;
};
const quantLine = (q: Quant) => {
  const he = q.humaneval ? ` · HumanEval ${Math.round(q.humaneval * 100)}%` : "";
  const star = RECOMMENDED && q.key === RECOMMENDED.key ? " ★ recommended for your Mac" : "";
  return `${onDisk(q) ? "✓" : "↓"} ${quantName(q)} — ${q.size_gb.toFixed(1)} GB${onDisk(q) ? ", on disk" : " download"}${he}${star}`;
};
const pick = <T,>(c: unknown, list: T[], render: (t: T) => string): T | undefined =>
  typeof c === "number" ? list[c] : list.find((t) => render(t) === c);

async function browse(pi: ExtensionAPI, ctx: ExtensionCommandContext, firstRun = false) {
  if (!ctx.hasUI) return;
  const ordered = [...groups].sort((a, b) => b.quants.filter(onDisk).length - a.quants.filter(onDisk).length || a.name.localeCompare(b.name));
  const title = firstRun
    ? `Choose a model to get started — ${RAM_GB} GB Mac · ★ recommended for you`
    : `Models — ${RAM_GB} GB Mac · ✓ downloaded · ↓ downloadable · ★ recommended`;
  const g = pick(await ctx.ui.select(title, ordered.map(groupLine)), ordered, groupLine);
  if (!g) return;
  const q = pick(await ctx.ui.select(`${g.name} — ${g.maker} · ${g.license}`, g.quants.map(quantLine)), g.quants, quantLine);
  if (!q) return;

  if (!onDisk(q)) {
    const ok = await ctx.ui.confirm(`Download ${g.name} ${quantName(q)}?`,
      `${q.size_gb.toFixed(1)} GB · one-time download, cached for future launches\nLicense: ${g.license}`);
    if (!ok) return;
    if (!(await download(ctx, q))) return;
  }
  await useModel(pi, ctx, q);
}

export default async function (pi: ExtensionAPI) {
  // Register at startup so --list-models and --model work before any session.
  await syncProvider(pi);

  // thinking off, exactly as localcode's runtime does it
  pi.on("before_provider_request", (event) => ({
    ...(event.payload as Record<string, unknown>),
    chat_template_kwargs: { enable_thinking: false },
  }));

  // In router mode the server serves only loaded models, so selecting one from
  // any picker (ours or the built-in /model) must load it first.
  pi.on("model_select", async (event, ctx) => {
    const m: any = (event as any).model ?? (event as any).next;
    if (!m || m.provider !== "localcode") return;
    try {
      const live = await serverModels();
      const entry = live.find((x) => x.id === m.id);
      if (!entry || entry.status === "loaded" || entry.status === "sleeping") return;
      if (ctx.hasUI) ctx.ui.setStatus("localcode", `loading ${m.name ?? m.id} …`);
      await post("/models/load", { model: m.id });
      for (let i = 0; i < 120; i++) {
        const now = (await serverModels()).find((x) => x.id === m.id);
        if (now && (now.status === "loaded" || now.status === "sleeping")) break;
        await new Promise((r) => setTimeout(r, 1000));
      }
    } finally {
      if (ctx.hasUI) ctx.ui.setStatus("localcode", "");
    }
  });

  pi.registerCommand("models", {
    description: "Choose a model — browse by family, then quant",
    handler: (_a, ctx) => browse(pi, ctx),
  });

  pi.on("session_start", async (_e, ctx) => {
    await syncProvider(pi);
    // Router mode serves only loaded models; make sure the active one is up.
    const active: any = ctx.model;
    if (active?.provider === "localcode") {
      const entry = (await serverModels()).find((m) => m.id === active.id);
      if (entry && entry.status !== "loaded" && entry.status !== "sleeping") {
        if (ctx.hasUI) ctx.ui.setStatus("localcode", `loading ${active.name ?? active.id} …`);
        await post("/models/load", { model: active.id }).catch(() => {});
        for (let i = 0; i < 120; i++) {
          const now = (await serverModels()).find((m) => m.id === active.id);
          if (now && (now.status === "loaded" || now.status === "sleeping")) break;
          await new Promise((r) => setTimeout(r, 1000));
        }
        if (ctx.hasUI) ctx.ui.setStatus("localcode", "");
      }
    }
    // First run: nothing downloaded yet → open the picker straight away.
    if (ctx.hasUI && ctx.mode === "tui" && !allQuants.some(onDisk)) {
      await browse(pi, ctx as ExtensionCommandContext, true);
    }
  });
}
