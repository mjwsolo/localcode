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
import { createWriteStream, existsSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir, totalmem } from "node:os";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";

type Quant = {
  key: string; label: string; filename: string; hf_repo: string; size_gb: number;
  revision: string; humaneval: number | null; recommended_at_ram_gb: number | null;
  mmproj_filename: string | null; mmproj_hf_filename: string | null;
};
type Group = { key: string; name: string; maker: string; architecture: string; license: string; notes: string; hf_repo: string; quants: Quant[] };

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

const QUANT_RE = /(UD-)?((?:IQ|Q|BF|F)\d+(?:_[A-Z0-9]+)*)\.gguf$/i;
const CACHE_DIR = join(process.env.LOCALCODE_HOME ?? join(homedir(), ".localcode"), "cache", "hf_quants");

type HfQuant = { label: string; filename: string; size_gb: number };

/** Every .gguf the repo ships — the same live listing localcode's picker uses. */
async function fetchQuants(repo: string): Promise<HfQuant[]> {
  const cache = join(CACHE_DIR, `${repo.replace(/\//g, "__")}.json`);
  try {
    const st = statSync(cache);
    if (Date.now() - st.mtimeMs < 24 * 3600 * 1000) return JSON.parse(readFileSync(cache, "utf8"));
  } catch {}
  try {
    const r = await fetch(`https://huggingface.co/api/models/${repo}/tree/main?recursive=true`,
                          { signal: AbortSignal.timeout(15_000) });
    const tree = (await r.json()) as any[];
    const out: HfQuant[] = [];
    for (const e of tree) {
      const path = String(e?.path ?? "");
      if (!path.toLowerCase().endsWith(".gguf")) continue;
      const base = path.split("/").pop()!;
      if (/^mmproj/i.test(base)) continue;
      const m = base.match(QUANT_RE);
      if (!m) continue;
      const bytes = Number(e?.lfs?.size ?? e?.size ?? 0);
      out.push({ label: `${m[1] ?? ""}${m[2]}`.toUpperCase(), filename: base, size_gb: bytes / 1e9 });
    }
    const best = new Map<string, HfQuant>();
    for (const q of out) {
      const prev = best.get(q.label);
      if (!prev || q.size_gb > prev.size_gb) best.set(q.label, q);
    }
    const deduped = [...best.values()].filter((q) => q.size_gb >= 0.8);
    deduped.sort((a, b) => a.size_gb - b.size_gb);
    out.length = 0; out.push(...deduped);
    try { mkdirSync(CACHE_DIR, { recursive: true }); writeFileSync(cache, JSON.stringify(out)); } catch {}
    return out;
  } catch {
    try { return JSON.parse(readFileSync(cache, "utf8")); } catch { return []; }
  }
}

/** Same rule as models_catalog.recommend(): weights inside ~55% of unified RAM. */
const fitBadge = (gb: number) => (gb <= 0.55 * RAM_GB ? "fits" : gb <= 0.65 * RAM_GB ? "tight" : "too big");
const FIT_GLYPH: Record<string, string> = { fits: "✓", tight: "~", "too big": "✗" };

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

/** Download a file straight from Hugging Face into MODELS_DIR, with progress
 *  and a .part temp file so an interrupted download never looks complete.
 *  (The router's own POST /models download proved to be a silent no-op in
 *  testing, so we do it ourselves, the way localcode's Python downloader does.) */
export async function hfDownload(
  ctx: ExtensionCommandContext, repo: string, revision: string, hfFile: string,
  destName: string, label: string, sizeGb: number,
): Promise<boolean> {
  const url = `https://huggingface.co/${repo}/resolve/${revision || "main"}/${hfFile}`;
  const dest = join(MODELS_DIR, destName);
  const part = `${dest}.part`;
  try {
    const r = await fetch(url, { redirect: "follow" });
    if (!r.ok || !r.body) { ctx.ui.notify(`Download failed: HTTP ${r.status} for ${hfFile}`, "error"); return false; }
    const total = Number(r.headers.get("content-length") ?? 0) || sizeGb * 1e9;
    const out = createWriteStream(part);
    let done = 0, lastPct = -1;
    for await (const chunk of r.body as any) {
      out.write(chunk); done += chunk.length;
      const pct = Math.floor((done / total) * 100);
      if (pct !== lastPct) {
        lastPct = pct;
        ctx.ui.setStatus("localcode", `downloading ${label} … ${pct}% of ${(total / 1e9).toFixed(1)} GB`);
      }
    }
    await new Promise((res, rej) => out.end((e: any) => (e ? rej(e) : res(null))));
    renameSync(part, dest);
    return true;
  } catch (e) {
    try { unlinkSync(part); } catch {}
    ctx.ui.notify(`Download failed: ${e}`, "error");
    return false;
  } finally {
    ctx.ui.setStatus("localcode", "");
  }
}

async function download(ctx: ExtensionCommandContext, q: Quant): Promise<boolean> {
  if (!(await hfDownload(ctx, q.hf_repo, q.revision, q.filename, q.filename, pretty(q), q.size_gb))) return false;
  // vision models need their mmproj sidecar next to the weights
  if (q.mmproj_filename) {
    await hfDownload(ctx, q.hf_repo, q.revision, q.mmproj_hf_filename ?? q.mmproj_filename,
                     q.mmproj_filename, `${pretty(q)} (vision sidecar)`, 1);
  }
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
  // Downloaded families first, then the ★ recommended family, then the rest —
  // so a first run's cursor rests on the model we actually recommend, never on
  // a research model that happens to sort first alphabetically.
  const recGroup = RECOMMENDED ? groupOf(RECOMMENDED).key : "";
  const ordered = [...groups].sort((a, b) =>
    b.quants.filter(onDisk).length - a.quants.filter(onDisk).length ||
    Number(b.key === recGroup) - Number(a.key === recGroup) ||
    a.name.localeCompare(b.name));
  const title = firstRun
    ? `Choose a model to get started — ${RAM_GB} GB Mac · ★ recommended for you`
    : `Models — ${RAM_GB} GB Mac · ✓ downloaded · ↓ downloadable · ★ recommended`;
  // family list → quant list, with a way back up
  for (;;) {
    const g = pick(await ctx.ui.select(title, ordered.map(groupLine)), ordered, groupLine);
    if (!g) return;

    const BACK = "← Back to all models";
    if (ctx.hasUI) ctx.ui.setStatus("localcode", `loading quants for ${g.name} …`);
    const live = await fetchQuants(g.hf_repo);
    if (ctx.hasUI) ctx.ui.setStatus("localcode", "");

    const rows = live.map((h) => {
      const downloaded = existsSync(join(MODELS_DIR, h.filename));
      const fit = fitBadge(h.size_gb);
      const star = RECOMMENDED && RECOMMENDED.filename === h.filename ? " ★" : "";
      const mark = downloaded ? "✓ downloaded" : `${FIT_GLYPH[fit]} ${fit}`;
      return `${h.label.padEnd(12)} ${h.size_gb.toFixed(1)} GB · ${mark}${star}`;
    });
    if (rows.length === 0) rows.push("(could not list quants — check your connection)");
    rows.push(BACK);

    const chosen = await ctx.ui.select(`${g.name} — ${g.maker} · ${g.license}`, rows);
    if (chosen === undefined) return;
    const idx = typeof chosen === "number" ? chosen : rows.indexOf(String(chosen));
    if (idx < 0 || rows[idx] === BACK) continue;
    const h = live[idx];
    if (!h) continue;
    await chooseHf(pi, ctx, g, h);
    return;
  }
}

async function chooseHf(pi: ExtensionAPI, ctx: ExtensionCommandContext, g: Group, h: HfQuant) {
  const id = h.filename.replace(/\.gguf$/, "");
  const ref = `${g.hf_repo}:${h.label.replace(/^UD-/, "")}`;
  if (!existsSync(join(MODELS_DIR, h.filename))) {
    const ok = await ctx.ui.confirm(`Download ${g.name} ${h.label}?`,
      `${h.size_gb.toFixed(1)} GB · one-time download, cached for future launches\nLicense: ${g.license}`);
    if (!ok) return;
    if (!(await hfDownload(ctx, g.hf_repo, "main", h.filename, h.filename, `${g.name} ${h.label}`, h.size_gb))) return;
  }
  ctx.ui.setStatus("localcode", `loading ${g.name} ${h.label} …`);
  await post("/models/load", { model: id }).catch(() => {});
  await syncProvider(pi);
  ctx.ui.setStatus("localcode", "");
  const model = ctx.modelRegistry.find("localcode", id) ?? ctx.modelRegistry.find("localcode", ref);
  if (!model) { ctx.ui.notify(`${g.name} ${h.label} is ready — pick it with /model.`, "warning"); return; }
  await pi.setModel(model);
  ctx.ui.notify(`Model: ${g.name} · ${h.label}`, "info");
}

async function chooseUnused(pi: ExtensionAPI, ctx: ExtensionCommandContext, g: Group, q: Quant) {
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

  for (const name of ["model"]) {
    pi.registerCommand(name, {
      description: "Choose a model — browse by family, then quant",
      handler: (_a, ctx) => browse(pi, ctx),
    });
  }

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
