/**
 * /models — localcode's model browser, built only from pi's native UI
 * primitives (ctx.ui.select / confirm / notify). No TUI code of our own.
 *
 * Two levels, matching localcode's catalog:
 *   1. model family   — Gemma 4 26B-A4B (Google) · 1 of 2 downloaded
 *   2. quant          — ✓ Q8 (28.0 GB, on disk)  /  ↓ IQ3_S (11.2 GB download) ★
 *
 * ✓ = downloaded and ready   ↓ = downloadable   ★ = recommended for this Mac
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir, totalmem } from "node:os";
import { spawn } from "node:child_process";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

type Quant = {
  key: string; label: string; filename: string; hf_repo: string;
  size_gb: number; revision: string; humaneval: number | null;
  recommended_at_ram_gb: number | null;
  mmproj_filename: string | null; mmproj_hf_filename: string | null;
};
type Group = {
  key: string; name: string; maker: string; architecture: string;
  license: string; notes: string; quants: Quant[];
};

const MODELS_DIR = process.env.LOCALCODE_MODELS_DIR ?? join(homedir(), ".local/share/localcode/models");
const { groups } = JSON.parse(readFileSync(new URL("./catalog.json", import.meta.url), "utf8")) as { groups: Group[] };
const RAM_GB = Math.round(totalmem() / 1024 ** 3);

const here = (q: Quant) => existsSync(join(MODELS_DIR, q.filename));
const isRecommended = (q: Quant) => q.recommended_at_ram_gb !== null && q.recommended_at_ram_gb <= RAM_GB;

/** The single model localcode recommends: largest whose threshold this Mac meets. */
const RECOMMENDED = groups
  .flatMap((g) => g.quants)
  .filter(isRecommended)
  .sort((a, b) => (b.recommended_at_ram_gb ?? 0) - (a.recommended_at_ram_gb ?? 0))[0];

/** Short quant name: "Qwen3.8-27B-UD-Q4_K_XL.gguf" -> "UD-Q4_K_XL" */
function quantName(q: Quant): string {
  const m = q.filename.replace(/\.gguf$/, "").match(/((UD-)?(IQ|Q|BF)[0-9][^-]*(_[A-Z0-9]+)*)$/);
  return m ? m[1] : q.label;
}

function groupLine(g: Group): string {
  const n = g.quants.filter(here).length;
  const status = n > 0 ? `${n} of ${g.quants.length} downloaded` : `${g.quants.length} available to download`;
  const star = g.quants.some((q) => RECOMMENDED && q.key === RECOMMENDED.key) ? " ★" : "";
  return `${g.name}${star} — ${g.maker} · ${status}`;
}

function quantLine(q: Quant): string {
  const mark = here(q) ? "✓" : "↓";
  const size = here(q) ? `${q.size_gb.toFixed(1)} GB, on disk` : `${q.size_gb.toFixed(1)} GB download`;
  const star = RECOMMENDED && q.key === RECOMMENDED.key ? " ★ recommended for your Mac" : "";
  const he = q.humaneval ? ` · HumanEval ${Math.round(q.humaneval * 100)}%` : "";
  return `${mark} ${quantName(q)} — ${size}${he}${star}`;
}

const pick = <T,>(choice: unknown, list: T[], render: (t: T) => string): T | undefined =>
  typeof choice === "number" ? list[choice] : list.find((t) => render(t) === choice);

export default function (pi: ExtensionAPI) {
  pi.registerCommand("models", {
    description: "Browse models: switch between downloaded ones, or download more",
    handler: async (_args, ctx) => {
      if (!ctx.hasUI) return;

      // Level 1 — families, downloaded ones first
      const ordered = [...groups].sort(
        (a, b) => b.quants.filter(here).length - a.quants.filter(here).length || a.name.localeCompare(b.name),
      );
      const gChoice = await ctx.ui.select(
        `Models — ${RAM_GB} GB Mac · ✓ downloaded · ↓ downloadable · ★ recommended`,
        ordered.map(groupLine),
      );
      if (gChoice === undefined) return;
      const group = pick(gChoice, ordered, groupLine);
      if (!group) return;

      // Level 2 — quants within the family
      const qChoice = await ctx.ui.select(
        `${group.name} — ${group.maker} · ${group.license}`,
        group.quants.map(quantLine),
      );
      if (qChoice === undefined) return;
      const q = pick(qChoice, group.quants, quantLine);
      if (!q) return;

      if (here(q)) {
        const id = q.filename.replace(/\.gguf$/, "");
        const model = ctx.modelRegistry.find("localcode", id);
        if (!model) {
          ctx.ui.notify(`${group.name} ${quantName(q)} is downloaded but not being served. Load it with /llama first.`, "warning");
          return;
        }
        await pi.setModel(model);
        ctx.ui.notify(`Model: ${group.name} ${quantName(q)}`, "info");
        return;
      }

      const go = await ctx.ui.confirm(
        `Download ${group.name} ${quantName(q)}?`,
        `${q.size_gb.toFixed(1)} GB from ${q.hf_repo}\nLicense: ${group.license}`,
      );
      if (!go) return;
      ctx.ui.setStatus("localcode-models", `downloading ${group.name} ${quantName(q)}…`);
      const files = [q.filename, ...(q.mmproj_hf_filename ?? q.mmproj_filename ? [q.mmproj_hf_filename ?? q.mmproj_filename!] : [])];
      const code = await new Promise<number>((resolve) => {
        const p = spawn("hf", ["download", q.hf_repo, ...files, "--revision", q.revision, "--local-dir", MODELS_DIR], { stdio: "ignore" });
        p.on("close", (c) => resolve(c ?? 1));
      });
      ctx.ui.setStatus("localcode-models", "");
      ctx.ui.notify(
        code === 0 ? `Downloaded ${group.name} ${quantName(q)}. Open /models again to select it.`
                   : `Download failed (exit ${code}).`,
        code === 0 ? "info" : "error",
      );
    },
  });
}
