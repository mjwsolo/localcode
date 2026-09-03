/**
 * /models — localcode's curated model picker, built entirely from pi's native
 * UI primitives (ctx.ui.select / confirm / notify). No TUI code of our own.
 *
 * Shows one list containing both halves:
 *   ✓ downloaded and ready        → select to switch the session to it
 *   ↓ in the catalog, not on disk → select to download it
 * with ★ marking the model localcode recommends for this Mac's memory.
 */
import { existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { homedir, totalmem } from "node:os";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

type Entry = {
  key: string; name: string; hf_repo: string; filename: string;
  size_gb: number; active_params: string; architecture: string;
  license: string; humaneval: number | null; notes: string;
  mmproj_filename: string | null; mmproj_hf_filename: string | null;
  revision: string; recommended_at_ram_gb: number | null;
};

const MODELS_DIR = process.env.LOCALCODE_MODELS_DIR ?? join(homedir(), ".local/share/localcode/models");
const CATALOG: Entry[] = JSON.parse(readFileSync(new URL("./catalog.json", import.meta.url), "utf8"));
const RAM_GB = Math.round(totalmem() / 1024 ** 3);

const onDisk = (e: Entry) => existsSync(join(MODELS_DIR, e.filename));
const gb = (bytes: number) => (bytes / 1024 ** 3).toFixed(1);

/** localcode recommends the largest model whose recommend-at threshold this Mac meets. */
function recommended(): Entry | undefined {
  const eligible = CATALOG.filter((e) => e.recommended_at_ram_gb !== null && e.recommended_at_ram_gb <= RAM_GB);
  return eligible.sort((a, b) => (b.recommended_at_ram_gb ?? 0) - (a.recommended_at_ram_gb ?? 0))[0];
}

function label(e: Entry, rec: Entry | undefined): string {
  const here = onDisk(e);
  const mark = here ? "✓" : "↓";
  const star = rec && e.key === rec.key ? " ★" : "";
  const size = here
    ? `${gb(statSync(join(MODELS_DIR, e.filename)).size)} GB on disk`
    : `${e.size_gb.toFixed(1)} GB download`;
  const he = e.humaneval ? ` · HumanEval ${Math.round(e.humaneval * 100)}%` : "";
  return `${mark} ${e.name}${star} — ${size} · ${e.active_params} active${he}`;
}

export default function (pi: ExtensionAPI) {
  pi.registerCommand("models", {
    description: "Browse localcode's model catalog: switch, or download",
    handler: async (_args, ctx) => {
      if (!ctx.hasUI) return;
      const rec = recommended();
      // downloaded first, then downloadable; each block largest-first
      const sorted = [...CATALOG].sort((a, b) =>
        Number(onDisk(b)) - Number(onDisk(a)) || b.size_gb - a.size_gb);
      const choice = await ctx.ui.select(
        `Models — ${RAM_GB} GB Mac · ★ = recommended for you`,
        sorted.map((e) => label(e, rec)),
      );
      if (choice === undefined) return;
      const picked = sorted[typeof choice === "number" ? choice : sorted.findIndex((e) => label(e, rec) === choice)];
      if (!picked) return;

      if (onDisk(picked)) {
        // Already here: switch the session to it. The router/server exposes it
        // under the gguf basename, which is what our provider registers.
        const id = picked.filename.replace(/\.gguf$/, "");
        const model = ctx.modelRegistry.find("localcode", id);
        if (!model) {
          ctx.ui.notify(`${picked.name} is on disk but the server is not serving it. Load it first (/llama), or restart with this model.`, "warning");
          return;
        }
        await pi.setModel(model);
        ctx.ui.notify(`Model: ${picked.name}`, "info");
        return;
      }

      const go = await ctx.ui.confirm(
        `Download ${picked.name}?`,
        `${picked.size_gb.toFixed(1)} GB from ${picked.hf_repo}\nLicense: ${picked.license}`,
      );
      if (!go) return;
      ctx.ui.setStatus("localcode-models", `downloading ${picked.name}…`);
      await new Promise<void>((resolve) => {
        const p = spawn("hf", ["download", picked.hf_repo, picked.filename,
          "--revision", picked.revision, "--local-dir", MODELS_DIR], { stdio: "ignore" });
        p.on("close", (code) => {
          ctx.ui.setStatus("localcode-models", "");
          ctx.ui.notify(code === 0 ? `Downloaded ${picked.name}. Run /models again to select it.`
                                   : `Download failed (exit ${code}).`, code === 0 ? "info" : "error");
          resolve();
        });
      });
    },
  });
}
