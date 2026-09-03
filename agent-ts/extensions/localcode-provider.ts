/**
 * localcode's model provider for pi.
 *
 * Registers every model the llama.cpp router (our bundled turboquant
 * llama-server) reports, at startup, with no interactive /login step. This is
 * the documented async-factory pattern from pi's extensions.md, and it is what
 * lets localcode own model naming instead of inheriting pi's.
 */
import { readFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Friendly display names, so the picker never shows a raw HF id like
// "unsloth/Muse-Glimmer-30B-GGUF:Q4_K_XL" next to a bare filename.
const CAT = JSON.parse(readFileSync(new URL("./catalog.json", import.meta.url), "utf8")) as {
  groups: { name: string; quants: { filename: string }[] }[];
};
const PRETTY = new Map<string, string>();
for (const g of CAT.groups)
  for (const q of g.quants) {
    const id = q.filename.replace(/\.gguf$/, "");
    const quant = id.match(/((UD-)?(IQ|Q|BF)[0-9][^-]*(_[A-Z0-9]+)*)$/)?.[1] ?? "";
    PRETTY.set(id, quant ? `${g.name} · ${quant}` : g.name);
  }
const pretty = (id: string) => PRETTY.get(id) ?? PRETTY.get(id.split("/").pop() ?? id) ?? id;

const BASE = (process.env.LLAMA_BASE_URL ?? "http://127.0.0.1:8080").replace(/\/+$/, "");

type RouterModel = {
  id: string;
  status?: { value?: string };
  meta?: { n_ctx?: number; n_ctx_train?: number };
  architecture?: { input_modalities?: string[] };
};

export default async function (pi: ExtensionAPI) {
  // Disable the model's thinking channel the way localcode's runtime.py does.
  // pi's --thinking flag only sets pi's own notion of thinking; llama.cpp needs
  // chat_template_kwargs.enable_thinking=false or the chat template's default
  // (thinking ON for Qwen 3.x) applies and the model burns tokens reasoning.
  pi.on("before_provider_request", (event) => ({
    ...event.payload,
    chat_template_kwargs: { enable_thinking: false },
  }));

  let models: RouterModel[] = [];
  try {
    const res = await fetch(`${BASE}/models`, { signal: AbortSignal.timeout(10_000) });
    models = ((await res.json()) as { data?: RouterModel[] }).data ?? [];
  } catch {
    return; // router not up yet; localcode's launcher owns that failure path
  }

  pi.registerProvider("localcode", {
    baseUrl: `${BASE}/v1`,
    apiKey: "local",  // llama-server started without --api-key accepts any value
    api: "openai-completions",
    models: models.map((m) => ({
      id: m.id,
      name: pretty(m.id),
      reasoning: false,
      input: m.architecture?.input_modalities?.includes("image") ? ["text", "image"] : ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: m.meta?.n_ctx ?? m.meta?.n_ctx_train ?? 32768,
      maxTokens: 8192,
    })),
  });
}
