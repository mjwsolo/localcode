/**
 * localcode's model provider for pi.
 *
 * Registers every model the llama.cpp router (our bundled turboquant
 * llama-server) reports, at startup, with no interactive /login step. This is
 * the documented async-factory pattern from pi's extensions.md, and it is what
 * lets localcode own model naming instead of inheriting pi's.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BASE = (process.env.LLAMA_BASE_URL ?? "http://127.0.0.1:8080").replace(/\/+$/, "");

type RouterModel = {
  id: string;
  status?: { value?: string };
  meta?: { n_ctx?: number; n_ctx_train?: number };
  architecture?: { input_modalities?: string[] };
};

export default async function (pi: ExtensionAPI) {
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
      name: m.id,
      reasoning: false,
      input: m.architecture?.input_modalities?.includes("image") ? ["text", "image"] : ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: m.meta?.n_ctx ?? m.meta?.n_ctx_train ?? 32768,
      maxTokens: 8192,
    })),
  });
}
