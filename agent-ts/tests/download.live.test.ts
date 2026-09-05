// Live download verification — run explicitly (downloads ~340MB from HF):
//   LOCALCODE_MODELS_DIR=$(mktemp -d) npx vitest run tests/download.live.test.ts
import { describe, expect, it } from "vitest";
import { statSync } from "node:fs";
import { join } from "node:path";

describe("hfDownload (live)", () => {
  it("streams a real gguf into MODELS_DIR", async () => {
    const { hfDownload } = await import("../extensions/localcode.ts");
    const ctx: any = { ui: { setStatus: () => {}, notify: (m: string) => console.log("notify:", m) } };
    const ok = await hfDownload(ctx, "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "main",
      "qwen2.5-0.5b-instruct-q2_k.gguf", "qwen2.5-0.5b-instruct-q2_k.gguf", "tiny test model", 0.4);
    expect(ok).toBe(true);
    const st = statSync(join(process.env.LOCALCODE_MODELS_DIR!, "qwen2.5-0.5b-instruct-q2_k.gguf"));
    expect(st.size).toBeGreaterThan(100e6);
    console.log("downloaded bytes:", st.size);
  }, 600_000);
});
