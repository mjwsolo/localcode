import { describe, expect, it } from "vitest";
import { statSync } from "node:fs";
import { join } from "node:path";
describe("real-size download", () => {
  it("streams gemma-4-12b Q4 (7.4GB) through the shipped function", async () => {
    const { hfDownload } = await import("$HOME/Desktop/Github/localcode-pi/agent-ts/extensions/localcode.ts");
    const ctx: any = { ui: { setStatus: (_: string, s: string) => s && process.stdout.write(`\r${s}   `), notify: (m: string) => console.log("notify:", m) } };
    const ok = await hfDownload(ctx, "unsloth/gemma-4-12b-it-GGUF", "main",
      "gemma-4-12b-it-UD-Q4_K_XL.gguf", "gemma-4-12b-it-UD-Q4_K_XL.gguf", "Gemma 4 12B Q4", 7.4);
    expect(ok).toBe(true);
    const st = statSync(join(process.env.LOCALCODE_MODELS_DIR!, "gemma-4-12b-it-UD-Q4_K_XL.gguf"));
    console.log("\nbytes:", st.size);
    expect(st.size).toBeGreaterThan(7e9);
  }, 3600_000);
});
