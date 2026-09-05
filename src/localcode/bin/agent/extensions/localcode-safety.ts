/**
 * localcode's safety net, on pi's `tool_call` hook.
 *
 * No prompts. Outright-dangerous commands (wipe a disk, rm -rf /, pipe curl
 * to a shell, force-push main) are blocked with a reason the model can read;
 * everything else runs.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BLOCKED: RegExp[] = [
  /rm\s+-rf\s+\/(?!\w)/i, /mkfs\./i, /dd\s+if=/i, />\s*\/dev\/sd/i,
  /chmod\s+-R\s+777\s+\//i, /curl.*\|\s*(bash|sh|zsh)/i, /wget.*\|\s*(bash|sh|zsh)/i,
  /sudo\s+rm/i, /:\(\)\{.*\}/, /git\s+push\s+.*--force\s+.*(main|master)/i,
  /git\s+reset\s+--hard\s+origin/i, /drop\s+(table|database)/i, /truncate\s+table/i,
  />\s*\/etc\//i, /python.*-c.*import\s+os.*remove/i,
];

export default function (pi: ExtensionAPI) {
  pi.on("tool_call", async (event) => {
    const input = event.input as Record<string, unknown>;

    if (event.toolName === "bash") {
      const cmd = String(input.command ?? "");
      for (const re of BLOCKED) {
        if (re.test(cmd)) {
          return { block: true, reason: `localcode blocked a dangerous command: ${cmd.slice(0, 80)}` };
        }
      }
      // No "Run this command?" prompts: the approval gate was removed on
      // 2026-09-05. It interrupted every npm install / rm / mv and added
      // nothing. Only the BLOCKED list above still applies.
      return;
    }
  });
}
