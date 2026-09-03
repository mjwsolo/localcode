/**
 * localcode's approval gate, ported to pi's `tool_call` hook.
 *
 * pi ships no permission system by design — it points at containers instead.
 * localcode promises the opposite ("asks before anything risky"), so this
 * restores it: dangerous commands are blocked outright, risky ones prompt, and
 * writes to sensitive files prompt. Rules mirror permissions_v2.py.
 */
import { basename } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BLOCKED: RegExp[] = [
  /rm\s+-rf\s+\/(?!\w)/i, /mkfs\./i, /dd\s+if=/i, />\s*\/dev\/sd/i,
  /chmod\s+-R\s+777\s+\//i, /curl.*\|\s*(bash|sh|zsh)/i, /wget.*\|\s*(bash|sh|zsh)/i,
  /sudo\s+rm/i, /:\(\)\{.*\}/, /git\s+push\s+.*--force\s+.*(main|master)/i,
  /git\s+reset\s+--hard\s+origin/i, /drop\s+(table|database)/i, /truncate\s+table/i,
  />\s*\/etc\//i, /python.*-c.*import\s+os.*remove/i,
];

const CONFIRM: RegExp[] = [
  /\brm\s+/i, /git\s+push/i, /git\s+checkout\s+--/i, /git\s+reset/i,
  /pip\s+(un)?install/i, /npm\s+(un)?install/i, /\bmv\s+/i, /\bdocker\s+/i, /\bkubectl\s+/i,
];

const SENSITIVE = [
  ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519", "id_ecdsa",
  "credentials", "credentials.json", ".ssh/", "shadow", "passwd", ".git/config",
  ".gitconfig", "secrets.yaml", "secrets.json", ".aws/credentials", ".npmrc",
];
const isSensitive = (p: string) => {
  const s = p.toLowerCase();
  return SENSITIVE.some((n) => s.includes(n.toLowerCase())) ||
    /(^|[._-])(token|api[_-]?key)([._-]|$)/i.test(basename(s));
};

export default function (pi: ExtensionAPI) {
  /** approved for the rest of this session, so we ask once not every time */
  const approved = new Set<string>();

  pi.on("tool_call", async (event, ctx) => {
    const input = event.input as Record<string, unknown>;

    if (event.toolName === "bash") {
      const cmd = String(input.command ?? "");
      for (const re of BLOCKED) {
        if (re.test(cmd)) {
          return { block: true, reason: `localcode blocked a dangerous command: ${cmd.slice(0, 80)}` };
        }
      }
      const risky = CONFIRM.find((re) => re.test(cmd));
      if (risky && !approved.has(cmd)) {
        if (!ctx.hasUI) {
          return { block: true, reason: `Needs approval, and there is no UI to ask: ${cmd.slice(0, 80)}` };
        }
        const ok = await ctx.ui.confirm("Run this command?", cmd.slice(0, 400));
        if (!ok) return { block: true, reason: "You declined this command." };
        approved.add(cmd);
      }
      return;
    }

    if (event.toolName === "write" || event.toolName === "edit") {
      const path = String(input.path ?? input.file ?? "");
      if (path && isSensitive(path)) {
        if (!ctx.hasUI) return { block: true, reason: `Refusing to modify a sensitive file: ${path}` };
        const ok = await ctx.ui.confirm("Modify this sensitive file?", path);
        if (!ok) return { block: true, reason: "You declined this write." };
      }
    }
  });
}
