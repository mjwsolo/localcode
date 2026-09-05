/**
 * Value-level secret redaction — port of localcode's redaction.py.
 *
 * pi persists the verbatim transcript (tool results included) to session
 * files on disk. A `cat .env`, a pasted API key, or a model echoing a token
 * all land there permanently. This scrubs the VALUE wherever it appears in
 * tool results and user input, before pi stores or resends it.
 *
 * Scope is deliberately narrow, same as upstream: only high-precision,
 * self-identifying credential formats (vendor prefixes, structural JWTs, PEM
 * blocks). No entropy heuristics — on a coding transcript those fire on git
 * SHAs, hashes and minified JS, and corrupting the user's own content in
 * their session log is worse than missing an unprefixed secret.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PATTERNS: [RegExp, string][] = [
  [/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g, "[redacted:private-key]"],
  [/\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b/g, "[redacted:aws-key]"],
  [/\bgithub_pat_[A-Za-z0-9_]{20,}/g, "[redacted:github-token]"],
  [/\bgh[pousr]_[A-Za-z0-9]{20,}/g, "[redacted:github-token]"],
  [/\bsk-ant-[A-Za-z0-9_-]{20,}/g, "[redacted:anthropic-key]"],
  [/\bsk-[A-Za-z0-9_-]{20,}/g, "[redacted:openai-key]"],
  [/\bsk_live_[0-9a-zA-Z]{20,}/g, "[redacted:stripe-key]"],
  [/\bhf_[A-Za-z0-9]{20,}/g, "[redacted:hf-token]"],
  [/\bxox[baprs]-[A-Za-z0-9-]{10,}/g, "[redacted:slack-token]"],
  [/\bAIza[0-9A-Za-z_-]{35}/g, "[redacted:google-key]"],
  [/\bnpm_[A-Za-z0-9]{36}/g, "[redacted:npm-token]"],
  [/\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/g, "[redacted:jwt]"],
];

const CANDIDATES = ["AKIA", "ASIA", "ABIA", "ACCA", "ghp_", "gho_", "ghu_", "ghs_", "ghr_",
  "github_pat_", "sk-", "sk_live_", "hf_", "xox", "AIza", "npm_", "eyJ", "PRIVATE KEY"];
const MIN_LEN = 14;

export function scrub(text: string): string {
  if (typeof text !== "string" || text.length < MIN_LEN) return text;
  if (!CANDIDATES.some((c) => text.includes(c))) return text;
  let out = text;
  for (const [re, marker] of PATTERNS) out = out.replace(re, marker);
  return out;
}

export default function (pi: ExtensionAPI) {
  // Tool results are the main leak path (cat .env, env dumps, curl output).
  pi.on("tool_result", (event) => {
    const content = (event as any).result?.content ?? (event as any).content;
    if (!Array.isArray(content)) return;
    let changed = false;
    const next = content.map((c: any) => {
      if (c?.type === "text" && typeof c.text === "string") {
        const s = scrub(c.text);
        if (s !== c.text) { changed = true; return { ...c, text: s }; }
      }
      return c;
    });
    if (changed) return { content: next };
  });

  // And anything the user pastes straight into the prompt.
  pi.on("input", (event) => {
    const s = scrub(event.text);
    if (s !== event.text) return { action: "transform" as const, text: s };
  });
}
