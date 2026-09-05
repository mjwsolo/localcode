/**
 * code_navigation — "search code by structure", the last README promise with
 * no answer on this front end. Port of tools/code_navigation.py.
 *
 * Python files get real definition extraction (functions, classes, top-level
 * assignments); TS/JS get declaration-pattern extraction; references use a
 * word-boundary match everywhere. Deterministic, no model guessing.
 */
import { readFileSync, statSync, readdirSync } from "node:fs";
import { join, relative, resolve, extname } from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const SKIP = new Set([".git", ".localcode", ".localcode-agent", "node_modules", ".venv", "venv", "dist", "build", "__pycache__"]);

function* files(root: string): Generator<string> {
  const st = statSync(root);
  if (st.isFile()) { yield root; return; }
  for (const name of readdirSync(root)) {
    if (SKIP.has(name)) continue;
    const p = join(root, name);
    try {
      const s = statSync(p);
      if (s.isDirectory()) yield* files(p);
      else if (s.isFile()) yield p;
    } catch {}
  }
}

type Sym = [name: string, line: number, kind: string];

/** Python: def/class/top-level assignment. Mirrors the ast walk closely enough
 *  for navigation (indentation-anchored declarations, not a full parser). */
function pySymbols(text: string): Sym[] {
  const out: Sym[] = [];
  text.split("\n").forEach((ln, i) => {
    let m = ln.match(/^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)/);
    if (m) { out.push([m[2], i + 1, "function"]); return; }
    m = ln.match(/^(\s*)class\s+([A-Za-z_]\w*)/);
    if (m) { out.push([m[2], i + 1, "class"]); return; }
    m = ln.match(/^([A-Za-z_]\w*)\s*(?::[^=]+)?=\s*/);
    if (m && !ln.trimStart().startsWith("#")) out.push([m[1], i + 1, "variable"]);
  });
  return out;
}

function tsSymbols(text: string): Sym[] {
  const out: Sym[] = [];
  text.split("\n").forEach((ln, i) => {
    let m = ln.match(/^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)/);
    if (m) { out.push([m[1], i + 1, "function"]); return; }
    m = ln.match(/^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)/);
    if (m) { out.push([m[1], i + 1, "class"]); return; }
    m = ln.match(/^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)/);
    if (m) { out.push([m[1], i + 1, "variable"]); return; }
    m = ln.match(/^\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)/);
    if (m) out.push([m[1], i + 1, "type"]);
  });
  return out;
}

const symbolsFor = (p: string, text: string): Sym[] =>
  extname(p) === ".py" ? pySymbols(text)
  : [".ts", ".tsx", ".js", ".jsx", ".mjs"].includes(extname(p)) ? tsSymbols(text)
  : [];

const codeNav = defineTool({
  name: "code_navigation",
  label: "Code navigation",
  description: "Find code symbols, definitions, or references deterministically. Prefer this over repeated grep/read guesses.",
  promptSnippet: "code_navigation: list symbols, find a definition, or find references",
  parameters: Type.Object({
    action: Type.Union([Type.Literal("symbols"), Type.Literal("definition"), Type.Literal("references")]),
    symbol: Type.Optional(Type.String({ description: "Required for definition/references" })),
    path: Type.Optional(Type.String({ description: "File or directory, repo-relative (default: whole repo)" })),
    max_results: Type.Optional(Type.Number()),
  }),
  async execute(_id, params, _signal, _onUpdate, ctx) {
    const repo = resolve(ctx.cwd);
    const target = resolve(repo, params.path ?? ".");
    if (!target.startsWith(repo)) return { content: [{ type: "text", text: "Error: path must stay inside the repository." }], details: {} };
    const symbol = (params.symbol ?? "").trim();
    if ((params.action === "definition" || params.action === "references") && !symbol)
      return { content: [{ type: "text", text: `Error: symbol is required for ${params.action}.` }], details: {} };
    const limit = Math.max(1, Math.min(params.max_results ?? 50, 500));
    const boundary = symbol ? new RegExp(`(?<![A-Za-z0-9_])${symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![A-Za-z0-9_])`) : null;
    const results: string[] = [];
    try {
      outer: for (const p of files(target)) {
        const rel = relative(repo, p);
        let text: string;
        try { text = readFileSync(p, "utf8"); } catch { continue; }
        if (params.action !== "references") {
          for (const [name, line, kind] of symbolsFor(p, text)) {
            if (params.action === "symbols" || name === symbol) results.push(`${rel}:${line}: ${kind} ${name}`);
            if (results.length >= limit) break outer;
          }
        } else {
          let n = 0;
          for (const ln of text.split("\n")) {
            n++;
            if (boundary!.test(ln)) results.push(`${rel}:${n}: ${ln.trim().slice(0, 240)}`);
            if (results.length >= limit) break outer;
          }
        }
      }
    } catch (e) {
      return { content: [{ type: "text", text: `Error: ${e}` }], details: {} };
    }
    const text = results.length
      ? results.slice(0, limit).join("\n") + (results.length >= limit ? `\n[limited to ${limit} results]` : "")
      : `No ${params.action} results found${symbol ? ` for '${symbol}'.` : "."}`;
    return { content: [{ type: "text", text }], details: { count: results.length } };
  },
});

export default function (pi: ExtensionAPI) {
  pi.registerTool(codeNav);
}
