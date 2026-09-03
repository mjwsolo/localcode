/**
 * web_search + web_fetch — the two network tools localcode ships and pi does not.
 *
 * Ported from localcode's tools/web_search.py and tools/web_fetch.py, keeping
 * the two things that matter: the SSRF guard (the model picks the host and is
 * never prompted, so loopback/private/link-local/CGNAT are refused), and the
 * untrusted-data fence around anything fetched from the web.
 */
import { lookup } from "node:dns/promises";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

/** Mirrors injection_defense.wrap_untrusted: web text is data, never instructions. */
const fence = (body: string, source: string) =>
  `<UNTRUSTED_DATA source="${source}">\n${body}\n</UNTRUSTED_DATA>`;

function blockedIp(ip: string): boolean {
  if (ip.includes(":")) {
    const v6 = ip.toLowerCase();
    if (v6 === "::1" || v6 === "::" ) return true;
    if (v6.startsWith("fe80") || v6.startsWith("fc") || v6.startsWith("fd")) return true;
    const m = v6.match(/::ffff:(\d+\.\d+\.\d+\.\d+)$/);
    return m ? blockedIp(m[1]) : false;
  }
  const p = ip.split(".").map(Number);
  if (p.length !== 4 || p.some((n) => Number.isNaN(n))) return true;
  const [a, b] = p;
  if (a === 0 || a === 10 || a === 127) return true;              // unspecified, private, loopback
  if (a === 169 && b === 254) return true;                         // link-local incl. cloud metadata
  if (a === 172 && b >= 16 && b <= 31) return true;                // private
  if (a === 192 && b === 168) return true;                         // private
  if (a === 100 && b >= 64 && b <= 127) return true;               // CGNAT / tailnet
  if (a >= 224) return true;                                       // multicast + reserved
  return false;
}

async function checkUrl(url: string): Promise<string | null> {
  if (!/^https?:\/\//i.test(url)) return `Error: url must start with http(s)://; got ${url}`;
  let host: string;
  try { host = new URL(url).hostname; } catch { return `Error: could not parse url ${url}`; }
  try {
    const addrs = await lookup(host, { all: true });
    if (addrs.length === 0) return `Error: could not resolve ${host}`;
    for (const a of addrs) if (blockedIp(a.address)) return `Error: refusing to fetch a private or local address (${host})`;
  } catch { return `Error: could not resolve ${host}`; }
  return null;
}

const stripHtml = (html: string) =>
  html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/[ \t]+/g, " ").replace(/\n\s*\n\s*\n+/g, "\n\n").trim();

const webSearch = defineTool({
  name: "web_search",
  label: "Web search",
  description: "Search the web for documentation, APIs, error solutions.",
  promptSnippet: "web_search: search the web for docs, APIs and error messages",
  parameters: Type.Object({ query: Type.String({ description: "What to search for" }) }),
  async execute(_id, params, signal) {
    const q = params.query;
    try {
      // DuckDuckGo's no-key HTML endpoint, same source localcode's ddgs uses.
      const r = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(q)}`, {
        signal, headers: { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" },
      });
      const html = await r.text();
      const out: string[] = [];
      const re = /<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;
      for (let m = re.exec(html); m && out.length < 5; m = re.exec(html)) {
        const href = decodeURIComponent((m[1].match(/uddg=([^&]+)/)?.[1]) ?? m[1]);
        out.push(`**${stripHtml(m[2])}**\n${href}\n${stripHtml(m[3])}\n`);
      }
      const text = out.length ? fence(out.join("\n"), `web_search ${JSON.stringify(q)}`) : "No results found";
      return { content: [{ type: "text", text }], details: { results: out.length } };
    } catch (e) {
      return { content: [{ type: "text", text: `Search error: ${e}` }], details: { results: 0 } };
    }
  },
});

const webFetch = defineTool({
  name: "web_fetch",
  label: "Fetch URL",
  description: "Fetch a URL and return its readable text content.",
  promptSnippet: "web_fetch: fetch a URL and read its text",
  parameters: Type.Object({
    url: Type.String({ description: "http(s) URL to fetch" }),
    max_chars: Type.Optional(Type.Number({ description: "Truncate the body (default 20000)" })),
  }),
  async execute(_id, params, signal) {
    const bad = await checkUrl(params.url);
    if (bad) return { content: [{ type: "text", text: bad }], details: { ok: false, status: 0, chars: 0 } };
    try {
      const r = await fetch(params.url, {
        signal, redirect: "follow",
        headers: { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" },
      });
      const raw = await r.text();
      const body = /html/i.test(r.headers.get("content-type") ?? "") ? stripHtml(raw) : raw;
      const cap = params.max_chars ?? 20000;
      const text = body.length > cap ? `${body.slice(0, cap)}\n…[truncated]` : body;
      return {
        content: [{ type: "text", text: fence(text, params.url) }],
        details: { ok: r.ok, status: r.status, chars: text.length },
      };
    } catch (e) {
      return { content: [{ type: "text", text: `Fetch error: ${e}` }], details: { ok: false, status: 0, chars: 0 } };
    }
  },
});

export default function (pi: ExtensionAPI) {
  pi.registerTool(webSearch);
  pi.registerTool(webFetch);
}
