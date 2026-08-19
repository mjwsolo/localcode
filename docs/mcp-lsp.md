# Comprehensive code intelligence via MCP (LSP)

LocalCode ships an MCP client (`src/localcode/mcp/`). Point it at a language‑server
MCP and the model gets **real, semantic** code intelligence as on‑demand tools —
hover/types, diagnostics, definition, references, rename — backed by an actual
language server, not regex. This is deliberately a **tool you call**, not a
persistent daemon: on a 16 GB Mac already running a large local model, an
always‑on `tsserver`/`pyright` is too much, so the language server only runs
while a query is in flight.

Config lives in `~/.localcode/mcp.json` (key `mcpServers`). Both recipes below
were verified end‑to‑end through LocalCode's own client.

## Native code intelligence (no setup)

Already built in, always available — reach for these first:

- `inspect_symbol` — a third‑party library's **real signature** from its installed
  types (`node_modules/**/*.d.ts`, site‑packages `.pyi`). Use before calling an
  API you're unsure of. (This is LSP "hover" for dependencies, filesystem‑only.)
- `code_navigation` — `symbols` / `definition` / `references` over **your** code
  (Python via AST).
- `project_check` — the real `tsc -p … --noEmit` / `ruff` diagnostics, run at the
  build‑completion gate.

Add an MCP language server (below) for the semantic ops those can't do:
cross‑file **semantic** references, go‑to‑implementation, call‑hierarchy, rename.

## TypeScript / JavaScript — `lsp-mcp` (verified)

Reliable because you hand it the **explicit** LSP command (no auto‑provisioning).
Exposes `get_diagnostics`, `get_info_on_location` (hover), `get_completions`,
`get_code_actions`. The flow is three steps: `start_lsp` (root dir) →
`open_document` → `get_diagnostics` / `get_info_on_location`.

```jsonc
// ~/.localcode/mcp.json
{
  "mcpServers": {
    "tslsp": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y", "tritlo/lsp-mcp",
        "typescript",
        "<abs path to>/typescript-language-server",   // e.g. `npm i -g typescript-language-server`, then `which`
        "--stdio"
      ]
    }
  }
}
```

Verified: `get_diagnostics` on a real project returned genuine
`TS7006 / TS2339` diagnostics from the language server, through LocalCode.

Rough edge: `lsp-mcp` logs `[INFO]/[ERROR]` to **stdout**, which is the JSON‑RPC
channel; the client recovers, but expect a benign parse warning at connect.

## Python — Serena (verified)

Serena auto‑manages language servers (pyright is bundled/downloaded) and adds the
full semantic set: `find_symbol`, `find_referencing_symbols`,
`find_implementations`, `find_declaration`, `rename_symbol`,
`get_symbols_overview`, plus semantic edits.

```jsonc
{
  "mcpServers": {
    "serena": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/oraios/serena",
        "serena", "start-mcp-server",
        "--project", "<abs path to your repo>",
        "--transport", "stdio",
        "--enable-web-dashboard", "False",
        "--enable-gui-log-window", "False",
        "--log-level", "ERROR"
      ]
    }
  }
}
```

Verified: `find_referencing_symbols` returned all real references (via pyright),
through LocalCode.

Caveats learned the hard way:
- **The project must be a git repo** — Serena/solidlsp enumerate files via git;
  a non‑git dir yields "no active language servers."
- **TypeScript in Serena is finicky** — its solidlsp did not auto‑start
  `typescript-language-server` in our environment even when installed. Prefer
  `lsp-mcp` (above) for TS/JS; use Serena for Python and its semantic‑nav set.

## Tool names the model sees

Each server's tools are exposed as `mcp_<server>_<tool>` (e.g.
`mcp_tslsp_get_diagnostics`, `mcp_serena_find_referencing_symbols`), so multiple
servers never collide. Cost measured: ~170 ms one‑time connect, ~2 ms/call — the
language server itself is what costs on first activation.
