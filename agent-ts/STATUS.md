# localcode on the new agent core — status against what users actually get today

Measured against README.md's promises and the current TUI's behaviour.
**done** = built and verified live · **built** = code exists, not verified in a
real TUI · **gap** = not there yet.

| # | What a localcode user expects | Status | Evidence |
|---|---|---|---|
| 1 | `pip install localcode` → `localcode` → it works | **built** | `frontend_agent.py` starts the server and hands over; a real turn returned through it. Wheel packaging of the binary still to do. |
| 2 | No API key, no account, no cloud inference | **done** | `/login` removed, model lists scoped to `localcode/*`, every call-home endpoint neutralised |
| 3 | First run recommends a model for this Mac and offers a download | **built** | picker opens from `session_start` when nothing is on disk — needs an interactive run to confirm |
| 4 | Browse model families, then every quant, with sizes | **done** | live HF tree listing, 24 quants for Gemma 4 12B, fit badges, ✓ downloaded markers, `← Back` |
| 5 | Reads and edits files | **done** | built in |
| 6 | Runs tests, builds, git, shell — asks before anything risky | **done** | `rm -rf /` blocked live; risky commands prompt once per command |
| 7 | Searches code by name and content | **done** | `grep`, `find`, `ls` |
| 8 | Searches code by *structure* | **gap** | localcode has `code_navigation` / `inspect_symbol`; nothing equivalent yet |
| 9 | Scaffolds and launches apps, then checks they respond | **done** | `launch_app` verified: reported `http://localhost:5173` correctly |
| 10 | Web search and fetch | **done** | ported with the SSRF guard; cloud-metadata address refused live |
| 11 | Remembers the task across messages | **done** | sessions, `--continue`, `--resume` |
| 12 | Thinking off by default | **done** | `chat_template_kwargs.enable_thinking=false` on every request |
| 13 | Nothing in the UI names another product | **done** | zero `pi.dev` strings; title, config dir, env prefix all localcode |
| 14 | Only commands that make sense here | **done** | 13 removed, entries deleted from the palette rather than renamed |

## Known gaps, in the order I would close them

1. **Ship the binary in the wheel** — 26 MB compressed, fits comfortably; needs
   `package-data` plus a build step. Without it, item 1 is only true for devs.
2. **Verify first run interactively** — move the GGUFs aside and launch.
3. **Structural code search** (item 8) — the one README promise with no answer.
4. **Cloud provider endpoints are still compiled in** — unreachable (no
   `/login`, lists scoped) but present. `unregisterProvider` at startup, or
   strip at build time.
5. **MCP and LSP** — not promised by the README, available as third-party
   extensions when wanted.
6. **Header branding** — the footer renders; the custom header does not,
   because `session_start` runs after the startup header is drawn.

## Regression watch (localcode behaviour that must not be lost)

thinking off · reasoning-cap no-think recovery · 1800s cap for slow models ·
`thinking_budget_tokens` vs `reasoning_budget_tokens` split · memory guard ·
thermal throttle · redaction · injection defense (web tools fence, approvals) ·
curated families + RAM recommendation.

Items 3-5 of that list are **not yet ported** and are tracked in REBRAND.md §D.
