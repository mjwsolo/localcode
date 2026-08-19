# localcode docs — preview site

A self-contained [Astro](https://astro.build) + [Starlight](https://starlight.astro.build)
docs site with a custom landing page. It lives entirely in `website/` and does
not touch the Python package, the existing `docs/` directory, or `mkdocs.yml`.

## Run it locally

Requires **Node ≥ 22.12** (Astro 7's floor — see `.nvmrc`).

```sh
cd website
npm install
npm run dev
```

Then open **<http://localhost:4321/localcode>**.

The `/localcode` path is not a typo: `base` is set to `/localcode` in
`astro.config.mjs` so the preview matches a GitHub Pages project-site deploy.
The dev server redirects `/` there.

### Production build

```sh
npm run build      # writes dist/
npm run preview    # serves dist/ at http://localhost:4321/localcode
```

### Type/diagnostic check

```sh
npm run check      # astro check (also aliased as npm run lint)
```

## Layout

```
website/
├── astro.config.mjs           Starlight config + sidebar IA
├── src/
│   ├── pages/index.astro      Custom landing page (owns "/")
│   ├── components/            FinderMark, SiteTitle, CopyCommand,
│   │                          TerminalDemo, MemoryChooser
│   ├── styles/
│   │   ├── tokens.css         Brand tokens (the only place colours are defined)
│   │   ├── landing.css        Landing page
│   │   └── docs.css           Starlight --sl-* → localcode token mapping
│   ├── assets/brand/          finder-mark.svg, lockup-horizontal.svg
│   └── content/docs/          Markdown pages
└── public/                    favicon.svg, social-preview.svg,
                               brand/hero-banner.svg
```

## Brand

**Tokens** (`src/styles/tokens.css`) — the palette is fixed:

The palette is a **cool ramp built on hue 225**. Neutrals sit on hue ~210 so
they read as the same family; the accent is the only saturated hue.

| Token | Value | Use |
| --- | --- | --- |
| `--lc-ink` | `#0B0E11` | Dark ground |
| `--lc-bone` | `#F5F7F8` | Light ground |
| `--lc-surface` | `#171C21` | Raised dark surface |
| `--lc-muted-dark` | `#5A6672` | Muted text on light |
| `--lc-muted-light` | `#8C98A3` | Muted text on dark |
| `--lc-amber` | `#47E6A1` | Accent — **dark backgrounds only** |
| `--lc-ember` | `#2457E6` | Accent — **light backgrounds only**, hue 225 |
| `--lc-verify` | `#3FCF8E` | Pass / verified |
| `--lc-fault` | `#FF5A5F` | Fail |

Use `--lc-accent`, which resolves to the correct one for the active theme,
rather than picking one by hand.

Two names are now historical: `--lc-amber` and `--lc-ember` are the dark-ground
and light-ground accents, not the warm hues they were originally named for. The
names are kept so the ~40 call sites across the landing page, the docs theme and
the brand SVGs stay in sync; renaming them is a mechanical follow-up, not a
design decision.

`#5B86FF` appears in the standalone brand SVGs (favicon, social preview,
banner). It is the on-dark lift of `#2457E6` — same hue 225, raised luminance so
the mark's core holds against ink at 16px. Those files cannot read CSS custom
properties, which is why the value is written out.

**Finder Mark** — two concentric orthogonal square rings with a solid core
offset down and right, hand-authored on a 32-unit grid with every edge on a
0.5 step so the 16px favicon render stays crisp. It is inlined as a component
(not an `<img>`) so the rings inherit `currentColor` and the core follows
`--lc-mark-core`.

**Hero composition** — the hero has **no image**. Its second column is the real
transcript panel (`TerminalDemo`), so the strongest thing on the page is the
product doing its job rather than a decorative shape. The Finder Mark appears
three times, each time structurally: the nav lockup, the privacy callout, and
the panel's title bar. Everything is inline SVG and text, so the hero ships zero
image bytes, stays selectable and searchable, and scales with zoom.

**`public/brand/hero-banner.svg`** — a 5 KB hand-authored banner for README and
social use, built from the same two parts: mark + wordmark on the left, a flat
version of the transcript panel on the right. Flat fills, orthogonal geometry,
one accent hue.

**GitHub repository social preview** lives outside this directory, at
`docs/assets/brand/github-social-preview.{svg,png}`, because it is a repo-level
asset uploaded through GitHub Settings rather than something the site serves.
It uses the same hue-225 ramp and is laid out for that slot rather than cropped
from this hero, but it does **not** use the Finder Mark: it pairs the wordmark
with a **House Mark**, the repository's existing emoji-house logo redrawn as
flat geometry on the same grid and stroke weights. The two marks share a
construction — a frame with an offset solid core — so they read as one family,
but the project does currently carry two. See `docs/assets/brand/README.md`,
"Open decision: two marks".

Deliberately absent everywhere: gradients, glow, blur, glassmorphism, flow
fields, particle or neural meshes, device mockups, and any generated or stock
imagery.

## Typography

Self-hosted via Fontsource — no runtime request to Google Fonts:

| Role | Face | Package |
| --- | --- | --- |
| Display | Martian Mono | `@fontsource/martian-mono` |
| Body | Inter | `@fontsource/inter` |
| Code | Commit Mono | `@fontsource/commit-mono` |

All three are open-licensed and redistributable. Every stack ends in a system
fallback (`ui-monospace` / `-apple-system`), so the page degrades cleanly if a
face fails to load.

### Outlined SVG text — pre-publication requirement

`src/assets/brand/lockup-horizontal.svg`, `public/social-preview.svg` and
`public/brand/hero-banner.svg` use **live `<text>`** in a Martian Mono font
stack. That renders correctly in a
browser (the face is loaded on the page) but **not** when the SVG is opened
standalone, embedded in a third-party surface, or sent to print. Convert the
text to outlines before either file is used as a distributed brand asset — this
is required before publication, not a nice-to-have.

## What is real and what is illustrative

- **Repo-verified.** The model chooser numbers come from
  `localcode.models_catalog.recommend()`. Slash commands, CLI flags, config
  keys, hook names, the outbound-network inventory and the verification gates
  were read out of `src/localcode/`. The confirmation behaviour documented on
  the Permissions page is `agent/helpers.py::_needs_confirmation`, which is the
  live gate — not the `autonomy.py` policy table, whose `PermissionManager`
  consumer the agent loop does not call.
- **Goal-classifier scope is load-bearing.** The evidence guarantee only
  applies to `build_app` and `edit_existing` goals. Both showcased prompts were
  run through `infer_goal_state` to confirm they classify as `edit_existing`;
  the earlier "make parse_duration accept …" wording classified as
  `general_task` and would have misrepresented the flow.
- **The JSONL event table is taken from the producer** — the
  `OutputManager._emit_event` call sites plus `headless_json.py`. It is
  deliberately *not* derived from `tests/protocol_fixtures/*.jsonl`, which
  exercise a consumer and describe a different, hypothetical vocabulary.
- **Illustrative.** The terminal transcript on the landing page
  (`src/components/TerminalDemo.astro`) is a **design preview**, labelled as
  such on the page. No repository artifact records that flow. The tool names
  and `pytest -q` are real; every line of output text is written for the page.
  It contains **no durations, timings, token counts or throughput figures** —
  the check line reads `48 passed`, with no elapsed time.
- **No measured throughput claims** appear anywhere on the site, and no
  benchmark command or screen is implied — there isn't one. Architectural
  statements about *why* something is faster or slower (active parameters,
  memory bandwidth) do appear; what is absent is any figure presented as a
  measurement of performance. The model picker's tok/s
  figures are documented as a calculated estimate (bytes-per-token over assumed
  realised bandwidth, plus a compute floor), not a measurement. The one other
  quantitative figure — TurboQuant's ~3.8× KV-cache compression versus `f16` —
  is attributed in-page as the fork's own figure for the quantisation scheme.
- **Claims are scoped to the default configuration.** `runtime.base_url` /
  `LOCALCODE_BASE_URL` accepts any URL with no validation, so a custom
  inference endpoint is documented as an outbound path that carries prompts and
  code context. The hero says "in the default configuration" for this reason.

## Deployment-time requirements

Two things are deliberately left unset because this preview has no deployment
origin yet. Both need doing before the site is published:

These are pre-publication requirements, not optional polish.

1. **`og:image`.** No Open Graph image tag is emitted. OG requires an absolute
   URL, and the only candidate origin
   (`https://mjwsolo.github.io/localcode/`) currently serves the MkDocs site and
   has no `social-preview.svg`. `public/social-preview.svg` is built and ready —
   add the tag in `astro.config.mjs` once the real origin is known.
2. **`site` / `base` in `astro.config.mjs`** are set for a GitHub Pages
   project-site layout (`https://mjwsolo.github.io` + `/localcode`). The
   sitemap is generated from them, so confirm they match the real deployment.

## Preview stubs

These pages carry an explicit "Preview stub" callout:

- `concepts/architecture` — condensed; needs module-level detail.
- `guides/mcp` — links to `docs/mcp-lsp.md` instead of a migrated recipe.
- `reference/error-codes` — links to the generated `docs/ERRORS.md` instead of
  importing it.
- `models-and-performance` — no throughput numbers until there is a
  methodology and broader hardware coverage.

## 404 page

The site uses Starlight's built-in 404 route. A `src/content/docs/404.md` entry
also resolves through the catch-all `[...slug]` route, which makes the
production build emit a route-conflict warning — so there deliberately isn't
one. If a custom 404 becomes worth the warning, that is where it goes.

Fully written: Install, First Change, Choose a Model, Permissions, Network
Boundary, Verification, Unified Memory, Offline, Headless, Skills & Hooks,
Undo, CLI, Slash Commands, Configuration, JSONL Events, Contributing.

## Relationship to the existing docs

`docs/` and `mkdocs.yml` are untouched, and the published MkDocs site at
<https://mjwsolo.github.io/localcode/> continues to build. This is a parallel
preview; if it is adopted, the remaining MkDocs pages should be migrated and
the GitHub Pages workflow repointed. Nothing here decides that.

## One correction worth making in the product

The committed `docs/ERRORS.md` tells users to "Run `localcode setup`" under
`E1001`, `E1002` and `E1003`. There is no `setup` subcommand — but the registry
it is generated from, `src/localcode/errors.py`, is already correct and contains
no such reference. The generated file is simply stale. Re-running
`python -m localcode.errors --emit-docs > docs/ERRORS.md` fixes it; no source
change is needed. Out of scope here, and noted on the Error Codes page.
