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
└── public/                    favicon.svg, social-preview.svg
```

## Brand

**Tokens** (`src/styles/tokens.css`) — the palette is fixed:

| Token | Value | Use |
| --- | --- | --- |
| `--lc-ink` | `#14110F` | Dark ground |
| `--lc-bone` | `#F7F4EF` | Light ground |
| `--lc-surface` | `#2A2622` | Raised dark surface |
| `--lc-muted-dark` | `#5C5751` | Muted text on light |
| `--lc-muted-light` | `#A8A099` | Muted text on dark |
| `--lc-amber` | `#FFB020` | Accent — **dark backgrounds only** |
| `--lc-ember` | `#B24A0B` | Accent — **light backgrounds only** |
| `--lc-verify` | `#5FD08A` | Pass / verified |
| `--lc-fault` | `#FF6B57` | Fail |

Use `--lc-accent`, which resolves to amber or ember for the active theme,
rather than picking one by hand.

**Finder Mark** — two concentric orthogonal square rings with a solid core
offset down and right, hand-authored on a 32-unit grid with every edge on a
0.5 step so the 16px favicon render stays crisp. It is inlined as a component
(not an `<img>`) so the rings inherit `currentColor` and the core follows
`--lc-mark-core`.

Deliberately absent: violet/blue gradients, glassmorphism, glow, neural
meshes, generic AI imagery.

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

### Font bundling — still outstanding

`src/assets/brand/lockup-horizontal.svg` and `public/social-preview.svg` use
**live `<text>`** in a Martian Mono font stack. That renders correctly in a
browser (the face is loaded on the page) but **not** when the SVG is opened
standalone, embedded in a third-party surface, or sent to print. Before either
file is used as a distributed brand asset, convert the text to outlines.

## What is real and what is illustrative

- **Repo-verified.** The model chooser numbers come from
  `localcode.models_catalog.recommend()`. Slash commands, CLI flags, config
  keys, hook names, autonomy levels, the outbound-network inventory and the
  verification gates were read out of `src/localcode/`.
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
- **No performance claims** appear anywhere on the site. The one quantitative
  figure that remains — TurboQuant's ~3.8× KV-cache compression versus `f16` —
  is attributed in-page as the fork's own figure for the quantisation scheme,
  not a measurement of any machine.

## Deployment-time requirements

Two things are deliberately left unset because this preview has no deployment
origin yet. Both need doing before the site is published:

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
