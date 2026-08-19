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
  keys, hook names, verification commands, autonomy levels and the JSONL event
  shape were all read out of `src/localcode/`. The JSONL examples are the
  checked-in fixtures in `tests/protocol_fixtures/`.
- **Illustrative.** The terminal transcript on the landing page is a **design
  preview**, labelled as such on the page itself. No repository contains a
  recorded transcript of that flow. It shows no timings, token counts, or
  throughput figures, because none of them would be real.
- **No throughput claims** appear anywhere on the site.

## Preview stubs

These pages carry an explicit "Preview stub" callout:

- `concepts/architecture` — condensed; needs module-level detail.
- `guides/mcp` — links to `docs/mcp-lsp.md` instead of a migrated recipe.
- `reference/error-codes` — links to the generated `docs/ERRORS.md` instead of
  importing it.
- `models-and-performance` — no throughput numbers until there is a
  methodology and broader hardware coverage.

Fully written: Install, First Change, Choose a Model, Permissions, Network
Boundary, Verification, Unified Memory, Offline, Headless, Skills & Hooks,
Undo, CLI, Slash Commands, Configuration, JSONL Events, Contributing.

## Relationship to the existing docs

`docs/` and `mkdocs.yml` are untouched, and the published MkDocs site at
<https://mjwsolo.github.io/localcode/> continues to build. This is a parallel
preview; if it is adopted, the remaining MkDocs pages should be migrated and
the GitHub Pages workflow repointed. Nothing here decides that.

## One correction worth making in the product

`docs/ERRORS.md` — generated from `src/localcode/errors.py` — tells users to
"Run `localcode setup`" in the remediation text for `E1001`, `E1002` and
`E1003`. There is no `setup` subcommand. This preview does not repeat that
instruction anywhere; fixing the registry strings is a separate change.
