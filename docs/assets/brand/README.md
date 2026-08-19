# Brand assets

## `github-social-preview.png` — GitHub repository social preview

Upload this one under **Settings → General → Social preview → Upload an image**
on <https://github.com/mjwsolo/localcode>. It is the exact 1280×640 GitHub asks
for, so GitHub stores it without re-cropping.

| | |
| --- | --- |
| Upload artifact | `github-social-preview.png` — 1280×640, 59,985 bytes, PNG |
| Editable source | `github-social-preview.svg` — 1280×640, ~4.8 KB, live `<text>` |
| Mark source | `house-mark.svg` — 32×32, themeable |
| Build record | `github-social-preview.manifest.json` — output hash + renderer |
| Re-export | `cd website && npm run social:export` |
| Verify | `cd website && npm run social:check` |

To preview before uploading, just open either file — `open
docs/assets/brand/github-social-preview.png`. The SVG needs a browser (or the
docs site) for the webfonts to resolve; the PNG has them baked in.

### Editing it

Change the text in the SVG, then re-export:

```sh
cd website
npm run social:export     # rewrites the PNG and the manifest
npm run social:check      # verifies; writes nothing; non-zero on mismatch
```

Both need `npm ci` to have run in `website/`, plus `npx playwright install
chromium` once.

The exporter loads Martian Mono, Inter and Commit Mono from
`website/node_modules/@fontsource` and renders through a browser. Do not
substitute rsvg-convert, cairosvg or Preview: the card's text is live `<text>`
in faces that are not installed system-wide, so those tools quietly swap in a
system font and produce something that looks close enough to ship and is wrong.

#### What is actually pinned, and what is not

The renderer is the Chromium build pinned by the `playwright` version in
`website/package-lock.json` — currently **playwright 1.62.1 → chromium-1234** —
not whatever Chrome the machine happens to have. Bumping playwright changes the
renderer deliberately and visibly instead of silently.

Every input is content-hashed into `github-social-preview.manifest.json`: the
SVG, each font file, the playwright version, the Chromium revision, and the
sha256 of the PNG itself. `npm run social:check` re-renders and compares, so a
changed font package, an edited SVG, a bumped renderer or a corrupted PNG all
fail loudly rather than passing on a dimension check.

**The limitation, stated plainly:** the output hash is reproducible for a given
(inputs, renderer, **platform**) triple, not universally. Skia's text
rasterisation and default hinting differ between macOS, Linux and Windows, so
the same Chromium can emit slightly different bytes for identical glyph
outlines. The manifest records the platform it was produced on
(`darwin-arm64`), and on a different platform `social:check` reports that as a
NOTE rather than as corruption. Byte-identical output across operating systems
would need a containerised renderer, which is more machinery than one brand
asset justifies.

### What it is built from — and where the mark comes from

The card pairs the **House Mark** with the **lowercase `localcode` wordmark**.

The House Mark is **this repository's own logo, redrawn**. `../logo/light.png`
and `../logo/dark.png` pair the wordmark with the 🏠 emoji, and that glyph is
supplied by the reader's operating system — it is a different drawing on Apple,
Google, Microsoft and Android, it carries bevels, gradients and a drop shadow
that fight a flat system, and it cannot be recoloured. So the shape was rebuilt
as vector geometry.

What was **kept**, because it is what makes the logo recognisable:

| From the emoji | In the House Mark |
| --- | --- |
| Gable roof overhanging the walls on both sides | Single constant slope, 11.5 rise over 14 run, eaves past the walls |
| Chimney, left of the apex | Two verticals terminating exactly on the roof line |
| Square body | Walls open at the top, where the roof covers them |
| Four-pane window, left | Frame plus two mullions, one pane lit |
| Doorway, right, reaching the base | Open-bottomed rectangle on the base line |

What was **dropped**, because it is platform-dependent or decorative: the
bevels, highlights and drop shadow; the grass; and the emoji's own colours.

The lit window pane is the "local" idea made literal, and it is deliberately
the same construction as the docs site's Finder Mark — a frame with an offset
solid core. In both marks the core sits in the bottom-right, so the two read as
one family rather than as two logos.

Colour is the cool ramp at hue 225 — `#0B0E11` ground, `#F5F7F8` text,
`#8C98A3` muted, `#5B86FF` accent (the on-dark lift of brand `#2457E6`).

Every text colour clears WCAG AA on the ground: 18.01:1 bone, 5.80:1 accent,
6.58:1 muted.

It is **not** a crop of the docs hero. It is laid out for the slot: all content
inside a 112 × 64 px safe inset, nothing below 25 px, and the block optically
centred so a consumer cropping to 1.91:1 or 16:9 takes only empty ground.
Verified legible down to a 200 px-wide thumbnail, where the mark still reads as
a house and the wordmark and both tagline lines are still readable. The mark
itself was proofed separately from 16 px to 144 px on both grounds.

## `house-mark.svg` — the mark on its own

32×32, on the same grid and stroke weights as the Finder Mark. Strokes inherit
`currentColor` and the lit pane reads a `--lc-mark-core` custom property, so it
themes anywhere it is inlined. Inline it rather than using `<img>` if you want
either to apply.

## Open decision: two marks

Three things now coexist, deliberately, and someone should choose between them:

1. `../logo/{light,dark}.png` and `../../logo.png` — the emoji house plus
   wordmark. **Untouched.** Still what `README.md` and `docs/index.md` render.
2. `house-mark.svg` — that same logo as flat geometry. Used by the social
   preview.
3. The **Finder Mark** (concentric squares) — used by the docs preview site in
   `website/`, per the approved preview direction.

(2) and (3) share a construction language on purpose, so the current state is
coherent rather than broken. But a project should ship one mark. Swapping the
docs site over to the House Mark, or the reverse, is a small mechanical change
in either direction; it is not made here because the docs hero direction was
already approved as-is.

## Retired

`hero-banner.svg` / `hero-banner.png` (a generated flow-field composition) and
`scripts/gen_hero_banner.py` that produced them were removed — the art was the
gradient/glow style the brand direction rules out, and the PNG was 338 KB. The
replacement hero banner is hand-authored at
`website/public/brand/hero-banner.svg` (5 KB).
