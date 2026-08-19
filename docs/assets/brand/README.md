# Brand assets

## `github-social-preview.png` — GitHub repository social preview

Upload this one under **Settings → General → Social preview → Upload an image**
on <https://github.com/mjwsolo/localcode>. It is the exact 1280×640 GitHub asks
for, so GitHub stores it without re-cropping.

| | |
| --- | --- |
| Upload artifact | `github-social-preview.png` — 1280×640, 59 KB, PNG |
| Editable source | `github-social-preview.svg` — 1280×640, ~4.8 KB, live `<text>` |
| Mark source | `house-mark.svg` — 32×32, themeable |
| Re-export | `python3 scripts/export_social_preview.py` |

To preview before uploading, just open either file — `open
docs/assets/brand/github-social-preview.png`. The SVG needs a browser (or the
docs site) for the webfonts to resolve; the PNG has them baked in.

### Editing it

Change the text in the SVG, then re-run the exporter:

```sh
python3 scripts/export_social_preview.py
```

The exporter renders through headless Chrome with Martian Mono, Inter and
Commit Mono loaded from `website/node_modules/@fontsource`, so **`npm ci` must
have run in `website/` first**. Anything else (rsvg-convert, cairosvg, Preview)
will silently substitute a system font, because those faces are not installed
system-wide. The script refuses to write the PNG unless the render came out at
exactly 1280×640.

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
