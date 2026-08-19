# Brand assets

## `github-social-preview.png` — GitHub repository social preview

Upload this one under **Settings → General → Social preview → Upload an image**
on <https://github.com/mjwsolo/localcode>. It is the exact 1280×640 GitHub asks
for, so GitHub stores it without re-cropping.

| | |
| --- | --- |
| Upload artifact | `github-social-preview.png` — 1280×640, 57 KB, PNG |
| Editable source | `github-social-preview.svg` — 1280×640, ~3.9 KB, live `<text>` |
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

### What it is built from

The approved identity: the **Finder Mark** (two concentric orthogonal square
rings with a solid core, offset down and right) plus the **lowercase
`localcode` wordmark**. Colour is the cool ramp at hue 225 — `#0B0E11` ground,
`#F5F7F8` text, `#8C98A3` muted, `#5B86FF` accent (the on-dark lift of brand
`#2457E6`).

Every text colour clears WCAG AA on the ground: 18.01:1 bone, 5.80:1 accent,
6.58:1 muted.

It is **not** a crop of the docs hero. It is laid out for the slot: all content
inside a 112 × 64 px safe inset, nothing below 25 px, and the block optically
centred so a consumer cropping to 1.91:1 or 16:9 takes only empty ground.
Verified legible down to a 200 px-wide thumbnail, where the wordmark and both
tagline lines still read.

## The wordmark logo in the README

`../logo/{light,dark}.png` — a house emoji plus the lowercase wordmark — is
still what `README.md` and `docs/index.md` render, and it is untouched. Note
that it does **not** match the Finder Mark identity used by the social preview
and the docs preview site; reconciling the two is an open decision, not
something these files assume.

## Retired

`hero-banner.svg` / `hero-banner.png` (a generated flow-field composition) and
`scripts/gen_hero_banner.py` that produced them were removed — the art was the
gradient/glow style the brand direction rules out, and the PNG was 338 KB. The
replacement hero banner is hand-authored at
`website/public/brand/hero-banner.svg` (5 KB).
