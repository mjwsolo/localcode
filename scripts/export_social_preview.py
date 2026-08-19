#!/usr/bin/env python3
"""Export the GitHub social preview SVG to an exact 1280x640 PNG.

GitHub's Settings > General > Social preview slot wants a raster at exactly
1280x640. This renders `docs/assets/brand/github-social-preview.svg` through
headless Chromium so the Martian Mono / Inter / Commit Mono text matches the
site instead of falling back to a system face -- which is what happens with
rsvg-convert or cairosvg, since none of those faces are installed system-wide.

The fonts are loaded straight from the docs site's node_modules via @font-face,
so the export needs `npm ci` to have run in `website/` first. No network.

    python3 scripts/export_social_preview.py

Deterministic: same SVG in, same pixels out. Verifies the result is exactly
1280x640 before writing, and fails loudly rather than shipping a wrong size.
"""
from __future__ import annotations

import base64
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SVG = ROOT / "docs/assets/brand/github-social-preview.svg"
PNG = ROOT / "docs/assets/brand/github-social-preview.png"
FONTS = ROOT / "website/node_modules/@fontsource"

W, H = 1280, 640

# (css family, weight, woff2 path relative to @fontsource)
FACES = [
    ("Martian Mono", 400, "martian-mono/files/martian-mono-latin-400-normal.woff2"),
    ("Martian Mono", 600, "martian-mono/files/martian-mono-latin-600-normal.woff2"),
    ("Inter", 400, "inter/files/inter-latin-400-normal.woff2"),
    ("Commit Mono", 400, "commit-mono/files/commit-mono-latin-400-normal.woff2"),
]

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    shutil.which("chromium") or "",
    shutil.which("google-chrome") or "",
]


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if path and pathlib.Path(path).exists():
            return path
    sys.exit("No Chrome/Chromium found. Install one, or export the SVG by hand at 1280x640.")


def font_face_css() -> str:
    blocks = []
    for family, weight, rel in FACES:
        path = FONTS / rel
        if not path.is_file():
            sys.exit(f"Missing font: {path}\nRun `npm ci` in website/ first.")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        blocks.append(
            f"@font-face{{font-family:'{family}';font-weight:{weight};font-display:block;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
        )
    return "".join(blocks)


def main() -> None:
    if not SVG.is_file():
        sys.exit(f"Missing source: {SVG}")
    chrome = find_chrome()

    # Inline the SVG (not an <img>) so the @font-face rules in the host document
    # apply to its <text>. An <img src="*.svg"> renders in an isolated document
    # that cannot see them, and would silently fall back to a system mono.
    page = (
        "<!doctype html><meta charset='utf-8'><style>"
        + font_face_css()
        + f"html,body{{margin:0;padding:0;background:#0B0E11}}"
        + f"svg{{display:block;width:{W}px;height:{H}px}}</style>"
        + SVG.read_text(encoding="utf-8")
    )

    with tempfile.TemporaryDirectory() as tmp:
        html = pathlib.Path(tmp) / "card.html"
        html.write_text(page, encoding="utf-8")
        out = pathlib.Path(tmp) / "out.png"
        # Chrome 151 writes the screenshot and then does not exit in
        # --headless=old, so this is bounded by a timeout and judged on whether
        # the file landed rather than on the return code. --headless (new mode)
        # hangs here without writing anything at all.
        try:
            subprocess.run(
                [
                    chrome,
                    "--headless=old",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=1",
                    f"--window-size={W},{H}",
                    f"--screenshot={out}",
                    f"--user-data-dir={tmp}/profile",
                    "--no-first-run",
                    "--no-default-browser-check",
                    # Let the embedded webfonts decode before the shot.
                    "--virtual-time-budget=5000",
                    html.as_uri(),
                ],
                timeout=90,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            pass
        if not out.is_file():
            sys.exit("Chrome produced no screenshot.")
        got = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(out)],
            check=True, capture_output=True, text=True,
        ).stdout
        w = int(got.split("pixelWidth:")[1].split()[0])
        h = int(got.split("pixelHeight:")[1].split()[0])
        if (w, h) != (W, H):
            sys.exit(f"Wrong export size: {w}x{h}, expected {W}x{H}")
        shutil.copyfile(out, PNG)

    print(f"{PNG.relative_to(ROOT)}  {W}x{H}  {PNG.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
