/**
 * Render docs/assets/brand/github-social-preview.svg to an exact 1280x640 PNG.
 *
 * WHY A SCRIPT AND NOT A DESIGN TOOL
 * The card's text is live <text> in three webfonts that are not installed
 * system-wide. rsvg-convert, cairosvg and Preview all silently substitute a
 * system face; the result looks close enough to ship and is wrong. This loads
 * the exact woff2 files from @fontsource and renders through a browser.
 *
 * WHAT "REPRODUCIBLE" MEANS HERE, PRECISELY
 * The renderer is the Chromium build pinned by the `playwright` version in
 * package-lock.json -- not whatever Chrome/Edge the machine happens to have.
 * Bump playwright and the Chromium revision changes with it, deliberately and
 * visibly. Every input is content-hashed into the manifest: the SVG, each font
 * file, the playwright version, and the Chromium revision.
 *
 * The output hash is therefore stable for a given (inputs, renderer, platform)
 * triple. It is NOT claimed to be stable across operating systems: Skia's text
 * rasterisation and default font hinting differ between macOS, Linux and
 * Windows, so the same Chromium can emit a few different bytes for identical
 * glyph outlines. The manifest records the platform it was produced on, and
 * `--check` says so plainly rather than pretending a cross-platform mismatch is
 * a corruption.
 *
 *   cd website
 *   npm run social:export   # render + write the PNG and its manifest
 *   npm run social:check    # verify what is on disk, write nothing
 *
 * Run from anywhere; paths resolve off this file.
 */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

// This script lives in website/scripts/ rather than the repo's own scripts/
// because `playwright` resolves out of website/node_modules -- the only
// lockfile-managed dependency tree in the repo. It writes to docs/assets/brand/
// at the repo root, which is where the GitHub-facing brand assets live.
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..', '..');
const SVG = join(REPO, 'docs/assets/brand/github-social-preview.svg');
const PNG = join(REPO, 'docs/assets/brand/github-social-preview.png');
const MANIFEST = join(REPO, 'docs/assets/brand/github-social-preview.manifest.json');
const FONTS = join(REPO, 'website/node_modules/@fontsource');

const W = 1280;
const H = 640;

/** family, weight, path under @fontsource */
const FACES = [
  ['Martian Mono', 400, 'martian-mono/files/martian-mono-latin-400-normal.woff2'],
  ['Martian Mono', 600, 'martian-mono/files/martian-mono-latin-600-normal.woff2'],
  ['Inter', 400, 'inter/files/inter-latin-400-normal.woff2'],
  ['Commit Mono', 400, 'commit-mono/files/commit-mono-latin-400-normal.woff2'],
];

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');
const die = (msg) => {
  console.error(`\n  ${msg}\n`);
  process.exit(1);
};

const check = process.argv.includes('--check');

if (!existsSync(SVG)) die(`Missing source: ${relative(REPO, SVG)}`);

// ── Inputs ──────────────────────────────────────────────────────────
const svgText = readFileSync(SVG, 'utf8');
const fontHashes = {};
const faceCss = FACES.map(([family, weight, rel]) => {
  const path = join(FONTS, rel);
  if (!existsSync(path)) {
    die(`Missing font: ${relative(REPO, path)}\n  Run \`npm ci\` in website/ first.`);
  }
  const bytes = readFileSync(path);
  fontHashes[rel] = sha256(bytes).slice(0, 16);
  return (
    `@font-face{font-family:'${family}';font-weight:${weight};font-display:block;` +
    `src:url(data:font/woff2;base64,${bytes.toString('base64')}) format('woff2');}`
  );
}).join('');

const playwrightVersion = JSON.parse(
  readFileSync(join(REPO, 'website/node_modules/playwright/package.json'), 'utf8'),
).version;
// The revision directory name is what actually pins the binary, e.g. chromium-1234.
const chromiumRevision =
  chromium.executablePath().match(/chromium[_a-z]*-(\d+)/)?.[1] ?? 'unknown';

// ── Render ──────────────────────────────────────────────────────────
// The SVG is inlined rather than referenced with <img src>, because an <img>
// renders in an isolated document that cannot see the host's @font-face rules.
const page_html =
  `<!doctype html><meta charset="utf-8"><style>${faceCss}` +
  `html,body{margin:0;padding:0;background:#0B0E11}` +
  `svg{display:block;width:${W}px;height:${H}px}</style>${svgText}`;

const browser = await chromium.launch();
let png;
try {
  const page = await browser.newPage({
    viewport: { width: W, height: H },
    deviceScaleFactor: 1,
  });
  await page.setContent(page_html, { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  png = await page.screenshot({ type: 'png', clip: { x: 0, y: 0, width: W, height: H } });
} finally {
  await browser.close();
}

// ── Verify the raster before it is allowed anywhere ─────────────────
// PNG IHDR: 8-byte signature, then length+type, then width/height big-endian.
const gotW = png.readUInt32BE(16);
const gotH = png.readUInt32BE(20);
if (gotW !== W || gotH !== H) die(`Wrong export size: ${gotW}x${gotH}, expected ${W}x${H}`);
if (png.length >= 1_000_000) die(`PNG is ${png.length} bytes; GitHub's cap is 1 MB`);

const outHash = sha256(png);
const manifest = {
  output: 'github-social-preview.png',
  width: W,
  height: H,
  bytes: png.length,
  sha256: outHash,
  renderer: {
    playwright: playwrightVersion,
    chromiumRevision,
    // Recorded because text rasterisation is platform-dependent; see the header.
    platform: `${process.platform}-${process.arch}`,
  },
  inputs: { svg: sha256(Buffer.from(svgText)).slice(0, 16), fonts: fontHashes },
};

if (check) {
  if (!existsSync(MANIFEST)) die('No manifest recorded. Run without --check first.');
  if (!existsSync(PNG)) die('No PNG on disk. Run without --check first.');
  const want = JSON.parse(readFileSync(MANIFEST, 'utf8'));
  const onDisk = sha256(readFileSync(PNG));

  const problems = [];
  if (onDisk !== want.sha256) {
    problems.push(`PNG on disk does not match the manifest\n      disk ${onDisk}\n      want ${want.sha256}`);
  }
  if (JSON.stringify(manifest.inputs) !== JSON.stringify(want.inputs)) {
    problems.push('inputs changed (SVG or fonts) -- re-export');
  }
  // If the inputs already changed, a pixel difference is the expected
  // consequence, not a second independent finding -- don't report it twice.
  if (outHash !== want.sha256 && problems.length === 0) {
    const samePlatform = manifest.renderer.platform === want.renderer.platform;
    const sameRenderer =
      manifest.renderer.chromiumRevision === want.renderer.chromiumRevision;
    if (!samePlatform) {
      console.log(
        `  NOTE  re-render differs, and this is ${manifest.renderer.platform} while the\n` +
        `        manifest was produced on ${want.renderer.platform}. Text rasterisation is\n` +
        `        platform-dependent, so this is expected, not corruption.`,
      );
    } else if (!sameRenderer) {
      problems.push(
        `renderer changed: chromium-${want.renderer.chromiumRevision} -> ` +
        `chromium-${manifest.renderer.chromiumRevision} -- re-export and review the diff`,
      );
    } else {
      problems.push(
        `same inputs, same renderer, same platform, DIFFERENT pixels\n` +
        `      got  ${outHash}\n      want ${want.sha256}`,
      );
    }
  }
  if (problems.length) die('FAILED\n  - ' + problems.join('\n  - '));
  console.log(
    `  OK  ${gotW}x${gotH}  ${png.length.toLocaleString()} bytes  sha256 ${outHash.slice(0, 16)}…\n` +
    `      chromium-${chromiumRevision} via playwright ${playwrightVersion} on ${manifest.renderer.platform}`,
  );
} else {
  writeFileSync(PNG, png);
  writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2) + '\n');
  console.log(
    `  ${relative(REPO, PNG)}\n` +
    `  ${gotW}x${gotH}  ${png.length.toLocaleString()} bytes  sha256 ${outHash.slice(0, 16)}…\n` +
    `  chromium-${chromiumRevision} via playwright ${playwrightVersion} on ${manifest.renderer.platform}`,
  );
}
