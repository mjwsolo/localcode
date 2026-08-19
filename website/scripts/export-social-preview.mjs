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
 * WHAT --check ENFORCES, IN ORDER
 *   1. Renderer identity -- playwright version AND chromium revision -- is
 *      compared first, on its own, and ANY drift fails. This is deliberately
 *      not conditional on the pixels differing: a toolchain change that happens
 *      to produce identical bytes today is still unrecorded, and the next edit
 *      to the SVG would be rendered by something the manifest never saw.
 *   2. Input hashes (SVG, each font) -- also independent of pixel equality.
 *   3. The committed PNG matches the sha256 the manifest claims.
 *   4. Pixels of a fresh render. Reported as an independent finding only when
 *      1-3 are all clean, since otherwise they already name the cause.
 *
 * THE ONE TOLERATED DIFFERENCE
 * A pixel-hash mismatch is downgraded to a NOTE in exactly one situation:
 * identical inputs, identical renderer identity, and a different platform.
 * Skia's text rasterisation and default font hinting differ between macOS,
 * Linux and Windows, so the same Chromium can emit a few different bytes for
 * identical glyph outlines. A platform change never excuses renderer or input
 * drift -- those are checked before the exception can apply, and still fail.
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
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

// This script lives in website/scripts/ rather than the repo's own scripts/
// because `playwright` resolves out of website/node_modules -- the only
// lockfile-managed dependency tree in the repo. It writes to docs/assets/brand/
// at the repo root, which is where the GitHub-facing brand assets live.
/**
 * Decide whether a re-render agrees with a recorded manifest. Pure: no I/O, no
 * globals, so the matrix below can be tested without launching a browser.
 *
 * Order matters and is load-bearing. Renderer identity and inputs are compared
 * independently of pixel equality, and BEFORE the cross-platform exception can
 * apply, so neither a lucky byte-match nor a simultaneous OS change can hide a
 * toolchain change.
 *
 * @param {{renderer: object, inputs: object}} current  freshly measured
 * @param {{renderer: object, inputs: object, sha256: string}} want  the manifest
 * @param {string} onDiskHash  sha256 of the committed PNG
 * @param {string} freshHash   sha256 of the render just produced
 * @returns {{problems: string[], notes: string[]}}
 */
export function evaluate(current, want, onDiskHash, freshHash) {
  const problems = [];
  const notes = [];

  // (a) Renderer identity. Both fields count: a playwright bump can change
  // rendering behaviour even when the Chromium revision is unchanged.
  const RENDERER_FIELDS = [
    ['playwright', 'playwright version'],
    ['chromiumRevision', 'chromium revision'],
  ];
  const rendererDrift = RENDERER_FIELDS.filter(
    ([key]) => current.renderer[key] !== want.renderer?.[key],
  );
  for (const [key, label] of rendererDrift) {
    problems.push(
      `renderer changed -- ${label}: ${want.renderer?.[key] ?? '(not recorded)'} -> ` +
      `${current.renderer[key]}\n      re-export and review the rendered diff before committing`,
    );
  }

  // (b) Inputs, also independent of pixel equality.
  const inputsDrift = JSON.stringify(current.inputs) !== JSON.stringify(want.inputs);
  if (inputsDrift) problems.push('inputs changed (SVG or fonts) -- re-export');

  // (c) The committed PNG is the artifact people upload; it must match what the
  // manifest claims, whatever this machine happens to re-render.
  if (onDiskHash !== want.sha256) {
    problems.push(
      `PNG on disk does not match the manifest\n      disk ${onDiskHash}\n      want ${want.sha256}`,
    );
  }

  const platformDrift = current.renderer.platform !== want.renderer?.platform;

  // (d) Pixels. Only an independent finding when nothing above explains it.
  if (freshHash !== want.sha256) {
    if (rendererDrift.length || inputsDrift) {
      // Already reported with the actual cause; don't restate it as a mystery.
    } else if (platformDrift) {
      // THE ONLY TOLERATED PIXEL DIFFERENCE: identical inputs, identical
      // renderer identity, different OS.
      notes.push(
        `re-render differs, and this is ${current.renderer.platform} while the manifest\n` +
        `        was produced on ${want.renderer?.platform}. Inputs and renderer identity are\n` +
        `        unchanged, so this is platform-dependent text rasterisation, not corruption.`,
      );
    } else {
      problems.push(
        'same inputs, same renderer, same platform, DIFFERENT pixels\n' +
        `      got  ${freshHash}\n      want ${want.sha256}`,
      );
    }
  } else if (platformDrift) {
    notes.push(
      `manifest was produced on ${want.renderer?.platform}, this is ` +
      `${current.renderer.platform} -- pixels match anyway.`,
    );
  }

  return { problems, notes };
}

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

const isMain =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (!isMain) {
  // Imported for `evaluate` alone -- do not render anything.
} else {

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

  const { problems, notes } = evaluate(manifest, want, onDisk, outHash);

  for (const n of notes) console.log(`  NOTE  ${n}`);
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

} // end CLI body
