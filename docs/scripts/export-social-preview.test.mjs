/**
 * The verification matrix for the social-preview exporter.
 *
 * These are the cases that matter: a renderer change must fail even when the
 * pixels happen to match, and a platform change must never excuse renderer or
 * input drift. Run with `npm run social:test`.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate } from './export-social-preview.mjs';

const PNG = 'a'.repeat(64);
const OTHER = 'b'.repeat(64);

const base = {
  renderer: { playwright: '1.62.1', chromiumRevision: '1234', platform: 'darwin-arm64' },
  inputs: { svg: 'aaaa', fonts: { x: 'bbbb' } },
};
const manifest = { ...structuredClone(base), sha256: PNG };
/** @param {object} over  overrides merged into a fresh copy of `base` */
const now = (over = {}) => {
  const c = structuredClone(base);
  Object.assign(c.renderer, over.renderer ?? {});
  if (over.inputs) Object.assign(c.inputs, over.inputs);
  return c;
};
const run = (current, onDisk = PNG, fresh = PNG) =>
  evaluate(current, structuredClone(manifest), onDisk, fresh);

test('clean: nothing drifted', () => {
  const { problems, notes } = run(now());
  assert.deepEqual(problems, []);
  assert.deepEqual(notes, []);
});

test('playwright drift fails even though the pixels are identical', () => {
  const { problems } = run(now({ renderer: { playwright: '1.63.0' } }));
  assert.equal(problems.length, 1);
  assert.match(problems[0], /playwright version: 1\.62\.1 -> 1\.63\.0/);
});

test('chromium drift fails even though the pixels are identical', () => {
  const { problems } = run(now({ renderer: { chromiumRevision: '1300' } }));
  assert.equal(problems.length, 1);
  assert.match(problems[0], /chromium revision: 1234 -> 1300/);
});

test('a platform change does NOT excuse renderer drift', () => {
  const { problems } = run(
    now({ renderer: { playwright: '1.63.0', platform: 'linux-x64' } }),
    PNG,
    OTHER,
  );
  assert.ok(problems.some((p) => /playwright version/.test(p)));
});

test('a platform change does NOT excuse input drift', () => {
  const { problems } = run(
    now({ renderer: { platform: 'linux-x64' }, inputs: { svg: 'cccc' } }),
    PNG,
    OTHER,
  );
  assert.ok(problems.some((p) => /inputs changed/.test(p)));
});

test('platform-only drift with differing pixels is the tolerated NOTE, not a failure', () => {
  const { problems, notes } = run(now({ renderer: { platform: 'linux-x64' } }), PNG, OTHER);
  assert.deepEqual(problems, []);
  assert.equal(notes.length, 1);
  assert.match(notes[0], /platform-dependent text rasterisation/);
});

test('same everything but different pixels is a hard failure', () => {
  const { problems } = run(now(), PNG, OTHER);
  assert.equal(problems.length, 1);
  assert.match(problems[0], /DIFFERENT pixels/);
});

test('a committed PNG that does not match the manifest fails', () => {
  const { problems } = run(now(), OTHER, PNG);
  assert.ok(problems.some((p) => /PNG on disk does not match/.test(p)));
});

test('a manifest with no renderer block fails loudly rather than passing', () => {
  const { problems } = evaluate(now(), { inputs: base.inputs, sha256: PNG }, PNG, PNG);
  assert.equal(problems.length, 2);
  assert.ok(problems.every((p) => /\(not recorded\)/.test(p)));
});
