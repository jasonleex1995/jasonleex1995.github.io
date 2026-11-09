#!/usr/bin/env node
/* Dot QR self-check — run it after touching qr.js or the presets:

     node projects/dot-qr/check.js

   No dependencies; exit code 0 means everything passed.
   1) Plain symbols: per-version fingerprints of 1,920 symbols (versions 1–40 × L/M/Q/H × masks).
      Those exact symbols were compared module-for-module against node-qrcode 1.5.4 on
      2026-11-09 and were identical, so a changed fingerprint means the encoder changed.
   2) Drawings in the middle: every preset in index.html × links × grid sizes. When a code is made,
      it reads back as exactly the link, its box shows the drawing exactly (plus a light margin),
      the drawing covers no function pattern except the centre alignment square, and the rendered
      image puts every module's colour where scanners look.
   3) maxInsetSide tells the truth: a drawing that big fits, one module bigger does not.
*/
'use strict';
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const DotQR = require('./qr.js');

let failed = 0, passed = 0;
function check(ok, msg) { if (ok) passed++; else { failed++; console.log('FAIL  ' + msg); } }

/* ---------- 1) Plain symbols vs. recorded fingerprints ---------- */
const KNOWN = ['9040d4e3873463e7', '21f70a04cf7fed24', '3f3315e2f205ed88', '63d6e087fb82dae8', '6115df06315dcdac',
  '4f966c160c88567c', 'fd2035f9f944e2de', '6fb22378210a8736', 'acdb6c672c1df46a', '0ca3dd878ce8b08c', 'a4d1a35ed9013a7f',
  '0b117889b808bef0', '64ee281b8964be87', '96cbc6ff036180e1', '5d276a58c06e180a', '81947ffcb9f4c5a6', 'a108b3ef4f8548b7',
  '1f249a7f169c5586', '687958d4367a3352', '6f4ea4f3bd386ecd', 'd53c12a8f545a4e8', '1a711fddc957f3da', '08c120cb688f5dde',
  'ab32ccb0eaed04e9', '9195263288d030fd', '7894ff7923818945', 'a6211f31141ce6d9', 'a9e4a614be591ab6', '4cbe647d34b54a30',
  'cd6b322ea84a3e9c', 'c73950b86e4553ab', '6da38865d0a1ecc3', '3bcd0033f31a8cd4', '5f8fa2534a618cff', '063f976e432f9e08',
  '21bd222e6d893c7e', '1c31cdc213d9cd88', '2d31bb716e80f6b4', '2bfc7d7368cf7249', '12de299c21083c60'];
for (let v = 1; v <= 40; v++) {
  const h = crypto.createHash('sha256');
  let allRead = true;
  for (let e = 0; e < 4; e++) {
    const cap = DotQR._numDataCodewords(v, e), cc = v <= 9 ? 8 : 16;
    const full = Math.min(v <= 9 ? 255 : 1e9, Math.floor((cap * 8 - 4 - cc) / 8));
    for (const L of [full, Math.max(1, full >> 1), Math.min(full, 5)]) {
      const bytes = new Uint8Array(L);
      for (let i = 0; i < L; i++) bytes[i] = (i * 37 + v * 11 + e * 5 + 7) & 0xff;
      for (let m = 0; m < 8; m++) {
        if (L !== full && m > 1) continue;
        const res = DotQR.makePlain(bytes, { version: v, ecl: e, mask: m });
        h.update(res.modules);
        if (!DotQR.verify(res, bytes).ok) allRead = false;
      }
    }
  }
  check(h.digest('hex').slice(0, 16) === KNOWN[v - 1], `plain symbols changed at version ${v}`);
  check(allRead, `a plain symbol at version ${v} does not read back`);
}

/* ---------- 2) Drawings in the middle ---------- */
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const presets = {};
for (const m of html.matchAll(/^\s*(\w+): '((?:[0-9a-f]{8} ){31}[0-9a-f]{8})',?$/gm)) presets[m[1]] = m[2];
check(Object.keys(presets).length >= 4, 'could not read the presets from index.html');

function bitsOf(hex) {
  const out = new Uint8Array(32 * 32);
  hex.split(' ').forEach((w, y) => { const v = parseInt(w, 16); for (let x = 0; x < 32; x++) out[y * 32 + x] = (v >>> (31 - x)) & 1; });
  return out;
}
function resample(bits, M, M2) {      // same nearest-neighbour rule as the page's grid switch
  const out = new Uint8Array(M2 * M2);
  for (let y = 0; y < M2; y++) for (let x = 0; x < M2; x++)
    out[y * M2 + x] = bits[Math.floor((y + 0.5) * M / M2) * M + Math.floor((x + 0.5) * M / M2)];
  return out;
}

// Everything a finished code with a drawing must satisfy.
function checkInset(res, d, bytes, tag) {
  const read = DotQR.verify(res, bytes);
  check(read.ok, `${tag}: does not read back (${read.why})`);
  const N = res.size, b = res.box, ver = res.version;
  const ox = b.x + Math.floor((b.w - d.w) / 2), oy = b.y + Math.floor((b.h - d.h) / 2);
  let wrong = 0, badCover = 0, clash = 0;
  const al = DotQR._alignmentPositions(ver), mid = al.length >= 3 ? al[(al.length - 1) >> 1] : -1;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const i = y * N + x, inBox = x >= b.x && x < b.x + b.w && y >= b.y && y < b.y + b.h;
    if (res.covered[i]) {
      const centre = mid >= 0 && Math.abs(x - mid) <= 2 && Math.abs(y - mid) <= 2;
      if (!centre || !inBox) badCover++;
    }
    if (!inBox) continue;
    const dx = x - ox, dy = y - oy;
    const want = (dx >= 0 && dy >= 0 && dx < d.w && dy < d.h) ? d.bits[dy * d.w + dx] : 0;
    if (res.modules[i] !== want) wrong++;
    if (res.isFunction[i] && !res.covered[i]) clash++;
  }
  const coveredCount = res.covered.reduce((a, c) => a + c, 0);
  check(wrong === 0, `${tag}: ${wrong} modules in the box differ from the drawing`);
  check(badCover === 0, `${tag}: ${badCover} covered modules are not the centre alignment square`);
  check(coveredCount === 0 || coveredCount === 25, `${tag}: the centre alignment square is only partly covered (${coveredCount}/25)`);
  check(clash === 0, `${tag}: ${clash} function modules sit visibly inside the drawing's box`);
  check(b.x >= 0 && b.y >= 0 && b.x + b.w <= N && b.y + b.h <= N && d.w + 2 <= b.w && d.h + 2 <= b.h, `${tag}: box lacks its margin`);
  const px = 6, W = (N + 8) * px, img = new Uint8Array(W * W);                         // rendering
  DotQR.renderModules(res, { px }, (x, y, w, h) => { for (let yy = y; yy < y + h; yy++) img.fill(1, yy * W + x, yy * W + x + w); });
  let badPix = 0;
  for (let my = 0; my < N; my++) for (let mx = 0; mx < N; mx++)
    for (const [fx, fy] of [[1, 1], [3, 3], [5, 5]]) if (img[((my + 4) * px + fy) * W + (mx + 4) * px + fx] !== res.modules[my * N + mx]) badPix++;
  check(badPix === 0, `${tag}: rendering puts ${badPix} pixels in the wrong colour`);
  let quiet = 0;
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) if (img[y * W + x] && (x < 4 * px || y < 4 * px || x >= W - 4 * px || y >= W - 4 * px)) quiet++;
  check(quiet === 0, `${tag}: ${quiet} dark pixels in the quiet zone`);
}

const SHORT = ['https://bit.ly/3xYz9Ab', 'https://jasonleex1995.github.io', 'mailto:someone@example.com', '안녕하세요 Dot QR'];
const LONG = 'https://www.example.com/articles/2026/11/a-much-longer-link?ref=newsletter&id=12345';
for (const [name, hex] of Object.entries(presets)) {
  const base = bitsOf(hex);
  for (const link of SHORT) for (const M of [16, 24, 32]) {
    const bytes = DotQR.utf8(link), d = DotQR.cropDrawing(M === 32 ? base : resample(base, 32, M), M);
    const tag = `${name} / grid ${M} / ${link.slice(0, 26)}`;
    const res = DotQR.makeInset(bytes, d, { cache: {} });
    check(!!res, `${tag}: should fit but no code was made`);
    if (res) checkInset(res, d, bytes, tag);
  }
  const d = DotQR.cropDrawing(base, 32);                       // too long a link for a preset-sized drawing
  check(DotQR.makeInset(DotQR.utf8(LONG), d, { cache: {} }) === null, `${name}: a ${LONG.length}-character link should not fit a ${d.w}×${d.h} drawing`);
}

// Shapes the presets don't cover: a tiny drawing in a big code (box must grow to cover the centre
// square whole) and a wide, flat one (box runs close to the edges, past alignment squares).
{
  const bytes = DotQR.utf8('https://jasonleex1995.github.io');
  for (const ver of [7, 9, 12]) {
    const tiny = { w: 2, h: 2, bits: Uint8Array.from([1, 0, 0, 1]) };
    const res = DotQR.makeInset(bytes, tiny, { version: ver, cache: {} });
    check(!!res, `tiny drawing at version ${ver}: no code made`);
    if (res) checkInset(res, tiny, bytes, `tiny drawing at version ${ver}`);
  }
  for (const [w, h] of [[34, 10], [10, 34], [30, 14]]) {
    const bits = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) bits[i] = (Math.imul(i + 7, 2654435761) >>> 11) & 1;
    const d = { w, h, bits };
    for (const ver of [8, 9, 10, 11]) {
      const res = DotQR.makeInset(bytes, d, { version: ver, cache: {} });
      if (res) checkInset(res, d, bytes, `${w}×${h} drawing at version ${ver}`);
    }
    const auto = DotQR.makeInset(bytes, d, { cache: {} });
    check(!!auto, `${w}×${h} drawing: no code made`);
  }
}

/* ---------- 3) maxInsetSide matches what makeInset can actually do ---------- */
for (const link of ['https://bit.ly/3xYz9Ab', 'https://jasonleex1995.github.io', 'https://jasonleex1995.github.io/projects/dot-qr/']) {
  const bytes = DotQR.utf8(link), s = DotQR.maxInsetSide(bytes, { cache: {} });
  const random = n => { const b = new Uint8Array(n * n); for (let i = 0; i < n * n; i++) b[i] = (Math.imul(i + n, 2654435761) >>> 9) & 1; b[0] = b[n - 1] = b[n * n - 1] = b[n * n - n] = 1; return { w: n, h: n, bits: b }; };
  const fit = DotQR.makeInset(bytes, random(s), { cache: {} });
  check(!!fit, `${link}: maxInsetSide says ${s} fits, but it doesn't`);
  if (fit) checkInset(fit, random(s), bytes, `${link} @ ${s}×${s}`);
  check(DotQR.makeInset(bytes, random(s + 1), { cache: {} }) === null, `${link}: ${s + 1}×${s + 1} fits although maxInsetSide says ${s}`);
}
check(DotQR.maxInsetSide(DotQR.utf8('https://jasonleex1995.github.io'), { cache: {} }) === 32, 'a 31-character link should hold a full 32×32 drawing');

/* ---------- 4) Edges ---------- */
check(DotQR.cropDrawing(new Uint8Array(32 * 32), 32) === null, 'an empty drawing must crop to null');
check(DotQR.makePlain(DotQR.utf8('x'.repeat(2000)), { ecl: DotQR.ECL.M }) !== null, 'a 2000-byte link should still fit a plain code');
check(DotQR.makePlain(DotQR.utf8('x'.repeat(3000)), { ecl: DotQR.ECL.M }) === null, 'a 3000-byte link cannot fit and must return null');

console.log(`${passed} checks passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
