#!/usr/bin/env node
/* Dot QR self-check — run it after touching qr.js or the presets:

     node projects/dot-qr/check.js

   No dependencies; exit code 0 means everything passed.
   1) Plain symbols: per-version fingerprints of 1,920 symbols (versions 1–40 × L/M/Q/H × masks).
      Those exact symbols were compared module-for-module against node-qrcode 1.5.4 on
      2026-11-09 and were identical, so a changed fingerprint means the encoder changed.
   2) Picture symbols: every preset in index.html × several links × grid sizes builds, reads
      back as exactly the link (DotQR.verify), and keeps most dots on the drawing.
   3) Rendering: whatever the Blend, the centre pixel of every module shows its true colour —
      that centre is what scanners read.
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

/* ---------- 2) Picture symbols ---------- */
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

const LINKS = ['https://jasonleex1995.github.io', 'https://bit.ly/3xYz9Ab',
  'https://www.example.com/articles/2026/11/a-much-longer-link?ref=newsletter&id=12345', '안녕하세요 Dot QR'];
for (const [name, hex] of Object.entries(presets)) {
  const base = bitsOf(hex);
  for (const link of LINKS) for (const M of [16, 32, 64]) {
    const bits = M === 32 ? base : resample(base, 32, M), bytes = DotQR.utf8(link);
    const ver = DotQR.autoVersion(bytes, M, { maxVersion: 20 });
    const tag = `${name} / grid ${M} / ${link.slice(0, 28)}`;
    if (!(ver > 0)) { check(false, `${tag}: no version chosen`); continue; }
    const N = DotQR.sizeOf(ver), gray = DotQR.fitDrawing(bits, M, N);
    const target = Uint8Array.from(gray, g => (g >= 0.5 ? 1 : 0));
    const res = DotQR.makeArt(bytes, { version: ver, ecl: DotQR.ECL.L, target, weight: DotQR.weights(gray, N) });
    const read = DotQR.verify(res, bytes);
    check(read.ok, `${tag}: does not read back (${read.why})`);
    check(res.match >= 0.6, `${tag}: only ${(res.match * 100).toFixed(0)}% of dots follow the drawing`);

    /* ---------- 3) Rendering: true colour across the middle of each module, drawing colour
                      in the corner of the modules that disagree with it ---------- */
    for (const centre of [0.5, 0.75, 1]) {
      const px = 20, W = (res.size + 8) * px, img = new Uint8Array(W * W);
      DotQR.renderModules(res, target, { px, centre }, (x, y, w, h) => {
        for (let yy = y; yy < y + h; yy++) img.fill(1, yy * W + x, yy * W + x + w);
      });
      const at = (mx, my, fx, fy) => img[((my + 4) * px + Math.floor(fy * px)) * W + (mx + 4) * px + Math.floor(fx * px)];
      let badMiddle = 0, badCorner = 0;
      for (let my = 0; my < res.size; my++) for (let mx = 0; mx < res.size; mx++) {
        const i = my * res.size + mx, v = res.modules[i];
        for (const fy of [0.3, 0.5, 0.7]) for (const fx of [0.3, 0.5, 0.7]) if (at(mx, my, fx, fy) !== v) { badMiddle++; break; }
        const outer = (centre < 1 && !res.isFunction[i]) ? target[i] : v;
        if (at(mx, my, 0.05, 0.05) !== outer) badCorner++;
      }
      check(badMiddle === 0, `${tag}: ${badMiddle} modules lose their true colour in the middle at Blend ${centre}`);
      check(badCorner === 0, `${tag}: ${badCorner} modules have the wrong colour at the corner at Blend ${centre}`);
    }
  }
}

/* ---------- 4) Edges ---------- */
check(DotQR.makePlain(DotQR.utf8('x'.repeat(2000)), { ecl: DotQR.ECL.M }) !== null, 'a 2000-byte link should still fit a plain code');
check(DotQR.makePlain(DotQR.utf8('x'.repeat(3000)), { ecl: DotQR.ECL.M }) === null, 'a 3000-byte link cannot fit and must return null');
check(DotQR.autoVersion(DotQR.utf8('https://example.com/?q=' + 'x'.repeat(700)), 32, { maxVersion: 20 }) === -1,
  'an overlong link must leave the picture out (autoVersion −1)');

console.log(`${passed} checks passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
