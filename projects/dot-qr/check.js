#!/usr/bin/env node
/* Dot QR self-check — run it after touching qr.js, photo.js or the presets:

     node projects/dot-qr/check.js

   No dependencies; exit code 0 means everything passed.
   1) Plain symbols: per-version fingerprints of 1,920 symbols (versions 1–40 × L/M/Q/H × masks).
      Those exact symbols were compared module-for-module against node-qrcode 1.5.4 on
      2026-11-09 and were identical, so a changed fingerprint means the encoder changed.
   2) Drawings in the middle: every preset in index.html × links × grid sizes. When a code is made,
      it reads back as exactly the link, its box shows the drawing exactly (plus a light margin),
      the drawing covers no function pattern except the centre alignment square, no data modules
      form a finder look-alike, and the rendered image puts every module's colour where scanners look.
   3) maxInsetSide tells the truth: a drawing that big fits, one module bigger does not.
   3b) Plain codes: with the mask chosen automatically, no finder look-alikes unless every mask has
       some — then the fewest (200 random links).
   4) Edges: empty drawings, the largest links.
   5) Photo → dots (photo.js) on synthetic phone photos of paper (strong shadow, stains, grain):
      every dot is exactly 0 or 1; blank paper gives no dots, also zoomed out past the photo's
      edges; a 2-px pen ring survives whole with nothing stray; a filled disk is solid; a fleck of
      dust or an even paper texture leaves nothing (Lines); a portrait keeps dark hair black and the
      face white; Invert flips; more ink never means fewer dots (tidying may nudge a few), and Shapes' ink stays between the
      two tone groups; black, noisy-black and white pictures; the crop past the photo's edge counts
      as white in both styles; tidying removes specks, fills pinholes, keeps diagonals.
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
const buttons = [...html.matchAll(/data-preset="(\w+)"/g)].map(m => m[1]);
check(Object.keys(presets).length >= 1, 'could not read the presets from index.html');
check(buttons.length === Object.keys(presets).length && buttons.every(b => b in presets),
  `preset buttons (${buttons}) and PRESETS (${Object.keys(presets)}) must match one to one`);

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
function checkInset(res, d, bytes, tag, curated) {
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
  // Finder look-alikes: none outside the box (the mask choice avoids them); none inside it either for
  // the curated presets (curated = true), since there the drawing itself would be the culprit.
  const fakes = DotQR._lookalikes(res).filter(([x, y]) => curated || !(x >= b.x && x < b.x + b.w && y >= b.y && y < b.y + b.h));
  check(fakes.length === 0, `${tag}: finder look-alike(s) at ${JSON.stringify(fakes)} — scanners like ZXing can take them for a corner square`);
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
    if (res) checkInset(res, d, bytes, tag, true);
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

// Regression: this drawing (an earlier Clawd preset, 24×18) with this link once got a mask whose data
// modules formed a finder look-alike near the top edge, and ZXing could not read the code even as a
// clean image.
{
  const rows = ['.....##############.....', '....################....', '...##################...', '...##################...',
    '...###..########..###...', '...###..########..###...', '...###..########..###...', '...###..########..###...',
    '...##################...', '########################', '########################', '...##################...',
    '...##################...', '...##################...', '....################....', '.....##.##....##.##.....',
    '.....##.##....##.##.....', '.....##.##....##.##.....'];
  const d = { w: 24, h: 18, bits: Uint8Array.from(rows.join(''), c => c === '#' ? 1 : 0) };
  const bytes = DotQR.utf8('mailto:hello@example.com'), res = DotQR.makeInset(bytes, d, { cache: {} });
  check(!!res, 'look-alike regression: no code made');
  if (res) checkInset(res, d, bytes, 'look-alike regression (old Clawd + mailto)', true);
  // The detector itself must still see the look-alike in the code that broke ZXing (that mask and
  // turn, forced), upright and turned on its side.
  const bad = DotQR.makeInset(bytes, d, { cache: {}, masks: [2], rotations: [1] });
  check(!!bad && DotQR._lookalikes(bad).length > 0, 'the look-alike detector misses the pattern that broke ZXing');
  if (bad) {
    const N = bad.size, side = new Uint8Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) side[x * N + y] = bad.modules[y * N + x];
    check(DotQR._lookalikes({ modules: side, size: N, rotation: 0 }).length > 0, 'the look-alike detector misses that pattern when the code is mirrored on its diagonal');
  }
}

// Regression: the default drawing (logo) with this link got a mask whose look-alike only shows when
// the code is seen turned (the scan once covered one orientation only) — ZXing failed it even upright.
if (presets.logo) {
  const bytes = DotQR.utf8('https://7qb.me/eLQzyWCASYl9y7GTcH'), d = DotQR.cropDrawing(bitsOf(presets.logo), 32);
  const res = DotQR.makeInset(bytes, d, { cache: {} });
  check(!!res, 'turned look-alike regression: no code made');
  if (res) checkInset(res, d, bytes, 'turned look-alike regression (logo + 7qb.me)', true);
  const old = DotQR.makeInset(bytes, d, { cache: {}, masks: [4], rotations: [2] });
  check(!!old && DotQR._lookalikes(old).length > 0, 'the look-alike detector misses the turned pattern (logo + 7qb.me, mask 4, turn 2)');
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

/* ---------- 3b) Plain codes avoid finder look-alikes ---------- */
// With the mask chosen automatically, a plain code avoids masks that leave a finder look-alike
// (about 1 in 12 random links otherwise got one that ZXing couldn't read even as a clean image).
// Explicit masks are untouched — the fingerprints in section 1 use them.
{
  let seed = 99, withFakes = 0, readBack = true;
  const rnd = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296;
  const A = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-';
  const links = ['https://LwBG_VkSj_T8h7zDZFRfn28exvMXa'];          // got a look-alike mask before (v3, mask 2)
  for (let i = 0; i < 199; i++) { const L = 10 + Math.floor(rnd() * 140); let s = 'https://'; while (s.length < L) s += A[Math.floor(rnd() * A.length)]; links.push(s); }
  let notFewest = 0;
  for (const link of links) {
    const bytes = DotQR.utf8(link), res = DotQR.makePlain(bytes, { ecl: DotQR.ECL.M }), fakes = DotQR._lookalikes(res).length;
    if (fakes) {                                                  // allowed only if every mask has at least as many
      withFakes++;
      const fewest = Math.min(...[0, 1, 2, 3, 4, 5, 6, 7].map(m => DotQR._lookalikes(DotQR.makePlain(bytes, { ecl: DotQR.ECL.M, mask: m })).length));
      if (fakes > fewest) notFewest++;
    }
    if (!DotQR.verify(res, bytes).ok) readBack = false;
  }
  check(notFewest === 0, `plain codes: ${notFewest} automatic masks leave more finder look-alikes than another mask would`);
  check(withFakes <= links.length / 50, `plain codes: ${withFakes} of ${links.length} have a look-alike — far more than the few that can't avoid one`);
  check(DotQR._lookalikes(DotQR.makePlain(DotQR.utf8(links[0]), { ecl: DotQR.ECL.M })).length === 0, `plain codes: ${links[0]} should get a mask without look-alikes`);
  check(readBack, 'plain codes: an automatic-mask code does not read back');
  // One control per quarter turn of the scan: each code (mask forced) has a single look-alike that the
  // scan only meets at that turn — and ZXing fails a clean render of it at exactly that turn.
  for (const [link, mask, turn, want] of [['https://i96zclsc', 4, 0, '[[14,12]]'], ['https://99rt8qex76fi10cej3nsgbyqdd0', 2, 1, '[[14,14]]'],
    ['https://nbh9cbsxsjqbby', 4, 2, '[[9,17]]'], ['https://984a831h39hv6y4', 7, 3, '[[4,9]]']]) {
    const got = JSON.stringify(DotQR._lookalikes(DotQR.makePlain(DotQR.utf8(link), { ecl: DotQR.ECL.M, mask })));
    check(got === want, `look-alike scan, turn ${turn}: ${link} with mask ${mask} should show ${want}, got ${got}`);
  }
}

/* ---------- 4) Edges ---------- */
check(DotQR.cropDrawing(new Uint8Array(32 * 32), 32) === null, 'an empty drawing must crop to null');
check(DotQR.makePlain(DotQR.utf8('x'.repeat(2000)), { ecl: DotQR.ECL.M }) !== null, 'a 2000-byte link should still fit a plain code');
check(DotQR.makePlain(DotQR.utf8('x'.repeat(3000)), { ecl: DotQR.ECL.M }) === null, 'a 3000-byte link cannot fit and must return null');

/* ---------- 5) Photo → dots (photo.js) ---------- */
// Synthetic 600×600 grey "phone photos of paper": a strong shadow across the page (236 → ~110),
// blotches, pixel grain — then a thin pen ring, a filled disk, or one fleck of dust on it.
const DotPhoto = require('./photo.js');
{
  const W = 600, crop = { x: 0, y: 0, size: W };
  let seed = 7;
  const rnd = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296;
  const grain = a => (rnd() + rnd() + rnd() - 1.5) * a;
  const G = 12, blotch = Array.from({ length: (G + 1) * (G + 1) }, () => grain(12));
  const blot = (x, y) => {                                     // smooth stains up to about ±12, bilinear between grid points
    const gx = x / W * G, gy = y / W * G, ix = Math.min(G - 1, Math.floor(gx)), iy = Math.min(G - 1, Math.floor(gy)), fx = gx - ix, fy = gy - iy;
    return blotch[iy * (G + 1) + ix] * (1 - fx) * (1 - fy) + blotch[iy * (G + 1) + ix + 1] * fx * (1 - fy) +
      blotch[(iy + 1) * (G + 1) + ix] * (1 - fx) * fy + blotch[(iy + 1) * (G + 1) + ix + 1] * fx * fy;
  };
  const pic = (f, w = W, h = W) => { const g = new Uint8ClampedArray(w * h); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) g[y * w + x] = f(x, y); return g; };
  const paper = (x, y) => 236 - 126 * (x + y) / (2 * W) + blot(x, y) + grain(8);
  const R = W * 0.3, ringDist = (x, y) => Math.abs(Math.hypot(x - W / 2, y - W / 2) - R);
  const count = b => b.reduce((a, v) => a + v, 0);
  const blank = pic(paper);
  const ringed = pic((x, y) => ringDist(x, y) < 1 ? 70 + grain(8) : paper(x, y));                  // a 2-px pen line
  const disk = pic((x, y) => Math.hypot(x - W / 2, y - W / 2) < R ? 50 + grain(8) : paper(x, y));
  const dust = pic((x, y) => x >= 304 && x < 314 && y >= 304 && y < 314 ? 40 : paper(x, y));     // inside one cell at 16, 24 and 32
  const ramp = pic((x, y) => (x * 7 + y * 3) % 256);
  const woven = pic((x, y) => paper(x, y) * (1 + 0.12 * Math.sin(2 * Math.PI * x / 75) * Math.sin(2 * Math.PI * y / 75)));   // textured paper
  // A portrait crop: dark hair over the top, a light face in the middle, a darkish background. Not a
  // page, so it must be judged against its own bright end — not against a fitted "paper" surface.
  const inFace = (x, y) => x >= 180 && x < 420 && y >= 270;
  const portrait = pic((x, y) => (y < 270 ? 55 : inFace(x, y) ? 190 : 75) + grain(8));
  const tall = pic((x, y) => 205 + grain(6), 450, 600);                                      // a portrait-shaped page
  const noisyBlack = pic(() => 3 + grain(6));
  // A strip of lettering, its paper brighter than the plain white that the bars of a zoomed-out crop count as.
  const strip = pic((x, y) => y >= 35 && y < 85 && x >= 60 && x < 540 && x % 48 < 30 ? 25 : 238 + grain(3), 600, 120);
  const narrow = pic((x, y) => 150 + 260 * (y / 200 - 0.5) ** 2 + grain(8), 800, 200);      // a narrow page, dim across the middle
  const dim = pic((x, y) => blank[y * W + x] * 0.5);                                         // the blank page in a dim room
  // A page lit evenly but for a shadow over its lower 70% (soft edge, 48 px), the pen ring across both parts.
  const shaded = k => pic((x, y) => {
    const s = Math.min(1, Math.max(0, (y - 0.3 * W + 24) / 48)), p = (236 + blot(x, y) + grain(8)) * (1 - s * (1 - k));
    return ringDist(x, y) < 1 ? p * 0.3 : p;
  });
  const lightShade = shaded(0.6), deepShade = shaded(0.3);
  // A photo with a caption band under it (as in the gallery): dark picture above, white caption with a little text.
  const caption = pic((x, y) => y < 66 ? 35 + grain(10) : y >= 80 && y < 100 && x >= 30 && x < 210 && x % 14 < 4 ? 30 : 245 + grain(2), 600, 120);
  for (const n of [16, 24, 32]) {
    const tag = `photo ${n}×${n}`, cell = W / n;
    const centre = (cx, cy) => Math.hypot((cx + 0.5) * cell - W / 2, (cy + 0.5) * cell - W / 2);
    for (const style of DotPhoto.STYLES) for (const ink of [-1, 0, 1]) for (const invert of [false, true]) {
      const b = DotPhoto.toDots(DotPhoto.measure(ramp, W, W, crop, n), { style, ink, invert });
      check(b.length === n * n && b.every(v => v === 0 || v === 1), `${tag} ${style} ink ${ink}: every dot must be exactly 0 or 1`);
    }
    const mb = DotPhoto.measure(blank, W, W, crop, n);
    for (const style of DotPhoto.STYLES) check(count(DotPhoto.toDots(mb, { style })) === 0, `${tag} ${style}: blank, shadowed, stained paper should give no dots`);
    // Lines finds the thin ring all the way round, and nothing far from it.
    const mr = DotPhoto.measure(ringed, W, W, crop, n), lines = DotPhoto.toDots(mr, {});
    let missed = 0, stray = 0;
    for (let cy = 0; cy < n; cy++) for (let cx = 0; cx < n; cx++) {
      const off = Math.abs(centre(cx, cy) - R);
      if (off < cell * 0.3 && !lines[cy * n + cx]) missed++;
      if (off > cell * 1.5 && lines[cy * n + cx]) stray++;
    }
    check(missed === 0, `${tag}: Lines lost ${missed} cells of a thin pen ring`);
    check(stray === 0, `${tag}: Lines put ${stray} dots far from the ring`);
    const inv = DotPhoto.toDots(mr, { invert: true });
    check(inv.every((v, i) => v === (lines[i] ^ 1)), `${tag}: Invert must flip every dot`);
    // A filled disk comes out solid in both styles — inside black, outside white.
    const md = DotPhoto.measure(disk, W, W, crop, n);
    for (const style of DotPhoto.STYLES) {
      const b = DotPhoto.toDots(md, { style });
      let bad = 0;
      for (let cy = 0; cy < n; cy++) for (let cx = 0; cx < n; cx++) {
        const r = centre(cx, cy);
        if ((r < R - cell * 1.2 && !b[cy * n + cx]) || (r > R + cell * 1.2 && b[cy * n + cx])) bad++;
      }
      check(bad === 0, `${tag} ${style}: a filled disk should be solid inside and clean outside (${bad} cells wrong)`);
    }
    check(count(DotPhoto.toDots(DotPhoto.measure(dust, W, W, crop, n), {})) === 0, `${tag}: a lone fleck of dust should not leave a dot`);
    check(count(DotPhoto.toDots(DotPhoto.measure(woven, W, W, crop, n), {})) === 0, `${tag}: Lines should read an even paper texture as grain, not as dots`);
    for (const style of DotPhoto.STYLES) {                    // more ink → more black dots (tidying may nudge by a few)
      const counts = [-1, -0.5, 0, 0.5, 1].map(ink => count(DotPhoto.toDots(DotPhoto.measure(ramp, W, W, crop, n), { style, ink })));
      check(counts.every((c, i) => i === 0 || c >= counts[i - 1] - 3), `${tag} ${style}: dot count should grow with ink (${counts})`);
    }
    // Zoomed out past the photo's edges (white bars), blank paper is still blank: the edge is no line.
    for (const style of DotPhoto.STYLES) {
      const bars = count(DotPhoto.toDots(DotPhoto.measure(tall, 450, 600, { x: -80, y: 0, size: 600 }, n), { style }));
      const frame = count(DotPhoto.toDots(DotPhoto.measure(blank, W, W, { x: -60, y: -60, size: 720 }, n), { style }));
      check(bars === 0 && frame === 0, `${tag} ${style}: zoomed out past the photo, blank paper should still give no dots (${bars}, ${frame})`);
      // A cell less than half on the photo stays white, even when the photo's paper is brighter than the bars.
      const m = DotPhoto.measure(strip, 600, 120, { x: 0, y: -240, size: 600 }, n), b = DotPhoto.toDots(m, { style });
      let off = 0, on = 0;
      for (let i = 0; i < n * n; i++) { if (m.cov[i] <= 0.5) off += b[i]; else on += b[i]; }
      check(off === 0 && on > 0, `${tag} ${style}: zoomed out on a strip of lettering, nothing off the photo should turn black (${off}) and the letters should show (${on})`);
      const mc = DotPhoto.measure(caption, 600, 120, { x: 0, y: -240, size: 600 }, n), bc = DotPhoto.toDots(mc, { style });
      let offC = 0;
      for (let i = 0; i < n * n; i++) if (mc.cov[i] <= 0.5) offC += bc[i];
      check(offC === 0, `${tag} ${style}: zoomed out on a photo with a caption band, nothing off the photo should turn black (${offC})`);
      // Past a narrow photo's edges the fitted light is only a guess; it must not set how bright the paper is.
      // (At 16 the strip is just 4 rows, and Shapes may split its lit rows from its dim ones.)
      const wide = n >= 24 ? count(DotPhoto.toDots(DotPhoto.measure(narrow, 800, 200, { x: 0, y: -300, size: 800 }, n), { style })) : 0;
      check(wide === 0, `${tag} ${style}: a narrow, unevenly lit blank page seen whole should give no dots (${wide})`);
      const dimDots = count(DotPhoto.toDots(DotPhoto.measure(dim, W, W, crop, n), { style }));
      check(dimDots === 0, `${tag} ${style}: blank paper in a dim room should give no dots (${dimDots})`);
    }
    // A page mostly in shadow. A light shadow is evened out: the ring shows in both parts and nothing else
    // does, bar the shadow's own edge. A deep one comes out black — the price of telling portraits from
    // pages in lighting() — but at Ink −1 the paper is white again and the ring still shows.
    for (const [pic2, ink, what] of [[lightShade, 0, 'mostly in light shadow'], [deepShade, -1, 'mostly in deep shadow, at Ink −1,']]) {
      const b = DotPhoto.toDots(DotPhoto.measure(pic2, W, W, crop, n), { ink });
      let missed = 0, stray = 0;
      for (let cy = 0; cy < n; cy++) for (let cx = 0; cx < n; cx++) {
        const off = Math.abs(centre(cx, cy) - R), edge = Math.abs((cy + 0.5) * cell - 0.3 * W) < 24 + 2 * cell;
        if (off < cell * 0.3 && !b[cy * n + cx]) missed++;
        if (off > cell * 1.5 && !edge && b[cy * n + cx]) stray++;
      }
      check(missed === 0 && stray === 0, `${tag}: a page ${what} should show the ring and nothing else (${missed} ring cells lost, ${stray} stray dots)`);
    }
    // The portrait: hair and background black, face white (Lines).
    {
      const b = DotPhoto.toDots(DotPhoto.measure(portrait, W, W, crop, n), {});
      let hair = 0, hairN = 0, face = 0, faceN = 0, bg = 0, bgN = 0;
      for (let cy = 0; cy < n; cy++) for (let cx = 0; cx < n; cx++) {
        const x0 = cx * cell, y0 = cy * cell, x1 = x0 + cell, y1 = y0 + cell;
        if (y1 <= 270) { hairN++; hair += b[cy * n + cx]; }
        else if (inFace(x0, y0) && inFace(x1 - 0.01, y1 - 0.01) && x0 >= 180 + cell && x1 <= 420 - cell && y0 >= 270 + cell) { faceN++; face += b[cy * n + cx]; }
        else if (y0 >= 270 + cell && (x1 <= 180 - cell || x0 >= 420 + cell)) { bgN++; bg += b[cy * n + cx]; }
      }
      check(hair >= 0.9 * hairN && bg >= 0.9 * bgN && face <= 0.1 * faceN, `${tag}: a portrait should keep dark hair and background black and the face white (hair ${hair}/${hairN}, background ${bg}/${bgN}, face ${face}/${faceN})`);
    }
    // Ink never pushes Shapes all the way: at −1 the disk is still there, at +1 the paper is still white.
    {
      const lo = DotPhoto.toDots(md, { style: 'shapes', ink: -1 }), hi = DotPhoto.toDots(md, { style: 'shapes', ink: 1 });
      let inside = 0, insideN = 0, outside = 0, outsideN = 0;
      for (let cy = 0; cy < n; cy++) for (let cx = 0; cx < n; cx++) {
        const r = centre(cx, cy);
        if (r < R - cell * 1.5) { insideN++; inside += lo[cy * n + cx]; }
        if (r > R + cell * 1.5) { outsideN++; outside += hi[cy * n + cx]; }
      }
      check(inside >= 0.9 * insideN && outside <= 0.1 * outsideN, `${tag}: Shapes ink ±1 should stay between the disk and the paper (disk at −1: ${inside}/${insideN}, paper at +1: ${outside}/${outsideN})`);
    }
    for (const style of DotPhoto.STYLES) {
      check(count(DotPhoto.toDots(DotPhoto.measure(noisyBlack, W, W, crop, n), { style })) === n * n, `${tag} ${style}: a nearly black, slightly noisy photo should be all black`);
      check(count(DotPhoto.toDots(DotPhoto.measure(pic(() => 0), W, W, crop, n), { style })) === n * n, `${tag} ${style}: a black picture should be all black`);
      check(count(DotPhoto.toDots(DotPhoto.measure(pic(() => 255), W, W, crop, n), { style })) === 0, `${tag} ${style}: a white picture should be all white`);
    }
    // A black 600×300 photo seen whole (square crop runs past its top and bottom): those parts are white.
    for (const style of DotPhoto.STYLES) {
      const wide = DotPhoto.toDots(DotPhoto.measure(pic(() => 0, W, W / 2), W, W / 2, { x: 0, y: -W / 4, size: W }, n), { style });
      let rowsOk = true;
      for (let y = 0; y < n; y++) {
        const c = (y + 0.5) * cell - W / 4, want = c > cell && c < W / 2 - cell ? 1 : c < -cell || c > W / 2 + cell ? 0 : -1;
        for (let x = 0; x < n; x++) if (want >= 0 && wide[y * n + x] !== want) rowsOk = false;
      }
      check(rowsOk, `${tag} ${style}: parts of the crop outside the photo must count as white`);
    }
  }
  // Tidying, on dots directly: a lone speck goes, a pinhole fills, a thin diagonal line stays.
  {
    const n = 12, at = (x, y) => y * n + x;
    const speck = new Uint8Array(n * n); speck[at(5, 6)] = 1;
    const hole = new Uint8Array(n * n).fill(1); hole[at(5, 6)] = 0;
    const diag = new Uint8Array(n * n); for (let i = 2; i < 10; i++) diag[at(i, i)] = 1;
    check(count(DotPhoto._despeckle(speck, n)) === 0, 'tidy: a lone black dot should go');
    check(count(DotPhoto._despeckle(hole, n)) === n * n, 'tidy: a lone white pinhole should fill');
    check(DotPhoto._despeckle(diag, n).every((v, i) => v === diag[i]), 'tidy: a one-dot diagonal line should stay as it is');
  }
  const g = DotPhoto.grayFromRGBA(Uint8ClampedArray.from([0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 255, 0, 255, 0, 0, 255, 255]), 5, 1);
  check(g[0] === 255 && g[1] === 0 && g[2] === 255, 'grayFromRGBA: transparent → white paper, black → 0, white → 255');
  check(g[3] === 150 && g[4] === 29, `grayFromRGBA: pure green → 150, pure blue → 29 (Rec. 601 weights), got ${g[3]}, ${g[4]}`);
}

console.log(`${passed} checks passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
