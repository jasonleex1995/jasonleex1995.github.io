/* ============================================================
   Dot QR — a QR code encoder that puts a drawing, untouched, in the middle of the code.
   Vanilla JS, no dependencies. Browser: window.DotQR · Node: module.exports.

   How the drawing gets in (the idea is Russ Cox's QArt codes, 2012): after the link and its
   4-bit terminator, every remaining data bit is free — decoders stop reading at the
   terminator. Reed–Solomon is linear, so every module of the symbol is an XOR (GF(2)-linear)
   function of those free bits, and Gaussian elimination can pick them so that every module
   in the drawing's box shows the drawing exactly. Nothing is damaged: the result is an
   ordinary QR code with zero errors. See makeInset().
   ============================================================ */
(function (root) {
  'use strict';

  /* ---------- GF(256), primitive polynomial x^8+x^4+x^3+x^2+1 (0x11D) ---------- */
  var EXP = new Uint8Array(512), LOG = new Uint8Array(256);
  (function () {
    var x = 1;
    for (var i = 0; i < 255; i++) {
      EXP[i] = x; LOG[x] = i;
      x <<= 1;
      if (x & 0x100) x ^= 0x11D;
    }
    for (i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
  })();
  function gmul(a, b) { return (a === 0 || b === 0) ? 0 : EXP[LOG[a] + LOG[b]]; }

  var GEN = {};
  // Generator polynomial Π(x − α^i), i = 0..degree−1. Coefficients highest degree first; g[0] = 1.
  function rsGenerator(degree) {
    if (GEN[degree]) return GEN[degree];
    var g = [1];
    for (var i = 0; i < degree; i++) {
      var next = [];
      for (var k = 0; k <= g.length; k++) next.push(0);
      for (var j = 0; j < g.length; j++) {
        next[j] ^= g[j];
        next[j + 1] ^= gmul(g[j], EXP[i]);
      }
      g = next;
    }
    return (GEN[degree] = g);
  }

  // The n error-correction codewords: remainder of data(x)·x^n mod g(x).
  function rsEncode(data, n) {
    var g = rsGenerator(n), rem = new Uint8Array(n);
    for (var i = 0; i < data.length; i++) {
      var f = data[i] ^ rem[0];
      for (var j = 0; j < n - 1; j++) rem[j] = rem[j + 1] ^ gmul(g[j + 1], f);
      rem[n - 1] = gmul(g[n], f);
    }
    return rem;
  }

  /* ---------- Version tables (ISO/IEC 18004), index = version, [0] unused ---------- */
  // Error-correction level index: 0 = L, 1 = M, 2 = Q, 3 = H.
  var ECC_PER_BLOCK = [
    [-1, 7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    [-1, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
    [-1, 13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    [-1, 17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
  ];
  var NUM_BLOCKS = [
    [-1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 8, 8, 9, 9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25],
    [-1, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49],
    [-1, 1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38, 40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68],
    [-1, 1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81]
  ];
  var ECL_FORMAT_BITS = [1, 0, 3, 2];
  var MIN_VERSION = 1, MAX_VERSION = 40;

  function sizeOf(ver) { return ver * 4 + 17; }

  // Modules left for codewords + remainder bits once all function patterns are drawn.
  function numRawDataModules(ver) {
    var size = sizeOf(ver), r = size * size;
    r -= 8 * 8 * 3;              // finders + separators
    r -= 15 * 2 + 1;             // two format-info copies + the dark module
    r -= (size - 16) * 2;        // timing patterns
    if (ver >= 2) {
      var na = Math.floor(ver / 7) + 2;
      r -= (na - 1) * (na - 1) * 25;   // alignment patterns not overlapping a finder
      r -= (na - 2) * 2 * 20;          // ...minus their overlap with the timing patterns
      if (ver >= 7) r -= 6 * 3 * 2;    // two version-info copies
    }
    return r;
  }

  function numDataCodewords(ver, ecl) {
    return Math.floor(numRawDataModules(ver) / 8) - ECC_PER_BLOCK[ecl][ver] * NUM_BLOCKS[ecl][ver];
  }

  function alignmentPositions(ver) {
    if (ver === 1) return [];
    var na = Math.floor(ver / 7) + 2;
    var step = Math.floor((ver * 8 + na * 3 + 5) / (na * 4 - 4)) * 2;
    var out = [6];
    for (var pos = ver * 4 + 10; out.length < na; pos -= step) out.splice(1, 0, pos);
    return out;
  }

  /* ---------- Per-version template: function patterns + data placement order ---------- */
  var TEMPLATES = {};
  function template(ver) {
    if (TEMPLATES[ver]) return TEMPLATES[ver];
    var size = sizeOf(ver), n = size * size;
    var mod = new Uint8Array(n), fn = new Uint8Array(n);
    function set(x, y, dark) { mod[y * size + x] = dark ? 1 : 0; fn[y * size + x] = 1; }
    var i, dx, dy;

    for (i = 0; i < size; i++) { set(6, i, i % 2 === 0); set(i, 6, i % 2 === 0); }    // timing

    [[3, 3], [size - 4, 3], [3, size - 4]].forEach(function (c) {                          // finders + separators
      for (dy = -4; dy <= 4; dy++) for (dx = -4; dx <= 4; dx++) {
        var x = c[0] + dx, y = c[1] + dy;
        if (x < 0 || x >= size || y < 0 || y >= size) continue;
        var d = Math.max(Math.abs(dx), Math.abs(dy));
        set(x, y, d !== 2 && d !== 4);
      }
    });

    var al = alignmentPositions(ver), na = al.length;                                      // alignment
    for (var a = 0; a < na; a++) for (var b = 0; b < na; b++) {
      if ((a === 0 && b === 0) || (a === 0 && b === na - 1) || (a === na - 1 && b === 0)) continue;
      for (dy = -2; dy <= 2; dy++) for (dx = -2; dx <= 2; dx++)
        set(al[a] + dx, al[b] + dy, Math.max(Math.abs(dx), Math.abs(dy)) !== 1);
    }

    for (i = 0; i < 9; i++) { fn[8 * size + i] = 1; fn[i * size + 8] = 1; }                // format areas
    for (i = 0; i < 8; i++) { fn[8 * size + (size - 1 - i)] = 1; fn[(size - 1 - i) * size + 8] = 1; }
    set(8, size - 8, true);                                                                 // dark module

    if (ver >= 7) {                                                                         // version info
      var rem = ver;
      for (i = 0; i < 12; i++) rem = (rem << 1) ^ ((rem >>> 11) * 0x1F25);
      var vbits = ver << 12 | rem;
      for (i = 0; i < 18; i++) {
        var bit = (vbits >>> i) & 1, p = size - 11 + i % 3, q = Math.floor(i / 3);
        set(p, q, bit); set(q, p, bit);
      }
    }

    var order = [];                                                                         // zigzag, bottom-right first
    for (var right = size - 1; right >= 1; right -= 2) {
      if (right === 6) right = 5;
      var upward = ((right + 1) & 2) === 0;
      for (var vert = 0; vert < size; vert++) for (var j = 0; j < 2; j++) {
        var xx = right - j, yy = upward ? size - 1 - vert : vert;
        if (!fn[yy * size + xx]) order.push(yy * size + xx);
      }
    }
    if (order.length !== numRawDataModules(ver)) throw new Error('placement mismatch at version ' + ver);

    var masks = [];
    for (var m = 0; m < 8; m++) {
      var mk = new Uint8Array(order.length);
      for (i = 0; i < order.length; i++) mk[i] = maskBit(m, order[i] % size, Math.floor(order[i] / size)) ? 1 : 0;
      masks.push(mk);
    }
    return (TEMPLATES[ver] = { ver: ver, size: size, mod: mod, fn: fn, order: Int32Array.from(order), masks: masks });
  }

  function maskBit(m, x, y) {   // x = column, y = row
    switch (m) {
      case 0: return (x + y) % 2 === 0;
      case 1: return y % 2 === 0;
      case 2: return x % 3 === 0;
      case 3: return (x + y) % 3 === 0;
      case 4: return (Math.floor(x / 3) + Math.floor(y / 2)) % 2 === 0;
      case 5: return x * y % 2 + x * y % 3 === 0;
      case 6: return (x * y % 2 + x * y % 3) % 2 === 0;
      default: return ((x + y) % 2 + x * y % 3) % 2 === 0;
    }
  }

  function formatBits(ecl, mask) {
    var data = ECL_FORMAT_BITS[ecl] << 3 | mask, rem = data;
    for (var i = 0; i < 10; i++) rem = (rem << 1) ^ ((rem >>> 9) * 0x537);
    return (data << 10 | rem) ^ 0x5412;
  }

  // The 15 format-bit positions, copy 1 then copy 2, as [x, y] (bit i of formatBits()).
  function formatPositions(size) {
    var a = [], b = [], i;
    for (i = 0; i <= 5; i++) a.push([8, i]);
    a.push([8, 7], [8, 8], [7, 8]);
    for (i = 9; i < 15; i++) a.push([14 - i, 8]);
    for (i = 0; i < 8; i++) b.push([size - 1 - i, 8]);
    for (i = 8; i < 15; i++) b.push([8, size - 15 + i]);
    return [a, b];
  }

  function drawFormat(mod, size, bits) {
    formatPositions(size).forEach(function (copy) {
      copy.forEach(function (p, i) { mod[p[1] * size + p[0]] = (bits >>> i) & 1; });
    });
    mod[(size - 8) * size + 8] = 1;
  }

  /* ---------- Block layout + interleaving ---------- */
  var LAYOUTS = {};
  function layout(ver, ecl) {
    var key = ver * 4 + ecl;
    if (LAYOUTS[key]) return LAYOUTS[key];
    var nb = NUM_BLOCKS[ecl][ver], ecLen = ECC_PER_BLOCK[ecl][ver];
    var raw = Math.floor(numRawDataModules(ver) / 8);
    var numShort = nb - raw % nb, shortLen = Math.floor(raw / nb);
    var blocks = [], start = 0, b, c;
    for (b = 0; b < nb; b++) {
      var dLen = shortLen - ecLen + (b < numShort ? 0 : 1);
      blocks.push({ start: start, dataLen: dLen });
      start += dLen;
    }
    // stream codeword p → which block, data or EC, index within that part
    var sBlock = new Int32Array(raw), sIdx = new Int32Array(raw), sEc = new Uint8Array(raw), p = 0;
    for (c = 0; c <= shortLen - ecLen; c++) for (b = 0; b < nb; b++)
      if (c < blocks[b].dataLen) { sBlock[p] = b; sIdx[p] = c; p++; }
    for (c = 0; c < ecLen; c++) for (b = 0; b < nb; b++) { sBlock[p] = b; sIdx[p] = c; sEc[p] = 1; p++; }
    return (LAYOUTS[key] = { ver: ver, ecl: ecl, ecLen: ecLen, raw: raw, dataCw: start, blocks: blocks, sBlock: sBlock, sIdx: sIdx, sEc: sEc });
  }

  // Data codewords → finished module matrix (QR orientation). `remainderFinal`, if given,
  // sets the final colour of the 0–7 remainder modules (decoders ignore them).
  function buildMatrix(ver, ecl, mask, dataCw, remainderFinal) {
    var t = template(ver), L = layout(ver, ecl), size = t.size;
    var mod = t.mod.slice();
    var ecs = L.blocks.map(function (bl) { return rsEncode(dataCw.subarray(bl.start, bl.start + bl.dataLen), L.ecLen); });
    var stream = new Uint8Array(L.raw);
    for (var p = 0; p < L.raw; p++) {
      var b = L.sBlock[p], i = L.sIdx[p];
      stream[p] = L.sEc[p] ? ecs[b][i] : dataCw[L.blocks[b].start + i];
    }
    var order = t.order, mk = t.masks[mask], nbits = L.raw * 8;
    for (var k = 0; k < order.length; k++) {
      if (k < nbits) mod[order[k]] = ((stream[k >>> 3] >>> (7 - (k & 7))) & 1) ^ mk[k];
      else mod[order[k]] = remainderFinal ? remainderFinal[k - nbits] : mk[k];
    }
    drawFormat(mod, size, formatBits(ecl, mask));
    return mod;
  }

  /* ---------- Data encoding (byte mode) ---------- */
  function utf8(text) {
    if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(text);
    var s = unescape(encodeURIComponent(text)), out = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }

  function pushBits(arr, val, len) { for (var i = len - 1; i >= 0; i--) arr.push((val >>> i) & 1); }

  function headerAndData(bytes, ver) {
    var bits = [];
    pushBits(bits, 4, 4);                                  // mode indicator 0100 = byte
    pushBits(bits, bytes.length, ver <= 9 ? 8 : 16);       // character count
    for (var i = 0; i < bytes.length; i++) pushBits(bits, bytes[i], 8);
    return bits;
  }

  function bitsNeeded(bytes, ver) { return 4 + (ver <= 9 ? 8 : 16) + bytes.length * 8; }

  function fits(bytes, ver, ecl) {
    if (ver <= 9 && bytes.length > 255) return false;
    return bitsNeeded(bytes, ver) <= numDataCodewords(ver, ecl) * 8;
  }

  function smallestVersion(bytes, ecl) {
    for (var v = MIN_VERSION; v <= MAX_VERSION; v++) if (fits(bytes, v, ecl)) return v;
    return -1;
  }

  function bitsToBytes(bits, nBytes) {
    var out = new Uint8Array(nBytes);
    for (var i = 0; i < bits.length; i++) if (bits[i]) out[i >>> 3] |= 0x80 >>> (i & 7);
    return out;
  }

  /* ---------- Plain QR: standard padding, mask chosen by the standard penalty ---------- */
  function makePlain(bytes, opts) {
    opts = opts || {};
    var ecl = opts.ecl == null ? 1 : opts.ecl;
    var ver = opts.version || smallestVersion(bytes, ecl);
    if (ver < 0 || !fits(bytes, ver, ecl)) return null;
    var cap = numDataCodewords(ver, ecl) * 8;
    var bits = headerAndData(bytes, ver);
    for (var t = Math.min(4, cap - bits.length); t > 0; t--) bits.push(0);   // terminator
    while (bits.length % 8) bits.push(0);
    for (var pad = 0xEC; bits.length < cap; pad ^= 0xEC ^ 0x11) pushBits(bits, pad, 8);
    var dataCw = bitsToBytes(bits, cap / 8);
    var size = sizeOf(ver), best = null;
    var masks = opts.mask != null ? [opts.mask] : [0, 1, 2, 3, 4, 5, 6, 7];
    masks.forEach(function (m) {
      var mod = buildMatrix(ver, ecl, m, dataCw);
      var score = masks.length > 1 ? penalty(mod, size) : 0, fakes = masks.length > 1 ? lookalikes(mod, size).length : 0;
      if (!best || fakes < best.fakes || (fakes === best.fakes && score < best.score)) best = { mod: mod, mask: m, score: score, fakes: fakes };
    });
    // The standard picks the lowest penalty; we pick the lowest penalty among masks without a finder
    // look-alike (see lookalikes), else the mask with the fewest — about 1 in 12 links otherwise gets a
    // mask ZXing can't read even as a clean image (node-qrcode picks the same masks). Any mask is valid;
    // decoders read it from the format bits. A mask asked for explicitly is used as is.
    return { size: size, version: ver, ecl: ecl, mask: best.mask, rotation: 0,
             modules: best.mod, isFunction: template(ver).fn.slice(), art: false };
  }

  // ISO/IEC 18004 §7.8.3 mask penalty (N1 runs, N2 blocks, N3 finder look-alikes, N4 balance).
  function penalty(mod, size) {
    var score = 0, x, y, i, dark = 0;
    for (var pass = 0; pass < 2; pass++) {
      for (y = 0; y < size; y++) {
        var run = 0, prev = -1, line = [];
        for (x = 0; x < size; x++) {
          var v = pass === 0 ? mod[y * size + x] : mod[x * size + y];
          line.push(v);
          if (v === prev) run++;
          else { if (run >= 5) score += run - 2; run = 1; prev = v; }
        }
        if (run >= 5) score += run - 2;
        for (i = 0; i + 11 <= size; i++) {          // 1011101 with 4 light modules on one side
          var s = line.slice(i, i + 11).join('');
          if (s === '10111010000' || s === '00001011101') score += 40;
        }
      }
    }
    for (y = 0; y + 1 < size; y++) for (x = 0; x + 1 < size; x++) {
      var c = mod[y * size + x];
      if (c === mod[y * size + x + 1] && c === mod[(y + 1) * size + x] && c === mod[(y + 1) * size + x + 1]) score += 3;
    }
    for (i = 0; i < size * size; i++) dark += mod[i];
    score += Math.floor(Math.abs(dark * 20 - size * size * 10) / (size * size)) * 10;
    return score;
  }

  /* ---------- Finder look-alikes ---------- */
  // Scanners find the three corner squares by looking for dark·light·dark·light·dark runs in roughly
  // 1:1:3:1:1 proportion that hold up vertically through their centre too. ZXing accepts anything
  // within half a module of those proportions. A mask (or steered free bits) can leave such a pattern
  // in the data area, and a decoder may then take it for a corner square: ZXing failed to read one
  // code even as a clean image (an earlier Clawd preset + mailto:hello@example.com, 2026-11-10), and
  // about 1 plain code in 12 had the same problem with the standard lowest-penalty mask. This replays
  // ZXing's row scan and cross-checks on the module grid (the ratios don't depend on scale) at all
  // four quarter turns, and returns the look-alikes that are not one of the three real corner squares.
  // It flags a little more than ZXing confirms (one hit is enough, no row skipping) — the safe side.
  // mod is in QR orientation.
  function finderLike(c) {
    var t = c[0] + c[1] + c[2] + c[3] + c[4], m = t / 7, v = m / 2;
    return c[0] > 0 && c[1] > 0 && c[2] > 0 && c[3] > 0 && c[4] > 0 &&
      Math.abs(m - c[0]) < v && Math.abs(m - c[1]) < v && Math.abs(3 * m - c[2]) < 3 * v &&
      Math.abs(m - c[3]) < v && Math.abs(m - c[4]) < v;
  }
  function crossCheck(at, lo, hi, start, maxCount, origTotal, strict) {   // ZXing's cross-checks, on modules
    var c = [0, 0, 0, 0, 0], i = start;
    while (i >= lo && at(i)) { c[2]++; i--; }
    if (i < lo) return NaN;
    while (i >= lo && !at(i) && c[1] <= maxCount) { c[1]++; i--; }
    if (i < lo || c[1] > maxCount) return NaN;
    while (i >= lo && at(i) && c[0] <= maxCount) { c[0]++; i--; }
    if (c[0] > maxCount) return NaN;
    i = start + 1;
    while (i <= hi && at(i)) { c[2]++; i++; }
    if (i > hi) return NaN;
    while (i <= hi && !at(i) && c[3] < maxCount) { c[3]++; i++; }
    if (i > hi || c[3] >= maxCount) return NaN;
    while (i <= hi && at(i) && c[4] < maxCount) { c[4]++; i++; }
    if (c[4] >= maxCount) return NaN;
    var tot = c[0] + c[1] + c[2] + c[3] + c[4];
    if (5 * Math.abs(tot - origTotal) >= (strict ? 1 : 2) * origTotal) return NaN;   // vertical 40 %, horizontal 20 %
    return finderLike(c) ? i - c[4] - c[3] - c[2] / 2 : NaN;
  }
  function rowScanLookalikes(dark, size) {
    var lo = -4, hi = size + 3, found = [];                        // the quiet zone counts as light
    for (var y = 0; y < size; y++) {
      var runs = [], x = lo;
      while (x <= hi) { var v = dark(x, y), s0 = x; while (x <= hi && dark(x, y) === v) x++; runs.push({ v: v, s: s0, n: x - s0 }); }
      for (var k = 0; k + 4 < runs.length; k++) {
        if (!runs[k].v) continue;
        var c = [runs[k].n, runs[k + 1].n, runs[k + 2].n, runs[k + 3].n, runs[k + 4].n];
        if (!finderLike(c)) continue;
        var total = c[0] + c[1] + c[2] + c[3] + c[4], col = Math.floor(runs[k + 2].s + c[2] / 2);
        var cy = crossCheck(function (i) { return dark(col, i); }, lo, hi, y, c[2], total);
        if (isNaN(cy)) continue;
        var row = Math.floor(cy);
        var cx = crossCheck(function (i) { return dark(i, row); }, lo, hi, col, c[2], total, true);
        if (!isNaN(cx)) found.push([cx, cy]);
      }
    }
    return found;
  }
  function lookalikes(mod, size, skip) {                          // skip(x, y): ignore look-alikes centred there
    var at = function (x, y) { return x >= 0 && y >= 0 && x < size && y < size && mod[y * size + x] === 1; };
    // What ZXing registers depends on which way the image runs (rounding of a centre on a module edge,
    // the asymmetric cross-check), and a phone may see the code at any quarter turn: scan all four.
    var found = [];
    [0, 1, 2, 3].forEach(function (r) {                           // the code turned r quarter turns clockwise
      rowScanLookalikes(function (x, y) {
        for (var k = 0; k < r; k++) { var nx = y; y = size - 1 - x; x = nx; }
        return at(x, y);
      }, size).forEach(function (p) {
        var X = p[0], Y = p[1];
        for (var k = 0; k < r; k++) { var nX = Y; Y = size - X; X = nX; }
        found.push([X, Y]);
      });
    });
    var real = [[3.5, 3.5], [size - 3.5, 3.5], [3.5, size - 3.5]], out = [];
    found.forEach(function (p) {
      if (real.some(function (r) { return Math.abs(p[0] - r[0]) < 3 && Math.abs(p[1] - r[1]) < 3; })) return;
      if (skip && skip(Math.min(size - 1, Math.floor(p[0])), Math.min(size - 1, Math.floor(p[1])))) return;
      if (!out.some(function (o) { return Math.abs(o[0] - p[0]) < 2 && Math.abs(o[1] - p[1]) < 2; })) out.push(p);
    });
    return out;
  }

  /* ---------- Rotation helpers (QR orientation ↔ display orientation) ---------- */
  // r = number of clockwise quarter turns applied to the QR to get the displayed image.
  function rotIndex(r, size, x, y) {
    switch (r & 3) {
      case 0: return y * size + x;
      case 1: return x * size + (size - 1 - y);
      case 2: return (size - 1 - y) * size + (size - 1 - x);
      default: return (size - 1 - x) * size + y;
    }
  }
  function rotateArray(arr, size, r) {
    if (!(r & 3)) return arr.slice();
    var out = new arr.constructor(arr.length);
    for (var y = 0; y < size; y++) for (var x = 0; x < size; x++) out[rotIndex(r, size, x, y)] = arr[y * size + x];
    return out;
  }
  function unrotateArray(arr, size, r) {
    if (!(r & 3)) return arr.slice();
    var out = new arr.constructor(arr.length);
    for (var y = 0; y < size; y++) for (var x = 0; x < size; x++) out[y * size + x] = arr[rotIndex(r, size, x, y)];
    return out;
  }

  /* ---------- The picture: QArt-style free-bit selection ---------- */
  // Everything about (version, ecl, link) that does not depend on the drawing, mask or rotation.
  function artSystem(bytes, ver, ecl) {
    var t = template(ver), L = layout(ver, ecl);
    var cap = L.dataCw * 8;
    var fixedBits = headerAndData(bytes, ver);
    if (fixedBits.length + 4 > cap) return null;
    fixedBits.push(0, 0, 0, 0);                     // terminator: decoders stop here, the rest is free
    var U = fixedBits.length;
    var fixedCw = bitsToBytes(fixedBits, L.dataCw);  // free bits = 0

    var nbits = L.raw * 8, order = t.order;
    var fixedQ = new Uint8Array(t.size * t.size);    // 1 = a module whose colour the link alone decides
    var blockSys = L.blocks.map(function (bl, b) {
      var lo = Math.max(U, bl.start * 8), hi = (bl.start + bl.dataLen) * 8;
      var k = Math.max(0, hi - lo), W = (k + 31) >>> 5;
      return { b: b, lo: lo, k: k, W: W, mods: [], rows: [], consts: [],
               ecFixed: rsEncode(fixedCw.subarray(bl.start, bl.start + bl.dataLen), L.ecLen) };
    });

    // EC codewords of a unit data byte at each position of each block that holds free bits.
    blockSys.forEach(function (bs) {
      if (!bs.k) return;
      var bl = L.blocks[bs.b], firstByte = (bs.lo >>> 3) - bl.start, unit = [];
      for (var i = firstByte; i < bl.dataLen; i++) {
        var e = new Uint8Array(bl.dataLen); e[i] = 1;
        unit[i] = rsEncode(e, L.ecLen);
      }
      bs.unit = unit;
    });

    for (var kk = 0; kk < nbits; kk++) {
      var p = kk >>> 3, bitInByte = kk & 7, b = L.sBlock[p], bs = blockSys[b], bl = L.blocks[b];
      var pos = order[kk], row = null, cst = 0;
      if (!L.sEc[p]) {
        var g = (bl.start + L.sIdx[p]) * 8 + bitInByte;      // global data-bit index
        if (g >= U) { row = new Uint32Array(bs.W); var j = g - bs.lo; row[j >>> 5] = 1 << (j & 31); }
        else cst = fixedBits[g];
      } else {
        var ei = L.sIdx[p];
        cst = (bs.ecFixed[ei] >>> (7 - bitInByte)) & 1;
        if (bs.k) {
          row = new Uint32Array(bs.W);
          for (var jj = 0; jj < bs.k; jj++) {
            var gg = bs.lo + jj, byteIdx = (gg >>> 3) - bl.start, val = 0x80 >>> (gg & 7);
            var ecb = gmul(bs.unit[byteIdx][ei], val);
            if ((ecb >>> (7 - bitInByte)) & 1) row[jj >>> 5] |= 1 << (jj & 31);
          }
        }
      }
      bs.mods.push(kk); bs.rows.push(row); bs.consts.push(cst);
      if (!row) fixedQ[pos] = 1;
    }
    var freeBits = cap - U;
    return { ver: ver, ecl: ecl, t: t, L: L, U: U, fixedBits: fixedBits, blockSys: blockSys,
             freeBits: freeBits, dataModules: order.length, fixedQ: fixedQ };
  }

  // Greedy GF(2) elimination in priority order (Gauss–Jordan, basis kept fully reduced).
  // Returns the k free-bit values. A module whose row is dependent on higher-priority
  // ones is simply left to whatever value they imply.
  function solveBlock(bs, orderIdx, rhs) {
    var k = bs.k, W = bs.W;
    var basis = new Array(k), bRhs = new Uint8Array(k), piv = new Uint32Array(W), pivList = [];
    var tmp = new Uint32Array(W), snap = new Uint32Array(W), rank = 0, w, u;
    for (var q = 0; q < orderIdx.length && rank < k; q++) {
      var m = orderIdx[q], row = bs.rows[m];
      if (!row) continue;
      tmp.set(row);
      var r = rhs[m];
      for (w = 0; w < W; w++) snap[w] = tmp[w] & piv[w];
      for (w = 0; w < W; w++) {
        var bits = snap[w];
        while (bits) {
          var low = bits & -bits, p = (w << 5) + (31 - Math.clz32(low)), br = basis[p];
          for (u = 0; u < W; u++) tmp[u] ^= br[u];
          r ^= bRhs[p];
          bits ^= low;
        }
      }
      var np = -1;
      for (w = 0; w < W; w++) if (tmp[w]) { np = (w << 5) + (31 - Math.clz32(tmp[w] & -tmp[w])); break; }
      if (np < 0) continue;
      var nr = tmp.slice(), pw = np >>> 5, pb = 1 << (np & 31);
      for (var t = 0; t < pivList.length; t++) {
        var p2 = pivList[t], r2 = basis[p2];
        if (r2[pw] & pb) { for (u = 0; u < W; u++) r2[u] ^= nr[u]; bRhs[p2] ^= r; }
      }
      basis[np] = nr; bRhs[np] = r; piv[pw] |= pb; pivList.push(np); rank++;
    }
    var f = new Uint8Array(k);
    for (var i = 0; i < k; i++) f[i] = basis[i] ? bRhs[i] : 0;
    return f;
  }

  function hash01(i) {            // deterministic pseudo-random in [0, 1) — same input, same code
    var h = Math.imul(i ^ 0x9E3779B9, 0x85EBCA6B);
    h ^= h >>> 13; h = Math.imul(h, 0xC2B2AE35); h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  }

  // Solve every block for one mask. tq = wanted final colour of each module (QR orientation),
  // orders = per block, its modules most-important-first. Returns the finished matrix.
  function steer(sys, mask, tq, orders) {
    var t = sys.t, L = sys.L, order = t.order, mk = t.masks[mask], nbits = L.raw * 8;
    var dataBits = sys.fixedBits.slice();
    for (var g = dataBits.length; g < L.dataCw * 8; g++) dataBits.push(0);
    sys.blockSys.forEach(function (bs, bi) {
      if (!bs.k) return;
      var rhs = new Uint8Array(bs.mods.length);
      for (var i = 0; i < bs.mods.length; i++) { var kk = bs.mods[i]; rhs[i] = tq[order[kk]] ^ mk[kk] ^ bs.consts[i]; }
      var f = solveBlock(bs, orders[bi], rhs);
      for (var j = 0; j < bs.k; j++) dataBits[bs.lo + j] = f[j];
    });
    var rem = new Uint8Array(order.length - nbits);
    for (var q = nbits; q < order.length; q++) rem[q - nbits] = tq[order[q]];
    return buildMatrix(sys.ver, sys.ecl, mask, bitsToBytes(dataBits, L.dataCw), rem);
  }

  // Crop an M×M drawing (1 = dark) to the box around what was drawn; null when nothing is drawn.
  function cropDrawing(bits, M) {
    var x0 = M, y0 = M, x1 = -1, y1 = -1, x, y;
    for (y = 0; y < M; y++) for (x = 0; x < M; x++) if (bits[y * M + x]) {
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    if (x1 < 0) return null;
    var w = x1 - x0 + 1, h = y1 - y0 + 1, out = new Uint8Array(w * h);
    for (y = 0; y < h; y++) for (x = 0; x < w; x++) out[y * w + x] = bits[(y + y0) * M + x + x0];
    return { w: w, h: h, bits: out, x: x0, y: y0 };
  }

  var MAX_INSET_VERSION = 13;     // from version 14 on, several alignment squares sit inside the middle

  // The drawing, intact, in the middle of the code. Every module of the drawing's box (plus a light
  // margin) is steered to the drawing's colour exactly, so the code has zero errors. The smallest
  // version where that works is used: the box must miss every function pattern — except the
  // alignment square at the very centre of version 7–13 codes, which scanners don't need, so the
  // drawing covers it — and must miss the modules that carry the link itself (they can't be steered;
  // turning the code a quarter turn moves them). Among the masks that work, the one with the lowest
  // standard penalty wins, because that is what the penalty is for: easy scanning.
  // drawing = cropDrawing(...); opts: version | minVersion/maxVersion, rotations, masks, margin, cache.
  function makeInset(bytes, drawing, opts) {
    opts = opts || {};
    var ecl = opts.ecl == null ? 0 : opts.ecl, margin = opts.margin == null ? 1 : opts.margin;
    var lo = opts.version || opts.minVersion || MIN_VERSION, hi = Math.min(opts.version || opts.maxVersion || MAX_INSET_VERSION, MAX_INSET_VERSION);
    for (var v = lo; v <= hi; v++) {
      var plan = insetPlan(bytes, drawing.w, drawing.h, v, ecl, margin, opts.rotations || [0, 1, 2, 3], opts.cache);
      var res = plan && insetSolve(plan, drawing, opts.masks || [0, 1, 2, 3, 4, 5, 6, 7]);
      if (res) return res;
    }
    return null;
  }

  // Largest square drawing (side, in modules) that can sit in the middle of a code for these bytes.
  function maxInsetSide(bytes, opts) {
    opts = opts || {};
    var ecl = opts.ecl == null ? 0 : opts.ecl, margin = opts.margin == null ? 1 : opts.margin, best = 0;
    for (var v = MIN_VERSION; v <= MAX_INSET_VERSION; v++) {
      for (var s = sizeOf(v) - 14 - 2 * margin; s > best; s--) {
        if (insetPlan(bytes, s, s, v, ecl, margin, [0, 1, 2, 3], opts.cache)) { best = s; break; }
      }
    }
    return best;
  }

  // Everything that depends only on the box size: where the box goes, which quarter turns keep it
  // clear of function patterns and of the link's own modules. Null if no turn works at this version.
  function insetPlan(bytes, w, h, ver, ecl, margin, rots, cache) {
    if (!fits(bytes, ver, ecl)) return null;
    var size = sizeOf(ver), n = size * size, t = template(ver);
    var al = alignmentPositions(ver), mid = al.length >= 3 ? al[(al.length - 1) >> 1] : -1;
    var Rw = w + 2 * margin, Rh = h + 2 * margin;
    if (mid >= 0) { Rw = Math.max(Rw, 5); Rh = Math.max(Rh, 5); }      // cover the centre square whole, never half
    if (Rw > size - 14 || Rh > size - 14) return null;                // can't even clear the timing lines
    var x0 = Math.floor((size - Rw) / 2), y0 = Math.floor((size - Rh) / 2);
    var inR = new Uint8Array(n);                                      // display orientation
    for (var y = y0; y < y0 + Rh; y++) for (var x = x0; x < x0 + Rw; x++) inR[y * size + x] = 1;
    var okRots = rots.filter(function (r) {
      for (var qy = 0; qy < size; qy++) for (var qx = 0; qx < size; qx++) {
        if (!t.fn[qy * size + qx] || !inR[rotIndex(r, size, qx, qy)]) continue;
        if (!(mid >= 0 && Math.abs(qx - mid) <= 2 && Math.abs(qy - mid) <= 2)) return false;
      }
      return true;
    });
    if (!okRots.length) return null;
    var key = ver + '|' + ecl, sys = cache && cache[key];
    if (!sys) { sys = artSystem(bytes, ver, ecl); if (cache) cache[key] = sys; }
    if (!sys || Rw * Rh > sys.freeBits) return null;
    okRots = okRots.filter(function (r) {
      for (var qi = 0; qi < n; qi++) if (sys.fixedQ[qi] && inR[rotIndex(r, size, qi % size, (qi / size) | 0)]) return false;
      return true;
    });
    if (!okRots.length) return null;
    return { sys: sys, t: t, size: size, n: n, inR: inR, rots: okRots, margin: margin, box: { x: x0, y: y0, w: Rw, h: Rh } };
  }

  // Steer the box to the drawing for each workable turn and mask. Keep the first turn that has a mask
  // leaving no finder look-alike outside the box, with its lowest-penalty such mask. If no turn has one,
  // take each turn's best mask (fewest look-alikes, then lowest penalty) and keep the turn whose best
  // leaves the fewest, the earlier turn on a tie. A single mask asked for — the quick redraw during a
  // stroke — skips the look-alike test.
  // Null if (rarely) the box's modules can't all be set at once.
  function insetSolve(plan, d, masks) {
    var sys = plan.sys, t = plan.t, size = plan.size, n = plan.n, inR = plan.inR, b = plan.box, x, y, i;
    var ox = b.x + Math.floor((b.w - d.w) / 2), oy = b.y + Math.floor((b.h - d.h) / 2), want = new Uint8Array(n);
    for (y = b.y; y < b.y + b.h; y++) for (x = b.x; x < b.x + b.w; x++) {
      var dx = x - ox, dy = y - oy;
      want[y * size + x] = (dx >= 0 && dy >= 0 && dx < d.w && dy < d.h) ? d.bits[dy * d.w + dx] : 0;
    }
    var fallback = null;
    for (var ri = 0; ri < plan.rots.length; ri++) {
      var r = plan.rots[ri], tq = new Uint8Array(n), wq = new Float32Array(n), boxQ = new Uint8Array(n);
      for (y = 0; y < size; y++) for (x = 0; x < size; x++) {
        var di = rotIndex(r, size, x, y), qi = y * size + x;
        boxQ[qi] = inR[di];
        tq[qi] = inR[di] ? want[di] : (hash01(qi * 7 + 3) < 0.5 ? 1 : 0);   // outside: ordinary QR texture
        wq[qi] = (inR[di] ? 2 : 0) + hash01(qi);                            // the box first
      }
      var order = t.order;
      var orders = sys.blockSys.map(function (bs) {
        var idx = [];
        for (var k = 0; k < bs.mods.length; k++) idx.push(k);
        idx.sort(function (a, c) { return wq[order[bs.mods[c]]] - wq[order[bs.mods[a]]]; });
        return idx;
      });
      var best = null;
      for (var mi = 0; mi < masks.length; mi++) {
        var mod = steer(sys, masks[mi], tq, orders), clean = true;
        for (var q = 0; q < n && clean; q++) if (boxQ[q] && !t.fn[q] && mod[q] !== tq[q]) clean = false;
        if (!clean) continue;
        if (masks.length === 1) { best = { pen: 0, mod: mod, mask: masks[mi], fakes: 0 }; continue; }
        var seen = mod.slice();                                     // as drawn: the drawing covers the centre square
        for (q = 0; q < n; q++) if (boxQ[q] && t.fn[q]) seen[q] = tq[q];
        // Only look-alikes outside the box count: inside it every module is the drawing, whatever the mask.
        var fakes = lookalikes(seen, size, function (x, y) { return boxQ[y * size + x]; }).length;
        var cand = { pen: penalty(mod, size), mod: mod, mask: masks[mi], fakes: fakes };
        if (!best || fakes < best.fakes || (fakes === best.fakes && cand.pen < best.pen)) best = cand;
      }
      if (best && (best.fakes === 0 || masks.length === 1)) return finish(best, r);
      if (best && (!fallback || best.fakes < fallback.best.fakes)) fallback = { best: best, r: r };
    }
    return fallback ? finish(fallback.best, fallback.r) : null;

    function finish(pick, r) {
      var shown = rotateArray(pick.mod, size, r), fnD = rotateArray(t.fn, size, r), covered = new Uint8Array(n);
      for (var j = 0; j < n; j++) if (inR[j] && fnD[j]) { covered[j] = 1; shown[j] = want[j]; }
      return {
        size: size, version: sys.ver, ecl: sys.ecl, mask: pick.mask, rotation: r, art: true,
        modules: shown,             // what gets drawn (display orientation)
        isFunction: fnD,
        covered: covered,           // function modules painted over by the drawing (only ever the centre square)
        box: b
      };
    }
  }

  /* ---------- Rendering ---------- */
  // Calls fill(x, y, w, h) for every run of dark modules, in pixel units, quiet zone included.
  function renderModules(res, opts, fill) {
    var size = res.size, px = opts.px, quiet = opts.quiet == null ? 4 : opts.quiet;
    for (var y = 0; y < size; y++) {
      var run = -1;
      for (var x = 0; x <= size; x++) {
        var dark = x < size && res.modules[y * size + x] === 1;
        if (dark && run < 0) run = x;
        if (!dark && run >= 0) { fill((run + quiet) * px, (y + quiet) * px, (x - run) * px, px); run = -1; }
      }
    }
    return (size + quiet * 2) * px;
  }

  /* ---------- Self-check: read the finished symbol back ---------- */
  function verify(res, expectedBytes) {
    var size = res.size, ver = (size - 17) / 4;
    var mod = unrotateArray(res.modules, size, res.rotation || 0);
    var covered = res.covered ? unrotateArray(res.covered, size, res.rotation || 0) : null;
    var pos = formatPositions(size), read = [0, 0];
    pos.forEach(function (copy, ci) { copy.forEach(function (p, i) { read[ci] |= mod[p[1] * size + p[0]] << i; }); });
    if (read[0] !== read[1]) return { ok: false, why: 'format copies differ' };
    var ecl = -1, mask = -1;
    for (var e = 0; e < 4; e++) for (var m = 0; m < 8; m++) if (formatBits(e, m) === read[0]) { ecl = e; mask = m; }
    if (ecl < 0) return { ok: false, why: 'format bits invalid' };
    var t = template(ver), L = layout(ver, ecl), order = t.order, mk = t.masks[mask];
    for (var i = 0; i < size * size; i++)
      if (t.fn[i] && t.mod[i] !== mod[i] && !isFormatModule(i, size) && !(covered && covered[i])) return { ok: false, why: 'function pattern damaged' };
    var stream = new Uint8Array(L.raw);
    for (var k = 0; k < L.raw * 8; k++) stream[k >>> 3] |= ((mod[order[k]] ^ mk[k]) & 1) << (7 - (k & 7));
    var data = new Uint8Array(L.dataCw), ecs = L.blocks.map(function () { return new Uint8Array(L.ecLen); });
    for (var p = 0; p < L.raw; p++) {
      if (L.sEc[p]) ecs[L.sBlock[p]][L.sIdx[p]] = stream[p];
      else data[L.blocks[L.sBlock[p]].start + L.sIdx[p]] = stream[p];
    }
    for (var b = 0; b < L.blocks.length; b++) {
      var bl = L.blocks[b], want = rsEncode(data.subarray(bl.start, bl.start + bl.dataLen), L.ecLen);
      for (var j = 0; j < L.ecLen; j++) if (want[j] !== ecs[b][j]) return { ok: false, why: 'error correction mismatch in block ' + b };
    }
    var bitPos = 0;
    function take(nb) { var v = 0; for (var q = 0; q < nb; q++, bitPos++) v = v << 1 | ((data[bitPos >>> 3] >>> (7 - (bitPos & 7))) & 1); return v; }
    if (take(4) !== 4) return { ok: false, why: 'not byte mode' };
    var count = take(ver <= 9 ? 8 : 16);
    if ((bitPos + count * 8) > L.dataCw * 8) return { ok: false, why: 'length overflows' };
    var got = new Uint8Array(count);
    for (var c = 0; c < count; c++) got[c] = take(8);
    if (L.dataCw * 8 - bitPos >= 4 && take(4) !== 0) return { ok: false, why: 'no terminator after the data' };
    if (expectedBytes) {
      if (got.length !== expectedBytes.length) return { ok: false, why: 'length differs' };
      for (c = 0; c < count; c++) if (got[c] !== expectedBytes[c]) return { ok: false, why: 'content differs' };
    }
    return { ok: true, bytes: got, ecl: ecl, mask: mask };
  }

  function isFormatModule(i, size) {
    var x = i % size, y = Math.floor(i / size);
    return (y === 8 && (x <= 8 || x >= size - 8)) || (x === 8 && (y <= 8 || y >= size - 8));
  }

  var API = {
    ECL: { L: 0, M: 1, Q: 2, H: 3 },
    utf8: utf8, sizeOf: sizeOf, fits: fits, smallestVersion: smallestVersion,
    makePlain: makePlain, cropDrawing: cropDrawing, makeInset: makeInset, maxInsetSide: maxInsetSide,
    renderModules: renderModules, verify: verify,
    // exposed for tests
    _rsEncode: rsEncode, _numDataCodewords: numDataCodewords, _numRawDataModules: numRawDataModules,
    _alignmentPositions: alignmentPositions, _template: template, _layout: layout, _formatBits: formatBits,
    // Finder look-alikes of a finished code, as display-orientation modules [x, y].
    _lookalikes: function (res) {
      var size = res.size, r = res.rotation || 0;
      return lookalikes(unrotateArray(res.modules, size, r), size).map(function (p) {
        var i = rotIndex(r, size, Math.min(size - 1, Math.floor(p[0])), Math.min(size - 1, Math.floor(p[1])));
        return [i % size, (i / size) | 0];
      });
    }
  };
  if (typeof module === 'object' && module.exports) module.exports = API;
  else root.DotQR = API;
})(this);
