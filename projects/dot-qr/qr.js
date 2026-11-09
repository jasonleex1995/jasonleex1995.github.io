/* ============================================================
   Dot QR — a QR code encoder that can steer its modules toward a picture.
   Vanilla JS, no dependencies. Browser: window.DotQR · Node: module.exports.

   How the picture gets in (both ideas are from published work):
   1) QArt codes (Russ Cox, 2012). After the link and its 4-bit terminator, every
      remaining data bit is free — decoders stop reading at the terminator. Reed–Solomon
      is linear, so every module of the symbol is an XOR (GF(2)-linear) function of
      those free bits. Gaussian elimination then picks the free bits so that as many
      modules as possible — most important first — match the drawing. The result has
      zero errors: it is an ordinary, valid QR code.
   2) Halftone rendering (Chu et al., "Halftone QR Codes", SIGGRAPH Asia 2013).
      Scanners sample the centre of each module, so a module whose value disagrees
      with the drawing can be painted in the drawing's colour with only a small centre
      square in its true colour. See renderModules().
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
      var score = masks.length > 1 ? penalty(mod, size) : 0;
      if (!best || score < best.score) best = { mod: mod, mask: m, score: score };
    });
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
    }
    var freeBits = cap - U;
    return { ver: ver, ecl: ecl, t: t, L: L, U: U, fixedBits: fixedBits, blockSys: blockSys,
             freeBits: freeBits, dataModules: order.length };
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

  function hash01(i) {            // deterministic jitter in [0, 1) — stable output, no flicker
    var h = Math.imul(i ^ 0x9E3779B9, 0x85EBCA6B);
    h ^= h >>> 13; h = Math.imul(h, 0xC2B2AE35); h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  }

  // target/weight: display orientation, length size². Tries the given masks × rotations,
  // keeps the one whose finished modules agree best (weighted) with the drawing.
  function makeArt(bytes, opts) {
    var ver = opts.version, ecl = opts.ecl == null ? 0 : opts.ecl;
    var sys = opts.system && opts.system.ver === ver && opts.system.ecl === ecl ? opts.system : artSystem(bytes, ver, ecl);
    if (!sys) return null;
    var t = sys.t, L = sys.L, size = t.size, n = size * size, order = t.order, nbits = L.raw * 8;
    var target = opts.target, weight = opts.weight;
    var masks = opts.masks || [0, 1, 2, 3, 4, 5, 6, 7], rots = opts.rotations || [0, 1, 2, 3];
    var best = null;

    rots.forEach(function (r) {
      var tq = new Uint8Array(n), wq = new Float32Array(n);
      for (var y = 0; y < size; y++) for (var x = 0; x < size; x++) {
        var di = rotIndex(r, size, x, y), qi = y * size + x;
        tq[qi] = target[di]; wq[qi] = weight[di] + 1e-4 * hash01(di);
      }
      var orders = sys.blockSys.map(function (bs) {
        var idx = []; for (var i = 0; i < bs.mods.length; i++) idx.push(i);
        idx.sort(function (a, b) { return wq[order[bs.mods[b]]] - wq[order[bs.mods[a]]]; });
        return idx;
      });
      masks.forEach(function (m) {
        var mk = t.masks[m];
        var dataBits = sys.fixedBits.slice();
        for (var g = dataBits.length; g < L.dataCw * 8; g++) dataBits.push(0);
        sys.blockSys.forEach(function (bs, bi) {
          if (!bs.k) return;
          var rhs = new Uint8Array(bs.mods.length);
          for (var i = 0; i < bs.mods.length; i++) {
            var kk = bs.mods[i];
            rhs[i] = tq[order[kk]] ^ mk[kk] ^ bs.consts[i];
          }
          var f = solveBlock(bs, orders[bi], rhs);
          for (var j = 0; j < bs.k; j++) dataBits[bs.lo + j] = f[j];
        });
        var dataCw = bitsToBytes(dataBits, L.dataCw);
        var rem = new Uint8Array(order.length - nbits);
        for (var q = nbits; q < order.length; q++) rem[q - nbits] = tq[order[q]];
        var mod = buildMatrix(ver, ecl, m, dataCw, rem);
        var score = 0;
        for (var i2 = 0; i2 < n; i2++) if (mod[i2] === tq[i2]) score += wq[i2];
        if (!best || score > best.score) best = { score: score, mod: mod, mask: m, rot: r, tq: tq };
      });
    });

    var fn = t.fn, agree = 0, dataN = 0;
    for (var i3 = 0; i3 < n; i3++) if (!fn[i3]) { dataN++; if (best.mod[i3] === best.tq[i3]) agree++; }
    return {
      size: size, version: ver, ecl: ecl, mask: best.mask, rotation: best.rot, art: true,
      modules: rotateArray(best.mod, size, best.rot),
      isFunction: rotateArray(fn, size, best.rot),
      match: agree / dataN,                       // share of data modules that match the drawing
      control: sys.freeBits / sys.dataModules,    // share of data modules we could set freely
      score: best.score                           // weighted agreement (compare runs of the same drawing)
    };
  }

  /* ---------- Picking a version for a drawing ---------- */
  // Share of data modules the free bits can steer.
  function freeRatio(bytes, ver, ecl) {
    if (!fits(bytes, ver, ecl)) return -1;
    var free = numDataCodewords(ver, ecl) * 8 - bitsNeeded(bytes, ver) - 4;
    return free < 0 ? -1 : free / numRawDataModules(ver);
  }

  // Prefer the version that maps an M×M drawing 1:1 (size M+1); otherwise the smallest
  // version at least that big whose free share reaches `minRatio`.
  function autoVersion(bytes, M, opts) {
    opts = opts || {};
    var ecl = opts.ecl == null ? 0 : opts.ecl, minRatio = opts.minRatio || 0.4, maxVer = opts.maxVersion || 20;
    var exact = (M + 1 - 17) / 4;
    if (exact === Math.floor(exact) && exact >= 1 && exact <= maxVer && freeRatio(bytes, exact, ecl) >= minRatio) return exact;
    var startVer = Math.max(1, Math.ceil((M + 1 - 17) / 4));
    for (var v = startVer; v <= maxVer; v++) if (freeRatio(bytes, v, ecl) >= minRatio) return v;
    for (v = Math.min(startVer, maxVer); v >= 1; v--) if (freeRatio(bytes, v, ecl) >= minRatio) return v;
    return -1;
  }

  /* ---------- Fitting an M×M drawing onto the N×N module grid ---------- */
  function cubic(x) {             // Keys cubic convolution, a = −0.5 (the usual "bicubic")
    x = Math.abs(x);
    if (x < 1) return (1.5 * x - 2.5) * x * x + 1;
    if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
  }

  // Returns gray levels in [0, 1] (1 = dark) for each module, display orientation.
  //  · N is M or M+1 → placed 1:1 (bicubic at ~1:1 would erase lone dots).
  //  · N > M+1      → bicubic, then every drawn pixel keeps at least its nearest module.
  //  · N < M        → area average.
  function fitDrawing(bits, M, N) {
    var out = new Float32Array(N * N), x, y, i;
    if (N === M || N === M + 1) {
      for (y = 0; y < M; y++) for (x = 0; x < M; x++) out[y * N + x] = bits[y * M + x];
      return out;
    }
    if (N > M) {
      var s = M / N, taps = [];
      for (i = 0; i < N; i++) {
        var c = (i + 0.5) * s - 0.5, base = Math.floor(c) - 1, w = [], idx = [];
        for (var k = 0; k < 4; k++) { idx.push(Math.min(M - 1, Math.max(0, base + k))); w.push(cubic(c - (base + k))); }
        taps.push({ idx: idx, w: w });
      }
      var tmp = new Float32Array(M * N);
      for (y = 0; y < M; y++) for (x = 0; x < N; x++) {
        var tx = taps[x], acc = 0;
        for (k = 0; k < 4; k++) acc += tx.w[k] * bits[y * M + tx.idx[k]];
        tmp[y * N + x] = acc;
      }
      for (y = 0; y < N; y++) for (x = 0; x < N; x++) {
        var ty = taps[y], a2 = 0;
        for (k = 0; k < 4; k++) a2 += ty.w[k] * tmp[ty.idx[k] * N + x];
        out[y * N + x] = Math.min(1, Math.max(0, a2));
      }
      for (y = 0; y < M; y++) for (x = 0; x < M; x++) {                 // keep lone dots / thin gaps
        var nx = Math.round((x + 0.5) / s - 0.5), ny = Math.round((y + 0.5) / s - 0.5), o = ny * N + nx;
        if (bits[y * M + x] && out[o] < 0.5) out[o] = 0.75;
        else if (!bits[y * M + x] && out[o] >= 0.5) out[o] = 0.25;
      }
      return out;
    }
    var r = M / N;                                                        // area average
    for (y = 0; y < N; y++) for (x = 0; x < N; x++) {
      var y0 = y * r, y1 = y0 + r, x0 = x * r, x1 = x0 + r, sum = 0;
      for (var sy = Math.floor(y0); sy < Math.ceil(y1); sy++) {
        var hy = Math.min(y1, sy + 1) - Math.max(y0, sy);
        for (var sx = Math.floor(x0); sx < Math.ceil(x1); sx++) {
          var hx = Math.min(x1, sx + 1) - Math.max(x0, sx);
          if (bits[sy * M + sx]) sum += hx * hy;
        }
      }
      out[y * N + x] = sum / (r * r);
    }
    return out;
  }

  // Which modules matter most: confident ones (gray far from 0.5), and among those the
  // ones next to a colour boundary — they carry the outline, thin lines and lone dots.
  // The figure (whichever colour does not own the border) outranks the background, so the
  // modules nobody can control — and the link's own band — land in the background.
  function weights(gray, N) {
    var n = N * N, t = new Uint8Array(n), d = new Uint8Array(n), w = new Float32Array(n), x, y, i;
    for (i = 0; i < n; i++) t[i] = gray[i] >= 0.5 ? 1 : 0;
    var ring = 0, ringDark = 0;
    for (y = 0; y < N; y++) for (x = 0; x < N; x++) {
      if (Math.min(x, y, N - 1 - x, N - 1 - y) > 1) continue;
      ring++; ringDark += t[y * N + x];
    }
    var figure = ringDark * 3 < ring ? 1 : ringDark * 3 > ring * 2 ? 0 : -1;   // −1: no clear background
    var CAP = 4;
    for (y = 0; y < N; y++) for (x = 0; x < N; x++) {
      var me = t[y * N + x], best = CAP;
      for (var r = 1; r < CAP && best === CAP; r++) {
        for (var dy = -r; dy <= r && best === CAP; dy++) for (var dx = -r; dx <= r; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
          var xx = x + dx, yy = y + dy;
          var other = (xx < 0 || yy < 0 || xx >= N || yy >= N) ? 0 : t[yy * N + xx];   // quiet zone is light
          if (other !== me) { best = r; break; }
        }
      }
      d[y * N + x] = best;
    }
    var c = (N - 1) / 2;
    for (y = 0; y < N; y++) for (x = 0; x < N; x++) {
      i = y * N + x;
      var conf = Math.min(1, Math.abs(gray[i] * 2 - 1) * 1.5 + 0.05);
      var rr = Math.sqrt((x - c) * (x - c) + (y - c) * (y - c)) / (c * Math.SQRT2);
      w[i] = conf * (0.6 + 1 / d[i]) * (1.15 - 0.3 * rr) * (t[i] === figure ? 1.3 : 1);
    }
    return w;
  }

  /* ---------- Rendering ---------- */
  // Calls fill(x, y, w, h) for every dark rectangle, in pixel units, quiet zone included.
  // A module whose value differs from the drawing is painted in the drawing's colour with a
  // centre square of side `centre` (0–1 of a module) in its true colour. centre = 1 → plain QR.
  function renderModules(res, target, opts, fill) {
    var size = res.size, px = opts.px, quiet = opts.quiet == null ? 4 : opts.quiet;
    var centre = opts.centre == null ? 1 : opts.centre;
    var cpx = Math.max(1, Math.min(px, Math.round(px * centre))), off = Math.floor((px - cpx) / 2);
    for (var y = 0; y < size; y++) {
      var runStart = -1;
      for (var x = 0; x <= size; x++) {
        var i = y * size + x, full = false;
        if (x < size) {
          var v = res.modules[i];
          var outer = (target && !res.isFunction[i]) ? target[i] : v;
          if (outer === v) full = v === 1;
          else {
            if (outer === 1) {            // dark square with a light centre: paint the ring as 4 bars
              var X = (x + quiet) * px, Y = (y + quiet) * px;
              fill(X, Y, px, off);
              fill(X, Y + off + cpx, px, px - off - cpx);
              fill(X, Y + off, off, cpx);
              fill(X + off + cpx, Y + off, px - off - cpx, cpx);
            } else fill((x + quiet) * px + off, (y + quiet) * px + off, cpx, cpx);
          }
        }
        if (full && runStart < 0) runStart = x;
        if (!full && runStart >= 0) { fill((runStart + quiet) * px, (y + quiet) * px, (x - runStart) * px, px); runStart = -1; }
      }
    }
    return (size + quiet * 2) * px;
  }

  /* ---------- Self-check: read the finished symbol back ---------- */
  function verify(res, expectedBytes) {
    var size = res.size, ver = (size - 17) / 4;
    var mod = unrotateArray(res.modules, size, res.rotation || 0);
    var pos = formatPositions(size), read = [0, 0];
    pos.forEach(function (copy, ci) { copy.forEach(function (p, i) { read[ci] |= mod[p[1] * size + p[0]] << i; }); });
    if (read[0] !== read[1]) return { ok: false, why: 'format copies differ' };
    var ecl = -1, mask = -1;
    for (var e = 0; e < 4; e++) for (var m = 0; m < 8; m++) if (formatBits(e, m) === read[0]) { ecl = e; mask = m; }
    if (ecl < 0) return { ok: false, why: 'format bits invalid' };
    var t = template(ver), L = layout(ver, ecl), order = t.order, mk = t.masks[mask];
    for (var i = 0; i < size * size; i++) if (t.fn[i] && t.mod[i] !== mod[i] && !isFormatModule(i, size)) return { ok: false, why: 'function pattern damaged' };
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
    makePlain: makePlain, makeArt: makeArt, artSystem: artSystem,
    autoVersion: autoVersion, freeRatio: freeRatio,
    // Function-pattern map (finders, timing, alignment, format/version info) in display orientation.
    functionMap: function (ver, rot) { var t = template(ver); return rotateArray(t.fn, t.size, rot || 0); },
    fitDrawing: fitDrawing, weights: weights, renderModules: renderModules, verify: verify,
    // exposed for tests
    _rsEncode: rsEncode, _numDataCodewords: numDataCodewords, _numRawDataModules: numRawDataModules,
    _alignmentPositions: alignmentPositions, _template: template, _layout: layout, _formatBits: formatBits
  };
  if (typeof module === 'object' && module.exports) module.exports = API;
  else root.DotQR = API;
})(this);
