/* Dot QR — photo → black-and-white dots.
   Pure functions, no DOM: the page loads it as window.DotPhoto, check.js require()s it.

   Every dot comes out as exactly one of two values: 1 = black, 0 = white. Grey (0–255) only
   exists while deciding which of the two a dot should be.

   1. measure — for each grid cell of the square crop, two tones:
        mean  the cell's average (area-weighted)            → for Shapes
        dark  a low percentile: "is there ink in this cell?" → for Lines (a thin line survives)
      both over the part of the cell that is on the photo; cov says how much that is. Zoomed out
      past the photo's edge, the rest reads as plain paper, so the edge itself never draws a line.
   2. lighting — phone photos of paper are never evenly lit. A smooth surface is fitted to the
      cells that aren't much darker than it, and every tone is read relative to it (1 = paper).
   3. decide:
        lines  — black where a cell is clearly darker than its surroundings, or much darker than
                 the paper. "Clearly" is measured against the picture's own grain (the flattest
                 blocks), so blank paper stays blank. The default: drawings, logos, faces.
        shapes — one cut-off for the whole picture (Otsu's method) on the evened-out tones; if
                 there is no real dark group (blank paper), nothing is black. Bold silhouettes.
      ink runs from −1 (fewer black dots) to 1 (more); 0 is the automatic setting.
   4. tidy — a dot with no same-coloured neighbour flips (lone specks go, pinholes fill); invert.
*/
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.DotPhoto = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* RGBA bytes (canvas ImageData) → luminance 0–255; transparent pixels count as white paper. */
  function grayFromRGBA(data, w, h) {
    var g = new Uint8ClampedArray(w * h);
    for (var i = 0, p = 0; i < w * h; i++, p += 4) {
      var a = data[p + 3] / 255;
      g[i] = (0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]) * a + 255 * (1 - a);
    }
    return g;
  }

  /* ---------- 1. measure ---------- */

  /* For each of n cells along one axis: [pixel, overlap, pixel, overlap, …] over [a, a + step). */
  function spans(start, step, n) {
    var out = [];
    for (var i = 0; i < n; i++) {
      var a = start + i * step, b = a + step, s = [];
      for (var p = Math.floor(a); p < b; p++) {
        var ov = Math.min(b, p + 1) - Math.max(a, p);
        if (ov > 1e-9) s.push(p, ov);
      }
      out.push(s);
    }
    return out;
  }

  /* Average of each cell over its on-photo part (each photo pixel counts by how much of it falls in
     the cell), and the share of the cell that is on the photo. A cell entirely off the photo is white. */
  function meanTones(gray, w, h, crop, n) {
    var step = crop.size / n, xs = spans(crop.x, step, n), ys = spans(crop.y, step, n);
    var mean = new Float32Array(n * n), cov = new Float32Array(n * n);
    for (var cy = 0; cy < n; cy++) {
      var ry = ys[cy];
      for (var cx = 0; cx < n; cx++) {
        var rx = xs[cx], sum = 0, area = 0, on = 0;
        for (var j = 0; j < ry.length; j += 2) {
          var py = ry[j], wy = ry[j + 1], row = py * w, inY = py >= 0 && py < h;
          for (var k = 0; k < rx.length; k += 2) {
            var px = rx[k], wt = wy * rx[k + 1];
            if (inY && px >= 0 && px < w) { sum += wt * gray[row + px]; on += wt; }
            area += wt;
          }
        }
        mean[cy * n + cx] = on > 0 ? sum / on : 255;
        cov[cy * n + cx] = on / area;
      }
    }
    return { mean: mean, cov: cov };
  }

  /* Low percentile of each cell: the tone of its darkest pixel-wide strip, so a thin line crossing
     a cell still shows. Small cells (zoomed far in) just use the mean. */
  var histBuf = null;
  function darkTones(gray, w, h, crop, n, mean) {
    var step = crop.size / n;
    if (step < 3) return mean.slice();
    var q = Math.min(0.12, Math.max(0.01, 1 / step)), N = n * n;
    if (!histBuf || histBuf.length < N * 256) histBuf = new Uint32Array(N * 256);
    var hist = histBuf, cnt = new Uint32Array(N), out = new Float32Array(N), i;
    hist.fill(0, 0, N * 256);
    var x0 = Math.max(0, Math.floor(crop.x)), x1 = Math.min(w, Math.ceil(crop.x + crop.size));
    var y0 = Math.max(0, Math.floor(crop.y)), y1 = Math.min(h, Math.ceil(crop.y + crop.size));
    var colCell = new Int32Array(Math.max(0, x1 - x0));
    for (var x = x0; x < x1; x++) { var c = Math.floor((x + 0.5 - crop.x) / step); colCell[x - x0] = c >= 0 && c < n ? c : -1; }
    for (var y = y0; y < y1; y++) {
      var cy = Math.floor((y + 0.5 - crop.y) / step);
      if (cy < 0 || cy >= n) continue;
      for (x = x0; x < x1; x++) {
        var cx = colCell[x - x0];
        if (cx < 0) continue;
        var cell = cy * n + cx;
        hist[cell * 256 + gray[y * w + x]]++; cnt[cell]++;
      }
    }
    for (i = 0; i < N; i++) {
      if (cnt[i] < 4) { out[i] = mean[i]; continue; }
      var need = Math.max(1, Math.round(cnt[i] * q)), acc = 0, v = 0;     // over the on-photo pixels only
      for (; v < 256; v++) { acc += hist[i * 256 + v]; if (acc >= need) break; }
      out[i] = v;
    }
    return out;
  }

  /* Both tones and the photo coverage for every cell of the square crop {x, y, size} (photo pixels). */
  function measure(gray, w, h, crop, n) {
    var mt = meanTones(gray, w, h, crop, n);
    return { n: n, mean: mt.mean, dark: darkTones(gray, w, h, crop, n, mt.mean), cov: mt.cov };
  }

  /* ---------- 2. lighting ---------- */

  function median(a) { var s = Array.prototype.slice.call(a).sort(function (x, y) { return x - y; }); return s.length ? s[s.length >> 1] : 0; }

  /* Smooth paper brightness: a quadratic surface fitted to the cells not much darker than it
     (re-fitted a few times, so dark drawing or shadow doesn't drag it down). Only a picture that is
     mostly paper gets evened out this way: when most cells sit well below the surface, or the surface
     is far darker than the picture's bright end (a photo of a scene, not of a page), one flat level is
     used instead — the picture's bright end. */
  function lighting(t, n, cov) {
    var N = n * n, X = new Float64Array(N), Y = new Float64Array(N), q = new Float64Array(N), i, r, c, it;
    for (i = 0; i < N; i++) { X[i] = (i % n + 0.5) / n - 0.5; Y[i] = (Math.floor(i / n) + 0.5) / n - 0.5; }
    var full = new Uint8Array(N), seen = [];                      // judge the paper from cells wholly on the photo
    for (i = 0; i < N; i++) if (!cov || cov[i] > 0.999) { full[i] = 1; seen.push(t[i]); }
    if (!seen.length) for (i = 0; i < N; i++) { full[i] = 1; seen.push(t[i]); }
    var med = median(seen), use = new Uint8Array(N);
    for (i = 0; i < N; i++) use[i] = full[i] && t[i] >= med ? 1 : 0;
    for (it = 0; it < 5; it++) {
      var A = [];
      for (r = 0; r < 6; r++) A.push(new Float64Array(7));
      for (i = 0; i < N; i++) {
        if (!use[i]) continue;
        var f = [1, X[i], Y[i], X[i] * X[i], X[i] * Y[i], Y[i] * Y[i]];
        for (r = 0; r < 6; r++) { for (c = 0; c < 6; c++) A[r][c] += f[r] * f[c]; A[r][6] += f[r] * t[i]; }
      }
      for (r = 0; r < 6; r++) A[r][r] += 1e-6;
      for (c = 0; c < 6; c++) {                           // Gauss–Jordan with partial pivoting
        var p = c;
        for (r = c + 1; r < 6; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
        var tmp = A[c]; A[c] = A[p]; A[p] = tmp;
        for (r = 0; r < 6; r++) if (r !== c) { var k = A[r][c] / A[c][c]; for (var k2 = c; k2 < 7; k2++) A[r][k2] -= k * A[c][k2]; }
      }
      var co = [];
      for (r = 0; r < 6; r++) co.push(A[r][6] / A[r][r]);
      var res = [];
      for (i = 0; i < N; i++) {
        q[i] = co[0] + co[1] * X[i] + co[2] * Y[i] + co[3] * X[i] * X[i] + co[4] * X[i] * Y[i] + co[5] * Y[i] * Y[i];
        if (use[i]) res.push(t[i] - q[i]);
      }
      var m = median(res), dev = res.map(function (v) { return Math.abs(v - m); });
      var s = Math.max(1, 1.4826 * median(dev));
      for (i = 0; i < N; i++) use[i] = full[i] && t[i] - q[i] > -2.5 * s ? 1 : 0;
    }
    var ratio = [], level = [], out = new Float32Array(N);
    for (i = 0; i < N; i++) if (full[i]) { ratio.push(t[i] / Math.max(1, q[i])); level.push(q[i]); }
    var sorted = seen.sort(function (a, b) { return a - b; }), bright = sorted[Math.floor((sorted.length - 1) * 0.9)];
    // Not a page when most cells sit well below the fitted surface, or when that "paper" is far darker
    // than the picture's bright end (a dark background under a lit face is not paper).
    if (median(ratio) < 0.85 || median(level) < 0.6 * bright) return out.fill(Math.max(16, bright));
    var top = 16;
    for (i = 0; i < N; i++) top = Math.max(top, q[i]);
    for (i = 0; i < N; i++) out[i] = Math.max(0.4 * top, q[i]);   // light varies at most 2.5× across a page
    return out;
  }

  /* Tones relative to the paper: 1 = paper, 0 = black (capped at 1.5 for highlights). The off-photo
     share of a cell counts as paper. */
  function relative(t, n, cov) {
    var bg = lighting(t, n, cov), u = new Float32Array(n * n);
    for (var i = 0; i < n * n; i++) { var c = cov ? cov[i] : 1; u[i] = c * Math.min(1.5, t[i] / bg[i]) + (1 - c); }
    return u;
  }

  /* ---------- 3. decide ---------- */

  /* Gaussian blur on an n×n grid, edges clamped. */
  function blur(t, n, sigma) {
    var r = Math.max(1, Math.ceil(sigma * 3)), k = [], s = 0, i, x, y;
    for (i = -r; i <= r; i++) { var v = Math.exp(-i * i / (2 * sigma * sigma)); k.push(v); s += v; }
    for (i = 0; i < k.length; i++) k[i] /= s;
    var tmp = new Float32Array(n * n), out = new Float32Array(n * n);
    for (y = 0; y < n; y++) for (x = 0; x < n; x++) {
      var a = 0;
      for (i = -r; i <= r; i++) a += t[y * n + Math.min(n - 1, Math.max(0, x + i))] * k[i + r];
      tmp[y * n + x] = a;
    }
    for (y = 0; y < n; y++) for (x = 0; x < n; x++) {
      var b = 0;
      for (i = -r; i <= r; i++) b += tmp[Math.min(n - 1, Math.max(0, y + i)) * n + x] * k[i + r];
      out[y * n + x] = b;
    }
    return out;
  }

  /* The picture's own grain: spread of v in the flattest quarter of blocks (paper, sky, skin…). */
  function grain(v, n) {
    var B = Math.max(2, n >> 3), out = [];
    for (var by = 0; by + B <= n; by += B) for (var bx = 0; bx + B <= n; bx += B) {
      var s = 0, s2 = 0, c = 0;
      for (var y = by; y < by + B; y++) for (var x = bx; x < bx + B; x++) { var a = v[y * n + x]; s += a; s2 += a * a; c++; }
      out.push(Math.sqrt(Math.max(0, s2 / c - (s / c) * (s / c))));
    }
    out.sort(function (a, b) { return a - b; });
    return out[Math.floor(out.length * 0.25)];
  }

  /* Otsu's threshold on 0–255 tones: the cut-off that best separates a dark and a light group. */
  function otsu(t) {
    var hist = new Float64Array(256), i, sum = 0, total = t.length;
    for (i = 0; i < total; i++) hist[Math.max(0, Math.min(255, Math.round(t[i])))]++;
    for (i = 0; i < 256; i++) sum += i * hist[i];
    var sumB = 0, wB = 0, best = -1, cut = 128;
    for (i = 0; i < 256; i++) {
      wB += hist[i];
      if (!wB) continue;
      var wF = total - wB;
      if (!wF) break;
      sumB += i * hist[i];
      var d = sumB / wB - (sum - sumB) / wF, between = wB * wF * d * d;
      if (between > best) { best = between; cut = i + 0.5; }
    }
    return best > 0 ? cut : 128;
  }

  function linesDots(dark, n, ink, cov) {
    var u = relative(dark, n, cov), N = n * n, i, m;
    if (cov) {                                                    // surroundings from the photo only: off-photo
      var uc = new Float32Array(N), c = new Float32Array(N);      // cells would read as bright paper and draw
      for (i = 0; i < N; i++) { uc[i] = u[i] * cov[i]; c[i] = cov[i]; }   // a line along a shadowed edge
      var bu = blur(uc, n, 3 * n / 32), bc = blur(c, n, 3 * n / 32);
      m = new Float32Array(N);
      for (i = 0; i < N; i++) m[i] = bc[i] > 1e-6 ? bu[i] / bc[i] : u[i];
    } else m = blur(u, n, 3 * n / 32);
    var r = new Float32Array(N);
    for (i = 0; i < N; i++) r[i] = m[i] - u[i];                  // how much darker than the surroundings
    var noise = Math.max(0.006, grain(r, n));
    var kappa = ink >= 0 ? 3 - ink : 3 - 2 * ink;                 // × grain: what counts as clearly darker
    var off = Math.max(kappa * noise, ink >= 0 ? 0.07 - 0.04 * ink : 0.07 - 0.12 * ink);
    var solid = 0.45 + 0.25 * ink;                                // this much darker than paper is black anyway
    var bits = new Uint8Array(n * n);
    for (i = 0; i < n * n; i++) bits[i] = r[i] > off || u[i] < solid ? 1 : 0;
    return bits;
  }

  var SCALE = 170;                                                // relative tone 1.5 → 255 for Otsu
  function shapesDots(mean, n, ink, cov) {
    var u = relative(mean, n, cov), s = new Float32Array(n * n), i;
    for (i = 0; i < n * n; i++) s[i] = u[i] * SCALE;
    var cut = otsu(s) / SCALE, a = 0, ca = 0, b = 0, cb = 0;
    for (i = 0; i < n * n; i++) { if (u[i] < cut) { a += u[i]; ca++; } else { b += u[i]; cb++; } }
    var light = cb ? b / cb : 1, darkest = ca ? a / ca : 0;
    if (light - darkest < 0.12) {                                 // no real dark group: one even tone —
      var onPhoto = [];                                           // blank paper or fog stays white, a dark photo goes black
      for (i = 0; i < n * n; i++) if (!cov || cov[i] > 0.5) onPhoto.push(mean[i]);
      cut = median(onPhoto.length ? onPhoto : mean) < 100 ? 2 : 0; darkest = 0;
    }
    cut += 0.8 * ink * (ink >= 0 ? light - cut : cut - darkest);  // ink moves the cut toward either group, never past it
    var bits = new Uint8Array(n * n);
    for (i = 0; i < n * n; i++) bits[i] = u[i] < cut ? 1 : 0;
    return bits;
  }

  /* ---------- 4. tidy ---------- */

  /* A dot with no same-coloured neighbour, diagonals included, flips: lone specks go, pinholes fill.
     (Diagonal neighbours count, so a thin diagonal line — a staircase of corner-touching dots — stays.)
     Corner dots are left alone. */
  function despeckle(bits, n) {
    var out = bits.slice();
    for (var y = 0; y < n; y++) for (var x = 0; x < n; x++) {
      var v = bits[y * n + x], same = 0, cnt = 0;
      for (var dy = -1; dy <= 1; dy++) for (var dx = -1; dx <= 1; dx++) {
        var xx = x + dx, yy = y + dy;
        if ((dx || dy) && xx >= 0 && yy >= 0 && xx < n && yy < n) { cnt++; if (bits[yy * n + xx] === v) same++; }
      }
      if (same === 0 && cnt >= 5) out[y * n + x] = v ^ 1;
    }
    return out;
  }

  /* measure() result → n×n dots (Uint8Array, 1 = black, 0 = white).
     opts: style 'lines' (default) | 'shapes', ink −1 … 1 (0 = automatic), invert. */
  function toDots(m, opts) {
    opts = opts || {};
    var n = m.n, ink = Math.max(-1, Math.min(1, +opts.ink || 0));
    var bits = despeckle(opts.style === 'shapes' ? shapesDots(m.mean, n, ink, m.cov) : linesDots(m.dark, n, ink, m.cov), n);
    if (opts.invert) for (var i = 0; i < n * n; i++) bits[i] ^= 1;
    return bits;
  }

  return {
    STYLES: ['lines', 'shapes'],
    grayFromRGBA: grayFromRGBA, measure: measure, toDots: toDots,
    _lighting: lighting, _otsu: otsu, _despeckle: despeckle
  };
}));
