/**
 * src/render/draw.js — Canvas 2D 렌더 (브라우저 전용)
 *
 * 정본 v1.4 구현 절:
 *   §1.1   논리 1280×720 · 아레나 {350,0,580,720} · 좌우 패널 350
 *   §1.2   아레나 오버레이 띠 (내용은 hud.js 가 그린다 — 레이어 1에서 호출)
 *   §7.1   불변식 4개 (I-1 테두리 = 적 전용 · I-2 색은 거짓말하지 않는다 · I-4 적 탄 불가림)
 *   §7.2   팔레트 = rules.palette 가 유일한 거처. ★ 이 파일에 색 리터럴이 없다
 *   §7.3   cbMode off / cvd / mono
 *   §7.4   적 탄 = 자홍 + 흰 스페큘러 코어 + 검은 하드 외곽선 2px + 불투명 + 레이어 9
 *   §7.5   기체 부착 3중 — 히트박스 코어 링(10) · 슬롯 스트립(6) · 상태 배지(6) · 림 오라(6)
 *   §7.6   적 = 중립 차콜 본체 + 속성색 외곽선 + 코어 글리프 + 림 라이트
 *   §7.8   픽업 = 무광·납작·외곽선 0·글로우 0 · ≤6px · 레이어 2
 *   §7.12.8 플레이어 탄의 밝은 코어 — coreRadiusRatio 0.45 · coreLightnessAdd 25 (L*)
 *   §9.10  절차적 도형 — 스프라이트 0바이트
 *   §10.1  렌더 보간은 **위치 lerp만**. 로직 금지
 *   §12.3  레이어 스택 (하 → 상)
 *
 * ★ core 는 이 파일을 모른다 (단방향). 이 파일은 world 를 **읽기만** 한다 (§9.1).
 */

import { drawArenaBands } from './hud.js';

// ---------------------------------------------------------------------------
// 색 — sRGB ↔ CIE Lab. §7.12.8 의 「L*+25」와 §7.3 의 「채도 0」이 실제 수를 요구한다
// ---------------------------------------------------------------------------
function hexToRgb(hex) {
  const h = hex.charCodeAt(0) === 35 ? hex.slice(1) : hex;
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function clamp255(v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

function rgbToHex(r, g, b) {
  const n = (clamp255(Math.round(r)) << 16) | (clamp255(Math.round(g)) << 8) | clamp255(Math.round(b));
  const s = n.toString(16);
  return `#${'000000'.slice(s.length)}${s}`;
}

function srgbToLinear(c) {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

function linearToSrgb(v) {
  const c = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
  return c * 255;
}

const WHITE_X = 0.95047;
const WHITE_Z = 1.08883;
const LAB_EPS = 216 / 24389;
const LAB_KAPPA = 24389 / 27;

function labF(t) { return t > LAB_EPS ? Math.cbrt(t) : (LAB_KAPPA * t + 16) / 116; }
function labFInv(t) { const t3 = t * t * t; return t3 > LAB_EPS ? t3 : (116 * t - 16) / LAB_KAPPA; }

function hexToLab(hex) {
  const [r8, g8, b8] = hexToRgb(hex);
  const r = srgbToLinear(r8);
  const g = srgbToLinear(g8);
  const b = srgbToLinear(b8);
  const X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / WHITE_X;
  const Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
  const Z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / WHITE_Z;
  const fx = labF(X);
  const fy = labF(Y);
  const fz = labF(Z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function labToHex(L, a, bb) {
  const fy = (L + 16) / 116;
  const fx = fy + a / 500;
  const fz = fy - bb / 200;
  const X = labFInv(fx) * WHITE_X;
  const Y = labFInv(fy);
  const Z = labFInv(fz) * WHITE_Z;
  const r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  const g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  const b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  return rgbToHex(linearToSrgb(r), linearToSrgb(g), linearToSrgb(b));
}

/** §7.12.8 — 같은 hue 의 밝은 코어. L* 만 올린다 (a,b 불변 = hue·채도 보존) */
function lighten(hex, dL) {
  const [L, a, b] = hexToLab(hex);
  return labToHex(L + dL, a, b);
}

/** §7.3 mono — 「채도를 0으로」. Lab 의 a,b 를 0으로 두면 L* 사다리가 그대로 남는다 */
function desaturate(hex) {
  const [L] = hexToLab(hex);
  return labToHex(L, 0, 0);
}

function rgba(hex, alpha) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ---------------------------------------------------------------------------
// 팔레트 해석 (§7.2 · §7.3)
// ---------------------------------------------------------------------------
/**
 * §7.3 — cbMode ∈ off / cvd / mono.
 *   off  : palette.element
 *   cvd  : palette.elementCvd + 글리프 ×1.5 + 적 외곽선 3px + 배경 명도 상한 0.22 + 라벨 강제
 *   mono : cvd 기저 + **palette 전체** 무채화 (element·threat·status·pickup·hud·enemyBody·… 전부)
 *
 * ★ 공장 기본값은 rules.visual.a11y.cbMode 다. 사용자 오버라이드(`opts.cbMode ?? …`, §9.4.3 · §14)는
 *   OPTIONS 화면(§14)의 소관이며 1주차 범위 밖이다 — 여기서는 공장 기본값만 읽는다.
 */
export function resolvePalette(rules) {
  const p = rules.palette;
  const mode = rules.visual.a11y.cbMode;
  if (mode !== 'off' && mode !== 'cvd' && mode !== 'mono') {
    throw new Error(`draw: 미지의 cbMode "${mode}" — 어휘 = off / cvd / mono (§7.3)`);
  }
  const cvd = mode !== 'off';
  const mono = mode === 'mono';
  const f = mono ? desaturate : (c) => c;

  const element = {};
  const src = cvd ? p.elementCvd : p.element;
  const keys = Object.keys(src);
  for (let i = 0; i < keys.length; i += 1) element[keys[i]] = f(src[keys[i]]);

  return {
    mode,
    cvd,
    mono,
    element,
    /** §7.12.8 — 탄 코어. 비율(0.45)이라 H3 의 projRadius 클램프를 자동으로 상속한다 */
    elementCore: (() => {
      const out = {};
      const add = rules.visual.playerBullet.coreLightnessAdd;
      for (let i = 0; i < keys.length; i += 1) out[keys[i]] = f(lighten(src[keys[i]], add));
      return out;
    })(),
    threat: {
      enemyBullet: f(p.threat.enemyBullet),
      telegraph: f(p.threat.telegraph),
      bulletCore: f(p.threat.bulletCore),
      outline: f(p.threat.outline),
    },
    status: { band: f(p.status.band) },
    pickup: { coin: f(p.pickup.coin), xp: f(p.pickup.xp) },
    enemyBody: f(p.enemyBody),
    partDestroyed: f(p.partDestroyed),
    neutralGray: f(p.neutralGray),
    hud: {
      panelBg: f(p.hud.panelBg), panelRule: f(p.hud.panelRule),
      textPrimary: f(p.hud.textPrimary), textDim: f(p.hud.textDim), hpFill: f(p.hud.hpFill),
    },
    bg: p.bg,
    /** §7.3 — 배경 명도 상한은 cvd/mono 에서 0.22 로 내려간다 */
    bgMaxLightness: cvd ? p.bg.cvdMaxLightness : p.bg.maxLightness,
    /** §7.3 — 글리프 크기 ×1.0 / ×1.5 */
    glyphScale: cvd ? 1.5 : 1.0,
    /** §7.6 · §7.3 — 적 외곽선 2px / 3px */
    enemyOutlinePx: cvd ? 3 : 2,
  };
}

// ---------------------------------------------------------------------------
// 속성 글리프 (§7.2 — ● 원 / ▲ 삼각(위) / ◆ 마름모 / ✚ 십자(사엽))
//   ★ 이 4실루엣이 색맹·mono 에서 속성의 **유일한** 채널이다 (§7.3). 형태를 바꾸면 그 보증이 깨진다.
// ---------------------------------------------------------------------------
export function glyphPath(ctx, element, x, y, r) {
  ctx.beginPath();
  if (element === 'normal') {
    ctx.arc(x, y, r, 0, Math.PI * 2);
    return;
  }
  if (element === 'fire') {                       // ▲ 정삼각(위)
    const h = r * 1.15;
    ctx.moveTo(x, y - h);
    ctx.lineTo(x + h * 0.866, y + h * 0.5);
    ctx.lineTo(x - h * 0.866, y + h * 0.5);
    ctx.closePath();
    return;
  }
  if (element === 'water') {                      // ◆ 마름모
    const h = r * 1.3;
    ctx.moveTo(x, y - h);
    ctx.lineTo(x + h * 0.72, y);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x - h * 0.72, y);
    ctx.closePath();
    return;
  }
  if (element === 'grass') {                      // ✚ 십자(사엽)
    const a = r * 1.15;
    const b = a * 0.36;
    ctx.moveTo(x - b, y - a); ctx.lineTo(x + b, y - a); ctx.lineTo(x + b, y - b);
    ctx.lineTo(x + a, y - b); ctx.lineTo(x + a, y + b); ctx.lineTo(x + b, y + b);
    ctx.lineTo(x + b, y + a); ctx.lineTo(x - b, y + a); ctx.lineTo(x - b, y + b);
    ctx.lineTo(x - a, y + b); ctx.lineTo(x - a, y - b); ctx.lineTo(x - b, y - b);
    ctx.closePath();
    return;
  }
  throw new Error(`draw: 글리프가 없는 속성 "${element}" (§7.2 — 어휘 4종)`);
}

// ---------------------------------------------------------------------------
// 적 본체 도형 (§9.10 — shapeId 12종 동결)
//   ★ 정본은 **어휘 12개를 동결**하고 각 도형의 **기하를 인쇄하지 않았다** → 보고 대상.
//     여기서는 "반경 r 안에 들어오는 실루엣"으로만 해석했다. 기하가 확정되면 이 함수만 바뀐다.
// ---------------------------------------------------------------------------
function poly(ctx, x, y, r, pts) {
  ctx.beginPath();
  for (let i = 0; i < pts.length; i += 2) {
    const px = x + pts[i] * r;
    const py = y + pts[i + 1] * r;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

function regular(ctx, x, y, r, n, rot) {
  ctx.beginPath();
  for (let i = 0; i < n; i += 1) {
    const a = rot + (i * Math.PI * 2) / n;
    const px = x + Math.cos(a) * r;
    const py = y + Math.sin(a) * r;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

function shapePath(ctx, shapeId, x, y, r) {
  switch (shapeId) {
    case 'wedge':  return poly(ctx, x, y, r, [0, 1, -0.9, -0.7, 0, -0.35, 0.9, -0.7]);
    case 'delta':  return poly(ctx, x, y, r, [0, 1, -0.85, -0.6, 0.85, -0.6]);
    case 'hexPod': return regular(ctx, x, y, r, 6, Math.PI / 6);
    case 'orb':    { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); return undefined; }
    case 'cross':  return poly(ctx, x, y, r, [-0.35, -1, 0.35, -1, 0.35, -0.35, 1, -0.35,
      1, 0.35, 0.35, 0.35, 0.35, 1, -0.35, 1, -0.35, 0.35, -1, 0.35, -1, -0.35, -0.35, -0.35]);
    case 'spike':  return poly(ctx, x, y, r, [0, 1, -0.45, 0, -0.2, 0, -0.2, -1, 0.2, -1, 0.2, 0, 0.45, 0]);
    case 'ring':   { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.arc(x, y, r * 0.55, 0, Math.PI * 2, true); return undefined; }
    case 'slab':   return poly(ctx, x, y, r, [-1, -0.55, 1, -0.55, 1, 0.55, -1, 0.55]);
    case 'fin':    return poly(ctx, x, y, r, [0, 1, -0.3, -1, 0.95, -0.15]);
    case 'claw':   return poly(ctx, x, y, r, [-0.9, -0.8, -0.25, 0.2, 0, 1, 0.25, 0.2, 0.9, -0.8, 0, -0.15]);
    case 'dart':   return poly(ctx, x, y, r, [0, 1, -0.5, -0.9, 0, -0.5, 0.5, -0.9]);
    case 'bulb':   { ctx.beginPath(); ctx.ellipse(x, y, r * 0.78, r, 0, 0, Math.PI * 2); return undefined; }
    default:
      throw new Error(`draw: 어휘 밖의 shapeId "${shapeId}" (§9.10 — 12종 동결)`);
  }
}

// ---------------------------------------------------------------------------
// 보간 (§10.1 — 위치 lerp만. 렌더가 소유하는 상태이며 world 를 건드리지 않는다)
// ---------------------------------------------------------------------------
function makeTrack(size) {
  return { x: new Float64Array(size), y: new Float64Array(size), gen: new Int32Array(size).fill(-1) };
}

/**
 * ★ world 는 이전 위치를 모른다 (core 는 렌더를 모른다). 그러므로 보간에 필요한
 *   직전 틱 스냅샷은 **렌더가 자기 것으로** 들고 있는다. gen 이 다르면 그 슬롯은 재사용된
 *   다른 개체이므로 보간하지 않는다 (스폰 순간 화면을 가로지르는 유령 방지).
 */
export function makeInterp(world) {
  const caps = world.data.rules.caps;
  return {
    enabled: world.data.rules.loop.interpolate,
    tick: -1,
    player: { x: world.player.x, y: world.player.y },
    enemies: makeTrack(caps.enemies),
    playerBullets: makeTrack(caps.playerBullets),
    enemyBullets: makeTrack(caps.enemyBullets),
    pickups: makeTrack(caps.pickups),
    drones: makeTrack(caps.drones),
  };
}

function captureTrack(track, pool) {
  const items = pool.items;
  for (let i = 0; i < items.length; i += 1) {
    const e = items[i];
    if (!e.alive) { track.gen[i] = -1; continue; }
    track.x[i] = e.x;
    track.y[i] = e.y;
    track.gen[i] = e.gen;
  }
}

/** ★ main.js 가 step() **직전**에 부른다 → track 은 언제나 "직전 틱의 위치"다 */
export function captureInterp(interp, world) {
  interp.player.x = world.player.x;
  interp.player.y = world.player.y;
  captureTrack(interp.enemies, world.enemies);
  captureTrack(interp.playerBullets, world.playerBullets);
  captureTrack(interp.enemyBullets, world.enemyBullets);
  captureTrack(interp.pickups, world.pickups);
  captureTrack(interp.drones, world.drones);
  interp.tick = world.tick;
}

function lerpX(interp, track, e, alpha) {
  if (!interp.enabled || track.gen[e.idx] !== e.gen) return e.x;
  return track.x[e.idx] + (e.x - track.x[e.idx]) * alpha;
}

function lerpY(interp, track, e, alpha) {
  if (!interp.enabled || track.gen[e.idx] !== e.gen) return e.y;
  return track.y[e.idx] + (e.y - track.y[e.idx]) * alpha;
}

// ---------------------------------------------------------------------------
// 렌더 상태 (순수 장식 — 결정성 무관, core 바깥, §10.1)
// ---------------------------------------------------------------------------
/** §7.7 — 히트 피드백 파티클 풀 (render 전용, 결정성 무관). 초과 = evictOldest(초과 정책 = particle) */
function makeHitPool(cap) {
  const buf = new Array(cap);
  for (let i = 0; i < cap; i += 1) {
    buf[i] = { active: false, tier: 'neutral', x: 0, y: 0, element: 'normal', age: 0, life: 0, killed: false, seed: 0 };
  }
  return { buf, cap, head: 0 };   // head = 다음 쓸 슬롯 → 링 덮어쓰기가 곧 evictOldest
}

/** §7.7 — tier 랭크(maxOnly 판정용). super > neutral > resist */
function tierRank(t) { return t === 'super' ? 2 : t === 'neutral' ? 1 : 0; }

export function makeFx(world) {
  const capE = world.data.rules.caps.enemies;
  return {
    stancePrev: world.player.stance,
    ringT: -1,          // §7.5 전환 링. 음수 = 비활성
    ringElement: 'normal',
    ringInvested: false,
    bgScroll: 0,
    // §7.7 — 히트 피드백 3중 감각. core 의 hitFx 링을 소진해 여기로 옮긴다(updateFx).
    hits: makeHitPool(world.data.rules.caps.particles),
    hitSeq: 0,                                    // 스파크 각도 분산용 결정적 카운터(무-RNG)
    // 개체(idx)별 상태 — gen 으로 풀 재사용 슬롯을 구분한다(다른 개체엔 안 샌다)
    freezeT: new Float64Array(capE),              // §7.7 ×2 임팩트 프리즈 잔여(게임초)
    freezeGen: new Int32Array(capE).fill(-1),
    markerT: new Float64Array(capE),              // §7.7 markerCooldownSecPerEntity 잔여
    markerGen: new Int32Array(capE).fill(-1),
    markerRank: new Int8Array(capE),              // 이 창에서 이미 보인 최고 tier(maxOnly)
  };
}

/**
 * ★ 렌더 전용 장식 갱신. **실제 경과 게임초**를 받는다 (배속을 그대로 탄다 = §6.2).
 *   판정에 아무 영향이 없으므로 결정성과 무관하다.
 */
export function updateFx(fx, world, dtGame) {
  const p = world.player;
  if (p.stance !== fx.stancePrev) {
    fx.stancePrev = p.stance;
    fx.ringT = 0;
    fx.ringElement = p.stance;
    // §7.5 — 투자 0인 속성으로의 전환은 **탈색 링**. 노말(Q)은 은색 정상 확장
    fx.ringInvested = p.stance === 'normal' ? true : p.invest[p.stance] > 0;
  }
  if (fx.ringT >= 0) {
    fx.ringT += dtGame;
    if (fx.ringT > world.data.rules.visual.stance.ringExpandSec) fx.ringT = -1;
  }
  const bg = world.data.rules.palette.bg;
  fx.bgScroll = (fx.bgScroll + bg.maxScrollSpeed * dtGame) % 4096;

  updateHitFx(fx, world, dtGame);
}

/**
 * §7.7 — core 의 히트 이벤트(이번 스텝분)를 소진해 3중 감각 파티클로 옮긴다.
 *   ★ main.js 가 **매 스텝 뒤** 부르므로 world.hitFx 는 언제나 「그 스텝의 히트」다(다음 step 진입 시 리셋).
 *   ★ 결정성 무관 — 여기 있는 어떤 값도 core 로 되돌아가지 않는다(순수 장식).
 *   dedup: markerPolicy "maxOnly" + markerCooldownSecPerEntity — 한 개체에 창(0.25s) 안에서는
 *     **더 높은 tier 만** 새로 그린다(4무기 동시타격의 버스트 스팸을 막고 「최고 배율 1개」를 남긴다).
 *     ★ 처치(killed)는 1회성이라 dedup 을 우회해 항상 처치 FX 를 보인다.
 */
function updateHitFx(fx, world, dtGame) {
  const vh = world.data.rules.visual.hitFx;
  // 개체별 타이머 감쇠(프리즈·마커 쿨다운) — 시간만 줄인다. gen 은 스폰 때만 쓴다
  for (let i = 0; i < fx.freezeT.length; i += 1) {
    if (fx.freezeT[i] > 0) { fx.freezeT[i] -= dtGame; if (fx.freezeT[i] < 0) fx.freezeT[i] = 0; }
    if (fx.markerT[i] > 0) { fx.markerT[i] -= dtGame; if (fx.markerT[i] < 0) fx.markerT[i] = 0; }
  }
  // 파티클 수명
  const hb = fx.hits.buf;
  for (let i = 0; i < hb.length; i += 1) {
    const h = hb[i];
    if (h.active) { h.age += dtGame; if (h.age >= h.life) h.active = false; }
  }
  // 이번 스텝의 히트 이벤트 소진
  const src = world.hitFx;
  const cd = vh.markerCooldownSecPerEntity;
  for (let i = 0; i < src.count; i += 1) {
    const ev = src.buf[i];
    const idx = ev.enemyIdx;
    const rank = tierRank(ev.tier);
    const tracked = idx >= 0 && idx < fx.markerT.length;
    // maxOnly + 쿨다운: 창 안에서 같거나 낮은 tier 는 억제, 높은 tier 는 교체(처치는 우회)
    if (!ev.killed && tracked
        && fx.markerGen[idx] === ev.enemyGen && fx.markerT[idx] > 0 && rank <= fx.markerRank[idx]) {
      continue;
    }
    if (tracked) { fx.markerGen[idx] = ev.enemyGen; fx.markerT[idx] = cd; fx.markerRank[idx] = rank; }
    // ×2 임팩트 프리즈 — 살아있는 개체(idx,gen)에 묶는다. 죽은 적은 묶을 대상이 없다(처치 FX 로 간다)
    if (ev.tier === 'super' && !ev.killed && idx >= 0 && idx < fx.freezeT.length) {
      fx.freezeT[idx] = vh.superFreezeSec; fx.freezeGen[idx] = ev.enemyGen;
    }
    spawnHit(fx, vh, ev);
  }
}

function spawnHit(fx, vh, ev) {
  const pool = fx.hits;
  const h = pool.buf[pool.head];
  pool.head = (pool.head + 1) % pool.cap;         // evictOldest = 링 덮어쓰기
  h.active = true;
  h.tier = ev.tier;
  h.x = ev.x; h.y = ev.y;
  h.element = ev.element;
  h.killed = ev.killed;
  h.age = 0;
  // 수명 = resistArcLifeSec(0.25) 를 세 tier 공통 페이드로 재사용한다 — 「얼마나」는 render 결정(새 키 0).
  h.life = vh.resistArcLifeSec;
  h.seed = fx.hitSeq; fx.hitSeq += 1;
}

// ---------------------------------------------------------------------------
// 레이어 0 — 배경 (§7.9 · §12.3: 채도 ≤0.25 / 명도 ≤ bgMaxLightness, **소프트 엣지만**)
//   ★ 테마 hue 는 stages.json 의 소관이고 1주차에는 스테이지가 없다 → 무채색 차콜 (보고 대상)
// ---------------------------------------------------------------------------
function drawBackground(ctx, world, pal, fx) {
  const v = world.data.rules.view;
  const a = v.arena;
  const base = labToHex(pal.bgMaxLightness * 100 * 0.45, 0, 0);
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, v.logicalW, v.logicalH);

  ctx.save();
  ctx.beginPath();
  ctx.rect(a.x, a.y, a.w, a.h);
  ctx.clip();

  // §7.9 — parallaxLayers 2, maxScrollSpeed 90. 소프트 엣지만 (하드 엣지 = 게임플레이 전용)
  const layers = world.data.rules.palette.bg.parallaxLayers;
  for (let L = 0; L < layers; L += 1) {
    const depth = (L + 1) / layers;
    const step = 64 + L * 40;
    const speed = depth;
    const lum = pal.bgMaxLightness * 100 * (0.55 + 0.35 * depth);
    ctx.fillStyle = rgba(labToHex(lum, 0, 0), 0.5);
    const off = (fx.bgScroll * speed) % step;
    for (let y = a.y - step + off; y < a.y + a.h + step; y += step) {
      for (let x = a.x; x < a.x + a.w; x += step) {
        const r = 1.2 + depth * 1.6;
        ctx.beginPath();
        ctx.arc(x + ((L * 23) % step), y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// 레이어 1 — 지면 장판 · 지면 텔레그래프 (§12.3)
// ---------------------------------------------------------------------------
function drawGroundZones(ctx, world, pal) {
  const vz = world.data.rules.visual.zone;
  const items = world.zones.items;
  for (let i = 0; i < items.length; i += 1) {
    const z = items[i];
    if (!z.alive || z.fromPlayer) continue;                 // 플레이어 장판은 레이어 3
    const pulse = 0.5 + 0.5 * Math.sin(world.time * vz.pulseHz * Math.PI * 2);
    ctx.fillStyle = rgba(pal.threat.enemyBullet, vz.fillAlpha);
    ctx.beginPath();
    ctx.arc(z.x, z.y, z.radius, 0, Math.PI * 2);
    ctx.fill();
    // §12.3 — 활성 장판 = 외곽선 불투명 + 검은 외곽선 + 내부 0.30 + 1Hz 맥동
    ctx.lineWidth = world.data.rules.visual.telegraph.strokePx + 2;
    ctx.strokeStyle = pal.threat.outline;
    ctx.stroke();
    ctx.lineWidth = world.data.rules.visual.telegraph.strokePx;
    ctx.strokeStyle = rgba(pal.threat.enemyBullet, 0.7 + 0.3 * pulse);
    ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// 레이어 2 — 픽업 (§7.8: 글로우·외곽선 **금지**, ≤6px, 무광·납작)
// ---------------------------------------------------------------------------
function drawPickups(ctx, world, pal, interp, alpha) {
  const items = world.pickups.items;
  for (let i = 0; i < items.length; i += 1) {
    const q = items[i];
    if (!q.alive) continue;
    const x = lerpX(interp, interp.pickups, q, alpha);
    const y = lerpY(interp, interp.pickups, q, alpha);
    if (q.kind === 'coin') {                                 // 원반(중앙 구멍)
      ctx.fillStyle = pal.pickup.coin;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.arc(x, y, 1.8, 0, Math.PI * 2, true);
      ctx.fill('evenodd');
    } else if (q.kind === 'xp') {                            // 작은 마름모
      ctx.fillStyle = pal.pickup.xp;
      glyphPath(ctx, 'water', x, y, 2.6);
      ctx.fill();
    } else {                                                 // heal — §7.12.4 목록 밖이므로 호박 금지
      ctx.fillStyle = pal.hud.hpFill;
      glyphPath(ctx, 'grass', x, y, 3);
      ctx.fill();
    }
  }
}

// ---------------------------------------------------------------------------
// 레이어 4 — 플레이어 탄 (§7.4: additive · 알파 ≤ playerBulletMaxAlpha · 외곽선 금지 · 속성 글리프)
// ---------------------------------------------------------------------------
function drawPlayerBullets(ctx, world, pal, interp, alpha) {
  const r = world.data.rules.render;
  const vb = world.data.rules.visual.playerBullet;
  const items = world.playerBullets.items;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';                  // §7.4 — additive
  ctx.globalAlpha = r.playerBulletMaxAlpha;                  // §7.4 · §12.3 — 0.80 상한
  for (let i = 0; i < items.length; i += 1) {
    const b = items[i];
    if (!b.alive) continue;
    const x = lerpX(interp, interp.playerBullets, b, alpha);
    const y = lerpY(interp, interp.playerBullets, b, alpha);
    // §4.4 · I-2 — live 각인(orbit·aura)은 슬롯의 **현재** 각인을 보여야 색이 거짓말하지 않는다
    const el = b.stampMode === 'live' ? world.slots[b.slot].stampElement : b.element;
    glyphPath(ctx, el, x, y, b.radius);
    ctx.fillStyle = pal.element[el];
    ctx.fill();
    // §7.12.8 — 같은 hue 의 밝은 코어. 비율이라 H3 클램프를 자동 상속. 외곽선 없음(I-1)
    glyphPath(ctx, el, x, y, b.radius * vb.coreRadiusRatio);
    ctx.fillStyle = pal.elementCore[el];
    ctx.fill();
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// 레이어 5 — 적 기체 (§7.6: 중립 차콜 본체 + 속성색 외곽선 + 코어 글리프 + 림 라이트)
// ---------------------------------------------------------------------------
function drawEnemies(ctx, world, pal, fx, interp, alpha) {
  const vg = world.data.rules.visual.glyph;
  const el = world.data.rules.elite;
  const freezeScale = world.data.rules.visual.hitFx.superFreezeScale;   // §7.7 ×2 임팩트 프리즈
  const archetypes = world.data.enemies.archetypes;
  const items = world.enemies.items;

  for (let i = 0; i < items.length; i += 1) {
    const e = items[i];
    if (!e.alive) continue;
    const x = lerpX(interp, interp.enemies, e, alpha);
    const y = lerpY(interp, interp.enemies, e, alpha);
    let def = null;
    for (let j = 0; j < archetypes.length; j += 1) {
      if (archetypes[j].id === e.archetypeId) { def = archetypes[j]; break; }
    }
    if (def === null) throw new Error(`draw: 미지의 아키타입 "${e.archetypeId}" (§9.7)`);
    const color = pal.element[e.element];

    // §7.7 — ×2 히트의 0.04초 임팩트 프리즈: 개체(idx,gen)가 프리즈 중이면 본체를 ×superFreezeScale.
    //   ★ 게임 클럭·판정과 무관한 개체 단위 렌더 연출이다(§7.7 note). e.radius 는 원본 유지, r 만 스케일.
    const frozen = fx.freezeT[e.idx] > 0 && fx.freezeGen[e.idx] === e.gen;
    const r = frozen ? e.radius * freezeScale : e.radius;

    // 본체 — 속성별로 칠하지 않는다 (§7.6: 3층 분리의 근거)
    shapePath(ctx, def.shapeId, x, y, r);
    ctx.fillStyle = pal.enemyBody;
    ctx.fill();

    // 림 라이트 — 진행 방향 **반대쪽**, 알파 0.35
    const sp = Math.sqrt(e.vx * e.vx + e.vy * e.vy);
    if (sp > 0) {
      ctx.save();
      ctx.clip();
      ctx.fillStyle = rgba(color, 0.35);
      ctx.beginPath();
      ctx.arc(x - (e.vx / sp) * r * 0.75, y - (e.vy / sp) * r * 0.75, r * 0.9, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // 외곽선 — 속성색 2px (cvd 3px). 본체 크기 무관 항상
    ctx.lineWidth = pal.enemyOutlinePx;
    ctx.strokeStyle = color;
    shapePath(ctx, def.shapeId, x, y, r);
    ctx.stroke();

    // §7.6 엘리트 — 회전하는 이중 외곽선 + 개체 위 속성색 HP 바 + 상시 글리프
    if (e.elite) {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(world.time * Math.PI * 0.5);
      ctx.lineWidth = pal.enemyOutlinePx;
      ctx.strokeStyle = rgba(color, 0.7);
      shapePath(ctx, def.shapeId, 0, 0, r * el.sizeMult * 0.82);
      ctx.stroke();
      ctx.restore();
      // HP 바는 UI 요소 — 프리즈 팝에 흔들리지 않게 원본 e.radius 에 앵커
      const bw = e.radius * 2;
      ctx.fillStyle = rgba(pal.threat.outline, 0.8);
      ctx.fillRect(x - bw / 2, y - e.radius - 8, bw, 3);
      ctx.fillStyle = color;
      ctx.fillRect(x - bw / 2, y - e.radius - 8, bw * (e.hp / e.hpMax), 3);
    }

    // 코어 글리프 — max(6, bodyPx × bodyRatio), 상한 maxPx. cvd 는 ×1.5 + 항상 렌더
    const bodyPx = r * 2;
    if (e.elite || pal.cvd || bodyPx >= vg.lodMinBodyPx) {
      const g = Math.min(vg.maxPx, Math.max(6, bodyPx * vg.bodyRatio)) * 0.5 * pal.glyphScale;
      glyphPath(ctx, e.element, x, y, g);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }
}

// ---------------------------------------------------------------------------
// 레이어 6 — 플레이어 기체 + 슬롯 스트립 + 상태 배지 + 림 오라 (§7.5)
// ---------------------------------------------------------------------------
function drawPlayer(ctx, world, pal, fx, interp, alpha) {
  const rp = world.data.rules.player;
  const vs = world.data.rules.visual.stance;
  const p = world.player;
  const x = interp.enabled ? interp.player.x + (p.x - interp.player.x) * alpha : p.x;
  const y = interp.enabled ? interp.player.y + (p.y - interp.player.y) * alpha : p.y;
  const stanceColor = pal.element[p.stance];

  ctx.save();
  // §2.4 — i-frame 깜빡임 8Hz, alpha 0.35 ↔ 1.0
  if (p.iframeSec > 0) {
    const ph = Math.sin(world.time * world.data.rules.visual.iframeBlinkHz * Math.PI * 2);
    ctx.globalAlpha = ph > 0 ? 1.0 : 0.35;
  }

  // ④ 림 오라 — 현재 스탠스 색 소프트 글로우, 알파 0.40
  const grad = ctx.createRadialGradient(x, y, rp.spriteRadius * 0.4, x, y, rp.spriteRadius * 1.8);
  grad.addColorStop(0, rgba(stanceColor, vs.auraAlpha));
  grad.addColorStop(1, rgba(stanceColor, 0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, rp.spriteRadius * 1.8, 0, Math.PI * 2);
  ctx.fill();

  // 기체 본체 — ★ 정본에 플레이어 기체의 도형·색이 없다 (§9.10 은 적·보스만 확정) → 보고 대상.
  //   팔레트 어휘를 늘리지 않기 위해 §7.6 의 중립 차콜 본체 + 스탠스색 외곽선을 그대로 재사용했다.
  ctx.beginPath();
  ctx.moveTo(x, y - rp.spriteRadius);
  ctx.lineTo(x + rp.spriteRadius * 0.8, y + rp.spriteRadius * 0.75);
  ctx.lineTo(x, y + rp.spriteRadius * 0.35);
  ctx.lineTo(x - rp.spriteRadius * 0.8, y + rp.spriteRadius * 0.75);
  ctx.closePath();
  ctx.fillStyle = pal.enemyBody;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = stanceColor;
  ctx.stroke();

  // ② 슬롯 스트립 (임뷰 칩) — 기체 하단 +14px, 7px × 4칸, 간격 2px. 좌→우 = 슬롯 1..4
  const n = world.slots.length;
  const w = n * vs.pipPx + (n - 1) * vs.pipGapPx;
  let sx = x - w / 2 + vs.pipPx / 2;
  const sy = y + vs.pipOffsetYPx;
  for (let i = 0; i < n; i += 1) {
    const s = world.slots[i];
    const r = vs.pipPx / 2;
    if (s.stampElement === 'normal') {
      // 부여 안 됨 = 은색 ● (I-2 — 실제로 ×1 이므로 속성색을 쓰면 화면이 거짓말한다)
      ctx.fillStyle = s.weaponId === null ? rgba(pal.element.normal, 0.25) : pal.element.normal;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.fillStyle = pal.element[s.stampElement];
      glyphPath(ctx, s.stampElement, sx, sy, r);
      ctx.fill();
    }
    sx += vs.pipPx + vs.pipGapPx;
  }

  // ③ 상태이상 배지 — 기체 위 −16px. ∿ 둔화 / ✳ 스턴 + 잔여 감소 바 12×2px (§7.12.4-② 호박)
  const st = p.stunSec > 0 ? 'stun' : p.slowSec > 0 ? 'slow' : null;
  if (st !== null) {
    const by = y - 16;
    ctx.strokeStyle = pal.status.band;
    ctx.lineWidth = 2;
    if (st === 'slow') {
      ctx.beginPath();
      for (let i = 0; i <= 12; i += 1) {
        const px = x - 6 + i;
        const py = by + Math.sin((i / 12) * Math.PI * 2) * 3;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    } else {
      for (let i = 0; i < 3; i += 1) {
        const a = (i * Math.PI) / 3;
        ctx.beginPath();
        ctx.moveTo(x - Math.cos(a) * 6, by - Math.sin(a) * 6);
        ctx.lineTo(x + Math.cos(a) * 6, by + Math.sin(a) * 6);
        ctx.stroke();
      }
    }
    const dur = st === 'stun' ? p.stunSec : p.slowSec;
    ctx.fillStyle = pal.status.band;
    ctx.fillRect(x - 6, by + 7, 12 * Math.min(1, dur), 2);
  }
  ctx.restore();

  // §7.5 전환 링 — 투자 있으면 속성색 정상 확장 / 투자 0이면 확장 중 회색으로 **탈색**
  if (fx.ringT >= 0) {
    const t = fx.ringT / vs.ringExpandSec;
    const from = pal.element[fx.ringElement];
    const to = fx.ringInvested ? from : pal.neutralGray;
    ctx.save();
    ctx.lineWidth = vs.ringStrokePx;
    ctx.strokeStyle = rgba(fx.ringInvested ? from : mix(from, to, t), 1 - t);
    ctx.beginPath();
    ctx.arc(x, y, vs.ringMaxRadiusPx * t, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
  return { x, y };
}

function mix(a, b, t) {
  const [ar, ag, ab] = hexToRgb(a);
  const [br, bg, bb] = hexToRgb(b);
  return rgbToHex(ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t);
}

// ---------------------------------------------------------------------------
// 레이어 7 — 히트 피드백 3중 감각 (§7.7). ★ 적 탄(9)보다 **아래**에 그린다 → I-4 (탄 불가림) 보존.
//   ×2 (super)  : 공격 속성색 화이트-핫 코어 + 확장 버스트 링 + 스파크 3개 (발광). 처치면 흰 링 확장.
//   ×1 (neutral): 짧은 백색 플래시 + 스파크 1개. 처치면 소형 파열.
//   ×0.5(resist): **회색 방패 호**(sweep 120° · stroke 2px · 발광 아님) + 스파크 0 → "튕겼다".
//   ★ 색은 palette 가 소유(§7.2). 규격값(sweep·stroke·수명·스파크 수·프리즈)은 visual.hitFx 가 소유(§9.4.3).
// ---------------------------------------------------------------------------
const GOLDEN_ANGLE = 2.399963229728653;           // 스파크 각도 분산(무-RNG · 결정적)

function drawResistArc(ctx, pal, vh, h, t) {
  // 둔한 회색 방패 호. 아래(+y = 플레이어/입사탄 쪽)를 향해 살짝 바깥으로 밀린다 = 튕겨냄
  const sweep = (vh.resistArcSweepDeg * Math.PI) / 180;
  const base = Math.PI / 2;
  const r = 12 + t * 6;
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';    // 발광 금지 — 방어는 둔하다(§7.7 "약한 스파크")
  ctx.globalAlpha = (1 - t) * 0.9;
  ctx.lineWidth = vh.resistArcStrokePx;
  ctx.strokeStyle = pal.neutralGray;
  ctx.beginPath();
  ctx.arc(h.x, h.y, r, base - sweep / 2, base + sweep / 2);
  ctx.stroke();
  ctx.restore();
}

function drawBurst(ctx, pal, vh, h, t) {
  const isSuper = h.tier === 'super';
  const color = isSuper ? pal.element[h.element] : pal.threat.bulletCore;  // neutral = 백색 플래시
  const nSpark = isSuper ? vh.particles.super : vh.particles.neutral;
  const fade = 1 - t;
  const baseR = isSuper ? 10 : 5;
  const grow = isSuper ? 22 : 9;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';        // 발광 = 플레이어 FX(I-1: 외곽선 없음)

  // 화이트-핫 코어 — §7.7 ×2 "속성색 화이트-핫 1프레임" (첫 ~1/6 구간만 강한 백색)
  const hot = 1 - t * 6;
  if (hot > 0) {
    ctx.globalAlpha = hot * (isSuper ? 0.95 : 0.6);
    ctx.fillStyle = pal.threat.bulletCore;
    ctx.beginPath();
    ctx.arc(h.x, h.y, isSuper ? 7 : 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // 확장 버스트 링 — 속성색(super) / 백색(neutral). 처치면 더 크게(대형 파열)
  const rr = baseR + t * grow * (h.killed ? 1.8 : 1);
  ctx.globalAlpha = fade * (isSuper ? 0.85 : 0.5);
  ctx.lineWidth = isSuper ? 3 : 1.5;
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.arc(h.x, h.y, rr, 0, Math.PI * 2);
  ctx.stroke();

  // 스파크 — 중심에서 방사. 각도는 seed 로 분산(결정적, 무-RNG)
  const sparkLen = (isSuper ? 14 : 8) * (0.4 + t);
  ctx.lineWidth = isSuper ? 2 : 1.5;
  ctx.strokeStyle = color;
  ctx.globalAlpha = fade * (isSuper ? 0.9 : 0.55);
  for (let k = 0; k < nSpark; k += 1) {
    const ang = h.seed * GOLDEN_ANGLE + (k * Math.PI * 2) / Math.max(1, nSpark);
    const cx = h.x + Math.cos(ang) * baseR;
    const cy = h.y + Math.sin(ang) * baseR;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(ang) * sparkLen, cy + Math.sin(ang) * sparkLen);
    ctx.stroke();
  }

  // 처치 FX — 흰 링 확장 (§7.7 처치: 속성색 대형 파열 + 흰 링)
  if (h.killed) {
    ctx.globalAlpha = fade * 0.8;
    ctx.lineWidth = 2;
    ctx.strokeStyle = pal.threat.bulletCore;
    ctx.beginPath();
    ctx.arc(h.x, h.y, baseR + t * grow * 2.4, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawHitFx(ctx, world, pal, fx) {
  const vh = world.data.rules.visual.hitFx;
  const hb = fx.hits.buf;
  for (let i = 0; i < hb.length; i += 1) {
    const h = hb[i];
    if (!h.active) continue;
    const t = h.life > 0 ? h.age / h.life : 1;     // 0→1 진행
    if (t >= 1) continue;
    if (h.tier === 'resist') drawResistArc(ctx, pal, vh, h, t);
    else drawBurst(ctx, pal, vh, h, t);            // super · neutral
  }
}

// ---------------------------------------------------------------------------
// 레이어 8 — 공중 텔레그래프 (§7.4: 자홍 점선 + 알파 0.50, 채움 금지)
// ---------------------------------------------------------------------------
function drawTelegraphs(ctx, world, pal) {
  const vt = world.data.rules.visual.telegraph;
  const items = world.telegraphs.items;
  ctx.save();
  ctx.globalAlpha = vt.airAlpha;
  ctx.setLineDash([vt.dashPx, vt.dashPx]);
  ctx.lineWidth = vt.strokePx;
  ctx.strokeStyle = pal.threat.telegraph;
  for (let i = 0; i < items.length; i += 1) {
    const t = items[i];
    if (!t.alive) continue;
    ctx.beginPath();
    ctx.arc(t.x, t.y, t.r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// 레이어 9 — 적 탄 (§7.4: 불투명 · additive 금지 · 검은 하드 외곽선 2px · 흰 스페큘러 코어)
//   ★ I-4 — 이 레이어 위에 면적이 있는 것은 아무것도 없다 (히트박스 도트 r=4 만 예외, §7.5)
// ---------------------------------------------------------------------------
function drawEnemyBullets(ctx, world, pal, interp, alpha) {
  const items = world.enemyBullets.items;
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1.0;                                     // §7.4 — 알파 1.0 고정
  for (let i = 0; i < items.length; i += 1) {
    const b = items[i];
    if (!b.alive) continue;
    const x = lerpX(interp, interp.enemyBullets, b, alpha);
    const y = lerpY(interp, interp.enemyBullets, b, alpha);

    // §7.4 · S14 — (status ≠ null) ⟺ (육각) ⟺ (호박 테두리). 3중 동치
    if (b.status !== null) regular(ctx, x, y, b.radius, 6, Math.PI / 6);
    else { ctx.beginPath(); ctx.arc(x, y, b.radius, 0, Math.PI * 2); }
    ctx.fillStyle = pal.threat.enemyBullet;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = pal.threat.outline;
    ctx.stroke();
    if (b.status !== null) {                                 // §7.12.4-⑤ — 호박 테두리
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = pal.status.band;
      regular(ctx, x, y, b.radius + 2, 6, Math.PI / 6);
      ctx.stroke();
    }
    // 흰 스페큘러 점 — hue 가 없다 (플레이어 탄의 「같은 hue 의 밝은 판」과 배타, §7.12.8)
    ctx.fillStyle = pal.threat.bulletCore;
    ctx.beginPath();
    ctx.arc(x - b.radius * 0.28, y - b.radius * 0.28, Math.max(1, b.radius * 0.24), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// 레이어 10 — 히트박스 도트 + 스탠스 링 (§7.5 ① — **적 탄보다 위**)
//   그려진 흰 코어 도트의 반지름 = 히트박스 반지름 = 4. 크기도 거짓말하지 않는다 (§2.3)
// ---------------------------------------------------------------------------
function drawHitboxDot(ctx, world, pal, px, py) {
  const rp = world.data.rules.player;
  const vs = world.data.rules.visual.stance;
  if (!world.data.rules.hud.hitboxAlwaysVisible) return;
  const p = world.player;

  // §7.8 — 자석 반경 가시화: 은색 점선 원, 알파 0.12 상시
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = rgba(pal.element.normal, 0.12);
  ctx.beginPath();
  ctx.arc(px, py, rp.magnetRadius * (1 + world.shopMagnetPct), 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  ctx.fillStyle = pal.threat.bulletCore;
  ctx.beginPath();
  ctx.arc(px, py, rp.hitboxRadius, 0, Math.PI * 2);          // r = 4 = 실제 히트박스
  ctx.fill();
  // 스탠스 색 링 stroke 2px, r 4 → 6. 투자 0인 스탠스면 은색 (I-2)
  const invested = p.stance === 'normal' ? false : p.invest[p.stance] > 0;
  ctx.lineWidth = vs.dotRingPx;
  ctx.strokeStyle = invested ? pal.element[p.stance] : pal.element.normal;
  ctx.beginPath();
  ctx.arc(px, py, rp.hitboxRadius + vs.dotRingPx / 2, 0, Math.PI * 2);
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// 전체 (§12.3 레이어 스택 — 이 함수의 호출 순서가 곧 그 표다)
// ---------------------------------------------------------------------------
export function drawWorld(ctx, world, pal, fx, interp, alpha) {
  const v = world.data.rules.view;
  const a = v.arena;

  drawBackground(ctx, world, pal, fx);                        // 0

  ctx.save();
  ctx.beginPath();
  ctx.rect(a.x, a.y, a.w, a.h);
  ctx.clip();                                                 // §1.1 — 아레나 밖으로 새지 않는다

  drawArenaBands(ctx, world, pal);                            // 1 — 띠 (내용의 소유자는 hud.js)
  drawGroundZones(ctx, world, pal);                           // 1
  drawPickups(ctx, world, pal, interp, alpha);                // 2
  drawPlayerBullets(ctx, world, pal, interp, alpha);          // 4
  drawEnemies(ctx, world, pal, fx, interp, alpha);            // 5 (§7.7 임팩트 프리즈 = 본체 팝)
  const pp = drawPlayer(ctx, world, pal, fx, interp, alpha);  // 6
  drawHitFx(ctx, world, pal, fx);                             // 7 — §7.7 3중 감각 (적 탄 9보다 아래 = I-4)
  drawTelegraphs(ctx, world, pal);                            // 8
  drawEnemyBullets(ctx, world, pal, interp, alpha);           // 9
  drawHitboxDot(ctx, world, pal, pp.x, pp.y);                 // 10

  ctx.restore();
}

// rgba 만 외부(main.js)가 쓴다. lighten/desaturate/mix/hexToLab/labToHex/shapePath 는 importer 0 →
// 모듈-프라이빗으로 강등(죽은 export 표면 제거, 기능 영향 없음)
export { rgba };
