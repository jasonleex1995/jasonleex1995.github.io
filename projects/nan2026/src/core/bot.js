/**
 * src/core/bot.js — 결정적 AI 플레이어 (순수 core, §10.2 · §10.4.1)
 *
 * ★ 변종 B — CONTINUOUS STEERING (지각 지연이 있는 연속 조향).
 *   설계: 의도(표적 · 위협 SET · 스탠스)는 reactionSec 마다만 갱신한다(= 눈 감는 창 = 지연).
 *   그러나 이동 벡터는 **매 틱** 그 기억된 의도 + 라이브 기하로 새로 합성한다:
 *     · 사격 위치(표적 아래 + x정렬)로의 인력
 *     · 기억된(스냅샷·외삽) 탄/몸통/장판/빔/경계로부터의 1/거리 반발
 *   합벡터가 자연히 «빈 공간을 향하되 총구는 유지»한다. 새 위협은 다음 갱신 전엔 보이지 않는다
 *   (지연의 모델). 알던 위협은 기억한 속도로 외삽해 매끄럽게 추적한다(사람의 예측).
 *
 * 정본 v1.4 구현 절:
 *   §10.2  8번째 RNG 스트림 `bot`. 독립이므로 봇 추첨이 콘텐츠 시퀀스를 흔들지 않는다.
 *   §10.4.1 정책 4축 + 난이도는 봇의 반응 지연으로만 들어온다.
 *   §5.7   출력 = makeInput() 모양(불리언). §9.1 순수성 — rng 는 world.rng.bot 만.
 *   §10.3  봇 상태는 ensureBot 에서 최초 1회만 alloc. 핫패스 0 alloc(스냅샷은 미리 잡은 배열에 채운다).
 */

import { TICK_HZ } from './step.js';
import { elementMul } from './elements.js';

// ── 조향 가중치(튜닝 축 — bot.js 인라인 리터럴, §데이터 아님) ───────────────────
const T = {
  W_ATTRACT_X: 0.85,   // 사격선 x정렬 인력(강하게 — 무기는 위로 나간다 §1.1)
  W_ATTRACT_Y: 0.35,   // 표적 아래 스탠드오프로의 인력
  ATTRACT_CAP: 180,    // 인력 성분 상한(px) — 근접 위협이 인력을 압도하게
  W_BULLET: 4.5,       // 적탄 반발
  W_CONTACT: 5.0,      // 적 몸통·장판 반발(접촉사 61% — 최우선 회피)
  W_LASER: 4.2,        // 빔 반발
  W_WALL: 2.6,         // 경계 반발(구석에 몰리지 않게)
  PERCEPT: 340,        // 지각 반경(px) — 이 안의 위협만 스냅샷(지연·성능)
  CONTACT_MARGIN_MUL: 0.55, // 접촉 여유(탄보다 관대 — 요격을 살린다)
  CLUSTER_X: 62,       // 밀집 클러스터 x창(px)
  LOWHP_REP_MUL: 1.45, // 저체력 시 반발 증폭
  LOWHP_ATTRACT_MUL: 0.6, // 저체력 시 인력 감쇠
  DEAD: 3,             // 목표 근처 떨림 방지(px)
  MOB_LINE_OFFSET: 360,  // 모브 사격선 = 하단에서 위로 얼마(px) — 중단이 명중·클리어 최적(실측)
  BOSS_ATTRACT_X: 4.5,   // 보스전 x정렬 인력(강 — 열을 지켜 명중을 유지)
  BOSS_CAP_X: 500,       // 보스전 x인력 상한(px) — 크게(반발보다 우선)
  PICK_MAX_DIST: 140,    // 이 거리 안의 픽업만 좇는다(balanced) — 자석 밖 근접만
};

// 스냅샷 용량(ensureBot 에서 1회 alloc)
const CAP_BUL = 72;
const CAP_CON = 44;
const CAP_LAS = 8;

/** 봇 상태를 최초 1회 만든다(§10.3). 정책은 meta.bot.baseline 에서 시작한다. */
function ensureBot(world) {
  if (world.bot !== undefined) return world.bot;
  const b = world.data.meta.bot;
  world.bot = {
    policy: {
      draft: b.baseline.draft,
      farm: b.baseline.farm,
      stance: b.baseline.stance,
      shop: b.baseline.shop,
      forceNoElement: false,
    },
    input: { left: false, right: false, up: false, down: false,
      stanceNormal: false, stanceFire: false, stanceWater: false, stanceGrass: false },
    decideT: 0,                   // 다음 재결정까지 남은 게임초(= 반응 지연)
    stanceT: 0,                   // 스탠스 전환 쿨다운(게임초)
    wantStance: 'normal',
    armorIdx: -1, armorGen: -1,   // §8.13 격파 중인 armor 부위(화력 집중)
    // ── 기억된 의도(reactionSec 마다 갱신) ──
    tgtIdx: -1, tgtGen: -1,       // 추적 표적 개체(라이브 위치로 조준)
    tgtBoss: false,               // 표적이 보스/중간보스(사격선 위로 안 올라감)
    pickX: 0, pickY: 0, hasPick: false, // 파밍 목적지(스냅샷)
    fallbackX: 0,                 // 표적 소실 시 x
    aimJx: 0, aimJy: 0,           // 손 오차(갱신 시 1회 추첨, 창 동안 고정)
    margin: 0,                    // 반응 시간 동안 갈 수 있는 거리(px)
    lowHp: false,
    // ── 스냅샷 위협 SET(미리 alloc, 매 갱신 채움 — 핫패스 0 alloc) ──
    snapElapsed: 0,               // 스냅샷 이후 경과(외삽용)
    nBul: 0, bx: new Float64Array(CAP_BUL), by: new Float64Array(CAP_BUL),
    bvx: new Float64Array(CAP_BUL), bvy: new Float64Array(CAP_BUL), br: new Float64Array(CAP_BUL),
    nCon: 0, cx: new Float64Array(CAP_CON), cy: new Float64Array(CAP_CON),
    cvx: new Float64Array(CAP_CON), cvy: new Float64Array(CAP_CON), cr: new Float64Array(CAP_CON),
    nLas: 0, lx: new Float64Array(CAP_LAS), ly: new Float64Array(CAP_LAS),
    la: new Float64Array(CAP_LAS), lr: new Float64Array(CAP_LAS),
  };
  return world.bot;
}

/** 정책 오버라이드(시뮬이 축을 바꿔가며 돌린다). 부분 갱신. */
export function setBotPolicy(world, patch) {
  const b = ensureBot(world);
  const keys = Object.keys(patch);
  for (let i = 0; i < keys.length; i += 1) b.policy[keys[i]] = patch[keys[i]];
  return b.policy;
}

/** §10.4.1 — 반응 지연(게임초). 난이도 배속이 클수록 「눈 감는 창」이 길어진다. */
function reactionSec(world) {
  const b = world.data.meta.bot;
  const diff = world.data.meta.difficulty[world.difficultyId];
  const speed = diff === undefined ? 1 : diff.speed;
  const jitter = (world.rng.bot.f() * 2 - 1) * b.reactionJitterMs;
  const ms = b.reactionMs + jitter;
  const ticks = Math.round((ms / 1000) * TICK_HZ * speed);
  return (ticks < 1 ? 1 : ticks) / TICK_HZ;
}

/** 가장 가까운 살아있는 적(보스 코어 포함, 중간보스 제외). */
function nearestEnemy(world) {
  const en = world.enemies.items;
  const p = world.player;
  let best = null;
  let bestD = 0;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive) continue;
    if (e.midBossId !== '') continue;   // §8.9 중간보스는 조준 대상에서 제외(주차 방지)
    const dx = e.x - p.x;
    const dy = e.y - p.y;
    const d = dx * dx + dy * dy;
    if (best === null || d < bestD) { bestD = d; best = e; }
  }
  return best;
}

/** §8.13 소프트게이트 — armor 부위를 먼저 부순다(화력 집중, sticky). */
function nearestArmor(world) {
  const en = world.enemies.items;
  const p = world.player;
  let best = null;
  let bestD = 0;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive || !e.isBoss || e.partType !== 'armor') continue;
    const dx = e.x - p.x;
    const dy = e.y - p.y;
    const d = dx * dx + dy * dy;
    if (best === null || d < bestD) { bestD = d; best = e; }
  }
  return best;
}

/**
 * ★ 모브 표적 — «밀집한 죽일 수 있는 클러스터»의 x 에 사격선을 건다.
 *   무기는 위로 나가므로(§1.1) x정렬이 명중의 전제다. 각 모브에 대해 x창(CLUSTER_X) 안의
 *   이웃 수를 세고(밀집도) 최대인 개체를 고른다 — 열(column)이 가장 많은 적을 관통한다.
 *   동점이면 더 가깝고 더 아래(사거리 안)인 개체.
 */
function clusterTarget(world) {
  const { CLUSTER_X } = T;
  const en = world.enemies.items;
  const p = world.player;
  let best = null;
  let bestScore = -1;
  let bestD = 0;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive || e.isBoss || e.midBossId !== '') continue;
    let n = 0;
    for (let j = 0; j < en.length; j += 1) {
      const o = en[j];
      if (!o.alive || o.isBoss || o.midBossId !== '') continue;
      const ddx = o.x - e.x;
      if (ddx < CLUSTER_X && ddx > -CLUSTER_X) n += 1;
    }
    const dx = e.x - p.x;
    const dy = e.y - p.y;
    const d = dx * dx + dy * dy;
    if (n > bestScore || (n === bestScore && d < bestD)) {
      bestScore = n; best = e; bestD = d;
    }
  }
  return best;
}

/** §8.13 sticky armor → 보스 코어 → 모브 클러스터 순으로 사격 표적을 고른다. */
function pickFoe(world) {
  const b = world.bot;
  // 고정된 armor 가 살아있으면 계속 그것을(화력 집중)
  if (b.armorIdx >= 0) {
    const held = world.enemies.items[b.armorIdx];
    if (held !== undefined && held.alive && held.gen === b.armorGen
      && held.isBoss && held.partType === 'armor') return held;
    b.armorIdx = -1; b.armorGen = -1;
  }
  const armor = nearestArmor(world);
  if (armor !== null) {
    b.armorIdx = armor.idx; b.armorGen = armor.gen;
    return armor;
  }
  const nb = nearestEnemy(world);
  if (nb !== null && nb.isBoss) return nb;   // 보스 코어(armor 없음)
  const cl = clusterTarget(world);
  return cl !== null ? cl : nb;
}

/** 가장 가까운 픽업(파밍 대상). 없으면 null. */
function nearestPickup(world) {
  const it = world.pickups.items;
  const p = world.player;
  let best = null;
  let bestD = 0;
  for (let i = 0; i < it.length; i += 1) {
    const k = it[i];
    if (!k.alive) continue;
    const dx = k.x - p.x;
    const dy = k.y - p.y;
    const d = dx * dx + dy * dy;
    if (best === null || d < bestD) { bestD = d; best = k; }
  }
  return best;
}

/** §10.4.1 stance — 어떤 스탠스를 원하는가(사격 표적을 상성으로 때린다). */
function desiredStance(world, fireTarget) {
  const b = world.bot;
  if (b.policy.stance === 'static') return 'normal';
  const matrix = world.data.elements.matrix;
  const investable = world.data.elements.investable;

  let targetElement = null;
  if (b.policy.stance === 'majorityOnScreen') {
    const en = world.enemies.items;
    const count = Object.create(null);
    let bestN = 0;
    for (let i = 0; i < en.length; i += 1) {
      const e = en[i];
      if (!e.alive) continue;
      const c = (count[e.element] === undefined ? 0 : count[e.element]) + 1;
      count[e.element] = c;
      if (c > bestN) { bestN = c; targetElement = e.element; }
    }
  } else {
    const e = fireTarget !== null && fireTarget !== undefined ? fireTarget : nearestEnemy(world);
    if (e !== null) targetElement = e.element;
  }
  if (targetElement === null) return world.player.stance;
  for (let i = 0; i < investable.length; i += 1) {
    if (elementMul(matrix, investable[i], targetElement) > 1) return investable[i];
  }
  return world.player.stance;
}

/**
 * ★ 위협 SET 스냅샷 — reactionSec 마다 1회. 미리 잡은 배열에 채운다(핫패스 0 alloc).
 *   지각 반경(PERCEPT) 안의 위협만 담는다(지연 · 성능). 이후 매 틱 외삽해 반발을 만든다.
 */
function snapshotThreats(world, b) {
  const { PERCEPT, CONTACT_MARGIN_MUL } = T;
  const p = world.player;
  const rp = world.data.rules.player;
  const margin = b.margin;
  const percept2 = PERCEPT * PERCEPT;

  // ── 적탄 ──
  let n = 0;
  const eb = world.enemyBullets.items;
  for (let i = 0; i < eb.length && n < CAP_BUL; i += 1) {
    const bu = eb[i];
    if (!bu.alive) continue;
    const rx = bu.x - p.x;
    const ry = bu.y - p.y;
    if (rx * rx + ry * ry > percept2) continue;
    b.bx[n] = bu.x; b.by[n] = bu.y; b.bvx[n] = bu.vx; b.bvy[n] = bu.vy;
    b.br[n] = rp.hitboxRadius + bu.hitRadius + margin;
    n += 1;
  }
  b.nBul = n;

  // ── 적 몸통(모브만 — 보스/중간보스는 별도) + 장판을 한 배열에 ──
  let m = 0;
  const en = world.enemies.items;
  const cMargin = margin * CONTACT_MARGIN_MUL;
  for (let i = 0; i < en.length && m < CAP_CON; i += 1) {
    const e = en[i];
    if (!e.alive || e.isBoss || e.midBossId !== '') continue;
    const rx = e.x - p.x;
    const ry = e.y - p.y;
    if (rx * rx + ry * ry > percept2) continue;
    b.cx[m] = e.x; b.cy[m] = e.y; b.cvx[m] = e.vx; b.cvy[m] = e.vy;
    b.cr[m] = rp.hitboxRadius + e.radius + cMargin;
    m += 1;
  }
  const zs = world.zones.items;
  for (let i = 0; i < zs.length && m < CAP_CON; i += 1) {
    const z = zs[i];
    if (!z.alive || z.fromPlayer) continue;
    const rx = z.x - p.x;
    const ry = z.y - p.y;
    if (rx * rx + ry * ry > percept2) continue;
    b.cx[m] = z.x; b.cy[m] = z.y; b.cvx[m] = 0; b.cvy[m] = 0;
    b.cr[m] = rp.hitboxRadius + z.radius + margin;
    m += 1;
  }
  b.nCon = m;

  // ── 빔(telegraph laser) — 선. ──
  let k = 0;
  const ts = world.telegraphs.items;
  for (let i = 0; i < ts.length && k < CAP_LAS; i += 1) {
    const t2 = ts[i];
    if (!t2.alive || t2.kind !== 'laser') continue;
    b.lx[k] = t2.x; b.ly[k] = t2.y; b.la[k] = t2.a;
    b.lr[k] = rp.hitboxRadius + t2.r * 0.5 + margin;
    k += 1;
  }
  b.nLas = k;
}

// 조향 스크래치(모듈 스코프 — 핫패스 0 alloc)
const _steer = { x: 0, y: 0 };

/**
 * ★ 매 틱 조향 벡터 — 기억된 의도(표적·위협 SET) + 라이브 기하.
 *   결과(px 스케일)를 _steer 에 쓴다. 인력(표적 사격위치) + 반발(외삽 위협) + 경계.
 */
function steer(world, b) {
  const { W_ATTRACT_X, W_ATTRACT_Y, ATTRACT_CAP, W_BULLET, W_CONTACT, W_LASER, W_WALL,
    LOWHP_REP_MUL, LOWHP_ATTRACT_MUL, MOB_LINE_OFFSET, BOSS_ATTRACT_X, BOSS_CAP_X } = T;
  const p = world.player;
  const bounds = world.bounds;
  const el = b.snapElapsed;
  const repMul = b.lowHp ? LOWHP_REP_MUL : 1;

  // ── 표적 사격 위치(라이브) ──
  let tx;
  let ty;
  if (b.hasPick) {
    tx = b.pickX; ty = b.pickY;
  } else {
    const t = b.tgtIdx >= 0 ? world.enemies.items[b.tgtIdx] : undefined;
    if (t !== undefined && t.alive && t.gen === b.tgtGen) {
      tx = t.x;
      if (b.tgtBoss) {
        // 보스/중간보스: 아래에 서서 위로 쏜다(사격선 위로 안 올라감)
        const rp = world.data.rules.player;
        const stand = rp.hitboxRadius + t.radius + b.margin;
        ty = t.y + stand;
      } else {
        ty = bounds.maxY - MOB_LINE_OFFSET;   // 모브: 하단 사격선 유지
      }
    } else {
      tx = b.fallbackX; ty = bounds.maxY - MOB_LINE_OFFSET;
    }
  }
  tx += b.aimJx; ty += b.aimJy;

  // 저체력이면 목적지를 하단 중앙으로 당긴다(후퇴)
  if (b.lowHp && !b.hasPick) {
    tx = tx * 0.45 + ((bounds.minX + bounds.maxX) * 0.5) * 0.55;
    ty = bounds.maxY;
  }

  const aMul = b.lowHp ? LOWHP_ATTRACT_MUL : 1;
  // ★ 보스전 사격 uptime — 실측: 안 하면 봇이 탄을 피하느라 x정렬을 못 유지해 보스전 명중률 ≈1%.
  //   보스/중간보스 표적에는 x인력을 강하게(높은 상한) 걸어 «열을 지킨다» — 위로 쏘는 무기의 전제.
  const wax = b.tgtBoss ? BOSS_ATTRACT_X : W_ATTRACT_X;
  const capx = b.tgtBoss ? BOSS_CAP_X : ATTRACT_CAP;
  let adx = (tx - p.x) * wax * aMul;
  let ady = (ty - p.y) * W_ATTRACT_Y * aMul;
  if (adx > capx) adx = capx; else if (adx < -capx) adx = -capx;
  if (ady > ATTRACT_CAP) ady = ATTRACT_CAP; else if (ady < -ATTRACT_CAP) ady = -ATTRACT_CAP;
  let ax = adx;
  let ay = ady;

  // ── 반발: 적탄(외삽) ──
  for (let i = 0; i < b.nBul; i += 1) {
    const ex = b.bx[i] + b.bvx[i] * el;
    const ey = b.by[i] + b.bvy[i] * el;
    const cx = ex - p.x;
    const cy = ey - p.y;
    const danger = b.br[i];
    const d2 = cx * cx + cy * cy;
    if (d2 > danger * danger) continue;
    const d = Math.sqrt(d2);
    const w = ((danger - d) / danger) * W_BULLET * repMul;
    if (d > 0.0001) { ax -= (cx / d) * danger * w; ay -= (cy / d) * danger * w; }
    else ay += danger * w;
  }
  // ── 반발: 몸통·장판(외삽) ──
  for (let i = 0; i < b.nCon; i += 1) {
    const ex = b.cx[i] + b.cvx[i] * el;
    const ey = b.cy[i] + b.cvy[i] * el;
    const cx = ex - p.x;
    const cy = ey - p.y;
    const danger = b.cr[i];
    const d2 = cx * cx + cy * cy;
    if (d2 > danger * danger) continue;
    const d = Math.sqrt(d2);
    // 몸통과 장판을 구분하지 않고 강한 가중(접촉사 회피). 장판은 vel 0 이라 외삽 안 됨.
    const w = ((danger - d) / danger) * W_CONTACT * repMul;
    if (d > 0.0001) { ax -= (cx / d) * danger * w; ay -= (cy / d) * danger * w; }
    else ay += danger * w;
  }
  // ── 반발: 빔(선까지의 수직거리) ──
  for (let i = 0; i < b.nLas; i += 1) {
    const rx = p.x - b.lx[i];
    const ry = p.y - b.ly[i];
    const a = b.la[i];
    const sn = -Math.sin(a);
    const cs = Math.cos(a);
    const signed = sn * rx + cs * ry;
    const perp = signed < 0 ? -signed : signed;
    const danger = b.lr[i];
    if (perp > danger) continue;
    const side = signed >= 0 ? 1 : -1;
    const w = ((danger - perp) / danger) * W_LASER * repMul;
    ax += sn * side * danger * w;
    ay += cs * side * danger * w;
  }

  // ── 경계 반발(라이브 — 벽은 안 움직인다) ──
  const wall = b.margin;
  if (wall > 0) {
    if (p.x - bounds.minX < wall) ax += ((wall - (p.x - bounds.minX)) / wall) * wall * W_WALL;
    if (bounds.maxX - p.x < wall) ax -= ((wall - (bounds.maxX - p.x)) / wall) * wall * W_WALL;
    if (p.y - bounds.minY < wall) ay += ((wall - (p.y - bounds.minY)) / wall) * wall * W_WALL;
    if (bounds.maxY - p.y < wall) ay -= ((wall - (bounds.maxY - p.y)) / wall) * wall * W_WALL;
  }

  _steer.x = ax; _steer.y = ay;
}

/**
 * ★ 봇의 한 틱 입력. step(world, botInput(world, dt), dt) 로 쓰인다.
 *   의도(표적·위협 SET·스탠스)는 반응 지연마다만 갱신하고, 이동은 매 틱 조향한다.
 */
export function botInput(world, dt) {
  const b = ensureBot(world);
  const p = world.player;
  const bt = world.data.meta.bot;
  const rp = world.data.rules.player;
  const inp = b.input;

  // 스탠스 키는 매 틱 내린다 — 상승 엣지를 만들 틱에만 올린다
  inp.stanceNormal = false; inp.stanceFire = false; inp.stanceWater = false; inp.stanceGrass = false;

  // ── 의도 갱신 (반응 지연마다 = 눈 감는 창) ──────────────────────────────────
  b.decideT -= dt;
  if (b.decideT <= 0) {
    b.decideT = reactionSec(world);
    b.snapElapsed = 0;
    b.margin = rp.moveSpeed * (bt.reactionMs / 1000);
    b.lowHp = p.hp < p.hpMax * rp.lowHpThreshold;

    const farm = b.policy.farm;
    const foe = pickFoe(world);
    const foeIsBoss = foe !== null && (foe.isBoss || foe.midBossId !== '');
    // ★ 보스/중간보스가 표적이면 픽업을 좇지 않는다 — 실측: 보스전에 픽업을 좇다 열을 벗어나
    //   보스전 명중률이 ≈1% 로 붕괴했다. 보스전엔 사격선을 지킨다.
    let pick = (farm === 'passive' || foeIsBoss) ? null : nearestPickup(world);
    // ★ 먼 픽업은 좇지 않는다(maxFarm 제외) — 실측: 화력선을 벗어나 픽업을 좇으면 모브 명중 시간이
    //   줄어 레벨이 안 오른다. 자석(90px)이 근처는 자동 수거하므로 «가까운» 픽업만 살짝 우회한다.
    if (pick !== null && farm !== 'maxFarm') {
      const pdx = pick.x - p.x;
      const pdy = pick.y - p.y;
      if (pdx * pdx + pdy * pdy > T.PICK_MAX_DIST * T.PICK_MAX_DIST) pick = null;
    }
    // 안전하거나 maxFarm 이면 픽업을 목적지로
    if (pick !== null && (farm === 'maxFarm' || p.hp > p.hpMax * 0.5)) {
      b.hasPick = true; b.pickX = pick.x; b.pickY = pick.y;
      b.tgtIdx = -1; b.tgtGen = -1; b.tgtBoss = false;
    } else {
      b.hasPick = false;
      if (foe !== null) {
        b.tgtIdx = foe.idx; b.tgtGen = foe.gen;
        b.tgtBoss = foe.isBoss || foe.midBossId !== '';
        b.fallbackX = foe.x;
      } else {
        b.tgtIdx = -1; b.tgtGen = -1; b.tgtBoss = false;
        b.fallbackX = (world.bounds.minX + world.bounds.maxX) * 0.5;
      }
    }
    // §10.4.1 aimErrorPx — 손 오차(창 동안 고정)
    b.aimJx = (world.rng.bot.f() * 2 - 1) * bt.aimErrorPx;
    b.aimJy = (world.rng.bot.f() * 2 - 1) * bt.aimErrorPx;

    snapshotThreats(world, b);
    b.wantStance = desiredStance(world, foe);
  }

  // ── 연속 조향 (매 틱) ──────────────────────────────────────────────────────
  steer(world, b);
  const DEAD = T.DEAD;
  inp.left = _steer.x < -DEAD;
  inp.right = _steer.x > DEAD;
  inp.up = _steer.y < -DEAD;
  inp.down = _steer.y > DEAD;

  b.snapElapsed += dt;

  // ── 스탠스 — stanceSwitchMs 간격으로만 전환 ──────────────────────────────
  b.stanceT -= dt;
  if (b.stanceT <= 0 && b.wantStance !== p.stance) {
    b.stanceT = bt.stanceSwitchMs / 1000;
    if (b.wantStance === 'normal') inp.stanceNormal = true;
    else if (b.wantStance === 'fire') inp.stanceFire = true;
    else if (b.wantStance === 'water') inp.stanceWater = true;
    else if (b.wantStance === 'grass') inp.stanceGrass = true;
  }

  return inp;
}

/**
 * §10.4.1 draft — 어떤 카드를 고르는가(인덱스). 결정 불가 시 0.
 */
export function botDraftPick(world, draft) {
  const b = ensureBot(world);
  const cards = draft.cards;
  if (cards.length === 0) return 0;
  if (b.policy.draft === 'random') return Math.floor(world.rng.bot.f() * cards.length);

  // §9.5(v1.5) — 진화 준비: Lv≥6 무기의 «짝 패시브»가 부족하면 그 패시브 카드를 최우선으로 집는다.
  //   진화가 후반 화력의 핵심이므로 어떤 정책이든(무투자 정책도) 이 콤보는 노린다 — 진화 게이트의 전제.
  for (let si = 0; si < world.slots.length; si += 1) {
    const s = world.slots[si];
    if (s.weaponId === null || s.level < 6) continue;
    const req = world.weaponDefs[s.family].evolution.requiresPassive;
    let lv = 0;
    for (let j = 0; j < world.passives.length; j += 1) {
      if (world.passives[j].id === req.id) { lv = world.passives[j].level; break; }
    }
    if (lv >= req.level) continue;
    for (let i = 0; i < cards.length; i += 1) {
      if (cards[i].category === 'passive' && cards[i].passiveId === req.id) return i;
    }
  }

  const noElement = b.policy.forceNoElement;
  let order;
  if (b.policy.draft === 'weaponRush') order = ['newWeapon', 'weaponLevel', 'passive', 'elementLevel'];
  else if (b.policy.draft === 'elementRush' || b.policy.draft === 'specialist') {
    order = ['elementLevel', 'newWeapon', 'weaponLevel', 'passive'];
  } else if (b.policy.draft === 'greedyDps') order = ['weaponLevel', 'newWeapon', 'passive', 'elementLevel'];
  else order = ['newWeapon', 'weaponLevel', 'elementLevel', 'passive'];   // generalist (§13.5.1 사다리)

  for (let k = 0; k < order.length; k += 1) {
    if (noElement && order[k] === 'elementLevel') continue;
    for (let i = 0; i < cards.length; i += 1) if (cards[i].category === order[k]) return i;
  }
  return 0;
}

/** §10.4.1 shop — 정책대로 산다. */
export function botShopPlan(world) {
  const b = ensureBot(world);
  if (b.policy.shop === 'survivalFirst') return ['potion', 'shield', 'defense', 'maxhp'];
  if (b.policy.shop === 'thrifty') return ['defense', 'maxhp'];
  return ['potion', 'shield', 'defense', 'maxhp', 'movespeed', 'magnet', 'resist', 'bomb', 'reroll', 'timeToken'];
}

/** thrifty 가 남겨야 하는 최소 잔액(컨티뉴 값). 다른 정책은 0. */
export function botShopReserve(world) {
  const b = ensureBot(world);
  return b.policy.shop === 'thrifty' ? world.data.meta.flow.continueCost : 0;
}
