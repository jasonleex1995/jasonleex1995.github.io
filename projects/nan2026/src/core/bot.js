/**
 * src/core/bot.js — 결정적 AI 플레이어 (순수 core, §10.2 · §10.4.1)
 *
 * 정본 v1.4 구현 절:
 *   §10.2  **8번째 RNG 스트림 `bot` 의 거처.** 스트림은 독립이므로 봇의 추첨이 theme/draft/spawn/…
 *          시퀀스를 흔들지 않는다 → 같은 마스터 시드에서 게임과 시뮬이 같은 콘텐츠를 본다.
 *   §10.4.1 정책 4축(draft × farm × stance × shop) + 난이도는 **봇의 반응 지연으로만** 들어온다:
 *          latencyTicks = round(reactionMs/1000 × TICK_HZ × speed).
 *          → core 는 전 난이도에서 게임시간 기준 **동일**하고, 달라지는 것은 봇의 눈 감는 창이다.
 *   §5.7   봇의 출력 = makeInput() 모양(불리언 상태). **게임과 같은 입구**로 들어간다.
 *   §9.1   순수성 — window/Date/Math.random 0. rng 는 world.rng.bot 만.
 *   §10.3  봇 상태는 world.bot 에 최초 1회만 alloc(ensureX 관용구). 핫패스 0 alloc.
 *
 * ★ 스탠스 키는 **상승 엣지**로 소비된다(step.readInput) → 전환하고 싶은 틱에만 true 를 낸다.
 */

import { TICK_HZ } from './step.js';
import { elementMul } from './elements.js';

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
      forceNoElement: false,      // §10.4 probes — 속성 투자 금지(noElementPass 측정용)
    },
    input: { left: false, right: false, up: false, down: false,
      stanceNormal: false, stanceFire: false, stanceWater: false, stanceGrass: false },
    decideT: 0,                   // 다음 재결정까지 남은 게임초(= 반응 지연)
    tgtX: 0, tgtY: 0,             // 이번 결정의 목표 위치
    stanceT: 0,                   // 스탠스 전환 쿨다운(게임초)
    wantStance: 'normal',
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

/** 가장 가까운 살아있는 적. 없으면 null. */
function nearestEnemy(world) {
  const en = world.enemies.items;
  const p = world.player;
  let best = null;
  let bestD = 0;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive) continue;
    // ★ §8.9 「선택적」 — 베이스라인 봇은 **무시하는 쪽을 선택한다.** 중간보스를 조준 대상에서
    //   빼면 그 아래에 주차하지 않는다(회피는 그대로 그 장판·빔을 위협으로 본다).
    //   실측: 이것을 안 하면 «출처 불명»(중간보스 zone/laser)이 전 사인의 최대 항목이 된다 —
    //   즉 봇이 「선택」을 모르면 게임이 실제보다 훨씬 잔인해 보인다.
    if (e.midBossId !== '') continue;
    const dx = e.x - p.x;
    const dy = e.y - p.y;
    const d = dx * dx + dy * dy;
    if (best === null || d < bestD) { bestD = d; best = e; }
  }
  return best;
}

/**
 * §10.4.1 — 회피: dodgeLookaheadSec 안에 나에게 가장 가까이 접근하는 적 탄을 찾아 **그 접근점의 반대**로.
 *   반환 = 위협이 있으면 회피 방향(정규화), 없으면 null. 결정적(순수 기하).
 */
const _dodge = { x: 0, y: 0 };
function dodgeVector(world) {
  const p = world.player;
  const eb = world.enemyBullets.items;
  const bt = world.data.meta.bot;
  const rp = world.data.rules.player;
  const look = bt.dodgeLookaheadSec;
  // ★ 여유는 지어낸 상수가 아니라 **반응 시간 동안 내가 움직일 수 있는 거리**다 —
  //   이보다 늦게 알아채면 원리적으로 피할 수 없다(latency 가 난이도인 이유, §10.4.1).
  const margin = rp.moveSpeed * (bt.reactionMs / 1000);
  let ax = 0;
  let ay = 0;
  let threats = 0;

  for (let i = 0; i < eb.length; i += 1) {
    const b = eb[i];
    if (!b.alive) continue;
    // 상대 위치·속도로 최근접 시각 t* 를 구한다(0..look 로 클램프)
    const rx = b.x - p.x;
    const ry = b.y - p.y;
    const vv = b.vx * b.vx + b.vy * b.vy;
    let t = vv > 0 ? -(rx * b.vx + ry * b.vy) / vv : 0;
    if (t < 0) t = 0;
    if (t > look) t = look;
    const cx = rx + b.vx * t;
    const cy = ry + b.vy * t;
    const d2 = cx * cx + cy * cy;
    const danger = rp.hitboxRadius + b.hitRadius + margin;
    if (d2 > danger * danger) continue;                // 이 탄은 나를 스치지 않는다
    // ★ 위협 **전부**를 1/거리 가중으로 합산한다 → 합벡터가 자연히 «빈 공간»을 가리킨다
    const d = Math.sqrt(d2);
    const w = (danger - d) / danger;                   // 가까울수록 크게
    if (d > 0) { ax -= (cx / d) * w; ay -= (cy / d) * w; }
    else { ay -= w; }
    threats += 1;
  }
  // ★ 장판(zone)과 빔(laser)도 위협이다 — 실측: 이것을 안 보면 «출처 불명»(장판·빔) 피해가
  //   전 사인의 최대 항목이 된다(중간보스의 zone/laser). 탄만 피하는 봇은 사람의 하한이 아니다.
  const zs = world.zones.items;
  for (let i = 0; i < zs.length; i += 1) {
    const z = zs[i];
    if (!z.alive || z.fromPlayer) continue;            // 플레이어 장판(기뢰)은 위협이 아니다
    const rx = z.x - p.x;
    const ry = z.y - p.y;
    const d2 = rx * rx + ry * ry;
    const danger = rp.hitboxRadius + z.radius + margin;
    if (d2 > danger * danger) continue;
    const d = Math.sqrt(d2);
    const w = (danger - d) / danger;
    if (d > 0) { ax -= (rx / d) * w; ay -= (ry / d) * w; }
    else { ay -= w; }
    threats += 1;
  }
  const ts = world.telegraphs.items;
  for (let i = 0; i < ts.length; i += 1) {
    const t2 = ts[i];
    if (!t2.alive || t2.kind !== 'laser') continue;
    // 빔은 선이다 — 선까지의 수직거리로 위험을 잰다(진행각 a, 폭 r)
    const rx = p.x - t2.x;
    const ry = p.y - t2.y;
    const perp = Math.abs(-Math.sin(t2.a) * rx + Math.cos(t2.a) * ry);
    const danger = rp.hitboxRadius + t2.r * 0.5 + margin;
    if (perp > danger) continue;
    const w = (danger - perp) / danger;
    const side = (-Math.sin(t2.a) * rx + Math.cos(t2.a) * ry) >= 0 ? 1 : -1;
    ax += -Math.sin(t2.a) * side * w;                  // 빔의 «옆으로» 벗어난다
    ay += Math.cos(t2.a) * side * w;
    threats += 1;
  }
  if (threats === 0) return null;

  // 벽에 몰리지 않도록 경계에서 밀어낸다(구석에서 갇혀 맞는 것이 가장 흔한 사인)
  const bd = world.bounds;
  const wall = margin;
  if (p.x - bd.minX < wall) ax += (wall - (p.x - bd.minX)) / wall;
  if (bd.maxX - p.x < wall) ax -= (wall - (bd.maxX - p.x)) / wall;
  if (p.y - bd.minY < wall) ay += (wall - (p.y - bd.minY)) / wall;
  if (bd.maxY - p.y < wall) ay -= (wall - (bd.maxY - p.y)) / wall;

  const len = Math.sqrt(ax * ax + ay * ay);
  if (len > 0) { _dodge.x = ax / len; _dodge.y = ay / len; }
  else { _dodge.x = 0; _dodge.y = -1; }                // 완전 대칭이면 위로 뺀다
  return _dodge;
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

/**
 * §10.4.1 stance — 어떤 스탠스를 원하는가.
 *   greedyNearest    : 최근접 적을 ×2 로 때리는 속성
 *   majorityOnScreen : 화면 다수 속성을 ×2 로 때리는 속성
 *   static           : 전환하지 않는다(노말 고정) — stanceValue 게이트의 대조군
 */
function desiredStance(world) {
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
    const e = nearestEnemy(world);
    if (e !== null) targetElement = e.element;
  }
  if (targetElement === null) return world.player.stance;

  // 그 적을 ×2 로 때리는 투자 가능 속성을 고른다(없으면 현 상태 유지)
  for (let i = 0; i < investable.length; i += 1) {
    if (elementMul(matrix, investable[i], targetElement) > 1) return investable[i];
  }
  return world.player.stance;
}

/**
 * ★ 봇의 한 틱 입력. step(world, botInput(world, dt), dt) 로 쓰인다.
 *   반응 지연 창 동안은 **직전 결정을 유지**한다(눈 감는 창의 모델).
 */
export function botInput(world, dt) {
  const b = ensureBot(world);
  const p = world.player;
  const bt = world.data.meta.bot;
  const rp0 = world.data.rules.player;
  const bounds = world.bounds;
  const inp = b.input;

  // 스탠스 키는 매 틱 내린다 — 전환하고 싶은 틱에만 올려 상승 엣지를 만든다
  inp.stanceNormal = false; inp.stanceFire = false; inp.stanceWater = false; inp.stanceGrass = false;

  // ── 재결정 (반응 지연마다) ────────────────────────────────────────────────
  b.decideT -= dt;
  if (b.decideT <= 0) {
    b.decideT = reactionSec(world);

    // ★ 의도(사격선·파밍)를 먼저 세우고, 회피를 **그 위에 더한다**.
    //   회귀(실측): 회피가 의도를 «대체»하면 화면에 적이 많을수록 봇이 영원히 도망만 다닌다 —
    //   스테이지1 120초에서 명중이 잠재 화력의 20%에 그쳐 처치율 21%가 나왔다. 사람은 피하면서도
    //   총구를 맞춘다. 그래서 회피는 **변위**이지 목적지가 아니다.
    const farm = b.policy.farm;
    const foe = nearestEnemy(world);
    const pick = farm === 'passive' ? null : nearestPickup(world);
    if (pick !== null && (farm === 'maxFarm' || world.player.hp > world.player.hpMax * 0.5)) {
      b.tgtX = pick.x; b.tgtY = pick.y;                // 안전하면(또는 maxFarm) 주우러 간다
    } else {
      b.tgtX = foe === null ? (bounds.minX + bounds.maxX) / 2 : foe.x;   // 사격선을 맞춘다
      b.tgtY = bounds.maxY;                            // 기본은 하단(§2.6 과 같은 안전 위치)
    }
    const dodge = dodgeVector(world);
    if (dodge !== null) {
      // 비켜서는 거리 = «내다보는 시간 동안 실제로 갈 수 있는 거리». 새 데이터 키를 만들지 않고
      // dodgeLookaheadSec × 현재 이동속도로 유도한다(빠를수록 크게 비킨다 = 사람의 감각과 같다).
      const rp = world.data.rules.player;
      const disp = bt.dodgeLookaheadSec * rp.moveSpeed * (1 + world.stats.moveSpeedMul);
      b.tgtX += dodge.x * disp;                        // 위협 반대로 «비켜서되» 목적지는 유지한다
      b.tgtY += dodge.y * disp;
    }
    // ★★ 사격선 유지 — **표적보다 위로 올라가지 않는다.**
    //   회귀(실측): 파밍과 회피가 겹치면 봇이 아레나 최상단(y=56)까지 올라가 보스(y=170)보다
    //   **위에서 위로** 쐈다 → 보스전 실효 DPS 1.3 (목표 49). 이 게임의 주 무기는 전부 위로 나간다
    //   (§1.1) — 표적 아래에 있는 것은 «전술»이 아니라 **사격의 전제**다.
    //   ★ 단, **자리를 지키는 표적**(보스·중간보스)에만 건다. 내려오는 잡몹에까지 걸면 봇이
    //     계속 물러나며 요격을 포기한다 — 실측으로 처치율이 39% → 24% 로 떨어졌다.
    if (foe !== null && (foe.isBoss || foe.midBossId !== '')) {
      const stand = rp0.hitboxRadius + foe.radius + rp0.moveSpeed * (bt.reactionMs / 1000);
      if (b.tgtY < foe.y + stand) b.tgtY = foe.y + stand;
    }
    // §10.4.1 aimErrorPx — 사람의 손 오차
    b.tgtX += (world.rng.bot.f() * 2 - 1) * bt.aimErrorPx;
    b.tgtY += (world.rng.bot.f() * 2 - 1) * bt.aimErrorPx;

    b.wantStance = desiredStance(world);
  }

  // ── 이동 — 목표로 향하는 4방향 불리언 ─────────────────────────────────────
  const dead = 4;                                      // 목표 근처에서 떨림 방지
  inp.left = b.tgtX < p.x - dead;
  inp.right = b.tgtX > p.x + dead;
  inp.up = b.tgtY < p.y - dead;
  inp.down = b.tgtY > p.y + dead;

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
 *   generalist  : 무기 슬롯을 먼저 채우고(newWeapon) → 무기 레벨 → 패시브 → 속성
 *   weaponRush  : 무기(신규·레벨) 최우선
 *   elementRush : 속성 최우선
 *   specialist  : 속성 최우선이되 두 축만 (elementRush 와 같은 선호, 투자 축은 stance 가 결정)
 *   greedyDps   : 무기 레벨 > 신규 무기 > 패시브 > 속성
 *   random      : rng.bot 균등
 */
export function botDraftPick(world, draft) {
  const b = ensureBot(world);
  const cards = draft.cards;
  if (cards.length === 0) return 0;
  if (b.policy.draft === 'random') return Math.floor(world.rng.bot.f() * cards.length);

  const noElement = b.policy.forceNoElement;
  let order;
  if (b.policy.draft === 'weaponRush') order = ['newWeapon', 'weaponLevel', 'passive', 'elementLevel'];
  else if (b.policy.draft === 'elementRush' || b.policy.draft === 'specialist') {
    order = ['elementLevel', 'newWeapon', 'weaponLevel', 'passive'];
  } else if (b.policy.draft === 'greedyDps') order = ['weaponLevel', 'newWeapon', 'passive', 'elementLevel'];
  else order = ['newWeapon', 'weaponLevel', 'passive', 'elementLevel'];   // generalist

  for (let k = 0; k < order.length; k += 1) {
    if (noElement && order[k] === 'elementLevel') continue;               // 속성 투자 금지 프로브
    for (let i = 0; i < cards.length; i += 1) if (cards[i].category === order[k]) return i;
  }
  // 전부 걸러졌으면(예: 속성 카드만 남았는데 금지) 첫 장
  return 0;
}

/**
 * §10.4.1 shop — 정책대로 산다. 살 수 있는 것이 없을 때까지 반복한다.
 *   survivalFirst : potion → shield → defense → maxhp 순, 그 뒤 저축
 *   thrifty       : 컨티뉴 값(continueCost)은 남기고 그 위로만
 *   spender       : 살 수 있으면 순서대로 전부
 * ★ 구매 자체는 shop.js 가 한다 — 여기선 **무엇을 살지**만 정한다(정책과 규칙의 분리).
 */
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
