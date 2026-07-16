/**
 * src/core/state.js
 *
 * 정본 v1.4 구현 절:
 *   §1.1   좌표 — 논리 1280×720 · arena {350,0,580,720} · playerBoundsInset {56,56,20,20}
 *   §2.1   체력 · 방어  §2.3 히트박스  §2.4 i-frame  §2.6 런 시작 상태  §2.7 상태이상
 *   §4.2   투자 (fire/water/grass)   §4.3 부여
 *   §9.5   무기 슬롯 = 패밀리 계약. levels[] 부분 오버라이드 누적 (§9.3의 유일한 예외)
 *   §9.6   패시브 12훅 1:1 — 가산 풀
 *   §9.6.1 passiveHooks — src = base ∪ (evolved ? evolution.params : {}) + H1~H4
 *   §10.2  RNG 8 스트림 주입   §10.3 L2 유지 규칙 — 사전할당 풀 + alive 플래그,
 *                              인덱스 오름차순 순회만, **core 내 객체 생성 금지(핫패스 0 bytes/tick)**
 *   §12.1  2층 캡 — 풀 크기 = rules.caps (B층 안전망)
 *   §9.1   core 순수성
 *
 * ★ 모든 풀은 여기서 **한 번** 할당된다. step() 은 아무것도 new 하지 않는다.
 * ★ 마스터 시드와 무기 모듈 레지스트리는 **주입**된다 — core 는 시계도 파일도 모른다.
 */

import { makeStreams } from './rng.js';
import { recomputeStamps, NORMAL } from './stance.js';

// ---------------------------------------------------------------------------
// 사전할당 풀 (§10.3)
// ---------------------------------------------------------------------------
/**
 * alive 플래그 + free 스택. 순회는 언제나 인덱스 오름차순이며,
 * alloc/release 는 호출 순서만의 함수다 → 결정적.
 */
function makePool(size, factory) {
  const items = new Array(size);
  const free = new Int32Array(size);
  for (let i = 0; i < size; i += 1) {
    const it = factory(i);
    it.alive = false;
    it.idx = i;
    it.gen = 0;
    items[i] = it;
    free[i] = size - 1 - i;      // pop 순서가 0, 1, 2 … 가 되도록
  }
  return {
    items,
    size,
    freeTop: size,
    live: 0,
    /** 여유가 없으면 null — 초과 정책(§12.1)은 호출자가 소유한다 */
    alloc() {
      if (this.freeTop === 0) return null;
      this.freeTop -= 1;
      const it = this.items[free[this.freeTop]];
      it.alive = true;
      it.gen += 1;
      this.live += 1;
      return it;
    },
    release(it) {
      if (!it.alive) return;
      it.alive = false;
      this.freeTop += 1;
      free[this.freeTop - 1] = it.idx;
      this.live -= 1;
    },
  };
}

// ---------------------------------------------------------------------------
// 엔티티 팩토리 — 필드는 여기서 전부 만들어진다 (히든 클래스 고정 + 0 alloc/tick)
// ---------------------------------------------------------------------------
function makeEnemy() {
  return {
    alive: false, idx: 0, gen: 0,
    archetypeId: '', band: '', element: NORMAL,
    x: 0, y: 0, vx: 0, vy: 0,
    hp: 0, hpMax: 0, radius: 0,
    contactDmg: 0, xp: 0, score: 0, coin: 0,
    elite: false,
    // §3.1-4항 — 잡몹은 코어가 아니다. 보스 코어가 이 풀을 쓰게 되면 여기서 켠다
    isCore: false, aliveArmorPartCount: 0,
    // §2.7 — 상태이상은 플레이어 전용이지만 구조는 대칭으로 둔다 (오라 진화의 끌어당김 등)
    slowSec: 0, stunSec: 0,
    // 이미터 스케줄 (emitters.js 소관 — 자리만 예약)
    emitT: 0, emitPhase: 0,
    moveT: 0, mp0: 0, mp1: 0, mp2: 0,
  };
}

function makePlayerBullet(capEnemies) {
  return {
    alive: false, idx: 0, gen: 0,
    slot: 0, family: '',
    x: 0, y: 0, vx: 0, vy: 0,
    dmg: 0, localMul: 1, radius: 0,
    element: NORMAL, stampMode: 'spawn',
    pierceLeft: 0, hitCooldownSec: 0,
    age: 0, lifetimeSec: 0,
    // §9.5 — 관통·재히트의 정확한 표현. 적 슬롯별 마지막 히트 시각.
    //   hitEpoch 로 세대를 구분하므로 스폰 때 배열을 지울 필요가 없다 (0 alloc/tick).
    hitEpoch: 0,
    hitStamp: new Int32Array(capEnemies),
    hitAt: new Float64Array(capEnemies),
    // ★ hitGen — 적 슬롯이 풀 재사용(release → 새 적 alloc, gen++)되면 같은 idx 라도
    //   다른 개체다. 관통탄의 재히트 가드가 (hitStamp==hitEpoch && hitGen==e.gen)여야
    //   재사용된 슬롯의 새 적을 조용히 통과하지 않는다 (seeker.targetGen 방어와 대칭).
    hitGen: new Int32Array(capEnemies),
    // 패밀리별 스크래치 (부메랑 왕복 · 시커 타겟 등)
    s0: 0, s1: 0, s2: 0, target: -1, targetGen: -1,
  };
}

function makeEnemyBullet() {
  return {
    alive: false, idx: 0, gen: 0,
    bulletId: '',
    x: 0, y: 0, vx: 0, vy: 0,
    dmg: 0, radius: 0, hitRadius: 0,
    // §4.1 — 적 탄에 element 가 없다. 스키마가 이미 그것을 강제한다 (§9.7)
    status: null, statusDurationSec: 0,
    accel: 0, turnRateDegSec: 0, retargetSec: 0, retargetT: 0,
    age: 0,
  };
}

function makePickup() {
  return { alive: false, idx: 0, gen: 0, kind: '', value: 0, x: 0, y: 0, vx: 0, vy: 0, magnet: false };
}

function makeZone() {
  return { alive: false, idx: 0, gen: 0, x: 0, y: 0, radius: 0, dmg: 0, activeSec: 0, age: 0, fromPlayer: false };
}

function makeDrone() {
  return { alive: false, idx: 0, gen: 0, slot: 0, x: 0, y: 0, ox: 0, oy: 0, fireT: 0 };
}

function makeTelegraph() {
  return { alive: false, idx: 0, gen: 0, kind: '', x: 0, y: 0, r: 0, a: 0, age: 0, durSec: 0, owner: -1 };
}

// ---------------------------------------------------------------------------
// §9.6 — 패시브 12훅. 각 스탯의 소유자는 정확히 1개 패시브다 (1:1)
// ---------------------------------------------------------------------------
/**
 * §3.1 · §9.6 — 훅의 기본값.
 *   *Mul / *Add : 0 (가산 풀의 항등원. dmgMul 은 §3.1-2항에서 1 + Σ 가 된다)
 *   elementBonusMul : ★ 1.0 (§3.1 — 이것만 곱의 항등원이다. k 이지 가산항이 아니다)
 */
function makeStats() {
  return {
    dmgMul: 0, fireRateMul: 0, areaMul: 0, pierceAdd: 0, projCountAdd: 0,
    elementBonusMul: 1, ghostSecOnHit: 0, hitBulletClearRadius: 0,
    maxHpAdd: 0, moveSpeedMul: 0, xpGainMul: 0, coinGainMul: 0,
  };
}

/** 보유 패시브 → 스탯 캐시. 패시브 변경 시에만 호출한다 */
export function recomputeStats(world) {
  const st = world.stats;
  const fresh = makeStats();
  const keys = Object.keys(fresh);
  for (let i = 0; i < keys.length; i += 1) st[keys[i]] = fresh[keys[i]];
  const list = world.data.passives.passives;
  for (let i = 0; i < world.passives.length; i += 1) {
    const p = world.passives[i];
    if (p.id === null) continue;
    let def = null;
    for (let j = 0; j < list.length; j += 1) if (list[j].id === p.id) { def = list[j]; break; }
    if (def === null) throw new Error(`state: 미지의 패시브 "${p.id}" (§9.6)`);
    const v = def.values[p.level - 1];   // §9.6 — values 는 각 레벨의 **절대 총량**이지 증분이 아니다
    if (def.stat === 'elementBonusMul') st.elementBonusMul = v;   // ★ k 는 대입이지 합산이 아니다
    else st[def.stat] += v;
  }
  // §2.1(v1.4) — maxHpAdd(패시브 bulkhead)와 상점 maxhp 는 hpMax 를 직접 바꾸고,
  //   **모든 hpMax 증가는 그 증가분만큼 hp 를 채운다**(단일 규칙 — 델타만 회복).
  const base = world.data.rules.player.hpMax;
  const prevMax = world.player.hpMax;
  world.player.hpMax = base + world.shopHpAdd + st.maxHpAdd;
  if (world.player.hpMax > prevMax) world.player.hp += world.player.hpMax - prevMax;
  if (world.player.hp > world.player.hpMax) world.player.hp = world.player.hpMax;
  for (let i = 0; i < world.slots.length; i += 1) world.slots[i].effDirty = true;
}

// ---------------------------------------------------------------------------
// §9.5 · §9.6.1 — 무기 슬롯의 유효 파라미터
// ---------------------------------------------------------------------------
function makeSlot(index) {
  return {
    index,
    weaponId: null, family: null,
    level: 0, evolved: false,
    stampElement: NORMAL,
    cooldownT: 0,
    effDirty: true,
    eff: {},          // 재사용. 매 틱 새로 만들지 않는다
    // 패밀리별 지속 상태 (오빗 각도 · 오버드라이브 램프 · 마인 배치 타이머 …)
    a0: 0, a1: 0, a2: 0, a3: 0,
  };
}

/** weapons[].levels[0..level-1] 을 base 에 순서대로 덮는다 (§9.3의 유일한 부분 오버라이드 예외) */
function resolveLevels(def, level, out) {
  const keys = Object.keys(def.base);
  for (let i = 0; i < keys.length; i += 1) out[keys[i]] = def.base[keys[i]];
  for (let L = 0; L < level; L += 1) {
    const row = def.levels[L];
    const rk = Object.keys(row);
    for (let i = 0; i < rk.length; i += 1) out[rk[i]] = row[rk[i]];
  }
  return out;
}

/**
 * ★ §9.6.1 — 슬롯의 유효 파라미터를 계산해 slot.eff 에 **제자리로** 쓴다.
 *
 *   src = resolveLevels(base, level) ∪ (w.evolved ? evolution.params : {})
 *   fireRateMul  : eff[rateKey]  = src[rateKey] / (1 + v)
 *   areaMul      : eff[areaKey]  = src[areaKey] × (1 + v)     // areaKeys 중 src 에 있는 것 전부
 *   pierceAdd    : eff.pierce    = src.pierce + v             // pierceApplies == false 면 무효
 *   projCountAdd : eff[countKey] = src[countKey] + v          // countKey == null 이면 무효 (H4)
 *   H3           : projRadius 는 render.playerBulletMaxRadiusPx 로 클램프 (판정·렌더 동시)
 *
 * ★ §9.6.1(v1.4) 확정 — 여기서 `src` 는 **`resolveLevels(base, level)` 로 레벨 오버라이드가
 *   적용된 유효 파라미터 집합**에 evolution.params 를 합집합한 것이다. 적용 순서 =
 *   resolveLevels → ∪evolution.params → H1~H4. 「인쇄된 base 블록 그대로 읽으면
 *   levels[].pierce·count 가 증발한다」던 문면 결함(예: seeker Lv5 count·pierce)을
 *   정본이 base.* → src.* 로 고쳐 닫았다. 코드(eff = 그 src)는 이미 정합.
 */
export function recomputeEff(world, slot) {
  if (!slot.effDirty) return slot.eff;
  const eff = slot.eff;
  for (const k of Object.keys(eff)) delete eff[k];
  if (slot.weaponId === null) { slot.effDirty = false; return eff; }

  const def = world.weaponDefs[slot.family];
  resolveLevels(def, slot.level, eff);

  // src = base ∪ (evolved ? evolution.params : {})
  if (slot.evolved) {
    const ep = def.evolution.params;
    const ek = Object.keys(ep);
    for (let i = 0; i < ek.length; i += 1) eff[ek[i]] = ep[ek[i]];
  }

  const hooks = world.data.rules.passiveHooks[slot.family];
  const st = world.stats;

  // H1 — fireRateMul 은 12 패밀리 전부에 적용된다. 주기(간격)이므로 나눗셈
  eff[hooks.rateKey] = eff[hooks.rateKey] / (1 + st.fireRateMul);

  // H2 — areaMul 은 "닿는 범위"만. 산포(spreadDeg·jitterDeg·arcDeg)는 areaKeys 에 없다
  for (let i = 0; i < hooks.areaKeys.length; i += 1) {
    const k = hooks.areaKeys[i];
    if (Object.prototype.hasOwnProperty.call(eff, k)) eff[k] = eff[k] * (1 + st.areaMul);
  }

  // pierceAdd — pierceApplies == false 면 무효. pierce: -1(무제한)에는 적용되지 않는다
  if (hooks.pierceApplies && eff.pierce !== -1) eff.pierce += st.pierceAdd;

  // H4 — countKey == null 이면 그 패시브는 그 무기에 무효다 (aura · nova · drone)
  if (hooks.countKey !== null) eff[hooks.countKey] += st.projCountAdd;

  // H3 — projRadius 클램프. 판정 반경과 렌더 반경을 **동시에** (I-2)
  const maxR = world.data.rules.render.playerBulletMaxRadiusPx;
  if (Object.prototype.hasOwnProperty.call(eff, 'projRadius') && eff.projRadius > maxR) {
    eff.projRadius = maxR;
  }

  slot.effDirty = false;
  return eff;
}

// ---------------------------------------------------------------------------
// 월드 생성
// ---------------------------------------------------------------------------
/**
 * @param opts.data     schema.validate() 를 통과한 9파일
 * @param opts.seed     uint32 마스터 시드. ★ core 바깥(main.js)에서 생성해 주입한다 (§10.2)
 * @param opts.weapons  { [family]: { update(world, slot, eff, dt, api) } }
 *                      — src/core/weapons/** 의 12 update 함수 레지스트리 (§9.5).
 *                      ★ 주입 이유: 정본 §9.1 은 파일 배치만 확정하고 **합성 계약을 인쇄하지 않았다**.
 *                        주입이면 core 밖 import 0 을 유지하면서 1주차에 2~3개만 꽂을 수 있다.
 * @param opts.hooks    { enemies, emitters } — 각각 (world, dt) 를 받는 함수 또는 null.
 *                      1주차에는 null 이며 step() 은 적의 등속 적분만 한다.
 */
export function createWorld(opts) {
  const data = opts.data;
  const rules = data.rules;
  const caps = rules.caps;
  const rp = rules.player;

  // §1.1 — 이동 가능 영역 (파생: 540 × 608 @ (370, 56))
  const a = rules.view.arena;
  const ins = rules.view.playerBoundsInset;
  const bounds = {
    minX: a.x + ins.left, maxX: a.x + a.w - ins.right,
    minY: a.y + ins.top, maxY: a.y + a.h - ins.bottom,
  };

  // §4.2 — 투자축은 elements.investable 이 소유한다. 이 파일에 "fire" 를 박지 않는다
  const invest = {};
  for (let i = 0; i < data.elements.investable.length; i += 1) invest[data.elements.investable[i]] = 0;

  // 패밀리 → 정의 (id == family, §9.5)
  const weaponDefs = {};
  for (let i = 0; i < data.weapons.weapons.length; i += 1) {
    const w = data.weapons.weapons[i];
    weaponDefs[w.family] = w;
  }

  const world = {
    data,
    seed: opts.seed >>> 0,
    rng: makeStreams(opts.seed),
    weaponFns: opts.weapons,
    // §9.1 이 파일 배치(enemies.js · emitters.js)만 확정하고 **합성 계약을 인쇄하지 않았다** →
    // 주입으로 둔다. 1주차에는 둘 다 null 이며 step() 은 적의 등속 적분만 한다.
    hooks: {
      enemies: opts.hooks === undefined || opts.hooks.enemies === undefined ? null : opts.hooks.enemies,
      emitters: opts.hooks === undefined || opts.hooks.emitters === undefined ? null : opts.hooks.emitters,
    },
    weaponDefs,
    bounds,

    tick: 0,
    time: 0,          // §0.2 — 게임초. 배속은 core 밖(main.js 의 tickDur)에만 있다

    player: {
      x: (bounds.minX + bounds.maxX) / 2,   // §2.6(v1.4) — 이동 가능 영역 하단 중앙 = (minX+maxX)/2, maxY
      y: bounds.maxY,                       //   bounds 파생(리터럴 아님) → 새 키 없음. 마커 해소됨
      vx: 0, vy: 0,
      hp: rp.hpMax, hpMax: rp.hpMax,
      defense: rp.defenseBase,
      statusResist: 0,                      // §11.2 상점 resist 누적 (상한 0.60). §2.7 resistAffects="duration"
      iframeSec: 0,
      stance: rp.startStance,               // §2.6 — 노말
      stanceCooldown: 0,
      invest,                               // §2.6 — fire 0 / water 0 / grass 0 (§4.2 investable)
      level: 1, xp: 0, xpToNext: 0,
      coins: 0,
      bombs: rules.bomb.stockStart,
      shields: 0, tokens: 0, rerolls: 0,
      slowSec: 0, stunSec: 0, ghostSec: 0,
      dirX: 0, dirY: 0,
      lastHorizontal: 0, lastVertical: 0,   // §2.2 SOCD = lastInput
      hit: false,                           // 이번 틱에 피격했는가 (렌더/점수용)
    },
    shopHpAdd: 0,        // §11.2 maxhp 구매분. core 는 상점을 모르지만 hpMax 의 합에는 참여한다
    shopMoveSpeedPct: 0,
    shopMagnetPct: 0,

    // §5.7 — 직전 틱의 키 상태. SOCD(lastInput)와 상승 엣지 판정의 유일한 근거. 재사용(0 alloc)
    prevInput: { left: false, right: false, up: false, down: false,
      stanceNormal: false, stanceFire: false, stanceWater: false, stanceGrass: false },

    // §3.1 — 데미지 컨텍스트. 매 틱 재사용한다 (핫패스 0 alloc)
    dmgCtx: { matrix: null, dmgMulSum: 0, elementBonusMul: 1, coreGateMul: 0 },

    slots: new Array(rp.weaponSlots),
    passives: new Array(rp.passiveSlots),
    stats: makeStats(),

    enemies: makePool(caps.enemies, makeEnemy),
    playerBullets: makePool(caps.playerBullets, () => makePlayerBullet(caps.enemies)),
    enemyBullets: makePool(caps.enemyBullets, makeEnemyBullet),
    pickups: makePool(caps.pickups, makePickup),
    zones: makePool(caps.zones, makeZone),
    drones: makePool(caps.drones, makeDrone),
    telegraphs: makePool(caps.telegraphs, makeTelegraph),

    // §6.4 — 레벨업 드래프트 큐. 소화는 호출자(상태 기계)의 몫이며 core 는 세기만 한다
    draftQueue: 0,
    draftsSeen: 0,
    elementPity: 0,      // §11.1 elementCardPity — 속성 카드가 "등장"하지 않은 연속 드래프트 수
    autoEquipDone: false, // §9.9 onboarding.autoEquipFirstElement — 투자 0→1 최초 전이에서만

    // §13.1.1 capHits — A층/B층 defer·reject 발화 카운터 (시뮬 게이트가 읽는다)
    capHits: { playerBullet: 0, enemyBullet: 0, enemy: 0, pickup: 0, zone: 0, drone: 0, telegraph: 0 },

    over: false,
  };

  for (let i = 0; i < rp.weaponSlots; i += 1) world.slots[i] = makeSlot(i);
  for (let i = 0; i < rp.passiveSlots; i += 1) world.passives[i] = { id: null, level: 0 };

  // §2.6 — forward Lv1 → 슬롯 1 고정
  giveWeapon(world, rp.startWeaponId);
  recomputeStats(world);
  recomputeStamps(world);
  world.player.xpToNext = xpToNext(world, 1);
  return world;
}

// ---------------------------------------------------------------------------
// 성장
// ---------------------------------------------------------------------------
/** §11.1 · §5.3 — 새 무기는 "가장 앞의 빈 슬롯에 자동 배치" (draft.slotAssign = "append") */
export function giveWeapon(world, weaponId) {
  const def = world.weaponDefs[weaponId];
  if (def === undefined) throw new Error(`state: 미지의 무기 "${weaponId}" (§9.5 — id == family)`);
  for (let i = 0; i < world.slots.length; i += 1) {
    const s = world.slots[i];
    if (s.weaponId !== null) continue;
    s.weaponId = def.id;
    s.family = def.family;
    s.level = 1;
    s.evolved = false;
    s.cooldownT = 0;
    s.effDirty = true;
    recomputeStamps(world);   // §4.3 재계산 시점 ② — 새 무기 획득
    return i;
  }
  return -1;                  // §11.1 — 만석이면 newWeapon 카드가 애초에 풀에 없다
}

/** §9.5 — Lv7 → Lv8 레벨업 카드 그 자체가 진화 카드다 */
export function levelUpWeapon(world, slotIndex) {
  const s = world.slots[slotIndex];
  if (s.weaponId === null) throw new Error(`state: 빈 슬롯 ${slotIndex} 의 레벨업 (§9.5)`);
  if (s.level >= 8) return false;   // Lv8 에서 종료. Lv9 없음
  s.level += 1;
  if (s.level === 8) s.evolved = true;
  s.effDirty = true;
  return true;
}

/** §9.6 — 획득과 레벨업이 같은 passive 카테고리 */
export function givePassive(world, passiveId) {
  const maxLevel = world.data.passives.maxLevel;
  for (let i = 0; i < world.passives.length; i += 1) {
    if (world.passives[i].id === passiveId) {
      if (world.passives[i].level >= maxLevel) return false;
      world.passives[i].level += 1;
      recomputeStats(world);
      return true;
    }
  }
  for (let i = 0; i < world.passives.length; i += 1) {
    if (world.passives[i].id !== null) continue;
    world.passives[i].id = passiveId;
    world.passives[i].level = 1;
    recomputeStats(world);
    return true;
  }
  return false;               // 6칸 만석 + 미보유 → 카드가 풀에 없다 (§11.1)
}

/** §5.3 — 슬롯 재정렬(스왑). 드래프트 화면에서만 호출된다 */
export function swapSlots(world, i, j) {
  const t = world.slots[i];
  world.slots[i] = world.slots[j];
  world.slots[j] = t;
  world.slots[i].index = i;
  world.slots[j].index = j;
  recomputeStamps(world);     // §4.3 재계산 시점 ③ — 슬롯 재정렬
}

// ---------------------------------------------------------------------------
// 스폰 (§12.1 — 초과 정책은 caps.overflow 가 소유한다)
// ---------------------------------------------------------------------------
/**
 * §12.1 — playerBullet 초과 = "rejectSpawn". 오래된 것 재활용 절대 금지.
 * §4.4  — element 각인은 **생성 순간**에 일어난다 (spawn 모드). live 모드는 stampFor() 가 매 적용마다 재평가.
 * @returns 탄 또는 null (풀 만석)
 */
export function spawnPlayerBullet(world, slot, eff, x, y, vx, vy, localMul) {
  const b = world.playerBullets.alloc();
  if (b === null) { world.capHits.playerBullet += 1; return null; }
  b.slot = slot.index;
  b.family = slot.family;
  b.x = x; b.y = y; b.vx = vx; b.vy = vy;
  b.dmg = eff.dmg;
  b.localMul = localMul;
  b.radius = eff.projRadius;
  b.element = slot.stampElement;                                  // §4.4 spawn 각인
  b.stampMode = world.weaponDefs[slot.family].elementStampMode;
  b.pierceLeft = eff.pierce;
  b.hitCooldownSec = eff.hitCooldownSec;
  b.age = 0;
  b.lifetimeSec = eff.lifetimeSec;
  b.hitEpoch += 1;                                                // 히트 기록 세대 교체 = 배열 클리어 불필요
  b.s0 = 0; b.s1 = 0; b.s2 = 0; b.target = -1; b.targetGen = -1;
  return b;
}

/** §12.1 — enemy 초과 = "defer". 스포너가 다음 틱에 재시도한다 (웨이브가 공짜로 사라지지 않는다) */
export function spawnEnemy(world, archetypeId, element, x, y, hp, elite) {
  const e = world.enemies.alloc();
  if (e === null) { world.capHits.enemy += 1; return null; }
  const defs = world.data.enemies.archetypes;
  let def = null;
  for (let i = 0; i < defs.length; i += 1) if (defs[i].id === archetypeId) { def = defs[i]; break; }
  if (def === null) throw new Error(`state: 미지의 아키타입 "${archetypeId}" (§9.7)`);
  const band = world.data.enemies.bands[def.band];
  const el = world.data.rules.elite;

  e.archetypeId = def.id;
  e.band = def.band;
  e.element = element;                    // §8.6 — element 는 아키타입 필드가 아니다. 편성이 주입한다
  e.x = x; e.y = y; e.vx = 0; e.vy = 0;
  e.hp = elite ? hp * el.hpMult : hp;     // §8.6 — 엘리트 = 접두 플래그다. 별도 개체가 아니다
  e.hpMax = e.hp;
  e.radius = elite ? def.radius * el.sizeMult : def.radius;
  e.contactDmg = elite ? def.contactDmg * el.contactDmgMul : def.contactDmg;
  e.xp = elite ? def.xp * el.xpMult : def.xp;
  e.score = def.score;
  e.coin = elite ? el.coin : band.coin;
  e.elite = elite;
  e.isCore = false; e.aliveArmorPartCount = 0;
  e.slowSec = 0; e.stunSec = 0;
  e.emitT = 0; e.emitPhase = 0; e.moveT = 0;
  return e;
}

/** §12.1 — pickup 초과 = "merge": 신규 값을 최근접 기존 픽업에 합산 (★ 손실 0 = 무-노가다 기둥 보존) */
export function spawnPickup(world, kind, value, x, y) {
  const p = world.pickups.alloc();
  if (p === null) {
    world.capHits.pickup += 1;
    const items = world.pickups.items;
    let best = null;
    let bestD = Infinity;
    for (let i = 0; i < items.length; i += 1) {     // 인덱스 오름차순 — 동점이면 낮은 인덱스 (§10.3)
      const q = items[i];
      if (!q.alive || q.kind !== kind) continue;
      const dx = q.x - x;
      const dy = q.y - y;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = q; }
    }
    if (best !== null) best.value += value;
    return best;
  }
  p.kind = kind; p.value = value;
  p.x = x; p.y = y; p.vx = 0; p.vy = 0; p.magnet = false;
  return p;
}

/** §12.1 — enemyBullet 초과 = "rejectSpawn" */
export function spawnEnemyBullet(world, bulletId, x, y, vx, vy) {
  const b = world.enemyBullets.alloc();
  if (b === null) { world.capHits.enemyBullet += 1; return null; }
  const defs = world.data.bullets.bullets;
  let def = null;
  for (let i = 0; i < defs.length; i += 1) if (defs[i].id === bulletId) { def = defs[i]; break; }
  if (def === null) throw new Error(`state: 미지의 탄 "${bulletId}" (§9.7)`);
  b.bulletId = def.id;
  b.x = x; b.y = y; b.vx = vx; b.vy = vy;
  b.dmg = def.dmg;
  b.radius = def.radius;
  b.hitRadius = def.radius * def.hitboxScale;   // §2.3
  b.status = def.status;
  b.statusDurationSec = def.statusDurationSec;
  b.accel = def.accel;
  b.turnRateDegSec = def.turnRateDegSec;
  b.retargetSec = def.retargetSec;
  b.retargetT = 0;
  b.age = 0;
  return b;
}

/**
 * §9.9 meta.xp — curve "poly": 레벨 L → L+1 에 필요한 XP = base × L^exp.
 * ★ §9.9(v1.4) 확정 — xpToNext 는 float 다. **레벨별 반올림이 없다**: §13.5 의 누적 검산
 *   Σ_{L=1}^{53} 6·L^1.32 = 26,450(연속 합)이 이 독법을 강제한다. float 누산, 표시만 정수.
 */
export function xpToNext(world, level) {
  const xp = world.data.meta.xp;
  if (xp.curve !== 'poly') throw new Error(`state: 미지의 xp.curve "${xp.curve}" (§9.9)`);
  return xp.base * Math.pow(level, xp.exp);
}

export { makePool };
