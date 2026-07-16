/**
 * src/core/step.js
 *
 * 정본 v1.4 구현 절:
 *   §10.1  고정 타임스텝 — TICK_DT = 1/60 게임초. ★ dt 는 **인자**다. core 는 speed 를 모른다
 *   §5.7   매 고정 틱마다 키 상태를 **폴링**한다 (이벤트 큐 아님 — 헤드리스 재현성의 전제)
 *   §2.2   이동 — 관성 없음(moveResponseTau = 0), 대각선 정규화, SOCD = lastInput
 *   §2.3   히트박스   §2.4 i-frame (모든 피해원 공유, 게임초당 최대 1회)
 *   §2.5   몸통 충돌   §2.7 상태이상 (스턴 중에도 자동발사·스탠스 전환은 유효)
 *   §3.1 · §3.2  데미지
 *   §4.3 · §4.4  스탠스 부여 · 각인
 *   §6.4   드래프트 발생 — core 는 큐에 세기만 한다. 소화는 상태 기계의 몫
 *   §8.6   밴드 코인 드랍 · 엘리트   §9.6 패시브 (reactive · afterimage · xpGain · coinGain)
 *   §10.3  L2 — 인덱스 오름차순 순회만 · 핫패스 0 alloc · Map/Set 순회 없음
 *   §12.1  캡 초과 정책
 *   §9.1   core 순수성
 *
 * ★ 이 파일에 배속이 없다. `main.js` 의 `tickDur` 가 이 함수를 몇 번 부를지 정한다 (§6.1).
 * ★ tools/sim.mjs 는 이 파일을 그대로 import 한다 — 렌더·오디오·DOM 은 애초에 없다 (§10.4).
 */

import { playerToEnemy, enemyToPlayer } from './damage.js';
import { recomputeEff, spawnPickup, xpToNext } from './state.js';
import { tickStance, requestStance, stampFor } from './stance.js';

/** §10.1 — 잠금. 배속과 무관한 상수 */
export const TICK_HZ = 60;
export const TICK_DT = 1 / 60;

const NORMAL = 'normal';
/** §2.2 — diagonalNormalize: true (×0.70710678). 정본이 인쇄한 상수 그대로 */
const DIAG = 0.70710678;

/**
 * §5.7 — 폴링된 키 상태 1틱분. **이벤트가 아니라 상태다.**
 * 봇(§10.2 bot 스트림)도 이 모양을 만들어 넘긴다 → 시뮬과 게임이 같은 입구를 쓴다.
 */
export function makeInput() {
  return { left: false, right: false, up: false, down: false,
    stanceNormal: false, stanceFire: false, stanceWater: false, stanceGrass: false };
}

/**
 * ★ 고정 dt 1틱.
 * @param world  createWorld() 가 만든 월드
 * @param input  이번 틱의 키 상태 스냅샷 (makeInput() 모양)
 * @param dt     게임초. 호출자가 TICK_DT 를 넘긴다. ★ 여기서 speed 를 곱하지 않는다
 */
export function step(world, input, dt) {
  if (world.over) return;

  world.tick += 1;
  world.time += dt;
  world.player.hit = false;

  readInput(world, input, dt);      // 1. 입력 스냅샷
  movePlayer(world, dt);            // 2. 이동
  fireWeapons(world, dt);           // 3. 무기 발사
  if (world.hooks.enemies !== null) world.hooks.enemies(world, dt);
  if (world.hooks.emitters !== null) world.hooks.emitters(world, dt);
  moveBullets(world, dt);           // 4. 탄 이동
  collide(world, dt);               // 5. 충돌
  pickups(world, dt);               // 6. 픽업
  levelUps(world);                  // 7. XP / 레벨
}

// ---------------------------------------------------------------------------
// 1. 입력 스냅샷 (§5.7 · §2.2 SOCD)
// ---------------------------------------------------------------------------
function readInput(world, input, dt) {
  const p = world.player;
  const prev = world.prevInput;

  // §2.2 — SOCD "lastInput": 반대키 동시 입력이면 **마지막에 눌린 키**가 이긴다.
  //   폴링 모델이므로 "눌린 순간" = 직전 틱 대비 상승 엣지다 (§5.5의 edgeTrigger 와 같은 기제).
  if (input.left && !prev.left) p.lastHorizontal = -1;
  if (input.right && !prev.right) p.lastHorizontal = 1;
  if (input.up && !prev.up) p.lastVertical = -1;
  if (input.down && !prev.down) p.lastVertical = 1;

  let dx = 0;
  if (input.left && input.right) dx = p.lastHorizontal;
  else if (input.left) dx = -1;
  else if (input.right) dx = 1;

  let dy = 0;
  if (input.up && input.down) dy = p.lastVertical;
  else if (input.up) dy = -1;
  else if (input.down) dy = 1;

  p.dirX = dx;
  p.dirY = dy;

  // §2.7 — 스탠스 전환은 **스턴 중에도 유효하다** ("스턴은 위치를 잠그는 것이지 판단을 잠그는 것이 아니다")
  // §5.7(v1.4) — 스탠스 키 동시 입력 → 마지막에 눌린 키.
  //   ★ 한 틱에 둘 이상이 동시에 상승 엣지면 Q<W<E<R 순 스캔으로 마지막(R 우선)을 채택한다
  //     (§4.1 키 순서의 결정적 사상 — 정본이 이 순서를 확정했다).
  let want = null;
  if (input.stanceNormal && !prev.stanceNormal) want = NORMAL;
  if (input.stanceFire && !prev.stanceFire) want = 'fire';
  if (input.stanceWater && !prev.stanceWater) want = 'water';
  if (input.stanceGrass && !prev.stanceGrass) want = 'grass';
  if (want !== null) requestStance(world, want);
  tickStance(world, dt);

  prev.left = input.left; prev.right = input.right;
  prev.up = input.up; prev.down = input.down;
  prev.stanceNormal = input.stanceNormal; prev.stanceFire = input.stanceFire;
  prev.stanceWater = input.stanceWater; prev.stanceGrass = input.stanceGrass;

  // §2.4 · §2.7 — 타이머
  if (p.iframeSec > 0) { p.iframeSec -= dt; if (p.iframeSec < 0) p.iframeSec = 0; }
  if (p.slowSec > 0) { p.slowSec -= dt; if (p.slowSec < 0) p.slowSec = 0; }
  if (p.stunSec > 0) { p.stunSec -= dt; if (p.stunSec < 0) p.stunSec = 0; }
  if (p.ghostSec > 0) { p.ghostSec -= dt; if (p.ghostSec < 0) p.ghostSec = 0; }
}

// ---------------------------------------------------------------------------
// 2. 이동 (§2.2 — 관성 없음. 지수 스무딩 항은 존재하고 기본값이 0이다)
// ---------------------------------------------------------------------------
function movePlayer(world, dt) {
  const p = world.player;
  const rp = world.data.rules.player;
  const b = world.bounds;

  let dx = p.dirX;
  let dy = p.dirY;
  if (p.stunSec > 0) { dx = 0; dy = 0; }          // §2.7 — 스턴 중 이동 입력 무시

  // §2.2 파생 상한: moveSpeed × (1 + 상점 %) × (1 + 패시브 moveSpeedMul)
  let v = rp.moveSpeed * (1 + world.shopMoveSpeedPct) * (1 + world.stats.moveSpeedMul);
  if (p.slowSec > 0) v *= world.data.rules.status.slowMoveSpeedMul;   // §2.7 — 강도는 불변

  if (rp.diagonalNormalize && dx !== 0 && dy !== 0) { dx *= DIAG; dy *= DIAG; }

  const tvx = dx * v;
  const tvy = dy * v;
  const tau = rp.moveResponseTau;
  if (tau > 0) {
    // ★ 항은 존재하고 값이 0이다 — "살짝 미끄럽게"가 필요해도 숫자만 바뀐다 (§2.2 · C-4)
    const k = 1 - Math.exp(-dt / tau);
    p.vx += (tvx - p.vx) * k;
    p.vy += (tvy - p.vy) * k;
  } else {
    p.vx = tvx;                                   // 가속/감속 없음, 즉시 정지
    p.vy = tvy;
  }

  p.x += p.vx * dt;
  p.y += p.vy * dt;
  if (p.x < b.minX) p.x = b.minX;
  if (p.x > b.maxX) p.x = b.maxX;
  if (p.y < b.minY) p.y = b.minY;
  if (p.y > b.maxY) p.y = b.maxY;
}

// ---------------------------------------------------------------------------
// 3. 무기 발사 (§2.7 — 자동발사는 스턴 중에도 계속된다)
// ---------------------------------------------------------------------------
function fireWeapons(world, dt) {
  const slots = world.slots;
  for (let i = 0; i < slots.length; i += 1) {     // 인덱스 오름차순 = 슬롯 순서 (§4.3 과 같은 순서)
    const s = slots[i];
    if (s.weaponId === null) continue;
    const fn = world.weaponFns[s.family];
    if (fn === undefined) {
      // §9.3 정신 — 폴백 금지. 조용히 안 쏘는 무기는 "밸런스가 조용히 드리프트하는" 바로 그 자리다
      throw new Error(`step: 무기 모듈 없음 "${s.family}" — src/core/weapons/${s.family}.js 를 레지스트리에 주입하라 (§9.5)`);
    }
    fn.update(world, s, recomputeEff(world, s), dt);
  }
}

/**
 * §9.5(v1.4) 무기 런타임 계약 D2 — 플레이어 탄 release 의 유일한 관문.
 *   (1) b.family 로 현재 슬롯을 해소한다 (world.slots 선형 탐색 · id==family 1:1 · 없으면 스킵)
 *   (2) 그 family 모듈에 onExpire 가 있으면 release **직전에 정확히 1회** 부른다
 *       onExpire(world, slot, recomputeEff(world, slot), bullet) — fan 진화 폭발 등.
 *   (3) playerBullets.release(b)
 * ★ 탄이 풀로 반환되기 **전에** onExpire 가 불리므로 LIFO 재사용 경합이 구조적으로 불가능하다
 *   (v1.3 fan.sweepExpired 의 스캔-마커 해킹을 대체). 모든 플레이어 탄 release 는 이 함수 경유.
 */
function releasePlayerBullet(world, b) {
  const fn = world.weaponFns[b.family];
  if (fn !== undefined && fn.onExpire !== undefined) {
    let slot = null;
    for (let i = 0; i < world.slots.length; i += 1) {
      if (world.slots[i].family === b.family) { slot = world.slots[i]; break; }
    }
    if (slot !== null) fn.onExpire(world, slot, recomputeEff(world, slot), b);
  }
  world.playerBullets.release(b);
}

// ---------------------------------------------------------------------------
// 4. 탄 이동
// ---------------------------------------------------------------------------
function moveBullets(world, dt) {
  const a = world.data.rules.view.arena;
  const pad = 64;

  const pb = world.playerBullets.items;
  for (let i = 0; i < pb.length; i += 1) {
    const b = pb[i];
    if (!b.alive) continue;
    b.x += b.vx * dt;
    b.y += b.vy * dt;
    b.age += dt;
    if (b.age >= b.lifetimeSec
        || b.x < a.x - pad || b.x > a.x + a.w + pad
        || b.y < a.y - pad || b.y > a.y + a.h + pad) {
      releasePlayerBullet(world, b);          // D2 — 수명 만료·화면 밖도 onExpire 경유
    }
  }

  const eb = world.enemyBullets.items;
  for (let i = 0; i < eb.length; i += 1) {
    const b = eb[i];
    if (!b.alive) continue;
    if (b.accel !== 0) {                          // §9.7 — 가속은 탄의 속성이다
      const sp = Math.sqrt(b.vx * b.vx + b.vy * b.vy);
      if (sp > 0) {
        const ns = sp + b.accel * dt;
        b.vx = (b.vx / sp) * ns;
        b.vy = (b.vy / sp) * ns;
      }
    }
    b.x += b.vx * dt;
    b.y += b.vy * dt;
    b.age += dt;
    if (b.x < a.x - pad || b.x > a.x + a.w + pad || b.y < a.y - pad || b.y > a.y + a.h + pad) {
      world.enemyBullets.release(b);
    }
  }

  // 적의 등속 적분. moveId 8종의 스크립트는 src/core/enemies.js 의 소관이며 vx/vy 를 쓴다 (§8.4)
  const en = world.enemies.items;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive) continue;
    if (e.stunSec > 0) { e.stunSec -= dt; continue; }
    let m = 1;
    if (e.slowSec > 0) { e.slowSec -= dt; m = world.data.rules.status.slowMoveSpeedMul; }
    e.x += e.vx * m * dt;
    e.y += e.vy * m * dt;
    e.moveT += dt;
    // §8.7 — 아레나를 벗어난 적은 보상을 몰수당한다 (enemyExitForfeitsReward)
    if (e.y > a.y + a.h + pad || e.x < a.x - pad * 2 || e.x > a.x + a.w + pad * 2) {
      world.enemies.release(e);
    }
  }
}

// ---------------------------------------------------------------------------
// 5. 충돌
//   ★ collide.gridCellPx(64) 의 균일 그리드는 src/core/collide.js 의 소관이다 (§9.1).
//     1주차 규모(적 ≤96 × 탄 ≤256)에서는 직접 순회가 등가이며 결정성도 동일하다.
// ---------------------------------------------------------------------------
function collide(world, dt) {
  const ctx = world.dmgCtx;
  ctx.matrix = world.data.elements.matrix;
  ctx.dmgMulSum = world.stats.dmgMul;                     // §3.1-2항 — 가산 풀
  ctx.elementBonusMul = world.stats.elementBonusMul;      // §3.1-3항 — resonance 의 k
  ctx.coreGateMul = world.data.rules.boss.coreGateMul;    // §3.1-4항

  const pb = world.playerBullets.items;
  const en = world.enemies.items;

  // (a) 플레이어 탄 → 적
  for (let i = 0; i < pb.length; i += 1) {
    const b = pb[i];
    if (!b.alive) continue;
    for (let j = 0; j < en.length; j += 1) {              // 인덱스 오름차순 (§10.3)
      const e = en[j];
      if (!e.alive) continue;
      const dx = e.x - b.x;
      const dy = e.y - b.y;
      const rr = e.radius + b.radius;
      if (dx * dx + dy * dy > rr * rr) continue;

      // §9.5 — hitCooldownSec: 같은 대상을 다시 때리기까지의 최소 간격.
      //        0.0 = 한 대상에 정확히 1회 (재히트 없음)
      // ★ hitGen — 히트 기록은 (hitEpoch, e.gen) 쌍으로 유효하다. 적 슬롯이 풀 재사용되면
      //   같은 idx 라도 e.gen 이 올라 다른 개체다 → 스탬프가 살아 있어도 "새 적"으로 취급해
      //   재히트 가드를 통과시킨다 (재사용 슬롯의 새 적을 관통탄이 조용히 무시하던 손실 차단).
      if (b.hitStamp[e.idx] === b.hitEpoch && b.hitGen[e.idx] === e.gen) {
        if (b.hitCooldownSec === 0) continue;
        if (world.time - b.hitAt[e.idx] < b.hitCooldownSec) continue;
      }
      b.hitStamp[e.idx] = b.hitEpoch;
      b.hitGen[e.idx] = e.gen;
      b.hitAt[e.idx] = world.time;

      // §4.4 — spawn 은 각인된 값, live(orbit·aura) 는 현재 스탠스를 재평가
      const stamp = stampFor(world, b.slot, b.stampMode, b.element);
      e.hp -= playerToEnemy(ctx, b.dmg, b.localMul, stamp, e);

      if (e.hp <= 0) { killEnemy(world, e); }

      // pierce: -1 = 무제한 (§9.6.1). 0 = 첫 히트에 소멸
      if (b.pierceLeft === -1) continue;
      if (b.pierceLeft > 0) { b.pierceLeft -= 1; continue; }
      releasePlayerBullet(world, b);          // D2 — 관통 소진도 onExpire 경유
      break;
    }
  }

  // (b) 적 탄 → 플레이어 (§2.3 히트박스 r = 4)
  const p = world.player;
  const rp = world.data.rules.player;
  const eb = world.enemyBullets.items;
  for (let i = 0; i < eb.length; i += 1) {
    const b = eb[i];
    if (!b.alive) continue;
    const dx = p.x - b.x;
    const dy = p.y - b.y;
    const rr = rp.hitboxRadius + b.hitRadius;
    if (dx * dx + dy * dy > rr * rr) continue;
    // §2.4(v1.4) — i-frame 중 통과하는 탄은 소멸하지 않는다. **피해를 실제로 준 그 탄 하나만**
    //   소멸(아래 release, 실드 흡수 포함). 광역 소거는 폭탄·hitBulletClearRadius 만.
    if (p.iframeSec > 0) continue;
    if (applyHit(world, b.dmg)) {
      if (b.status !== null) applyStatus(world, b.status, b.statusDurationSec);
      world.enemyBullets.release(b);
    }
  }

  // (c) 몸통 충돌 (§2.5 — 데미지 있음, i-frame 공유. 밀리지 않는다, 관통 통과)
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive) continue;
    if (p.iframeSec > 0) break;
    const dx = p.x - e.x;
    const dy = p.y - e.y;
    const rr = rp.hitboxRadius + e.radius;
    if (dx * dx + dy * dy > rr * rr) continue;
    applyHit(world, e.contactDmg);
    break;
  }
}

/**
 * §2.4 · §3.2 — 모든 피해원이 공유하는 단 하나의 게이트.
 * @returns 실제로 피해가 적용됐는가 (실드 흡수도 true — 피격 자체는 일어났다)
 */
function applyHit(world, raw) {
  const p = world.player;
  const rp = world.data.rules.player;
  if (p.iframeSec > 0) return false;              // 게임초당 최대 1회

  p.hit = true;
  p.iframeSec = rp.iframeSec;

  // §3.2 — 실드: taken 0 + 실드 −1 + i-frame 발동. 무피격 판정 유지 (§11.3)
  if (p.shields > 0) {
    p.shields -= 1;
  } else {
    p.hp -= enemyToPlayer(rp, p, raw);
    if (p.hp <= 0) { p.hp = 0; world.over = true; }
  }

  // §9.6 — afterimage: 피격 시 N초간 적의 조준·유도 대상에서 제외
  if (world.stats.ghostSecOnHit > 0) p.ghostSec = world.stats.ghostSecOnHit;
  // §9.6 — reactive: 피격 시 반경 N px 의 적 탄 소거
  const r = world.stats.hitBulletClearRadius;
  if (r > 0) {
    const eb = world.enemyBullets.items;
    for (let i = 0; i < eb.length; i += 1) {
      const b = eb[i];
      if (!b.alive) continue;
      const dx = b.x - p.x;
      const dy = b.y - p.y;
      if (dx * dx + dy * dy <= r * r) world.enemyBullets.release(b);
    }
  }
  return true;
}

/** §2.7 — stackMode "refresh": 중첩 금지. 잔여 = max(잔여, 신규 × (1 − statusResist)) */
function applyStatus(world, status, durSec) {
  const p = world.player;
  const d = durSec * (1 - p.statusResist);        // resistAffects = "duration". 강도(0.55)는 불변
  if (status === 'slow') { if (d > p.slowSec) p.slowSec = d; return; }
  if (status === 'stun') { if (d > p.stunSec) p.stunSec = d; return; }
  throw new Error(`step: 미지의 상태이상 "${status}" (§9.7)`);
}

/**
 * §8.6 — 처치 보상. chaff·line 은 코인 0 (이것이 LOCKED "일부 잡몹만"의 정의다).
 * §9.5(v1.4) 무기 런타임 계약 D3 — **step.js 가 export.** 탄 충돌 밖에서 피해를 주는
 *   무기 모듈(fan 진화 폭발 등)이 import 해서 `if (e.hp <= 0) killEnemy(world, e)` 로 부른다.
 * ★ 진입 시 `if (!e.alive) return` 으로 **멱등** — 같은 틱에 두 피해원이 부르면 두 번째는 무해.
 * ★ S11 안전: world.rng.drop 텍스트가 이 파일(step.js)에 잔류하므로 weapons 파일 스캔에 안 걸림.
 */
export function killEnemy(world, e) {
  if (!e.alive) return;                                             // D3 멱등 가드
  spawnPickup(world, 'xp', e.xp, e.x, e.y);
  const band = world.data.enemies.bands[e.band];
  const el = world.data.rules.elite;
  // §2.1(v1.4) — 회복 픽업의 회복량 = healPickupPct × 드랍 순간 hpMax. 절대량을 value 로 싣는다
  //   (상점 potion.healPct 결속 해제 · 죽은 인자 제거).
  const healValue = world.data.rules.player.healPickupPct * world.player.hpMax;
  if (e.elite) {
    spawnPickup(world, 'coin', el.coin, e.x, e.y);                   // 확정 드랍
    if (world.rng.drop.f() < el.healDropChance) spawnPickup(world, 'heal', healValue, e.x, e.y);
  } else if (band.coinDropChance > 0 && world.rng.drop.f() < band.coinDropChance) {
    spawnPickup(world, 'coin', band.coin, e.x, e.y);
  }
  world.enemies.release(e);
}

// ---------------------------------------------------------------------------
// 6. 픽업 (§2.6 magnetRadius · §12.1 merge)
// ---------------------------------------------------------------------------
function pickups(world, dt) {
  const p = world.player;
  const rp = world.data.rules.player;
  const mag = rp.magnetRadius * (1 + world.shopMagnetPct);
  // §2.6(v1.4) — 픽업 회수 2단계: magnetRadius 자석 → 획득 반경 = spriteRadius 접촉.
  //   둘 다 bounds/스프라이트에서 파생되는 구조 규칙(리터럴 아님) → 새 키 없음.
  const grab = rp.spriteRadius;
  const items = world.pickups.items;

  for (let i = 0; i < items.length; i += 1) {
    const q = items[i];
    if (!q.alive) continue;
    const dx = p.x - q.x;
    const dy = p.y - q.y;
    const d2 = dx * dx + dy * dy;
    if (!q.magnet && d2 <= mag * mag) q.magnet = true;
    if (q.magnet) {
      const d = Math.sqrt(d2);
      if (d > 0) {
        const v = rp.moveSpeed;                    // 자석은 플레이어보다 느리지 않아야 회수가 성립한다
        q.x += (dx / d) * v * dt;
        q.y += (dy / d) * v * dt;
      }
    }
    if (d2 <= grab * grab) {
      collect(world, q);
      world.pickups.release(q);
    }
  }
}

function collect(world, q) {
  const p = world.player;
  if (q.kind === 'xp') { p.xp += q.value * (1 + world.stats.xpGainMul); return; }      // §9.6 study
  if (q.kind === 'coin') { p.coins += q.value * (1 + world.stats.coinGainMul); return; } // §9.6 salvage
  if (q.kind === 'heal') {
    // §2.1(v1.4) — value 는 이미 절대 회복량(killEnemy 가 healPickupPct×hpMax 로 실었다).
    //   다른 픽업 value 와 같은 의미(절대량) → hp += value. 상점 결속·죽은 인자 없음.
    p.hp += q.value;
    if (p.hp > p.hpMax) p.hp = p.hpMax;
    return;
  }
  throw new Error(`step: 미지의 픽업 "${q.kind}"`);
}

// ---------------------------------------------------------------------------
// 7. XP / 레벨 (§6.4 — core 는 큐에 세기만 한다. 소화는 상태 기계의 몫)
// ---------------------------------------------------------------------------
function levelUps(world) {
  const p = world.player;
  // §6.4 — 동시 다중 레벨업 = 병합 없이 **순차** 드래프트 (xp.levelUpQueueMode = "serial")
  while (p.xp >= p.xpToNext) {
    p.xp -= p.xpToNext;
    p.level += 1;
    p.xpToNext = xpToNext(world, p.level);
    world.draftQueue += 1;
  }
}
