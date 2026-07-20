/**
 * src/core/weapons/lance.js — 랜스 (§9.5)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표):
 *   base            : dmg cooldownSec count pierce hitCooldownSec targetMode
 *                     beamWidthPx chargeSec rangePx
 *   evolution.params: evoFullHeight
 *
 * §9.6.1 훅은 state.recomputeEff 가 이미 적용했다:
 *   rateKey "cooldownSec" (H1) · countKey "count" (+projCountAdd) · pierceApplies true (+pierceAdd)
 *   areaKeys ["beamWidthPx", "rangePx"] — 폭도 사거리도 areaMul 을 받는다
 *
 * ★ 랜스는 **탄이 아니다**. 정면(화면 위쪽, §1.1)으로 뻗는 폭 beamWidthPx 의 선을 그어 그 안의 적을
 *   **가까운 순으로 pierce 마리**까지 꿴다. hitCooldownSec 0 = 한 발에 한 대상 1회.
 * ★ 진화(레일건) evoFullHeight — 사거리가 아레나 세로 전체가 되고 관통이 무제한이 된다.
 *
 * 슬롯 스크래치: a0 = 발사 주기 타이머(차지 구간 포함)
 *   a0 이 chargeSec 이하인 구간이 «차지»이며(렌더가 그 구간을 그린다), 0 에 닿는 순간 발사한다.
 */

import { playerToEnemy, noteDamage } from '../damage.js';
import { stampFor } from '../stance.js';
import { killEnemy } from '../step.js';

const FORWARD = 'forward';

/** §3.1 의 컨텍스트. ★ 모듈 스코프 1회 (§10.3) */
const ctx = { matrix: null, dmgMulSum: 0, elementBonusMul: 1, coreGateMul: 0 };

/**
 * 빔 하나. 중심 x = bx, 위로 length 만큼. 가까운 순(= y 가 큰 순)으로 limit 마리까지 적용한다.
 *   ★ 정렬 배열을 만들지 않는다(§10.3 0 alloc): «직전에 맞힌 y 보다 작은 것 중 최대 y» 를 반복해 고른다.
 */
function beam(world, slot, eff, bx, length, limit, stamp) {
  const p = world.player;
  const en = world.enemies.items;
  const halfW = eff.beamWidthPx * 0.5;
  const topY = p.y - length;
  let bound = p.y;                                   // 이 값보다 y 가 작은 적만 후보(가까운 순 진행)
  let hits = 0;

  while (hits < limit) {
    let best = null;
    let bestY = 0;
    for (let i = 0; i < en.length; i += 1) {
      const e = en[i];
      if (!e.alive) continue;
      if (e.y >= bound || e.y < topY) continue;      // 이미 지난 구간 / 사거리 밖
      if (e.x < bx - halfW - e.radius || e.x > bx + halfW + e.radius) continue;
      if (best === null || e.y > bestY) { bestY = e.y; best = e; }
    }
    if (best === null) return;
    bound = best.y;
    const dealt = playerToEnemy(ctx, eff.dmg, 1, stamp, best);
    best.hp -= dealt;
    noteDamage(world, slot.family, dealt);
    if (best.hp <= 0) killEnemy(world, best);
    hits += 1;
  }
}

function fire(world, slot, eff) {
  ctx.matrix = world.data.elements.matrix;
  ctx.dmgMulSum = world.stats.dmgMul;
  ctx.elementBonusMul = world.stats.elementBonusMul;
  ctx.coreGateMul = world.data.rules.boss.coreGateMul;

  const p = world.player;
  const stamp = stampFor(world, slot.index, 'spawn', slot.stampElement);
  const arena = world.data.rules.view.arena;

  let length = eff.rangePx;
  let limit = eff.pierce;
  // ★ slot.evolved 분기 정확히 1개 (§9.5) — 레일건: 아레나 세로 전체 + 무제한 관통
  if (slot.evolved && eff.evoFullHeight) { length = arena.h; limit = world.enemies.size; }

  const n = eff.count;
  // count 개의 평행 빔을 폭 간격으로 정면에 나란히 세운다(count 1 이면 정중앙)
  let bx = p.x - (n - 1) * eff.beamWidthPx * 0.5;
  for (let i = 0; i < n; i += 1) {
    beam(world, slot, eff, bx, length, limit, stamp);
    bx += eff.beamWidthPx;
  }
}

export function update(world, slot, eff, dt) {
  if (eff.targetMode !== FORWARD) {
    throw new Error(`lance: 미구현 targetMode "${eff.targetMode}" — weapons.json 은 forward 만 쓴다 (§9.5)`);
  }
  slot.a0 -= dt;
  if (slot.a0 > 0) return;                            // 차지 구간 포함(a0 ≤ chargeSec 이 «차지»)
  slot.a0 += eff.cooldownSec;
  fire(world, slot, eff);
}

export default { update };
