/**
 * src/core/weapons/nova.js — 노바 (§9.5)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표):
 *   base            : dmg intervalSec radius expandSec telegraphSec
 *   evolution.params: evoRing2Radius evoClearBullets evoSecondaryDmgMul
 *
 * §9.6.1 훅은 state.recomputeEff 가 이미 적용했다:
 *   rateKey "intervalSec" (H1) · countKey **null** · pierceApplies false
 *   areaKeys ["radius", "evoRing2Radius"] — 2단 링도 areaMul 을 받는다(진화 시에만 존재)
 *
 * ★ expandSec·telegraphSec 는 **연출의 시간**이다. 판정은 §8.5 와 같은 「적용 1회」이며 폭발 시점에
 *   한 번 적용한다 — 확장 애니메이션은 렌더의 몫이고 core 의 판정을 나누지 않는다.
 * ★ 진화(슈퍼노바)는 2단 링(evoRing2Radius, dmg × evoSecondaryDmgMul)과 **적 탄 소거**를 더한다.
 *
 * 슬롯 스크래치: a0 = 다음 폭발까지 남은 시간
 */

import { hitEnemy } from '../damage.js';
import { stampFor } from '../stance.js';
import { killEnemy } from '../step.js';

/** §3.1 의 컨텍스트. ★ 모듈 스코프 1회 (§10.3) */
const ctx = { matrix: null, dmgMulSum: 0, elementBonusMul: 1, coreGateMul: 0 };

/** 반경 r 안의 적에게 dmg × localMul 을 1회 적용한다. */
function ring(world, slot, eff, r, localMul, stamp) {
  const p = world.player;
  const en = world.enemies.items;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive) continue;
    const dx = e.x - p.x;
    const dy = e.y - p.y;
    if (dx * dx + dy * dy > r * r) continue;
    const dealt = hitEnemy(world, ctx, slot.family, eff.dmg, localMul, stamp, e);
    if (dealt > 0 && e.hp <= 0) killEnemy(world, e);
  }
}

function detonate(world, slot, eff) {
  ctx.matrix = world.data.elements.matrix;
  ctx.dmgMulSum = world.stats.dmgMul;
  ctx.elementBonusMul = world.stats.elementBonusMul;
  ctx.coreGateMul = world.data.rules.boss.coreGateMul;

  const stamp = stampFor(world, slot.index, 'spawn', slot.stampElement);
  ring(world, slot, eff, eff.radius, 1, stamp);

  // ★ slot.evolved 분기 정확히 1개 (§9.5) — 2단 링 + 적 탄 소거
  if (slot.evolved) {
    ring(world, slot, eff, eff.evoRing2Radius, eff.evoSecondaryDmgMul, stamp);
    if (eff.evoClearBullets) {
      const p = world.player;
      const r = eff.evoRing2Radius;
      const eb = world.enemyBullets.items;
      for (let i = 0; i < eb.length; i += 1) {
        const b = eb[i];
        if (!b.alive) continue;
        const dx = b.x - p.x;
        const dy = b.y - p.y;
        if (dx * dx + dy * dy <= r * r) world.enemyBullets.release(b);
      }
    }
  }
}

export function update(world, slot, eff, dt) {
  slot.a0 -= dt;
  if (slot.a0 > 0) return;
  slot.a0 += eff.intervalSec;
  detonate(world, slot, eff);
}

export default { update };
