/**
 * src/core/weapons/mine.js — 마인필드 (§9.5)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표):
 *   base            : dmg placeIntervalSec armSec triggerRadius blastRadius maxAlive
 *   evolution.params: evoClusterCount evoClusterRadius evoSecondaryDmgMul
 *
 * §9.6.1 훅은 state.recomputeEff 가 이미 적용했다:
 *   rateKey "placeIntervalSec" (H1) · countKey "maxAlive" (+projCountAdd) · pierceApplies false
 *   areaKeys ["blastRadius", "triggerRadius", "evoClusterRadius"]
 *
 * ★ 기뢰는 zones 풀의 **플레이어 장판**(fromPlayer)이다. step.hazards 는 나이만 먹이고
 *   수명·반납은 **이 파일이 소유**한다(§12.1 — 새 풀을 만들지 않는다).
 *     z.radius = triggerRadius (기폭 감지 반경) · z.dmg = 폭발 피해 · z.age = 무장 경과
 * ★ 진화(클러스터 마인): 폭발 자리에서 evoClusterCount 발을 evoClusterRadius 원주에 **균등 배치**해
 *   각각 2차 폭발(dmg × evoSecondaryDmgMul)을 낸다 — 각도가 균등이라 결정적이다(RNG 0).
 *
 * 슬롯 스크래치: a0 = 다음 설치까지 남은 시간
 */

import { spawnZone } from '../state.js';
import { hitEnemy } from '../damage.js';
import { stampFor } from '../stance.js';
import { killEnemy } from '../step.js';
import { TAU } from '../angle.js';

/** §3.1 의 컨텍스트. ★ 모듈 스코프 1회 (§10.3) */
const ctx = { matrix: null, dmgMulSum: 0, elementBonusMul: 1, coreGateMul: 0 };

/** (x,y) 반경 r 안의 적에게 dmg × localMul 을 1회 적용한다. */
function blastAt(world, family, eff, x, y, r, localMul, stamp) {
  const en = world.enemies.items;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive) continue;
    const dx = e.x - x;
    const dy = e.y - y;
    if (dx * dx + dy * dy > r * r) continue;
    const dealt = hitEnemy(world, ctx, family, eff.dmg, localMul, stamp, e);
    if (dealt > 0 && e.hp <= 0) killEnemy(world, e);
  }
}

/**
 * ★ §9.5(v1.5) — 기폭은 «판을 흔든다»: 폭발 반경 안의 적 탄을 지우고(탄막 제거) 잡몹을 둔화(이동 방해).
 *   보스는 둔화 면제(위치·이동은 보스 스크립트 소유). base·진화 공통.
 */
function disrupt(world, x, y, r) {
  const eb = world.enemyBullets.items;
  for (let i = 0; i < eb.length; i += 1) {
    const b = eb[i];
    if (!b.alive) continue;
    const dx = b.x - x;
    const dy = b.y - y;
    if (dx * dx + dy * dy <= r * r) world.enemyBullets.release(b);
  }
  const en = world.enemies.items;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive || e.isBoss) continue;
    const dx = e.x - x;
    const dy = e.y - y;
    if (dx * dx + dy * dy <= r * r && e.slowSec < 1) e.slowSec = 1;   // 1초 둔화(리터럴 1 허용)
  }
}

/** 이 슬롯의 살아있는 기뢰 수 */
function liveMines(world) {
  const it = world.zones.items;
  let n = 0;
  for (let i = 0; i < it.length; i += 1) if (it[i].alive && it[i].fromPlayer) n += 1;
  return n;
}

export function update(world, slot, eff, dt) {
  ctx.matrix = world.data.elements.matrix;
  ctx.dmgMulSum = world.stats.dmgMul;
  ctx.elementBonusMul = world.stats.elementBonusMul;
  ctx.coreGateMul = world.data.rules.boss.coreGateMul;
  const stamp = stampFor(world, slot.index, 'spawn', slot.stampElement);

  // ── 설치 — placeIntervalSec 마다, maxAlive 까지 ──────────────────────────
  slot.a0 -= dt;
  if (slot.a0 <= 0) {
    slot.a0 += eff.placeIntervalSec;
    if (liveMines(world) < eff.maxAlive) {
      spawnZone(world, world.player.x, world.player.y, eff.triggerRadius, eff.dmg, eff.armSec, true);
    }
  }

  // ── 기폭 — armSec 이 지난 기뢰의 triggerRadius 에 적이 들어오면 터진다 ──
  const zs = world.zones.items;
  const en = world.enemies.items;
  for (let i = 0; i < zs.length; i += 1) {
    const z = zs[i];
    if (!z.alive || !z.fromPlayer) continue;
    if (z.age < eff.armSec) continue;                 // 아직 무장 전

    let hit = false;
    for (let j = 0; j < en.length && !hit; j += 1) {
      const e = en[j];
      if (!e.alive) continue;
      const dx = e.x - z.x;
      const dy = e.y - z.y;
      const rr = z.radius + e.radius;
      if (dx * dx + dy * dy <= rr * rr) hit = true;
    }
    if (!hit) continue;

    blastAt(world, slot.family, eff, z.x, z.y, eff.blastRadius, 1, stamp);
    disrupt(world, z.x, z.y, eff.blastRadius);         // ★ v1.5 — 탄막 제거 + 잡몹 둔화
    // ★ slot.evolved 분기 정확히 1개 (§9.5) — 클러스터: 원주 균등 배치의 2차 폭발
    if (slot.evolved) {
      const n = eff.evoClusterCount;
      for (let k = 0; k < n; k += 1) {
        const a = (k / n) * TAU;
        blastAt(world, slot.family, eff, z.x + Math.cos(a) * eff.evoClusterRadius,
          z.y + Math.sin(a) * eff.evoClusterRadius, eff.blastRadius, eff.evoSecondaryDmgMul, stamp);
      }
    }
    world.zones.release(z);                            // 소유자가 반납한다
  }
}

export default { update };
