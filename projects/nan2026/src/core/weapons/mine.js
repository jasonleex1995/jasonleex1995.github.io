/**
 * src/core/weapons/mine.js — 마인필드 (§9.5)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표):
 *   base            : dmg placeIntervalSec armSec triggerRadius blastRadius maxAlive blockHp
 *   evolution.params: evoClusterCount evoClusterRadius evoSecondaryDmgMul
 *
 * ★ §9.5(v1.5, 사용자 결정 2026-08-01) — 마인 = «탄 막는 설치물». 발밑에 자동 설치되어:
 *   ① 탄 막기 — blastRadius 안의 적 탄을 막고(소거) 그때마다 blockHp(z.hp)를 1 깎는다.
 *      hp 0 = «소멸»(다 막고 닳아 사라짐 — 폭발 아님).
 *   ② 폭발 — 적 «기체»가 triggerRadius 에 닿으면 터진다(blastRadius 광역 피해). hp 와 무관.
 *   즉 탄에는 방패로 닳고, 적에는 지뢰로 터진다. v1.5(#24)의 «탄막 제거+둔화»는 폐기.
 *
 * ★ 기뢰는 zones 풀의 **플레이어 장판**(fromPlayer)이다. step.hazards 는 나이만 먹이고 수명·반납은
 *   **이 파일이 소유**한다(§12.1). z.radius=triggerRadius(접촉) · z.dmg=폭발피해 · z.hp=탄막이체력 · z.age=무장경과
 * ★ 진화(클러스터 마인): 폭발 자리에서 evoClusterCount 발을 원주에 균등 배치해 2차 폭발(결정적, RNG 0).
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

  // ── 설치 — placeIntervalSec 마다, maxAlive 까지. blockHp(탄막이 체력) 세팅 ──
  slot.a0 -= dt;
  if (slot.a0 <= 0) {
    slot.a0 += eff.placeIntervalSec;
    if (liveMines(world) < eff.maxAlive) {
      const z = spawnZone(world, world.player.x, world.player.y, eff.triggerRadius, eff.dmg, eff.armSec, true);
      if (z !== null) z.hp = eff.blockHp;
    }
  }

  const zs = world.zones.items;
  const en = world.enemies.items;
  const eb = world.enemyBullets.items;
  const br2 = eff.blastRadius * eff.blastRadius;
  for (let i = 0; i < zs.length; i += 1) {
    const z = zs[i];
    if (!z.alive || !z.fromPlayer) continue;
    if (z.age < eff.armSec) continue;                 // 아직 무장 전

    // ① 탄 막기 — blastRadius 안의 적 탄을 막고(소거) hp 를 깎는다. hp 0 = «소멸»(폭발 아님)
    for (let j = 0; j < eb.length && z.hp > 0; j += 1) {
      const b = eb[j];
      if (!b.alive) continue;
      // 잡몹 탄만 막는다. 보스('boss')·중간보스('mb…') 탄은 관통(무력화 방지, §9.5 v1.5.1과 동일 방침).
      if (b.srcArch === 'boss' || b.srcArch.startsWith('mb')) continue;
      const dx = b.x - z.x;
      const dy = b.y - z.y;
      if (dx * dx + dy * dy <= br2) { world.enemyBullets.release(b); z.hp -= 1; }
    }
    if (z.hp <= 0) { world.zones.release(z); continue; }   // 탄 다 막고 닳아 소멸(폭발 없음)

    // ② 폭발 — 적 «기체»가 triggerRadius 에 닿으면 터진다(광역 피해). hp 와 무관.
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
