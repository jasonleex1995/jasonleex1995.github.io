/**
 * src/core/weapons/orbit.js — 오빗 (§9.5)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표):
 *   base            : dmg hitCooldownSec projRadius orbitRadius angularSpeedDegSec bodyCount
 *   evolution.params: evoBulletClearCooldownSec
 *
 * §9.6.1 훅은 state.recomputeEff 가 이미 적용했다:
 *   rateKey "hitCooldownSec" (H1) · countKey "bodyCount" (+projCountAdd) · pierceApplies false
 *   areaKeys ["orbitRadius", "projRadius"]
 *
 * ★ 표현의 선택 — 공전체는 **playerBullets 풀의 «정지한 탄»**이다. 이유: 그 풀만이 적 슬롯별
 *   재히트 기록(hitStamp/hitAt/hitGen)을 이미 들고 있고, §9.5 의 hitCooldownSec(대상별 재타격 간격)이
 *   정확히 그 기록을 요구한다. 새 풀도, collide 의 새 분기도 필요 없다.
 *   - 매 틱 위치를 궤도 위로 **직접** 세팅하고 age 를 0 으로 되돌린다 → 수명으로 사라지지 않는다.
 *   - pierceLeft = -1 (무제한) → 한 대상을 때려도 소멸하지 않는다.
 *   - stampMode 는 weaponDefs 가 'live' 를 주므로 collide 가 **적용 순간의 스탠스**를 재평가한다(§4.4).
 *
 * 슬롯 스크래치: a0 = 공전 각(rad) / a1 = 진화(이지스) 탄 소거 쿨다운
 */

import { spawnPlayerBullet } from '../state.js';
import { DEG2RAD, TAU } from '../angle.js';

/** 이 family 의 살아있는 공전체 수 */
function bodyCount(world, slot) {
  const it = world.playerBullets.items;
  let n = 0;
  for (let i = 0; i < it.length; i += 1) if (it[i].alive && it[i].family === slot.family) n += 1;
  return n;
}

/** 부족한 만큼 공전체를 만든다. 탄이지만 «날지 않는» 탄이다. */
function ensureBodies(world, slot, eff) {
  const p = world.player;
  let have = bodyCount(world, slot);
  while (have < eff.bodyCount) {
    const b = spawnPlayerBullet(world, slot, eff, p.x, p.y, 0, 0, 1);
    if (b === null) return;                       // §12.1 rejectSpawn — 다음 틱에 다시 시도
    b.pierceLeft = -1;                            // 무제한: 때려도 소멸하지 않는다
    b.lifetimeSec = eff.hitCooldownSec;           // age 를 매 틱 0 으로 되돌리므로 만료되지 않는다
    b.age = 0;
    have += 1;
  }
}

/** 궤도 위에 균등 배치하고 나이를 되돌린다(수명·이탈로 사라지지 않게). */
function place(world, slot, eff) {
  const p = world.player;
  const it = world.playerBullets.items;
  const n = eff.bodyCount;
  let k = 0;
  for (let i = 0; i < it.length; i += 1) {
    const b = it[i];
    if (!b.alive || b.family !== slot.family) continue;   // §5.3 family-키로만 식별
    const a = slot.a0 + (k / n) * TAU;
    b.x = p.x + Math.cos(a) * eff.orbitRadius;
    b.y = p.y + Math.sin(a) * eff.orbitRadius;
    b.vx = 0; b.vy = 0;
    b.age = 0;
    // ★ 공전체는 영속(수명 없음)이라 스폰 때 각인한 dmg/반경/쿨다운이 **레벨업해도 안 갱신**됐다 —
    //   기존 공전체가 Lv1 수치로 굳었다. 매 틱 현재 eff 로 다시 각인해 레벨 성장이 반영되게 한다.
    b.dmg = eff.dmg;
    b.radius = eff.projRadius;
    b.hitCooldownSec = eff.hitCooldownSec;
    k += 1;
  }
}

/** 진화(이지스) — 공전체에 닿은 적 탄을 지운다. 쿨다운이 «방패의 재사용 대기»다. */
function aegis(world, slot, eff) {
  const it = world.playerBullets.items;
  const eb = world.enemyBullets.items;
  for (let i = 0; i < it.length; i += 1) {
    const b = it[i];
    if (!b.alive || b.family !== slot.family) continue;
    for (let j = 0; j < eb.length; j += 1) {
      const g = eb[j];
      if (!g.alive) continue;
      const dx = g.x - b.x;
      const dy = g.y - b.y;
      const rr = b.radius + g.hitRadius;
      if (dx * dx + dy * dy <= rr * rr) world.enemyBullets.release(g);
    }
  }
}

export function update(world, slot, eff, dt) {
  slot.a0 += eff.angularSpeedDegSec * DEG2RAD * dt;
  if (slot.a0 > TAU) slot.a0 -= TAU;              // 유계 유지(장시간 런의 부동소수 방어)

  ensureBodies(world, slot, eff);
  place(world, slot, eff);

  // ★ slot.evolved 분기 정확히 1개 (§9.5)
  if (slot.evolved) {
    slot.a1 -= dt;
    if (slot.a1 <= 0) { slot.a1 = eff.evoBulletClearCooldownSec; aegis(world, slot, eff); }
  }
}

export default { update };
