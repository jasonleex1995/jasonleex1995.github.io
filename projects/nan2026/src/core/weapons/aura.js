/**
 * src/core/weapons/aura.js — 펄스필드 (§9.5)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표):
 *   base            : dmg radius tickIntervalSec falloff
 *   evolution.params: evoPullForce
 *
 * §9.6.1 훅은 state.recomputeEff 가 이미 적용했다:
 *   rateKey "tickIntervalSec" (H1) · countKey **null**(H4 무효) · pierceApplies false · areaKeys ["radius"]
 *
 * ★ §4.4 elementStampMode "live" — 장은 **지금 이 순간의 스탠스**로 때린다. 적용 때마다 stampFor 로
 *   현재 부여를 재평가한다(각인해 두면 I-2「그린 색 = 적용된 배율」이 깨진다).
 * ★ §9.5 L2297 — 진화(싱귤래리티)의 끌어당김은 **chaff 밴드만**이다(엘리트·보스·파트 제외).
 *
 * 슬롯 스크래치: a0 = 다음 틱까지 남은 시간
 */

import { hitEnemy } from '../damage.js';
import { stampFor } from '../stance.js';
import { killEnemy } from '../step.js';

const CHAFF = 'chaff';

/** §3.1 의 컨텍스트. ★ 모듈 스코프 1회 — 핫패스에서 새로 만들지 않는다 (§10.3) */
const ctx = { matrix: null, dmgMulSum: 0, elementBonusMul: 1, coreGateMul: 0 };

/**
 * 반경 안의 적에게 1회 적용. 지역 배율 = **중심 1.0 → 가장자리 falloff 선형**(§3.1 1항의 지역 배율).
 */
function pulse(world, slot, eff) {
  ctx.matrix = world.data.elements.matrix;
  ctx.dmgMulSum = world.stats.dmgMul;
  ctx.elementBonusMul = world.stats.elementBonusMul;
  ctx.coreGateMul = world.data.rules.boss.coreGateMul;

  const p = world.player;
  const en = world.enemies.items;
  const r = eff.radius;

  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive) continue;
    const dx = e.x - p.x;
    const dy = e.y - p.y;
    const d2 = dx * dx + dy * dy;
    if (d2 > r * r) continue;
    // ★ live — 적용하는 그 순간의 스탠스를 다시 읽는다
    const stamp = stampFor(world, slot.index, 'live', slot.stampElement);
    const d = Math.sqrt(d2);
    const local = 1 + (eff.falloff - 1) * (r > 0 ? d / r : 0);
    const dealt = hitEnemy(world, ctx, slot.family, eff.dmg, local, stamp, e);
    if (dealt > 0 && e.hp <= 0) killEnemy(world, e);
  }
}

/**
 * 진화(싱귤래리티) — chaff 를 중심으로 끌어당긴다. **매 틱** 작용하는 힘이다(피해는 주기).
 *   §9.5 — 엘리트·보스·파트는 제외한다(밀집 학살이지 보스 위치 조작이 아니다).
 */
function pull(world, eff, dt) {
  const p = world.player;
  const en = world.enemies.items;
  const r = eff.radius;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive || e.isBoss || e.elite || e.band !== CHAFF) continue;
    const dx = p.x - e.x;
    const dy = p.y - e.y;
    const d2 = dx * dx + dy * dy;
    if (d2 > r * r || d2 === 0) continue;
    const d = Math.sqrt(d2);
    const s = eff.evoPullForce * dt / d;
    e.x += dx * s;
    e.y += dy * s;
  }
}

/**
 * ★ §9.5(v1.5) — 펄스필드의 정체성 = «탄막 제거». 반경 안의 적 탄을 소거한다(피해 아님).
 *   base 는 이것만 한다(무피해 방어 무기). 진화(싱귤래리티)에서 pulse 피해가 열린다.
 */
function clearBullets(world, eff) {
  const p = world.player;
  const r = eff.radius;
  const eb = world.enemyBullets.items;
  for (let i = 0; i < eb.length; i += 1) {
    const b = eb[i];
    if (!b.alive) continue;
    const dx = b.x - p.x;
    const dy = b.y - p.y;
    if (dx * dx + dy * dy <= r * r) world.enemyBullets.release(b);
  }
}

export function update(world, slot, eff, dt) {
  slot.a0 -= dt;
  const tick = slot.a0 <= 0;
  if (tick) { slot.a0 += eff.tickIntervalSec; clearBullets(world, eff); }   // base·진화 공통 — 탄막 제거
  // ★ slot.evolved 분기 정확히 1개 (§9.5) — 진화(싱귤래리티): 끌어당김(매 틱) + 피해(주기)
  if (slot.evolved) {
    pull(world, eff, dt);
    if (tick) pulse(world, slot, eff);
  }
}

export default { update };
