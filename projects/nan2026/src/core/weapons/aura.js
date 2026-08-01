/**
 * src/core/weapons/aura.js — 펄스필드 (§9.5, v1.5 «영역 슬로우»)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표):
 *   base            : dmg radius tickIntervalSec falloff  (★v1.5: dmg·tickIntervalSec·falloff 미사용 —
 *                     펄스필드는 이제 피해/제거가 아니라 «슬로우 필드»다. 레벨은 radius 만 키운다)
 *   evolution.params: evoPullForce
 *
 * ★ §9.5(v1.5, 사용자 결정 2026-08-01) — 펄스필드의 정체성 = «영역 슬로우». v1.4의 «탄막 제거»(너무 쉬움)
 *   폐기. 반경 안의 적 탄을 base 는 느리게(×0.5), 진화(싱귤래리티)는 완전 정지(×0)시킨다. 탄을 지우지
 *   않는다 — «이 범위 안에서만» 느려지고, 벗어나면 원속도로 돌아간다(step.moveBullets 의 slowMul).
 *   진화 = 완전 정지 + 잡몹 끌어당김(블랙홀). 피해 없음(순수 제어 무기).
 *
 * §9.6.1 훅(state.recomputeEff): countKey null · pierceApplies false · areaKeys ["radius"](필드 확대).
 * 슬롯 스크래치: 없음(슬로우는 매 틱 연속 적용, 주기 없음).
 */

const CHAFF = 'chaff';
const SLOW = 0.5;   // base — 적 탄 이동 ×0.5 (화이트리스트 리터럴)
const STOP = 0;     // 진화 — 완전 정지

/**
 * 반경 안의 적 탄에 이동 배율 factor 를 «이번 틱» 세팅한다(더 강한 슬로우가 이긴다). moveBullets 가
 *   적용 후 1 로 리셋 → 필드 밖으로 나가면 원속도. factor=0 이면 필드 안에서 정지(진화).
 */
function slowField(world, eff, factor) {
  const p = world.player;
  const r = eff.radius;
  const eb = world.enemyBullets.items;
  for (let i = 0; i < eb.length; i += 1) {
    const b = eb[i];
    if (!b.alive) continue;
    const dx = b.x - p.x;
    const dy = b.y - p.y;
    if (dx * dx + dy * dy <= r * r && factor < b.slowMul) b.slowMul = factor;
  }
}

/**
 * 진화(싱귤래리티) — chaff 를 중심으로 끌어당긴다(매 틱). §9.5 — 엘리트·보스·파트 제외(위치 조작 아님).
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

export function update(world, slot, eff, dt) {
  // ★ slot.evolved 분기 정확히 1개 (§9.5) — 진화(싱귤래리티): 완전 정지 + 끌어당김
  if (slot.evolved) {
    slowField(world, eff, STOP);
    pull(world, eff, dt);
  } else {
    slowField(world, eff, SLOW);
  }
}

export default { update };
