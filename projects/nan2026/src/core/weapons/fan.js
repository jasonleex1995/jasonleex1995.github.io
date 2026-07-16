/**
 * src/core/weapons/fan.js — 팬아웃 (§9.5)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표):
 *   base            : dmg cooldownSec count projSpeed projRadius lifetimeSec pierce
 *                     hitCooldownSec targetMode arcDeg
 *   evolution.params: evoBlastRadius evoSecondaryDmgMul
 *
 * §9.6.1 훅은 state.recomputeEff 가 이미 적용했다:
 *   rateKey "cooldownSec" · countKey "count" · pierceApplies true
 *   areaKeys ["projRadius", "evoBlastRadius"] — ★ evoBlastRadius 는 evolution.params 에 살며
 *   `src = base ∪ (evolved ? evolution.params : {})` 덕에 진화 시에만 areaMul 을 받는다
 *   ★ H2 — arcDeg 는 **산포**라 areaKeys 에 없다. 여기서도 건드리지 않는다
 *
 * 탄 스크래치: s0 = 1 이면 "아직 안 터진 우리 탄" (진화의 소멸 감지 마커)
 */

import { DEG2RAD } from '../angle.js';
import { spawnPlayerBullet } from '../state.js';
import { playerToEnemy } from '../damage.js';
import { stampFor } from '../stance.js';

const FORWARD = 'forward';

/** §3.1 의 컨텍스트. ★ 모듈 스코프 1회 — 핫패스에서 새로 만들지 않는다 (§10.3) */
const ctx = { matrix: null, dmgMulSum: 0, elementBonusMul: 1, coreGateMul: 0 };

/**
 * 진화(플레어 팬) — 소멸한 탄 자리에서 evoBlastRadius 안의 적에게
 * dmg × evoSecondaryDmgMul (§3.1 의 1항 지역 배율 3종 중 하나).
 *
 * ★★ 보고 대상 — 이 폭발은 **적을 죽일 수 없다.**
 *   step.js 의 `killEnemy()` 는 export 되지 않고, 그것을 여기서 재현하려면 §8.6 의 코인·힐
 *   드랍 때문에 **drop 스트림**이 필요한데 **S11 이 weapons/** 의 pattern 외 스트림을 정적으로 금지**한다.
 *   (★ 이 문장에 그 스트림 이름을 점 표기로 적을 수 없다 — S11 이 주석까지 스캔한다. 보고 참조)
 *   → 즉 "탄이 아닌 경로로 적에게 피해를 주는 무기"는 **처치를 완결할 수단이 구조적으로 없다**
 *     (fan 진화 · orbit · aura · mine · barrage · nova = 12 중 6 패밀리가 같은 벽에 걸린다).
 *   현재 동작: 피해는 정확히 누산되고, hp <= 0 이 된 적은 **다음 탄이 닿는 순간** step.js 가
 *   정상 경로로 죽인다(보상 정상). 즉 피해 손실 0 · 처치 크레딧만 지연된다.
 *   지어내지 않았다 — 정본이 death hook 도 kill API 도 인쇄하지 않았다.
 */
function blast(world, slot, eff, b) {
  ctx.matrix = world.data.elements.matrix;
  ctx.dmgMulSum = world.stats.dmgMul;
  ctx.elementBonusMul = world.stats.elementBonusMul;
  ctx.coreGateMul = world.data.rules.boss.coreGateMul;

  // §4.4 — 폭발은 그 탄의 것이므로 **탄에 각인된 속성**을 그대로 쓴다 (step.js 의 충돌 경로와 동일)
  const stamp = stampFor(world, slot.index, b.stampMode, b.element);
  const en = world.enemies.items;
  const r = eff.evoBlastRadius;

  for (let i = 0; i < en.length; i += 1) {          // §10.3 인덱스 오름차순
    const e = en[i];
    if (!e.alive) continue;
    const dx = e.x - b.x;
    const dy = e.y - b.y;
    if (dx * dx + dy * dy > r * r) continue;
    e.hp -= playerToEnemy(ctx, eff.dmg, eff.evoSecondaryDmgMul, stamp, e);
  }
}

/**
 * 소멸한 우리 탄을 찾아 터뜨린다.
 * ★ 보고 대상 — 정본에 **탄 소멸 훅이 없다.** update(world, slot, eff, dt) 만으로는
 *   "탄이 소멸할 때"를 관측할 수 없어 마커(s0) + 스캔으로 근사했다. 풀이 LIFO 라
 *   **직전 슬롯이 그 탄을 먼저 재사용하면 그 폭발 1회를 놓친다**(spawnPlayerBullet 이 s0 을 0 으로
 *   되돌리므로 오폭은 없다 — 누락만 있다). 정확한 처방은 step.js 가 release 시점에
 *   패밀리 훅을 부르는 것이며, 그 계약은 정본이 인쇄하지 않았다.
 */
function sweepExpired(world, slot, eff) {
  const items = world.playerBullets.items;
  for (let i = 0; i < items.length; i += 1) {
    const b = items[i];
    if (b.alive || b.s0 !== 1 || b.family !== slot.family) continue;
    b.s0 = 0;
    blast(world, slot, eff, b);
  }
}

/** `count` 발을 arcDeg 부채꼴에 균등 배치. 정면 = 화면 위쪽 (§1.1 — y 는 아래가 +) */
function volley(world, slot, eff) {
  const p = world.player;
  const n = eff.count;
  const stepDeg = n > 1 ? eff.arcDeg / (n - 1) : 0;
  let deg = n > 1 ? -eff.arcDeg * 0.5 : 0;

  for (let i = 0; i < n; i += 1) {
    const r = deg * DEG2RAD;
    const b = spawnPlayerBullet(world, slot, eff, p.x, p.y,
      Math.sin(r) * eff.projSpeed, -Math.cos(r) * eff.projSpeed, 1);
    if (b !== null) b.s0 = 1;    // 소멸 감지 마커. 미진화면 sweepExpired 가 안 돌아 아무 일도 없다
    deg += stepDeg;
  }
}

export function update(world, slot, eff, dt) {
  if (eff.targetMode !== FORWARD) {
    throw new Error(`fan: 계약 밖의 targetMode "${eff.targetMode}" — 허용 = forward (§9.5)`);
  }

  // ★ w.evolved 분기 정확히 1개 (§9.5 "진화의 코드 표현") — 플레어 팬
  if (slot.evolved) sweepExpired(world, slot, eff);

  slot.cooldownT -= dt;
  if (slot.cooldownT > 0) return;
  slot.cooldownT += eff.cooldownSec;
  volley(world, slot, eff);
}

export default { update };
