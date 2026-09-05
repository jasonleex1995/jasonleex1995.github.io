/**
 * src/core/weapons/spiral.js — 스파이럴 (§9.5, v1.10 ㉟ 신설)
 *
 * 폐쇄된 파라미터 계약:
 *   base            : dmg cooldownSec count projSpeed projRadius lifetimeSec pierce hitCooldownSec targetMode
 *                     ampPx freqHz
 *   evolution.params: evoAmpMul evoLifetimeMul
 *
 * 동사: 정면으로 올라가며 좌우로 «나선»을 그리는 탄. count 줄기가 위상을 360°/count 씩 나눠 서로 엇갈린다 — 세로 띠를
 *   넓게 덮는다(정면 직사·산탄과 다른 «면» 커버). x 속도 = amp·ω·cos(ω·age + 위상), y 속도 = projSpeed.
 *   진화(토네이도): 진폭 × evoAmpMul, 수명 × evoLifetimeMul — 나선이 화면 폭을 쓸고 오래 남는다. 짝 = 자세 안정기(자이로).
 *
 * 탄의 vx 는 이 모듈이 «매 틱» 다시 쓴다(오빗이 공의 좌표를 매 틱 쓰는 것과 같은 규약). 적분은 step 이 한다.
 * 훅: rateKey cooldownSec · countKey count · pierceApplies true · areaKeys [] · speedKeys [projSpeed] · durationKeys [lifetimeSec]
 * 탄: s0 = 위상(rad). 스크래치: cooldownT 만. 레퍼런스: 슈팅 고전 나선탄 · 뱀서 산타 물(면 커버)
 */

import { TAU } from '../angle.js';
import { spawnPlayerBullet } from '../state.js';

const FORWARD = 'forward';

export function update(world, slot, eff, dt) {
  if (eff.targetMode !== FORWARD) {
    throw new Error(`spiral: 계약 밖의 targetMode "${eff.targetMode}" — 허용 = forward (§9.5)`);
  }
  // ★ w.evolved 분기 정확히 1개 — 토네이도
  const amp = slot.evolved ? eff.ampPx * eff.evoAmpMul : eff.ampPx;
  const w = TAU * eff.freqHz;
  const it = world.playerBullets.items;
  for (let i = 0; i < it.length; i += 1) {
    const b = it[i];
    if (!b.alive || b.slot !== slot.index) continue;
    b.vx = amp * w * Math.cos(w * b.age + b.s0);   // 나선 — 적분하면 x = amp·sin(ω·age + 위상)
  }
  slot.cooldownT -= dt;
  if (slot.cooldownT > 0) return;
  slot.cooldownT += eff.cooldownSec;
  const p = world.player;
  const n = eff.count;
  for (let k = 0; k < n; k += 1) {
    const b = spawnPlayerBullet(world, slot, eff, p.x, p.y, 0, -eff.projSpeed, 1);
    if (b === null) return;
    b.s0 = (TAU * k) / n;                            // 위상 분할
    if (slot.evolved) b.lifetimeSec = eff.lifetimeSec * eff.evoLifetimeMul;
  }
}

export default { update };
