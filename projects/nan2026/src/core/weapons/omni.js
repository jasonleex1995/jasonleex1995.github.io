/**
 * src/core/weapons/omni.js — 리어가드 (§9.5)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표 — 이 파일은 계약 밖의 키를 읽지 않는다):
 *   base            : dmg cooldownSec projSpeed projRadius lifetimeSec pierce hitCooldownSec
 *                     dirCount dirOffsetDeg rearBias
 *   evolution.params: evoRingRotDeg
 *
 * ★ 이 파일은 JSON 을 스스로 읽지 않는다. `eff` 는 state.recomputeEff 가 §9.6.1 의
 *   훅(H1~H4)을 **이미 적용해서** 넘긴 값이다 — 여기서 패시브를 다시 곱하면 이중 적용이다.
 *     rateKey "cooldownSec" (H1) · countKey "dirCount" (+projCountAdd)
 *     pierceApplies true (+pierceAdd) · areaKeys ["projRadius"] (×(1+areaMul), H3 클램프)
 *
 * ★ 리어가드의 정체성 = 사각이 없다. 매 cooldownSec 마다 dirCount 발을 360° 에 균등 배치한다.
 *   **화면 뒤쪽(아래, vy>0)으로 나가는 탄에만** dmg×rearBias 를 실어 뒤를 지킨다 (§9.5 L2284).
 *   rearBias 는 spawnPlayerBullet 의 localMul 인자로 흘러 §3.1 term-1 안에서 곱해진다.
 *
 * 슬롯 스크래치 (state.makeSlot 이 자리를 미리 잡아둔다):
 *   cooldownT = 볼리 주기 / a0 = 진화(링 버스트)의 누적 회전각(라디안, wrapAngle 로 유계)
 *
 * ★ omni 는 targetMode 가 없다 (자기중심 방사). eff.targetMode 를 읽지 않는다.
 * ★ 각도는 전부 라디안. 한 바퀴 = 2·Math.PI (2 는 화이트리스트, Math.PI 는 속성 접근 = 리터럴 아님).
 *   도(°) 파라미터는 DEG2RAD 로만 변환한다 (weapons/** 에서 180 같은 리터럴은 불법, §9.1 D1).
 */

import { DEG2RAD, wrapAngle } from '../angle.js';
import { spawnPlayerBullet } from '../state.js';

/**
 * 한 번의 볼리 = dirCount 발을 360° 에 균등 배치. 시작각 = dirOffsetDeg + (진화 시) 누적 링 회전.
 * §1.1 — y 는 아래가 +. a=0 → 위(정면). a 가 뒤(아래)를 향하면(vy>0) rearBias 를 싣는다.
 */
function ring(world, slot, eff) {
  const p = world.player;
  const n = eff.dirCount;
  const stepRad = (2 * Math.PI) / n;
  const startRad = eff.dirOffsetDeg * DEG2RAD + slot.a0;   // a0 = 0 (미진화) 또는 누적 회전(진화)

  for (let i = 0; i < n; i += 1) {
    const a = startRad + stepRad * i;
    const vx = Math.sin(a) * eff.projSpeed;
    const vy = -Math.cos(a) * eff.projSpeed;               // a=0 → vy = -projSpeed (위/정면)
    // 뒤쪽(아래) 반구 = 발사각이 **아래 성분**을 가짐 = |wrapAngle(a)| > π/2 (§9.5 L2284). 그 탄에만 rearBias.
    //   ★ vy>0 로 판정하지 않는다: 정확히 수평인 탄(a=±90°, 짝수 dirCount·offset 0)의 vy 는 부동소수
    //     노이즈(±1e-14)라 좌/우가 부호 비대칭이 된다(왼쪽만 rear 오분류). 각도 판정은 두 수평탄을
    //     **모두 front** 로 놓아 좌우 대칭이며 엔진 간 결정적이다(정본 "아래 성분 없음 = rear 아님"에 부합).
    const localMul = Math.abs(wrapAngle(a)) > Math.PI / 2 ? eff.rearBias : 1;
    spawnPlayerBullet(world, slot, eff, p.x, p.y, vx, vy, localMul);
  }
}

export function update(world, slot, eff, dt) {
  slot.cooldownT -= dt;
  if (slot.cooldownT > 0) return;
  slot.cooldownT += eff.cooldownSec;

  ring(world, slot, eff);

  // ★ slot.evolved 분기 정확히 1개 (§9.5 "진화의 코드 표현")
  // 링 버스트 — 볼리마다 링이 evoRingRotDeg 만큼 돌아 사각이 완전히 사라진다.
  //   누적각은 wrapAngle 로 [-π,π] 에 가둬 장시간 런에서도 부동소수가 자라지 않는다 (결정성).
  if (slot.evolved) slot.a0 = wrapAngle(slot.a0 + eff.evoRingRotDeg * DEG2RAD);
}

export default { update };
