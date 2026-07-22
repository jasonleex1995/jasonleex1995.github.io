/**
 * src/core/formations.js — 편대 배치 (§9.9.2, 순수 core 모듈)
 *
 * 편대는 «모양»이다. 그 모양이 **어디에** 놓이는지는 누가 스폰시켰는지가 정한다 —
 * 웨이브면 스폰 라인 중앙, 소환(§8.9-R9 mbNest)이면 소환자의 자리. 그래서 원점을 인자로 받는다.
 *
 * 구현 어휘: scatter · arc · lineH · vWedge (그 외 columnV·pincer·미지 = scatter 폴백).
 * scatter 만 `rng.spawn` 을 쓴다 — 나머지는 순수 함수다(§10.2 결정성).
 */

import { DEG2RAD } from './angle.js';

/**
 * 편대의 i번째(총 count) 개체 좌표를 out 에 쓴다. 원점(originX, originY)이 편대의 기준점이다.
 * 아레나 가로 밖으로 새지 않게 구조적으로 클램프한다(밸런스 값이 아니라 좌표계 경계다).
 */
export function formationPos(world, formationId, i, count, originX, originY, out) {
  const a = world.data.rules.view.arena;
  const forms = world.data.stages.formations;
  const rng = world.rng.spawn;

  let x = originX;
  let y = originY;

  if (formationId === 'arc') {
    const f = forms.arc;
    const t = count > 1 ? i / (count - 1) : 0.5;
    const ang = (-f.spanDeg / 2 + t * f.spanDeg) * DEG2RAD;
    x = originX + Math.sin(ang) * f.radiusPx;
    // 가운데가 앞선(=최대 y, 하강 방향으로 선두) 아래로 볼록한 호. (1-cos) 은 가운데 0·날개 양수라
    //   **빼야** 가운데가 앞선다(더하면 날개가 앞서는 ∩ 로 뒤집힌다 — 주석과 반대였다).
    y = originY - (1 - Math.cos(ang)) * f.radiusPx * 0.3;
  } else if (formationId === 'lineH') {
    const f = forms.lineH;
    x = originX + (i - (count - 1) / 2) * f.gapPx;
    y = originY;
  } else if (formationId === 'vWedge') {
    const f = forms.vWedge;
    if (i === 0) { x = originX; y = originY; }
    else {
      const rank = Math.ceil(i / 2);
      const side = (i % 2 === 1) ? -1 : 1;
      const ar = f.angleDeg * DEG2RAD;
      x = originX + side * rank * f.gapPx * Math.sin(ar);
      y = originY - rank * f.gapPx * Math.cos(ar);          // 날개가 위로·바깥으로 = 아래로 향한 V
    }
  } else {
    // scatter + 폴백(columnV·pincer·미지) — rng.spawn 산포. jitterPx = y 계단, minSepPx = 가장자리 여백.
    //   ★ 원점 중심으로 흩는다(소환 편대가 소환자 자리에 놓이는 계약). 웨이브는 originX=아레나 중앙이라
    //     결과·rng 소비가 기존과 **완전히 동일**하고(중앙±(a.w-2minSep)/2 = a.x+minSep…a.x+a.w-minSep),
    //     소환(mbNest)만 originX=소환자로 옮겨간다. 아래 클램프가 아레나 밖을 막는다.
    const f = forms.scatter;
    x = originX + (rng.f() - 0.5) * (a.w - 2 * f.minSepPx);
    y = originY - rng.f() * f.jitterPx;
  }

  if (x < a.x) x = a.x;
  if (x > a.x + a.w) x = a.x + a.w;
  out.x = x;
  out.y = y;
  return out;
}

export default { formationPos };
