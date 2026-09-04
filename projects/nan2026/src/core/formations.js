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
  } else if (formationId === 'wall') {
    // §8.19.2(v1.9) 도입의 «벽» — 몸을 촘촘히 세우되 줄마다 «차선»(lane)을 한 칸 비운다.
    //   ★ v1.8 의 벽은 순틈이 gapPx - 지름 = 3px 라 «완성된 한 줄»이 물리적으로 통과 불가였다.
    //     그 벽을 두껍게 하면 «강제 피격»이 몸 수에 비례해 늘어 §2.1 ①(완벽하면 안 맞는다)이 깨진다
    //     — 세 각도의 심사가 전부 같은 자리에서 걸렸다. 그래서 질량은 차선이 «먼저» 열려야 풀린다.
    //   차선 = 연속한 laneSlots 칸을 비운 자리. 순틈 = (laneSlots + 1)·gapPx - 몸 지름 이고
    //     S52 가 그 값이 fairness.minGapWidthPx 이상임을 정적으로 강제한다(§12.4 의 어휘를 재사용).
    //   차선의 자리는 줄마다 laneStrideCols 칸씩 옮겨간다 — 난수 0(§10.2), 상태 0. 벽은 «막는 것»이
    //     아니라 «바늘귀가 움직이는 것»이 된다: 비켜 갈 길은 항상 있고, 가만히 있으면 맞는다.
    const f = forms.wall;
    const slots = f.perRow;                       // 한 줄의 «칸» 수(몸 수가 아니다)
    const bodies = slots - f.laneSlots;           // 한 줄에 실제로 서는 몸 수
    const col = i % bodies;
    const row = Math.floor(i / bodies);
    // ★ 차선의 자리는 «지그재그»로 옮겨간다 — 나머지연산으로 감으면 오른쪽 끝에서 왼쪽 끝으로
    //   한 줄 만에 튀어 플레이어가 따라갈 수 없다(실측: 464px 점프, 0.78초에 갈 수 있는 거리는 219px).
    //   삼각파는 줄마다 정확히 laneStrideCols 칸씩만 움직이고 끝에서 되튄다 — 길이 «이어진다».
    const span = bodies;                                   // 차선이 설 수 있는 칸 0..bodies
    const p = (row * f.laneStrideCols) % (2 * span);
    const lane = p <= span ? p : 2 * span - p;
    const slot = col < lane ? col : col + f.laneSlots;      // 차선 칸을 건너뛴다
    // ★ 원점 기준이 «칸 격자»다(몸 수가 아니다) — 마지막 줄이 덜 차도 줄이 좌우로 흔들리지 않는다.
    x = originX + (slot - (slots - 1) / 2) * f.gapPx;
    y = originY - row * f.rowGapPx;
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
