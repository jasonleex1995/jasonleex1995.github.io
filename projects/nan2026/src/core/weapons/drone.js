/**
 * src/core/weapons/drone.js — 옵션 (§9.5)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표):
 *   base            : dmg projSpeed projRadius lifetimeSec pierce hitCooldownSec targetMode
 *                     droneCount anchorOffsets droneFireSec droneRangePx
 *   evolution.params: evoTrailDelaySec
 *
 * §9.6.1 훅은 state.recomputeEff 가 이미 적용했다:
 *   rateKey "droneFireSec" (H1) · countKey **null** — ★ anchorOffsets 가 droneCount 만큼만 인쇄돼 있어
 *   projCountAdd 로 위성 수를 늘리면 «자리 없는 위성»이 생긴다(§9.6.1 L2474). pierceApplies true ·
 *   areaKeys ["droneRangePx", "projRadius"]
 *
 * ★ 위성은 drones 풀에 산다(사전할당, §12.1). 이 파일이 **소유자**다 — 배치·발사·반납을 스스로 한다.
 *   위성이 쏘는 것은 평범한 플레이어 탄이므로 이동·충돌은 step 이 그대로 처리한다.
 *
 * 슬롯 스크래치: a0 = 진화(잔상 편대)의 위치 이력 샘플 타이머 / a1 = 회수 내부 쿨다운(§9.5 v1.7)
 */

import { spawnPlayerBullet } from '../state.js';

const NEAREST = 'nearest';

/** droneRangePx 안의 최근접 적. 없으면 null. §10.3 인덱스 오름차순(동점은 낮은 인덱스) */
function nearestEnemy(world, x, y, radius) {
  const en = world.enemies.items;
  let best = null;
  let bestD = radius * radius;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive) continue;
    const dx = e.x - x;
    const dy = e.y - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = e; }
  }
  return best;
}

/** 이 슬롯의 살아있는 위성 수 (§5.3 family-키로 식별) */
function liveDrones(world, slot) {
  const it = world.drones.items;
  let n = 0;
  for (let i = 0; i < it.length; i += 1) if (it[i].alive && it[i].family === slot.family) n += 1;
  return n;
}

/** droneCount 만큼 위성을 유지한다. ox/oy 는 update 가 매 틱 현재 eff 로 다시 각인한다(레벨업 반영). */
function ensureDrones(world, slot, eff) {
  let have = liveDrones(world, slot);
  const want = eff.droneCount;
  while (have < want) {
    const d = world.drones.alloc();
    if (d === null) { world.capHits.drone += 1; return; }   // §12.1 초과 = 이번 틱 포기
    const off = eff.anchorOffsets[have];
    d.family = slot.family;
    d.ox = off[0]; d.oy = off[1];
    d.x = world.player.x + d.ox;
    d.y = world.player.y + d.oy;
    d.fireT = 0;
    have += 1;
  }
}

export function update(world, slot, eff, dt) {
  // §9.5(v1.7) 회수 내부 쿨다운 — step.droneSalvage 가 처치 시 세운다. 감소는 여기가 단일 소유.
  if (slot.a1 > 0) { slot.a1 -= dt; if (slot.a1 < 0) slot.a1 = 0; }
  if (eff.targetMode !== NEAREST) {
    throw new Error(`drone: 미구현 targetMode "${eff.targetMode}" — weapons.json 은 nearest 만 쓴다 (§9.5)`);
  }
  ensureDrones(world, slot, eff);

  // ★ slot.evolved 분기 정확히 1개 (§9.5) — 잔상 편대: 고정 앵커 대신 **플레이어의 과거 위치**를 따른다.
  //   이력 버퍼 없이 evoTrailDelaySec 를 «따라붙는 부드러움»으로 환원한다(지수 추종) — 결정적·0 alloc.
  const trail = slot.evolved;
  const follow = trail ? dt / (eff.evoTrailDelaySec + dt) : 1;

  const p = world.player;
  const it = world.drones.items;
  let k = 0;
  for (let i = 0; i < it.length; i += 1) {
    const d = it[i];
    if (!d.alive || d.family !== slot.family) continue;

    // ★ 앵커를 매 틱 현재 eff 로 다시 각인 — 레벨업이 droneCount·anchorOffsets 를 바꾸면 기존 위성도
    //   새 편대 자리로 옮긴다(스폰 때 굳지 않게). k = 이 family 위성의 순번(0..droneCount-1).
    const off = eff.anchorOffsets[k];
    d.ox = off[0]; d.oy = off[1];
    k += 1;

    const tx = p.x + d.ox;
    const ty = p.y + d.oy;
    if (trail) { d.x += (tx - d.x) * follow; d.y += (ty - d.y) * follow; }
    else { d.x = tx; d.y = ty; }

    d.fireT -= dt;
    if (d.fireT > 0) continue;
    d.fireT += eff.droneFireSec;
    const e = nearestEnemy(world, d.x, d.y, eff.droneRangePx);
    let vx = 0;
    let vy = -eff.projSpeed;                       // 사거리 안에 적이 없으면 정면(§1.1 위쪽)
    if (e !== null) {
      const dx = e.x - d.x;
      const dy = e.y - d.y;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len > 0) { vx = (dx / len) * eff.projSpeed; vy = (dy / len) * eff.projSpeed; }
    }
    spawnPlayerBullet(world, slot, eff, d.x, d.y, vx, vy, 1);
  }
}

export default { update };
