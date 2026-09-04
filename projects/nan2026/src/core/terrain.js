/**
 * src/core/terrain.js — 지형 장판 (§8.21, v1.10 ⑦ 신설 · 순수 core 모듈)
 *
 * 사용자 결정(2026-09-04): 「공격이 아니라 **유틸을 방해하는** 지형 — 늪 웅덩이는 느리게, 빙원은 가속도가 붙고,
 *   화산은 갑자기 과열돼서 멈춘다」. 그래서 이 장판은 **피해 0** 이다 — 피해 장판은 적 공격(magmaZone·mortar)이
 *   이미 한다(§8.5). 기계는 3종이고 **속성당 하나**, 테마는 겉모습만 다르다(§7.13):
 *     slow    (풀 — 늪·숲)  : 안에 있으면 둔화 = status.slowMoveSpeedMul (기존 둔화 상태를 «매 틱 갱신»해 쓴다, 새 키 0)
 *     inertia (물 — 빙원·바다): 안에서는 이동 응답이 rules.terrain.inertia.responseTauSec 로 미끄러진다(§2.2 의 «항은 존재하고
 *                             값이 0» 이던 지수 스무딩을 지형이 켠다)
 *     heat    (불 — 화산·사막): 안에 있는 동안 player.heat 가 fullSec 에 차고, 다 차면 stallSec 스턴(§2.7) 후 0. 밖에서는 coolSec 에 식는다
 *
 * 소유: 스폰 시각표·위치(rng.terrain) · 흐름(scrollSpeedPx, 위 → 아래) · 반납(아레나 아래로 나가면) · 「플레이어가 어느
 *   장판 안인가」의 술어(terrainUnder). 효과의 «적용»은 step.movePlayer 가 한다(이동의 단일 소유자, §2.2).
 * ★ 결정성 — rng 는 `terrain` 스트림만 쓴다(x 위치). 시각표는 run.terrainNextT(게임초). 핫패스 0 alloc.
 * ★ 지형이라 예고가 없다 — 피해가 없으니 §2.1 ① 의 대상이 아니고, 항상 보인다(§12.3 레이어 1).
 */

import { TERRAIN_KINDS } from './schema.mjs';

export const T_SLOW = 0;
export const T_INERTIA = 1;
export const T_HEAT = 2;

/** 현재 스테이지 엔트리(stage.js 를 import 하지 않는다 — 순환). */
function stageOf(world) {
  const id = world.run.order[world.run.stageIndex];
  const list = world.data.stages.stages;
  for (let i = 0; i < list.length; i += 1) if (list[i].id === id) return list[i];
  throw new Error(`terrain: 미지의 스테이지 id "${id}" (§9.9)`);
}

/** 지형 종 → 인덱스. 폴백 금지(§9.3). */
function kindIndex(kind) {
  const k = TERRAIN_KINDS.indexOf(kind);
  if (k < 0) throw new Error(`terrain: 미지의 terrainKind "${kind}" (§8.21)`);
  return k;
}

/**
 * ★ 훅 진입점 — stage.tickRun 이 매 고정 틱 부른다(페이즈 무관: 흐름·반납은 늘, 스폰은 MOB 에서만).
 *   (1) 흐름 — 모든 장판이 scrollSpeedPx 로 내려간다. 아레나 아래로 완전히 나가면 반납.
 *   (2) 스폰 — MOB 페이즈, terrainKind ≠ null, everySec 마다, 무대에 maxOnScreen 미만일 때.
 *       x = 아레나 안 균일(rng.terrain), y = spawnLineY − radius (위에서 «들어온다»).
 */
export function terrainTick(world, dt) {
  const run = world.run;
  const tr = world.data.rules.terrain;
  const a = world.data.rules.view.arena;
  const it = world.terrain.items;
  for (let i = 0; i < it.length; i += 1) {
    const t = it[i];
    if (!t.alive) continue;
    t.y += tr.scrollSpeedPx * dt;
    if (t.y - t.radius > a.y + a.h) world.terrain.release(t);
  }
  if (run.phase !== 'MOB') return;
  const st = stageOf(world);
  if (st.terrainKind === null) return;                         // finale — 테마가 없으니 지형도 없다
  if (world.time < run.terrainNextT) return;
  run.terrainNextT = world.time + tr.everySec;
  if (world.terrain.live >= tr.maxOnScreen) return;
  const t = world.terrain.alloc();
  if (t === null) return;                                       // caps.terrain — rejectSpawn
  const r = tr.radiusPx;
  t.kind = kindIndex(st.terrainKind);
  t.radius = r;
  t.x = a.x + r + world.rng.terrain.f() * (a.w - 2 * r);
  t.y = world.data.rules.view.spawnLineY - r;
}

/** 점 (x, y) 가 어느 지형 안인가 — kind 인덱스 또는 -1. 겹치면 «먼저 스폰된 것»(인덱스 오름차순). 0 alloc. */
export function terrainUnder(world, x, y) {
  const it = world.terrain.items;
  for (let i = 0; i < it.length; i += 1) {
    const t = it[i];
    if (!t.alive) continue;
    const dx = x - t.x; const dy = y - t.y;
    if (dx * dx + dy * dy <= t.radius * t.radius) return t.kind;
  }
  return -1;
}

/** 스테이지 전이·런 시작에서 무대를 비운다(이전 테마의 지형이 다음 테마로 넘어가지 않는다). */
export function clearTerrain(world) {
  const it = world.terrain.items;
  for (let i = 0; i < it.length; i += 1) if (it[i].alive) world.terrain.release(it[i]);
  world.run.terrainNextT = 0;
  world.player.heat = 0;
}

export default { terrainTick, terrainUnder, clearTerrain };
