/**
 * src/core/midboss.js — 중간보스 (§8.9, 순수 core 모듈)
 *
 * 「선택적」의 정확한 의미를 코드로 옮긴 파일이다:
 *   등장 후 midBossLeaveAfterSec(30) 게임초가 지나면 **화면 위로 이탈**한다. 이탈 = 보상 0.
 *   무시해도 **쫓아오지 않는다**(추격 금지 — 그러면 "선택"이 거짓이 된다). 다만 **이탈 전까지는
 *   계속 쏜다** → 무시 = "30초간 추가 탄막을 회피하며 웨이브를 파밍한다"는 선택. 공짜 회피가 아니다.
 *
 * 정본 절:
 *   §8.9   3종 전 테마 공용(`bosses.json`, tier "mid", parts [], phases 1개) · 단일 몸체
 *          mbHammer(anchor + fan/zone 교대) · mbLancer(charge + laser) · mbNest(anchor + aimed + 소환)
 *   §8.9   등장 시각 = `stages.phase.midBossAtSec[stageIndex]`(배열) — 그 길이 == `curve.midBossCount[i]`
 *          (S29). hp = `bosses[].hp × curve.bossHpScale[stageIndex]` — **enemyHpScale 이 아니다**
 *          (그러면 "DPS 체크 = 보스전 예고편"이 소멸한다).
 *   §8.9   `midBossElementRule "notThemeAndNotNormal"` = **런타임 주입**. 저작값 element 는 null 이고
 *          그것이 곧 「주입 대상」 표식이다(S15). 최종 스테이지는 테마가 없으므로 "테마가 아님"이
 *          **공허참** → 후보 = {fire,water,grass} 전부이고 `rng.spawn` 이 **비복원 2개**를 뽑는다.
 *   §8.9   `midBossForcedLeaveOnCrisis` — 위기 세션(새떼)이 시작되면 남아 있어도 즉시 이탈한다.
 *   §8.9-R9 `summon` 은 `midBossSummonsAllowed`(= ["mbNest"])를 통과한 개체만 non-null(S17).
 *          중간보스는 **잡몹 페이즈**에 있으므로 소환된 잡몹의 XP 획득은 정상이다(§6.4 전제 유지).
 *   §9.8.2 `moveId` 는 중간보스 판본에만 인쇄된 필드다 — `charge` 의 **유일한 사용자**가 mbLancer 라
 *          이 파일이 써지는 순간 동결 어휘 8종의 1/8 이 처음으로 도달 가능해진다.
 *
 * ★ 이 파일이 **소유**하는 것: 등장 스케줄 · 속성 주입 · 이동(anchor/charge) · 이탈 · 소환.
 *   발사는 emitters.js(중간보스 분기), 처치 보상은 step.killEnemy(midBossId 분기)가 소유한다.
 * ★ 결정성 — rng 는 `spawn` 스트림만 쓴다(종·속성 추첨). 이동·소환은 순수 함수다.
 */

import { spawnMidBoss, spawnEnemy } from './state.js';
import { formationPos } from './formations.js';

const ELEMENTS3 = ['fire', 'water', 'grass'];   // §4.1 — 노말을 뺀 3종(주입 후보)

/**
 * 현재 스테이지의 **테마 속성**. finale 은 저작값이 null 이고 그것이 곧 "테마가 없다"이다.
 *   ★ stage.js 를 import 하지 않는다 — stage.js 가 이 파일을 부르므로(훅) 순환이 된다.
 *     필요한 건 data + run 뿐이라 여기서 직접 읽는 편이 의존 방향을 한쪽으로 유지한다.
 */
function themeElement(world) {
  const id = world.run.order[world.run.stageIndex];
  const stages = world.data.stages.stages;
  for (let i = 0; i < stages.length; i += 1) if (stages[i].id === id) return stages[i].element;
  throw new Error(`midboss: 미지의 스테이지 id "${id}" (§9.9)`);
}

/** bosses.json 의 tier "mid" 3종. 최초 1회만 만든다(§10.3 핫패스 0 alloc). */
function ensureMidDefs(world) {
  if (world.midDefs !== undefined) return world.midDefs;
  const out = [];
  const bs = world.data.bosses.bosses;
  for (let i = 0; i < bs.length; i += 1) if (bs[i].tier === 'mid') out.push(bs[i]);
  if (out.length === 0) throw new Error('midboss: tier "mid" 개체가 없다 (§8.9)');
  world.midDefs = out;
  return out;
}

/** §8.9 — 이 스테이지의 등장 시각 배열. S29 가 길이 == midBossCount 를 이미 강제한다. */
function atSecList(world) {
  return world.data.stages.phase.midBossAtSec[world.run.stageIndex];
}

/**
 * §8.9 `notThemeAndNotNormal` 의 주입. 테마 속성을 뺀 나머지에서 `rng.spawn` 이 뽑는다.
 *   최종 스테이지(테마 없음)는 후보 3종 전부 · **같은 스테이지에서 이미 쓴 속성은 제외**(비복원).
 */
function injectElement(world) {
  const run = world.run;
  const theme = themeElement(world);
  // §8.9(v1.5, 사용자 결정) — 중간보스는 «무조건 테마 속성»(화산=불 중간보스, 테마 일관성).
  //   finale(테마 null)만 기존처럼 비-노말 후보에서 직전과 다르게 뽑는다(§8.16 서로 다른 속성).
  if (theme !== null) { run.midBossElementPrev = theme; return theme; }
  let n = 0;
  const pick = [];
  for (let i = 0; i < ELEMENTS3.length; i += 1) {
    const el = ELEMENTS3[i];
    if (el === run.midBossElementPrev) continue;
    pick.push(el);
    n += 1;
  }
  if (n === 0) { pick.push(ELEMENTS3[0]); n = 1; }    // 방어(도달 불가: 후보가 최소 1개 남는다)
  const el = pick[Math.floor(world.rng.spawn.f() * n) % n];
  run.midBossElementPrev = el;
  return el;
}

/** §8.9 — 한 마리 등장. 아레나 상단 중앙(스폰 라인)에서 들어온다. */
function spawnOne(world) {
  const defs = ensureMidDefs(world);
  const def = defs[Math.floor(world.rng.spawn.f() * defs.length) % defs.length];
  const a = world.data.rules.view.arena;
  const hp = def.hp * world.data.stages.curve.bossHpScale[world.run.stageIndex];
  const e = spawnMidBoss(world, def, injectElement(world), hp,
    a.x + a.w / 2, world.data.rules.view.spawnLineY);
  return e;
}

/** 살아있는 중간보스. 없으면 null. (동시 1마리 — 등장 스케줄이 그것을 보장한다) */
function liveMidBoss(world) {
  const it = world.enemies.items;
  for (let i = 0; i < it.length; i += 1) if (it[i].alive && it[i].midBossId !== '') return it[i];
  return null;
}

/** 이탈 — 화면 위로 사라진다. 보상 0(반납만 한다). */
function leave(world, e) {
  world.enemies.release(e);
}

/**
 * §8.9 `anchor` — enterSpeed 로 yHoldPx 까지 내려온 뒤 그 높이에서 좌우로 swayAmpPx 왕복한다.
 *   ★ 왕복 속도는 **enterSpeed 를 재사용**한다(중간보스 moveParams 에 속도 키가 하나뿐이다 —
 *     새 데이터 키를 만들지 않는다). mp0 = 진입 완료 플래그 · mp1 = 왕복 위상 · mp2 = 왕복 중심 x.
 */
function moveAnchor(world, e, mp, dt) {
  if (e.mp0 === 0) {
    e.y += mp.enterSpeed * dt;
    if (e.y >= mp.yHoldPx) { e.y = mp.yHoldPx; e.mp0 = 1; e.mp2 = e.x; }
    return;
  }
  const w = mp.swayAmpPx > 0 ? mp.enterSpeed / mp.swayAmpPx : 0;   // 각속도 = 선속도 / 진폭
  e.mp1 += w * dt;
  e.x = e.mp2 + Math.sin(e.mp1) * mp.swayAmpPx;
}

/**
 * §8.9 `charge` — windUpSec 동안 멈춰서 **조준을 굳히고**(그 사이가 곧 회피 리드) dashSpeed 로 돌진한다.
 *   아레나를 관통해 빠져나가면 다시 스폰 라인에서 조준 → 다음 돌진. ★ `charge` 의 유일한 사용자다(§9.8.2).
 *   mp0 = 상태(0 조준 / 1 돌진·아직 진입 전 / 2 돌진·진입함) · mp1 = 조준 타이머 · mp2 = 돌진 각(rad)
 *   ★ 「진입 전」 상태가 따로 있는 이유(회귀): 스폰 라인은 **아레나 위쪽 밖**이라, 진입 여부를 안 보면
 *     돌진하자마자 «밖에 있다»로 판정돼 즉시 리셋 → 영원히 조준만 반복한다(실측된 버그).
 */
function moveCharge(world, e, mp, dt) {
  const a = world.data.rules.view.arena;
  if (e.mp0 === 0) {
    e.vx = 0; e.vy = 0;
    e.mp1 += dt;
    if (e.mp1 >= mp.windUpSec) {
      const p = world.player;
      e.mp2 = Math.atan2(p.y - e.y, p.x - e.x);      // 조준을 «굳힌다» — 돌진 중엔 다시 안 본다
      e.mp0 = 1; e.mp1 = 0;
    }
    return;
  }
  e.x += Math.cos(e.mp2) * mp.dashSpeed * dt;
  e.y += Math.sin(e.mp2) * mp.dashSpeed * dt;
  const inside = e.x >= a.x && e.x <= a.x + a.w && e.y >= a.y && e.y <= a.y + a.h;
  if (inside) { e.mp0 = 2; return; }
  if (e.mp0 !== 2) return;                            // 아직 한 번도 안 들어왔다 = 진입 중
  // 관통해 빠져나갔다 → 스폰 라인으로 복귀해 다시 조준한다(추격 금지 — 리셋이지 추적이 아니다)
  if (e.x < a.x) e.x = a.x;
  if (e.x > a.x + a.w) e.x = a.x + a.w;
  e.y = world.data.rules.view.spawnLineY;
  e.mp0 = 0; e.mp1 = 0;
}

const _pos = { x: 0, y: 0 };   // 재사용(핫패스 0 alloc)

/**
 * §8.9-R9 — mbNest 의 소환. everySec 마다 count 마리를 formationId 모양으로 낸다.
 *   ★ 편대의 **원점은 소환자**다(산란모함이 자기 자리에서 알을 뿌린다). 편대는 «모양»이고
 *     그 모양이 어디에 놓이는지는 누가 스폰시켰는지가 정한다 — 웨이브면 스폰 라인, 소환이면 모함.
 */
function summon(world, e, def, dt) {
  const sm = def.summon;
  if (sm === null) return;
  e.summonT += dt;
  if (e.summonT < sm.everySec) return;
  e.summonT -= sm.everySec;
  const curve = world.data.stages.curve;
  const idx = world.run.stageIndex;
  const arch = world.data.enemies.archetypes;
  let a = null;
  for (let i = 0; i < arch.length; i += 1) if (arch[i].id === sm.archetypeId) { a = arch[i]; break; }
  if (a === null) throw new Error(`midboss: 미지의 소환 아키타입 "${sm.archetypeId}" (§8.9-R9)`);
  const hp = a.hp * curve.enemyHpScale[idx];
  for (let i = 0; i < sm.count; i += 1) {
    formationPos(world, sm.formationId, i, sm.count, e.x, e.y, _pos);
    spawnEnemy(world, sm.archetypeId, e.element, _pos.x, _pos.y, hp, false);
  }
}

/**
 * ★ 훅 진입점 — stage.tickRun 의 MOB 분기가 매 고정 틱 부른다.
 *   (1) 위기 강제 이탈 → (2) 등장 스케줄 → (3) 이탈 타이머 → (4) 이동 → (5) 소환.
 *   발사는 emitters.js 가, 처치 보상은 step.killEnemy 가 소유한다.
 */
export function midBoss(world, dt) {
  const run = world.run;
  const ph = world.data.stages.phase;
  const e = liveMidBoss(world);

  // (1) §8.9 midBossForcedLeaveOnCrisis — 새떼가 오면 무대를 비운다
  if (e !== null && ph.midBossForcedLeaveOnCrisis && run.crisis) { leave(world, e); return; }

  // (2) 등장 — 예정 시각을 지났고 아직 안 나온 마리가 있으면 낸다(동시 1마리)
  if (e === null && !run.crisis) {
    const list = atSecList(world);
    if (run.midBossNext < list.length && run.phaseT >= list[run.midBossNext]) {
      run.midBossNext += 1;
      spawnOne(world);
    }
    return;
  }
  if (e === null) return;

  // ★ §2.7 「스턴 = 개체 정지」 — 이동·소환을 멈춘다(발사는 emitters, 시계는 moveBullets 가 이미 얼린다).
  //   stunSec 은 여기서 감소시키지 않는다 — step.moveBullets 가 단일 소유자다(이중 감소 방지).
  if (e.stunSec > 0) return;

  // (3) 이탈 — 등장 후 midBossLeaveAfterSec. 그 전까지는 계속 쏜다(공짜 회피가 아니다)
  //   ★ e.moveT 는 step.moveBullets 가 이미 매 틱 올린다 — 여기서 또 올리면 시계가 2배로 간다.
  if (e.moveT >= ph.midBossLeaveAfterSec) { leave(world, e); return; }

  // (4) 이동 — §9.8.2 moveId(anchor | charge)
  const defs = ensureMidDefs(world);
  let def = null;
  for (let i = 0; i < defs.length; i += 1) if (defs[i].id === e.midBossId) { def = defs[i]; break; }
  if (def === null) throw new Error(`midboss: 미지의 중간보스 "${e.midBossId}" (§8.9)`);
  if (def.moveId === 'charge') moveCharge(world, e, def.moveParams, dt);
  else if (def.moveId === 'anchor') moveAnchor(world, e, def.moveParams, dt);
  else throw new Error(`midboss: 미구현 moveId "${def.moveId}" — §8.9 는 anchor|charge 만 쓴다`);

  // (5) 소환 — midBossSummonsAllowed 를 통과한 개체만 summon 이 non-null 이다(S17)
  summon(world, e, def, dt);
}

/** 페이즈/스테이지 전이에서 무대를 비운다(잡몹 페이즈가 끝나면 중간보스는 남지 않는다). */
export function clearMidBoss(world) {
  const e = liveMidBoss(world);
  if (e !== null) leave(world, e);
}

export default { midBoss, clearMidBoss };
