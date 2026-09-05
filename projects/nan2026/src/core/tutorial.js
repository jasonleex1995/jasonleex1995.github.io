/**
 * src/core/tutorial.js — 튜토리얼 (§6.7, v1.10 ㊴ 신설)
 *
 * ★ 사용자(2026-09-05): 「게임이 익숙하지 않은 사람들이 게임과 키에 익숙할 수 있게 튜토리얼 스테이지를 구성하려고 해.
 *   적을 되게 쉽게 구성해서 한번 체험할 수 있게.」 — **읽는 설명이 아니라 «해 보는» 순서**다.
 *   그래서 타이틀의 요약 문장(「QWER 스탠스 · 상성 ×2 …」)을 지우고 그 자리를 이 절이 가져간다.
 *
 * 구조 — 스텝의 목록은 `data/tutorial.json` 이 소유한다(§9.2 매니페스트 11번째 파일). 이 모듈은 «규칙»만 갖는다:
 *   ① 스텝에 들어가면 그 스텝의 적을 놓는다(spawn) · 필요하면 속성 투자를 준다(grant)
 *   ② 매 틱 목표(goal)를 재고, 충족되면 다음 스텝으로 — 목표 어휘는 닫혀 있다(아래 GOALS, schema·check 가 강제)
 *   ③ **튜토리얼에서는 죽지 않는다** — HP 가 safeHpFloor 아래로 내려가면 가득 채운다(연습에 사망은 없다)
 *
 * ★ 스테이지 디렉터(stage.tickRun)를 쓰지 않는다 — 페이즈·웨이브·보스 스폰이 전부 스테이지의 규칙이기 때문이다.
 *   그 자리에 이 함수가 `hooks.run` 으로 들어간다(같은 계약: (world, dt), 고정 틱). 적 이동·적 사격·충돌·드래프트는
 *   전부 평소의 step 이 그대로 처리한다 — «튜토리얼 전용 게임»이 아니라 같은 게임의 얇은 진행자다.
 */

import { spawnEnemy, spawnBossCore, spawnBossPart } from './state.js';
import { terrainUnder } from './terrain.js';
import { investElement } from './stance.js';
import { TERRAIN_KINDS } from './schema.mjs';

/** §6.7 — 목표 어휘(닫힘). schema.mjs · check.mjs 가 같은 목록을 갖는다(독립 사본). */
export const GOALS = ['move', 'clear', 'level', 'superHit', 'resistHit', 'stances', 'survive', 'terrain', 'boss', 'confirm'];

const STANCE_ELEMENTS = ['fire', 'water', 'grass'];
const REFILL_SEC = 1.5;   // 표적이 다 떨어지고 이만큼 지나면 다시 놓는다(구조 상수 — 밸런스 값 아님)

export function makeTutorialState() {
  return {
    i: 0,                 // 현재 스텝
    t: 0,                 // 그 스텝에 머문 게임초
    entered: false,       // 이 스텝의 spawn/grant 를 이미 했는가
    movedPx: 0,           // move 목표 누적
    lastX: 0, lastY: 0,
    superHits: 0,         // superHit 목표 누적(×2 를 맞은 개체 수)
    superFlag: [],        // 개체마다 «이미 셌는가» 래치 — 개체가 죽어 사라져도 셈이 되돌아가지 않는다
    resistHits: 0,        // ×½ 로 때린 개체 수(㊷ — 반대 방향도 «해 보고» 배운다)
    resistFlag: [],
    refillT: 0,           // 표적이 다 떨어졌을 때의 재보급 대기(초) — 튜토리얼은 «막히지 않는다»
    stanceSeen: [],       // stances 목표 — 눌러 본 속성
    terrainHits: 0,
    ids: [],              // 이 스텝이 놓은 개체(idx, gen)
    confirm: false,       // confirm 목표 — 드라이버가 Space 로 올린다
    done: false,          // 전 스텝 완료
  };
}

/** 현재 스텝 정의 (없으면 null = 끝) */
export function tutorialStep(world) {
  const steps = world.data.tutorial.steps;
  const tu = world.tut;
  return tu.i < steps.length ? steps[tu.i] : null;
}

/** confirm 목표를 올린다(드라이버의 Space). 다른 목표엔 영향이 없다. */
export function tutorialConfirm(world) { world.tut.confirm = true; }

/** 스텝의 적을 놓는다. 아레나 가운데 기준 좌우 대칭 · 난수 0(§10.2 — 튜토리얼은 매번 같아야 배울 수 있다). */
function spawnStepEnemies(world, st) {
  const tu = world.tut;
  const cfg = world.data.tutorial;
  const a = world.data.rules.view.arena;
  const arch = world.data.enemies.archetypes;
  for (let k = 0; k < st.spawn.length; k += 1) {
    const sp = st.spawn[k];
    let def = null;
    for (let i = 0; i < arch.length; i += 1) if (arch[i].id === sp.archetypeId) { def = arch[i]; break; }
    if (def === null) throw new Error(`tutorial: 미지의 아키타입 "${sp.archetypeId}" (§6.7)`);
    for (let n = 0; n < sp.count; n += 1) {
      const x = a.x + a.w / 2 + (n - (sp.count - 1) / 2) * cfg.spawnGapPx;
      const e = spawnEnemy(world, def.id, sp.element, x, cfg.spawnYPx, Math.max(1, def.hp * cfg.hpMul), false);
      if (e !== null) { e.vy = 0; e.vx = 0; tu.ids.push(e.idx, e.gen); tu.superFlag.push(false); tu.resistFlag.push(false); }
    }
  }
}

function enterStep(world) {
  const tu = world.tut;
  const st = tutorialStep(world);
  tu.entered = true;
  tu.t = 0; tu.superHits = 0; tu.terrainHits = 0; tu.confirm = false; tu.ids.length = 0; tu.refillT = 0;
  tu.superFlag.length = 0; tu.resistHits = 0; tu.resistFlag.length = 0;
  // ★ 스텝은 «깨끗한 판»에서 시작한다 — 앞 스텝에서 안 죽고 남은 적·탄이 다음 가르침을 흐린다
  //   (실측: 3단계는 레벨업으로 끝나므로 잡몹이 남고, 4단계에서 풀 적과 섞여 「무엇을 때리라는 건지」가 사라졌다).
  for (const e of world.enemies.items) if (e.alive) world.enemies.release(e);
  for (const b of world.enemyBullets.items) if (b.alive) world.enemyBullets.release(b);
  tu.movedPx = 0; tu.lastX = world.player.x; tu.lastY = world.player.y;
  if (st === null) return;
  const cfg = world.data.tutorial;
  const a = world.data.rules.view.arena;

  // grant — 속성 투자 1(스탠스가 «각인»으로 이어지는 것을 보이려면 투자가 있어야 한다).
  //   ★ 반드시 investElement 로 준다 — 그 함수만이 §4.3 각인 재계산(recomputeStamps)을 부른다(직접 대입하면 각인이 안 내려간다).
  if (st.grant !== null) investElement(world, st.grant.invest);

  spawnStepEnemies(world, st);

  // boss 스텝 — 코어 1 + 모듈 2. 정본의 보스 정의를 그대로 쓰되 HP 만 연습용으로 낮춘다(§8.13 봉인 규칙은 진짜다).
  if (st.goal.kind === 'boss') {
    const b = world.data.bosses.bosses.find((x) => x.tier === 'stage');
    if (b === undefined) throw new Error('tutorial: tier "stage" 보스가 없다 (§6.7)');
    const cx = a.x + a.w / 2;
    const cy = cfg.spawnYPx + 40;
    const core = spawnBossCore(world, b.id, b.core, b.core.hp * cfg.hpMul * 0.25, cx, cy);
    if (core !== null) { core.sealedNow = true; tu.ids.push(core.idx, core.gen); tu.superFlag.push(false); tu.resistFlag.push(false); }
    const mods = b.parts.filter((p) => p.partType === 'armament').slice(0, 2);
    for (let k = 0; k < mods.length; k += 1) {
      const p = spawnBossPart(world, b.id, mods[k], mods[k].hp * cfg.hpMul * 0.25, cx, cy);
      if (p !== null) { tu.ids.push(p.idx, p.gen); tu.superFlag.push(false); tu.resistFlag.push(false); }
    }
  }
}

/** 이 스텝이 놓은 개체 중 살아 있는 수 (idx·gen 쌍으로 확인 — 풀 재사용에 속지 않는다) */
function aliveOfStep(world) {
  const tu = world.tut;
  const items = world.enemies.items;
  let n = 0;
  for (let k = 0; k < tu.ids.length; k += 2) {
    const e = items[tu.ids[k]];
    if (e.alive && e.gen === tu.ids[k + 1]) n += 1;
  }
  return n;
}

function coreAliveOfStep(world) {
  const tu = world.tut;
  const items = world.enemies.items;
  for (let k = 0; k < tu.ids.length; k += 2) {
    const e = items[tu.ids[k]];
    if (e.alive && e.gen === tu.ids[k + 1] && e.isCore) return e;
  }
  return null;
}

function goalMet(world, st) {
  const tu = world.tut;
  const p = world.player;
  switch (st.goal.kind) {
    case 'move': return tu.movedPx >= st.goal.value;
    case 'clear': return aliveOfStep(world) === 0;
    case 'level': return p.level >= st.goal.value;
    case 'superHit': return tu.superHits >= st.goal.value;
    case 'resistHit': return tu.resistHits >= st.goal.value;
    case 'stances': return tu.stanceSeen.length >= st.goal.value;
    case 'survive': return tu.t >= st.goal.value;
    case 'terrain': return tu.terrainHits >= st.goal.value;
    case 'boss': return coreAliveOfStep(world) === null && tu.ids.length > 0;
    case 'confirm': return tu.confirm;
    default: throw new Error(`tutorial: 미지의 목표 "${st.goal.kind}" (§6.7)`);
  }
}

/**
 * ★ 훅 진입점 — step 이 매 고정 틱 부른다(hooks.run 자리). 스테이지 디렉터의 자리를 대신한다.
 */
export function tickTutorial(world, dt) {
  const tu = world.tut;
  if (tu.done) return;
  const cfg = world.data.tutorial;
  const p = world.player;

  if (!tu.entered) enterStep(world);
  const st = tutorialStep(world);
  if (st === null) { tu.done = true; return; }
  tu.t += dt;

  // ③ 튜토리얼에서는 죽지 않는다 — 연습에 사망은 없다(사용자: 「되게 쉽게」).
  if (p.hp <= p.hpMax * cfg.safeHpFloor) p.hp = p.hpMax;
  world.over = false;

  // 목표 누적 — 이동 거리 · ×2 히트 · 스탠스 · 지형
  tu.movedPx += Math.abs(p.x - tu.lastX) + Math.abs(p.y - tu.lastY);
  tu.lastX = p.x; tu.lastY = p.y;
  // ×2 히트 — ★ hitFx 는 «이번 틱» 링이고 step 이 훅보다 **먼저** 비운다(훅은 collide 앞이다). 그래서 링이 아니라
  //   개체가 들고 있는 누적(e.dmgSuper, §11.3 초효과 지분의 소유자)을 본다.
  //   ★★ 개체마다 «이미 셌다» 래치를 둔다 — 「살아 있는 개체 수」로 세면 **맞고 죽는 즉시 셈이 되돌아가** 목표가 영원히
  //     안 찼다(플레이테스트 2026-09-05: 4단계에서 안 넘어감). 죽어도 래치는 남는다.
  {
    const items = world.enemies.items;
    for (let k = 0; k < tu.ids.length; k += 2) {
      const f = k >> 1;
      const e = items[tu.ids[k]];
      if (e.gen !== tu.ids[k + 1]) continue;
      if (!tu.superFlag[f] && e.dmgSuper > 0) { tu.superFlag[f] = true; tu.superHits += 1; }
      // ×½ — 피해는 들어갔는데 ×2 지분이 0 이면 «상성이 아닌» 히트다. 이 스텝의 표적은 전부 역상성이므로 그것이 곧 ×½ 다.
      if (!tu.resistFlag[f] && e.dmgSuper === 0 && e.dmgTotal > 0) { tu.resistFlag[f] = true; tu.resistHits += 1; }
    }
  }
  if (STANCE_ELEMENTS.indexOf(p.stance) >= 0 && tu.stanceSeen.indexOf(p.stance) < 0) tu.stanceSeen.push(p.stance);
  if (terrainUnder(world, p.x, p.y) !== null) tu.terrainHits += 1;

  // terrain 스텝 — 배울 장판을 «직접» 놓는다(스테이지의 지형 스폰 규칙은 런의 것이다, §8.21).
  if (st.goal.kind === 'terrain' && world.terrain.live === 0 && tu.terrainHits === 0) {
    const t = world.terrain.alloc();
    if (t !== null) {
      t.kind = TERRAIN_KINDS.indexOf('slow');
      t.radius = world.data.rules.terrain.radiusPx;
      t.x = world.data.rules.view.arena.x + world.data.rules.view.arena.w / 2;
      t.y = p.y - 160;
      t.fadeT = -1;
    }
  }

  // boss 스텝 — 코어 봉인을 매 틱 갱신한다(§8.13 하드 게이트와 같은 규칙: 모듈이 하나라도 살아 있으면 코어 무적)
  const core = coreAliveOfStep(world);
  if (core !== null) {
    let mods = 0;
    const items = world.enemies.items;
    for (let k = 0; k < tu.ids.length; k += 2) {
      const e = items[tu.ids[k]];
      if (e.alive && e.gen === tu.ids[k + 1] && e.isBoss && !e.isCore) mods += 1;
    }
    core.sealedNow = mods > 0;
  }

  // ★ 막히지 않는다 — 표적이 필요한 스텝인데 다 죽었고 목표는 아직이면 다시 놓아 준다(REFILL_SEC 뒤).
  //   실측(플레이테스트 2026-09-05): 4단계에서 풀 적 4기를 다 잡았는데 셈이 모자라 «영원히 안 넘어가는» 상태가 됐다.
  if (st.spawn.length > 0 && st.goal.kind !== 'clear' && aliveOfStep(world) === 0 && !goalMet(world, st)) {
    tu.refillT += dt;
    if (tu.refillT >= REFILL_SEC) { tu.refillT = 0; spawnStepEnemies(world, st); }
  } else tu.refillT = 0;

  if (goalMet(world, st)) {
    tu.i += 1;
    tu.entered = false;
    if (tu.i >= world.data.tutorial.steps.length) tu.done = true;
  }
}

export default { makeTutorialState, tickTutorial, tutorialStep, tutorialConfirm, GOALS };
