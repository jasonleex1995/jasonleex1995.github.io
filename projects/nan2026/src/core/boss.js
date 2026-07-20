/**
 * src/core/boss.js — 복합 보스 (순수 core 훅, §8.11~§8.16)
 *
 * B1b 범위 = **스폰 · 이동 · 격파의 골격**. 발사(patternSet 이미터)·페이즈 전환(phaseThresholds)·
 *   파트파괴 이동페널티·중간보스는 B2.
 *
 * 정본 v1.4 구현 절:
 *   §8.11  복합 구조 — 코어 + 파트(적 풀 공유). 코어 hp 0 = 보스 사망. 파트 파괴는 사망 아님.
 *          hp = data.hp × bossHpScale[런포지션] (tier stage/mid) · finale 는 절대값(스케일 없음).
 *   §8.12  partType(mobility/armament/armor/core). armor 파괴 = 코어 소프트게이트 1단 해제(§3.1-4).
 *   §8.12.1 movePattern(sway/orbitArc/holdCenter) — 전신 이동. 파트는 코어+anchor 를 매 틱 따라붙는다.
 *   §8.13  코어 소프트게이트 coreGateMul^(살아있는 armor 수) — damage.js 가 이미 적용(여기선 스폰만).
 *   §10.2  world.rng.boss 만 사용(B2 발사에서). B1b 이동은 결정적(moveT 의 함수, RNG 0).
 *   §9.1   순수성 — window/Date/Math.random 0. import 는 core 내부(state.js·stage.js)만.
 *
 * ★ 처치 규칙(코인·게이트·반납)은 step.killEnemy 의 killBossEntity 가 소유한다(D3 계약과 대칭).
 *   이 파일은 **스폰 신호(run.bossSpawned)에 반응해 스폰**하고 **매 틱 위치를 갱신**할 뿐이다.
 */

import { spawnBossCore, spawnBossPart } from './state.js';
import { PHASE, stageEntry } from './stage.js';

/** bosses.json 에서 id 로 조회. */
function findBoss(world, id) {
  const list = world.data.bosses.bosses;
  for (let i = 0; i < list.length; i += 1) if (list[i].id === id) return list[i];
  throw new Error(`boss: 미지의 보스 "${id}" (§9.8)`);
}

/** 보스 아레나를 깨끗이 — 남은 잡몹(비-보스)과 적 탄을 반납한다(§8.11 등장 직전). */
function clearField(world) {
  const en = world.enemies.items;
  for (let i = 0; i < en.length; i += 1) if (en[i].alive && !en[i].isBoss) world.enemies.release(en[i]);
  const eb = world.enemyBullets.items;
  for (let i = 0; i < eb.length; i += 1) if (eb[i].alive) world.enemyBullets.release(eb[i]);
}

/** 살아있는 보스 코어. 없으면 null(동시 1개). */
function findCore(world) {
  const en = world.enemies.items;
  for (let i = 0; i < en.length; i += 1) { const e = en[i]; if (e.alive && e.isBoss && e.isCore) return e; }
  return null;
}

/**
 * §8.11/§13.6 — 현재 스테이지 보스를 스폰한다. 파트를 먼저(낮은 풀 idx = collide 가 먼저 맞아
 *   코어를 가린다 ≈ partHitPriority outermostFirst), 코어를 나중에 놓는다.
 */
export function spawnBoss(world) {
  const entry = stageEntry(world);
  const def = findBoss(world, entry.bossId);
  const run = world.run;
  run.bossPhase = 0; run.bossTransitionT = 0; run.bossMoveSpeedMul = 1; run.bossMoveAmpMul = 1;  // 새 보스 = 1페이즈
  run.bossTokenUsed = false;
  const scale = def.tier === 'final' ? 1 : world.data.stages.curve.bossHpScale[world.run.stageIndex];
  const arena = world.data.rules.view.arena;
  const cx = arena.x + arena.w / 2;
  const cy = def.movePatternParams.yHoldPx;

  clearField(world);

  let armorCount = 0;
  for (let i = 0; i < def.parts.length; i += 1) if (def.parts[i].partType === 'armor') armorCount += 1;

  for (let i = 0; i < def.parts.length; i += 1) {
    const part = def.parts[i];
    spawnBossPart(world, def.id, part, part.hp * scale, cx, cy);
  }
  spawnBossCore(world, def.id, def.core, def.core.hp * scale, cx, cy, armorCount);
}

/**
 * §8.12.1 — 전신 이동. 코어를 movePattern 대로 움직이고 파트를 코어+anchor 로 따라붙인다.
 *   보스 개체는 vx/vy=0 이며 위치를 여기서 직접 세팅한다(step.moveBullets 의 등속 적분은 무해).
 *   sway: x = cx + ampPx·sin(w·moveT), w = speedPxSec/ampPx → 최대 측속 = speedPxSec.
 *   orbitArc 는 B1b 에서 sway 로 근사(B2 에서 정식 구현).
 */
function moveBoss(world) {
  const core = findCore(world);
  if (core === null) return;
  const run = world.run;
  const def = findBoss(world, core.bossId);
  const mp = def.movePatternParams;
  const arena = world.data.rules.view.arena;
  const cx = arena.x + arena.w / 2;

  // §8.12 — mobility 파괴 시 speedPxSec ×0.5 · ampPx →0(스웨이 정지). 배율은 run 이 소유.
  const effAmp = mp.ampPx * run.bossMoveAmpMul;
  const effSpeed = mp.speedPxSec * run.bossMoveSpeedMul;
  if (def.movePattern === 'holdCenter' || effAmp <= 0) {
    core.x = cx;
    core.y = mp.yHoldPx;
  } else {
    const w = effSpeed / effAmp;                       // 최대 측속 = effSpeed
    core.x = cx + effAmp * Math.sin(w * core.moveT);
    core.y = mp.yHoldPx;
  }

  const en = world.enemies.items;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive || !e.isBoss || e.isCore) continue;
    e.x = core.x + e.anchorX;
    e.y = core.y + e.anchorY;
  }
}

/**
 * §8.11 — 코어 HP 임계(phaseThresholds [0.6,0.3])를 지나면 페이즈 전환.
 *   전환 = phaseTransitionSec 동안 보스 무적(collide) + 타이머 정지(tickRun). 끝나면 전 파트의 phase 를
 *   각인해 patternSet 이 새 페이즈로 바뀌고(이미터 악절도 처음부터), "몰아치고 쉰다"의 층이 오른다.
 */
function advancePhase(world, dt) {
  const run = world.run;
  const core = findCore(world);
  if (core === null) return;

  if (run.bossTransitionT > 0) {
    run.bossTransitionT -= dt;
    if (run.bossTransitionT <= 0) {
      run.bossTransitionT = 0;
      const en = world.enemies.items;
      for (let i = 0; i < en.length; i += 1) {
        const e = en[i];
        if (e.alive && e.isBoss) { e.phase = run.bossPhase; e.emitT = 0; e.emitPhase = 0; }
      }
    }
    return;                                            // 전환 중엔 재판정 안 함
  }

  const thr = world.data.rules.boss.phaseThresholds;   // [0.6, 0.3]
  const ratio = core.hpMax > 0 ? core.hp / core.hpMax : 0;
  let target = 0;
  for (let i = 0; i < thr.length; i += 1) if (ratio < thr[i]) target = i + 1;
  if (target > run.bossPhase) {
    run.bossPhase = target;
    run.bossTransitionT = world.data.rules.boss.phaseTransitionSec;
  }
}

/**
 * ★ 훅 진입점 — step.js 가 매 틱 부른다(hooks.boss). BOSS 페이즈에만 활성.
 *   run.bossSpawned(스테이지.tickRun 이 BOSS 진입 시 false 로 세팅)를 보고 1회 스폰 후 매 틱 이동.
 */
export function bossHook(world, dt) {
  const run = world.run;
  if (run.phase !== PHASE.BOSS) return;
  if (!run.bossSpawned) { spawnBoss(world); run.bossSpawned = true; }
  advancePhase(world, dt);
  moveBoss(world);
}

export default { bossHook };
