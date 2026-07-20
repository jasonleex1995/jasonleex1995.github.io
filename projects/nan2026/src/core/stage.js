/**
 * src/core/stage.js — 런 오케스트레이션 (순수 core, §6.5 전역 상태 기계의 전투/진행 절반)
 *
 * 정본 v1.4 구현 절:
 *   §8.1   themeDraw — 6 테마 중 5 비복원 추첨(rng.theme) + finale 부착 = 6 포지션. 스테이지 1 은
 *          introOk 필수. **증명(§8.1)**: 1종만 탈락 → 물·불·풀 각 최소 1회 · stage-1 introOk 항상 가능.
 *   §6.3   페이즈 길이(전부 stages.phase / rules.boss 의 게임초). mobPhaseSec 120 · crisisStartSec 95
 *          · introSec 3 · bossTimerSec 180.
 *   §6.5   전역 상태 기계의 **전투/진행 절반** — MOB→BOSS_INTRO→BOSS→STAGE_CLEAR→(다음/승리).
 *          ★ 메뉴·전환 화면(TITLE/DIFFICULTY/THEME_BANNER/HEAL/SHOP/RESULTS)과 DRAFT/SHOP 결정,
 *            그리고 **STAGE_CLEAR→advanceStage** 는 **드라이버(main.js 사람 UI / sim 봇)의 몫**이다.
 *            이 파일은 게임클럭이 흐르는 전투 페이즈만 tickRun 으로 진행한다. 즉 헤드리스 루프는
 *            `while(!over){ if(결정 대기) 드라이버가 해소; else step(); }` 이며 — main.js 와 sim 이
 *            **같은 전이 함수(advanceStage·applyStageClearHeal…)를 호출**하므로 전투는 "certified=shipped"
 *            (§10.4)로 동일 재현된다. STAGE_CLEAR 에서 tickRun 은 스스로 전이하지 않는다(드라이버 대기).
 *   §9.1   순수성 — window/Date/Math.random/… 0. import 는 core 내부만(현재 0). rng 는 world.rng.theme.
 *   §10.3  런 상태는 initRun 1회만 alloc(핫패스 tickRun 은 0 alloc). themeDraw 셔플의 slice 는 런 1회.
 *
 * ★ 배선(B1 다음 슬라이스)에서 채우는 사이드이펙트(이 파일은 신호만 세팅):
 *     - MOB→BOSS_INTRO 전이 시 mobPhaseExitClearBullets + XP 자동수집(phaseEndAutocollect).
 *     - BOSS 진입 시 run.bossSpawned=false 를 보스 훅(boss.js)이 보고 스폰.
 *     - 보스 코어 격파 시 killEnemy(step.js)가 run.cleared=true 를 세팅 → 여기서 STAGE_CLEAR/승리로 소화.
 */

import { addBossClear, addRunClear } from './score.js';

export const PHASE = {
  MOB: 'MOB',                 // 잡몹 페이즈 (내부 마지막 25초 = 위기 서브구간)
  BOSS_INTRO: 'BOSS_INTRO',   // 보스 등장 연출 (무적·무발사·타이머 정지)
  BOSS: 'BOSS',               // 보스전 (180s 타이머)
  STAGE_CLEAR: 'STAGE_CLEAR', // 보스 격파 — 드라이버가 드래프트/회복/상점 후 advanceStage 호출
};

/**
 * §8.1 — 스테이지 순서를 결정적으로 뽑는다(런 시작 1회). rng.theme 만 소비(다른 스트림 불변).
 *   반환 = 6 포지션 [themed×5, finale]. 스테이지 1 은 introOk. 비복원(중복 없음).
 */
function drawStageOrder(world) {
  const td = world.data.stages.themeDraw;
  const stages = world.data.stages.stages;
  const rng = world.rng.theme;

  const introOk = Object.create(null);
  for (let i = 0; i < stages.length; i += 1) introOk[stages[i].id] = stages[i].introOk;

  const pool = td.pool.slice();                       // 6 테마 복사 (런 1회 = slice 허용, §10.3)
  // Fisher-Yates (rng.theme). i 내림차순, j = floor(f·(i+1)) ∈ [0,i]
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng.f() * (i + 1));
    const t = pool[i]; pool[i] = pool[j]; pool[j] = t;
  }

  // 스테이지 1 = 셔플 순서상 첫 introOk 테마(§8.1 stage1RequiresIntroOk). introOk 3종이라 항상 존재.
  let s1 = 0;
  if (td.stage1RequiresIntroOk) {
    s1 = -1;
    for (let i = 0; i < pool.length; i += 1) { if (introOk[pool[i]]) { s1 = i; break; } }
    if (s1 < 0) throw new Error('stage: introOk 테마가 풀에 없다 (§8.1 증명 위반)');
  }

  const order = [pool[s1]];
  for (let i = 0; i < pool.length && order.length < td.count; i += 1) {
    if (i === s1) continue;                            // s1 은 이미 넣었다
    order.push(pool[i]);                               // 셔플 순서대로 채운다 (count-1 개 → 1종 탈락)
  }
  order.push(td.finalStageId);                         // 6번째 = finale (추첨 대상 아님)
  return order;
}

/** 런 상태를 만든다(런 시작 1회). world.run 을 세팅하고 돌려준다. */
export function initRun(world) {
  world.run = {
    order: drawStageOrder(world),   // [6] stage id
    stageIndex: 0,                  // 0..5 (런 포지션)
    phase: PHASE.MOB,
    phaseT: 0,                      // 현재 페이즈 경과(게임초)
    crisis: false,                  // 잡몹 페이즈 마지막 25초 서브구간(§8.10)
    bossTimer: 0,                   // 보스 타이머 잔여(BOSS 진입 시 bossTimerSec)
    bossSpawned: false,             // BOSS 페이즈 보스 스폰 1회 가드(보스 훅이 본다)
    bossPhase: 0,                   // §8.11 보스 페이즈(코어 HP 임계 [0.6,0.3] → 0/1/2, patternSet 선택)
    bossTransitionT: 0,             // 페이즈 전환 잔여(>0 = 보스 무적 + 타이머 정지, §6.3)
    bossMoveSpeedMul: 1,            // §8.12 mobility 파괴 시 speedPxSec ×0.5
    bossMoveAmpMul: 1,              //   그리고 ampPx →0 (스웨이 정지)
    bossTokenUsed: false,           // §11.3 timeTokenForfeitsTimeBonus — 이 보스전에 토큰을 썼는가
    cleared: false,                 // 보스 코어 격파 신호(killEnemy 가 세팅 → tickRun 이 소화)
    won: false,                     // finale 격파 = 런 클리어(승리)
    deathCause: null,               // null | 'hp' | 'timeout'
  };
  return world.run;
}

/** 현재 런 포지션의 stages.json 엔트리. */
export function stageEntry(world) {
  const id = world.run.order[world.run.stageIndex];
  const stages = world.data.stages.stages;
  for (let i = 0; i < stages.length; i += 1) if (stages[i].id === id) return stages[i];
  throw new Error(`stage: 미지의 스테이지 id "${id}" (§9.9)`);
}

/** 현재 포지션이 finale(마지막)인가. */
export function isFinale(world) {
  return world.run.order[world.run.stageIndex] === world.data.stages.themeDraw.finalStageId;
}

/**
 * ★ 런 디렉터 — 게임클럭이 흐르는 페이즈의 시간 진행/전이. step() 가 매 고정 틱 부른다(hooks.run).
 *   전투(스폰·보스)는 enemies/boss 훅이, 결정(드래프트/상점)은 드라이버가 소화한다.
 */
export function tickRun(world, dt) {
  const run = world.run;
  const ph = world.data.stages.phase;
  const boss = world.data.rules.boss;
  run.phaseT += dt;

  if (run.phase === PHASE.MOB) {
    // 위기 서브구간 = 마지막 crisisDurationSec (§8.10). 독립 상태 아님(§6.5).
    run.crisis = run.phaseT >= ph.crisisStartSec;
    if (run.phaseT >= ph.mobPhaseSec) {
      run.crisis = false;
      run.phase = PHASE.BOSS_INTRO;
      run.phaseT = 0;
      // (배선) mobPhaseExitClearBullets + phaseEndAutocollect 는 여기 전이에서 수행한다
    }
    return;
  }

  if (run.phase === PHASE.BOSS_INTRO) {
    // 연출 중 보스 무적·무발사·타이머 정지(§6.3). 시간만 흐른다.
    if (run.phaseT >= boss.introSec) {
      run.phase = PHASE.BOSS;
      run.phaseT = 0;
      run.bossTimer = ph.bossTimerSec;   // timerStartsAfterIntro (§8.11)
      run.bossSpawned = false;           // 보스 훅이 이 틱 이후 스폰
    }
    return;
  }

  if (run.phase === PHASE.BOSS) {
    // ★ 격파 신호를 타이머보다 **먼저** 소화한다 — 이미 코어가 죽은 보스는 타이머 잔여와 무관하게
    //   클리어/승리해야 한다(§6.3). killEnemy 가 collide(run 훅 뒤)에서 run.cleared 를 세팅하므로
    //   신호는 항상 다음 틱에 소비되는데, 그 틱의 bossTimer 감소가 먼저 0 을 넘으면 격파가 시간초과
    //   패배로 뒤집힌다 — 순서를 역전해 막는다.
    if (run.cleared) {
      run.cleared = false;
      // §11.3 — 보스 격파 보너스 + 잔여 타이머의 시간 보너스(토큰을 쓴 보스전은 0)
      addBossClear(world, run.bossTimer, run.bossTokenUsed);
      if (isFinale(world)) { addRunClear(world); run.won = true; world.over = true; return; }
      run.phase = PHASE.STAGE_CLEAR;
      run.phaseT = 0;
      return;
    }
    // §6.3 — 타이머 만료 = 즉사(timerExpire "kill"). 페이즈 전환 중엔 정지(timerPausesOnPhaseTransition).
    if (run.bossTransitionT <= 0) {
      run.bossTimer -= dt;
      if (run.bossTimer <= 0) {
        run.bossTimer = 0;
        run.deathCause = 'timeout';
        world.over = true;
      }
    }
    return;
  }

  // PHASE.STAGE_CLEAR — 드라이버가 advanceStage 를 부를 때까지 대기(게임클럭 정지 = 여기서 아무것도 안 함).
}

/** §2.1 — 스테이지 클리어 회복(flow.stageClearHealPct). 드라이버가 HEAL 연출 시점에 부른다. */
export function applyStageClearHeal(world) {
  const pct = world.data.meta.flow.stageClearHealPct;
  const p = world.player;
  const heal = pct * p.hpMax;
  p.hp += heal;
  if (p.hp > p.hpMax) p.hp = p.hpMax;
}

/**
 * §6.5 — 다음 스테이지로. 드라이버가 STAGE_CLEAR 에서 드래프트 소화·회복·(1~5차)상점 뒤 부른다.
 *   런 포지션을 올리고 MOB 페이즈로 리셋한다. (스폰 상태 리셋·필드 클리어는 배선에서.)
 */
export function advanceStage(world) {
  const run = world.run;
  run.stageIndex += 1;
  run.phase = PHASE.MOB;
  run.phaseT = 0;
  run.crisis = false;
  run.bossTimer = 0;
  run.bossSpawned = false;
  run.cleared = false;
  return run;
}
