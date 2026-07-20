/**
 * src/core/score.js — 런 점수 (순수 core, §11.3)
 *
 * 정본 v1.4 구현 절:
 *   §11.3  처치 점수(초효과 보너스) · 시간 보너스 · 보스/중간보스/런 클리어 · 스테이지 무피격 ·
 *          퍼펙트 · 남은 코인 환산 · 난이도 배율 · **floor 는 마지막에 딱 한 번**(roundMode).
 *   §11.3  attribution "damageShare" — 처치의 초효과 보너스는 **막타가 아니라 누적 피해 지분**이다:
 *          그 개체에 넣은 ×2 피해 / 총 피해 ≥ superEffectiveDamageShare(0.5) 면 보너스.
 *   §11.3  shieldPreservesNoHit — 실드가 막은 피격은 무피격을 깨지 않는다.
 *          timeTokenForfeitsTimeBonus — 토큰을 쓴 보스전은 그 시간 보너스가 0.
 *   §11.4  컨티뉴는 퍼펙트와 **모든** 스테이지 무피격을 소급 무효로 만든다.
 *   §9.1   순수성 — window/Date/Math.random 0. import 0 (world 만 만진다).
 *
 * ★ 값은 전부 meta.score 에 산다. 이 파일에 점수 수치가 없다.
 */

/** 런 점수 누적기. createWorld 가 1회 만든다(스테이지 수 = flow.stagePar 길이 = 6). */
export function makeScore(data) {
  const n = data.meta.flow.stagePar.length;
  const noHit = new Array(n);
  for (let i = 0; i < n; i += 1) noHit[i] = true;      // 아직 안 맞았다
  return {
    kills: 0,          // 처치 점수(초효과 보너스 포함)
    bossClear: 0,      // 보스 격파 보너스
    midBossClear: 0,   // 중간보스 격파 보너스
    time: 0,           // 시간 보너스(보스전 잔여 초 × timeBonusPerGameSec)
    runClear: 0,       // 최종 클리어 보너스
    noHit,             // 스테이지별 무피격 유지 여부
    continues: 0,      // 사용한 컨티뉴 수(퍼펙트·무피격을 소급 무효)
  };
}

/**
 * §11.3 — 개체 처치 점수. 그 개체에 넣은 ×2 피해 지분이 기준을 넘으면 초효과 보너스를 더한다.
 *   dmgTotal 이 0(피해 없이 사라진 개체)이면 지분이 정의되지 않으므로 보너스 없음.
 */
export function addKill(world, e) {
  const s = world.data.meta.score;
  let v = e.score;
  if (e.dmgTotal > 0 && e.dmgSuper / e.dmgTotal >= s.superEffectiveDamageShare) {
    v += e.score * s.superEffectiveKillBonusRatio;
  }
  world.score.kills += v;
  // §13.1.1 — 시뮬 텔레메트리(있을 때만). 게임 실행엔 world.tele 가 없다 = 무영향(§10.2).
  const t = world.tele;
  if (t !== undefined) {
    const k = e.archetypeId === '' ? (e.midBossId !== '' ? e.midBossId : 'boss') : e.archetypeId;
    t.kills[k] = (t.kills[k] === undefined ? 0 : t.kills[k]) + 1;
    if (world.run !== undefined && world.run.crisis) t.crisisKills += 1;
  }
}

/**
 * §11.3 — 피격. 실드가 막았으면(absorbed) 무피격을 유지한다(shieldPreservesNoHit).
 *   런이 아직 없으면(테스트 월드) 아무것도 하지 않는다.
 */
export function noteHit(world, absorbed) {
  if (absorbed || world.run === undefined) return;
  const i = world.run.stageIndex;
  if (i >= 0 && i < world.score.noHit.length) world.score.noHit[i] = false;
}

/**
 * §11.3 — 보스 격파. 잔여 타이머가 시간 보너스가 되며, 그 보스전에 시간 토큰을 썼으면 0 이다.
 */
export function addBossClear(world, timerLeftSec, tokenUsed) {
  const s = world.data.meta.score;
  world.score.bossClear += s.bossClearBonus;
  if (!tokenUsed && timerLeftSec > 0) world.score.time += timerLeftSec * s.timeBonusPerGameSec;
}

/** §11.3 — 중간보스 격파 보너스. */
export function addMidBossClear(world) {
  world.score.midBossClear += world.data.meta.score.midBossClearBonus;
}

/** §6.5 — 최종 스테이지 격파 = 런 클리어 보너스. */
export function addRunClear(world) {
  world.score.runClear += world.data.meta.score.runClearBonus;
}

/** §11.4 — 컨티뉴: 퍼펙트 + **모든** 스테이지 무피격을 소급 무효로 만든다. */
export function noteContinue(world) {
  world.score.continues += 1;
  for (let i = 0; i < world.score.noHit.length; i += 1) world.score.noHit[i] = false;
}

/**
 * §11.3 — 최종 집계. 합 × difficulty.scoreMul 를 **마지막에 한 번만** floor 한다(roundMode).
 *   반환은 표시용 내역 + 총점. 죽어도 집계된다(§11.3).
 */
export function tally(world) {
  const s = world.data.meta.score;
  const sc = world.score;

  let noHitCount = 0;
  for (let i = 0; i < sc.noHit.length; i += 1) if (sc.noHit[i]) noHitCount += 1;
  const noHitBonus = noHitCount * s.stageNoHitBonus;
  const perfect = sc.continues === 0 && noHitCount === sc.noHit.length;
  const perfectBonus = perfect ? s.perfectBonus : 0;
  const coinBonus = Math.floor(world.player.coins) * s.coinToScore;

  const raw = sc.kills + sc.bossClear + sc.midBossClear + sc.time
    + sc.runClear + noHitBonus + perfectBonus + coinBonus;

  const diff = world.data.meta.difficulty[world.difficultyId];
  if (diff === undefined) throw new Error(`score: 미지의 난이도 "${world.difficultyId}" (§11.3)`);

  return {
    kills: sc.kills,
    bossClear: sc.bossClear,
    midBossClear: sc.midBossClear,
    time: sc.time,
    runClear: sc.runClear,
    noHitCount,
    noHitBonus,
    perfect,
    perfectBonus,
    coinBonus,
    scoreMul: diff.scoreMul,
    total: Math.floor(raw * diff.scoreMul),      // ★ floor 는 여기 한 번뿐
  };
}
