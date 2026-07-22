/**
 * tests/score.test.mjs — §11.3 런 점수 (src/core/score.js).
 *
 * 정본 계약:
 *   처치 점수의 초효과 보너스는 **막타가 아니라 누적 피해 지분**(attribution "damageShare").
 *   실드가 막은 피격은 무피격을 유지(shieldPreservesNoHit) · 토큰을 쓴 보스전은 시간 보너스 0.
 *   컨티뉴는 퍼펙트 + 모든 스테이지 무피격을 소급 무효. floor 는 마지막에 딱 한 번(roundMode).
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld, spawnEnemy } from '../src/core/state.js';
import { killEnemy, applyHit } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { enemies } from '../src/core/enemies.js';
import { emitters } from '../src/core/emitters.js';
import { bossHook } from '../src/core/boss.js';
import { initRun, tickRun } from '../src/core/stage.js';
import { addKill, addBossClear, addMidBossClear, addRunClear, noteContinue, noteHit, tally } from '../src/core/score.js';

function mkWorld(difficulty) {
  const w = createWorld({
    data: loadData(), seed: 1, weapons, difficulty,
    hooks: { run: tickRun, enemies, emitters, boss: bossHook },
  });
  initRun(w);
  return w;
}
const S = () => loadData().meta.score;

suite('score/처치 §11.3', () => {
  test('초효과 보너스는 누적 피해 지분 기준 (막타 아님)', () => {
    const w = mkWorld();
    const s = S();
    const def = loadData().enemies.archetypes.find((a) => a.id === 'drifter');

    // 지분 미달 — ×2 피해가 절반 미만
    const a = spawnEnemy(w, 'drifter', 'normal', 600, 200, def.hp, false);
    a.dmgTotal = 100; a.dmgSuper = 10;
    addKill(w, a);
    assert.eq(w.score.kills, a.score, '지분 미달 = 보너스 없음');

    // 지분 충족 — 정확히 기준선도 포함(≥)
    const b = spawnEnemy(w, 'drifter', 'normal', 620, 200, def.hp, false);
    b.dmgTotal = 100; b.dmgSuper = 100 * s.superEffectiveDamageShare;
    const before = w.score.kills;
    addKill(w, b);
    assert.near(w.score.kills - before, b.score * (1 + s.superEffectiveKillBonusRatio), 1e-9,
      '지분 충족 = 점수 × (1 + 보너스비율)');
  });

  test('피해가 0인 개체는 보너스 없음 (지분이 정의되지 않는다)', () => {
    const w = mkWorld();
    const def = loadData().enemies.archetypes.find((a) => a.id === 'drifter');
    const e = spawnEnemy(w, 'drifter', 'normal', 600, 200, def.hp, false);
    e.dmgTotal = 0; e.dmgSuper = 0;
    addKill(w, e);
    assert.eq(w.score.kills, e.score, '지분 미정의 = 기본 점수만');
  });

  test('killEnemy 가 처치 점수를 준다 (경로 연결)', () => {
    const w = mkWorld();
    const def = loadData().enemies.archetypes.find((a) => a.id === 'drifter');
    const e = spawnEnemy(w, 'drifter', 'normal', 600, 200, def.hp, false);
    const sc = e.score;
    killEnemy(w, e);
    assert.eq(w.score.kills, sc, 'killEnemy → addKill');
  });
});

suite('score/무피격 · 보너스', () => {
  test('실드가 막은 피격은 무피격을 유지한다 (shieldPreservesNoHit)', () => {
    const w = mkWorld();
    w.player.shields = 1;
    applyHit(w, 10);
    assert.ok(w.score.noHit[0], '실드 흡수 = 무피격 유지');
    w.player.iframeSec = 0;
    applyHit(w, 10);                                    // 이번엔 실드 없음
    assert.eq(w.score.noHit[0], false, '실제 피격 = 무피격 깨짐');
  });

  test('보스 격파 = 격파 보너스 + 잔여 타이머 × timeBonusPerGameSec', () => {
    const w = mkWorld();
    const s = S();
    addBossClear(w, 40, false);
    assert.eq(w.score.bossClear, s.bossClearBonus, '격파 보너스');
    assert.eq(w.score.time, 40 * s.timeBonusPerGameSec, '시간 보너스 = 잔여 × 계수');
  });

  test('토큰을 쓴 보스전은 시간 보너스 0 (timeTokenForfeitsTimeBonus)', () => {
    const w = mkWorld();
    addBossClear(w, 40, true);
    assert.eq(w.score.time, 0, '토큰 사용 = 시간 보너스 몰수');
    assert.eq(w.score.bossClear, S().bossClearBonus, '격파 보너스는 그대로');
  });

  test('중간보스 · 런 클리어 보너스', () => {
    const w = mkWorld();
    addMidBossClear(w); addRunClear(w);
    assert.eq(w.score.midBossClear, S().midBossClearBonus, '중간보스');
    assert.eq(w.score.runClear, S().runClearBonus, '런 클리어');
  });
});

suite('score/집계 tally', () => {
  test('전 스테이지 무피격 + 컨티뉴 0 + 승리 = 퍼펙트', () => {
    const w = mkWorld();
    w.run.won = true;                                   // ★ 퍼펙트는 승리한 런에만 성립(§11.3)
    const t = tally(w);
    assert.eq(t.noHitCount, w.score.noHit.length, '6/6 무피격');
    assert.ok(t.perfect, '퍼펙트 성립');
    assert.eq(t.perfectBonus, S().perfectBonus, '퍼펙트 보너스');
  });

  test('미승리 런은 무피격이어도 퍼펙트가 아니다 (타임아웃 익스플로잇 방지)', () => {
    const w = mkWorld();
    w.run.won = false; w.run.stageIndex = 2;            // 스테이지 3에서 사망(1~2만 클리어)
    const t = tally(w);
    assert.eq(t.noHitCount, 2, '클리어한 2 스테이지만 무피격으로 센다');
    assert.ok(!t.perfect, '미승리 = 퍼펙트 아님');
    assert.eq(t.perfectBonus, 0, '퍼펙트 보너스 0');
  });

  test('컨티뉴는 퍼펙트 + 모든 무피격을 소급 무효로 만든다 (§11.4)', () => {
    const w = mkWorld();
    noteContinue(w);
    const t = tally(w);
    assert.eq(t.noHitCount, 0, '모든 스테이지 무피격 무효');
    assert.eq(t.noHitBonus, 0, '무피격 보너스 0');
    assert.ok(!t.perfect, '퍼펙트 상실');
    assert.eq(t.perfectBonus, 0, '퍼펙트 보너스 0');
  });

  test('남은 코인 환산 + 난이도 배율, floor 는 마지막 한 번', () => {
    const s = S();
    const w = mkWorld('hell');                          // scoreMul 2.5
    // 소수 코인을 만들어 「마지막에 한 번 floor」를 시험한다
    w.player.coins = 10.7;
    for (let i = 0; i < w.score.noHit.length; i += 1) w.score.noHit[i] = false;  // 보너스 격리
    w.score.kills = 1;
    const t = tally(w);
    const raw = 1 + 10.7 * s.coinToScore;              // ★ floor 는 마지막 한 번뿐 — 코인을 미리 안 floor 한다
    assert.eq(t.coinBonus, 10.7 * s.coinToScore, '코인은 floor 없이 환산(단일 floor 규약)');
    assert.eq(t.scoreMul, loadData().meta.difficulty.hell.scoreMul, '난이도 배율');
    assert.eq(t.total, Math.floor(raw * t.scoreMul), '총점 = floor(합 × 배율) — floor 1회');
  });

  test('난이도별 배율이 총점에 반영된다', () => {
    const mk = (d) => { const w = mkWorld(d); w.score.kills = 1000; return tally(w).total; };
    assert.gt(mk('disaster'), mk('normal'), 'disaster > normal');
    assert.gt(mk('hell'), mk('hard'), 'hell > hard');
  });

  test('미지 난이도는 소리내어 실패', () => {
    const w = mkWorld('nope');
    assert.throws(() => tally(w), '미지 난이도 → throw');
  });
});
