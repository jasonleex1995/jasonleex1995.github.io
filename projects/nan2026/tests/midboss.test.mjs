/**
 * tests/midboss.test.mjs — 중간보스(§8.9)의 정본 계약 단위 테스트.
 *
 * 원칙(MEMORY ★★): 값은 데이터/정본에서 유도한다(하드코딩 매직넘버 지양).
 *
 * 커버:
 *   스케줄 — midBossAtSec 시각에 등장 / 스테이지당 마릿수 == curve.midBossCount / 동시 1마리
 *   속성   — notThemeAndNotNormal(테마 속성 아님 · 노말 아님) / 최종 스테이지는 후보 3종 + 비복원
 *   HP     — bosses[].hp × curve.bossHpScale[stageIndex] (★ enemyHpScale 이 아니다)
 *   이탈   — midBossLeaveAfterSec 뒤 사라지고 **보상 0** / midBossForcedLeaveOnCrisis
 *   이동   — anchor: yHoldPx 까지 하강 후 swayAmpPx 왕복
 *            charge: ★ 회귀 — 스폰 라인(아레나 밖 위쪽)에서 시작해도 실제로 돌진한다
 *   소환   — mbNest 만 summon 이 non-null(S17) / everySec 마다 count 마리 / 원점 = 소환자
 *   발사   — §8.9-R8 이미터 2개가 **각자의 스케줄**로 돈다(mbHammer = fan 탄 + zone 장판 둘 다)
 *   처치   — xp/coin 확정 드랍 + 중간보스 격파 점수 + 반납
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld, spawnMidBoss } from '../src/core/state.js';
import { step, makeInput, TICK_DT, killEnemy } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { emitters } from '../src/core/emitters.js';
import { initRun } from '../src/core/stage.js';
import { midBoss } from '../src/core/midboss.js';

const dt = TICK_DT;

function mkRun(seed = 1, withEmitters = false) {
  const world = createWorld({
    data: loadData(), seed, weapons,
    hooks: { enemies: null, emitters: withEmitters ? emitters : null, run: null, boss: null },
  });
  initRun(world);
  world.player.hp = 1e9; world.player.hpMax = 1e9;      // 관측 중 사망하지 않게
  return world;
}
function midOf(world) {
  for (const e of world.enemies.items) if (e.alive && e.midBossId !== '') return e;
  return null;
}
function defOf(world, id) {
  for (const b of world.data.bosses.bosses) if (b.id === id) return b;
  throw new Error(`no boss ${id}`);
}
function midDefs(world) {
  return world.data.bosses.bosses.filter((b) => b.tier === 'mid');
}
/** run 훅 없이 «잡몹 페이즈 시계»만 흘린다(중간보스 계약만 본다) */
function tickMob(world, n) {
  const ph = world.data.stages.phase;
  for (let i = 0; i < n; i += 1) {
    world.run.phaseT += dt;
    world.run.crisis = world.run.phaseT >= ph.crisisStartSec;
    midBoss(world, dt);
    step(world, makeInput(), dt);
  }
}
function livePickups(world) {
  return world.pickups.items.filter((p) => p.alive);
}

// ══════════════════════════════════════════════════════════════════════
suite('midboss — 등장 스케줄 (§8.9)', () => {
  test('midBossAtSec 시각에 등장하고, 그 전엔 없다', () => {
    const w = mkRun(2);
    const at = w.data.stages.phase.midBossAtSec[0][0];
    tickMob(w, Math.floor(at / dt) - 2);
    assert.eq(midOf(w), null, '예정 시각 전엔 없다');
    tickMob(w, 4);
    assert.ne(midOf(w), null, '예정 시각에 등장');
  });

  test('스테이지당 등장 마릿수 == curve.midBossCount', () => {
    const counts = loadData().stages.curve.midBossCount;
    for (let idx = 0; idx < counts.length; idx += 1) {
      const w = mkRun(5 + idx);
      w.run.stageIndex = idx;
      const ph = w.data.stages.phase;
      tickMob(w, Math.floor(ph.crisisStartSec / dt));
      // §8.9(v1.5) 동시 다수 — «등장 수»는 스폰 카운터로 센다(겹쳐 나오므로 null 전이로 못 센다).
      assert.eq(w.run.midBossNext, counts[idx], `스테이지 ${idx + 1} 등장 수 = midBossCount`);
    }
  });

  test('§8.9(v1.5) 동시 다수 — 15초 간격 + 30초 수명 = 겹쳐서 나온다(우르르)', () => {
    const w = mkRun(9);
    w.run.stageIndex = 4;                               // 4마리(22,37,52,67), 15초 간격
    const n = Math.floor(w.data.stages.phase.crisisStartSec / dt);
    let maxLive = 0;
    for (let i = 0; i < n; i += 1) {
      tickMob(w, 1);
      let live = 0;
      for (const e of w.enemies.items) if (e.alive && e.midBossId !== '') live += 1;
      if (live > maxLive) maxLive = live;
    }
    assert.gte(maxLive, 2, '최소 2마리가 동시에 떠 있는 순간이 있다');
  });
});

// ══════════════════════════════════════════════════════════════════════
suite('midboss — 속성 주입 · HP (§8.9)', () => {
  test('themeElseNonTheme — 테마 스테이지는 테마 속성으로 주입 (§8.9 v1.5)', () => {
    const data = loadData();
    let checked = 0;
    for (let seed = 1; seed <= 12; seed += 1) {
      const w = mkRun(seed);
      const stageId = w.run.order[0];
      const theme = data.stages.stages.find((s) => s.id === stageId).element;
      tickMob(w, Math.floor(data.stages.phase.midBossAtSec[0][0] / dt) + 2);
      const e = midOf(w);
      assert.ne(e, null, '등장했다');
      assert.ne(e.element, 'normal', '노말이 아니다');
      assert.eq(e.element, theme, `테마(${theme}) 속성이다 — 중간보스=테마`);
      checked += 1;
    }
    assert.eq(checked, 12, '12 시드 전부 검사했다 (vacuous 아님)');
  });

  test('최종 스테이지(테마 없음) — 주입 후보 3종 전부 + 시드마다 2종 이상(비복원)', () => {
    const data = loadData();
    const lastIdx = data.stages.curve.midBossCount.length - 1;
    const seen = new Set();
    for (let seed = 1; seed <= 10; seed += 1) {
      const w = mkRun(seed);
      w.run.stageIndex = lastIdx;
      const perSeed = new Set();
      const n = Math.floor(data.stages.phase.crisisStartSec / dt);
      for (let i = 0; i < n; i += 1) {
        tickMob(w, 1);
        for (const e of w.enemies.items) {
          if (e.alive && e.midBossId !== '') { perSeed.add(e.element); seen.add(e.element); }
        }
      }
      assert.gte(perSeed.size, 2, '연속 주입은 직전과 다르다 → 최소 2종');
    }
    assert.eq(seen.size, 3, '10 시드에 걸쳐 후보 3종이 전부 나온다(테마가 없으므로)');
  });

  test('hp = bosses[].hp × bossHpScale[stageIndex] (enemyHpScale 이 아니다)', () => {
    const data = loadData();
    const idx = 4;
    const w = mkRun(3);
    w.run.stageIndex = idx;
    tickMob(w, Math.floor(data.stages.phase.midBossAtSec[idx][0] / dt) + 2);
    const e = midOf(w);
    const def = defOf(w, e.midBossId);
    assert.near(e.hpMax, def.hp * data.stages.curve.bossHpScale[idx], 1e-6, 'bossHpScale');
    assert.ne(data.stages.curve.bossHpScale[idx], data.stages.curve.enemyHpScale[idx],
      '두 곡선이 실제로 다르다 (vacuous 아님)');
  });
});

// ══════════════════════════════════════════════════════════════════════
suite('midboss — 이탈 (§8.9 「선택적」의 정의)', () => {
  test('midBossLeaveAfterSec 뒤 사라지고 보상은 0이다', () => {
    const w = mkRun(2);
    const ph = w.data.stages.phase;
    tickMob(w, Math.floor(ph.midBossAtSec[0][0] / dt) + 2);
    const e = midOf(w);
    assert.ne(e, null, '등장했다');
    const before = livePickups(w).length;
    const scoreBefore = w.score.midBossClear;
    // ★ v1.5 — 수명이 다하면 «즉시 반납»이 아니라 위로 서서히 빠져나가는 «퇴장 연출»(mp0=-1)이 시작된다.
    tickMob(w, Math.floor(ph.midBossLeaveAfterSec / dt) + 2);
    assert.eq(e.alive, true, '아직 살아서 위로 빠져나가는 중(연출)');
    assert.eq(e.mp0, -1, '퇴장 상태');
    const yMid = e.y;
    tickMob(w, 20);
    assert.lt(e.y, yMid, '위로 상승 중');
    tickMob(w, 200);                                    // off-screen 까지 충분히
    assert.eq(e.alive, false, '퇴장 완료 = 반납');
    assert.eq(livePickups(w).length, before, '퇴장 = 드랍 0');
    assert.eq(w.score.midBossClear, scoreBefore, '퇴장 = 격파 점수 0');
  });

  test('midBossForcedLeaveOnCrisis — 새떼가 오면 즉시 이탈', () => {
    const w = mkRun(2);
    const ph = w.data.stages.phase;
    assert.eq(ph.midBossForcedLeaveOnCrisis, true, '정본이 강제 이탈을 켜 뒀다');
    // 위기 직전에 등장하도록 시계를 옮긴다(이탈 타이머가 끝나기 전에 위기가 온다)
    w.run.phaseT = ph.crisisStartSec - 2;
    w.run.midBossNext = 0;
    tickMob(w, 2);                                       // 예정 시각을 이미 지났으므로 즉시 등장
    assert.ne(midOf(w), null, '등장했다');
    tickMob(w, Math.floor(2.5 / dt));                    // 위기 진입
    assert.eq(w.run.crisis, true, '위기 구간에 들어왔다');
    assert.eq(midOf(w), null, '강제 이탈');
  });
});

// ══════════════════════════════════════════════════════════════════════
suite('midboss — 이동 (§9.8.2 moveId)', () => {
  test('anchor — yHoldPx 까지 내려와 swayAmpPx 안에서 왕복한다', () => {
    const w = mkRun(1);
    const def = defOf(w, 'mbHammer');
    const mp = def.moveParams;
    const a = w.data.rules.view.arena;
    const e = spawnMidBoss(w, def, 'fire', 1e9, a.x + a.w / 2, w.data.rules.view.spawnLineY);
    let minX = 1e9; let maxX = -1e9; let maxY = -1e9;
    for (let i = 0; i < Math.floor(20 / dt); i += 1) {
      midBoss(w, dt); step(w, makeInput(), dt);
      minX = Math.min(minX, e.x); maxX = Math.max(maxX, e.x); maxY = Math.max(maxY, e.y);
    }
    assert.near(maxY, mp.yHoldPx, 1e-6, 'yHoldPx 에서 멈춘다');
    assert.gt(maxX - minX, mp.swayAmpPx, '실제로 왕복한다 (vacuous 아님)');
    assert.lte(maxX - minX, mp.swayAmpPx * 2 + 1e-6, '진폭은 swayAmpPx 를 넘지 않는다');
  });

  test('회귀: charge — 스폰 라인(아레나 밖 위쪽)에서 시작해도 돌진이 성립한다', () => {
    const w = mkRun(1);
    const def = defOf(w, 'mbLancer');
    const a = w.data.rules.view.arena;
    const e = spawnMidBoss(w, def, 'fire', 1e9, a.x + a.w / 2, w.data.rules.view.spawnLineY);
    assert.lt(w.data.rules.view.spawnLineY, a.y, '스폰 라인은 아레나 위쪽 «밖»이다 (회귀의 전제)');
    let maxY = -1e9;
    for (let i = 0; i < Math.floor(10 / dt); i += 1) {
      midBoss(w, dt); step(w, makeInput(), dt);
      maxY = Math.max(maxY, e.y);
    }
    assert.gt(maxY, a.y + a.h * 0.5, '아레나를 가로질러 돌진한다(조준만 반복하지 않는다)');
  });
});

// ══════════════════════════════════════════════════════════════════════
suite('midboss — 소환 · 발사 (§8.9-R8/R9)', () => {
  test('summon 은 midBossSummonsAllowed 를 통과한 개체만 non-null (S17)', () => {
    const w = mkRun(1);
    const allowed = w.data.rules.boss.midBossSummonsAllowed;
    let nonNull = 0;
    for (const d of midDefs(w)) {
      if (d.summon !== null) { nonNull += 1; assert.eq(allowed.includes(d.id), true, `${d.id} 는 허용 목록에 있다`); }
      else assert.eq(allowed.includes(d.id), false, `${d.id} 는 허용 목록에 없다`);
    }
    assert.gt(nonNull, 0, '소환하는 개체가 실제로 있다 (vacuous 아님)');
  });

  test('mbNest — everySec 마다 count 마리를 소환자 자리에서 낸다', () => {
    const w = mkRun(1);
    const def = defOf(w, 'mbNest');
    const sm = def.summon;
    const a = w.data.rules.view.arena;
    const e = spawnMidBoss(w, def, 'fire', 1e9, a.x + a.w / 2, w.data.rules.view.spawnLineY);
    const ticks = Math.floor(sm.everySec / dt) + 2;
    for (let i = 0; i < ticks; i += 1) { midBoss(w, dt); step(w, makeInput(), dt); }
    const mobs = w.enemies.items.filter((x) => x.alive && x.midBossId === '');
    assert.eq(mobs.length, sm.count, '한 주기에 count 마리');
    for (const m of mobs) assert.eq(m.archetypeId, sm.archetypeId, '지정 아키타입');
    // 원점이 소환자다 — y 는 모함 근처(scatter 의 jitterPx 안)
    const jitter = w.data.stages.formations.scatter.jitterPx;
    for (const m of mobs) assert.lte(Math.abs(m.y - e.y), jitter + 1e-6, '소환자 자리에서 나온다');
  });

  test('§8.9-R8 mbHammer — 이미터 2개가 각자의 스케줄로 돈다(탄 + 장판)', () => {
    const w = mkRun(1, true);
    const def = defOf(w, 'mbHammer');
    assert.eq(def.patternSet[0].emitterIds.length, 2, '이미터가 실제로 2개다 (vacuous 아님)');
    const a = w.data.rules.view.arena;
    const e = spawnMidBoss(w, def, 'fire', 1e9, a.x + a.w / 2, w.data.rules.view.spawnLineY);
    let sawBullet = false; let sawZone = false;
    for (let i = 0; i < Math.floor(12 / dt); i += 1) {
      midBoss(w, dt); step(w, makeInput(), dt);
      if (w.enemyBullets.live > 0) sawBullet = true;
      for (const z of w.zones.items) if (z.alive && !z.fromPlayer) sawZone = true;
    }
    assert.eq(sawBullet, true, 'fan 이미터가 쐈다');
    assert.eq(sawZone, true, 'zone 이미터가 장판을 깔았다');
    assert.gt(e.emitPhase, 0, '1번 악절이 돌았다');
    assert.gt(e.emitPhase2, 0, '2번 악절도 따로 돌았다');
  });
});

// ══════════════════════════════════════════════════════════════════════
suite('midboss — 처치 보상 (§8.9, 거처 = bosses[] 개체 필드)', () => {
  test('xp·coin 확정 드랍 + 격파 점수 + 반납', () => {
    const w = mkRun(1);
    const def = defOf(w, 'mbHammer');
    const e = spawnMidBoss(w, def, 'fire', 10, 300, 300);
    const before = w.score.midBossClear;
    killEnemy(w, e);
    const ps = livePickups(w);
    const xp = ps.filter((p) => p.kind === 'xp').reduce((s, p) => s + p.value, 0);
    const coin = ps.filter((p) => p.kind === 'coin').reduce((s, p) => s + p.value, 0);
    assert.eq(xp, def.xp, 'xp 는 개체 필드가 소유');
    assert.eq(coin, def.coin, 'coin 은 개체 필드가 소유');
    assert.eq(w.score.midBossClear - before, w.data.meta.score.midBossClearBonus, '격파 보너스');
    assert.eq(e.alive, false, '반납');
  });
});
