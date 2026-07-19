/**
 * tests/boss.test.mjs — 복합 보스(boss.js) + 처치 규칙(step.killBossEntity)의 정본(§8.11~§8.14) 단위 테스트.
 *
 * 커버:
 *   spawnBoss — 코어+파트 스폰 · HP × bossHpScale[포지션] · finale 절대HP · aliveArmorPartCount = armor 수
 *   killBossEntity — armor 파트 처치 = 게이트 1단 해제 / mobility 는 불변 / 코어 처치 = cleared + 전 개체 반납
 *   bossHook — BOSS 페이즈 1회 스폰(가드) + 파트가 코어+anchor 추종
 *   clearField — 보스 등장 시 잔존 잡몹·적탄 정리
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld, spawnEnemy, spawnEnemyBullet } from '../src/core/state.js';
import { killEnemy } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { enemies } from '../src/core/enemies.js';
import { emitters } from '../src/core/emitters.js';
import { bossHook, spawnBoss } from '../src/core/boss.js';
import { initRun, tickRun, stageEntry, PHASE } from '../src/core/stage.js';

function mkRunWorld(seed, stageIndex) {
  const w = createWorld({ data: loadData(), seed, weapons, hooks: { run: tickRun, enemies, emitters, boss: bossHook } });
  initRun(w);
  if (stageIndex !== undefined) w.run.stageIndex = stageIndex;
  return w;
}
function bossDefFor(w) {
  const id = stageEntry(w).bossId;
  return w.data.bosses.bosses.find((b) => b.id === id);
}
function scanBoss(w) {
  let core = null; const parts = [];
  for (const e of w.enemies.items) if (e.alive && e.isBoss) { if (e.isCore) core = e; else parts.push(e); }
  return { core, parts };
}
function livePickups(w, kind) {
  let n = 0; for (const p of w.pickups.items) if (p.alive && p.kind === kind) n += 1; return n;
}

suite('boss/spawnBoss', () => {
  test('코어 + 파트 스폰, HP × bossHpScale[포지션], aliveArmorPartCount = armor 수', () => {
    const w = mkRunWorld(1, 0);
    const def = bossDefFor(w);
    assert.eq(def.tier, 'stage', '포지션 0 = 스테이지 보스');
    spawnBoss(w);
    const { core, parts } = scanBoss(w);
    assert.ok(core, '코어 스폰됨');
    assert.eq(parts.length, def.parts.length, `파트 수 = ${def.parts.length}`);
    const scale = w.data.stages.curve.bossHpScale[0];
    assert.near(core.hp, def.core.hp * scale, 1e-6, '코어 HP = core.hp × bossHpScale[0]');
    assert.near(core.hpMax, core.hp, 1e-9, 'hpMax = hp');
    const armorCount = def.parts.filter((p) => p.partType === 'armor').length;
    assert.eq(core.aliveArmorPartCount, armorCount, 'aliveArmorPartCount = armor 파트 수');
  });

  test('후반 포지션: 코어 HP 가 bossHpScale 로 커진다 (포지션 3)', () => {
    const w = mkRunWorld(1, 3);                       // 여전히 themed 스테이지(0..4)
    const def = bossDefFor(w);
    spawnBoss(w);
    const { core } = scanBoss(w);
    assert.near(core.hp, def.core.hp * w.data.stages.curve.bossHpScale[3], 1e-6, '포지션 3 스케일');
    assert.gt(w.data.stages.curve.bossHpScale[3], w.data.stages.curve.bossHpScale[0], '진행할수록 커진다');
  });

  test('finale(tetrarch): 절대 HP (bossHpScale 미적용)', () => {
    const w = mkRunWorld(1, 5);                       // finale 포지션
    const def = w.data.bosses.bosses.find((b) => b.tier === 'final');
    spawnBoss(w);
    const { core } = scanBoss(w);
    assert.near(core.hp, def.core.hp, 1e-6, 'finale 코어 HP = 절대값(스케일 없음)');
  });

  test('파트가 코어+anchor 위치에 배치된다', () => {
    const w = mkRunWorld(2, 0);
    spawnBoss(w);
    const { core, parts } = scanBoss(w);
    for (const p of parts) {
      assert.near(p.x, core.x + p.anchorX, 1e-6, '파트 x = 코어+anchorX');
      assert.near(p.y, core.y + p.anchorY, 1e-6, '파트 y = 코어+anchorY');
    }
  });
});

suite('boss/처치 규칙 (killBossEntity)', () => {
  test('armor 파트 처치 → 코어 aliveArmorPartCount −1 (§3.1-4 게이트 1단 해제), 코어 생존', () => {
    const w = mkRunWorld(3, 0);
    spawnBoss(w);
    const { core, parts } = scanBoss(w);
    const armor = parts.filter((p) => p.partType === 'armor');
    assert.gt(armor.length, 0, 'armor 파트 존재 (양성 경로)');
    const before = core.aliveArmorPartCount;
    killEnemy(w, armor[0]);
    assert.eq(core.aliveArmorPartCount, before - 1, 'armor 처치 = 게이트 1단 해제');
    assert.ok(!armor[0].alive, '파트 반납됨');
    assert.ok(core.alive, '코어 생존 (파트 파괴 ≠ 보스 사망)');
    assert.eq(w.run.cleared, false, '파트 파괴는 클리어 아님');
  });

  test('비-armor 파트(mobility/armament) 처치는 aliveArmorPartCount 불변 (armor 만 게이트)', () => {
    const w = mkRunWorld(4, 0);
    spawnBoss(w);
    const { core, parts } = scanBoss(w);
    const nonArmor = parts.filter((p) => p.partType !== 'armor');   // 스테이지 보스 = 2 armor + 1 비-armor
    assert.gt(nonArmor.length, 0, '비-armor 파트 존재 (양성 경로)');
    const before = core.aliveArmorPartCount;
    killEnemy(w, nonArmor[0]);
    assert.eq(core.aliveArmorPartCount, before, '비-armor 처치 = 게이트 불변');
  });

  test('코어 처치 → run.cleared + 모든 보스 개체 반납 + boss.coin 드랍', () => {
    const w = mkRunWorld(5, 0);
    spawnBoss(w);
    const { core } = scanBoss(w);
    const coinsBefore = livePickups(w, 'coin');
    killEnemy(w, core);
    assert.ok(w.run.cleared, 'stage clear 신호');
    let bossLeft = 0; for (const e of w.enemies.items) if (e.alive && e.isBoss) bossLeft += 1;
    assert.eq(bossLeft, 0, '코어+모든 파트 반납');
    assert.gt(livePickups(w, 'coin'), coinsBefore, 'boss.coin 픽업 드랍');
  });

  test('보스 처치는 잡몹 드랍(xp) 경로를 타지 않는다', () => {
    const w = mkRunWorld(6, 0);
    spawnBoss(w);
    const { core } = scanBoss(w);
    const xpBefore = livePickups(w, 'xp');
    killEnemy(w, core);
    assert.eq(livePickups(w, 'xp'), xpBefore, '보스 개체는 xp 드랍 없음 (§8.11 xp 0)');
  });
});

suite('boss/bossHook · clearField', () => {
  test('bossHook: BOSS 페이즈서 1회만 스폰(가드), 비-BOSS 페이즈선 아무것도 안 함', () => {
    const w = mkRunWorld(1, 0);
    w.run.phase = PHASE.MOB;
    bossHook(w);
    assert.eq(scanBoss(w).core, null, 'MOB 페이즈선 보스 스폰 없음');

    w.run.phase = PHASE.BOSS; w.run.bossSpawned = false;
    bossHook(w);
    assert.ok(w.run.bossSpawned, '스폰 후 가드 세팅');
    assert.ok(scanBoss(w).core, '코어 스폰됨');
    bossHook(w);                                       // 재호출
    let cores = 0; for (const e of w.enemies.items) if (e.alive && e.isBoss && e.isCore) cores += 1;
    assert.eq(cores, 1, '재스폰 없음 (bossSpawned 가드)');
  });

  test('clearField: 보스 등장 시 잔존 잡몹·적탄 정리', () => {
    const w = mkRunWorld(1, 0);
    const def = loadData().enemies.archetypes.find((a) => a.id === 'drifter');
    spawnEnemy(w, 'drifter', 'normal', 600, 200, def.hp, false);
    spawnEnemy(w, 'drifter', 'water', 620, 220, def.hp, false);
    spawnEnemyBullet(w, 'pelletS', 600, 300, 0, 100);
    assert.gte(w.enemies.live, 2, '잡몹 존재');
    assert.gte(w.enemyBullets.live, 1, '적탄 존재');
    spawnBoss(w);
    let mobs = 0; for (const e of w.enemies.items) if (e.alive && !e.isBoss) mobs += 1;
    assert.eq(mobs, 0, '잡몹 전부 정리됨');
    assert.eq(w.enemyBullets.live, 0, '적탄 전부 정리됨');
    assert.ok(scanBoss(w).core, '보스는 남는다');
  });
});
