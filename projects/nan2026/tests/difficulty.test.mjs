/**
 * tests/difficulty.test.mjs — §11.3 난이도 (v1.10 ㊿).
 *
 * 정본 계약:
 *   난이도는 «필요한 진화 무기 수»다 — 노멀 3 / 하드 4 / 헬 5(meta.difficulty[].evolutionsExpected).
 *   그 요구를 hpMul 이 «적 체력»으로 표현하고, state.difficultyHpMul 이 **유일한 문**이다:
 *   잡몹·중간보스·보스 코어·보스 파트 네 스포너 전부가 이 문을 지난다. 하나라도 빠지면
 *   그 적만 난이도를 안 탄다 = 조용한 구멍.
 *   ★ 난이도 id 를 하드코딩하지 않는다 — 표(meta.difficulty)를 읽는다. 「디재스터」가 그래서 죽었다.
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld, spawnEnemy, spawnMidBoss, spawnBossCore, spawnBossPart, difficultyHpMul } from '../src/core/state.js';
import { weapons } from '../src/core/weapons/index.js';
import { enemies } from '../src/core/enemies.js';
import { emitters } from '../src/core/emitters.js';
import { bossHook } from '../src/core/boss.js';
import { initRun, tickRun } from '../src/core/stage.js';

function mkWorld(difficulty) {
  const w = createWorld({
    data: loadData(), seed: 1, weapons, difficulty,
    hooks: { run: tickRun, enemies, emitters, boss: bossHook },
  });
  initRun(w);
  return w;
}
/** 표에 실재하는 난이도 id (stunMinDifficulty 같은 스칼라 키는 뺀다) */
function tierIds() {
  const d = loadData().meta.difficulty;
  return Object.keys(d).filter((k) => d[k] !== null && typeof d[k] === 'object');
}

suite('difficulty/표 §11.3', () => {
  test('난이도는 셋이고 네 열이 모두 순증한다', () => {
    const d = loadData().meta.difficulty;
    const ids = tierIds();
    assert.eq(ids.length, 3, '난이도 셋 (튜토리얼은 난이도가 아니다)');
    // ㊿ — 저작값(bosses.json·stages.curve)은 «가장 어려운 난이도»의 값이다. 쉬운 난이도는 그 할인.
    assert.eq(d[ids[ids.length - 1]].hpMul, 1, '가장 어려운 난이도가 기준선 — hpMul 1');
    for (let i = 0; i < ids.length - 1; i += 1) assert.gt(1, d[ids[i]].hpMul, `${ids[i]}: hpMul < 1`);
    for (const col of ['speed', 'scoreMul', 'hpMul', 'evolutionsExpected']) {
      for (let i = 1; i < ids.length; i += 1) {
        assert.gt(d[ids[i]][col], d[ids[i - 1]][col], `${col}: ${ids[i]} > ${ids[i - 1]}`);
      }
    }
  });

  test('진화 요구는 정수이고 무기 슬롯 수를 넘지 않는다', () => {
    const d = loadData().meta.difficulty;
    const slots = loadData().rules.player.weaponSlots;
    for (const id of tierIds()) {
      const e = d[id].evolutionsExpected;
      assert.eq(e, Math.round(e), `${id}: 정수`);
      assert.gt(e, 0, `${id}: ≥ 1`);
      assert.gt(slots + 1, e, `${id}: ≤ weaponSlots(${slots})`);
    }
  });

  test('미지 난이도는 소리내어 실패한다 (폴백 금지)', () => {
    const w = mkWorld('nope');
    assert.throws(() => difficultyHpMul(w), '미지 난이도 → throw');
  });
});

suite('difficulty/체력 배율 §11.3', () => {
  test('difficultyHpMul 이 표의 hpMul 을 그대로 준다', () => {
    const d = loadData().meta.difficulty;
    for (const id of tierIds()) assert.eq(difficultyHpMul(mkWorld(id)), d[id].hpMul, `${id}`);
  });

  test('네 스포너 전부가 hpMul 을 탄다 — 잡몹·중간보스·보스 코어·보스 파트', () => {
    const data = loadData();
    const ids = tierIds();
    const hi = ids[0];                              // 저작값과 «다른» 난이도(할인 쪽)로 배율이 실제로 걸리는지 본다
    const mul = data.meta.difficulty[hi].hpMul;
    const arch = data.enemies.archetypes[0];
    const mb = data.bosses.bosses.find((b) => b.tier === 'mid');
    const bs = data.bosses.bosses.find((b) => Array.isArray(b.parts) && b.parts.length > 0);

    const cases = [
      ['잡몹', (w) => spawnEnemy(w, arch.id, 'normal', 600, 200, 1000, false, false)],
      ['중간보스', (w) => spawnMidBoss(w, mb, 'fire', 1000, 600, 200)],
      ['보스 코어', (w) => spawnBossCore(w, bs.id, bs.core, 1000, 600, 200)],
      ['보스 파트', (w) => spawnBossPart(w, bs.id, bs.parts[0], 1000, 600, 200)],
    ];
    for (const [name, spawn] of cases) {
      const base = spawn(mkWorld(ids[ids.length - 1]));
      const disc = spawn(mkWorld(hi));
      assert.eq(base.hp, 1000, `${name}: 가장 어려운 난이도 = 저작값 그대로`);
      assert.near(disc.hp, 1000 * mul, 1e-6, `${name}: hp × hpMul`);
      assert.eq(disc.hpMax, disc.hp, `${name}: hpMax 도 같이 움직인다 (체력바가 거짓말하지 않는다)`);
    }
  });

  test('엘리트 배율과 난이도 배율은 «곱»으로 함께 걸린다', () => {
    const data = loadData();
    const ids = tierIds();
    const hi = ids[0];
    const el = data.rules.elite.hpMult;
    const mul = data.meta.difficulty[hi].hpMul;
    const arch = data.enemies.archetypes[0];
    const e = spawnEnemy(mkWorld(hi), arch.id, 'normal', 600, 200, 1000, true, false);
    assert.near(e.hp, 1000 * el * mul, 1e-6, '엘리트 × 난이도');
  });
});
