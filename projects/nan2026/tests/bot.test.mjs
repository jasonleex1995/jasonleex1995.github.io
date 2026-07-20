/**
 * tests/bot.test.mjs — 결정적 AI 플레이어(src/core/bot.js), §10.2 · §10.4.1.
 *
 * 정본 계약:
 *   - 8번째 스트림 `bot` 만 소비한다 → theme/draft/spawn/… 시퀀스를 흔들지 않는다(독립 스트림).
 *   - 출력은 makeInput() 모양(불리언 상태). 게임과 같은 입구로 들어간다.
 *   - 같은 시드·같은 정책 = 같은 입력 시퀀스(헤드리스 재현성의 전제).
 *   - stance 'static' 은 절대 전환하지 않는다(stanceValue 게이트의 대조군).
 *   - forceNoElement 프로브는 속성 카드를 고르지 않는다.
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld, spawnEnemy, spawnEnemyBullet } from '../src/core/state.js';
import { TICK_DT } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { enemies } from '../src/core/enemies.js';
import { emitters } from '../src/core/emitters.js';
import { bossHook } from '../src/core/boss.js';
import { initRun, tickRun } from '../src/core/stage.js';
import { botInput, botDraftPick, botShopPlan, setBotPolicy } from '../src/core/bot.js';

function mkWorld(seed = 1, difficulty) {
  const w = createWorld({
    data: loadData(), seed, weapons, difficulty,
    hooks: { run: tickRun, enemies, emitters, boss: bossHook },
  });
  initRun(w);
  return w;
}
function snap(i) {
  return `${i.left ? 1 : 0}${i.right ? 1 : 0}${i.up ? 1 : 0}${i.down ? 1 : 0}`
    + `${i.stanceNormal ? 1 : 0}${i.stanceFire ? 1 : 0}${i.stanceWater ? 1 : 0}${i.stanceGrass ? 1 : 0}`;
}

suite('bot/결정성 · 스트림', () => {
  test('같은 시드 → 같은 입력 시퀀스', () => {
    function seq(seed) {
      const w = mkWorld(seed);
      const def = loadData().enemies.archetypes.find((a) => a.id === 'drifter');
      spawnEnemy(w, 'drifter', 'water', 600, 200, def.hp, false);
      const out = [];
      for (let t = 0; t < 120; t += 1) out.push(snap(botInput(w, TICK_DT)));
      return out.join('|');
    }
    assert.eq(seq(5), seq(5), '동일 시드 = 동일 입력');
  });

  test('봇은 rng.bot 만 소비한다 — 다른 스트림을 흔들지 않는다 (§10.2 독립)', () => {
    const a = mkWorld(7);
    const before = `${a.rng.spawn.f()},${a.rng.draft.f()},${a.rng.pattern.f()}`;
    const b = mkWorld(7);
    for (let t = 0; t < 300; t += 1) botInput(b, TICK_DT);      // 봇을 300틱 돌린 뒤
    const after = `${b.rng.spawn.f()},${b.rng.draft.f()},${b.rng.pattern.f()}`;
    assert.eq(before, after, '봇 추첨이 spawn/draft/pattern 을 이동시키지 않는다');
  });

  test('출력은 makeInput 모양의 불리언 8필드', () => {
    const w = mkWorld();
    const i = botInput(w, TICK_DT);
    for (const k of ['left', 'right', 'up', 'down', 'stanceNormal', 'stanceFire', 'stanceWater', 'stanceGrass']) {
      assert.eq(typeof i[k], 'boolean', `${k} 는 불리언`);
    }
  });
});

suite('bot/정책', () => {
  test("stance 'static' 은 전환 키를 절대 내지 않는다 (stanceValue 대조군)", () => {
    const w = mkWorld(3);
    setBotPolicy(w, { stance: 'static' });
    const def = loadData().enemies.archetypes.find((a) => a.id === 'drifter');
    for (const el of ['fire', 'water', 'grass']) spawnEnemy(w, 'drifter', el, 600, 200, def.hp, false);
    let switches = 0;
    for (let t = 0; t < 600; t += 1) {
      const i = botInput(w, TICK_DT);
      if (i.stanceFire || i.stanceWater || i.stanceGrass) switches += 1;
    }
    assert.eq(switches, 0, 'static 은 속성 스탠스로 전환하지 않는다');
  });

  test("stance 'greedyNearest' 는 최근접 적을 ×2 로 때리는 스탠스로 전환한다", () => {
    const w = mkWorld(3);
    setBotPolicy(w, { stance: 'greedyNearest' });
    const def = loadData().enemies.archetypes.find((a) => a.id === 'drifter');
    spawnEnemy(w, 'drifter', 'fire', w.player.x, w.player.y - 60, def.hp, false);   // 불 → 물이 정답
    let sawWater = false;
    for (let t = 0; t < 600 && !sawWater; t += 1) if (botInput(w, TICK_DT).stanceWater) sawWater = true;
    assert.ok(sawWater, '불 적에 물 스탠스를 요청한다');
  });

  test('forceNoElement 프로브는 속성 카드를 고르지 않는다', () => {
    const w = mkWorld(2);
    setBotPolicy(w, { forceNoElement: true, draft: 'elementRush' });
    const draft = { cards: [
      { category: 'elementLevel', key: 'e' },
      { category: 'weaponLevel', key: 'w' },
    ] };
    assert.eq(draft.cards[botDraftPick(w, draft)].category, 'weaponLevel', '속성 카드를 건너뛴다');
  });

  test('draft 정책이 선호 카테고리를 바꾼다', () => {
    const cards = [
      { category: 'passive', key: 'p' },
      { category: 'elementLevel', key: 'e' },
      { category: 'newWeapon', key: 'n', weaponId: 'fan' },
    ];
    const w1 = mkWorld(1); setBotPolicy(w1, { draft: 'generalist' });
    assert.eq(cards[botDraftPick(w1, { cards })].category, 'newWeapon', 'generalist = 무기 슬롯 먼저');
    const w2 = mkWorld(1); setBotPolicy(w2, { draft: 'elementRush' });
    assert.eq(cards[botDraftPick(w2, { cards })].category, 'elementLevel', 'elementRush = 속성 먼저');
  });

  test('shop 정책이 구매 계획을 바꾼다', () => {
    const w = mkWorld(1);
    setBotPolicy(w, { shop: 'survivalFirst' });
    assert.eq(botShopPlan(w)[0], 'potion', 'survivalFirst = 생존 우선');
    setBotPolicy(w, { shop: 'spender' });
    assert.gt(botShopPlan(w).length, 4, 'spender = 더 많이 산다');
  });
});

suite('bot/회피', () => {
  test('다가오는 탄을 피해 이동한다', () => {
    const w = mkWorld(4);
    const p = w.player;
    // 플레이어 바로 위에서 정면으로 내려오는 탄
    spawnEnemyBullet(w, 'pelletS', p.x, p.y - 120, 0, 260);
    let moved = false;
    for (let t = 0; t < 60 && !moved; t += 1) {
      const i = botInput(w, TICK_DT);
      if (i.left || i.right || i.up || i.down) moved = true;
    }
    assert.ok(moved, '위협이 있으면 움직인다');
  });

  test('난이도가 높을수록 반응이 느려진다 (latency = 난이도의 유일한 통로, §10.4.1)', () => {
    // 같은 상황에서 normal 과 disaster 의 «첫 반응까지 걸린 틱»을 비교한다
    function firstReactTick(difficulty) {
      const w = mkWorld(8, difficulty);
      const p = w.player;
      spawnEnemyBullet(w, 'pelletS', p.x, p.y - 300, 0, 200);
      for (let t = 0; t < 240; t += 1) {
        const i = botInput(w, TICK_DT);
        if (i.left || i.right || i.up || i.down) return t;
      }
      return 240;
    }
    assert.gte(firstReactTick('disaster'), firstReactTick('normal'), 'disaster 의 눈 감는 창이 더 길다');
  });
});
