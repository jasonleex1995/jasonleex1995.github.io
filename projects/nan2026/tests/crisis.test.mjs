/**
 * tests/crisis.test.mjs — §8.10 위기 세션(새떼) 편성·속성·웨이브 정지.
 *
 * 정본 계약:
 *   - 총량 = crisisTotal(60) × swarmTotalScale[포지션], 서브웨이브 crisisSubWaves(6) 파.
 *   - 편성 = 서브웨이브당 swarmChaff 9 + swarmLancer 1 (9:1). 스케일링이 랜서를 0 으로 잘라선 안 된다.
 *   - 속성 = themePure(전부 테마) | finaleRotating(1·2 물 → 3·4 불 → 5·6 풀, §8.16).
 *   - crisisSuspendsWaves — 위기 중 정상 웨이브 정지.
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld } from '../src/core/state.js';
import { TICK_DT } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { enemies } from '../src/core/enemies.js';
import { emitters } from '../src/core/emitters.js';
import { bossHook } from '../src/core/boss.js';
import { initRun, tickRun, PHASE } from '../src/core/stage.js';

function mkCrisisWorld(seed, stageId, pos) {
  const w = createWorld({ data: loadData(), seed, weapons, hooks: { run: tickRun, enemies, emitters, boss: bossHook } });
  initRun(w);
  w.run.order[pos] = stageId;
  w.run.stageIndex = pos;
  w.run.phase = PHASE.MOB;
  w.run.crisis = true;
  return w;
}
function swarmStats(w) {
  let chaff = 0; let lancer = 0; const elems = new Set();
  for (const e of w.enemies.items) {
    if (!e.alive || !e.archetypeId.startsWith('swarm')) continue;
    elems.add(e.element);
    if (e.archetypeId === 'swarmLancer') lancer += 1; else chaff += 1;
  }
  return { chaff, lancer, total: chaff + lancer, elems: [...elems].sort() };
}

suite('crisis/§8.10 새떼', () => {
  test('총량 = round(crisisTotal × swarmTotalScale) 이고 랜서가 잘리지 않는다 (전 포지션)', () => {
    const d = loadData(); const ph = d.stages.phase; const sc = d.stages.curve.swarmTotalScale;
    for (let pos = 0; pos < sc.length; pos += 1) {
      const w = mkCrisisWorld(1, pos === 5 ? 'finale' : 'sea', pos);
      w.run.phaseT = ph.crisisStartSec + ph.crisisDurationSec;    // 6파 전부 캐치업
      enemies(w, TICK_DT);
      const st = swarmStats(w);
      assert.eq(st.total, Math.round(ph.crisisTotal * sc[pos]), `pos${pos}(scale ${sc[pos]}) 총량`);
      assert.gt(st.lancer, 0, `pos${pos} 랜서 > 0 (9:1 편성이 스케일링에 잘리지 않는다)`);
      assert.gt(st.chaff, st.lancer, `pos${pos} chaff 가 다수 (9:1)`);
    }
  });

  test('정확히 crisisSubWaves 파 — 시간이 더 지나도 추가 스폰 없음', () => {
    const d = loadData(); const ph = d.stages.phase;
    const w = mkCrisisWorld(3, 'sea', 3);
    w.run.phaseT = ph.crisisStartSec + ph.crisisDurationSec * 2;   // 위기 길이의 2배가 지나도
    enemies(w, TICK_DT);
    assert.eq(w.spawner.crisisSpawned, ph.crisisSubWaves, '정확히 6파');
    // ★ 총량은 crisisTotal 그 자체가 아니라 «스테이지 곡선을 태운 값»이다 (§8.10 swarmTotalScale).
    //   하드코딩하면 곡선을 만질 때마다 이 테스트가 «실패»가 아니라 «거짓말»을 한다.
    const scale = d.stages.curve.swarmTotalScale[2];               // 스테이지 3 = 인덱스 2
    const want = Math.round(ph.crisisTotal * scale);
    assert.eq(swarmStats(w).total, want, `총 ${want}기 (상한 초과 스폰 없음)`);
  });

  test('themePure = 전부 테마 속성 (sea → 물)', () => {
    const d = loadData(); const ph = d.stages.phase;
    const w = mkCrisisWorld(1, 'sea', 0);
    w.run.phaseT = ph.crisisStartSec + ph.crisisDurationSec;
    enemies(w, TICK_DT);
    assert.deepEq(swarmStats(w).elems, ['water'], 'sea 위기 = 전부 물 (정답 스탠스의 페이오프)');
  });

  test('finaleRotating = 서브웨이브 1·2 물 → 3·4 불 → 5·6 풀', () => {
    const d = loadData(); const ph = d.stages.phase;
    const interval = ph.crisisDurationSec / ph.crisisSubWaves;
    const want = ['water', 'water', 'fire', 'fire', 'grass', 'grass'];
    for (let k = 0; k < ph.crisisSubWaves; k += 1) {
      const w = mkCrisisWorld(1, 'finale', 5);
      // 서브웨이브 창의 **중간점**을 쓴다 — 경계 정각은 부동소수로 floor 가 한쪽으로 떨어진다
      w.run.phaseT = ph.crisisStartSec + (k + 0.5) * interval;     // k+1 파까지 캐치업
      enemies(w, TICK_DT);
      assert.ok(swarmStats(w).elems.includes(want[k]), `서브웨이브 ${k + 1} 속성에 ${want[k]} 포함`);
    }
  });

  test('crisisSuspendsWaves — 위기 중엔 정상 웨이브를 스폰하지 않는다', () => {
    const d = loadData(); const ph = d.stages.phase;
    const w = mkCrisisWorld(2, 'sea', 0);
    w.run.phaseT = ph.crisisStartSec + ph.crisisDurationSec;
    for (let t = 0; t < 60; t += 1) enemies(w, TICK_DT);
    let nonSwarm = 0;
    for (const e of w.enemies.items) if (e.alive && !e.archetypeId.startsWith('swarm')) nonSwarm += 1;
    assert.eq(nonSwarm, 0, '위기 중 비-새떼 스폰 0');
  });

  test('결정성: 같은 시드·포지션 → 같은 편성', () => {
    const d = loadData(); const ph = d.stages.phase;
    function run() {
      const w = mkCrisisWorld(7, 'sea', 2);
      w.run.phaseT = ph.crisisStartSec + ph.crisisDurationSec;
      enemies(w, TICK_DT);
      const st = swarmStats(w);
      return `${st.chaff}/${st.lancer}`;
    }
    assert.eq(run(), run(), '동일 시드 = 동일 편성');
  });
});
