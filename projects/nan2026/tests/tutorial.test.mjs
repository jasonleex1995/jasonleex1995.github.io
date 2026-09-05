/**
 * tests/tutorial.test.mjs — §6.7(v1.10 ㊴) 튜토리얼의 계약.
 *
 * 커버:
 *   · 데이터 — 스텝 id 고유 · 목표 어휘 닫힘 · 마지막은 confirm(플레이어가 끝낸다)
 *   · 진행   — 각 목표가 «실제로» 충족되면 다음 스텝으로 간다(9스텝 전부를 스크립트로 통과시킨다)
 *   · 안전   — 튜토리얼에서는 죽지 않는다(safeHpFloor 아래로 내려가면 회복, world.over 는 항상 false)
 *   · 봉인   — 보스 스텝의 코어는 모듈이 살아 있는 동안 무적이다(§8.13 과 같은 규칙)
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld, spawnEnemy } from '../src/core/state.js';
import { step, makeInput, TICK_DT, killEnemy } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { emitters } from '../src/core/emitters.js';
import { initRun } from '../src/core/stage.js';
import { hitEnemy } from '../src/core/damage.js';
import { stampFor } from '../src/core/stance.js';
import { makeTutorialState, tickTutorial, tutorialStep, tutorialConfirm, GOALS } from '../src/core/tutorial.js';

function mk() {
  const world = createWorld({
    data: loadData(), seed: 1, weapons,
    hooks: { enemies: null, emitters, run: tickTutorial, boss: null },
    startWeaponId: 'forward',
  });
  initRun(world);
  world.tut = makeTutorialState();
  return world;
}
function tick(w, n, input) { for (let i = 0; i < n; i += 1) step(w, input || makeInput(), TICK_DT); }
function killAllMobs(w) { for (const e of w.enemies.items) if (e.alive && !e.isBoss) e.hp = 0; }

suite('tutorial — 데이터 (§6.7)', () => {
  test('스텝 id 는 고유하고, 목표는 닫힌 어휘이며, 마지막 스텝은 confirm 이다', () => {
    const d = loadData();
    const steps = d.tutorial.steps;
    assert.gt(steps.length, 3, '스텝이 여러 개다');
    const ids = new Set();
    for (const s of steps) {
      assert.eq(ids.has(s.id), false, `중복 없음: ${s.id}`);
      ids.add(s.id);
      assert.ok(GOALS.indexOf(s.goal.kind) >= 0, `목표 어휘: ${s.goal.kind}`);
      assert.ok(s.body.length > 0 && s.hint.length > 0, `${s.id}: 설명과 지시가 있다`);
    }
    assert.eq(steps[steps.length - 1].goal.kind, 'confirm', '마지막은 플레이어가 끝낸다');
    // 가르치는 개념이 실제로 들어 있다(공허 통과 방지)
    const kinds = steps.map((s) => s.goal.kind);
    for (const k of ['move', 'clear', 'level', 'superHit', 'stances', 'survive', 'terrain', 'boss']) {
      assert.ok(kinds.indexOf(k) >= 0, `${k} 를 가르친다`);
    }
  });
});

suite('tutorial — 진행 (§6.7)', () => {
  test('9개 스텝을 목표 충족으로 전부 통과한다 (스크립트)', () => {
    const w = mk();
    const steps = w.data.tutorial.steps;
    const guard = 60 * 600;                       // 10분치 틱 상한(무한 루프 방지)
    let t = 0;
    while (!w.tut.done && t < guard) {
      const s = tutorialStep(w);
      if (s === null) break;
      const k = s.goal.kind;
      // 목표별 «플레이어가 할 일»을 스크립트로 대신한다
      if (k === 'move') {
        const inp = makeInput(); inp.left = t % 20 < 10; inp.right = !inp.left;
        tick(w, 1, inp);
      } else if (k === 'clear' || k === 'boss') {
        // ★ hp = 0 은 «죽음»이 아니다 — 처치는 killEnemy 가 완결한다(§9.5 무기 런타임 계약)
        for (const e of w.enemies.items) {
          if (!e.alive) continue;
          if (k === 'clear') killEnemy(w, e);
          else if (e.isBoss && !e.isCore) killEnemy(w, e);
          else if (e.isCore && !e.sealedNow) killEnemy(w, e);
        }
        tick(w, 1);
      } else if (k === 'level') {
        w.player.level = Math.max(w.player.level, s.goal.value);
        tick(w, 1);
      } else if (k === 'superHit') {
        // ×2 히트를 «실제 무기»로 만든다 — [W] 를 눌러 불 스탠스가 되고(각인은 stance.js 가 한다) 풀 적 아래에 선다
        const inp = makeInput();
        if (w.player.stance !== 'fire') inp.stanceFire = true;
        const e = w.enemies.items.find((x) => x.alive && x.element === 'grass');
        if (e) w.player.x = e.x;
        tick(w, 1, inp);
      } else if (k === 'stances') {
        const want = ['fire', 'water', 'grass'][Math.min(w.tut.stanceSeen.length, 2)];
        const inp = makeInput();
        if (w.player.stance !== want) { inp.stanceFire = want === 'fire'; inp.stanceWater = want === 'water'; inp.stanceGrass = want === 'grass'; }
        tick(w, 1, inp);
      } else if (k === 'survive') {
        tick(w, 1);
      } else if (k === 'terrain') {
        // 튜토리얼이 놓아 준 장판 위로 간다
        const ter = w.terrain.items.find((x) => x.alive);
        if (ter) { w.player.x = ter.x; w.player.y = ter.y; }
        tick(w, 1);
      } else if (k === 'confirm') {
        tutorialConfirm(w);
        tick(w, 1);
      } else throw new Error(`테스트가 모르는 목표: ${k}`);
      t += 1;
    }
    assert.eq(w.tut.done, true, `전 스텝 통과 (스텝 ${w.tut.i}/${steps.length}, ${(t / 60).toFixed(0)}초)`);
    assert.lt(t, guard, '상한 안에서 끝났다');
  });

  test('㊵-c 회귀 — ×2 를 맞은 적이 «죽어도» 셈이 되돌아가지 않는다 (4단계에서 안 넘어가던 버그)', () => {
    const w = mk();
    const steps = w.data.tutorial.steps;
    const idx = steps.findIndex((s) => s.goal.kind === 'superHit');
    w.tut.i = idx; w.tut.entered = false;
    tick(w, 1);
    const targets = w.enemies.items.filter((e) => e.alive);
    assert.gt(targets.length, steps[idx].goal.value, '목표보다 적이 많다(vacuous 아님)');
    const ctx = { matrix: w.data.elements.matrix, dmgMulSum: 0, elementBonusMul: 1 };
    const stamp = stampFor(w, 0, 'spawn', 'fire');
    // 한 마리씩 «×2 로 때리고 바로 죽인다» — 옛 판은 살아 있는 개체 수로 세서 첫 마리 뒤로는 하나도 안 세졌다
    let counted = 0;
    for (let k = 0; k < steps[idx].goal.value && w.tut.i === idx; k += 1) {
      const e = w.enemies.items.find((x) => x.alive && x.element === 'grass');
      assert.ok(e !== undefined, `${k + 1}번째 표적이 있다`);
      assert.gt(hitEnemy(w, ctx, 'forward', 1, 1, stamp, e, 0), 0, '×2 히트가 실제로 들어갔다');
      tick(w, 1);                                  // 훅이 래치를 본다
      counted = Math.max(counted, w.tut.superHits);   // ★ 스텝이 넘어가면 카운터는 0 으로 리셋된다 — 넘기 «직전» 값을 본다
      killEnemy(w, e);                             // 그리고 바로 죽인다 — 옛 판은 여기서 셈이 되돌아갔다
      tick(w, 1);
      counted = Math.max(counted, w.tut.superHits);
    }
    assert.gte(counted, steps[idx].goal.value, `×2 ${counted}회가 세졌다(죽어도 되돌아가지 않는다)`);
    assert.gt(w.tut.i, idx, '다음 스텝으로 넘어갔다');
  });

  test('막히지 않는다 — 표적을 다 잡았는데 목표가 남으면 다시 놓아 준다', () => {
    const w = mk();
    const steps = w.data.tutorial.steps;
    const idx = steps.findIndex((s) => s.goal.kind === 'superHit');
    w.tut.i = idx; w.tut.entered = false;
    tick(w, 1);
    for (const e of w.enemies.items) if (e.alive) killEnemy(w, e);   // 목표는 0인 채로 표적만 전멸
    tick(w, 2);
    assert.eq(w.enemies.items.filter((e) => e.alive).length, 0, '한동안은 비어 있다');
    tick(w, Math.ceil(2.0 / TICK_DT));
    assert.gt(w.enemies.items.filter((e) => e.alive).length, 0, '재보급됐다');
    assert.eq(w.tut.i, idx, '스텝은 그대로(목표는 아직)');
  });

  test('스텝은 «깨끗한 판»에서 시작한다 — 앞 스텝의 잔여 적이 남지 않는다', () => {
    const w = mk();
    const steps = w.data.tutorial.steps;
    w.tut.i = steps.findIndex((s) => s.goal.kind === 'level');   // 레벨업으로 끝나므로 적이 남는 스텝
    w.tut.entered = false;
    tick(w, 2);
    const before = w.enemies.items.filter((e) => e.alive).length;
    assert.gt(before, 0, '적이 있다');
    w.player.level = 99;                                          // 목표 즉시 충족 → 다음 스텝
    tick(w, 2);
    const mine = new Set();
    for (let k = 0; k < w.tut.ids.length; k += 2) mine.add(w.tut.ids[k]);
    for (const e of w.enemies.items) if (e.alive) assert.ok(mine.has(e.idx), '무대의 적은 전부 이 스텝의 것');
  });

  test('죽지 않는다 — HP 가 safeHpFloor 아래로 내려가면 가득 채운다 (적이 쏘는 스텝을 통째로 버틴다)', () => {
    const w = mk();
    const steps = w.data.tutorial.steps;
    w.tut.i = steps.findIndex((s) => s.goal.kind === 'survive');
    w.tut.entered = false;
    let minHp = Infinity;
    for (let i = 0; i < 60 * 20; i += 1) {
      w.player.iframeSec = 0;                      // 무적 프레임을 꺼서 최대한 맞게 한다
      tick(w, 1);
      minHp = Math.min(minHp, w.player.hp);
      assert.eq(w.over, false, `t${i}: 사망 없음`);
    }
    assert.gt(minHp, 0, `HP 는 0 에 닿지 않았다 (최저 ${minHp.toFixed(0)})`);
    w.player.hp = 1;
    tick(w, 2);
    assert.eq(w.player.hp, w.player.hpMax, 'safeHpFloor 아래 = 가득 회복');
  });

  test('보스 스텝 — 모듈이 살아 있으면 코어는 무적(§8.13 과 같은 규칙)', () => {
    const w = mk();
    const steps = w.data.tutorial.steps;
    w.tut.i = steps.findIndex((s) => s.goal.kind === 'boss');
    w.tut.entered = false;
    tick(w, 2);
    const core = w.enemies.items.find((e) => e.alive && e.isCore);
    const mods = w.enemies.items.filter((e) => e.alive && e.isBoss && !e.isCore);
    assert.ok(core !== undefined, '코어가 섰다');
    assert.eq(mods.length, 2, '모듈 2기');
    assert.eq(core.sealedNow, true, '모듈이 살아 있으니 코어는 봉인');
    const ctx = { matrix: w.data.elements.matrix, dmgMulSum: 0, elementBonusMul: 1 };
    const hp0 = core.hp;
    assert.eq(hitEnemy(w, ctx, 'forward', 9999, 1, stampFor(w, 0, 'spawn', 'normal'), core, 0), 0, '봉인 코어 = 피해 0');
    assert.eq(core.hp, hp0, '코어 HP 불변');
    for (const m of mods) killEnemy(w, m);
    tick(w, 2);
    assert.eq(core.sealedNow, false, '모듈이 전부 죽으면 봉인 해제');
    assert.gt(hitEnemy(w, ctx, 'forward', 10, 1, stampFor(w, 0, 'spawn', 'normal'), core, 0), 0, '이제 딜이 들어간다');
  });
});
