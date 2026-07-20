/**
 * tests/render.test.mjs — 렌더 경로가 «한 판 내내» 던지지 않는다는 회귀망.
 *
 * ★ 이 파일이 존재하는 이유(실측된 사고): `drawEnemies` 가 개체의 모양을 `archetypeId` 로 찾는 바람에
 *   보스 코어·파트(archetypeId '')가 화면에 있는 **모든 프레임에서 던졌고**, 렌더가 패널을 그리기도
 *   전에 중단됐다. 순수 core 테스트는 그것을 절대 못 잡는다 — 렌더를 실제로 통과시켜야 잡힌다.
 *
 * 방법: 2D 컨텍스트를 **스텁**으로 세운다(모든 메서드 = noop, 모든 대입 = 삼킴). 그러면 그리기 자체는
 *   아무 일도 안 하지만, **데이터 조회 실패**(미지 shapeId · 없는 팔레트 키 · undefined 필드 접근)는
 *   그대로 예외로 드러난다. 즉 «칠은 안 보고 배선만 본다».
 *
 * 커버: 한 판(잡몹 → 중간보스 → 위기 → 보스 인트로 → 보스)의 전 프레임 + 결정 화면 4종.
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld } from '../src/core/state.js';
import { step, makeInput, TICK_DT } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { enemies } from '../src/core/enemies.js';
import { emitters } from '../src/core/emitters.js';
import { tickRun, initRun } from '../src/core/stage.js';
import { bossHook } from '../src/core/boss.js';
import { resolvePalette, drawWorld, makeInterp, captureInterp, makeFx, updateFx } from '../src/render/draw.js';
import { drawPanels, drawShop, drawResults, drawDeath, drawDraft } from '../src/render/hud.js';
import { buildDraft } from '../src/core/draft.js';
import { tally } from '../src/core/score.js';

/** 모든 호출을 삼키는 2D 컨텍스트. 조회 실패만 통과시킨다. */
function stubCtx() {
  const noop = () => {};
  const target = {
    canvas: { width: 1280, height: 800 },
    createLinearGradient: () => ({ addColorStop: noop }),
    createRadialGradient: () => ({ addColorStop: noop }),
    measureText: () => ({ width: 10 }),
  };
  return new Proxy(target, {
    get(t, k) { return (k in t) ? t[k] : noop; },
    set() { return true; },
  });
}

function mkRun(seed) {
  const world = createWorld({
    data: loadData(), seed, weapons,
    hooks: { enemies, emitters, run: tickRun, boss: bossHook },
  });
  initRun(world);
  world.player.hp = 1e9; world.player.hpMax = 1e9;     // 관측 중 죽지 않게(전 페이즈를 보려면 살아야 한다)
  return world;
}

suite('render — 한 판 전 프레임이 던지지 않는다 (회귀망)', () => {
  test('잡몹 → 중간보스 → 위기 → 보스까지 전 프레임 렌더', () => {
    const w = mkRun(7);
    const pal = resolvePalette(w.data.rules);
    const fx = makeFx(w);
    const interp = makeInterp(w);
    const ctx = stubCtx();
    let frames = 0; let midFrames = 0; let bossFrames = 0;
    const total = Math.floor(190 / TICK_DT);
    for (let i = 0; i < total; i += 1) {
      captureInterp(interp, w);
      step(w, makeInput(), TICK_DT);
      updateFx(fx, w, TICK_DT);
      drawWorld(ctx, w, pal, fx, interp, 1);           // 던지면 여기서 테스트가 깨진다
      drawPanels(ctx, w, pal);
      frames += 1;
      for (const e of w.enemies.items) {
        if (!e.alive) continue;
        if (e.midBossId !== '') { midFrames += 1; break; }
      }
      for (const e of w.enemies.items) {
        if (e.alive && e.isBoss) { bossFrames += 1; break; }
      }
    }
    assert.eq(frames, total, '전 프레임을 그렸다');
    assert.gt(midFrames, 0, '중간보스가 실제로 화면에 있었다 (vacuous 아님)');
    assert.gt(bossFrames, 0, '보스가 실제로 화면에 있었다 (vacuous 아님)');
  });

  test('결정 화면(드래프트·상점·결과·사망)도 던지지 않는다', () => {
    const w = mkRun(3);
    const pal = resolvePalette(w.data.rules);
    const ctx = stubCtx();
    const draft = buildDraft(w);
    drawDraft(ctx, w, pal, draft, 0);
    const ids = Object.keys(w.data.meta.shop);
    assert.gt(ids.length, 0, '상점 품목이 실제로 있다 (vacuous 아님)');
    drawShop(ctx, w, pal, ids, 0, false);
    drawShop(ctx, w, pal, ids, ids.length - 1, true);   // 나가기 확인 상태
    drawResults(ctx, w, pal, tally(w), 'seed-3');
    drawDeath(ctx, w, pal, w.data.meta.flow.continueCost);
    assert.ok(true, '네 화면 모두 예외 없이 통과');
  });
});
