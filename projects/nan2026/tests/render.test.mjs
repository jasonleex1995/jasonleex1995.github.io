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
import { createWorld, giveWeapon, levelUpWeapon } from '../src/core/state.js';
import { step, makeInput, TICK_DT } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { enemies } from '../src/core/enemies.js';
import { emitters } from '../src/core/emitters.js';
import { tickRun, initRun } from '../src/core/stage.js';
import { bossHook } from '../src/core/boss.js';
import { resolvePalette, drawWorld, makeInterp, captureInterp, makeFx, updateFx, bulletDensityAlpha } from '../src/render/draw.js';
import { drawPanels, drawResults, drawDraft, wrapLines, passiveWeaponLine, weaponRowLayout } from '../src/render/hud.js';
import { BODY_STATS } from '../src/core/schema.mjs';
import { giveWeapon as giveW, passiveAffectsSlot, recomputeEff } from '../src/core/state.js';
import { buildDraft } from '../src/core/draft.js';
import { tally } from '../src/core/score.js';

/**
 * 브라우저를 흉내 내는 검증 2D 컨텍스트(v1.10 ㊱). 칠은 안 하지만 브라우저가 **던지는 것은 똑같이 던지고**, 브라우저가 조용히
 *   삼키는 잘못은 로그로 남긴다:
 *   · arc/ellipse/arcTo 음수 반지름 = IndexSizeError · roundRect 음수 = RangeError · 그라데이션 비유한 인자 = NotSupportedError
 *   · save/restore 깊이 · globalAlpha · globalCompositeOperation 을 추적한다(프레임이 끝나면 전부 원상이어야 한다)
 *   · fillStyle/strokeStyle 에 색이 아닌 문자열(NaN 등)이 들어오면 로그(브라우저는 무시해 «직전 색»으로 칠한다 = 조용한 버그)
 * ★ 이 스텁이 «전부 삼키는» 스텁이던 때, 둔화 장판 페이드 끝의 음수 반지름 arc() 를 못 잡았다 — 실제 브라우저는 예외를 던져
 *   프레임을 중간에 끊었고, 굳은 globalAlpha(≈0.02)로 다음 프레임의 배경이 칠해져 런 끝까지 «모든 물체가 잔상»을 남겼다
 *   (플레이테스트 스크린샷 · 숲 위기 진입). 스텁은 브라우저만큼 엄격해야 회귀망이 된다.
 */
const COLOR_RE = /^(#[0-9a-fA-F]{3,8}|rgba?\([-\d.e\s,]+\)|hsla?\([-\d.e\s,%]+\))$/;
function stubCtx() {
  const noop = () => {};
  const state = { globalAlpha: 1, globalCompositeOperation: 'source-over', fillStyle: '#000', strokeStyle: '#000', lineWidth: 1 };
  const stack = [];
  const log = [];
  const grad = { addColorStop: (o, c) => { if (typeof c === 'string' && !COLOR_RE.test(c)) log.push(`gradient color ${c}`); } };
  const finite = (...a) => a.every((v) => Number.isFinite(v));
  const target = {
    canvas: { width: 1280, height: 800 },
    save: () => { stack.push({ ...state }); },
    restore: () => { if (stack.length === 0) { log.push('restore on empty stack'); return; } Object.assign(state, stack.pop()); },
    arc: (x, y, r) => { if (!finite(x, y, r)) return; if (r < 0) throw new Error(`IndexSizeError: arc radius ${r}`); },
    ellipse: (x, y, rx, ry) => { if (!finite(x, y, rx, ry)) return; if (rx < 0 || ry < 0) throw new Error(`IndexSizeError: ellipse ${rx},${ry}`); },
    arcTo: (x1, y1, x2, y2, r) => { if (!finite(x1, y1, x2, y2, r)) return; if (r < 0) throw new Error(`IndexSizeError: arcTo ${r}`); },
    roundRect: (x, y, w, h, r) => { if (typeof r === 'number' && r < 0) throw new RangeError(`roundRect ${r}`); },
    createLinearGradient: (...a) => { if (!finite(...a)) throw new Error(`NotSupportedError: linear gradient ${a}`); return grad; },
    createRadialGradient: (x0, y0, r0, x1, y1, r1) => {
      if (!finite(x0, y0, r0, x1, y1, r1)) throw new Error(`NotSupportedError: radial gradient ${[x0, y0, r0, x1, y1, r1]}`);
      if (r0 < 0 || r1 < 0) throw new Error(`IndexSizeError: radial gradient ${r0},${r1}`);
      return grad;
    },
    measureText: () => ({ width: 10 }),
    __depth: () => stack.length,
    __state: state,
    __log: log,
  };
  return new Proxy(target, {
    get(t, k) { if (k in t) return t[k]; if (k in state) return state[k]; return noop; },
    set(t, k, v) {
      if ((k === 'fillStyle' || k === 'strokeStyle' || k === 'shadowColor') && typeof v === 'string' && !COLOR_RE.test(v)) log.push(`${k} ${v}`);
      if (k === 'globalAlpha' && !(v >= 0 && v <= 1)) log.push(`globalAlpha ${v}`);
      state[k] = v; return true;
    },
  });
}

/** 한 프레임 뒤의 캔버스 상태가 원상인지 — 깊이 0 · 알파 1 · source-over · 잘못된 색 0. 아니면 그 프레임 번호로 실패한다. */
function assertClean(ctx, frame) {
  assert.eq(ctx.__depth(), 0, `f${frame}: save/restore 균형`);
  assert.eq(ctx.__state.globalAlpha, 1, `f${frame}: globalAlpha 원상`);
  assert.eq(ctx.__state.globalCompositeOperation, 'source-over', `f${frame}: 합성 모드 원상`);
  assert.eq(ctx.__log.length, 0, `f${frame}: 잘못된 색/알파 0 — ${ctx.__log.slice(0, 3).join(' | ')}`);
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
      assertClean(ctx, frames);
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

  test('둔화 장판이 있는 숲 · 위기 진입(장판 페이드 → 반지름 0)에서도 전 프레임이 깨끗하다 (㊱ 잔상 회귀)', () => {
    const d = loadData();
    const w = createWorld({ data: d, seed: 19, weapons, hooks: { enemies, emitters, run: tickRun, boss: bossHook }, startWeaponId: 'forward' });
    initRun(w); w.run.order[0] = 'forest'; w.run.stageIndex = 0;
    for (const id of ['lance', 'pinball', 'beam', 'aura', 'nova']) giveWeapon(w, id);
    for (let i = 0; i < w.slots.length; i += 1) { if (w.slots[i].weaponId === null) continue; for (let k = 0; k < 6; k += 1) levelUpWeapon(w, i); }
    w.player.hp = 1e9; w.player.hpMax = 1e9;
    const pal = resolvePalette(d.rules); const fx = makeFx(w); const interp = makeInterp(w); const ctx = stubCtx();
    let faded = 0; let crisisFrames = 0;
    const total = Math.floor(120 / TICK_DT);
    for (let i = 0; i < total; i += 1) {
      captureInterp(interp, w);
      step(w, makeInput(), TICK_DT);
      updateFx(fx, w, TICK_DT);
      for (const t of w.terrain.items) if (t.alive && t.fadeT >= 0) { faded += 1; break; }
      if (w.run.crisis) crisisFrames += 1;
      drawWorld(ctx, w, pal, fx, interp, (i % 3) / 3);
      drawPanels(ctx, w, pal);
      assertClean(ctx, i);
    }
    assert.gt(crisisFrames, 0, '위기에 실제로 들어갔다');
    assert.gt(faded, 0, '페이드 중인 장판을 실제로 그렸다 (vacuous 아님)');
  });

  test('결정 화면(드래프트·결과)도 던지지 않는다 (v1.5: 상점·사망 화면 폐지)', () => {
    const w = mkRun(3);
    const pal = resolvePalette(w.data.rules);
    const ctx = stubCtx();
    const draft = buildDraft(w);
    drawDraft(ctx, w, pal, draft, 0);
    drawResults(ctx, w, pal, tally(w), 'seed-3');
    assert.ok(true, '두 화면 모두 예외 없이 통과');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('render · §7.4 밀도 알파 (v1.10 ㉖ — 만렙 빌드의 백지 화면 회귀)', () => {
  test('탄 수 ≤ densityRef 면 상한, 그 위로는 상한 × ref / live, minAlpha 아래로는 안 간다 — 총 밝기(수 × 알파)가 늘지 않는다', () => {
    const d = loadData(); const r = d.rules.render;
    assert.ok(r.playerBulletDensityRef > 0 && r.playerBulletMinAlpha > 0 && r.playerBulletMinAlpha < r.playerBulletMaxAlpha, '키가 있고 순서가 맞다');
    assert.eq(bulletDensityAlpha(r, 0), r.playerBulletMaxAlpha, '0발 = 상한');
    assert.eq(bulletDensityAlpha(r, r.playerBulletDensityRef), r.playerBulletMaxAlpha, 'ref 발 = 상한');
    const a2 = bulletDensityAlpha(r, r.playerBulletDensityRef * 2);
    assert.near(a2, r.playerBulletMaxAlpha / 2, 1e-12, '2배면 절반');
    assert.near(2 * r.playerBulletDensityRef * a2, r.playerBulletDensityRef * r.playerBulletMaxAlpha, 1e-9, '수 × 알파 = 일정');
    assert.eq(bulletDensityAlpha(r, 100000), r.playerBulletMinAlpha, '바닥 = minAlpha');
    for (let n = 1; n < 600; n += 7) assert.ok(bulletDensityAlpha(r, n) >= bulletDensityAlpha(r, n + 7) - 1e-12, `단조 비증가 @${n}`);
    // 풀 상한(caps.playerBullets)에서의 총 밝기가 ref × 상한의 몇 배인가 — 바닥 때문에 조금 넘지만 4배는 안 넘는다
    const cap = d.rules.caps.playerBullets;
    assert.ok(cap * bulletDensityAlpha(r, cap) <= 4 * r.playerBulletDensityRef * r.playerBulletMaxAlpha, `풀 상한 ${cap}발의 총 밝기 ≤ 4 × 기준`);
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('render · §11.1 ㊴ 카드 줄바꿈 — 어떤 문장도 카드를 넘지 않는다', () => {
  const measure = (t) => t.length * 10;          // 글자당 10px 인 가짜 폰트(순수 함수라 테스트가 결정적이다)

  test('공백이 없는 긴 한 덩어리도 maxW 안으로 쪼갠다 (플레이테스트 회귀: 「다중 장전」 카드가 삐져나갔다)', () => {
    const s = '[탄] 탄 무기(벌컨·팬아웃·스파이럴·시커·리턴·미사일·핀볼)의 발사 수 +N';
    const lines = wrapLines(measure, s, 200);
    assert.gt(lines.length, 1, '여러 줄로 나뉜다');
    for (const l of lines) assert.ok(measure(l) <= 200, `줄이 폭 안에 있다: "${l}" (${measure(l)}px)`);
    assert.eq(lines.join('').replace(/ /g, ''), s.replace(/ /g, ''), '글자는 하나도 잃지 않는다');
  });

  test('실제 데이터의 모든 카드 문장이 카드 폭 안에 들어간다 (패시브·무기·특성 desc 전수)', () => {
    const d = loadData();
    const maxW = 264;                            // drawDraft 의 cw(300) − 36
    const strings = [];
    for (const p of d.passives.passives) strings.push(p.desc, p.name);
    for (const w of d.weapons.weapons) { strings.push(w.desc, w.name, w.evolution.desc, w.evolution.name); }
    for (const t of d.traits.traits) strings.push(t.desc, t.name);
    for (const st of d.tutorial.steps) strings.push(st.body, st.hint);
    let n = 0;
    for (const s of strings) {
      for (const l of wrapLines(measure, s, maxW)) {
        assert.ok(measure(l) <= maxW, `카드 폭 초과: "${l}"`);
        n += 1;
      }
    }
    assert.gt(n, 60, `검사한 줄 ${n} (vacuous 아님)`);
  });

  test('한 글자가 폭보다 넓어도 무한 루프에 빠지지 않는다 (극단)', () => {
    const lines = wrapLines(measure, '가나다라', 3);
    assert.eq(lines.length, 4, '글자마다 한 줄');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('render · §11.1 ㊸ 패시브 카드의 «내 무기» 줄', () => {
  test('기체 패시브에는 줄이 붙지 않는다 — 무기와 무관하기 때문이다 (강화 격벽이 「내 무기: 벌컨…」을 달던 회귀)', () => {
    const w = mkRun(5);
    for (const p of w.data.passives.passives) {
      const line = passiveWeaponLine(w, p.id);
      if (BODY_STATS.indexOf(p.stat) >= 0) assert.eq(line, null, `${p.id}(기체) = 줄 없음`);
      else assert.ok(line !== null && typeof line.text === 'string', `${p.id}(무기 분류) = 줄 있음`);
    }
  });

  test('㊸ 답은 «저작값»이 아니라 «지금 그 무기의 유효 파라미터»에서 나온다', () => {
    const w = mkRun(9);
    for (const s of w.slots) { s.weaponId = null; s.family = ''; }
    giveW(w, 'forward');
    const slot = w.slots.find((s) => s.family === 'forward');
    assert.eq(passiveAffectsSlot(w, slot, 'pierceAdd'), true, '벌컨은 관통 코팅을 받는다');
    // 유효 파라미터에서 pierce 가 «무제한(-1)»이 되면 관통 +N 은 의미가 없다 → 화면도 그렇게 답해야 한다
    const eff = recomputeEff(w, slot);
    const keep = eff.pierce;
    eff.pierce = -1; slot.effDirty = false;
    assert.eq(passiveAffectsSlot(w, slot, 'pierceAdd'), false, '무제한 관통이면 «혜택 없음»');
    eff.pierce = keep;
    // 키 자체가 없으면(그 무기의 계약에 없는 파라미터) 역시 없음
    delete eff.lifetimeSec;
    assert.eq(passiveAffectsSlot(w, slot, 'durationMul'), false, '수명 키가 없으면 장기 배터리 무효');
  });

  test('무기 분류 패시브는 «내가 든 무기 중 듣는 것»만 센다', () => {
    const w = mkRun(6);
    for (const s of w.slots) { s.weaponId = null; s.family = ''; }
    giveW(w, 'lance');                                   // 빔 하나만 든다
    const beam = passiveWeaponLine(w, 'highvolt');       // 빔 피해
    assert.eq(beam.hit.length, 1, '랜스 하나');
    assert.ok(beam.text.indexOf('랜스') >= 0, `${beam.text}`);
    const bullet = passiveWeaponLine(w, 'autoload');     // 탄 발사 수
    assert.eq(bullet.hit.length, 0, '탄 무기가 없다');
    assert.eq(bullet.text, '지금 내 무기엔 효과 없음', bullet.text);
  });
});

suite('render/무기 행 배치 §9.5 (v1.10 ㊿-d)', () => {
  // 한글 = 전각, ASCII ≈ 0.56 배 — 캔버스 없이 결정적으로 재는 근사 폰트
  const mm = (t, px) => [...t].reduce((a, c) => a + (/[가-힣]/.test(c) ? px : px * 0.56), 0);
  const W = () => loadData().rules.view.panelRightW - loadData().rules.hud.panelPadPx * 2;
  const SM = () => loadData().rules.hud.fontSmallPx;
  const BD = () => loadData().rules.hud.fontBodyPx;

  test('진화 임박 행: 이름·칩·힌트·레벨이 «겹치지 않는다» — 14 무기 전부', () => {
    const d = loadData();
    const w = W();
    const lvText = `Lv.${7}/10`;
    const lvW = mm(lvText, SM());
    let shown = 0;
    for (const wd of d.weapons.weapons) {
      const req = wd.evolution.requiresPassive;
      const pdef = d.passives.passives.find((q) => q.id === req.id);
      const full = `${pdef.name} 0/${req.level}`;
      const chip = '궤도';                       // 분류 칩 중 가장 긴 것(2자) = 최악
      const lay = weaponRowLayout(w, mm(wd.name, BD()), mm(chip, SM()), lvW, [full, `0/${req.level}`], (t) => mm(t, SM()));
      assert.ok(lay.hint !== '', `${wd.name}: 진화 힌트가 그려진다`);
      if (lay.hint === full) shown += 1;
      const hintRight = lay.hintX + mm(lay.hint, SM());
      const levelLeft = w - 46 - lvW;
      assert.ok(hintRight <= levelLeft, `${wd.name}: 힌트 우측 ${hintRight.toFixed(0)} ≤ 레벨 좌측 ${levelLeft.toFixed(0)}`);
      if (lay.showChip) {
        assert.ok(lay.chipX + mm(chip, SM()) <= lay.hintX, `${wd.name}: 칩이 힌트와 안 겹친다`);
      }
      assert.ok(26 + mm(wd.name, BD()) <= lay.chipX, `${wd.name}: 이름이 칩/힌트와 안 겹친다`);
    }
    assert.eq(shown, d.weapons.weapons.length, '14 무기 모두 «패시브 이름 + 진행»을 온전히 보여준다');
  });

  test('폭이 모자라면 칩 → 짧은 힌트 → 힌트 없음 순으로 양보한다', () => {
    const long = '아주아주아주 긴 무기 이름';
    const hints = ['자기 코일 0/3', '0/3'];
    const wide = weaponRowLayout(400, 30, 28, 46, hints, (t) => mm(t, 14));
    assert.eq(wide.hint, hints[0], '넉넉하면 전부');
    assert.eq(wide.showChip, true, '넉넉하면 칩도');
    const mid = weaponRowLayout(240, mm(long, 16), 28, 46, hints, (t) => mm(t, 14));
    assert.ok(mid.hint === hints[1] || mid.hint === '' || mid.showChip === false, '좁으면 양보한다');
    const tiny = weaponRowLayout(120, mm(long, 16), 28, 46, hints, (t) => mm(t, 14));
    assert.eq(tiny.hint, '', '아주 좁으면 힌트를 접는다 (겹쳐 찍지 않는다)');
    assert.eq(tiny.showChip, true, '힌트를 접었으면 칩은 남는다');
  });

  test('진화 임박이 아니면 힌트가 없다 (hints 빈 배열)', () => {
    const lay = weaponRowLayout(318, 32, 28, 55, [], (t) => mm(t, 14));
    assert.eq(lay.hint, '', '힌트 없음');
    assert.eq(lay.showChip, true, '칩은 그린다');
  });
});
