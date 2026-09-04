/**
 * tests/terrain.test.mjs — 지형 장판 (§8.21, v1.10 ⑦)의 정본 계약.
 *
 * 커버:
 *   스폰   — MOB 에서 everySec 마다, 무대에 maxOnScreen 미만일 때, 아레나 안 x · 위에서 들어온다 / 최종 스테이지는 0
 *   흐름   — scrollSpeedPx 로 내려가고 아레나 아래로 나가면 반납 / advanceStage 가 무대를 비운다
 *   효과   — slow: 안에서 둔화(status.slowMoveSpeedMul) · 밖으로 나가면 다음 틱에 풀림
 *            inertia: 안에서 방향을 뒤집으면 vx 가 «서서히» 뒤집힌다(tau) · 밖에서는 즉시
 *            heat: fullSec 만에 차고 → stallSec 스턴 → 0 · 밖에서 coolSec 에 식는다 · 스턴 중엔 안 찬다
 *   무해   — 어느 지형 안에 서 있어도 HP 가 줄지 않는다(피해 0 = 사용자 결정 「유틸 방해」)
 *   결정성 — 같은 시드 = 같은 위치(rng.terrain)
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld } from '../src/core/state.js';
import { step, makeInput, TICK_DT } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { initRun, tickRun, advanceStage, PHASE } from '../src/core/stage.js';
import { terrainUnder, sectionOf, T_SLOW, T_INERTIA, T_HEAT } from '../src/core/terrain.js';
import { enemies } from '../src/core/enemies.js';
import { bossHook, wipeFrontY } from '../src/core/boss.js';
import { spawnEnemy, spawnEnemyBullet } from '../src/core/state.js';
import { TERRAIN_KINDS } from '../src/core/schema.mjs';

const dt = TICK_DT;

/** 적·발사 없이 런 시계만 흐르는 세계 — 지형과 플레이어만 본다 */
function mkRun(seed, stageId, pos = 0) {
  const w = createWorld({ data: loadData(), seed, weapons, hooks: { enemies: null, emitters: null, run: tickRun, boss: null } });
  initRun(w);
  w.run.order[pos] = stageId;
  w.run.stageIndex = pos;
  w.player.hp = 1e9; w.player.hpMax = 1e9;
  return w;
}
function live(w) { return w.terrain.items.filter((t) => t.alive); }
function tick(w, n, input = makeInput()) { for (let i = 0; i < n; i += 1) step(w, input, dt); }
/** 첫 장판이 화면 가운데쯤 올 때까지 흘리고, 플레이어를 그 중심에 세운다 */
function standInFirst(w) {
  for (let i = 0; i < 60 * 30; i += 1) {
    step(w, makeInput(), dt);
    const t = live(w)[0];
    if (t && t.y > 200) { w.player.x = t.x; w.player.y = t.y; return t; }
  }
  throw new Error('지형이 안 온다');
}

suite('terrain — 스폰·흐름 (§8.21)', () => {
  test('MOB 에서 everySec 마다 스폰, 무대 상한 maxOnScreen, x 는 아레나 안, y 는 위에서', () => {
    const w = mkRun(3, 'bog');
    const tr = w.data.rules.terrain; const a = w.data.rules.view.arena;
    tick(w, 1);
    assert.eq(live(w).length, 1, '첫 틱에 하나 (terrainNextT 0)');
    const t0 = live(w)[0];
    assert.ok(t0.x >= a.x + t0.radius && t0.x <= a.x + a.w - t0.radius, 'x 가 아레나 안');
    assert.lt(t0.y, a.y, '위에서 들어온다');
    assert.eq(t0.radius, tr.radiusPx, '반지름 = rules.terrain.radiusPx');
    assert.eq(t0.kind, TERRAIN_KINDS.indexOf('slow'), 'bog = slow');
    tick(w, Math.round(tr.everySec / dt) + 2);
    assert.eq(live(w).length, 2, 'everySec 뒤 둘');
    tick(w, Math.round(tr.everySec / dt) * 6);
    assert.lte(live(w).length, tr.maxOnScreen, '무대 상한');
    assert.gte(live(w).length, 2, '흐르며 계속 있다');
  });

  test('scrollSpeedPx 로 내려가고 아레나 아래로 나가면 반납된다', () => {
    const w = mkRun(3, 'bog');
    const tr = w.data.rules.terrain; const a = w.data.rules.view.arena;
    tick(w, 1);
    const t = live(w)[0]; const y0 = t.y;
    tick(w, 60);
    assert.near(t.y - y0, tr.scrollSpeedPx, 1.5, '1초에 scrollSpeedPx 만큼');
    const need = Math.ceil(((a.y + a.h + t.radius) - t.y) / tr.scrollSpeedPx / dt) + 3;
    const gen = t.gen;
    tick(w, need);
    assert.ok(!t.alive || t.gen !== gen, '아래로 나가면 반납');
  });

  test('최종 스테이지(테마 없음)는 지형이 0 · advanceStage 는 무대를 비운다', () => {
    const w = mkRun(3, 'bog', 4);
    w.run.order[5] = 'finale';
    tick(w, 60 * 8);
    assert.gt(live(w).length, 0, '스테이지 5(bog)엔 지형이 있다');
    w.player.heat = 0.7;
    advanceStage(w);
    assert.eq(live(w).length, 0, '전이에서 비운다');
    assert.eq(w.player.heat, 0, '열도 0');
    tick(w, 60 * 10);
    assert.eq(live(w).length, 0, 'finale 은 지형 0');
  });

  test('결정성 — 같은 시드 = 같은 위치, 다른 시드 = 다른 위치 (rng.terrain)', () => {
    const xs = (seed) => { const w = mkRun(seed, 'glacier'); tick(w, 60 * 12); return live(w).map((t) => Math.round(t.x)).join(','); };
    assert.eq(xs(9), xs(9), '같은 시드');
    assert.ne(xs(9), xs(10), '다른 시드');
  });
});

suite('terrain — 효과 (§8.21 · §2.2 · §2.7)', () => {
  test('slow — 안에서 둔화(status.slowMoveSpeedMul), 밖으로 나가면 다음 틱에 풀린다', () => {
    const w = mkRun(3, 'bog');
    const rp = w.data.rules.player; const mul = w.data.rules.status.slowMoveSpeedMul;
    const t = standInFirst(w);
    assert.eq(terrainUnder(w, w.player.x, w.player.y), T_SLOW, '장판 안');
    const inp = makeInput(); inp.right = true;
    tick(w, 3, inp);
    assert.gt(w.player.slowSec, 0, '둔화 상태');
    assert.near(Math.abs(w.player.vx), rp.moveSpeed * (1 + w.stats.moveSpeedMul) * mul, 1e-6, '속도 = 기준 × 둔화');
    // 밖으로
    w.player.x = t.x + t.radius + 200; w.player.y = 600;
    tick(w, 2, inp);
    assert.eq(terrainUnder(w, w.player.x, w.player.y), -1, '밖');
    assert.eq(w.player.slowSec, 0, '풀렸다');
    assert.near(Math.abs(w.player.vx), rp.moveSpeed * (1 + w.stats.moveSpeedMul), 1e-6, '속도 정상');
  });

  test('inertia — 안에서 방향을 뒤집으면 vx 가 서서히 뒤집힌다(responseTauSec), 밖에서는 즉시', () => {
    const w = mkRun(4, 'glacier');
    const tr = w.data.rules.terrain;
    const t = standInFirst(w);
    assert.eq(terrainUnder(w, w.player.x, w.player.y), T_INERTIA, '장판 안');
    const right = makeInput(); right.right = true;
    const left = makeInput(); left.left = true;
    tick(w, 30, right);                          // 오른쪽으로 충분히
    w.player.x = t.x; w.player.y = t.y;          // 장판 중심에 다시 (관찰용)
    const v0 = w.player.vx;
    assert.gt(v0, 0, '오른쪽 속도');
    step(w, left, dt);
    assert.gt(w.player.vx, 0, '한 틱 뒤에도 아직 오른쪽 (관성)');
    let n = 1;
    while (w.player.vx > 0 && n < 600) { w.player.x = t.x; w.player.y = t.y; step(w, left, dt); n += 1; }
    const expect = Math.log(2) * tr.inertia.responseTauSec / dt;    // tau·ln2 만에 0 을 지난다
    assert.ok(Math.abs(n - expect) <= 3, `뒤집히는 데 ${n}틱 ≈ tau·ln2 = ${expect.toFixed(1)}틱`);
    // 밖에서는 즉시
    w.player.x = t.x + t.radius + 200; w.player.y = 600;
    tick(w, 5, right);
    step(w, left, dt);
    assert.lt(w.player.vx, 0, '밖에서는 한 틱에 뒤집힌다 (moveResponseTau 0)');
  });

  test('heat — fullSec 에 차서 stallSec 스턴 후 0 · 스턴 중엔 안 찬다 · 밖에서 coolSec 에 식는다', () => {
    const w = mkRun(5, 'volcano');
    const tr = w.data.rules.terrain;
    const t = standInFirst(w);
    assert.eq(terrainUnder(w, w.player.x, w.player.y), T_HEAT, '장판 안');
    let stunAt = -1; let n = 0;
    while (stunAt < 0 && n < 600) { w.player.x = t.x; w.player.y = t.y; step(w, makeInput(), dt); n += 1; if (w.player.stunSec > 0) stunAt = n; }
    assert.ok(stunAt > 0, '정지가 온다');
    assert.ok(Math.abs(stunAt * dt - tr.heat.fullSec) <= 2 * dt, `fullSec(${tr.heat.fullSec}) 만에 정지 (실제 ${(stunAt * dt).toFixed(2)})`);
    assert.near(w.player.stunSec, tr.heat.stallSec, 1e-6, '정지 길이 = stallSec');
    assert.eq(w.player.heat, 0, '정지 순간 열 0');
    w.player.x = t.x; w.player.y = t.y; step(w, makeInput(), dt);
    assert.eq(w.player.heat, 0, '스턴 중엔 안 찬다');
    // 밖에서 식는다
    let m = 0; while (w.player.stunSec > 0 && m < 200) { w.player.x = t.x; w.player.y = t.y; step(w, makeInput(), dt); m += 1; }
    w.player.x = t.x; w.player.y = t.y; tick(w, 30);                // 0.5초 → heat ≈ 1/3
    const h = w.player.heat;
    assert.gt(h, 0.2, '다시 찬다');
    w.player.x = t.x + t.radius + 200; w.player.y = 600;
    tick(w, Math.round(tr.heat.coolSec * h / dt) + 2);
    assert.eq(w.player.heat, 0, 'coolSec × 열 만에 식는다');
  });

  test('무해 — 세 지형 어디에 서 있어도 HP 가 줄지 않는다 (피해 0)', () => {
    for (const id of ['bog', 'glacier', 'volcano']) {
      const w = mkRun(6, id);
      w.player.hp = 100; w.player.hpMax = 100;
      const t = standInFirst(w);
      for (let i = 0; i < 60 * 12; i += 1) { w.player.x = t.x; w.player.y = t.y; step(w, makeInput(), dt); }
      assert.eq(w.player.hp, 100, `${id}: 12초 서 있어도 HP 100`);
    }
  });
});

suite('terrain — 구간과 위기 페이드 (§8.21 ④ v1.10 ⑧)', () => {
  test('sectionOf — early → midboss → crisis → boss 를 시계·페이즈로 가른다', () => {
    const w = mkRun(3, 'bog');
    const ph = w.data.stages.phase; const at = ph.midBossAtSec[0];
    assert.eq(sectionOf(w), 'early', '시작 = early');
    w.run.phaseT = at[0] + 1; assert.eq(sectionOf(w), 'midboss', '첫 중간보스 뒤 = midboss');
    w.run.crisis = true; assert.eq(sectionOf(w), 'crisis', '위기 = crisis');
    w.run.phase = PHASE.BOSS_INTRO; assert.eq(sectionOf(w), 'boss', 'BOSS_INTRO = boss');
    w.run.phase = PHASE.BOSS; assert.eq(sectionOf(w), 'boss', 'BOSS = boss');
    w.run.phase = PHASE.STAGE_CLEAR; assert.eq(sectionOf(w), null, '그 밖은 null');
  });

  test('spawnIn 에 crisis 가 없다 — 위기엔 새 지형이 안 나오고, 있던 것은 fadeSec 안에 줄어들며 사라진다(효과는 즉시 꺼짐)', () => {
    const w = mkRun(3, 'bog');
    const tr = w.data.rules.terrain;
    assert.ok(tr.spawnIn.indexOf('crisis') < 0, '정본: 위기 제외');
    const t = standInFirst(w);
    assert.eq(terrainUnder(w, w.player.x, w.player.y), T_SLOW, '위기 전엔 효과');
    // 위기 강제 — 격파 앞당김 경로 대신 시계 상한으로
    w.run.phaseT = w.data.stages.phase.crisisStartSec - dt;
    step(w, makeInput(), dt);
    assert.eq(w.run.crisis, true, '위기');
    assert.ok(t.alive && t.fadeT >= 0, '남은 지형이 사라지는 중');
    assert.eq(terrainUnder(w, t.x, t.y), -1, '사라지는 중엔 효과 없음');
    const before = live(w).length;
    tick(w, Math.round(tr.fadeSec / dt) + 2);
    assert.eq(live(w).length, 0, `fadeSec 뒤 0 (전 ${before})`);
    tick(w, Math.round(tr.everySec / dt) * 3);
    assert.eq(live(w).length, 0, '위기 중 새 스폰 0');
  });
});

suite('boss — 등장 쓸어내기 (§8.22 v1.10 ⑧)', () => {
  /** 잡몹·탄·지형이 남은 채 잡몹 페이즈를 끝내는 세계 */
  function mkAtPhaseEnd(seed, stageId) {
    const w = createWorld({ data: loadData(), seed, weapons, hooks: { enemies, emitters: null, run: tickRun, boss: bossHook } });
    initRun(w);
    w.run.order[0] = stageId;
    w.player.hp = 1e9; w.player.hpMax = 1e9;
    const ph = w.data.stages.phase;
    w.run.phaseT = ph.mobPhaseSec - 0.5;
    w.run.crisis = true; w.run.crisisAtSec = ph.crisisStartSec;   // 위기 끝자락 — 새떼가 남아 있다
    tick(w, 20);
    // 무대에 확실히 남긴다: 위·아래에 잡몹, 탄, 지형
    const def = w.data.enemies.archetypes.find((a) => a.id === 'drifter');
    spawnEnemy(w, 'drifter', 'normal', 600, 100, def.hp, false);
    spawnEnemy(w, 'drifter', 'normal', 620, 650, def.hp, false);
    spawnEnemyBullet(w, 'pelletS', 640, 700, 0, 50);
    const t = w.terrain.alloc(); t.kind = 0; t.radius = 72; t.x = 640; t.y = 600; t.fadeT = -1;
    return w;
  }
  const mobs = (w) => w.enemies.items.filter((e) => e.alive && !e.isBoss).length;

  test('앞선이 위에서 아래로 내려가며 지나간 것만 지운다 — entryWipeSec 뒤 무대엔 보스뿐, 그 뒤 지형 무리', () => {
    const w = mkAtPhaseEnd(2, 'bog');
    const b = w.data.rules.boss; const tr = w.data.rules.terrain;
    assert.gte(mobs(w), 2, '잡몹이 남아 있다');
    // 페이즈 끝 → BOSS_INTRO + wipeT 0
    tick(w, Math.round(0.5 / dt) + 1);
    assert.eq(w.run.phase, PHASE.BOSS_INTRO, '강림 연출');
    assert.gte(w.run.wipeT, 0, '쓸어내기 시작');
    assert.gt(mobs(w), 0, '시작 직후엔 아래쪽 잡몹이 아직 산다(즉시 증발이 아니다)');
    // 중간 — 앞선 아래는 산다
    tick(w, Math.round(b.entryWipeSec * 0.4 / dt));
    const front = wipeFrontY(w);
    for (const e of w.enemies.items) if (e.alive && !e.isBoss) assert.gt(e.y, front - 1e-6, '앞선 위의 잡몹은 없다');
    let below = 0; for (const e of w.enemies.items) if (e.alive && !e.isBoss && e.y > front) below += 1;
    assert.gt(below, 0, '앞선 아래엔 아직 있다');
    // 끝 — 전부 없고 지형 무리
    tick(w, Math.round(b.entryWipeSec * 0.7 / dt) + 2);
    assert.eq(w.run.wipeT, -1, '쓸어내기 끝');
    assert.eq(mobs(w), 0, '비-보스 적 0');
    assert.eq(w.enemyBullets.live, 0, '적탄 0');
    const core = w.enemies.items.find((e) => e.alive && e.isBoss && e.isCore);
    assert.ok(core, '보스는 남는다');
    const lv = w.terrain.items.filter((t) => t.alive);
    assert.eq(lv.length, tr.bossEntryCount, `지형 무리 ${tr.bossEntryCount}개`);
    const a = w.data.rules.view.arena;
    for (const t of lv) {
      assert.ok(t.y > a.y && t.y < a.y + a.h, '아레나 «안»에 이미 놓여 있다(위에서 오지 않는다)');
      assert.ok(t.fadeT < 0, '사라지는 중이 아니다');
      const dx = t.x - w.player.x; const dy = t.y - w.player.y;
      assert.gt(dx * dx + dy * dy, (t.radius + 40) * (t.radius + 40) - 1e-6, '플레이어 바로 위엔 없다');
    }
  });

  test('보스 구간에도 평소 주기로 계속 흘러온다 (spawnIn 에 boss) · 결정성', () => {
    function run(seed) {
      const w = mkAtPhaseEnd(seed, 'volcano');
      const tr = w.data.rules.terrain; const b = w.data.rules.boss;
      tick(w, Math.round((0.5 + b.entryWipeSec) / dt) + 3);
      const burst = w.terrain.items.filter((t) => t.alive).map((t) => `${Math.round(t.x)},${Math.round(t.y)}`).join(' ');
      tick(w, Math.round((tr.everySec * 2 + 0.2) / dt));
      return { burst, liveNow: w.terrain.items.filter((t) => t.alive).length, phase: w.run.phase };
    }
    const a = run(5); const b = run(5); const c = run(6);
    assert.eq(a.burst, b.burst, '같은 시드 = 같은 무리');
    assert.ne(a.burst, c.burst, '다른 시드 = 다른 무리');
    assert.gte(a.liveNow, 1, '보스 구간에도 지형이 있다');
  });

  test('finale — 쓸어내기는 하되 지형 무리는 0', () => {
    const w = mkAtPhaseEnd(2, 'finale');
    w.run.order[5] = 'finale'; w.run.stageIndex = 5;
    const b = w.data.rules.boss;
    tick(w, Math.round((0.5 + b.entryWipeSec) / dt) + 3);
    assert.eq(w.run.phase, PHASE.BOSS_INTRO, '강림');
    assert.eq(mobs(w), 0, '잡몹 0');
    assert.eq(w.terrain.items.filter((t) => t.alive).length, 0, '지형 0');
  });
});
