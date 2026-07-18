/**
 * tests/enemies.test.mjs — src/core/enemies.js (스포너 + 이동 스크립트 훅) 계약 단위 테스트
 *
 * 대상 (임무):
 *   · 결정성 — 같은 시드 → 같은 스폰 시퀀스(scatter 좌표까지 비트 동일) / 다른 시드 → 상이 (§10.2)
 *   · element 편성 주입 — 적의 element 는 웨이브 레코드에서 온다. 서로 다른 element 가 **섞여** 내려온다 (§8.6)
 *   · moveId 속도 — dive: vy=speed·vx=0 / weave: vy=speed·vx=ampPx·ω·cos(ω·moveT) (§8.4)
 *   · 캡 준수 — live ≤ caps.enemies · 풀 무결(freeTop+live==size), 언제나 (§12.1 · §10.3)
 *   · 화면 이탈 몰수 — 하단으로 이탈한 적은 release 되고 **보상(XP 픽업)이 0** (§8.7)
 *   · 슬라이스 통합 — 스폰 → 자동발사 피격 → 처치 → XP 드랍이 실제로 배선돼 있다 (재미 검증의 뼈대)
 *
 * ★ 값은 전부 data/정본에서 유도한다 (하드코딩 매직넘버 지양).
 */
import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld, spawnEnemy } from '../src/core/state.js';
import { step, makeInput, TICK_DT } from '../src/core/step.js';
import { enemies } from '../src/core/enemies.js';
import { weapons } from '../src/core/weapons/index.js';
import { TAU } from '../src/core/angle.js';

const dt = TICK_DT;

/** 훅을 주입한 월드 */
function mk(seed = 1) {
  return createWorld({ data: loadData(), seed, weapons, hooks: { enemies } });
}
/** 자동 발사가 적을 죽이지 않게 무기를 침묵(스포너·이동만 관찰) */
function silence(w) { for (const s of w.slots) s.weaponId = null; }
/** 아키타입 정의 (매직넘버 대신 데이터에서) */
function arch(w, id) { return w.data.enemies.archetypes.find((a) => a.id === id); }
/** enemies.js 와 같은 규칙으로 슬라이스 로스터를 유도한다(구현 이동 × 플레이 밴드 × 테마) */
function sliceRoster(w) {
  const MOVES = ['dive', 'weave'];
  const BANDS = ['chaff', 'line'];
  return w.data.enemies.archetypes
    .filter((a) => MOVES.includes(a.moveId) && BANDS.includes(a.band) && (a.themeOnly === null || a.themeOnly === 'sea'))
    .map((a) => a.id);
}
/** alive 적 스냅샷 서명 — idx 오름차순, 좌표·속도까지 */
function signature(w) {
  const out = [];
  const items = w.enemies.items;
  for (let i = 0; i < items.length; i += 1) {
    const e = items[i];
    if (!e.alive) continue;
    out.push(`${e.idx}:${e.archetypeId}:${e.element}:${e.x.toFixed(4)}:${e.y.toFixed(4)}:${e.vx.toFixed(4)}:${e.vy.toFixed(4)}`);
  }
  return out.join('|');
}

// ─────────────────────────────────────────────────────────────────────────
suite('enemies · 결정성 (§10.2 — rng.spawn 만 사용)', () => {
  test('같은 시드 → 같은 스폰 시퀀스 (scatter 좌표까지 비트 동일)', () => {
    const a = mk(20260716);
    const b = mk(20260716);
    silence(a); silence(b);
    for (let i = 0; i < 200; i += 1) { step(a, makeInput(), dt); step(b, makeInput(), dt); }
    assert.ok(a.enemies.live > 0, '웨이브가 실제로 스폰됐다 (0행 아님)');
    assert.eq(signature(a), signature(b), '같은 시드 = 비트 동일한 스폰/이동');
  });

  test('다른 시드 → 상이 (scatter 가 rng.spawn 을 실제로 탄다)', () => {
    const a = mk(1);
    const b = mk(2);
    silence(a); silence(b);
    for (let i = 0; i < 200; i += 1) { step(a, makeInput(), dt); step(b, makeInput(), dt); }
    assert.ok(a.enemies.live > 0 && b.enemies.live > 0, '양쪽 다 스폰됨');
    assert.ne(signature(a), signature(b), '다른 시드 = scatter 좌표가 갈린다');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('enemies · element 편성 주입 (§8.6 — 상성의 핵심)', () => {
  test('첫 웨이브: element 는 웨이브가 주입, 아키타입은 로스터[0]', () => {
    const w = mk(7);
    silence(w);
    // sea stage-1 첫 해금 웨이브 = scatter water 16. 아키타입은 로스터[0](= drifter). 데이터에서 유도.
    const stage = w.data.stages.stages.find((s) => s.id === 'sea');
    const wave0 = stage.waves.find((v) => v.unlockStageMin <= 1);
    const roster = sliceRoster(w);
    const expectId = roster[0];
    const expectDef = arch(w, expectId);
    const hpMult = w.data.enemies.bands[expectDef.band].hpMult;
    const expectCount = Math.max(2, Math.round(wave0.count / hpMult));
    step(w, makeInput(), dt);   // 첫 틱에 wave0 스폰
    const items = w.enemies.items;
    let n = 0;
    for (let i = 0; i < items.length; i += 1) {
      const e = items[i];
      if (!e.alive) continue;
      n += 1;
      assert.eq(e.archetypeId, expectId, '아키타입 = 로스터[0] (골격은 웨이브, 종류는 로스터)');
      assert.eq(e.element, wave0.element, 'element = 웨이브 레코드 (아키타입 필드 아님)');
    }
    assert.eq(n, expectCount, `첫 웨이브 = 밴드 클램프된 count(${expectCount})만큼 스폰`);
  });

  test('서로 다른 element 가 섞여 내려온다 (스탠스를 바꿀 이유)', () => {
    const w = mk(7);
    silence(w);
    const seen = new Set();
    let maxDistinctAlive = 0;
    for (let i = 0; i < 2400; i += 1) {
      step(w, makeInput(), dt);
      const live = new Set();
      const items = w.enemies.items;
      for (let j = 0; j < items.length; j += 1) if (items[j].alive) live.add(items[j].element);
      for (const el of live) seen.add(el);
      if (live.size > maxDistinctAlive) maxDistinctAlive = live.size;
    }
    // sea stage-1 해금 웨이브의 element 집합 = water/grass/fire/normal
    assert.ok(seen.has('water'), 'water 등장');
    assert.ok(seen.has('grass'), 'grass 등장');
    assert.ok(seen.has('fire'), 'fire 등장');
    assert.gte(seen.size, 3, '≥3 종의 element 가 등장');
    assert.gte(maxDistinctAlive, 2, '한 화면에 ≥2 종의 element 가 동시에 살아 있다 (실제 혼재)');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('enemies · moveId 속도 (§8.4)', () => {
  test('dive — vy = moveParams.speed, vx = 0', () => {
    const w = mk(3);
    silence(w);
    const drifter = arch(w, 'drifter');   // moveId dive
    assert.eq(drifter.moveId, 'dive', '전제: drifter 는 dive');
    step(w, makeInput(), dt);             // wave0 = drifter 스폰 + 이동 세팅
    const items = w.enemies.items;
    let n = 0;
    for (let i = 0; i < items.length; i += 1) {
      const e = items[i];
      if (!e.alive || e.archetypeId !== 'drifter') continue;
      n += 1;
      assert.eq(e.vy, drifter.moveParams.speed, 'dive vy = speed');
      assert.eq(e.vx, 0, 'dive vx = 0 (직하강)');
    }
    assert.gt(n, 0, 'drifter 가 실제로 존재');
  });

  test('weave — vy = speed, vx = ampPx·ω·cos(ω·moveT) (사인 좌우, 부호가 반주기에 뒤집힘)', () => {
    const w = mk(5);
    silence(w);
    const spitter = arch(w, 'spitter');   // moveId weave
    assert.eq(spitter.moveId, 'weave', '전제: spitter 는 weave');
    const mp = spitter.moveParams;
    const om = TAU * mp.freqHz;

    // 스케줄과 무관하게 moveT=0 인 spitter 하나를 직접 주입해 이동 법칙만 관찰
    const cx = w.data.rules.view.arena.x + w.data.rules.view.arena.w / 2;
    const e = spawnEnemy(w, 'spitter', 'fire', cx, w.data.rules.view.spawnLineY, spitter.hp, false);
    assert.eq(e.moveT, 0, 'spawn 직후 moveT 0');

    enemies(w, dt);   // applyMovement 이 e.vx/vy 를 세팅 (스케줄러는 wave0 도 스폰하지만 무관)
    assert.eq(e.vy, mp.speed, 'weave vy = speed');
    assert.near(e.vx, mp.ampPx * om * Math.cos(om * 0), 1e-9, 'moveT 0: vx = ampPx·ω');
    assert.gt(e.vx, 0, 'moveT 0 에서 vx > 0 (cos0 = 1)');

    // 반주기(T/2 = 1/(2·freqHz)) 후: cos(π) = -1 → vx 부호 반전
    e.moveT = 1 / (2 * mp.freqHz);
    enemies(w, dt);
    assert.near(e.vx, mp.ampPx * om * Math.cos(om * e.moveT), 1e-9, '반주기: vx = ampPx·ω·cos(π)');
    assert.lt(e.vx, 0, '반주기에서 vx < 0 (좌우로 흔든다)');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('enemies · 캡 준수 (§12.1 · §10.3)', () => {
  test('오래 돌려도 live ≤ caps.enemies · 풀 무결', () => {
    const w = mk(11);
    silence(w);   // 아무도 안 죽으니 스포너가 최대 압박을 만든다
    const cap = w.data.rules.caps.enemies;
    const concurrent = w.data.rules.fairness.enemyConcurrentMax;
    let peak = 0;
    for (let i = 0; i < 3000; i += 1) {
      step(w, makeInput(), dt);
      const p = w.enemies;
      assert.lte(p.live, cap, 'live ≤ caps.enemies (풀 = B층 안전망)');
      assert.eq(p.freeTop + p.live, p.size, '풀 무결: freeTop + live == size');
      if (p.live > peak) peak = p.live;
    }
    assert.gt(peak, 0, '실제로 스폰이 일어났다 (공허 통과 아님)');
    assert.lte(peak, concurrent, '동시 오써링 상한(enemyConcurrentMax) 을 넘지 않는다');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('enemies · 화면 이탈 몰수 (§8.7 — step 이 집행, 이동은 이 훅)', () => {
  test('하단으로 이탈한 적은 release 되고 XP 픽업이 0 (보상 몰수)', () => {
    const w = mk(9);
    silence(w);
    const a = w.data.rules.view.arena;
    const cx = a.x + a.w / 2;
    const drifter = arch(w, 'drifter');
    // 하단 이탈 직전에 배치 (exit 조건: y > a.y + a.h + 64)
    const e = spawnEnemy(w, 'drifter', 'water', cx, a.y + a.h + 60, drifter.hp, false);
    const gen0 = e.gen;
    assert.ok(e.alive, '배치됨');

    let left = false;
    for (let i = 0; i < 30 && !left; i += 1) {
      step(w, makeInput(), dt);                 // 훅이 vy=speed 세팅, step 이 적분+이탈 판정
      assert.eq(w.pickups.live, 0, '이탈 경로 어디에서도 XP 픽업이 생기지 않는다 (몰수)');
      if (!e.alive || e.gen !== gen0) left = true;
    }
    assert.ok(left, '적이 하단으로 빠져 release 됐다');
    assert.eq(w.pickups.live, 0, '최종적으로도 보상 0 (killEnemy 를 거치지 않았다)');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('enemies · 슬라이스 통합 (재미의 뼈대: 스폰 → 피격 → 처치 → XP)', () => {
  test('무기를 살려두면 실제로 적이 죽어 XP 픽업이 드랍된다', () => {
    const w = mk(4);   // 무기 침묵 안 함 = forward 자동발사가 상단의 적을 때린다
    let sawXpPickup = false;
    for (let i = 0; i < 1500 && !sawXpPickup; i += 1) {
      step(w, makeInput(), dt);
      const items = w.pickups.items;
      for (let j = 0; j < items.length; j += 1) {
        if (items[j].alive && items[j].kind === 'xp') { sawXpPickup = true; break; }
      }
    }
    // 화면 이탈(몰수)은 픽업을 안 만든다 → XP 픽업의 존재 = 처치가 실제로 일어났다는 증거
    assert.ok(sawXpPickup, '스폰된 적이 자동발사에 맞아 죽고 XP 를 떨궜다 (슬라이스 end-to-end)');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('enemies · 슬라이스 로스터 다양성 (피드백 #3 — 대비를 화면에)', () => {
  test('로스터는 데이터 유도 · 구현 이동 × 플레이 밴드 × 테마 부합만', () => {
    const w = mk(1);
    const roster = sliceRoster(w);
    assert.gte(roster.length, 3, '로스터 ≥ 3종 (drifter/spitter 둘만이 아니다)');
    for (const id of roster) {
      const a = arch(w, id);
      assert.ok(['dive', 'weave'].includes(a.moveId), `${id}: 구현된 이동만 (dive/weave)`);
      assert.ok(['chaff', 'line'].includes(a.band), `${id}: 플레이 가능한 밴드만 (chaff/line)`);
      assert.ok(a.themeOnly === null || a.themeOnly === 'sea', `${id}: 테마 부합만`);
    }
    assert.eq(roster[0], 'drifter', '로스터[0] = drifter (element 테스트 전제)');
  });

  test('오래 돌리면 ≥3종의 아키타입이 실제로 스폰된다 (느린 탱커 ↔ 빠른 약골)', () => {
    const w = mk(13);
    silence(w);
    const seen = new Set();
    for (let i = 0; i < 3000; i += 1) {
      step(w, makeInput(), dt);
      const items = w.enemies.items;
      for (let j = 0; j < items.length; j += 1) if (items[j].alive) seen.add(items[j].archetypeId);
    }
    assert.gte(seen.size, 3, `≥3종 스폰됨 (실제 ${seen.size}종: ${[...seen].sort().join(', ')})`);
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('enemies · HP↔속도 튜닝 (피드백 #3 — 느린=탱키/빠른=약함)', () => {
  test('로스터를 속도 내림차순으로 정렬하면 effHP 가 단조 증가한다', () => {
    const w = mk(1);
    const bands = w.data.enemies.bands;
    const rows = sliceRoster(w).map((id) => {
      const a = arch(w, id);
      return { id, speed: a.moveParams.speed, effHp: a.hp * bands[a.band].hpMult };
    });
    // 빠른 → 느린 순으로 정렬. 그러면 effHP 는 비감소여야 한다 (느릴수록 더 단단하다).
    rows.sort((p, q) => q.speed - p.speed);
    for (let i = 1; i < rows.length; i += 1) {
      assert.gte(rows[i].effHp, rows[i - 1].effHp,
        `${rows[i - 1].id}(spd ${rows[i - 1].speed}, effHP ${rows[i - 1].effHp}) → `
        + `${rows[i].id}(spd ${rows[i].speed}, effHP ${rows[i].effHp}): 더 느린데 더 약하면 안 된다`);
    }
    // 대비가 실제로 크다: 가장 빠른 것과 가장 느린 것의 effHP 배율 ≥ 3
    const fastest = rows[0];
    const slowest = rows[rows.length - 1];
    assert.gte(slowest.effHp / fastest.effHp, 3,
      `가장 느린 ${slowest.id}(effHP ${slowest.effHp}) 는 가장 빠른 ${fastest.id}(effHP ${fastest.effHp}) 의 ≥3배 (체감되는 대비)`);
  });
});
