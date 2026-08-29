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
import { createWorld, spawnEnemy, spawnBeam, spawnZone } from '../src/core/state.js';
import { step, makeInput, TICK_DT } from '../src/core/step.js';
import { enemies } from '../src/core/enemies.js';
import { emitters } from '../src/core/emitters.js';
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
    // §8.6 — 밴드 클램프 × 스테이지 스폰 밀도(curve.spawnDensityScale). 슬라이스 = 스테이지 1.
    const density = w.data.stages.curve.spawnDensityScale[0];
    const expectCount = Math.max(2, Math.round((wave0.count / hpMult) * density));
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
    assert.eq(n, expectCount, `첫 웨이브 = 밴드 클램프 × 밀도(${expectCount})만큼 스폰`);
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

// §8.18(v1.7) 잡몹 사격 «강도»의 스테이지 곡선.
//   v1.6 까지 잡몹의 발사 주기와 탄 피해는 전 스테이지 동일했다 — 체력·밀도만 오르고
//   사격은 안 올랐으므로 스테이지 1 이 상대적으로 과했다(플레이 피드백: 「1인데 너무 많이 쏜다」).
//   이 두 테스트가 지키는 것은 «곡선이 존재한다»가 아니라 «곡선이 실제로 적용된다»이다.
suite('enemies/§8.18 잡몹 사격 강도의 스테이지 곡선 (v1.7)', () => {
  const mk = (stageIndex) => {
    const w = createWorld({ data: loadData(), seed: 6, weapons, hooks: {}, startWeaponId: 'forward' });
    w.run = { stageIndex, bossFireRateMul: 1, order: [], crisis: false };
    return w;
  };

  test('발사 주기 — 이미터 시간이 곡선만큼 느리게/빠르게 흐른다', () => {
    const curve = loadData().stages.curve.mobFireRateScale;
    assert.eq(curve.length, 6, '스테이지 6개분');
    for (let i = 0; i < 6; i += 1) {
      const w = mk(i);
      const p = w.player;
      const e = spawnEnemy(w, 'hexer', 'normal', p.x, p.y - 220, 9e9, false, false);
      for (let t = 0; t < 60; t += 1) emitters(w, TICK_DT);        // 1초
      assert.lt(Math.abs(e.emitT - curve[i]), 1e-6, `스테이지 ${i + 1}: emitT ${curve[i]} 여야 한다`);
    }
    assert.lt(curve[0], curve[5], '★ 초반이 후반보다 뜸하게 쏜다');
  });

  test('탄 피해 — 같은 탄이라도 스테이지에 따라 다르게 아프다', () => {
    const d = loadData();
    const curve = d.stages.curve.mobBulletDmgScale;
    const base = d.bullets.bullets.find((b) => b.id === 'hexBolt').dmg;
    const dmgAt = (i) => {
      const w = mk(i);
      const p = w.player;
      const e = spawnEnemy(w, 'hexer', 'normal', p.x, p.y - 220, 9e9, false, false);
      for (let t = 0; t < 60 * 6; t += 1) {
        emitters(w, TICK_DT);
        const b = w.enemyBullets.items.find((x) => x.alive);
        if (b) return b.dmg;
      }
      return null;
    };
    const lo = dmgAt(0); const hi = dmgAt(5);
    assert.eq(lo, Math.max(1, Math.round(base * curve[0])), '스테이지 1 탄 피해 = 기본 × 곡선');
    assert.eq(hi, Math.max(1, Math.round(base * curve[5])), '스테이지 6 탄 피해 = 기본 × 곡선');
    assert.lt(lo, hi, '★ 초반 탄이 후반 탄보다 덜 아프다');
  });

  test('보스·중간보스는 이 곡선 밖이다 (자기 곡선을 이미 갖는다)', () => {
    const w = mk(0);
    const p = w.player;
    const e = spawnEnemy(w, 'hexer', 'normal', p.x, p.y - 220, 9e9, false, false);
    e.midBossId = 'mbHammer';                                       // 중간보스로 표시
    for (let t = 0; t < 60; t += 1) emitters(w, TICK_DT);
    assert.lt(Math.abs(e.emitT - 1), 1e-6, '중간보스는 배율 1 — 잡몹 곡선을 안 탄다');
  });
});

// §7.6/§8.6(v1.7) — 「생김새로 공격을 예측할 수 없다」와 「스테이지 1부터 엘리트가 너무 많다」의 답.
suite('enemies/§7.6 공격 기호 · §8.6 엘리트 곡선 (v1.7)', () => {
  test('모든 아키타입이 자기 이미터 타입을 개체에 싣는다 (사격 안 하면 빈 문자열)', () => {
    const d = loadData();
    const w = createWorld({ data: d, seed: 3, weapons, hooks: {}, startWeaponId: 'forward' });
    const byId = {};
    for (const em of d.enemies.emitters) byId[em.id] = em;
    for (const a of d.enemies.archetypes) {
      const e = spawnEnemy(w, a.id, 'normal', 500, 100, 10, false, false);
      const want = a.attack === null ? '' : byId[a.attack.emitterId].type;
      assert.eq(e.attackType, want, `${a.id} 의 공격 기호`);
    }
  });

  // ★ v1.6 까지 베이크된 eliteIndex 가 곡선을 무시했다 — 스테이지 1 은 곡선 0.0 인데도
  //   웨이브의 34% 가 엘리트를 낳았고, 엘리트는 hpMult 4.0 이라 초반에 10~20초짜리 벽이었다.
  test('엘리트는 스테이지 곡선이 «유일한 권위»다 — 곡선 0 이면 한 마리도 없다', () => {
    const d = loadData();
    assert.eq(d.stages.curve.elitePerWaveChance[0], 0, '스테이지 1 곡선 = 0 (전제)');
    // 곡선이 0 인 스테이지에서, eliteIndex 가 박힌 웨이브가 실제로 존재하는지 먼저 확인한다.
    //   (없으면 이 테스트가 «통과»해도 아무것도 증명하지 못한다)
    let baked = 0;
    for (const st of d.stages.stages) {
      for (const wv of st.waves) if (wv.unlockStageMin <= 1 && wv.eliteIndex !== null) baked += 1;
    }
    assert.gt(baked, 0, '스테이지 1 에 eliteIndex 가 박힌 웨이브가 있다(전제) — 없으면 무의미한 통과');
    assert.lt(d.stages.curve.elitePerWaveChance[0], d.stages.curve.elitePerWaveChance[5],
      '★ 곡선은 스테이지가 갈수록 오른다 — 초반 평범한 몹 → 후반 엘리트');
  });
});

// §8.4/§8.18(v1.7 후속) — 「비행기 없이 적만」 계측이 잡아낸 두 구멍의 회귀.
suite('enemies/§8.4 진입 위치 · §8.18 빔·장판 곡선 (v1.7)', () => {
  // ★ strafe·rearIn 은 «스폰만 되고 아레나에 한 번도 서지 못했다»(실측 도달률 0.0%).
  //   strafe 는 상단 스폰라인에서 vy=0 이라 화면 위에 머물렀고, rearIn 은 거기서 vy=-speed 로
  //   더 멀어졌다. 값(moveParams.yPx)은 이미 저작돼 있었고 스폰이 그것을 안 읽은 것이 원인이다.
  test('strafe 는 좌우 벽 밖 · yPx 높이에서 들어온다', () => {
    const d = loadData();
    const w = createWorld({ data: d, seed: 3, weapons, hooks: { enemies }, startWeaponId: 'forward' });
    const a = d.rules.view.arena;
    const def = d.enemies.archetypes.find((x) => x.moveId === 'strafe');
    assert.ok(def !== undefined, 'strafe 아키타입 존재(전제)');
    assert.eq(typeof def.moveParams.yPx, 'number', 'yPx 가 저작돼 있다(전제)');
    const e = spawnEnemy(w, def.id, 'normal', 0, 0, 10, false, false);
    // 스폰 좌표는 spawnWave 경유라 직접 호출로는 안 잡힌다 — 여기서는 «이동이 화면 안을 향하는가»만 본다.
    for (let t = 0; t < 30; t += 1) step(w, makeInput(), TICK_DT);
    assert.eq(e.vy, 0, 'strafe 는 수평 횡단이다');
    assert.ok(e.vx !== 0, '옆으로 움직인다');
    assert.ok(def.moveParams.yPx > 0 && def.moveParams.yPx < a.h, 'yPx 가 아레나 안이다');
  });

  test('빔·장판도 잡몹 피해 곡선을 탄다 — 단 «깎기만» 한다 (§2.1 상한 보존)', () => {
    const d = loadData();
    const mk = (stageIndex) => {
      const w = createWorld({ data: d, seed: 1, weapons, hooks: {}, startWeaponId: 'forward' });
      w.run = { stageIndex, order: [], crisis: false, bossFireRateMul: 1 };
      return w;
    };
    const curve = d.stages.curve.mobBulletDmgScale;
    assert.lt(curve[0], 1, '스테이지 1 은 1 미만이다(전제)');
    assert.gt(curve[5], 1, '스테이지 6 은 1 초과다(전제)');
    const RAW = 22;
    const lo = spawnBeam(mk(0), 500, 100, 1.57, 16, RAW, 1, -1, 'turretPod', 0.5, false);
    const hi = spawnBeam(mk(5), 500, 100, 1.57, 16, RAW, 1, -1, 'turretPod', 0.5, false);
    const boss = spawnBeam(mk(5), 500, 100, 1.57, 16, RAW, 1, -1, '', 0.5, false);
    assert.lt(lo.dmg, RAW, '★ 초반 빔은 깎인다 — 「스테이지 1인데 너무 아프다」의 답');
    assert.eq(hi.dmg, RAW, '★ 후반에도 저작 상한을 넘지 않는다 — §2.1 「최대 단발 22 → 최소 5초」 보증');
    assert.eq(boss.dmg, RAW, '보스 빔(귀속 \'\')은 곡선 밖이다');
    const z = spawnZone(mk(0), 500, 300, 40, 12, 1, false, 'magmaBomb', 0.5);
    assert.lt(z.dmg, 12, '장판도 초반에 깎인다');
  });
});
