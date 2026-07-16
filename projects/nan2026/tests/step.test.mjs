/**
 * tests/step.test.mjs — src/core/step.js 계약 단위 테스트
 *
 * 대상 (임무):
 *   · 이동: bounds 클램프 · 대각 정규화 · SOCD lastInput (§2.2)
 *   · i-frame: 모든 피해원 공유, **게임초당 최대 1회** (§2.4) — 회귀
 *   · 관통 hitGen: 적 슬롯 재사용(release→재alloc, gen++) 후 관통탄이 새 적을 통과하지 않는다 — 회귀
 *   · killEnemy: 진입 !alive 가드 = 멱등 + §8.6 XP 드랍 (D3) — 회귀
 *   · 레벨업: serial 큐 (§6.4) · 스탠스 런 유지 · 발사 → 탄 스폰
 *
 * world = createWorld({ data, seed, weapons }) · hooks 없이 (1주차: 적 등속 적분만).
 * ★ 결정성 = 같은 시드+입력열 → 비트 동일 / 다른 시드 → 상이.
 */
import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld, spawnEnemy, spawnPlayerBullet, spawnEnemyBullet, xpToNext } from '../src/core/state.js';
import { step, makeInput, killEnemy, TICK_DT } from '../src/core/step.js';
import { investElement, requestStance } from '../src/core/stance.js';
import { weapons } from '../src/core/weapons/index.js';

function mk(seed = 1) {
  return createWorld({ data: loadData(), seed, weapons });
}
/** 자동 발사가 테스트 대상 풀을 오염시키지 않도록 무기를 침묵시킨다 (family 는 남겨 spawn 계약 유지) */
function silence(w) {
  for (const s of w.slots) s.weaponId = null;
}
function down(k) { const i = makeInput(); i[k] = true; return i; }

// ─────────────────────────────────────────────────────────────────────────
suite('step · 이동 (§2.2)', () => {
  test('등속 — 관성 없음 (moveResponseTau 0): Δ = moveSpeed × dt', () => {
    const w = mk();
    silence(w);
    const rp = w.data.rules.player;
    const y0 = w.player.y;
    step(w, down('up'), TICK_DT);
    assert.near(w.player.y, y0 - rp.moveSpeed * TICK_DT, 1e-9, 'up 1틱 = -moveSpeed·dt');
    assert.eq(w.player.vy, -rp.moveSpeed, 'vy 즉시 = -moveSpeed (가속 없음)');
  });

  test('bounds 클램프 — 위로 계속 = minY 에 정확히 고정', () => {
    const w = mk();
    silence(w);
    for (let i = 0; i < 400; i += 1) step(w, down('up'), TICK_DT);
    assert.eq(w.player.y, w.bounds.minY, 'y 클램프 = minY (파생값)');
    for (let i = 0; i < 400; i += 1) step(w, down('left'), TICK_DT);
    assert.eq(w.player.x, w.bounds.minX, 'x 클램프 = minX');
    // 좌표 유한 불변식
    assert.finite(w.player.x, 'x 유한');
    assert.finite(w.player.y, 'y 유한');
  });

  test('대각 정규화 — 두 축 동시 = ×0.70710678 (속도 상한 보존)', () => {
    const w = mk();
    silence(w);
    const rp = w.data.rules.player;
    assert.ok(rp.diagonalNormalize, '정본: diagonalNormalize true');
    const DIAG = 0.70710678;
    const x0 = w.player.x; const y0 = w.player.y;
    const inp = makeInput(); inp.up = true; inp.left = true;
    step(w, inp, TICK_DT);
    assert.near(w.player.x, x0 - rp.moveSpeed * DIAG * TICK_DT, 1e-6, 'x 대각 성분');
    assert.near(w.player.y, y0 - rp.moveSpeed * DIAG * TICK_DT, 1e-6, 'y 대각 성분');
    // 두 축 성분 크기가 같다 = 45° 정규화
    assert.near(Math.abs(x0 - w.player.x), Math.abs(y0 - w.player.y), 1e-9, '두 축 동일 성분');
  });

  test('SOCD lastInput — 반대키 동시 입력이면 마지막에 눌린 키가 이긴다', () => {
    const w = mk();
    silence(w);
    // 틱1: left 먼저 → 왼쪽으로
    let x = w.player.x;
    step(w, down('left'), TICK_DT);
    assert.lt(w.player.x, x, '틱1 left → x 감소');
    // 틱2: left 유지 + right 추가(상승 엣지) → 마지막에 눌린 right 가 이긴다
    x = w.player.x;
    const both = makeInput(); both.left = true; both.right = true;
    step(w, both, TICK_DT);
    assert.gt(w.player.x, x, '틱2 left+right 동시 → 마지막 눌린 right 승 → x 증가');
    assert.eq(w.player.lastHorizontal, 1, 'lastHorizontal = 1 (right)');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('step · i-frame 게임초당 1회 (§2.4) — 회귀', () => {
  test('정지 몸통 충돌이 매 틱 겹쳐도 1.0 게임초당 정확히 1회만 피격', () => {
    const w = mk();
    silence(w);
    const px = w.player.x; const py = w.player.y;
    const def = w.data.enemies.archetypes.find((a) => a.id === 'columnAnt');
    const e = spawnEnemy(w, 'columnAnt', 'normal', px, py, def.hp, false);
    let hits = 0; let prev = w.player.hp;
    // 60틱 = 1.0 게임초 → 정확히 1회
    for (let i = 0; i < 60; i += 1) {
      e.x = px; e.y = py;                 // 적을 계속 겹쳐 둔다 (매 틱 피해원 존재)
      step(w, makeInput(), TICK_DT);
      if (w.player.hp < prev) { hits += 1; prev = w.player.hp; }
    }
    assert.eq(hits, 1, '1.0 게임초 = 1회 (i-frame 게이트)');
    // 120틱 = 2.0 게임초 → 정확히 2회 (게이트가 진짜로 시간당 1회임을 증명; 미게이트면 매 틱)
    for (let i = 0; i < 60; i += 1) {
      e.x = px; e.y = py;
      step(w, makeInput(), TICK_DT);
      if (w.player.hp < prev) { hits += 1; prev = w.player.hp; }
    }
    assert.eq(hits, 2, '2.0 게임초 = 2회 (미게이트였다면 수십 회)');
    assert.lt(hits, 120, '틱마다 맞지 않는다 = 게이트 존재');
    // HP 불변식: [0, hpMax]
    assert.gte(w.player.hp, 0, 'hp >= 0');
    assert.lte(w.player.hp, w.player.hpMax, 'hp <= hpMax');
  });

  test('i-frame 중 겹친 탄은 소멸하지 않고 통과한다 (§2.4 v1.4)', () => {
    const w = mk();
    silence(w);
    const px = w.player.x; const py = w.player.y;
    // 피격을 유발할 탄 하나 (피해 적용 → 소멸) + i-frame 중 겹칠 탄 하나 (통과)
    spawnEnemyBullet(w, 'pelletS', px, py, 0, 0);
    spawnEnemyBullet(w, 'pelletS', px, py, 0, 0);
    assert.eq(w.enemyBullets.live, 2, '두 탄 배치');
    step(w, makeInput(), TICK_DT);
    // 정확히 한 탄만 소멸 (피해 준 그 탄). 나머지는 i-frame 중이라 통과 = 여전히 살아있다
    assert.eq(w.enemyBullets.live, 1, 'i-frame 중 겹친 탄은 통과 (1개만 소멸)');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('step · 관통 hitGen (§9.5) — 회귀', () => {
  test('적 슬롯 재사용(gen++) 후 관통탄이 새 적을 조용히 통과하지 않는다', () => {
    const w = mk();
    silence(w);
    const slot = w.slots[0];                              // family forward, stampElement normal
    const eff = { dmg: 50, projRadius: 6, pierce: 0, hitCooldownSec: 0, lifetimeSec: 999 };
    const b = spawnPlayerBullet(w, slot, eff, 600, 300, 0, 0, 1);
    b.pierceLeft = -1;                                    // 무제한 관통 (탄이 살아남아 재히트 가능)
    b.lifetimeSec = 999;
    const A = spawnEnemy(w, 'drifter', 'normal', 600, 300, 3, false);
    const idxA = A.idx; const genA = A.gen;
    step(w, makeInput(), TICK_DT);                        // 탄 dmg 50 > 3 → A 처치, 탄 생존
    assert.ok(!A.alive, 'A 처치됨');
    assert.ok(b.alive, '무제한 관통탄 생존');
    const B = spawnEnemy(w, 'drifter', 'normal', 600, 300, 3, false);
    assert.eq(B.idx, idxA, 'B 가 A 의 슬롯 재사용 (같은 idx)');
    assert.eq(B.gen, genA + 1, 'gen++ = 다른 개체');
    step(w, makeInput(), TICK_DT);                        // hitGen: (hitEpoch, e.gen) 쌍으로 판정
    assert.ok(!B.alive, 'B 가 피격 처치됨 — 재사용 슬롯을 통과하지 않는다 (hitGen 가드)');
  });

  test('같은 개체(gen 동일)는 hitCooldownSec 0 에서 재히트하지 않는다 (음성 대칭)', () => {
    const w = mk();
    silence(w);
    const slot = w.slots[0];
    const eff = { dmg: 1, projRadius: 6, pierce: 0, hitCooldownSec: 0, lifetimeSec: 999 };
    const b = spawnPlayerBullet(w, slot, eff, 600, 300, 0, 0, 1);
    b.pierceLeft = -1; b.lifetimeSec = 999;
    const A = spawnEnemy(w, 'drifter', 'normal', 600, 300, 100, false);   // 안 죽는 hp
    const hp0 = A.hp;
    step(w, makeInput(), TICK_DT);                        // 1히트
    const hp1 = A.hp;
    assert.lt(hp1, hp0, '1히트 = hp 감소');
    step(w, makeInput(), TICK_DT);                        // 같은 gen + hitCooldownSec 0 → 재히트 없음
    assert.eq(A.hp, hp1, 'hitCooldownSec 0 = 같은 대상 정확히 1회 (재히트 없음)');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('step · killEnemy 멱등 + §8.6 드랍 (D3) — 회귀', () => {
  test('killEnemy = XP 드랍 + 풀 release, 두 번째 호출은 무해 (멱등)', () => {
    const w = mk();
    const e = spawnEnemy(w, 'drifter', 'normal', 500, 300, 6, false);   // chaff, coin 0
    const xp = e.xp; const ex = e.x; const ey = e.y;
    const pickBefore = w.pickups.live;
    const enemyLive = w.enemies.live;
    killEnemy(w, e);
    assert.ok(!e.alive, '처치 후 release');
    assert.eq(w.enemies.live, enemyLive - 1, '적 풀 live 감소');
    assert.eq(w.pickups.live, pickBefore + 1, 'XP 픽업 1개 (chaff coinDropChance 0 → 코인 없음)');
    // 드랍된 XP 픽업 검증
    let found = null;
    for (const q of w.pickups.items) if (q.alive && q.kind === 'xp') { found = q; break; }
    assert.ok(found !== null, 'XP 픽업 존재');
    assert.eq(found.value, xp, 'XP value = e.xp');
    assert.eq(found.x, ex, '드랍 위치 x'); assert.eq(found.y, ey, '드랍 위치 y');
    // 멱등 — 같은 틱에 두 피해원이 부르는 상황
    const pickAfter = w.pickups.live; const enemyAfter = w.enemies.live;
    killEnemy(w, e);
    assert.eq(w.pickups.live, pickAfter, '이중 처치는 드랍을 두 번 하지 않는다');
    assert.eq(w.enemies.live, enemyAfter, '이중 release 없음 (free 스택 무결)');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('step · 레벨업 · 스탠스 · 발사', () => {
  test('레벨업 serial — 동시 다중은 병합 없이 순차 (§6.4)', () => {
    const w = mk();
    silence(w);
    const need = xpToNext(w, 1) + xpToNext(w, 2);         // 2레벨분 + 여분
    w.player.xp = need + 0.5;
    const q0 = w.draftQueue;
    step(w, makeInput(), TICK_DT);
    assert.eq(w.player.level, 3, 'Lv1 → Lv3 (2회 순차)');
    assert.eq(w.draftQueue, q0 + 2, '드래프트 큐 +2 (레벨당 1장)');
    assert.near(w.player.xp, 0.5, 1e-9, '남은 XP = 0.5 (float 누산)');
    assert.lt(w.player.xp, w.player.xpToNext, '다음 임계 미만');
  });

  test('스탠스 런 유지 — 부여 후 스텝을 흘려도 각인이 유지된다 (§4.3)', () => {
    const w = mk();
    silence(w);
    investElement(w, 'fire');                             // invest.fire = 1
    requestStance(w, 'fire');                             // stance fire, 슬롯 1..N = fire
    assert.eq(w.player.stance, 'fire', '스탠스 fire');
    assert.eq(w.slots[0].stampElement, 'fire', 'invest 1 → 슬롯1(=idx0) fire');
    assert.eq(w.slots[1].stampElement, 'normal', '슬롯2 이후 무속성');
    for (let i = 0; i < 30; i += 1) step(w, makeInput(), TICK_DT);
    assert.eq(w.player.stance, 'fire', '스탠스 입력 없으면 런 내내 유지');
    assert.eq(w.slots[0].stampElement, 'fire', '각인 유지 (재계산 이벤트 없음)');
  });

  test('발사 → 탄 스폰: forward 는 첫 틱에 즉시 발사 (cooldownT 0)', () => {
    const w = mk();
    assert.eq(w.playerBullets.live, 0, '시작 시 탄 0');
    step(w, makeInput(), TICK_DT);
    assert.gte(w.playerBullets.live, 1, '첫 틱에 볼리 발사');
    let b = null;
    for (const x of w.playerBullets.items) if (x.alive) { b = x; break; }
    assert.eq(b.family, 'forward', '발사 무기 = forward');
    assert.lt(b.vy, 0, '정면 = 화면 위쪽 (vy < 0)');
  });
});

// ─────────────────────────────────────────────────────────────────────────
suite('step · 결정성 (§10.2/§10.3)', () => {
  test('같은 시드 + 같은 입력열 → 상태 비트 동일 / 다른 시드 → 상이', () => {
    const inputs = [down('up'), down('left'), makeInput(), down('right'), down('down')];
    function run(seed) {
      const w = mk(seed);
      for (let r = 0; r < 40; r += 1) step(w, inputs[r % inputs.length], TICK_DT);
      return JSON.stringify({
        x: w.player.x, y: w.player.y, tick: w.tick,
        pb: w.playerBullets.live, en: w.enemies.live,
        rngDraft: w.rng.draft.u32(),
      });
    }
    assert.eq(run(12345), run(12345), '같은 시드 = 비트 동일');
    assert.ne(run(12345), run(9999), '다른 시드 = 상이');
  });
});
