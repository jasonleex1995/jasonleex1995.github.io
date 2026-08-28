/**
 * tests/hazards.test.mjs — §8.5 zone(장판) · laser(빔) 인프라 (step.hazards + state.spawnZone/spawnBeam).
 *
 * 정본 계약:
 *   - 피해는 **적용 1회 = dmg**이며 i-frame 이 게이트한다("zone 의 dps 는 존재하지 않는다", §8.5).
 *   - 수명(activeSec)이 다하면 반납. 새 풀 없이 zones/telegraphs 풀을 재사용한다(§12.1 S2).
 *   - fromPlayer 장판(무기 A2)은 플레이어를 때리지 않는다.
 *   - 빔은 원점에서 a 방향 **반직선** — 뒤쪽·폭 밖은 맞지 않는다.
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld, spawnZone, spawnBeam, giveWeapon } from '../src/core/state.js';
import { step, makeInput, TICK_DT } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';

function mkWorld(seed = 1) { return createWorld({ data: loadData(), seed, weapons, hooks: {} }); }
function liveBeams(w) { let n = 0; for (const t of w.telegraphs.items) if (t.alive && t.kind === 'laser') n += 1; return n; }

suite('hazards/zone 장판 (§8.5)', () => {
  test('안에 있으면 피격, 그리고 i-frame 이 게이트한다(적용 1회 · dps 없음)', () => {
    const w = mkWorld(); const p = w.player;
    const z = spawnZone(w, p.x, p.y, 60, 10, 1.0, false);
    assert.ok(z, '장판 스폰');
    assert.eq(w.zones.live, 1, '장판 1개');
    p.iframeSec = 0;
    const hp0 = p.hp;
    step(w, makeInput(), TICK_DT);
    assert.lt(p.hp, hp0, '장판 안 = 피격');
    const hp1 = p.hp;
    step(w, makeInput(), TICK_DT);
    assert.eq(p.hp, hp1, 'i-frame 게이트 — 연속 틱에 추가 피해 없음');
  });

  test('반경 밖은 무피격', () => {
    const w = mkWorld(); const p = w.player;
    spawnZone(w, p.x + 500, p.y, 40, 10, 1.0, false);
    p.iframeSec = 0;
    const hp0 = p.hp;
    step(w, makeInput(), TICK_DT);
    assert.eq(p.hp, hp0, '반경 밖 = 무피격');
  });

  test('activeSec 경과 후 반납', () => {
    const w = mkWorld();
    spawnZone(w, 0, 0, 40, 10, 0.2, false);
    assert.eq(w.zones.live, 1, '살아있음');
    for (let t = 0; t < 20; t += 1) step(w, makeInput(), TICK_DT);   // 0.33s > 0.2s
    assert.eq(w.zones.live, 0, 'activeSec 후 반납');
  });

  test('fromPlayer 장판은 플레이어를 때리지 않는다 (무기 A2 용도)', () => {
    const w = mkWorld(); const p = w.player;
    spawnZone(w, p.x, p.y, 60, 10, 1.0, true);
    p.iframeSec = 0;
    const hp0 = p.hp;
    step(w, makeInput(), TICK_DT);
    assert.eq(p.hp, hp0, '플레이어 장판은 플레이어에게 무해');
  });

  test('§8.5 mortar 퓨즈 — warnSec 동안 무해, 그 뒤 폭발 피해, warnSec+activeSec 에 반납', () => {
    const w = mkWorld(); const p = w.player;
    const warnSec = 0.5, activeSec = 0.3;
    spawnZone(w, p.x, p.y, 60, 10, activeSec, false, '', warnSec);   // 발밑 퓨즈 폭탄(warnSec 9번째 인자)
    const hp0 = p.hp;
    for (let t = 0; t < 18; t += 1) { p.iframeSec = 0; step(w, makeInput(), TICK_DT); }   // ~0.30s < warnSec
    assert.eq(p.hp, hp0, '퓨즈 동안 무해(회피 창)');
    assert.eq(w.zones.live, 1, '폭발 전이라 살아있음');
    for (let t = 0; t < 15; t += 1) { p.iframeSec = 0; step(w, makeInput(), TICK_DT); }   // ~0.55s > warnSec
    assert.lt(p.hp, hp0, '퓨즈 경과 후 = 폭발 피해');
    for (let t = 0; t < 30; t += 1) step(w, makeInput(), TICK_DT);                         // > warnSec+activeSec
    assert.eq(w.zones.live, 0, 'warnSec+activeSec 후 반납');
  });
});

suite('hazards/laser 빔 (§8.5)', () => {
  test('빔 경로 위면 피격', () => {
    const w = mkWorld(); const p = w.player;
    // 플레이어 위 200px 원점에서 아래(+y, a=π/2)로 → 플레이어가 경로상
    const b = spawnBeam(w, p.x, p.y - 200, Math.PI / 2, 20, 15, 1.0, -1);
    assert.ok(b, '빔 스폰');
    assert.eq(liveBeams(w), 1, '빔 1개');
    p.iframeSec = 0;
    const hp0 = p.hp;
    step(w, makeInput(), TICK_DT);
    assert.lt(p.hp, hp0, '빔 경로 = 피격');
  });

  // §9.5(v1.7) 오빗 차폐 — 「펄스필드는 탄은 막는데 빔은 못 막는다」의 답.
  //   ★ 이 테스트가 있어야 하는 이유: 순수 기하로 구현하면 차단 창이 0.08~0.15초인데 빔 활성은
  //     0.5~2.2초라 «실측 0%»가 나온다. 컴파일되고 게이트를 통과해도 아무 일도 안 일어나는
  //     장식이 되는 것이다. 그래서 「피해가 실제로 줄었는가」를 못박는다.
  //   ★ 공전체를 손으로 옮기면 안 된다 — 같은 스텝의 orbit.place() 가 궤도로 되돌린다.
  //     링 «위상»(slot.a0)으로 제어해야 결정적이다.
  test('오빗 공전체가 빔 원점과 나 사이에 서면 피해가 준다 (§9.5 차폐)', () => {
    // 빔은 플레이어 200px 위 원점에서 아래로 → 선분은 플레이어 «바로 위» 수직선이다.
    //   a0 = -π/2 → 0번 공전체가 정확히 그 위에 선다(막힘). a0 = 0 → 좌우로 비킨다(열림).
    const run = (phase) => {
      const w = mkWorld(); const p = w.player;
      const si = giveWeapon(w, 'orbit');
      for (let t = 0; t < 20; t += 1) step(w, makeInput(), TICK_DT);
      w.slots[si].a0 = phase;
      spawnBeam(w, p.x, p.y - 200, Math.PI / 2, 20, 40, 1.0, -1);
      p.iframeSec = 0;
      const hp0 = p.hp;
      step(w, makeInput(), TICK_DT);
      return hp0 - p.hp;
    };
    const f = mkWorld().data.rules.fairness;
    assert.gt(f.beamBlockRadiusPx, 0, '판정 폭이 있다(전제)');
    assert.gt(f.beamBlockRatio, 0, '감산이 있다(전제)');

    const open = run(0);
    const blocked = run(-Math.PI / 2);
    assert.gt(open, 0, '막지 않으면 맞는다(전제)');
    assert.lt(blocked, open, '★ 공전체가 선분 위에 있으면 피해가 «실제로» 준다');
    assert.gt(blocked, 0, '감산이지 무효화가 아니다 — §2.1 관대함(영구 무효 없음)');
  });

  test('빔 뒤쪽(원점 반대편)은 무피격 — 반직선이다', () => {
    const w = mkWorld(); const p = w.player;
    spawnBeam(w, p.x, p.y + 100, Math.PI / 2, 20, 15, 1.0, -1);   // 원점이 플레이어 아래, 방향은 아래
    p.iframeSec = 0;
    const hp0 = p.hp;
    step(w, makeInput(), TICK_DT);
    assert.eq(p.hp, hp0, '빔 뒤쪽 = 무피격');
  });

  test('빔 폭 밖(옆)은 무피격', () => {
    const w = mkWorld(); const p = w.player;
    spawnBeam(w, p.x + 300, p.y - 200, Math.PI / 2, 20, 15, 1.0, -1);
    p.iframeSec = 0;
    const hp0 = p.hp;
    step(w, makeInput(), TICK_DT);
    assert.eq(p.hp, hp0, '폭 밖 = 무피격');
  });

  test('activeSec 경과 후 반납', () => {
    const w = mkWorld();
    spawnBeam(w, 0, 0, 0, 20, 15, 0.2, -1);
    assert.eq(liveBeams(w), 1, '빔 살아있음');
    for (let t = 0; t < 20; t += 1) step(w, makeInput(), TICK_DT);
    assert.eq(liveBeams(w), 0, 'activeSec 후 반납');
  });
});

// ══════════════════════════════════════════════════════════════════════
suite('hazards/laser 2단(§7.4 「충전이 곧 텔레그래프」)', () => {
  test('충전(warnSec) 구간은 무해, 활성(activeSec) 구간만 피해', () => {
    const w = mkWorld(); const p = w.player;
    const warnSec = 0.5; const activeSec = 0.5;
    // 플레이어 위 200px 원점에서 아래로 → 플레이어가 경로상. warnSec 동안은 안 맞아야 한다.
    spawnBeam(w, p.x, p.y - 200, Math.PI / 2, 20, 15, activeSec, -1, '', warnSec, false);
    const chargeTicks = Math.floor(warnSec / TICK_DT) - 1;
    assert.gt(chargeTicks, 0, '충전이 여러 틱이다 (vacuous 아님)');
    const hp0 = p.hp;
    for (let t = 0; t < chargeTicks; t += 1) { p.iframeSec = 0; step(w, makeInput(), TICK_DT); }
    assert.eq(p.hp, hp0, '충전 중엔 무해 — 예고 없이 즉발하지 않는다');
    p.iframeSec = 0;
    step(w, makeInput(), TICK_DT);
    step(w, makeInput(), TICK_DT);
    assert.lt(p.hp, hp0, '활성 진입 후 피해');
    // 총 수명 = warnSec + activeSec
    assert.eq(liveBeams(w), 1, '아직 활성');
    for (let t = 0; t < Math.ceil(activeSec / TICK_DT) + 1; t += 1) step(w, makeInput(), TICK_DT);
    assert.eq(liveBeams(w), 0, 'warnSec + activeSec 후 반납');
  });

  test('track=true — 충전 초반엔 추적, beamLockSec 잠금창부터 각이 멈추고 그대로 발사(§7.4 잠금창)', () => {
    const w = mkWorld(); const p = w.player;
    const lockSec = w.data.rules.fairness.beamLockSec;     // 잠금·뜸 창(회피 리드)
    const warnSec = lockSec + 0.5;                         // 추적 0.5s + 잠금 lockSec
    const ox = p.x; const oy = p.y - 200;
    const b = spawnBeam(w, ox, oy, Math.PI / 2, 20, 15, 0.5, -1, '', warnSec, true);
    p.iframeSec = 1e9;                                     // 관측 중 피격 무효(각만 본다)

    // ① 잠금창 전(충전 초반): 플레이어를 옮기면 각이 따라온다
    p.x = ox + 150;
    step(w, makeInput(), TICK_DT);
    assert.near(b.a, Math.atan2(p.y - oy, p.x - ox), 1e-6, '잠금창 전엔 추적');

    // ② 잠금창 진입 직전까지 진행 → 각 기록
    const trackUntilTicks = Math.floor((warnSec - lockSec) / TICK_DT);
    for (let t = 1; t < trackUntilTicks; t += 1) step(w, makeInput(), TICK_DT);
    const aLocked = b.a;
    // 잠금창 동안 플레이어를 크게 옮겨도 각이 안 변한다(뜸 = 회피 창)
    p.x = ox - 300;
    step(w, makeInput(), TICK_DT);
    assert.near(b.a, aLocked, 1e-9, '잠금창에선 각이 멈춘다(발사 전 뜸)');

    // ③ 활성(발사) 진입 후에도 잠긴 각 그대로
    for (let t = 0; t < Math.ceil(lockSec / TICK_DT) + 2; t += 1) step(w, makeInput(), TICK_DT);
    assert.near(b.a, aLocked, 1e-9, '발사도 잠긴 각 그대로');
  });
});
