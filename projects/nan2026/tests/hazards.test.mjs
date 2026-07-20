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
import { createWorld, spawnZone, spawnBeam } from '../src/core/state.js';
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
