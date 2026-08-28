/**
 * tests/weapons2.test.mjs — orbit · barrage · drone 의 정본(§9.5) 계약 단위 테스트.
 *
 * 원칙(MEMORY ★★): 값은 데이터/정본에서 유도한다(하드코딩 매직넘버 지양).
 *
 * 커버:
 *   orbit   — bodyCount 개가 orbitRadius 원주에 «균등» 배치 / 수명으로 사라지지 않는다(회귀:
 *             lifetimeSec=hitCooldownSec 인데 age 를 매 틱 0 으로 되돌린다) / 각속도 = angularSpeedDegSec
 *             / 이지스는 evolved 에서만 적 탄을 지운다
 *             / 클러스터는 evolved 에서만 (blastRadius 밖 · evoClusterRadius+blastRadius 안)
 *   barrage — cooldownSec 마다 예고 / 예고 반경은 evolved 에서 evoRadiusMul 배 / telegraphSec 전엔
 *             무피해·후엔 폭발+반납 / targetMode 'densest'(Lv8)는 «가장 밀집한 적» 위에 떨어진다
 *   drone   — anchorOffsets 자리에 droneCount 개 / droneRangePx 밖이면 발사 안 함 / droneFireSec 주기
 *             / 진화(잔상 편대)는 위치가 지연 추종한다
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import {
  createWorld, recomputeEff, giveWeapon, spawnEnemy, spawnEnemyBullet,
} from '../src/core/state.js';
import { step, makeInput, TICK_DT } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import { TAU, DEG2RAD, wrapAngle } from '../src/core/angle.js';

const dt = TICK_DT;

// ── 헬퍼 ────────────────────────────────────────────────────────────────
function mkWorld(seed = 1) {
  return createWorld({ data: loadData(), seed, weapons, hooks: { enemies: null, emitters: null }, startWeaponId: 'forward' });
}
function slotOf(world, family) {
  for (let i = 0; i < world.slots.length; i += 1) if (world.slots[i].family === family) return world.slots[i];
  return null;
}
/** family 하나만 남기고 원하는 level/evolved 로 세팅한 뒤 [slot, eff] 를 돌려준다 */
function setup(world, family, level, evolved) {
  if (slotOf(world, family) === null) giveWeapon(world, family);
  for (const s of world.slots) {
    if (s.weaponId !== null && s.family !== family) { s.weaponId = null; s.family = ''; }
  }
  const s = slotOf(world, family);
  s.level = level;
  s.evolved = !!evolved;
  s.cooldownT = 0; s.a0 = 0; s.a1 = 0; s.a2 = 0;
  s.effDirty = true;
  return [s, recomputeEff(world, s)];
}
/** weapons.json 원본 evolution.params (데이터 유도용) */
function evoOf(world, family) {
  for (const w of world.data.weapons.weapons) if (w.family === family) return w.evolution.params;
  throw new Error(`no weapon ${family}`);
}
function liveBullets(world, family) {
  const out = [];
  for (const b of world.playerBullets.items) if (b.alive && b.family === family) out.push(b);
  return out;
}
function liveZones(world) {
  const out = [];
  for (const z of world.zones.items) if (z.alive) out.push(z);
  return out;
}
function liveTelegraphs(world) {
  const out = [];
  for (const t of world.telegraphs.items) if (t.alive) out.push(t);
  return out;
}
function liveDrones(world) {
  const out = [];
  for (const d of world.drones.items) if (d.alive) out.push(d);
  return out;
}
/** 죽지 않을 만큼 두꺼운 적 하나 */
function fatEnemy(world, x, y) {
  const e = spawnEnemy(world, 'drifter', 'normal', x, y, 1e9, false);
  e.vx = 0; e.vy = 0;
  return e;
}
/** 무기 update 만 n 틱 돌린다(적 이동·스폰 없이 무기 계약만 본다) */
function tickWeapon(world, family, slot, n) {
  for (let i = 0; i < n; i += 1) {
    slot.effDirty = true;
    const eff = recomputeEff(world, slot);
    weapons[family].update(world, slot, eff, dt);
    // 이 파일이 소유하지 않는 나이 먹이기(step.hazards 몫)를 흉내낸다
    for (const z of world.zones.items) if (z.alive) z.age += dt;
    for (const t of world.telegraphs.items) if (t.alive) t.age += dt;
  }
}

// ══════════════════════════════════════════════════════════════════════
suite('weapons/orbit — 공전체 (§9.5)', () => {
  test('bodyCount 개가 orbitRadius 원주에 균등 배치된다', () => {
    const w = mkWorld();
    const [s, eff] = setup(w, 'orbit', 1, false);
    tickWeapon(w, 'orbit', s, 1);
    const bs = liveBullets(w, 'orbit');
    assert.eq(bs.length, eff.bodyCount, 'bodyCount 개');

    const angles = bs.map((b) => Math.atan2(b.y - w.player.y, b.x - w.player.x)).sort((a, b) => a - b);
    for (const b of bs) {
      const d = Math.hypot(b.x - w.player.x, b.y - w.player.y);
      assert.near(d, eff.orbitRadius, 1e-6, '반경 = orbitRadius');
    }
    for (let i = 1; i < angles.length; i += 1) {
      assert.near(wrapAngle(angles[i] - angles[i - 1]), TAU / eff.bodyCount, 1e-6, '균등 간격');
    }
  });

  test('회귀: 수명(lifetimeSec = hitCooldownSec)이 지나도 사라지지 않는다', () => {
    const w = mkWorld();
    const [s, eff] = setup(w, 'orbit', 1, false);
    const ticks = Math.ceil((eff.hitCooldownSec * 3) / dt);
    assert.gt(ticks, 1, '실제로 수명을 여러 번 넘긴다 (vacuous 아님)');
    tickWeapon(w, 'orbit', s, ticks);
    assert.eq(liveBullets(w, 'orbit').length, eff.bodyCount, '여전히 bodyCount 개');
  });

  test('각속도가 angularSpeedDegSec 와 같다', () => {
    const w = mkWorld();
    const [s, eff] = setup(w, 'orbit', 1, false);
    tickWeapon(w, 'orbit', s, 1);
    const b0 = liveBullets(w, 'orbit')[0];
    const a0 = Math.atan2(b0.y - w.player.y, b0.x - w.player.x);
    const n = 30;
    tickWeapon(w, 'orbit', s, n);
    const b1 = liveBullets(w, 'orbit')[0];
    const a1 = Math.atan2(b1.y - w.player.y, b1.x - w.player.x);
    assert.near(wrapAngle(a1 - a0), eff.angularSpeedDegSec * DEG2RAD * dt * n, 1e-6, '회전량');
  });

  test('이지스(적 탄 제거)는 evolved 에서만 일어난다', () => {
    for (const evolved of [false, true]) {
      const w = mkWorld();
      const [s, eff] = setup(w, 'orbit', 8, evolved);
      tickWeapon(w, 'orbit', s, 1);
      const b = liveBullets(w, 'orbit')[0];
      spawnEnemyBullet(w, w.data.bullets.bullets[0].id, b.x, b.y, 0, 0);
      assert.eq(w.enemyBullets.live, 1, '적 탄이 실제로 놓였다');
      s.a1 = 0;                                     // 방패 재사용 대기 해제
      tickWeapon(w, 'orbit', s, 1);
      assert.eq(w.enemyBullets.live, evolved ? 0 : 1, `evolved=${evolved} 일 때 제거 여부`);
    }
  });
});

// ══════════════════════════════════════════════════════════════════════
suite('weapons/barrage — 바라지 (§9.5)', () => {
  test('한 틱에 1발씩 strikeIntervalSec 간격으로 볼리를 깐다', () => {
    const w = mkWorld();
    const [s, eff] = setup(w, 'barrage', 8, false);
    assert.gt(eff.strikesPerVolley, 1, 'Lv8 볼리는 여러 발이다 (vacuous 아님)');
    tickWeapon(w, 'barrage', s, 1);
    assert.eq(liveTelegraphs(w).length, 1, '첫 틱엔 1발');
    assert.eq(liveTelegraphs(w)[0].kind, 'strike', 'kind = strike');
    assert.near(liveTelegraphs(w)[0].durSec, eff.telegraphSec, 1e-9, '예고 시간');
    tickWeapon(w, 'barrage', s, Math.ceil(eff.strikeIntervalSec / dt));
    assert.eq(liveTelegraphs(w).length, 2, '한 간격 뒤 2발');
  });

  test('예고 반경은 evolved 에서 정확히 evoRadiusMul 배다', () => {
    const rs = [];
    for (const evolved of [false, true]) {
      const w = mkWorld();
      const [s] = setup(w, 'barrage', 8, evolved);   // ★ 같은 레벨 — 차이는 진화 분기뿐
      tickWeapon(w, 'barrage', s, 1);
      rs.push(liveTelegraphs(w)[0].r);
    }
    assert.near(rs[1] / rs[0], evoOf(mkWorld(), 'barrage').evoRadiusMul, 1e-9, 'evoRadiusMul 배');
  });

  test('telegraphSec 전엔 무피해, 익으면 터지고 예고가 반납된다', () => {
    const w = mkWorld();
    const [s, eff] = setup(w, 'barrage', 1, false);
    tickWeapon(w, 'barrage', s, 1);
    const t = liveTelegraphs(w)[0];
    const e = fatEnemy(w, t.x, t.y);
    const hp0 = e.hp;

    const pre = Math.floor(eff.telegraphSec / dt) - 1;
    assert.gt(pre, 0, '예고가 여러 틱이다 (vacuous 아님)');
    tickWeapon(w, 'barrage', s, pre);
    assert.eq(e.hp, hp0, '예고 중엔 무피해');

    tickWeapon(w, 'barrage', s, 3);
    assert.lt(e.hp, hp0, '익으면 폭발');
    assert.eq(t.alive, false, '예고는 소유자가 반납한다');
  });

  test("Lv8 targetMode 'densest' 는 가장 밀집한 무리 위에 떨어진다", () => {
    const w = mkWorld();
    const [s, eff] = setup(w, 'barrage', 8, true);
    assert.eq(eff.targetMode, 'densest', 'Lv8 이 targetMode 를 바꾼다 (회귀: 예전엔 여기서 던졌다)');
    const a = w.data.rules.view.arena;
    const lone = fatEnemy(w, a.x + a.w * 0.15, a.y + a.h * 0.2);
    const cx = a.x + a.w * 0.75;
    const cy = a.y + a.h * 0.7;
    for (let i = 0; i < 5; i += 1) fatEnemy(w, cx + i, cy + i);    // 뭉친 무리
    tickWeapon(w, 'barrage', s, 1);
    const t = liveTelegraphs(w)[0];
    const dCluster = Math.hypot(t.x - cx, t.y - cy);
    const dLone = Math.hypot(t.x - lone.x, t.y - lone.y);
    assert.lt(dCluster, dLone, '외톨이보다 무리 쪽에 떨어진다');
    assert.lt(dCluster, 16, '무리 한복판이다');
  });
});

// ══════════════════════════════════════════════════════════════════════
suite('weapons/drone — 옵션 (§9.5)', () => {
  test('anchorOffsets 자리에 droneCount 개가 배치된다', () => {
    const w = mkWorld();
    const [s, eff] = setup(w, 'drone', 1, false);
    tickWeapon(w, 'drone', s, 1);
    const ds = liveDrones(w);
    assert.eq(ds.length, eff.droneCount, 'droneCount 개');
    const off = eff.anchorOffsets[0];
    assert.near(ds[0].x, w.player.x + off[0], 1e-6, 'anchor x');
    assert.near(ds[0].y, w.player.y + off[1], 1e-6, 'anchor y');
  });

  test('droneRangePx 밖이면 정면(§1.1), 안이면 그 적을 조준한다', () => {
    const w = mkWorld();
    const [s, eff] = setup(w, 'drone', 1, false);
    // 옆으로 멀리 — 사거리 밖이면 조준하지 않으므로 «정면(위)» 이어야 한다
    const e = fatEnemy(w, w.player.x + eff.droneRangePx * 2, w.player.y);
    tickWeapon(w, 'drone', s, 1);
    let bs = liveBullets(w, 'drone');
    assert.eq(bs.length, 1, '사거리 밖에서도 쏜다');
    assert.near(bs[0].vx, 0, 1e-6, '정면 — 가로 성분 0');
    assert.lt(bs[0].vy, 0, '정면 — 위로');

    e.x = w.player.x + eff.droneRangePx * 0.5;                     // 사거리 안으로
    tickWeapon(w, 'drone', s, Math.ceil(eff.droneFireSec / dt) + 1);
    bs = liveBullets(w, 'drone');
    assert.gt(bs.length, 1, '다음 주기에 또 쏜다');
    const last = bs[bs.length - 1];
    assert.gt(last.vx, 0, '사거리 안 — 적 쪽(오른쪽)으로 조준');
  });

  test('진화(잔상 편대)는 앵커를 지연 추종한다', () => {
    const w = mkWorld();
    const [s] = setup(w, 'drone', 8, true);
    tickWeapon(w, 'drone', s, 1);
    const d = liveDrones(w)[0];
    const before = { x: d.x, y: d.y };
    w.player.x += 120;                                             // 급이동
    tickWeapon(w, 'drone', s, 1);
    const moved = Math.abs(d.x - before.x);
    assert.gt(moved, 0, '따라오긴 한다');
    assert.lt(moved, 120, '한 틱에 다 따라잡지 않는다 (지연)');
  });
});
