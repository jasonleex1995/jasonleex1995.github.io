/**
 * tests/weapons.test.mjs — forward · fan · seeker 의 정본(§9.5) 계약 단위 테스트.
 *
 * 원칙(MEMORY ★★): 값은 데이터/정본에서 유도한다(하드코딩 매직넘버 지양).
 *   eff 는 recomputeEff 로 주입한다. 양성 + 경계 + 음성 + 회귀를 모두 건다.
 *
 * 커버:
 *   forward — 볼리 주기 = cooldownSec / spreadDeg 균등·대칭 / jitterDeg(rng.pattern·결정적)
 *             / 진화 on↔off 격리(램프는 evolved 에서만·비진화는 a2 불변)
 *             / i-frame 엣지 리셋(회귀: 피격 직후 그 한 틱)
 *   fan     — arcDeg 균등 / onExpire 는 evolved 일 때만 폭발 / 반경 경계
 *             / onExpire 정확히 1회(회귀 ④: release 경로 LIFO 재사용에도 탄당 1회)
 *   seeker  — 유도(각 오차 감소) / turnRate 클램프(정확히 turnRateDegSec·dt) / retarget
 *             (evolved=온-킬 즉시 / 비evolved=주기 전 직진) / distinct 타겟(evolved)
 *   ★ 리터럴 계약 — 세 파일 소스 파싱: 주석·문자열 제거 후 숫자 리터럴 ⊆ {0,1,-1,0.5,2}
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { suite, test, assert, loadData } from '../tools/test.mjs';
import {
  createWorld, recomputeEff, giveWeapon, levelUpWeapon, spawnPlayerBullet, spawnEnemy,
} from '../src/core/state.js';
import { step, makeInput, TICK_DT, killEnemy } from '../src/core/step.js';
import { weapons } from '../src/core/weapons/index.js';
import forward from '../src/core/weapons/forward.js';
import fan from '../src/core/weapons/fan.js';
import seeker from '../src/core/weapons/seeker.js';
import { DEG2RAD, wrapAngle } from '../src/core/angle.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEAPONS_DIR = join(HERE, '..', 'src', 'core', 'weapons');
const dt = TICK_DT;

// ── 헬퍼 ────────────────────────────────────────────────────────────────
function mkWorld(seed = 1) {
  return createWorld({ data: loadData(), seed, weapons, hooks: { enemies: null, emitters: null } });
}
function slotOf(world, family) {
  for (let i = 0; i < world.slots.length; i += 1) if (world.slots[i].family === family) return world.slots[i];
  return null;
}
/** family 무기를 원하는 level/evolved 로 세팅한 슬롯 + 갱신된 eff 를 돌려준다 */
function setup(world, family, level, evolved) {
  if (slotOf(world, family) === null) giveWeapon(world, family);
  const s = slotOf(world, family);
  s.level = level;
  s.evolved = !!evolved;
  s.cooldownT = 0; s.a0 = 0; s.a1 = 0; s.a2 = 0;
  s.effDirty = true;
  return { s, eff: recomputeEff(world, s) };
}
/** alive 플레이어 탄을 풀 인덱스 오름차순으로 (= 발사 순서) */
function liveBullets(world) {
  const out = [];
  const it = world.playerBullets.items;
  for (let i = 0; i < it.length; i += 1) if (it[i].alive) out.push(it[i]);
  return out;
}
/** 화면 위(0,-v)를 0도로 한 진행 각(도). 오른쪽 = +, 왼쪽 = − */
function headingDeg(b) { return Math.atan2(b.vx, -b.vy) / DEG2RAD; }
/** eff 로 볼리 1발분 스폰이 일어난 틱 인덱스 목록 (live 증가 = 볼리) */
function volleyTicks(mod, world, s, eff, ticks) {
  const ev = [];
  let prev = world.playerBullets.live;
  for (let t = 0; t < ticks; t += 1) {
    mod.update(world, s, eff, dt);
    if (world.playerBullets.live > prev) ev.push(t);
    prev = world.playerBullets.live;
  }
  return ev;
}
function addEnemy(world, x, y, element = 'normal', archetype = 'drifter') {
  const def = loadData().enemies.archetypes.find((a) => a.id === archetype);
  return spawnEnemy(world, archetype, element, x, y, def.hp, false);
}

// ══════════════════════════════════════════════════════════════════════════
// forward
// ══════════════════════════════════════════════════════════════════════════
suite('weapons/forward', () => {
  test('볼리 주기 = cooldownSec (첫 발 즉시, 이후 간격)', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'forward', 1, false);   // count 1 · burstCount 1
    const ev = volleyTicks(forward, w, s, eff, 90);
    assert.eq(ev[0], 0, '첫 볼리는 첫 틱에 즉시');
    assert.gte(ev.length, 2, '90틱(1.5s) 안에 최소 2볼리');
    const gap = ev[1] - ev[0];
    assert.near(gap * dt, eff.cooldownSec, dt * 2, '볼리 간격 = cooldownSec');
  });

  test('한 볼리 = count 발 (경계: count 1 → 산포 없이 정확히 1발)', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'forward', 1, false);
    assert.eq(eff.count, 1, 'Lv1 count = 1');
    forward.update(w, s, eff, dt);
    assert.eq(w.playerBullets.live, 1, 'count 만큼 스폰');
  });

  test('spreadDeg 균등·대칭 (count 2 → ±spreadDeg/2, 간격 = spreadDeg)', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'forward', 5, false);   // Lv5: count 2 · spreadDeg 6
    assert.eq(eff.count, 2, 'Lv5 count = 2');
    assert.gt(eff.spreadDeg, 0, 'Lv5 spreadDeg > 0');
    eff.jitterDeg = 0;                                  // 산포만 격리 (jitter 제거)
    forward.update(w, s, eff, dt);
    const b = liveBullets(w);
    assert.eq(b.length, 2, '볼리 = 2발');
    const d0 = headingDeg(b[0]); const d1 = headingDeg(b[1]);
    assert.near(d0, -eff.spreadDeg / 2, 1e-6, '첫 발 = −spreadDeg/2');
    assert.near(d1, +eff.spreadDeg / 2, 1e-6, '둘째 발 = +spreadDeg/2');
    assert.near(d1 - d0, eff.spreadDeg, 1e-6, '두 발 간격 = spreadDeg');
  });

  test('jitterDeg: |편차| ≤ jitterDeg 이고 rng.pattern 이라 결정적', () => {
    const wa = mkWorld(777); const a = setup(wa, 'forward', 1, false);
    assert.gt(a.eff.jitterDeg, 0, 'Lv1 jitterDeg > 0 (양성 경로)');
    forward.update(wa, a.s, a.eff, dt);
    const ba = liveBullets(wa)[0];
    assert.lte(Math.abs(headingDeg(ba)), a.eff.jitterDeg + 1e-9, '편차 ≤ jitterDeg');
    // 같은 시드 → 같은 jitter (rng.pattern 결정성)
    const wb = mkWorld(777); const b = setup(wb, 'forward', 1, false);
    forward.update(wb, b.s, b.eff, dt);
    assert.eq(headingDeg(liveBullets(wb)[0]), headingDeg(ba), '동일 시드 = 동일 jitter');
  });

  test('경계: jitterDeg 0 → 정확히 정면(vx = 0)', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'forward', 1, false);
    eff.jitterDeg = 0;
    forward.update(w, s, eff, dt);
    assert.near(liveBullets(w)[0].vx, 0, 1e-9, '산포·jitter 0 = 정면');
  });

  test('진화 격리: evolved 는 램프로 가속, 비evolved 는 램프 스크래치(a2) 불변', () => {
    // 비진화: 같은 창에서 a2 는 절대 변하지 않는다 (i-frame 을 걸어도)
    const wn = mkWorld();
    const n = setup(wn, 'forward', 1, false);
    wn.player.iframeSec = wn.data.rules.player.iframeSec;   // 진화라면 리셋을 유발할 신호
    for (let t = 0; t < 200; t += 1) forward.update(wn, n.s, n.eff, dt);
    assert.eq(n.s.a2, 0, '비evolved 는 a2(램프)를 건드리지 않는다');
    const nCount = liveBullets(wn).length;

    // 진화: 피격 없이 연사하면 램프가 차서 같은 창에서 볼리 수가 더 많다
    const we = mkWorld();
    const e = setup(we, 'forward', 1, true);
    we.player.iframeSec = 0;
    let eCount = 0; let prev = 0;
    for (let t = 0; t < 200; t += 1) {
      forward.update(we, e.s, e.eff, dt);
      if (we.playerBullets.live > prev) eCount += 1;
      prev = we.playerBullets.live;
    }
    assert.gt(e.s.a2, 0, 'evolved 는 램프 a2 가 찬다');
    assert.gt(eCount, nCount, '가속으로 evolved 볼리 수 > 비evolved');
  });

  test('회귀(i-frame 엣지): 피격 직후 그 한 틱에만 램프 리셋', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'forward', 1, true);
    const full = eff.evoRampSec;
    // 피격 순간: iframeSec = rules.player.iframeSec → iframeSec + dt ≥ iframeSec → 리셋
    s.a2 = full;
    w.player.iframeSec = w.data.rules.player.iframeSec;
    forward.update(w, s, eff, dt);
    assert.eq(s.a2, 0, '피격 엣지에서 a2 리셋');

    // 음성: i-frame 이 남아 흐르는 중(피격 아님) → 리셋 안 됨, 오히려 램프가 찬다
    s.a2 = 0;
    w.player.iframeSec = 0.1;                 // 0.1 + dt < 1.0 → 엣지 아님
    forward.update(w, s, eff, dt);
    assert.near(s.a2, dt, 1e-9, 'i-frame 잔여 중엔 리셋 없이 dt 만큼 램프');
  });

  test('음성: 계약 밖 targetMode 는 소리내어 실패 (§9.3 폴백 금지)', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'forward', 1, false);
    eff.targetMode = 'nearest';
    assert.throws(() => forward.update(w, s, eff, dt), '계약 밖 targetMode → throw');
  });
});

// ══════════════════════════════════════════════════════════════════════════
// fan
// ══════════════════════════════════════════════════════════════════════════
suite('weapons/fan', () => {
  test('arcDeg 균등 배치 (양 끝 ±arcDeg/2, 인접 간격 균일)', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'fan', 1, false);       // count 3 · arcDeg 40
    assert.gte(eff.count, 3, 'fan 은 다발');
    fan.update(w, s, eff, dt);
    const b = liveBullets(w).map(headingDeg);
    assert.eq(b.length, eff.count, 'count 만큼');
    assert.near(b[0], -eff.arcDeg / 2, 1e-6, '첫 발 = −arcDeg/2');
    assert.near(b[b.length - 1], +eff.arcDeg / 2, 1e-6, '끝 발 = +arcDeg/2');
    const stepD = eff.arcDeg / (eff.count - 1);
    for (let i = 1; i < b.length; i += 1) {
      assert.near(b[i] - b[i - 1], stepD, 1e-6, `간격 균일 @${i}`);
    }
  });

  test('경계: 최대 레벨 다발도 균등 (Lv8 count·arcDeg)', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'fan', 8, true);
    fan.update(w, s, eff, dt);
    const b = liveBullets(w).map(headingDeg);
    assert.eq(b.length, eff.count, 'Lv8 count 만큼');
    const stepD = eff.arcDeg / (eff.count - 1);
    for (let i = 1; i < b.length; i += 1) assert.near(b[i] - b[i - 1], stepD, 1e-6, `간격 균일 @${i}`);
  });

  test('onExpire 폭발은 evolved 일 때만 (반경 안 적에 피해)', () => {
    // evolved: 반경 안 적 hp 감소
    const we = mkWorld();
    const e = setup(we, 'fan', 8, true);
    const be = spawnPlayerBullet(we, e.s, e.eff, 600, 400, 0, 0, 1);
    const near = addEnemy(we, 600 + e.eff.evoBlastRadius * 0.5, 400);   // 반경 안
    const hp0 = near.hp;
    fan.onExpire(we, e.s, e.eff, be);
    assert.lt(near.hp, hp0, 'evolved 폭발 → 반경 안 적 피해');

    // 비evolved: 같은 배치라도 폭발 없음
    const wn = mkWorld();
    const n = setup(wn, 'fan', 7, false);              // Lv7: 진화 전
    const bn = spawnPlayerBullet(wn, n.s, n.eff, 600, 400, 0, 0, 1);
    const stay = addEnemy(wn, 605, 400);
    const shp = stay.hp;
    fan.onExpire(wn, n.s, n.eff, bn);
    assert.eq(stay.hp, shp, '비evolved 는 폭발하지 않는다');
  });

  test('경계: evoBlastRadius 밖의 적은 폭발에 무피해', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'fan', 8, true);
    const b = spawnPlayerBullet(w, s, eff, 600, 400, 0, 0, 1);
    const far = addEnemy(w, 600 + eff.evoBlastRadius + 30, 400);        // 반경 밖 (적 반경 감안 여유)
    const hp0 = far.hp;
    fan.onExpire(w, s, eff, b);
    assert.eq(far.hp, hp0, '반경 밖 = 무피해');
  });

  test('회귀④: release 경로가 onExpire 를 탄당 정확히 1회 (LIFO 재사용에도)', () => {
    // fan 을 계수 래퍼로 감싼 레지스트리로 월드를 만든다 → 모듈 진입 횟수를 직접 센다
    let calls = 0;
    const wrappedFan = { update: fan.update, onExpire(...a) { calls += 1; return fan.onExpire(...a); } };
    const reg = { forward: weapons.forward, fan: wrappedFan, seeker: weapons.seeker };
    const w = createWorld({ data: loadData(), seed: 3, weapons: reg, hooks: { enemies: null, emitters: null } });

    giveWeapon(w, 'fan');
    const s = slotOf(w, 'fan');
    while (s.level < 8) levelUpWeapon(w, s.index);      // Lv8 = 진화
    const eff = recomputeEff(w, s);
    s.cooldownT = 1e9;                                  // 이 창에서 fan 자동발사 봉쇄 (수동 탄만 만료 측정)

    const K = 5;
    for (let i = 0; i < K; i += 1) spawnPlayerBullet(w, s, eff, 600, 400, 0, 0, 1);
    assert.eq(w.playerBullets.live, K, 'K 발 수동 스폰');

    // lifetime 만큼 스텝 → 모든 fan 탄이 release 경로로 소멸
    for (let t = 0; t < 200; t += 1) step(w, makeInput(), dt);

    // fan 탄이 전부 사라졌는지 (forward 자동탄은 family 로 구분되어 무관)
    let fanLive = 0;
    for (const it of w.playerBullets.items) if (it.alive && it.family === 'fan') fanLive += 1;
    assert.eq(fanLive, 0, '모든 fan 탄 release');
    assert.eq(calls, K, 'onExpire 는 탄당 정확히 1회 (누락·중복 없음)');
  });

  test('음성: 계약 밖 targetMode 는 throw', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'fan', 1, false);
    eff.targetMode = 'nearest';
    assert.throws(() => fan.update(w, s, eff, dt), '계약 밖 targetMode → throw');
  });
});

// ══════════════════════════════════════════════════════════════════════════
// seeker
// ══════════════════════════════════════════════════════════════════════════
suite('weapons/seeker', () => {
  /** 위로 나는 seeker 탄 1발을 만들어 target 에 고정한다 (steer 격리를 위해 update 발사는 봉쇄) */
  function armed(w, level, evolved, enemy) {
    const { s, eff } = setup(w, 'seeker', level, evolved);
    s.cooldownT = 1e9;                                 // steer 만 돌리고 volley 는 막는다
    const b = spawnPlayerBullet(w, s, eff, enemy.x, enemy.y + 200, 0, -eff.projSpeed, 1); // 위로
    b.target = enemy.idx; b.targetGen = enemy.gen; b.s0 = eff.retargetSec;
    return { s, eff, b };
  }

  test('유도: 스티어 1틱이 타겟 쪽으로 각 오차를 줄인다', () => {
    const w = mkWorld();
    const en = addEnemy(w, 700, 400);
    const { s, eff, b } = armed(w, 1, false, en);
    b.x = en.x - 150;                                  // 탄을 옆으로 → 위로 날며 타겟은 위-오른쪽(각 오차 존재)
    const err0 = Math.abs(wrapAngle(Math.atan2(en.y - b.y, en.x - b.x) - Math.atan2(b.vy, b.vx)));
    assert.gt(err0, 0, '초기 각 오차 존재 (유도할 여지)');
    seeker.update(w, s, eff, dt);
    const err1 = Math.abs(wrapAngle(Math.atan2(en.y - b.y, en.x - b.x) - Math.atan2(b.vy, b.vx)));
    assert.lt(err1, err0, '각 오차 감소 = 유도');
  });

  test('turnRate 클램프: 큰 각차에서 회전량 = 정확히 turnRateDegSec·dt', () => {
    const w = mkWorld();
    // 탄은 위(−90°)로, 타겟은 정확히 오른쪽(0°) → 각차 90° ≫ maxTurn → 클램프
    const en = addEnemy(w, 700, 400);
    const { s, eff, b } = armed(w, 1, false, en);
    b.x = en.x - 100; b.y = en.y; b.vx = 0; b.vy = -eff.projSpeed;   // 타겟은 dx=+100, dy=0 (오른쪽)
    const before = Math.atan2(b.vy, b.vx);
    seeker.update(w, s, eff, dt);
    const after = Math.atan2(b.vy, b.vx);
    const turned = wrapAngle(after - before);
    const maxTurn = eff.turnRateDegSec * DEG2RAD * dt;
    assert.near(turned, maxTurn, 1e-9, '회전량 = turnRateDegSec·dt (초과 금지)');
  });

  test('retarget(evolved): 타겟 사망 즉시 다른 적으로 재조준', () => {
    const w = mkWorld();
    const a = addEnemy(w, 700, 400);
    const bEn = addEnemy(w, 720, 410);                 // 대체 후보
    const { s, eff, b } = armed(w, 3, true, a);        // evolved → retargetOnKill
    assert.eq(b.target, a.idx, '초기 타겟 = A');
    killEnemy(w, a);                                   // A 사망
    seeker.update(w, s, eff, dt);
    assert.ne(b.target, a.idx, 'A 를 더는 겨누지 않는다');
    assert.gte(b.target, 0, '새 타겟 확보');
    assert.ok(w.enemies.items[b.target].alive, '새 타겟은 살아있는 적');
    assert.eq(b.targetGen, w.enemies.items[b.target].gen, 'targetGen 동기화');
  });

  test('음성(비evolved): 타겟이 죽어도 주기 전엔 즉시 재조준하지 않는다', () => {
    const w = mkWorld();
    const a = addEnemy(w, 700, 400);
    addEnemy(w, 720, 410);
    const { s, eff, b } = armed(w, 3, false, a);       // 비evolved
    b.s0 = eff.retargetSec;                            // 주기 아직 안 참
    const vx0 = b.vx; const vy0 = b.vy;
    killEnemy(w, a);
    seeker.update(w, s, eff, dt);
    assert.eq(b.target, a.idx, '재조준 없이 stale 타겟 유지');
    assert.eq(b.vx, vx0, '놓친 타겟 = 직진(속도 불변) vx');
    assert.eq(b.vy, vy0, '놓친 타겟 = 직진(속도 불변) vy');
  });

  test('distinct 타겟: evolved 다발은 서로 다른 적을, 비evolved 는 같은 최근접을 겨눈다', () => {
    // evolved: 2발이 서로 다른 적
    const we = mkWorld();
    const a = addEnemy(we, 700, 400);
    const b = addEnemy(we, 705, 402);
    const e = setup(we, 'seeker', 3, true);            // Lv3 count 2 · evolved distinct
    assert.eq(e.eff.count, 2, 'Lv3 count = 2');
    e.s.cooldownT = 0;
    seeker.update(we, e.s, e.eff, dt);
    const shots = liveBullets(we);
    assert.eq(shots.length, 2, '2발 발사');
    assert.ne(shots[0].target, shots[1].target, 'evolved → 서로 다른 타겟');
    assert.gte(Math.min(shots[0].target, shots[1].target), 0, '둘 다 유효 타겟');

    // 비evolved: 2발이 같은 최근접을 겨눈다
    const wn = mkWorld();
    addEnemy(wn, 700, 400); addEnemy(wn, 705, 402);
    const n = setup(wn, 'seeker', 3, false);
    n.s.cooldownT = 0;
    seeker.update(wn, n.s, n.eff, dt);
    const sn = liveBullets(wn);
    assert.eq(sn[0].target, sn[1].target, '비evolved → 동일 최근접');
  });

  test('음성: 계약 밖 targetMode 는 throw (§9.5 nearest 만)', () => {
    const w = mkWorld();
    const { s, eff } = setup(w, 'seeker', 1, false);
    eff.targetMode = 'lowestHp';
    assert.throws(() => seeker.update(w, s, eff, dt), '미구현 targetMode → throw');
  });
});

// ══════════════════════════════════════════════════════════════════════════
// 리터럴 계약 (§9.1) — check.mjs S1 스캐너를 재현해 소스에서 직접 검증
// ══════════════════════════════════════════════════════════════════════════
suite('weapons/리터럴 계약 §9.1', () => {
  const ALLOWED = new Set(['0', '1', '-1', '0.5', '2']);
  // check.mjs stripStringsAndComments 와 동일
  function strip(src) {
    return src
      .replace(/\/\*[\s\S]*?\*\//g, ' ')
      .replace(/(^|[^:])\/\/[^\n]*/g, '$1 ')
      .replace(/`(?:\\.|[^`\\])*`/g, '""')
      .replace(/'(?:\\.|[^'\\])*'/g, '""')
      .replace(/"(?:\\.|[^"\\])*"/g, '""');
  }
  function literals(file) {
    const code = strip(readFileSync(join(WEAPONS_DIR, file), 'utf8'));
    const out = [];
    for (const m of code.matchAll(/(?<![\w$.])-?\d+(?:\.\d+)?\b/g)) out.push(m[0]);
    return out;
  }

  for (const file of ['forward.js', 'fan.js', 'seeker.js']) {
    test(`${file}: 숫자 리터럴 ⊆ {0,1,-1,0.5,2}`, () => {
      const lits = literals(file);
      assert.gt(lits.length, 0, '스캐너가 실제로 리터럴을 봤다 (vacuous 아님)');
      const bad = lits.filter((l) => !ALLOWED.has(l));
      assert.deepEq(bad, [], `허용 밖 리터럴 없음 (발견: ${bad.join(', ')})`);
    });
  }
});
