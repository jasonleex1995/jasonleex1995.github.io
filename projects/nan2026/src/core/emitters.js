/**
 * src/core/emitters.js — 적 공격(이미터) 발사 훅 (순수 core 모듈)
 *
 * 정본 v1.4 구현 절:
 *   §8.5   이미터 어휘 8종(straight·fan·aimed·ring·spiral·laser·zone·wall). 유도·상태이상은
 *          **탄 속성**이지 이미터가 아니다 → 이 파일은 발사 각/속도/개수만 만들고, 유도(turnRate)·
 *          slow/stun 은 spawnEnemyBullet 이 탄 데이터에서 실어 step 이 처리한다.
 *   §8.5   공통 파라미터 telegraphSec·everySec·offsetSec·repeat·restSec 로 케이던스가 유도된다.
 *          잡몹(attack 참조 이미터)은 S30 이 repeat==1 ∧ restSec==0 를 강제 → **메트로놈**.
 *          일반 악절(repeat≥2·restSec>0)도 같은 식으로 환원되게 구현한다(보스는 슬라이스 밖).
 *   §7.4 · §12.4  공정성(느린 속도·관대한 판정·텔레그래프 리드)은 **이미터 데이터가 이미 보장**한다
 *          (speed ≤ fairness.maxBulletSpeed 등은 check.mjs S6 가 로드에서 강제). 여기서 지어내지 않고
 *          데이터대로 발사한다. ★ 텔레그래프 리드: 각 발사는 그 볼리의 telegraphSec **뒤에** 일어난다
 *          (t_fire = firstDelaySec + telegraphSec + offsetSec + k·everySec). 텔레그래프의 **그리기**는
 *          렌더 소관(슬라이스 밖)이지만, 타이밍 리드를 여기서 보존하므로 렌더가 붙어도 core 무변경.
 *   §9.9.2 aimed 는 leadSec 만큼 플레이어 속도를 앞질러 조준한다. §9.6 afterimage(ghostSec)면 조준 제외.
 *   §10.2  결정성 — 발사 각/개수/타이밍이 전부 e.x/e.y/player 상태와 데이터의 **결정 함수**다.
 *          난수를 쓰지 않으므로(쓴다면 world.rng.pattern 만) 같은 시드 = 같은 탄 시퀀스.
 *   §10.3  인덱스 오름차순 순회 · 조회 인덱스(emitById/archById)는 최초 1회만 alloc(핫패스 0 alloc).
 *   §9.1   core 순수성 — window/Date/Math.random/… 0. import 는 core 내부(state.js·angle.js)만.
 *
 * ★ 합성 계약: state.js 가 hooks.emitters 자리를 인쇄 안 된 채로 뒀다(§9.1). step.js line 60:
 *     if (world.hooks.emitters !== null) world.hooks.emitters(world, dt);
 *   이 훅은 **발사만** 한다. 탄 이동·플레이어 충돌·i-frame·상태이상 적용은 step.moveBullets/collide 소관.
 *
 * ★ 스케줄 상태는 개체 필드 e.emitT(이미터 나이) · e.emitPhase(이미 쏜 볼리 수)에 산다.
 *   makeEnemy()가 자리를 예약했고 spawnEnemy()가 스폰마다 0 으로 리셋한다(슬롯 재사용 안전).
 */

import { spawnEnemyBullet } from './state.js';
import { DEG2RAD, TAU } from './angle.js';

/**
 * 조회 인덱스를 최초 1회만 만든다(§10.3 — 이후 핫패스 0 alloc). Map 순회 금지라 평범한 객체에 담아
 * **조회만** 한다. emitById: 이미터 id → 정의 / archById: 아키타입 id → 정의(attack 을 읽는다).
 */
function ensureLookup(world) {
  if (world.emitLookup !== undefined) return world.emitLookup;
  const emitById = Object.create(null);
  const ems = world.data.enemies.emitters;
  for (let i = 0; i < ems.length; i += 1) emitById[ems[i].id] = ems[i];
  const archById = Object.create(null);
  const arr = world.data.enemies.archetypes;
  for (let i = 0; i < arr.length; i += 1) archById[arr[i].id] = arr[i];
  world.emitLookup = { emitById, archById };
  return world.emitLookup;
}

/**
 * §8.5 — 이미터 나이 age(초) 시점까지 **쏘였어야 할 볼리 총수**.
 *   firstFire = firstDelaySec + telegraphSec + offsetSec  (첫 볼리 = 스폰 유예 + 텔레그래프 리드)
 *   악절 주기 period = repeat·everySec + restSec           (잡몹 메트로놈이면 = everySec)
 *   한 주기 안 볼리 j 는 j·everySec 에 발사, repeat 발 뒤 restSec 쉰다.
 * ★ 누적 카운트로 돌려주므로 큰 dt 도 놓치지 않는다(결정적 캐치업). age < firstFire → 0.
 */
function scheduledVolleys(age, em, firstDelaySec) {
  const firstFire = firstDelaySec + em.telegraphSec + em.offsetSec;
  if (age < firstFire) return 0;
  const period = em.repeat * em.everySec + em.restSec;
  const elapsed = age - firstFire;
  const cyc = Math.floor(elapsed / period);
  const within = elapsed - cyc * period;
  let inCycle = Math.floor(within / em.everySec) + 1;
  if (inCycle > em.repeat) inCycle = em.repeat;   // restSec 구간(볼리 없음)에 걸린 잔여
  return cyc * em.repeat + inCycle;
}

/**
 * §8.5 straight/fan/aimed 의 공용 발사기 — baseAngle 기준으로 count 발을 총폭 spreadDeg 로 편다.
 *   각은 **아래(+y)** 축에서 잰다: vx = sin(a)·speed, vy = cos(a)·speed → a=0 이면 직하강.
 *   count==1 이면 편차 0(정면). fan 은 baseAngle=0(아래) + spreadDeg=arcDeg 로 이 함수를 쓴다.
 */
function fireSpread(world, e, em, count, spreadDeg, baseAngle) {
  for (let i = 0; i < count; i += 1) {
    const off = count > 1 ? (i / (count - 1) - 0.5) * spreadDeg : 0;
    const a = baseAngle + off * DEG2RAD;
    spawnEnemyBullet(world, em.bulletId, e.x, e.y, Math.sin(a) * em.speed, Math.cos(a) * em.speed);
  }
}

/**
 * §8.5 aimed — 플레이어를 leadSec 만큼 앞질러 조준한 뒤 spreadDeg 로 편다.
 * §9.6 afterimage — ghostSec>0(피격 직후)면 조준 대상에서 제외 → 직하강으로 폴백(지어낸 규칙 아님).
 */
function fireAimed(world, e, em, p) {
  let dx;
  let dy;
  if (p.ghostSec > 0) {
    dx = 0; dy = 1;                                        // 조준 제외 = 아래로
  } else {
    dx = (p.x + p.vx * em.leadSec) - e.x;
    dy = (p.y + p.vy * em.leadSec) - e.y;
  }
  fireSpread(world, e, em, em.count, em.spreadDeg, Math.atan2(dx, dy));   // atan2(dx,dy): +y 축 기준각
}

/** §8.5 ring — count 발을 360° 등분, rotOffsetDeg 만큼 회전. base(아래)는 전원(全圓)이라 무의미. */
function fireRing(world, e, em) {
  for (let i = 0; i < em.count; i += 1) {
    const a = em.rotOffsetDeg * DEG2RAD + (i / em.count) * TAU;
    spawnEnemyBullet(world, em.bulletId, e.x, e.y, Math.sin(a) * em.speed, Math.cos(a) * em.speed);
  }
}

/**
 * §8.5 spiral — 원래는 durationSec 동안 rateSec 간격으로 rotStepDeg 씩 돌며 뿌리는 시간 확장 패턴이다.
 * ★ 슬라이스 근사: 한 볼리를 rotStepDeg 간격의 count 발로 놓고, 볼리마다 rotStepDeg 만큼 시작각을
 *   돌린다(회전감 보존). 시간 확장(durationSec/rateSec)은 모델링하지 않는다 — 슬라이스 로스터가
 *   spiral 을 쓰지 않으므로 미실행 경로다(보스/시그니처에서 붙을 때 확장).
 */
function fireSpiral(world, e, em, volleyIdx) {
  const spin = volleyIdx * em.rotStepDeg * DEG2RAD;
  for (let i = 0; i < em.count; i += 1) {
    const a = spin + i * em.rotStepDeg * DEG2RAD;
    spawnEnemyBullet(world, em.bulletId, e.x, e.y, Math.sin(a) * em.speed, Math.cos(a) * em.speed);
  }
}

/**
 * §8.5 wall — count 개 벽돌을 아레나 폭에 가로로 펼쳐 직하강, 가운데에 gapCount 칸(폭 gapWidthPx)을 비운다.
 * ★ 슬라이스 근사(미실행 경로): 로스터가 wall 을 쓰지 않는다. 좁은 틈으로 지나가는 판정만 재현한다.
 */
function fireWall(world, e, em) {
  const a = world.data.rules.view.arena;
  const total = em.count + em.gapCount;
  const gapStart = Math.floor((total - em.gapCount) / 2);   // 틈을 가운데로
  const step = a.w / (total + 1);
  let brick = 0;
  for (let slot = 0; slot < total; slot += 1) {
    if (slot >= gapStart && slot < gapStart + em.gapCount) continue;   // 틈
    const x = a.x + step * (slot + 1);
    spawnEnemyBullet(world, em.bulletId, x, e.y, 0, em.speed);
    brick += 1;
    if (brick >= em.count) break;
  }
}

/** 한 볼리를 타입대로 발사한다. laser·zone 은 탄이 아니라 빔/장판이라 spawnEnemyBullet 대상이 아니다. */
function fireVolley(world, e, em, volleyIdx, p) {
  const t = em.type;
  if (t === 'straight') fireSpread(world, e, em, em.count, em.spreadDeg, 0);
  else if (t === 'fan') fireSpread(world, e, em, em.count, em.arcDeg, 0);
  else if (t === 'aimed') fireAimed(world, e, em, p);
  else if (t === 'ring') fireRing(world, e, em);
  else if (t === 'spiral') fireSpiral(world, e, em, volleyIdx);
  else if (t === 'wall') fireWall(world, e, em);
  // laser·zone: 빔/장판 엔티티가 필요하고 step 이 적 빔/장판을 아직 처리하지 않는다(슬라이스 밖).
  //   슬라이스 로스터의 어떤 아키타입도 이 둘을 참조하지 않는다(도달 불가). 붙일 때 여기에 추가.
}

/**
 * ★ 훅 진입점 — step.js 가 매 고정 틱 부른다(hooks.emitters 로 주입).
 *   각 alive 적의 attack(archetype)을 읽어 firstDelaySec·telegraphSec 리드 뒤 emitter 케이던스대로
 *   탄을 쏜다. 이동·충돌·i-frame·상태이상은 step 소관 — 이 훅은 **발사만** 한다.
 */
export function emitters(world, dt) {
  const look = ensureLookup(world);
  const items = world.enemies.items;
  const p = world.player;
  for (let i = 0; i < items.length; i += 1) {
    const e = items[i];
    if (!e.alive) continue;
    if (e.stunSec > 0) continue;                    // 스턴 = 개체 정지(step.moveBullets 와 대칭). 나이도 얼린다
    const def = look.archById[e.archetypeId];
    if (def === undefined || def.attack === null) continue;   // 사격 안 함 = attack: null (§8.5)
    const em = look.emitById[def.attack.emitterId];
    if (em === undefined) {
      throw new Error(`emitters: 미지의 이미터 "${def.attack.emitterId}" (${e.archetypeId}, §9.7 — 폴백 금지)`);
    }
    e.emitT += dt;
    const want = scheduledVolleys(e.emitT, em, def.attack.firstDelaySec);
    while (e.emitPhase < want) {                     // 결정적 캐치업(보통 0~1회)
      fireVolley(world, e, em, e.emitPhase, p);
      e.emitPhase += 1;
    }
  }
}

export default { emitters };
