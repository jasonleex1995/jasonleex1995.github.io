/**
 * src/core/enemies.js — 적 스포너 + 이동 스크립트 훅 (순수 core 모듈)
 *
 * 정본 v1.4 구현 절:
 *   §8.4   적 이동 어휘 moveId — 이 파일이 매 틱 vx/vy 를 세팅하고 step.moveBullets 가 등속 적분한다.
 *          ★ 1주차 슬라이스: dive(직하강) + weave(사인 하강) 2종을 구현. 나머지 6종은 dive 폴백.
 *   §8.6   element 는 아키타입 필드가 아니다 — **웨이브 편성이 주입**한다(상성의 핵심).
 *          hp = archetype.hp × band.hpMult × curve.enemyHpScale[stage].
 *   §8.7   편대·스폰·웨이브 스케줄. ★ 슬라이스: sea 스테이지의 stage-1 해금 웨이브만 순환 스폰한다.
 *          waveClearAdvance = 전멸(live 0)이면 즉시 다음 웨이브, 아니면 waveIntervalSec 간격.
 *   §9.9.2 formations 파라미터(scatter·arc·lineH·vWedge 구현, 그 외 scatter 폴백).
 *   §10.2  world.rng.spawn 만 사용 → 결정성(같은 시드 = 같은 스폰 시퀀스).
 *   §10.3  인덱스 오름차순 순회 · 스폰 상태(world.spawner)는 최초 1회만 alloc(핫패스 0 alloc).
 *   §9.1   core 순수성 — window/Date/Math.random/… 0. import 는 core 내부만.
 *
 * ★ 이 파일은 state.js 가 인쇄 안 된 합성 계약 자리(hooks.enemies)에 주입된다(§9.1). step.js line 59:
 *     if (world.hooks.enemies !== null) world.hooks.enemies(world, dt);
 *   화면 이탈 보상 몰수(§8.7)와 slow/stun 감속·등속 적분은 step.moveBullets 소관이다 — 여기서 안 한다.
 *
 * ★ 슬라이스 범위의 이유(sea stage-1 해금 웨이브만): 그 필터의 웨이브들은 아키타입이 drifter(dive)와
 *   spitter(weave) 둘뿐 = 구현한 2종 이동만 쓰고, element 가 water/grass/fire/normal 로 **섞여** 있으며
 *   (스탠스를 바꿀 이유 = 상성 손맛), eliteIndex 가 전부 null 이라 엘리트 재롤(§8.6)이 슬라이스 밖으로
 *   깔끔히 빠진다. 정본 데이터를 그대로 읽으므로 지어낸 값이 0이다.
 */

import { spawnEnemy } from './state.js';
import { TAU, DEG2RAD } from './angle.js';

/** 슬라이스 스테이지 = sea, 스테이지 번호 1 (curve/해금 인덱스 0). 정본 stages.json 에서 유도한다 */
const SLICE_STAGE_ID = 'sea';
const SLICE_STAGE_NUMBER = 1;

/**
 * 스폰 상태를 최초 1회만 만든다(§10.3 — 이후 핫패스는 0 alloc).
 * ★ 편성·아키타입 인덱스·간격을 전부 주입된 데이터에서 유도한다(하드코딩 매직넘버 0).
 */
function ensureSpawner(world) {
  if (world.spawner !== undefined) return world.spawner;

  const stagesFile = world.data.stages;
  let stage = null;
  const list = stagesFile.stages;
  for (let i = 0; i < list.length; i += 1) {
    if (list[i].id === SLICE_STAGE_ID) { stage = list[i]; break; }
  }
  if (stage === null) throw new Error(`enemies: 슬라이스 스테이지 "${SLICE_STAGE_ID}" 없음 (§9.9)`);

  // §8.7 — 스테이지 번호 ≥ unlockStageMin 인 웨이브만. stage-1 이면 unlockStageMin == 1 인 것만 산다.
  const waves = [];
  for (let i = 0; i < stage.waves.length; i += 1) {
    if (stage.waves[i].unlockStageMin <= SLICE_STAGE_NUMBER) waves.push(stage.waves[i]);
  }
  if (waves.length === 0) throw new Error(`enemies: "${SLICE_STAGE_ID}" 에 stage-1 해금 웨이브가 0개 (§8.7)`);

  // 아키타입 id → 정의. Map 순회 금지(§10.3)라 평범한 객체에 담아 **조회만** 한다.
  const archIndex = Object.create(null);
  const archetypes = world.data.enemies.archetypes;
  for (let i = 0; i < archetypes.length; i += 1) archIndex[archetypes[i].id] = archetypes[i];

  const s = {
    stageId: stage.id,
    waves,
    archIndex,
    waveIndex: 0,
    nextWaveT: 0,   // 0 = 첫 틱에 즉시 첫 웨이브
  };
  world.spawner = s;
  return s;
}

/** §8.6 — hp = archetype.hp × band.hpMult × enemyHpScale[stageIdx] (stage-1 → 인덱스 0). */
function enemyHp(world, def) {
  const band = world.data.enemies.bands[def.band];
  const scale = world.data.stages.curve.enemyHpScale[SLICE_STAGE_NUMBER - 1];
  return def.hp * band.hpMult * scale;
}

/** §8.4 — moveId 별 하강 속도의 거처. dive.speed | weave.speed | anchor.enterSpeed … 를 유도한다. */
function descentSpeed(mp) {
  if (typeof mp.speed === 'number') return mp.speed;
  if (typeof mp.enterSpeed === 'number') return mp.enterSpeed;  // anchor 계열 폴백(슬라이스 밖)
  return 0;
}

/**
 * §9.9.2 — 편대별 i번째 개체의 스폰 좌표. spawn RNG(scatter) + formation 파라미터로 유도한다.
 * spawnEdge 는 슬라이스에서 top 만 유효(sea stage-1 전량 top). base y = view.spawnLineY.
 */
function placement(world, wave, i, count, out) {
  const a = world.data.rules.view.arena;
  const lineY = world.data.rules.view.spawnLineY;
  const cx = a.x + a.w / 2;
  const forms = world.data.stages.formations;
  const rng = world.rng.spawn;

  let x = cx;
  let y = lineY;

  if (wave.formationId === 'arc') {
    const f = forms.arc;
    const t = count > 1 ? i / (count - 1) : 0.5;
    const ang = (-f.spanDeg / 2 + t * f.spanDeg) * DEG2RAD;
    x = cx + Math.sin(ang) * f.radiusPx;
    y = lineY + (1 - Math.cos(ang)) * f.radiusPx * 0.3;   // 가운데가 앞선 아래로 볼록한 호
  } else if (wave.formationId === 'lineH') {
    const f = forms.lineH;
    x = cx + (i - (count - 1) / 2) * f.gapPx;
    y = lineY;
  } else if (wave.formationId === 'vWedge') {
    const f = forms.vWedge;
    if (i === 0) { x = cx; y = lineY; }
    else {
      const rank = Math.ceil(i / 2);
      const side = (i % 2 === 1) ? -1 : 1;
      const ar = f.angleDeg * DEG2RAD;
      x = cx + side * rank * f.gapPx * Math.sin(ar);
      y = lineY - rank * f.gapPx * Math.cos(ar);          // 날개가 위로·바깥으로 = 아래로 향한 V
    }
  } else {
    // scatter + 폴백(columnV·pincer·미지) — rng.spawn 산포. jitterPx = y 계단, minSepPx = 가장자리 여백.
    const f = forms.scatter;
    x = a.x + f.minSepPx + rng.f() * (a.w - 2 * f.minSepPx);
    y = lineY - rng.f() * f.jitterPx;
  }

  // 아레나 바깥으로 새지 않게 구조적 클램프(밸런스 값 아님 — 좌표계 경계다).
  if (x < a.x) x = a.x;
  if (x > a.x + a.w) x = a.x + a.w;
  out.x = x;
  out.y = y;
  return out;
}

const _pos = { x: 0, y: 0 };   // 재사용(핫패스 0 alloc)

/** §8.7 — 한 웨이브를 편성대로 스폰한다. element 는 편성이 주입(§8.6). */
function spawnWave(world, s) {
  const wave = s.waves[s.waveIndex];
  const def = s.archIndex[wave.archetypeId];
  if (def === undefined) throw new Error(`enemies: 미지의 아키타입 "${wave.archetypeId}" (§9.7)`);
  const hp = enemyHp(world, def);
  const concurrentMax = world.data.rules.fairness.enemyConcurrentMax;

  for (let i = 0; i < wave.count; i += 1) {
    // §8.7 초과 정책 = defer. 동시 오써링 상한을 넘으면 나머지는 이번 웨이브에서 놓는다(풀 캡이 B층 안전망).
    if (world.enemies.live >= concurrentMax) break;
    placement(world, wave, i, wave.count, _pos);
    // §8.6 — eliteIndex: 그 웨이브의 n번째 개체에 접두 플래그. stage-1 해금 웨이브는 전부 null.
    const elite = wave.eliteIndex !== null && i === wave.eliteIndex;
    spawnEnemy(world, wave.archetypeId, wave.element, _pos.x, _pos.y, hp, elite);
  }

  s.waveIndex = (s.waveIndex + 1) % s.waves.length;   // §8.7 waveListExhausted = "cycle"
}

/**
 * §8.4 — 매 틱 alive 적의 vx/vy 를 moveId 로 갱신한다. step.moveBullets 가 등속 적분한다.
 *   dive  : vy = speed, vx = 0 (직하강)
 *   weave : vy = speed, vx = ampPx·ω·cos(ω·moveT), ω = 2π·freqHz (사인 좌우의 해석적 속도)
 *   그 외 : dive 폴백(슬라이스)
 * ★ e.moveT 는 step 이 적분 후 dt 만큼 올린다 → 여기서는 **현재 moveT** 로 속도를 산출한다.
 */
function applyMovement(world) {
  const items = world.enemies.items;
  const arch = world.spawner.archIndex;
  for (let i = 0; i < items.length; i += 1) {
    const e = items[i];
    if (!e.alive) continue;
    const def = arch[e.archetypeId];
    const mp = def.moveParams;
    const speed = descentSpeed(mp);
    if (def.moveId === 'weave' && typeof mp.ampPx === 'number' && typeof mp.freqHz === 'number') {
      const w = TAU * mp.freqHz;
      e.vy = speed;
      e.vx = mp.ampPx * w * Math.cos(w * e.moveT);
    } else {
      // dive + 폴백
      e.vy = speed;
      e.vx = 0;
    }
  }
}

/**
 * ★ 훅 진입점 — step.js 가 매 고정 틱 부른다(hooks.enemies 로 주입).
 *   (1) 스폰 케이던스 → (2) 갓 스폰한 개체 포함 전 개체의 이동 속도 갱신.
 *   화면 이탈 몰수(§8.7)·slow/stun 감속·좌표 적분은 step.moveBullets 소관이다.
 */
export function enemies(world, dt) {
  const s = ensureSpawner(world);

  const concurrentMax = world.data.rules.fairness.enemyConcurrentMax;
  const interval = world.data.stages.phase.waveIntervalSec;

  if (world.enemies.live < concurrentMax) {
    // §8.7 waveClearAdvance — 전멸(live 0)이면 즉시 다음, 아니면 waveIntervalSec 간격.
    if (world.enemies.live === 0 || world.time >= s.nextWaveT) {
      spawnWave(world, s);
      s.nextWaveT = world.time + interval;
    }
  }

  applyMovement(world);
}

export default { enemies };
