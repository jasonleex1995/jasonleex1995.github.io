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
 * ★ 슬라이스 범위: sea stage-1 해금 웨이브를 **케이던스·편대·element 의 골격**으로 쓰되, 아키타입은
 *   더 다양한 **로스터**에서 뽑는다(아래). 이유 — sea 의 stage-1 해금 웨이브는 아키타입이 drifter/spitter
 *   둘뿐이라 "느린 탱커 ↔ 빠른 약골"의 대비가 화면에 안 뜬다(사용자 피드백 #3). 정본 웨이브는 아키타입을
 *   확정하지만(콘텐츠), 슬라이스가 **더 다양한 stage-1 적을 뽑는 것**은 임무가 명시 허용한 범위다
 *   ("enemies.js가 더 다양한 stage-1 적을 뽑게, 단 이동/공격이 구현된 것 위주로"). 로스터는 **데이터에서
 *   유도**한다(하드코딩 id 0): 구현된 이동(dive·weave) × 플레이 가능한 밴드(chaff·line) × 테마 부합
 *   (themeOnly ∈ {null, sea}). element 는 여전히 웨이브 레코드가 주입하므로 §8.6 혼재가 유지된다.
 *   eliteIndex 는 웨이브 레코드에서 오고(전부 null → 엘리트 재롤(§8.6)은 슬라이스 밖).
 */

import { spawnEnemy } from './state.js';
import { TAU } from './angle.js';

const DEG2RAD = Math.PI / 180;
const ANCHOR_SWAY_HZ = 0.35;   // §8.4 「좌우 소폭 왕복」의 주기. 진폭은 swayAmpPx 가 소유한다
import { formationPos } from './formations.js';
import { PHASE } from './stage.js';

/** 슬라이스 스테이지 = sea, 스테이지 번호 1 (curve/해금 인덱스 0). 런 미구동(테스트) 시 폴백. */
const SLICE_STAGE_ID = 'sea';
const SLICE_STAGE_NUMBER = 1;

/** ★ 슬라이스가 구현한 이동(§8.4)·플레이 가능한 밴드(§8.6). 로스터 필터의 근거이며 하드코딩 id 가 아니다. */
const IMPLEMENTED_MOVES = ['dive', 'weave', 'column', 'strafe', 'anchor', 'orbitDrift', 'rearIn', 'bounce'];      // step.moveBullets + enemies.applyMovement 가 실제로 미는 2종
const PLAYABLE_BANDS = ['chaff', 'line', 'turret', 'bruiser'];          // turret/bruiser 는 effHP 가 슬라이스 무기엔 과하다(스폰지)

/**
 * 스폰 상태를 최초 1회만 만든다(§10.3 — 이후 핫패스는 0 alloc).
 * ★ 편성·아키타입 인덱스·간격을 전부 주입된 데이터에서 유도한다(하드코딩 매직넘버 0).
 */
/**
 * §8.10 — 위기 편성의 스케일링을 스테이지 진입 시 **1회 확정**한다(핫패스 0 alloc·계산).
 *   crisisTotal(60) × swarmTotalScale[pos] 을 레코드에 나눌 때 레코드마다 반올림하면 총량이 새고
 *   (0.5 배율에서 30 → 36), 소수 캐리는 부동소수 잔차로 랜서를 0 으로 잘라 9:1 편성을 깬다.
 *   → **아키타입별 누적 반올림**: Σcount == round(crisisTotal × scale) 이고 9:1 비율도 보존된다. 결정적.
 */
function crisisPlan(world, curveIdx) {
  const recs = world.data.stages.phase.crisisWaves;
  const scale = world.data.stages.curve.swarmTotalScale[curveIdx];
  const cum = Object.create(null);        // archetypeId → 정확 누적
  const done = Object.create(null);       // archetypeId → 확정한 정수 누적
  const plan = new Array(recs.length);
  for (let i = 0; i < recs.length; i += 1) {
    const r = recs[i];
    const a = r.archetypeId;
    if (cum[a] === undefined) { cum[a] = 0; done[a] = 0; }
    cum[a] += r.count * scale;
    const target = Math.round(cum[a]);
    plan[i] = target - done[a];
    done[a] = target;
  }
  return plan;
}

/** stageId·curveIdx 로 스폰 상태를 만든다(스테이지 진입/전환 시 1회). curveIdx = 런 포지션(0..5). */
function buildSpawner(world, stageId, curveIdx) {
  const list = world.data.stages.stages;
  let stage = null;
  for (let i = 0; i < list.length; i += 1) if (list[i].id === stageId) { stage = list[i]; break; }
  if (stage === null) throw new Error(`enemies: 스테이지 "${stageId}" 없음 (§9.9)`);

  // §8.7 — unlockStageMin ≤ 스테이지번호(= 런포지션+1) 인 웨이브만. 진행할수록 웨이브가 해금된다.
  const stageNumber = curveIdx + 1;
  const waves = [];
  for (let i = 0; i < stage.waves.length; i += 1) {
    if (stage.waves[i].unlockStageMin <= stageNumber) waves.push(stage.waves[i]);
  }
  if (waves.length === 0) throw new Error(`enemies: "${stageId}" 해금 웨이브 0개 (§8.7)`);

  // 아키타입 id → 정의. Map 순회 금지(§10.3)라 평범한 객체에 담아 **조회만** 한다.
  const archIndex = Object.create(null);
  const archetypes = world.data.enemies.archetypes;
  for (let i = 0; i < archetypes.length; i += 1) archIndex[archetypes[i].id] = archetypes[i];

  // ★ 위기(새떼) 전용 아키타입은 정상 웨이브 로스터에서 제외한다(데이터 유도, 하드코딩 id 없음) —
  //   crisisWaves 가 선언한 종이 평범한 웨이브에 섞여 나오면 §8.10 위기의 «장면»이 미리 새 버린다(실측).
  const crisisArch = Object.create(null);
  const cw = world.data.stages.phase.crisisWaves;
  for (let i = 0; i < cw.length; i += 1) crisisArch[cw[i].archetypeId] = true;

  // ★ 로스터 — **정본이 저작한 stages[].roster 를 쓴다** (§8.3 · §8.6 · §9.9).
  //   v1.7 까지 이 코드는 저작 로스터를 «읽지 않고» archetypes 전량에서 자체 로스터를 만들었다.
  //   그 결과 §8.3 이 명시한 「후반 = 아키타입 해금이 함께 올라 «다른 적»이 나온다」가 0 으로
  //   반영됐다 — 스테이지 1 과 6 의 등장 종 집합이 사실상 같았고, 스테이지 1 에 96HP 짜리
  //   turretPod 가 섰다(실측). unlockStageMin 저작 전체가 죽은 데이터였다.
  //   ★ 해금 기준은 «런의 스테이지 번호»(1..6) — 테마는 셔플되지만 해금은 진행도를 따른다.
  const roster = [];
  for (let i = 0; i < stage.roster.length; i += 1) {
    const ent = stage.roster[i];
    if (ent.unlockStageMin > stageNumber) continue;                // 아직 안 열린 적
    const a = archIndex[ent.archetypeId];
    if (a === undefined) throw new Error(`enemies: 로스터의 미지 아키타입 "${ent.archetypeId}" (§9.9)`);
    if (crisisArch[a.id]) continue;                                // 위기 전용 → 정상 로스터 제외
    if (IMPLEMENTED_MOVES.indexOf(a.moveId) < 0) continue;         // 이동 미구현 → 애초에 제외
    roster.push(a.id);
  }
  if (roster.length === 0) throw new Error(`enemies: "${stageId}" 스테이지 ${stageNumber} 로스터 0종 (§8.6)`);

  return {
    stageId, curveIdx, waves, archIndex, roster,
    waveIndex: 0, wavesSpawned: 0, nextWaveT: 0,
    element: stage.element,                    // §8.10 themePure 위기 속성
    crisisRule: stage.crisisElementRule,       // "themePure" | "finaleRotating"
    crisisSpawned: 0,                          // 이미 내보낸 위기 서브웨이브 수
    crisisPlan: crisisPlan(world, curveIdx),   // 레코드별 확정 스폰 수(총량·9:1 보존)
  };
}

/**
 * 스폰 상태 확보. 스테이지가 바뀌면(런 진행) 재빌드한다. themeDraw 는 비복원이라 stageId 로 유일 식별.
 * ★ 핫패스 0-alloc(§10.3) — 스칼라를 인라인한다(래퍼 객체를 만들면 MOB 매 틱 리터럴이 새로 alloc 된다).
 */
function ensureSpawner(world) {
  let stageId; let curveIdx;
  if (world.run !== undefined && world.run.order !== undefined) {
    stageId = world.run.order[world.run.stageIndex];
    curveIdx = world.run.stageIndex;
  } else {
    stageId = SLICE_STAGE_ID;
    curveIdx = SLICE_STAGE_NUMBER - 1;
  }
  if (world.spawner !== undefined && world.spawner.stageId === stageId) return world.spawner;
  world.spawner = buildSpawner(world, stageId, curveIdx);
  return world.spawner;
}

/** §8.6 — hp = archetype.hp × band.hpMult × enemyHpScale[런포지션]. */
function enemyHp(world, def) {
  const band = world.data.enemies.bands[def.band];
  const scale = world.data.stages.curve.enemyHpScale[world.spawner.curveIdx];
  return def.hp * band.hpMult * scale;
}

/** §8.4 — moveId 별 하강 속도의 거처. dive.speed | weave.speed | anchor.enterSpeed … 를 유도한다. */
function descentSpeed(mp) {
  if (typeof mp.speed === 'number') return mp.speed;
  if (typeof mp.enterSpeed === 'number') return mp.enterSpeed;  // anchor 계열 폴백(슬라이스 밖)
  return 0;
}

/**
 * §9.9.2 — 편대별 i번째 개체의 스폰 좌표. 원점 = 스폰 라인 중앙.
 * spawnEdge 는 슬라이스에서 top 만 유효(sea stage-1 전량 top). base y = view.spawnLineY.
 */
function placement(world, wave, i, count, out, def) {
  // 편대의 원점 = 스폰 라인 중앙(웨이브). 모양 자체는 formations.js 가 소유한다(§9.9.2).
  const a = world.data.rules.view.arena;
  const mv = def === undefined ? '' : def.moveId;
  const mp = def === undefined ? null : def.moveParams;
  // §8.4(v1.7) — 이동 동사가 «어디서 들어오는가»를 정하는 두 경우. 이걸 안 읽어서 세 아키타입이
  //   스폰만 되고 아레나에 한 번도 서지 못했다(실측 도달률 flanker·thornWeaver·rearDart 전부 0.0%):
  //   · strafe 는 「좌/우 벽 진입 → 수평 횡단」인데 상단 스폰라인에서 vy=0 이라 화면 위에 머물렀다.
  //   · rearIn 은 「하단 밖에서 상승 진입」인데 상단에서 vy=-speed(위로)라 더 멀어졌다.
  //     (rearIn 구현 주석이 「스폰 y 는 스폰 측 책임」이라 적어 놓고 그 책임자가 없었다)
  //   값은 이미 저작돼 있었다 — moveParams.yPx(flanker 180 · thornWeaver 140) · warnSec(0.8).
  if (mv === 'strafe' && mp !== null && typeof mp.yPx === 'number') {
    // 좌우 벽 «밖»에서 시작한다. 어느 쪽인지는 편대 인덱스로 갈라 rng 를 쓰지 않는다(§10.2 결정성).
    const pad = world.data.rules.view.spawnPadPx;
    const left = (i % 2) === 0;
    out.x = left ? a.x - pad : a.x + a.w + pad;
    out.y = a.y + mp.yPx;
    return out;
  }
  if (mv === 'rearIn') {
    const pad = world.data.rules.view.spawnPadPx;
    out.x = a.x + a.w * ((i + 0.5) / Math.max(1, count));   // 하단을 균등 분할 — 편대 모양 대신 폭을 쓴다
    out.y = a.y + a.h + pad;
    return out;
  }
  return formationPos(world, wave.formationId, i, count,
    a.x + a.w / 2, world.data.rules.view.spawnLineY, out);
}

const _pos = { x: 0, y: 0 };   // 재사용(핫패스 0 alloc)

/** §8.7 — 한 웨이브를 편성대로 스폰한다. element 는 편성이 주입(§8.6). */
function spawnWave(world, s) {
  const wave = s.waves[s.waveIndex];
  // ★ 아키타입은 로스터 라운드로빈으로 다양화한다(웨이브 골격 = 편대·element·count·eliteIndex 는 그대로).
  //   wave 0 → roster[0](= drifter, 첫 필터 통과 아키타입)이라 element 테스트의 전제와 정합한다.
  const archetypeId = s.roster[s.waveIndex % s.roster.length];
  const def = s.archIndex[archetypeId];
  if (def === undefined) throw new Error(`enemies: 미지의 아키타입 "${archetypeId}" (§9.7)`);
  const hp = enemyHp(world, def);
  const concurrentMax = world.data.rules.fairness.enemyConcurrentMax;

  // ★ count 는 밴드로 나눠 구조적으로 클램프한다(밸런스 매직넘버 아님 — effHP ∝ hpMult 이므로 탱커 웨이브가
  //   벽이 되지 않게 총 HP 예산을 대략 보존한다). chaff(hpMult 1.0)는 원본 count 유지, line(2.5)은 줄어든다.
  const band = world.data.enemies.bands[def.band];
  // §8.6 — 스테이지별 스폰 밀도(curve.spawnDensityScale). ★ 실측으로 발견된 누락: 이것이 없으면
  //   스테이지 1 이 저작 의도(0.7배)보다 30% 더 몰려오고 후반은 반대로 헐거워진다.
  const density = world.data.stages.curve.spawnDensityScale[s.curveIdx];
  const count = Math.max(2, Math.round((wave.count / band.hpMult) * density));

  for (let i = 0; i < count; i += 1) {
    // §8.7 초과 정책 = defer. 동시 오써링 상한을 넘으면 나머지는 이번 웨이브에서 놓는다(풀 캡이 B층 안전망).
    if (world.enemies.live >= concurrentMax) break;
    placement(world, wave, i, count, _pos, def);
    // §8.6 — 엘리트 = 두 경로의 OR:
    //   (1) eliteIndex: 그 웨이브의 n번째 개체에 접두 플래그(베이크된 스포트라이트, perWaveMax 1).
    //   (2) 엘리트 재롤(§8.6, 이제 구현 — 예약된 rng.elite 스트림): 자격 개체(밴드∈bandAllowed ∧
    //       속성∈elementAllowed)가 런 포지션 곡선 확률 elitePerWaveChance[curveIdx] 로 엘리트가 된다.
    //       곡선은 포지션(초반 0 → 최종 1.0)으로 오른다 → 테마가 셔플돼도 «초반 헐거움·후반 전면 엘리트».
    //       ★ 단락평가로 자격 개체만 rng.elite 를 뽑는다(결정성: 같은 시드 = 같은 엘리트열, §10.2).
    const el = world.data.rules.elite;
    const eligible = el.bandAllowed.indexOf(def.band) >= 0 && el.elementAllowed.indexOf(wave.element) >= 0;
    const chance = world.data.stages.curve.elitePerWaveChance[s.curveIdx];
    // §8.6(v1.7) — ★ 곡선이 «유일한 권위»다. v1.6 까지 베이크된 eliteIndex 는 곡선을 통째로
    //   무시했다: 스테이지 1 은 곡선이 0.0 인데도 웨이브의 34% 가 엘리트를 낳았고, 엘리트는
    //   hpMult 4.0 이라 Lv1 무기로 10~20초짜리 벽이었다(플레이 피드백).
    //   이제 베이크된 스포트라이트도 «그 스테이지가 엘리트를 허용할 때만» 선다 — 초반엔 같은
    //   자리에 평범한 몹이 서고, 스테이지가 갈수록 그 자리가 엘리트가 된다.
    const stageAllows = chance > 0;
    const rerollElite = eligible && stageAllows && world.rng.elite.f() < chance;
    const bakedElite = stageAllows && eligible && wave.eliteIndex !== null && i === wave.eliteIndex;
    // §8.6 perWaveMax — 선언만 되어 있고 아무도 강제하지 않던 값이다(재롤이 웨이브당 여러 마리를
    //   만들 수 있었다). 이제 실제로 상한이다.
    // §8.6 perWaveMax(=1) 는 «베이크된 스포트라이트» 쪽 서술이다 — eliteIndex 가 단일 인덱스라
    //   구조적으로 웨이브당 1기다. 재롤에까지 상한을 걸면 안 된다: 정본 v1.5 가 「최종 1.0 = 자격
    //   전원」으로 «후반 전면 엘리트화»를 확정했으므로, 상한을 걸면 그 설계가 통째로 죽는다
    //   (실측: 걸었더니 스테이지 6 엘리트율이 1.2% 로 주저앉았다).
    const elite = bakedElite || rerollElite;
    spawnEnemy(world, archetypeId, wave.element, _pos.x, _pos.y, hp, elite);
  }

  s.waveIndex = (s.waveIndex + 1) % s.waves.length;   // §8.7 waveListExhausted = "cycle"
  s.wavesSpawned += 1;                                 // 런 구동 시 mobPhaseMaxWaves 상한의 근거
}

/**
 * §8.10 — 위기 서브웨이브의 속성. themePure = 60기 전부 테마 속성(정답 스탠스의 페이오프),
 *   finaleRotating(최종 전용) = 서브웨이브 1·2 물 → 3·4 불 → 5·6 풀 (§8.16 · §7.12.3).
 */
function crisisElement(s, subWave) {
  if (s.crisisRule === 'themePure') return s.element;
  if (s.crisisRule !== 'finaleRotating') throw new Error(`enemies: 미지의 crisisElementRule "${s.crisisRule}" (§8.10)`);
  if (subWave <= 2) return 'water';
  if (subWave <= 4) return 'fire';
  return 'grass';
}

/** §8.10 — 한 위기 서브웨이브(9 swarmChaff + 1 swarmLancer)를 편성대로 내보낸다. */
function spawnCrisisSubWave(world, s, subWave) {
  const recs = world.data.stages.phase.crisisWaves;
  const swarmMax = world.data.rules.fairness.swarmConcurrentMax;
  const el = crisisElement(s, subWave);

  for (let i = 0; i < recs.length; i += 1) {
    const r = recs[i];
    if (r.subWave !== subWave) continue;
    const def = s.archIndex[r.archetypeId];
    if (def === undefined) throw new Error(`enemies: 미지의 새떼 아키타입 "${r.archetypeId}" (§8.10)`);
    const hp = enemyHp(world, def);
    const count = s.crisisPlan[i];                  // 스테이지 진입 시 확정(총량·9:1 보존)
    for (let k = 0; k < count; k += 1) {
      if (world.enemies.live >= swarmMax) break;      // §12.4 swarmConcurrentMax (새떼 전용 상한)
      placement(world, r, k, count, _pos);            // 레코드가 formationId 를 들고 있다(arc/vWedge)
      spawnEnemy(world, r.archetypeId, el, _pos.x, _pos.y, hp, false);
    }
  }
}

/**
 * §8.10 — 위기 세션: 잡몹 페이즈 마지막 crisisDurationSec 동안 crisisSubWaves 파를 균등 간격으로.
 *   정상 웨이브는 멈춘다(crisisSuspendsWaves). 누적 카운트라 큰 dt 도 놓치지 않는다(결정적 캐치업).
 */
function spawnCrisis(world, s) {
  const ph = world.data.stages.phase;
  const elapsed = world.run.phaseT - ph.crisisStartSec;
  const interval = ph.crisisDurationSec / ph.crisisSubWaves;
  let want = Math.floor(elapsed / interval) + 1;
  if (want > ph.crisisSubWaves) want = ph.crisisSubWaves;
  while (s.crisisSpawned < want) {
    s.crisisSpawned += 1;
    spawnCrisisSubWave(world, s, s.crisisSpawned);     // subWave 는 1-based
  }
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
  const arena = world.data.rules.view.arena;
  for (let i = 0; i < items.length; i += 1) {
    const e = items[i];
    if (!e.alive) continue;
    if (e.isBoss) continue;              // 보스 개체는 archIndex 에 없다 — 이동은 boss.js 소관(§8.12.1)
    if (e.midBossId !== '') continue;    // 중간보스도 마찬가지 — 이동은 midboss.js 소관(§8.9)
    const def = arch[e.archetypeId];
    const mp = def.moveParams;
    const speed = descentSpeed(mp);
    const mv = def.moveId;

    if (mv === 'bounce') {
      // §8.4(v1.7) 벽 반사 — 대각으로 들어와 좌우 벽을 되튀며 내려온다.
      //   플레이어의 도탄 무기(리턴)와 «같은 규칙»을 적이 쓴다. 벽이 내 편만은 아니라는 것을 가르친다.
      //   ★ 진입 방향은 strafe 와 같은 규약으로 정한다 — 스폰 x 가 중앙보다 왼쪽인가. rng 금지(§10.2).
      //   ★ 반사는 여기서 «속도 부호»만 뒤집는다. 위치 되접기는 step 의 적 이동이 클램프로 처리한다.
      if (e.mp0 === 0) e.mp0 = e.x < (arena.x + arena.w * 0.5) ? 1 : -1;
      const hx = typeof mp.hSpeed === 'number' ? mp.hSpeed : speed;
      if (e.x <= arena.x + e.radius && e.mp0 < 0) e.mp0 = 1;
      else if (e.x >= arena.x + arena.w - e.radius && e.mp0 > 0) e.mp0 = -1;
      e.vx = hx * e.mp0;
      e.vy = speed;

    } else if (mv === 'weave' && typeof mp.ampPx === 'number' && typeof mp.freqHz === 'number') {
      const w = TAU * mp.freqHz;
      e.vy = speed;
      e.vx = mp.ampPx * w * Math.cos(w * e.moveT);

    } else if (mv === 'strafe') {
      // §8.4 — 좌/우 벽 진입 → 수평 횡단 → 반대편 이탈. yPx 고정.
      //   ★ 진입 방향은 «스폰 x 가 아레나 중앙보다 왼쪽인가»로 정한다 — rng 를 쓰지 않는다(§10.2 결정성).
      if (e.mp0 === 0) e.mp0 = e.x < (arena.x + arena.w * 0.5) ? 1 : -1;
      e.vx = speed * e.mp0;
      e.vy = 0;

    } else if (mv === 'anchor') {
      // §8.4 — 상단 진입 → yHoldPx 정지 → 좌우 소폭 왕복 → leaveAfterSec 후 하단 이탈.
      const hold = typeof mp.yHoldPx === 'number' ? mp.yHoldPx : 0;
      const sway = typeof mp.swayAmpPx === 'number' ? mp.swayAmpPx : 0;
      const leave = typeof mp.leaveAfterSec === 'number' ? mp.leaveAfterSec : 0;
      if (e.mp0 === 0 && e.y < hold) {
        e.vy = speed; e.vx = 0;                        // ① 진입
      } else {
        if (e.mp0 === 0) { e.mp0 = 1; e.mp1 = e.moveT; }   // 정지 시각을 잠근다
        const held = e.moveT - e.mp1;
        if (held >= leave) {
          e.vy = speed; e.vx = 0;                      // ③ 이탈
        } else {
          const w = TAU * ANCHOR_SWAY_HZ;              // ② 체류 — 속도로 준다(적분해도 진폭을 안 넘는다)
          e.vy = 0;
          e.vx = sway * w * Math.cos(w * held);
        }
      }

    } else if (mv === 'orbitDrift') {
      // §8.4 — 플레이어 쪽으로 호를 그리며 접근 → keepDistPx 유지.
      const keep = typeof mp.keepDistPx === 'number' ? mp.keepDistPx : 0;
      const turn = typeof mp.turnRateDegSec === 'number' ? mp.turnRateDegSec : 0;
      let dx = world.player.x - e.x;
      let dy = world.player.y - e.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > 0.0001) { dx /= d; dy /= d; } else { dx = 0; dy = 1; }
      const radial = d > keep ? 1 : -1;                // 멀면 접근, 가까우면 후퇴
      const tan = turn * DEG2RAD * keep;               // 각속도 × 반경 = 접선 속도
      let vx = dx * radial * speed - dy * tan;
      let vy = dy * radial * speed + dx * tan;
      const vm = Math.sqrt(vx * vx + vy * vy);
      if (vm > speed) { vx = vx / vm * speed; vy = vy / vm * speed; }
      e.vx = vx;
      e.vy = vy;

    } else if (mv === 'rearIn') {
      // §8.4 — 하단 밖에서 «상승» 진입. 스폰 y 와 warnSec 진입 표식은 스폰 측 책임이다.
      e.vy = -speed;
      e.vx = 0;

    } else {
      // dive · column + 폴백 — 직하강. column 의 «일렬 종대»는 이동이 아니라 스폰 편성(gapSec)이 만든다.
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
  const runMode = world.run !== undefined && world.run.order !== undefined;
  // §6.5 — 런 구동이면 스폰은 MOB 페이즈에만. BOSS_INTRO/BOSS/STAGE_CLEAR 엔 잔존 개체 이동만.
  //   ★ 스포너가 아직 없으면(그 스테이지에 잡몹을 한 번도 안 스폰) 움직일 잡몹도 없다 → 이동 생략.
  if (runMode && world.run.phase !== PHASE.MOB) {
    if (world.spawner !== undefined) applyMovement(world);
    return;
  }

  const s = ensureSpawner(world);

  // §8.10 — 위기 세션 구간: 정상 웨이브를 멈추고(crisisSuspendsWaves) 새떼 서브웨이브만 내보낸다.
  if (runMode && world.run.crisis) {
    spawnCrisis(world, s);
    applyMovement(world);
    return;
  }

  const concurrentMax = world.data.rules.fairness.enemyConcurrentMax;
  const interval = world.data.stages.phase.waveIntervalSec;
  // §6.3 — 런 구동은 mobPhaseMaxWaves 상한. 슬라이스(테스트)는 무한 순환(상한 없음).
  const wavesLeft = !runMode || s.wavesSpawned < world.data.stages.phase.mobPhaseMaxWaves;

  if (world.enemies.live < concurrentMax && wavesLeft) {
    // §8.7 waveClearAdvance — 전멸(live 0)이면 즉시 다음, 아니면 waveIntervalSec 간격.
    if (world.enemies.live === 0 || world.time >= s.nextWaveT) {
      spawnWave(world, s);
      s.nextWaveT = world.time + interval;
    }
  }

  applyMovement(world);
}

export default { enemies };
