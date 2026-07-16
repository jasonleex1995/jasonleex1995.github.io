#!/usr/bin/env node
/**
 * ============================================================================
 *  NAN 2026 — check.mjs   (정본 §13.4 정적 게이트 S1~S26 + §9.3 로더 규칙)
 * ============================================================================
 *
 *  사용법
 *  ------
 *    node tools/check.mjs                 # 전 검사 실행. 위반 있으면 exit 1
 *    node tools/check.mjs --allow-ambiguous
 *                                         # __AMBIGUOUS__ 만 남았으면 exit 0
 *                                         # (전사 진행 중 다른 검사를 보기 위한 용도)
 *    node tools/check.mjs --quiet          # 요약만 출력
 *
 *  필요 버전
 *  ---------
 *    Node.js >= 16.0.0  (ESM + node: 프로토콜 import). 의존성 0, 빌드 스텝 0.
 *    이 파일은 tools/ 에 산다 = 개발 전용. 배포물에 포함되나 실행되지 않는다(§9.1).
 *
 *  종료 코드
 *  ---------
 *    0 = 통과   1 = 위반/정본결함/모호 존재   2 = 실행 오류(파일 없음·JSON 파싱 실패)
 *
 *  출력 카테고리 (4종)
 *  -------------------
 *    [VIOLATION]     데이터가 정본을 위반. 고칠 곳 = data/*.json
 *    [CANON]         정본 자신의 결함. 검사를 문면대로 돌리면 정본이 확정한 콘텐츠가
 *                    실패하거나, 검사가 읽을 값이 존재하지 않는다. 고칠 곳 = CANON.md
 *    [AMBIGUOUS]     data 안의 "__AMBIGUOUS__" = 정본이 아직 답하지 않은 자리
 *    [STUB]          시뮬이 필요해 정적으로 검사 불가. 인터페이스만 정본대로 선언
 *
 *  ★ 이 파일은 값을 발명하지 않는다 (C-6). 정본이 값을 주지 않은 자리는 검사하지
 *    않고 [CANON] 또는 [STUB] 로 신고한다.
 * ============================================================================
 */

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, resolve, relative, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
const DATA_DIR = join(ROOT, 'data');
const SRC_DIR = join(ROOT, 'src');

const AMB = '__AMBIGUOUS__';
const ARGV = process.argv.slice(2);
const ALLOW_AMBIGUOUS = ARGV.includes('--allow-ambiguous');
const QUIET = ARGV.includes('--quiet');

// ---------------------------------------------------------------------------
// 리포트 수집기
// ---------------------------------------------------------------------------
const report = { violation: [], canon: [], ambiguous: [], stub: [], skip: [] };

const V = (check, msg) => report.violation.push({ check, msg });
const C = (check, msg) => report.canon.push({ check, msg });
const A = (path, note) => report.ambiguous.push({ path, note });
const S = (check, msg) => report.stub.push({ check, msg });
const SKIP = (check, msg) => report.skip.push({ check, msg });

// ---------------------------------------------------------------------------
// 유틸
// ---------------------------------------------------------------------------
const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isAmb = (v) => v === AMB;
const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const num = (v) => typeof v === 'number' && Number.isFinite(v);

/** 상대 오차 비교 (±pct %) */
function withinPct(actual, target, pct) {
  if (target === 0) return actual === 0;
  return Math.abs(actual - target) / Math.abs(target) <= pct / 100;
}

/**
 * §9.3 로더 규칙: 미지 키 = 에러 / 누락 키 = 에러 / 기본값 폴백 금지.
 * allowed 와 required 가 같은 집합인 것이 정본의 기본값이다
 * (예외는 §9.3이 명시한 weapons/passives 의 levels[] 부분 오버라이드뿐).
 */
function closedKeys(check, obj, allowed, path, opts = {}) {
  const optional = new Set(opts.optional || []);
  if (!isObj(obj)) {
    V(check, `${path}: 객체가 아니다 (실제 타입 ${Array.isArray(obj) ? 'array' : typeof obj})`);
    return false;
  }
  const allow = new Set(allowed);
  let ok = true;
  for (const k of Object.keys(obj)) {
    if (!allow.has(k)) {
      V(check, `${path}.${k}: 미지 키 = 에러 (§9.3). 정본의 인쇄 블록에 이 필드의 자리가 없다`);
      ok = false;
    }
  }
  for (const k of allow) {
    if (!has(obj, k) && !optional.has(k)) {
      V(check, `${path}.${k}: 누락 키 = 에러, 기본값 폴백 금지 (§9.3)`);
      ok = false;
    }
  }
  return ok;
}

/** "__AMBIGUOUS__" 전수 수집 — 정본이 아직 답하지 않은 자리 */
function scanAmbiguous(node, path) {
  if (isAmb(node)) { A(path); return; }
  if (Array.isArray(node)) {
    node.forEach((v, i) => scanAmbiguous(v, `${path}[${i}]`));
    return;
  }
  if (isObj(node)) {
    for (const k of Object.keys(node)) scanAmbiguous(node[k], `${path}.${k}`);
  }
}

/** 어휘 검사 (S3 공용) */
function vocab(check, value, allowedList, path) {
  if (isAmb(value)) return false;           // 모호값은 별도 집계 — 여기서 위반 취급 않는다
  if (!allowedList.includes(value)) {
    V(check, `${path}: 동결 어휘 밖의 값 ${JSON.stringify(value)} — 허용 = [${allowedList.join(', ')}]`);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// §9.2 파일 매니페스트 — 정확히 9개, 닫힘
// ---------------------------------------------------------------------------
const MANIFEST = [
  'rules', 'elements', 'weapons', 'passives', 'bullets',
  'enemies', 'bosses', 'stages', 'meta',
];
const SCHEMA_VERSION = 1;   // §9.4~§9.9 의 전 인쇄 블록이 1을 인쇄한다

const D = {};
function loadAll() {
  if (!existsSync(DATA_DIR)) {
    console.error(`FATAL: data 디렉터리가 없다: ${DATA_DIR}`);
    process.exit(2);
  }
  // 매니페스트가 닫혀 있으므로 여분 파일도 에러다 (§9.2 "정확히 9개, 닫힘")
  const present = readdirSync(DATA_DIR).filter((f) => extname(f) === '.json');
  const expect = new Set(MANIFEST.map((n) => `${n}.json`));
  for (const f of present) {
    if (!expect.has(f)) V('S2', `data/${f}: §9.2 매니페스트(정확히 9개, 닫힘) 밖의 파일`);
  }
  for (const name of MANIFEST) {
    const p = join(DATA_DIR, `${name}.json`);
    if (!existsSync(p)) {
      console.error(`FATAL: 매니페스트 파일 누락: data/${name}.json (§9.2)`);
      process.exit(2);
    }
    try {
      D[name] = JSON.parse(readFileSync(p, 'utf8'));
    } catch (e) {
      console.error(`FATAL: data/${name}.json JSON 파싱 실패 — ${e.message}`);
      process.exit(2);
    }
    // §9.3: schemaVersion 은 모든 파일 루트에 필수. 불일치 → 로드 실패
    if (!has(D[name], 'schemaVersion')) {
      V('S2', `data/${name}.json: schemaVersion 누락 (§9.3 — 모든 파일 루트에 필수)`);
    } else if (D[name].schemaVersion !== SCHEMA_VERSION) {
      V('S2', `data/${name}.json: schemaVersion ${D[name].schemaVersion} ≠ ${SCHEMA_VERSION} → 로드 실패 (§9.3)`);
    }
    scanAmbiguous(D[name], `${name}.json`);
  }
}

// ---------------------------------------------------------------------------
// 동결 어휘 (§13.4 S3)
// ---------------------------------------------------------------------------
const MOVE_IDS = ['dive', 'weave', 'column', 'strafe', 'anchor', 'orbitDrift', 'charge', 'rearIn'];              // §8.4 (8)
const EMITTER_TYPES = ['straight', 'fan', 'aimed', 'ring', 'spiral', 'laser', 'zone', 'wall'];                   // §8.5 (8)
const FORMATION_IDS = ['lineH', 'columnV', 'vWedge', 'arc', 'pincer', 'scatter'];                                // §8.7 (6)
const PART_TYPES = ['mobility', 'armament', 'armor', 'core'];                                                    // §8.12 (4)
const SHAPE_IDS = ['wedge', 'delta', 'hexPod', 'orb', 'cross', 'spike', 'ring', 'slab', 'fin', 'claw', 'dart', 'bulb']; // §9.10 (12)
const TARGET_MODES = ['forward', 'nearest', 'lowestHp', 'densest', 'randomInArena'];                             // §9.5 (5)
const FAMILIES = ['forward', 'fan', 'seeker', 'lance', 'orbit', 'aura', 'mine', 'boomerang', 'barrage', 'omni', 'drone', 'nova']; // §9.5 (12)
const PASSIVE_STATS = ['dmgMul', 'fireRateMul', 'areaMul', 'pierceAdd', 'projCountAdd', 'elementBonusMul',
  'ghostSecOnHit', 'hitBulletClearRadius', 'maxHpAdd', 'moveSpeedMul', 'xpGainMul', 'coinGainMul'];              // §9.6 (12)
const MOVE_PATTERNS = ['sway', 'orbitArc', 'holdCenter'];                                                        // §8.12.1 (3)
const BULLET_SHAPES = ['circle', 'hex'];                                                                         // §9.7 (2)
const BULLET_STATUS = [null, 'slow', 'stun'];                                                                     // §9.7
const SPAWN_EDGES = ['top', 'left', 'right', 'bottom'];                                                           // §8.7
const BOSS_TIERS = ['stage', 'mid', 'final'];                                                                     // §9.8
const RNG_STREAMS = ['theme', 'draft', 'spawn', 'elite', 'drop', 'pattern', 'boss', 'bot'];                       // §10.2 (8)

// §9.5 패밀리별 고유 파라미터 (계약의 "고유 n" 부분)
const FAMILY_OWN = {
  forward:   ['spreadDeg', 'jitterDeg', 'burstCount', 'burstIntervalSec', 'evoRampSec', 'evoRampFireRateMul'],
  fan:       ['arcDeg', 'evoBlastRadius', 'evoSecondaryDmgMul'],
  seeker:    ['turnRateDegSec', 'acquireRadius', 'retargetSec', 'evoDistinctTargets', 'evoRetargetOnKill'],
  lance:     ['beamWidthPx', 'chargeSec', 'rangePx', 'evoFullHeight'],
  orbit:     ['orbitRadius', 'angularSpeedDegSec', 'bodyCount', 'evoBulletClearCooldownSec'],
  aura:      ['radius', 'tickIntervalSec', 'falloff', 'evoPullForce'],
  mine:      ['placeIntervalSec', 'armSec', 'triggerRadius', 'blastRadius', 'maxAlive',
              'evoClusterCount', 'evoClusterRadius', 'evoSecondaryDmgMul'],
  boomerang: ['outRangePx', 'returnSpeed', 'canRehit', 'evoChainCount'],
  barrage:   ['strikeIntervalSec', 'strikesPerVolley', 'blastRadius', 'telegraphSec', 'evoRadiusMul'],
  omni:      ['dirCount', 'dirOffsetDeg', 'rearBias', 'evoRingRotDeg'],
  drone:     ['droneCount', 'anchorOffsets', 'droneFireSec', 'droneRangePx', 'evoTrailDelaySec'],
  nova:      ['intervalSec', 'radius', 'expandSec', 'telegraphSec', 'evoRing2Radius', 'evoClearBullets', 'evoSecondaryDmgMul'],
};
// §9.5 "공통: dmg cooldownSec count projSpeed projRadius lifetimeSec pierce hitCooldownSec targetMode"
const FAMILY_COMMON = ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius',
  'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'];
// §9.5 허용 targetMode (패밀리별). '—' = targetMode 키 자체가 없다
const FAMILY_TARGET_MODES = {
  forward: ['forward'], fan: ['forward'], seeker: ['nearest', 'lowestHp', 'randomInArena'],
  lance: ['forward', 'nearest'], orbit: null, aura: null, mine: null,
  boomerang: ['forward', 'nearest'], barrage: ['randomInArena', 'densest'],
  omni: null, drone: ['nearest', 'lowestHp', 'forward'], nova: null,
};
// §7.4 거동별 telegraphSec 하한
const TELEGRAPH_FLOOR_BY_TYPE = {
  straight: 0.55, fan: 0.60, aimed: 0.60, ring: 0.60,
  spiral: 0.60, wall: 0.80, zone: 0.90, laser: 1.20,
};
const TELEGRAPH_FLOOR_SLOW_BULLET = 0.80;   // §7.4 "상태이상(slow) 탄"
const TELEGRAPH_FLOOR_MIDBOSS = 1.20;       // §7.4 "중간보스 패턴"

// ===========================================================================
//  S1 — core 순수성 (§9.1 모듈 경계)
//  금지 식별자 + src/core/weapons/** 숫자 리터럴(0,1,-1,0.5,2만) + import 경계
//  ★ src/core/bot.js 포함 (§10.2)
// ===========================================================================
const CORE_FORBIDDEN_IDENTS = ['window', 'document', 'canvas', 'requestAnimationFrame',
  'Date', 'performance', 'Math.random', 'fetch', 'localStorage', 'console'];
const WEAPON_NUMERIC_ALLOWED = new Set(['0', '1', '-1', '0.5', '2']);

function listJsFiles(dir) {
  const out = [];
  if (!existsSync(dir)) return out;
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    const st = statSync(p);
    if (st.isDirectory()) out.push(...listJsFiles(p));
    else if (/\.(mjs|js)$/.test(e)) out.push(p);
  }
  return out;
}

/** 문자열/주석을 지워 오탐을 줄인다 (완전한 파서가 아니다 — 보수적 근사) */
function stripStringsAndComments(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/(^|[^:])\/\/[^\n]*/g, '$1 ')
    .replace(/`(?:\\.|[^`\\])*`/g, '""')
    .replace(/'(?:\\.|[^'\\])*'/g, '""')
    .replace(/"(?:\\.|[^"\\])*"/g, '""');
}

function S1_corePurity() {
  const coreDir = join(SRC_DIR, 'core');
  if (!existsSync(coreDir)) {
    SKIP('S1', `src/core/ 가 아직 없다 → core 순수성 검사 건너뜀 (구현 시작 전)`);
    return;
  }
  const files = listJsFiles(coreDir);
  for (const f of files) {
    const rel = relative(ROOT, f);
    const raw = readFileSync(f, 'utf8');
    const code = stripStringsAndComments(raw);

    // (1) 금지 식별자
    for (const ident of CORE_FORBIDDEN_IDENTS) {
      const pat = ident.includes('.')
        ? new RegExp(ident.replace('.', '\\s*\\.\\s*'))
        : new RegExp(`\\b${ident}\\b`);
      if (pat.test(code)) V('S1', `${rel}: core 금지 식별자 "${ident}" (§9.1)`);
    }

    // (2) import 경계 — src/core/** 는 src/core/** 외 import 금지
    for (const m of raw.matchAll(/\bfrom\s*['"]([^'"]+)['"]/g)) {
      const spec = m[1];
      const target = spec.startsWith('.') ? resolve(dirname(f), spec) : spec;
      const inCore = spec.startsWith('.') && !relative(coreDir, target).startsWith('..');
      if (!inCore) V('S1', `${rel}: core 밖 import "${spec}" — src/core/** 는 src/core/** 외 import 금지 (§9.1)`);
    }

    // (3) src/core/weapons/** 숫자 리터럴 — 0, 1, -1, 0.5, 2 만
    if (!relative(join(coreDir, 'weapons'), f).startsWith('..')) {
      for (const m of code.matchAll(/-?\d+(?:\.\d+)?/g)) {
        const lit = m[0];
        if (!WEAPON_NUMERIC_ALLOWED.has(lit)) {
          V('S1', `${rel}: weapons/** 숫자 리터럴 "${lit}" — 허용 = 0, 1, -1, 0.5, 2 뿐 (§9.1)`);
        }
      }
    }
  }
  if (!files.some((f) => /bot\.(m?js)$/.test(f))) {
    SKIP('S1', 'src/core/bot.js 가 아직 없다 (§10.2가 요구하는 8번째 스트림의 거처)');
  }
}

// ===========================================================================
//  S2 — 스키마 (§9.3 · §9.4~§9.9)
//  타입 · 필수 키 · 미지 키 거부 · 참조 무결성
//  ★ rules.json 루트 키 = 17개 목록 (§9.4)
// ===========================================================================
const RULES_ROOT_17 = ['loop', 'view', 'collide', 'caps', 'player', 'status', 'bomb', 'elite',
  'boss', 'fairness', 'hud', 'passiveHooks', 'input', 'palette', 'visual', 'render', 'audio'];

function S2_schema() {
  const r = D.rules;

  // --- rules.json 루트 = schemaVersion + 정확히 17 블록 (§9.4) --------------
  closedKeys('S2', r, ['schemaVersion', ...RULES_ROOT_17], 'rules');
  if (RULES_ROOT_17.length !== 17) C('S2', `내부 오류: 루트 목록이 ${RULES_ROOT_17.length}개 (정본은 17)`);

  closedKeys('S2', r.loop, ['tickHz', 'maxStepsPerFrame', 'maxFrameGapMs', 'interpolate'], 'rules.loop');
  closedKeys('S2', r.view, ['logicalW', 'logicalH', 'arena', 'panelLeftW', 'panelRightW', 'bandTopH',
    'bandHpH', 'bandXpH', 'playerBoundsInset', 'spawnLineY', 'minViewportW', 'minViewportH', 'maxDpr'], 'rules.view');
  if (isObj(r.view)) {
    closedKeys('S2', r.view.arena, ['x', 'y', 'w', 'h'], 'rules.view.arena');
    closedKeys('S2', r.view.playerBoundsInset, ['top', 'bottom', 'left', 'right'], 'rules.view.playerBoundsInset');
  }
  closedKeys('S2', r.collide, ['gridCellPx'], 'rules.collide');

  closedKeys('S2', r.caps, ['playerBullets', 'enemyBullets', 'enemies', 'pickups', 'zones', 'drones',
    'particles', 'telegraphs', 'damageNumbers', 'effectMarkers', 'overflow'], 'rules.caps');
  if (isObj(r.caps)) {
    closedKeys('S2', r.caps.overflow, ['playerBullet', 'enemyBullet', 'enemy', 'pickup', 'zone',
      'drone', 'telegraph', 'particle', 'damageNumber', 'effectMarker'], 'rules.caps.overflow');
    // §12.1 "모든 캡에 정책이 있다" — 10 캡 ⟺ 10 정책
    const capNames = Object.keys(r.caps).filter((k) => k !== 'overflow');
    if (isObj(r.caps.overflow) && capNames.length !== Object.keys(r.caps.overflow).length) {
      V('S2', `rules.caps: 캡 ${capNames.length}개 ↔ overflow 정책 ${Object.keys(r.caps.overflow).length}개 — §12.1 "모든 캡에 정책이 있다"`);
    }
  }

  closedKeys('S2', r.player, ['hpMax', 'hpSegment', 'spriteRadius', 'hitboxRadius', 'moveSpeed',
    'moveResponseTau', 'diagonalNormalize', 'iframeSec', 'defenseBase', 'damageFloorRatio',
    'lowHpThreshold', 'lowHpCriticalThreshold', 'magnetRadius', 'startWeaponId', 'startStance',
    'stanceSwitchCooldown', 'stancePersistAcrossStages', 'elementCapPerElement', 'elementCapTotal',
    'weaponSlots', 'passiveSlots', 'lives'], 'rules.player');
  closedKeys('S2', r.status, ['slowMoveSpeedMul', 'stackMode', 'resistAffects'], 'rules.status');
  closedKeys('S2', r.bomb, ['stockStart', 'stockMax', 'iframeSec', 'mobDmg', 'clearsEnemyBullets',
    'clearsDuringBoss', 'bossDmgRatio', 'bossDmgCap'], 'rules.bomb');
  closedKeys('S2', r.elite, ['perWaveMax', 'hpMult', 'sizeMult', 'contactDmgMul', 'xpMult', 'coin',
    'healDropChance', 'bandAllowed', 'elementAllowed'], 'rules.elite');

  // §9.4 인쇄 블록이 boss 스코프의 필드 집합을 확정한다 (C-7)
  closedKeys('S2', r.boss, ['partCount', 'partRegen', 'summonsAllowed', 'partHitPriority',
    'phaseThresholds', 'phaseTransitionSec', 'timerPausesOnPhaseTransition', 'introSec',
    'timerStartsAfterIntro', 'timerExpire', 'coreGateMul', 'mobilityPenalty', 'coreElement',
    'partNormalForbidden', 'partElementDistinctMin', 'partThemeElementMax', 'armorElementNotTheme',
    'armorPartCountRange', 'armorCoreRatioBandPct', 'coin', 'partCoin', 'optionalPartArmorRatio',
    'midBossSummonsAllowed', 'finale'], 'rules.boss');
  if (isObj(r.boss)) {
    closedKeys('S2', r.boss.finale, ['partCount', 'armorPartCount', 'exemptRules', 'allowNormalPeripheral'],
      'rules.boss.finale');
  }

  closedKeys('S2', r.fairness, ['minTelegraphSec', 'minStunTelegraphSec', 'maxStunSec', 'maxBulletSpeed',
    'maxAimedBulletSpeed', 'minBulletRadiusPx', 'minGapWidthPx', 'minSpawnRadiusPx',
    'maxSimultaneousEnemyBullets', 'enemyConcurrentMax', 'swarmConcurrentMax', 'crisisWaveResidualMax',
    'telegraphConcurrentMaxPerEntity', 'telegraphConcurrentMaxGlobal', 'playerWeaponsExempt'], 'rules.fairness');

  closedKeys('S2', r.hud, ['hitboxAlwaysVisible', 'showElementBudget', 'fontHeroPx', 'fontLargePx',
    'fontMediumPx', 'fontBodyPx', 'fontSmallPx', 'panelPadPx', 'keycapBoxPx', 'bossHpBarH',
    'hpBarSegGapPx', 'xpBarH', 'hpBarSegCount', 'panelCacheDirtyOnly', 'parGhostEnabled',
    'elementMatrixInPanel', 'coinShowsScoreValue', 'noHitIndicator', 'tokenKeycapGatedDisplay',
    'stanceHintTargetsMajorityElement', 'icons'], 'rules.hud');
  // §9.4.1: 전 폰트 크기 ≥ visual.text.minPx(14)
  if (isObj(r.hud) && isObj(r.visual) && isObj(r.visual.text) && num(r.visual.text.minPx)) {
    for (const k of ['fontHeroPx', 'fontLargePx', 'fontMediumPx', 'fontBodyPx', 'fontSmallPx']) {
      if (num(r.hud[k]) && r.hud[k] < r.visual.text.minPx) {
        V('S2', `rules.hud.${k} = ${r.hud[k]} < visual.text.minPx(${r.visual.text.minPx}) — §9.4.1`);
      }
    }
  }

  closedKeys('S2', r.passiveHooks, FAMILIES, 'rules.passiveHooks');
  if (isObj(r.passiveHooks)) {
    for (const f of FAMILIES) {
      if (!has(r.passiveHooks, f)) continue;
      closedKeys('S2', r.passiveHooks[f], ['rateKey', 'countKey', 'pierce', 'areaKeys'], `rules.passiveHooks.${f}`);
    }
  }

  closedKeys('S2', r.input, ['layout', 'socd', 'pauseOnBlur', 'bindings'], 'rules.input');
  if (isObj(r.input)) {
    closedKeys('S2', r.input.bindings, ['move', 'stanceNormal', 'stanceFire', 'stanceWater', 'stanceGrass',
      'bomb', 'timeToken', 'pause', 'options', 'draftPick', 'reroll', 'reorderToggle', 'grab',
      'confirm', 'cursor'], 'rules.input.bindings');
  }

  closedKeys('S2', r.palette, ['element', 'elementCvd', 'threat', 'status', 'pickup', 'enemyBody',
    'partDestroyed', 'neutralGray', 'hud', 'bg'], 'rules.palette');
  if (isObj(r.palette)) {
    closedKeys('S2', r.palette.element, ['normal', 'fire', 'water', 'grass'], 'rules.palette.element');
    closedKeys('S2', r.palette.elementCvd, ['normal', 'fire', 'water', 'grass'], 'rules.palette.elementCvd');
    closedKeys('S2', r.palette.threat, ['enemyBullet', 'telegraph', 'bulletCore', 'outline'], 'rules.palette.threat');
    closedKeys('S2', r.palette.status, ['band'], 'rules.palette.status');
    closedKeys('S2', r.palette.pickup, ['coin', 'xp'], 'rules.palette.pickup');
    closedKeys('S2', r.palette.hud, ['panelBg', 'panelRule', 'textPrimary', 'textDim', 'hpFill'], 'rules.palette.hud');
    closedKeys('S2', r.palette.bg, ['maxSaturation', 'maxLightness', 'cvdMaxLightness',
      'parallaxLayers', 'maxScrollSpeed'], 'rules.palette.bg');
  }

  // §9.4.3 — visual 전 키 인쇄
  closedKeys('S2', r.visual, ['iframeBlinkHz', 'statusBulletSpeedMul', 'stance', 'playerBullet',
    'glyph', 'telegraph', 'band', 'zone', 'timer', 'trail', 'hitFx', 'a11y', 'text'], 'rules.visual');
  if (isObj(r.visual)) {
    closedKeys('S2', r.visual.stance, ['ringExpandSec', 'ringMaxRadiusPx', 'ringStrokePx', 'emptyDesatSec',
      'dotRadiusPx', 'dotRingPx', 'auraAlpha', 'pipPx', 'pipPxCvd', 'pipGapPx', 'pipOffsetYPx',
      'hintPulseHz', 'hintPulseAlpha'], 'rules.visual.stance');
    closedKeys('S2', r.visual.playerBullet, ['coreRadiusRatio', 'coreLightnessAdd'], 'rules.visual.playerBullet');
    closedKeys('S2', r.visual.glyph, ['bodyRatio', 'maxPx', 'occludedSkip', 'lodMinBodyPx', 'lodDegradedBodyPx'], 'rules.visual.glyph');
    closedKeys('S2', r.visual.telegraph, ['strokePx', 'dashPx', 'dashTightenAtPct', 'dashTightenMul',
      'airAlpha', 'fillAlpha', 'emphasisBySpeed'], 'rules.visual.telegraph');
    closedKeys('S2', r.visual.band, ['plateAlpha', 'contentOpaque'], 'rules.visual.band');
    closedKeys('S2', r.visual.zone, ['fillAlpha', 'pulseHz'], 'rules.visual.zone');
    closedKeys('S2', r.visual.timer, ['warnScale', 'warnPulseHz', 'alertScale', 'alertPulseHz'], 'rules.visual.timer');
    closedKeys('S2', r.visual.trail, ['ghostCount', 'ghostAlpha'], 'rules.visual.trail');
    closedKeys('S2', r.visual.hitFx, ['numberTargets', 'numberAggregateSec', 'numberMinPx', 'numberOutlinePx',
      'numberLifeSec', 'numberDriftPx', 'markerPolicy', 'markerCooldownSecPerEntity', 'superFreezeSec',
      'superFreezeScale', 'resistArcSweepDeg', 'resistArcLifeSec', 'resistArcStrokePx', 'particles'], 'rules.visual.hitFx');
    if (isObj(r.visual.hitFx)) {
      closedKeys('S2', r.visual.hitFx.particles, ['super', 'neutral', 'resist'], 'rules.visual.hitFx.particles');
    }
    closedKeys('S2', r.visual.a11y, ['cbMode', 'reduceFlash', 'screenShake', 'shakeMaxPx',
      'fullscreenFlashMaxPerSec', 'fullscreenFlashMaxAlpha'], 'rules.visual.a11y');
    closedKeys('S2', r.visual.text, ['family', 'minPx', 'outlinePx', 'outlineColor'], 'rules.visual.text');
  }

  closedKeys('S2', r.render, ['playerFxCompositeAlpha', 'killFxCompositeAlpha', 'playerBulletMaxAlpha',
    'playerBulletMaxRadiusPx', 'particleMaxAlpha', 'particleMaxLifeSec', 'fxMinRealMs', 'targetFps',
    'degradeOnFrameMs', 'degradeRecoverFrames'], 'rules.render');

  closedKeys('S2', r.audio, ['busGain', 'cueRateLimitPerSec', 'bgm'], 'rules.audio');
  if (isObj(r.audio)) closedKeys('S2', r.audio.busGain, ['bgm', 'sfx'], 'rules.audio.busGain');

  // --- elements.json (§9.4.4) ---------------------------------------------
  closedKeys('S2', D.elements, ['schemaVersion', 'order', 'investable', 'matrix'], 'elements');

  // --- passives.json (§9.6) ------------------------------------------------
  closedKeys('S2', D.passives, ['schemaVersion', 'maxLevel', 'stats', 'passives'], 'passives');
  if (Array.isArray(D.passives.passives)) {
    for (const p of D.passives.passives) {
      closedKeys('S2', p, ['id', 'name', 'stat', 'values'], `passives[${p && p.id}]`);
      if (Array.isArray(p.values) && p.values.length !== D.passives.maxLevel) {
        V('S2', `passives[${p.id}].values: 길이 ${p.values.length} ≠ maxLevel(${D.passives.maxLevel}) — §9.6`);
      }
    }
    // §9.6 "폐쇄 스탯 어휘 12종, 12 패시브와 1:1"
    const stats = D.passives.passives.map((p) => p.stat);
    if (new Set(stats).size !== stats.length) V('S2', 'passives: stat 중복 — §9.6 "12훅 = 12 패시브 1:1"');
    if (D.passives.passives.length !== 12) V('S2', `passives: ${D.passives.passives.length}종 ≠ 12 (§9.6)`);
  }

  // --- bullets.json (§9.7) -------------------------------------------------
  closedKeys('S2', D.bullets, ['schemaVersion', 'bullets'], 'bullets');
  if (Array.isArray(D.bullets.bullets)) {
    for (const b of D.bullets.bullets) {
      closedKeys('S2', b, ['id', 'radius', 'hitboxScale', 'speed', 'dmg', 'shape', 'status',
        'statusDurationSec', 'accel', 'turnRateDegSec', 'retargetSec'], `bullets[${b && b.id}]`);
      // §9.7 "element 키가 존재하지 않는다 — 스키마가 '적 공격에는 속성이 없다'를 강제한다"
      if (isObj(b) && has(b, 'element')) {
        V('S2', `bullets[${b.id}].element: 존재해서는 안 되는 키 — §9.7/§4.1 "적의 공격에는 속성이 없다"`);
      }
    }
  }

  // --- enemies.json (§9.7) -------------------------------------------------
  closedKeys('S2', D.enemies, ['schemaVersion', 'bands', 'archetypes', 'emitters'], 'enemies');
  if (isObj(D.enemies.bands)) {
    closedKeys('S2', D.enemies.bands, ['chaff', 'line', 'turret', 'bruiser'], 'enemies.bands');
    for (const [bn, bv] of Object.entries(D.enemies.bands)) {
      // §8.10: xpRef 는 chaff 밴드에만 산다 (04-N-1)
      const allowed = bn === 'chaff'
        ? ['hpMult', 'coinDropChance', 'coin', 'xpRef']
        : ['hpMult', 'coinDropChance', 'coin'];
      closedKeys('S2', bv, allowed, `enemies.bands.${bn}`);
      // §9.7: bands[].sizePx 는 삭제되었다 (04-R17)
      if (isObj(bv) && has(bv, 'sizePx')) {
        V('S2', `enemies.bands.${bn}.sizePx: 삭제된 키 — 크기의 단일 진실 = archetypes[].radius (§9.7, 04-R17)`);
      }
    }
  }
  if (Array.isArray(D.enemies.archetypes)) {
    for (const a of D.enemies.archetypes) {
      closedKeys('S2', a, ['id', 'name', 'desc', 'band', 'shapeId', 'radius', 'moveId', 'moveParams',
        'attack', 'contactDmg', 'hp', 'xp', 'score', 'themeOnly'], `enemies.archetypes[${a && a.id}]`);
      // §9.7: 삭제 확정된 필드들
      for (const dead of ['tier', 'element', 'hpScalePerStage', 'spriteId', 'unlockStageMin']) {
        if (isObj(a) && has(a, dead)) {
          V('S2', `enemies.archetypes[${a.id}].${dead}: 삭제된 키 (§9.7)`);
        }
      }
      if (isObj(a) && a.attack !== null && isObj(a.attack)) {
        closedKeys('S2', a.attack, ['emitterId', 'firstDelaySec'], `enemies.archetypes[${a.id}].attack`);
      }
    }
  }
  if (Array.isArray(D.enemies.emitters)) {
    const COMMON = ['id', 'type', 'bulletId', 'from', 'telegraphSec', 'everySec', 'offsetSec', 'repeat', 'restSec'];
    const OWN = {
      straight: ['count', 'spreadDeg', 'speed'],
      fan: ['count', 'arcDeg', 'speed'],
      aimed: ['count', 'spreadDeg', 'speed', 'leadSec'],
      ring: ['count', 'speed', 'rotOffsetDeg'],
      spiral: ['count', 'speed', 'rotStepDeg', 'durationSec', 'rateSec'],
      laser: ['chargeSec', 'widthPx', 'activeSec', 'angleDeg', 'trackDuringCharge'],
      zone: ['radius', 'activeSec', 'dmg'],
      wall: ['count', 'gapCount', 'gapWidthPx', 'speed'],
    };
    for (const e of D.enemies.emitters) {
      if (!isObj(e)) continue;
      if (!vocab('S3', e.type, EMITTER_TYPES, `enemies.emitters[${e.id}].type`)) continue;
      closedKeys('S2', e, [...COMMON, ...(OWN[e.type] || [])], `enemies.emitters[${e.id}]`);
      // §8.5 "zone 의 dps 는 존재하지 않는다"
      if (has(e, 'dps')) V('S2', `enemies.emitters[${e.id}].dps: 존재하지 않는 키 — §8.5 "모든 피해는 적용 1회"`);
    }
  }

  // --- weapons.json (§9.5) -------------------------------------------------
  closedKeys('S2', D.weapons, ['schemaVersion', 'weapons'], 'weapons');
  if (Array.isArray(D.weapons.weapons)) {
    for (const w of D.weapons.weapons) {
      if (!isObj(w)) continue;
      closedKeys('S2', w, ['id', 'family', 'name', 'desc', 'elementStampMode', 'base', 'levels', 'evolution'],
        `weapons[${w.id}]`);
      if (isObj(w.evolution)) {
        closedKeys('S2', w.evolution, ['name', 'desc', 'params'], `weapons[${w.id}].evolution`);
        // §9.5: 진화 flags 는 폐기 — evo* 파라미터로만
        if (has(w.evolution, 'flags')) {
          V('S2', `weapons[${w.id}].evolution.flags: 폐기된 키 — §9.5 "임의 문자열은 AI가 발명할 수 있다"`);
        }
        // §9.5: evolution.family 변경 폐기
        if (has(w.evolution, 'family')) {
          V('S2', `weapons[${w.id}].evolution.family: 폐기된 키 — §9.5 "family 변경 금지"`);
        }
      }
      // §9.5 "id == family" (12종 1:1)
      if (w.id !== w.family) V('S2', `weapons[${w.id}]: id ≠ family(${w.family}) — §9.5 "id == family"`);

      const own = FAMILY_OWN[w.family];
      if (!own) continue;
      const contractSuperset = new Set([...FAMILY_COMMON, ...own]);

      // ★ 계약의 "미지 키" 절반만 검사 가능하다 — 아래 CANON 신고 참조
      const scanContract = (obj, path) => {
        if (!isObj(obj)) return;
        for (const k of Object.keys(obj)) {
          if (!contractSuperset.has(k)) {
            V('S2', `${path}.${k}: 패밀리 "${w.family}" 계약(공통 9 + 고유 ${own.length}) 밖의 키 = 미지 키 = 에러 (§9.5)`);
          }
        }
      };
      scanContract(w.base, `weapons[${w.id}].base`);
      if (Array.isArray(w.levels)) {
        w.levels.forEach((lv, i) => scanContract(lv, `weapons[${w.id}].levels[${i}]`));
      }
      if (isObj(w.evolution) && isObj(w.evolution.params)) {
        for (const k of Object.keys(w.evolution.params)) {
          if (!k.startsWith('evo')) {
            V('S2', `weapons[${w.id}].evolution.params.${k}: evo* 접두가 아니다 (§9.5)`);
          } else if (!own.includes(k)) {
            V('S2', `weapons[${w.id}].evolution.params.${k}: 패밀리 "${w.family}" 계약 밖의 evo 키 (§9.5)`);
          }
        }
      }
      // §9.5 허용 targetMode (패밀리별)
      const tm = FAMILY_TARGET_MODES[w.family];
      if (isObj(w.base)) {
        if (tm === null && has(w.base, 'targetMode')) {
          V('S3', `weapons[${w.id}].base.targetMode: "${w.family}"는 자기중심/무조준 → 계약에 targetMode 가 없다 (§9.5)`);
        } else if (tm && has(w.base, 'targetMode') && !tm.includes(w.base.targetMode)) {
          V('S3', `weapons[${w.id}].base.targetMode = "${w.base.targetMode}" — "${w.family}" 허용 = [${tm.join(', ')}] (§9.5)`);
        } else if (tm && !has(w.base, 'targetMode')) {
          V('S2', `weapons[${w.id}].base.targetMode: 누락 키 = 에러 (§9.5 계약)`);
        }
      }
      // §4.4 elementStampMode
      const stampLive = ['orbit', 'aura'];
      const wantStamp = stampLive.includes(w.family) ? 'live' : 'spawn';
      if (w.elementStampMode !== wantStamp) {
        V('S2', `weapons[${w.id}].elementStampMode = "${w.elementStampMode}" ≠ "${wantStamp}" — §4.4 (구조 결정 = 잠금 키)`);
      }
    }
  }

  // --- bosses.json (§9.8) --------------------------------------------------
  closedKeys('S2', D.bosses, ['schemaVersion', 'bosses'], 'bosses');
  if (Array.isArray(D.bosses.bosses)) {
    for (const b of D.bosses.bosses) {
      if (!isObj(b)) continue;
      vocab('S3', b.tier, BOSS_TIERS, `bosses[${b.id}].tier`);
      if (b.tier === 'mid') {
        closedKeys('S2', b, ['id', 'name', 'tier', 'themeId', 'element', 'hp', 'radius', 'shapeId',
          'contactDmg', 'moveId', 'moveParams', 'patternSet', 'summon', 'parts', 'xp', 'coin',
          'healDropChance', 'score'], `bosses[${b.id}]`);
        // §8.9 v1.2 정정: bosses[].leaveAfterSec 는 삭제되었다
        if (has(b, 'leaveAfterSec')) {
          V('S2', `bosses[${b.id}].leaveAfterSec: 삭제된 키 — 유일 소유자 = stages.phase.midBossLeaveAfterSec (§8.9)`);
        }
      } else {
        closedKeys('S2', b, ['id', 'name', 'tier', 'themeId', 'movePattern', 'movePatternParams',
          'armorCoreRatio', 'summon', 'core', 'parts'], `bosses[${b.id}]`);
        if (isObj(b.core)) {
          closedKeys('S2', b.core, ['element', 'hp', 'radius', 'contactDmg', 'shapeId', 'score'], `bosses[${b.id}].core`);
        }
        if (isObj(b.movePatternParams)) {
          closedKeys('S2', b.movePatternParams, ['speedPxSec', 'ampPx', 'yHoldPx'], `bosses[${b.id}].movePatternParams`);
        }
        for (const p of (Array.isArray(b.parts) ? b.parts : [])) {
          if (!isObj(p)) continue;
          closedKeys('S2', p, ['id', 'name', 'partType', 'element', 'hp', 'radius', 'anchor',
            'contactDmg', 'shapeId', 'score', 'patternSet'], `bosses[${b.id}].parts[${p.id}]`);
          // §9.8: 존재하지 않는 키들
          for (const dead of ['regenSec', 'onDestroy', 'hpShare']) {
            if (has(p, dead)) V('S2', `bosses[${b.id}].parts[${p.id}].${dead}: 존재하지 않는 키 (§9.8)`);
          }
        }
      }
      if (isObj(b.summon)) {
        closedKeys('S2', b.summon, ['archetypeId', 'count', 'everySec', 'formationId'], `bosses[${b.id}].summon`);
      }
    }
  }

  // --- stages.json (§9.9) --------------------------------------------------
  closedKeys('S2', D.stages, ['schemaVersion', 'themeDraw', 'curve', 'phase', 'themes', 'formations'], 'stages');
  closedKeys('S2', D.stages.themeDraw, ['pool', 'count', 'allowRepeat', 'stage1RequiresIntroOk', 'finalStageId'], 'stages.themeDraw');
  closedKeys('S2', D.stages.curve, ['enemyHpScale', 'xpScale', 'bossHpScale', 'spawnDensityScale',
    'midBossCount', 'elitePerWaveChance', 'swarmTotalScale', 'rearSpawnAllowed'], 'stages.curve');
  closedKeys('S2', D.stages.phase, ['mobPhaseSec', 'mobPhaseSkippable', 'mobPhaseMaxWaves', 'waveIntervalSec',
    'waveClearAdvance', 'mobPhaseExitFadeSec', 'mobPhaseExitClearBullets', 'phaseEndAutocollect',
    'enemyExitForfeitsReward', 'waveListExhausted', 'crisisStartSec', 'crisisDurationSec', 'crisisWarnSec',
    'crisisSuspendsWaves', 'crisisTotal', 'crisisSubWaves', 'crisisElementRule', 'midBossLeaveAfterSec',
    'midBossElementRule', 'midBossForcedLeaveOnCrisis', 'bossEntrySec', 'bossTimerSec', 'timerWarnSec',
    'timerRedAlertSec', 'statusStunMaxPerStage'], 'stages.phase');
  // §9.9.3: crisisSubWaveIntervalSec 은 파생값이지 키가 아니다 (새 키 0)
  if (isObj(D.stages.phase) && has(D.stages.phase, 'crisisSubWaveIntervalSec')) {
    V('S2', 'stages.phase.crisisSubWaveIntervalSec: 파생값이지 키가 아니다 — §9.9.3 (= crisisDurationSec / crisisSubWaves)');
  }
  // §9.9.2 formations — 6종 + 파라미터
  closedKeys('S2', D.stages.formations, FORMATION_IDS, 'stages.formations');
  const FORM_PARAMS = {
    lineH: ['gapPx'], columnV: ['gapSec'], vWedge: ['gapPx', 'angleDeg'],
    arc: ['radiusPx', 'spanDeg'], pincer: ['yStartPx', 'yStepPx'], scatter: ['jitterPx', 'minSepPx'],
  };
  if (isObj(D.stages.formations)) {
    for (const [f, params] of Object.entries(FORM_PARAMS)) {
      if (has(D.stages.formations, f)) closedKeys('S2', D.stages.formations[f], params, `stages.formations.${f}`);
    }
  }
  const FINAL_ID = isObj(D.stages.themeDraw) ? D.stages.themeDraw.finalStageId : 'finale';
  if (Array.isArray(D.stages.themes)) {
    for (const t of D.stages.themes) {
      if (!isObj(t)) continue;
      const base = ['id', 'name', 'element', 'introOk', 'skinId', 'bossId', 'roster', 'mix',
        'mixGranularity', 'waves', 'midBossAtSec', 'elitesAtSec'];
      // §8.16 · §9.9: finaleCrisisRotating 은 finale 엔트리에만 인쇄되었다
      const allowed = t.id === FINAL_ID ? [...base, 'finaleCrisisRotating'] : base;
      closedKeys('S2', t, allowed, `stages.themes[${t.id}]`);
      if (t.id !== FINAL_ID && has(t, 'finaleCrisisRotating')) {
        V('S2', `stages.themes[${t.id}].finaleCrisisRotating: finale 전용 키 (§8.16)`);
      }
      for (const rEnt of (Array.isArray(t.roster) ? t.roster : [])) {
        closedKeys('S2', rEnt, ['archetypeId', 'unlockStageMin'], `stages.themes[${t.id}].roster[${rEnt && rEnt.archetypeId}]`);
      }
      (Array.isArray(t.waves) ? t.waves : []).forEach((w, i) => {
        closedKeys('S2', w, ['formationId', 'archetypeId', 'count', 'element', 'spawnEdge', 'eliteIndex'],
          `stages.themes[${t.id}].waves[${i}]`);
        // §8.7: atSec 절대 타임라인은 폐기 — 순서 리스트다
        if (isObj(w) && has(w, 'atSec')) {
          V('S2', `stages.themes[${t.id}].waves[${i}].atSec: 폐기된 키 — 웨이브는 순서 리스트다 (§8.7)`);
        }
      });
    }
  }

  // --- meta.json (§9.9 · §11 · §10.4 · §13.1) -----------------------------
  closedKeys('S2', D.meta, ['schemaVersion', 'xp', 'draft', 'shop', 'score', 'flow', 'onboarding',
    'difficulty', 'bot', 'certify'], 'meta');
  closedKeys('S2', D.meta.xp, ['curve', 'base', 'exp', 'levelUpsPerRunTarget', 'levelUpQueueMode'], 'meta.xp');
  if (isObj(D.meta.draft)) {
    closedKeys('S2', D.meta.draft, ['optionCount', 'slotAssign', 'categoryWeights', 'newWeaponSlotScale',
      'weaponLevelEvolutionBonus', 'elementFirstLevelBonus', 'passiveNewBonus', 'distinctItemsPerDraft',
      'filterInvalid', 'newWeaponWhenSlotsFull', 'elementLevelOfferRequiresWeaponCount',
      'guaranteeElementCardOnFirstDraft', 'guaranteeNewWeaponUntilSlots', 'elementCardPity', 'reroll',
      'fallback', 'pauseGame'], 'meta.draft');
    closedKeys('S2', D.meta.draft.categoryWeights, ['newWeapon', 'weaponLevel', 'elementLevel', 'passive'], 'meta.draft.categoryWeights');
    closedKeys('S2', D.meta.draft.reroll, ['granularity', 'canRepeatPrevious', 'maxPerDraft'], 'meta.draft.reroll');
    closedKeys('S2', D.meta.draft.fallback, ['id', 'name', 'coins'], 'meta.draft.fallback');
  }
  closedKeys('S2', D.meta.onboarding, ['autoEquipFirstElement', 'stanceHintPulse', 'stanceHintPulseStageMax'], 'meta.onboarding');
  if (isObj(D.meta.flow)) {
    closedKeys('S2', D.meta.flow, ['themeBannerSec', 'stageClearSec', 'healSec', 'stageClearHealPct',
      'pauseResumeCountdownSec', 'attractIdleSec', 'continueCost', 'continueTimerRestoreSec',
      'continueIframeSec', 'continueHealToFull', 'continueMaxPerRun', 'menuSpeed', 'deathAnimSec',
      'edgeTriggerOnStateEnter', 'pauseAllowsAbandon', 'attract', 'stagePar'], 'meta.flow');
    closedKeys('S2', D.meta.flow.attract, ['difficulty', 'draftDwellSec', 'endAfterMobPhase'], 'meta.flow.attract');
  }
  if (isObj(D.meta.difficulty)) {
    closedKeys('S2', D.meta.difficulty, ['normal', 'hard', 'hell', 'disaster', 'stunMinDifficulty'], 'meta.difficulty');
    for (const k of ['normal', 'hard', 'hell', 'disaster']) {
      if (has(D.meta.difficulty, k)) closedKeys('S2', D.meta.difficulty[k], ['speed', 'scoreMul'], `meta.difficulty.${k}`);
    }
  }

  // --- 참조 무결성 (§9.3 "모든 *Id 는 로드 시 대상 존재 확인") ------------
  refIntegrity();
}

function refIntegrity() {
  const archIds = new Set((D.enemies.archetypes || []).map((a) => a.id));
  const emitIds = new Set((D.enemies.emitters || []).map((e) => e.id));
  const bulletIds = new Set((D.bullets.bullets || []).map((b) => b.id));
  const bossIds = new Set((D.bosses.bosses || []).map((b) => b.id));
  const weaponIds = new Set((D.weapons.weapons || []).map((w) => w.id));
  const themeIds = new Set((D.stages.themes || []).map((t) => t.id));
  const formIds = new Set(Object.keys(D.stages.formations || {}));

  const need = (set, id, where) => {
    if (id === null || id === undefined || isAmb(id)) return;
    if (!set.has(id)) V('S2', `${where}: 참조 무결성 실패 — "${id}" 가 존재하지 않는다 (§9.3)`);
  };

  for (const a of D.enemies.archetypes || []) {
    if (isObj(a.attack)) need(emitIds, a.attack.emitterId, `enemies.archetypes[${a.id}].attack.emitterId`);
  }
  for (const e of D.enemies.emitters || []) need(bulletIds, e.bulletId, `enemies.emitters[${e.id}].bulletId`);
  for (const b of D.bosses.bosses || []) {
    if (isObj(b.summon)) {
      need(archIds, b.summon.archetypeId, `bosses[${b.id}].summon.archetypeId`);
      need(formIds, b.summon.formationId, `bosses[${b.id}].summon.formationId`);
    }
    const sets = b.tier === 'mid' ? (b.patternSet || []) : [];
    for (const ps of sets) for (const id of (ps.emitterIds || [])) need(emitIds, id, `bosses[${b.id}].patternSet.emitterIds`);
    for (const p of b.parts || []) {
      for (const ps of (p.patternSet || [])) for (const id of (ps.emitterIds || [])) {
        need(emitIds, id, `bosses[${b.id}].parts[${p.id}].patternSet.emitterIds`);
      }
    }
    if (b.tier !== 'mid') need(themeIds, b.themeId, `bosses[${b.id}].themeId`);
  }
  for (const t of D.stages.themes || []) {
    need(bossIds, t.bossId, `stages.themes[${t.id}].bossId`);
    for (const r of t.roster || []) need(archIds, r.archetypeId, `stages.themes[${t.id}].roster.archetypeId`);
    (t.waves || []).forEach((w, i) => {
      need(archIds, w.archetypeId, `stages.themes[${t.id}].waves[${i}].archetypeId`);
      need(formIds, w.formationId, `stages.themes[${t.id}].waves[${i}].formationId`);
    });
  }
  need(weaponIds, D.rules.player && D.rules.player.startWeaponId, 'rules.player.startWeaponId');
  for (const id of (D.rules.boss && D.rules.boss.midBossSummonsAllowed) || []) {
    need(bossIds, id, 'rules.boss.midBossSummonsAllowed');
  }
  for (const t of D.stages.themeDraw ? D.stages.themeDraw.pool || [] : []) {
    need(themeIds, t, 'stages.themeDraw.pool');
  }
  need(themeIds, D.stages.themeDraw && D.stages.themeDraw.finalStageId, 'stages.themeDraw.finalStageId');
}

// ===========================================================================
//  S3 — 어휘 (동결 목록 밖의 값 = 실패)
// ===========================================================================
function S3_vocab() {
  for (const a of D.enemies.archetypes || []) {
    vocab('S3', a.moveId, MOVE_IDS, `enemies.archetypes[${a.id}].moveId`);
    vocab('S3', a.shapeId, SHAPE_IDS, `enemies.archetypes[${a.id}].shapeId`);
    vocab('S3', a.band, ['chaff', 'line', 'turret', 'bruiser'], `enemies.archetypes[${a.id}].band`);
  }
  for (const e of D.enemies.emitters || []) vocab('S3', e.type, EMITTER_TYPES, `enemies.emitters[${e.id}].type`);
  for (const b of D.bullets.bullets || []) {
    vocab('S3', b.shape, BULLET_SHAPES, `bullets[${b.id}].shape`);
    if (!isAmb(b.status) && !BULLET_STATUS.includes(b.status)) {
      V('S3', `bullets[${b.id}].status = ${JSON.stringify(b.status)} — 허용 = null | "slow" | "stun" (§9.7)`);
    }
  }
  for (const w of D.weapons.weapons || []) vocab('S3', w.family, FAMILIES, `weapons[${w.id}].family`);
  for (const p of D.passives.passives || []) vocab('S3', p.stat, PASSIVE_STATS, `passives[${p.id}].stat`);
  for (const b of D.bosses.bosses || []) {
    if (b.tier === 'mid') {
      vocab('S3', b.moveId, MOVE_IDS, `bosses[${b.id}].moveId`);
      vocab('S3', b.shapeId, SHAPE_IDS, `bosses[${b.id}].shapeId`);
    } else {
      vocab('S3', b.movePattern, MOVE_PATTERNS, `bosses[${b.id}].movePattern`);
      if (isObj(b.core)) vocab('S3', b.core.shapeId, SHAPE_IDS, `bosses[${b.id}].core.shapeId`);
    }
    for (const p of b.parts || []) {
      vocab('S3', p.partType, PART_TYPES, `bosses[${b.id}].parts[${p.id}].partType`);
      vocab('S3', p.shapeId, SHAPE_IDS, `bosses[${b.id}].parts[${p.id}].shapeId`);
    }
  }
  for (const t of D.stages.themes || []) {
    (t.waves || []).forEach((w, i) => {
      vocab('S3', w.formationId, FORMATION_IDS, `stages.themes[${t.id}].waves[${i}].formationId`);
      vocab('S3', w.spawnEdge, SPAWN_EDGES, `stages.themes[${t.id}].waves[${i}].spawnEdge`);
    });
  }
  // §9.6 stats 어휘 = 12종 폐쇄
  const stats = D.passives.stats || [];
  for (const s of stats) vocab('S3', s, PASSIVE_STATS, 'passives.stats');
}

// ===========================================================================
//  S4 — 아키타입 겹침 금지 (§8.6)
//  (moveId, emitterType) 쌍 중복 시 실패. band 가 다르면 허용
// ===========================================================================
function S4_archetypeOverlap() {
  const emitById = new Map((D.enemies.emitters || []).map((e) => [e.id, e]));
  const seen = new Map();
  for (const a of D.enemies.archetypes || []) {
    const et = a.attack && isObj(a.attack) ? (emitById.get(a.attack.emitterId) || {}).type : null;
    const key = `${a.band}|${a.moveId}|${et === undefined ? 'UNRESOLVED' : et}`;
    if (seen.has(key)) {
      V('S4', `아키타입 겹침: "${a.id}" 와 "${seen.get(key)}" 가 같은 (band=${a.band}, moveId=${a.moveId}, emitterType=${et}) — §8.6/S4`);
    } else seen.set(key, a.id);
  }
}

// ===========================================================================
//  S5 — 보스 R1~R7 전부 + partCount + armor 수 + tier:"final" 3중 동치 (§8.14)
// ===========================================================================
function S5_bossRules() {
  const rb = D.rules.boss;
  if (!isObj(rb)) return;
  const themeById = new Map((D.stages.themes || []).map((t) => [t.id, t]));
  const FINAL_ID = D.stages.themeDraw ? D.stages.themeDraw.finalStageId : 'finale';
  const finaleTheme = themeById.get(FINAL_ID);
  const exempt = new Set((rb.finale && rb.finale.exemptRules) || []);

  for (const b of D.bosses.bosses || []) {
    if (!isObj(b) || b.tier === 'mid') continue;
    const isFinal = b.tier === 'final';
    const tag = `bosses[${b.id}]`;
    const parts = Array.isArray(b.parts) ? b.parts : [];
    const armor = parts.filter((p) => p.partType === 'armor');
    const periph = parts;                              // 주변부 = core 를 뺀 전부
    const theme = isAmb(b.themeId) ? null : themeById.get(b.themeId);
    const themeEl = theme ? theme.element : null;

    // partCount — core 를 포함한다 (§13.6.2 "armor 2 + 선택 1 + core 1 = partCount 4")
    const wantPartCount = isFinal ? (rb.finale && rb.finale.partCount) : rb.partCount;
    if (num(wantPartCount) && parts.length + 1 !== wantPartCount) {
      V('S5', `${tag}: partCount = ${parts.length + 1}(부위 ${parts.length} + core) ≠ ${wantPartCount} (§8.11/§8.16)`);
    }

    // R1: core 속성 = 항상 노말
    if (isObj(b.core) && b.core.element !== rb.coreElement) {
      V('S5', `${tag}: R1 위반 — core.element = "${b.core.element}" ≠ "${rb.coreElement}" (§8.14)`);
    }
    // core 는 정확히 1개 (§8.12) — parts[] 에 core 가 있으면 안 된다
    if (parts.some((p) => p.partType === 'core')) {
      V('S5', `${tag}: parts[] 에 partType "core" — core 는 bosses[].core 가 유일한 자리 (§9.8)`);
    }

    // R2: 주변부 속성에 노말 금지 (최종의 allowNormalPeripheral 은 왕좌에만)
    const allowNormalPeriph = isFinal && rb.finale && rb.finale.allowNormalPeripheral === true;
    const normalPeriph = periph.filter((p) => p.element === 'normal');
    if (rb.partNormalForbidden === true) {
      if (!allowNormalPeriph && normalPeriph.length > 0) {
        V('S5', `${tag}: R2 위반 — 주변부 노말 [${normalPeriph.map((p) => p.id).join(', ')}] (§8.14)`);
      } else if (allowNormalPeriph && normalPeriph.length > 1) {
        V('S5', `${tag}: R2 — finale.allowNormalPeripheral 은 왕좌 1개만 면제인데 노말 주변부가 ${normalPeriph.length}개 (§8.16)`);
      } else if (allowNormalPeriph && normalPeriph.some((p) => p.partType === 'armor')) {
        V('S5', `${tag}: R2 — 노말 면제는 armament(왕좌)에만. armor 가 노말이면 게이트에 상성이 없다 (§8.16)`);
      }
    }

    // R3: 주변부는 서로 다른 속성 ≥ 2종
    if (!exempt.has('R3') || !isFinal) {
      const distinct = new Set(periph.map((p) => p.element).filter((e) => !isAmb(e)));
      if (num(rb.partElementDistinctMin) && distinct.size < rb.partElementDistinctMin) {
        V('S5', `${tag}: R3 위반 — 주변부 distinct 속성 ${distinct.size} < ${rb.partElementDistinctMin} (§8.14)`);
      }
    }

    // R4: 테마 속성은 최대 1개 부위 (최종 면제)
    if (!(isFinal && exempt.has('R4'))) {
      if (themeEl === null || themeEl === undefined) {
        if (!isFinal) C('S5', `${tag}: R4 를 평가할 테마 속성이 없다 (themeId="${b.themeId}")`);
      } else if (num(rb.partThemeElementMax)) {
        const n = periph.filter((p) => p.element === themeEl).length;
        if (n > rb.partThemeElementMax) {
          V('S5', `${tag}: R4 위반 — 테마 속성(${themeEl}) 부위 ${n} > ${rb.partThemeElementMax} (§8.14)`);
        }
      }
    }

    // R5: armor 부위 속성 ≠ 테마 속성 (최종은 테마 없음 → 공허참)
    if (rb.armorElementNotTheme === true && themeEl) {
      const bad = armor.filter((p) => p.element === themeEl);
      if (bad.length) V('S5', `${tag}: R5 위반 — armor [${bad.map((p) => p.id).join(', ')}] 속성 = 테마(${themeEl}) (§8.14)`);
    }

    // R6: armor 수 = 2 (최종은 finale.armorPartCount = 3, exemptRules)
    if (isFinal && exempt.has('R6')) {
      const want = rb.finale && rb.finale.armorPartCount;
      if (num(want) && armor.length !== want) {
        V('S5', `${tag}: R6(finale) 위반 — armor ${armor.length} ≠ finale.armorPartCount ${want} (§8.16)`);
      }
    } else if (Array.isArray(rb.armorPartCountRange)) {
      const [lo, hi] = rb.armorPartCountRange;
      if (armor.length < lo || armor.length > hi) {
        V('S5', `${tag}: R6 위반 — armor ${armor.length} ∉ armorPartCountRange [${lo}, ${hi}] (§8.13.2)`);
      }
    }

    // R7: φ ∈ [0.85·B, B), B = 0.4^-a − 1.  ★ R7 은 면제하지 않는다 (§8.16)
    const a = armor.length;
    const phi = b.armorCoreRatio;
    if (num(phi) && a > 0 && num(rb.coreGateMul) && Array.isArray(rb.armorCoreRatioBandPct)) {
      const B = Math.pow(rb.coreGateMul, -a) - 1;
      const lo = rb.armorCoreRatioBandPct[0] * B;
      const hi = rb.armorCoreRatioBandPct[1] * B;
      const okLo = phi >= lo - 1e-9;
      const okHi = rb.armorCoreRatioBandPct[1] === 1.0 ? phi < B : phi <= hi + 1e-9;
      if (!okLo || !okHi) {
        V('S5', `${tag}: R7 위반 — φ = ${phi} ∉ [${lo.toFixed(3)}, ${B.toFixed(3)}) (a=${a}, B=0.4^-${a}−1) (§8.13.1)`);
      }
    } else if (!num(phi) && !isAmb(phi)) {
      V('S5', `${tag}.armorCoreRatio: tier ∈ {stage, final} 에 필수 (§9.8)`);
    }

    // tier == "final" ⟺ finale 스테이지 전용 ⟺ bossHpScale 미적용
    if (isFinal) {
      if (!finaleTheme) {
        C('S5', `tier:"final" 보스 "${b.id}" 가 있으나 stages.themes 에 finalStageId("${FINAL_ID}") 엔트리가 없다`);
      } else if (finaleTheme.bossId !== b.id) {
        V('S5', `${tag}: tier:"final" 인데 finale.bossId = "${finaleTheme.bossId}" — 3중 동치 위반 (S5)`);
      }
    } else if (finaleTheme && finaleTheme.bossId === b.id) {
      V('S5', `${tag}: finale 의 보스인데 tier = "${b.tier}" ≠ "final" — 3중 동치 위반 (S5)`);
    }
    // 테마 보스가 finale 테마를 참조하면 안 된다
    if (!isFinal && b.themeId === FINAL_ID) {
      V('S5', `${tag}: tier:"stage" 인데 themeId = finale (S5)`);
    }
  }
}

// ===========================================================================
//  S6 — 공정성 (§12.4 · §7.4)
//  ★ enemies.json > emitters 만 검사한다 (fairness.playerWeaponsExempt, §9.5)
// ===========================================================================
function S6_fairness() {
  const f = D.rules.fairness;
  if (!isObj(f)) return;
  if (f.playerWeaponsExempt !== true) {
    V('S6', `rules.fairness.playerWeaponsExempt = ${f.playerWeaponsExempt} ≠ true — §9.5 "플레이어 무기는 fairness 의 대상이 아니다"`);
  }
  const bulletById = new Map((D.bullets.bullets || []).map((b) => [b.id, b]));

  // 이미터 → 소유 개체 역인덱스 (§7.4의 "중간보스 패턴" 하한을 평가하기 위해)
  const ownerOf = new Map();
  for (const b of D.bosses.bosses || []) {
    const push = (id, kind) => { if (!isAmb(id)) ownerOf.set(id, kind); };
    if (b.tier === 'mid') for (const ps of b.patternSet || []) for (const id of ps.emitterIds || []) push(id, 'mid');
    for (const p of b.parts || []) for (const ps of p.patternSet || []) for (const id of ps.emitterIds || []) push(id, 'boss');
  }

  for (const e of D.enemies.emitters || []) {
    if (!isObj(e)) continue;
    const tag = `enemies.emitters[${e.id}]`;
    const bul = e.bulletId === null ? null : bulletById.get(e.bulletId);

    // (1) telegraphSec — 절대 하한 + 거동별 표 (§7.4) + 소유 개체별 하한
    if (num(e.telegraphSec)) {
      const floors = [];
      if (num(f.minTelegraphSec)) floors.push([f.minTelegraphSec, 'fairness.minTelegraphSec']);
      if (TELEGRAPH_FLOOR_BY_TYPE[e.type] !== undefined) floors.push([TELEGRAPH_FLOOR_BY_TYPE[e.type], `§7.4 거동표(${e.type})`]);
      if (bul && bul.status === 'slow') floors.push([TELEGRAPH_FLOOR_SLOW_BULLET, '§7.4 상태이상(slow) 탄']);
      if (bul && bul.status === 'stun' && num(f.minStunTelegraphSec)) floors.push([f.minStunTelegraphSec, 'fairness.minStunTelegraphSec']);
      if (ownerOf.get(e.id) === 'mid') floors.push([TELEGRAPH_FLOOR_MIDBOSS, '§7.4 중간보스 패턴']);
      for (const [v, why] of floors) {
        if (e.telegraphSec < v - 1e-9) {
          V('S6', `${tag}.telegraphSec = ${e.telegraphSec} < ${v} (${why})`);
        }
      }
    }

    // (2) 탄 속도 — 조준탄은 별도 상한 (§12.4)
    const speeds = [];
    if (num(e.speed)) speeds.push(['emitter.speed', e.speed]);
    if (bul && num(bul.speed)) speeds.push([`bullets[${bul.id}].speed`, bul.speed]);
    for (const [where, sp] of speeds) {
      if (num(f.maxBulletSpeed) && sp > f.maxBulletSpeed) {
        V('S6', `${tag}: ${where} = ${sp} > fairness.maxBulletSpeed(${f.maxBulletSpeed})`);
      }
      if (e.type === 'aimed' && num(f.maxAimedBulletSpeed) && sp > f.maxAimedBulletSpeed) {
        V('S6', `${tag}: 조준탄 ${where} = ${sp} > fairness.maxAimedBulletSpeed(${f.maxAimedBulletSpeed})`);
      }
    }

    // (3) 최소 탄 반경 (§12.4)
    if (bul && num(bul.radius) && num(f.minBulletRadiusPx) && bul.radius < f.minBulletRadiusPx) {
      V('S6', `${tag} → bullets[${bul.id}].radius = ${bul.radius} < fairness.minBulletRadiusPx(${f.minBulletRadiusPx})`);
    }

    // (4) wall 의 통과 틈 (§12.4)
    if (e.type === 'wall') {
      if (num(e.gapWidthPx) && num(f.minGapWidthPx) && e.gapWidthPx < f.minGapWidthPx) {
        V('S6', `${tag}.gapWidthPx = ${e.gapWidthPx} < fairness.minGapWidthPx(${f.minGapWidthPx})`);
      }
      if (num(e.gapCount) && e.gapCount < 1) V('S6', `${tag}.gapCount = ${e.gapCount} — 틈 없는 벽은 회피 불가`);
    }

    // (5) 스턴 최대 지속 (§12.4)
    if (bul && bul.status === 'stun' && num(bul.statusDurationSec) && num(f.maxStunSec)
        && bul.statusDurationSec > f.maxStunSec) {
      V('S6', `${tag} → bullets[${bul.id}].statusDurationSec = ${bul.statusDurationSec} > fairness.maxStunSec(${f.maxStunSec})`);
    }
  }

  // moveId: charge 의 windUpSec (§8.4 "charge.windUpSec ≥ fairness.minTelegraphSec")
  const checkWind = (mp, tag) => {
    if (isObj(mp) && num(mp.windUpSec) && num(f.minTelegraphSec) && mp.windUpSec < f.minTelegraphSec) {
      V('S6', `${tag}.windUpSec = ${mp.windUpSec} < fairness.minTelegraphSec(${f.minTelegraphSec}) — §8.4`);
    }
  };
  for (const a of D.enemies.archetypes || []) if (a.moveId === 'charge') checkWind(a.moveParams, `enemies.archetypes[${a.id}].moveParams`);
  for (const b of D.bosses.bosses || []) if (b.moveId === 'charge') checkWind(b.moveParams, `bosses[${b.id}].moveParams`);

  // fairness.minSpawnRadiusPx — 런타임 규칙이며 정적으로 검사할 대상이 없다
  S('S6', `fairness.minSpawnRadiusPx(${f.minSpawnRadiusPx}) = "적 탄은 플레이어 반경 N px 이내에서 생성 불가"는 `
    + `발사 시점의 플레이어 위치에 의존한다 → 런타임 어서션. TODO: step.js 의 탄 생성 경로에 assert + `
    + `sim 의 certify.fairnessViolations 가 카운트 (§13.1 static.fairnessViolations.max = 0)`);
}

// ===========================================================================
//  S7 — 동시 텔레그래프 (개체당 ≤ telegraphConcurrentMaxPerEntity)
//  보스 patternSet 을 3페이즈 전부 전개해 정적 검사 (§12.4 · §8.9-R8)
// ===========================================================================
function gcd(a, b) { while (b) { const t = a % b; a = b; b = t; } return a; }
function lcm(a, b) { return (a / gcd(a, b)) * b; }

/**
 * 이미터 집합의 동시 텔레그래프 최대치를 주기 위에서 스캔한다.
 * 모델: 발사 시각 t_k = offsetSec + k·everySec, 텔레그래프 창 = [t_k − telegraphSec, t_k)
 * (repeat/restSec 의 의미가 정본에 없으므로 단순 주기 모델 — CANON 신고 참조)
 */
function maxConcurrentTelegraphs(emitters) {
  // 값이 하나라도 확정되지 않았으면 전개하지 않는다 (발명 금지)
  if (emitters.some((e) => !e || !num(e.everySec) || e.everySec <= 0
      || !num(e.telegraphSec) || !num(e.offsetSec))) {
    return { max: 0, unresolved: true };
  }
  const list = emitters;
  if (!list.length) return { max: 0, unresolved: false };
  // 주기 = everySec 들의 LCM (센티초 정수화). 상한 60 게임초에서 절단
  const cs = list.map((e) => Math.round(e.everySec * 100));
  let P = cs.reduce((x, y) => lcm(x, y), 1);
  if (P > 6000) P = 6000;
  let mx = 0;
  for (let t = 0; t < P; t += 1) {           // 0.01 게임초 스텝
    const now = t / 100;
    let n = 0;
    for (const e of list) {
      const per = e.everySec;
      // now 기준 다음 발사까지 남은 시간
      let phase = (now - e.offsetSec) % per;
      if (phase < 0) phase += per;
      const untilFire = per - phase;         // (0, per]
      if (untilFire <= e.telegraphSec) n += 1;
    }
    if (n > mx) mx = n;
  }
  return { max: mx, unresolved: false };
}

function S7_concurrentTelegraphs() {
  const cap = D.rules.fairness && D.rules.fairness.telegraphConcurrentMaxPerEntity;
  if (!num(cap)) return;
  const emitById = new Map((D.enemies.emitters || []).map((e) => [e.id, e]));
  const resolve_ = (ids) => (ids || []).map((id) => (isAmb(id) ? null : emitById.get(id)));

  for (const b of D.bosses.bosses || []) {
    if (b.tier === 'mid') {
      for (const [i, ps] of (b.patternSet || []).entries()) {
        const es = resolve_(ps.emitterIds);
        if (es.some((e) => !e)) { continue; }   // 모호/미해결 → AMBIGUOUS 집계에 이미 잡힌다
        const { max, unresolved } = maxConcurrentTelegraphs(es);
        if (unresolved) continue;
        if (max > cap) V('S7', `bosses[${b.id}].patternSet[${i}]: 동시 텔레그래프 ${max} > ${cap} (§12.4)`);
      }
      continue;
    }
    for (const p of b.parts || []) {
      for (const [ph, ps] of (p.patternSet || []).entries()) {
        const es = resolve_(ps.emitterIds);
        if (es.some((e) => !e)) continue;
        const { max, unresolved } = maxConcurrentTelegraphs(es);
        if (unresolved) continue;
        if (max > cap) V('S7', `bosses[${b.id}].parts[${p.id}].patternSet[${ph}] (페이즈 ${ph + 1}): 동시 텔레그래프 ${max} > ${cap} (§12.4)`);
      }
    }
  }
  // 잡몹 = 이미터 1개 → 자기 자신과의 중첩만 가능 (telegraphSec > everySec 이면 발화)
  for (const a of D.enemies.archetypes || []) {
    if (!isObj(a.attack)) continue;
    const e = emitById.get(a.attack.emitterId);
    if (!e) continue;
    const { max, unresolved } = maxConcurrentTelegraphs([e]);
    if (!unresolved && max > cap) {
      V('S7', `enemies.archetypes[${a.id}] → ${e.id}: 동시 텔레그래프 ${max} > ${cap} (telegraphSec ${e.telegraphSec} vs everySec ${e.everySec})`);
    }
  }
}

// ===========================================================================
//  S8 — 혼합 비율 (§8.2 · §8.2.1)
//  저작 리스트(stages[].waves[] 전량)의 개체 수 기준 속성 비율이 mix 에 ±3%p
//  + mix 가 counter/prey 규칙(70/10/10/10)을 따르는지
// ===========================================================================
const MIX_TOL_PP = 3.0;   // §8.2.1 "허용 오차 ±3%p (저작 리스트 대비)"

function counterOf(el) {   // matrix[c][el] == 2.0 인 c
  const m = D.elements.matrix || {};
  for (const c of Object.keys(m)) if (m[c] && m[c][el] === 2.0) return c;
  return null;
}
function preyOf(el) {      // matrix[el][p] == 2.0 인 p
  const m = D.elements.matrix || {};
  const row = m[el] || {};
  for (const p of Object.keys(row)) if (row[p] === 2.0) return p;
  return null;
}

function S8_mix() {
  const FINAL_ID = D.stages.themeDraw ? D.stages.themeDraw.finalStageId : 'finale';
  for (const t of D.stages.themes || []) {
    if (!isObj(t) || !isObj(t.mix)) continue;
    const tag = `stages.themes[${t.id}]`;

    // mix 는 4속성 실제 키로 전개된 가중치 맵 하나 (§8.2)
    closedKeys('S8', t.mix, ['normal', 'fire', 'water', 'grass'], `${tag}.mix`);
    const sum = Object.values(t.mix).filter(num).reduce((x, y) => x + y, 0);
    if (Math.abs(sum - 1.0) > 1e-6) V('S8', `${tag}.mix: 합 ${sum} ≠ 1.0`);

    // (1) 저작 리스트의 개체 수 기준 비율 vs mix — ±3%p
    const cnt = { normal: 0, fire: 0, water: 0, grass: 0 };
    let tot = 0;
    let skipped = 0;
    for (const w of t.waves || []) {
      if (!num(w.count) || isAmb(w.element)) { skipped += 1; continue; }
      if (!has(cnt, w.element)) continue;
      cnt[w.element] += w.count; tot += w.count;
    }
    if (skipped) A(`${tag}.waves`, `${skipped}개 웨이브가 모호값을 포함해 S8 분모에서 빠졌다`);
    if (tot > 0) {
      for (const el of ['normal', 'fire', 'water', 'grass']) {
        const actualPp = (cnt[el] / tot) * 100;
        const wantPp = (t.mix[el] || 0) * 100;
        if (Math.abs(actualPp - wantPp) > MIX_TOL_PP + 1e-9) {
          V('S8', `${tag}: 저작 리스트 ${el} = ${actualPp.toFixed(2)}%p vs mix ${wantPp.toFixed(2)}%p — |Δ| ${Math.abs(actualPp - wantPp).toFixed(2)} > ${MIX_TOL_PP}%p (§8.2.1)`);
        }
      }
    }

    // (2) mix 가 counter/prey 규칙을 따르는지 — 테마 속성이 있는 테마만 (§8.2)
    if (t.id === FINAL_ID || t.element === null) continue;   // §8.16: 최종은 테마 속성이 없다
    const T = t.element, c = counterOf(T), p = preyOf(T);
    if (!c || !p) { C('S8', `${tag}: counter/prey 를 elements.matrix 에서 유도할 수 없다 (element="${T}")`); continue; }
    const want = { [T]: 0.70, [c]: 0.10, [p]: 0.10, normal: 0.10 };
    for (const el of ['normal', 'fire', 'water', 'grass']) {
      if (Math.abs((t.mix[el] || 0) - want[el]) > 1e-9) {
        V('S8', `${tag}.mix.${el} = ${t.mix[el]} ≠ ${want[el]} — 70/10/10/10 규칙 (T=${T}, counter=${c}, prey=${p}) (§8.2)`);
      }
    }
  }
}

// ===========================================================================
//  S9 — 구조 (§13.4)
// ===========================================================================
function S9_structure() {
  const st = D.stages, pl = D.rules.player;
  const FINAL_ID = st.themeDraw ? st.themeDraw.finalStageId : 'finale';
  const archById = new Map((D.enemies.archetypes || []).map((a) => [a.id, a]));

  // (1) stages[].element ∈ {water, fire, grass, null} 이고 null 은 finale 만
  for (const t of st.themes || []) {
    const ok = ['water', 'fire', 'grass'].includes(t.element) || t.element === null;
    if (!ok) V('S9', `stages.themes[${t.id}].element = ${JSON.stringify(t.element)} — 허용 = water|fire|grass|null (§8.16)`);
    if (t.element === null && t.id !== FINAL_ID) {
      V('S9', `stages.themes[${t.id}].element = null 인데 finale 가 아니다 (§8.16)`);
    }
    if (t.id === FINAL_ID && t.element !== null) {
      V('S9', `stages.themes[${FINAL_ID}].element = ${JSON.stringify(t.element)} ≠ null (§8.16)`);
    }
  }

  // (2) 무기 levels 정확히 8행 (§9.5)
  for (const w of D.weapons.weapons || []) {
    if (!Array.isArray(w.levels)) { V('S9', `weapons[${w.id}].levels: 배열이 아니다`); continue; }
    if (w.levels.length !== 8) V('S9', `weapons[${w.id}].levels: ${w.levels.length}행 ≠ 8행 (§9.5)`);
  }

  // (3) 4 ≤ elementCapTotal < 3 × elementCapPerElement (§4.2)
  if (isObj(pl) && num(pl.elementCapTotal) && num(pl.elementCapPerElement)) {
    if (!(pl.elementCapTotal >= 4 && pl.elementCapTotal < 3 * pl.elementCapPerElement)) {
      V('S9', `player.elementCapTotal(${pl.elementCapTotal}) ∉ [4, 3 × elementCapPerElement(${pl.elementCapPerElement}) = ${3 * pl.elementCapPerElement}) (§4.2)`);
    }
  }

  // (4) rearIn / spawnEdge:"bottom" 은 rearSpawnAllowed[stage] 일 때만 (§8.4 · §8.7)
  const rear = (st.curve && st.curve.rearSpawnAllowed) || [];
  const firstRearStage = rear.findIndex((v) => v === true) + 1;   // 1-indexed. 0 = 영영 불가
  for (const t of st.themes || []) {
    const unlock = new Map((t.roster || []).map((r) => [r.archetypeId, r.unlockStageMin]));
    (t.waves || []).forEach((w, i) => {
      const a = archById.get(w.archetypeId);
      if (!a) return;
      const needsRear = a.moveId === 'rearIn' || w.spawnEdge === 'bottom';
      if (!needsRear) return;
      const u = unlock.get(w.archetypeId);
      const tag = `stages.themes[${t.id}].waves[${i}] (${w.archetypeId}, moveId=${a.moveId}, spawnEdge=${w.spawnEdge})`;
      if (isAmb(u) || !num(u)) {
        A(`${tag}.unlockStageMin`, 'rearSpawnAllowed 게이트를 평가할 unlockStageMin 이 없다');
        return;
      }
      if (firstRearStage === 0) V('S9', `${tag}: rearSpawnAllowed 가 전 스테이지 false 인데 후방 진입 (§8.4)`);
      else if (u < firstRearStage) {
        V('S9', `${tag}: unlockStageMin ${u} < 후방 스폰 최초 허용 스테이지 ${firstRearStage} — rearSpawnAllowed = [${rear.join(', ')}] (§8.4/§8.7)`);
      }
    });
  }

  // (5) 새떼에 swarm* 외 아키타입 금지 (§8.10)
  //     새떼 = 위기 세션 전용. 로스터/웨이브는 잡몹이므로 swarm* 가 있으면 안 된다
  for (const t of st.themes || []) {
    for (const r of t.roster || []) {
      if (/^swarm/.test(r.archetypeId)) V('S9', `stages.themes[${t.id}].roster: 새떼 전용 "${r.archetypeId}" 가 잡몹 로스터에 있다 (§8.10)`);
    }
    (t.waves || []).forEach((w, i) => {
      if (/^swarm/.test(w.archetypeId)) V('S9', `stages.themes[${t.id}].waves[${i}]: 새떼 전용 "${w.archetypeId}" 가 잡몹 웨이브에 있다 (§8.10)`);
    });
  }
  const swarms = (D.enemies.archetypes || []).filter((a) => /^swarm/.test(a.id));
  if (swarms.length !== 2) V('S9', `enemies.archetypes: 새떼 전용 아키타입 ${swarms.length}종 ≠ 2 (swarmChaff, swarmLancer — §8.6)`);

  // (6) crisisElementRule 준수 (최종만 로테이션) (§8.10 · §8.16)
  const ph = st.phase || {};
  if (ph.crisisElementRule !== 'themePure') {
    V('S9', `stages.phase.crisisElementRule = ${JSON.stringify(ph.crisisElementRule)} ≠ "themePure" (§8.10)`);
  }
  for (const t of st.themes || []) {
    const rot = t.finaleCrisisRotating === true;
    if (rot && t.id !== FINAL_ID) V('S9', `stages.themes[${t.id}].finaleCrisisRotating = true — 로테이션은 최종만 (§8.16)`);
    if (!rot && t.id === FINAL_ID) V('S9', `stages.themes[${FINAL_ID}].finaleCrisisRotating 이 true 가 아니다 (§8.16)`);
  }
  // §8.16: 최종의 위기 = 6서브웨이브 물×2 → 불×2 → 풀×2
  if (num(ph.crisisSubWaves) && ph.crisisSubWaves !== 6) {
    V('S9', `stages.phase.crisisSubWaves = ${ph.crisisSubWaves} ≠ 6 — §8.16 의 3속성 × 2 로테이션이 표현 불가`);
  }
}

// ===========================================================================
//  S10 — 성장 예산 (§11.1 · §13.1 static.growthBudget)
// ===========================================================================
function S10_growthBudget() {
  const g = D.meta.certify && D.meta.certify.static && D.meta.certify.static.growthBudget;
  if (!isObj(g)) return;
  closedKeys('S10', g, ['maxLevelUps', 'minTotalSink'], 'meta.certify.static.growthBudget');
  if (num(g.maxLevelUps) && num(g.minTotalSink) && !(g.maxLevelUps < g.minTotalSink)) {
    V('S10', `growthBudget: maxLevelUps(${g.maxLevelUps}) < minTotalSink(${g.minTotalSink}) 이 거짓 — §11.1 "전부 못 찍는다가 산술적으로 성립"`);
  }
  // totalSink 를 데이터에서 재유도한다 (§11.1 의 대차대조표)
  const pl = D.rules.player;
  if (isObj(pl) && num(D.passives.maxLevel)) {
    const newWeapon = pl.weaponSlots - 1;                 // 시작 무기 1 지급 → 4칸 중 3칸
    const weaponLevel = pl.weaponSlots * 7;               // 4무기 × Lv1→8 = 7픽 × 4
    const elementLevel = pl.elementCapTotal;              // elementCapTotal
    const passive = pl.passiveSlots * D.passives.maxLevel; // 6칸 × Lv5 (획득 픽 포함)
    const derived = newWeapon + weaponLevel + elementLevel + passive;
    if (num(g.minTotalSink) && derived !== g.minTotalSink) {
      V('S10', `growthBudget.minTotalSink = ${g.minTotalSink} ≠ 데이터 유도값 ${derived} `
        + `(무기 ${newWeapon} + 레벨 ${weaponLevel} + 속성 ${elementLevel} + 패시브 ${passive}) — §11.1`);
    }
  }
  S('S10', `S13.4-S10 문면("XP 곡선으로 계산한 최대 레벨업 횟수 < totalSink")의 좌변은 XP 곡선만으로 나오지 않는다 `
    + `— 스테이지별 총 XP는 웨이브 편성·처치율·farm 정책의 함수다. 정본이 그 좌변을 상수로 소유한다`
    + `(certify.static.growthBudget.maxLevelUps = ${g.maxLevelUps}) → 위 비교가 정적 검사의 전부이고, `
    + `TODO: sim(run 모드)이 실측 maxLevelUps 를 report/summary.json 에 출력해 이 상수를 교정한다`);
}

// ===========================================================================
//  S11 — RNG 스트림 (§10.2) — 명명된 8 스트림만, 스트림 간 공유 금지
// ===========================================================================
function S11_rngStreams() {
  if (!existsSync(SRC_DIR)) {
    SKIP('S11', `src/ 가 아직 없다 → RNG 스트림 검사 건너뜀. 동결 목록 = [${RNG_STREAMS.join(', ')}] (8종, §10.2)`);
    return;
  }
  const files = listJsFiles(SRC_DIR);
  const coreDir = join(SRC_DIR, 'core');
  const weaponsDir = join(coreDir, 'weapons');
  for (const f of files) {
    const rel = relative(ROOT, f);
    const raw = readFileSync(f, 'utf8');
    // stream(masterSeed, "name") 호출의 name 을 뽑는다
    for (const m of raw.matchAll(/\bstream\s*\(\s*[^,]+,\s*['"]([^'"]+)['"]\s*\)/g)) {
      const name = m[1];
      if (!RNG_STREAMS.includes(name)) {
        V('S11', `${rel}: 미등록 RNG 스트림 "${name}" — 동결 8종 = [${RNG_STREAMS.join(', ')}] (§10.2)`);
      }
    }
    // rng.pattern 만 적·플레이어 양쪽 접근 허용 (§9.5 · §10.2)
    for (const m of raw.matchAll(/\brng\s*\.\s*([A-Za-z]+)/g)) {
      const s = m[1];
      if (!RNG_STREAMS.includes(s)) continue;
      const inWeapons = !relative(weaponsDir, f).startsWith('..');
      if (inWeapons && s !== 'pattern') {
        V('S11', `${rel}: weapons/** 가 rng.${s} 에 접근 — 플레이어 무기에 허용된 스트림은 rng.pattern 뿐 (§9.5/§10.2 S11)`);
      }
    }
  }
}

// ===========================================================================
//  S12 — 2층 캡 (§12.1) — A층 오써링 예산 < B층 안전망 캡 + 파생값 무결성
// ===========================================================================
function S12_twoLayerCaps() {
  const f = D.rules.fairness, caps = D.rules.caps;
  if (!isObj(f) || !isObj(caps)) return;

  // 파생값 무결성: telegraphConcurrentMaxGlobal == enemyConcurrentMax × telegraphConcurrentMaxPerEntity
  if (num(f.telegraphConcurrentMaxGlobal) && num(f.enemyConcurrentMax) && num(f.telegraphConcurrentMaxPerEntity)) {
    const derived = f.enemyConcurrentMax * f.telegraphConcurrentMaxPerEntity;
    if (f.telegraphConcurrentMaxGlobal !== derived) {
      V('S12', `fairness.telegraphConcurrentMaxGlobal = ${f.telegraphConcurrentMaxGlobal} ≠ enemyConcurrentMax(${f.enemyConcurrentMax}) × telegraphConcurrentMaxPerEntity(${f.telegraphConcurrentMaxPerEntity}) = ${derived} — 파생값 (§12.1)`);
    }
  }
  // A층 < B층 (전역 대 전역)
  const rows = [
    // §12.1 정정: 위기 중 = 새떼 70 + 웨이브 잔존 10 = 80 이 A층 enemies 합
    ['enemies', (f.swarmConcurrentMax || 0) + (f.crisisWaveResidualMax || 0),
      `swarmConcurrentMax(${f.swarmConcurrentMax}) + crisisWaveResidualMax(${f.crisisWaveResidualMax})`, caps.enemies],
    ['enemyBullets', f.maxSimultaneousEnemyBullets, `maxSimultaneousEnemyBullets(${f.maxSimultaneousEnemyBullets})`, caps.enemyBullets],
    ['telegraphs', f.telegraphConcurrentMaxGlobal, `telegraphConcurrentMaxGlobal(${f.telegraphConcurrentMaxGlobal})`, caps.telegraphs],
  ];
  for (const [name, a, why, b] of rows) {
    if (!num(a) || !num(b)) continue;
    if (!(a < b)) V('S12', `2층 캡 위반 — ${name}: A층 ${a} (= ${why}) < B층 caps.${name} ${b} 이 거짓 (§12.1)`);
  }
  // A층 enemyConcurrentMax 자체도 B층 아래여야 한다
  if (num(f.enemyConcurrentMax) && num(caps.enemies) && !(f.enemyConcurrentMax < caps.enemies)) {
    V('S12', `2층 캡 위반 — fairness.enemyConcurrentMax(${f.enemyConcurrentMax}) < caps.enemies(${caps.enemies}) 이 거짓 (§12.1)`);
  }
}

// ===========================================================================
//  S13 — 스턴의 거처 (§13.4)
//  status=="stun" 탄을 쓰는 이미터는 보스 부위의 patternSet[2](페이즈 3)에만.
//  스테이지당 그런 개체 ≤ statusStunMaxPerStage(2)
// ===========================================================================
function S13_stunHome() {
  const stunBullets = new Set((D.bullets.bullets || []).filter((b) => b.status === 'stun').map((b) => b.id));
  if (!stunBullets.size) return;
  const stunEmitters = new Set((D.enemies.emitters || []).filter((e) => stunBullets.has(e.bulletId)).map((e) => e.id));
  if (!stunEmitters.size) {
    SKIP('S13', `status=="stun" 탄(${[...stunBullets].join(', ')})을 참조하는 이미터가 0개 — 검사할 대상이 없다`);
    return;
  }
  const maxPerStage = D.stages.phase && D.stages.phase.statusStunMaxPerStage;

  // 합법한 자리 = 보스 부위의 patternSet[2](페이즈 3) 뿐 → 그 밖의 사용처를 전수 신고한다
  for (const a of D.enemies.archetypes || []) {
    if (isObj(a.attack) && stunEmitters.has(a.attack.emitterId)) {
      V('S13', `enemies.archetypes[${a.id}]: 스턴 이미터 "${a.attack.emitterId}" — 스턴의 거처는 보스 부위 patternSet[2] 뿐 (§13.4-S13)`);
    }
  }
  for (const b of D.bosses.bosses || []) {
    if (b.tier === 'mid') {
      for (const ps of b.patternSet || []) for (const id of ps.emitterIds || []) {
        if (stunEmitters.has(id)) V('S13', `bosses[${b.id}] (중간보스): 스턴 이미터 "${id}" — 보스 부위 patternSet[2] 전용 (§13.4-S13)`);
      }
      continue;
    }
    let stunPartsInBoss = 0;
    for (const p of b.parts || []) {
      (p.patternSet || []).forEach((ps, ph) => {
        for (const id of ps.emitterIds || []) {
          if (!stunEmitters.has(id)) continue;
          if (ph !== 2) V('S13', `bosses[${b.id}].parts[${p.id}].patternSet[${ph}]: 스턴 이미터 "${id}" — 페이즈 3(index 2) 에만 (§13.4-S13)`);
          else stunPartsInBoss += 1;
        }
      });
    }
    if (num(maxPerStage) && stunPartsInBoss > maxPerStage) {
      V('S13', `bosses[${b.id}]: 스턴 이미터를 가진 개체 ${stunPartsInBoss} > statusStunMaxPerStage(${maxPerStage}) (§13.4-S13)`);
    }
  }
  // (3) §9.7: 스턴은 difficulty.stunMinDifficulty 이상에서만
  const smd = D.meta.difficulty && D.meta.difficulty.stunMinDifficulty;
  if (smd && D.meta.difficulty && !has(D.meta.difficulty, smd)) {
    V('S13', `meta.difficulty.stunMinDifficulty = "${smd}" 가 난이도 목록에 없다 (§9.7)`);
  }
}

// ===========================================================================
//  S14 — shape ↔ status 동치 (§9.7)
//  (bullets[].status === null) === (bullets[].shape === "circle")
// ===========================================================================
function S14_shapeStatusEquiv() {
  for (const b of D.bullets.bullets || []) {
    if (isAmb(b.status) || isAmb(b.shape)) continue;
    const l = b.status === null, r = b.shape === 'circle';
    if (l !== r) {
      V('S14', `bullets[${b.id}]: (status === null)=${l} ≠ (shape === "circle")=${r} `
        + `— status=${JSON.stringify(b.status)}, shape=${JSON.stringify(b.shape)}. §7.4 "육각 = 상태이상"이 거짓말이 된다`);
    }
  }
}

// ===========================================================================
//  S15 — 중간보스 속성 주입 (§8.9) : tier == "mid" ⟺ element == null
// ===========================================================================
function S15_midBossElement() {
  for (const b of D.bosses.bosses || []) {
    if (!isObj(b)) continue;
    const isMid = b.tier === 'mid';
    const nullEl = has(b, 'element') && b.element === null;
    if (isMid && !nullEl) {
      V('S15', `bosses[${b.id}]: tier=="mid" 인데 element 가 null 이 아니다 (${JSON.stringify(b.element)}) — 주입 대상 표식 (§8.9)`);
    }
    if (!isMid && has(b, 'element')) {
      V('S15', `bosses[${b.id}]: tier=="${b.tier}" 인데 최상위 element 키가 있다 — null 허용은 tier:"mid" 뿐 (§8.9/S15)`);
    }
  }
  const rule = D.stages.phase && D.stages.phase.midBossElementRule;
  if (rule !== 'notThemeAndNotNormal') {
    V('S15', `stages.phase.midBossElementRule = ${JSON.stringify(rule)} ≠ "notThemeAndNotNormal" (§8.9)`);
  }
}

// ===========================================================================
//  S16 — patternSet 길이 (§8.9-R8 · §9.8)
//  보스 부위 = emitterIds 길이 1 강제 / 중간보스 = 1~2
//  + patternSet 길이 = 페이즈 수(3), 중간보스 = 1
// ===========================================================================
function S16_patternSetLen() {
  const phases = Array.isArray(D.rules.boss && D.rules.boss.phaseThresholds)
    ? D.rules.boss.phaseThresholds.length + 1 : 3;
  for (const b of D.bosses.bosses || []) {
    if (!isObj(b)) continue;
    if (b.tier === 'mid') {
      const ps = b.patternSet || [];
      if (ps.length !== 1) V('S16', `bosses[${b.id}] (중간보스): patternSet 길이 ${ps.length} ≠ 1 (§9.8)`);
      ps.forEach((e, i) => {
        const n = Array.isArray(e.emitterIds) ? e.emitterIds.length : -1;
        if (n < 1 || n > 2) V('S16', `bosses[${b.id}].patternSet[${i}].emitterIds: 길이 ${n} ∉ [1, 2] (§8.9-R8)`);
      });
      // 중간보스는 parts: [] (§8.9 "단일 몸체. 부위 없음")
      if (!Array.isArray(b.parts) || b.parts.length !== 0) {
        V('S16', `bosses[${b.id}] (중간보스): parts 가 [] 가 아니다 — "단일 몸체. 부위 없음" (§8.9)`);
      }
      continue;
    }
    for (const p of b.parts || []) {
      const ps = p.patternSet || [];
      if (ps.length !== phases) {
        V('S16', `bosses[${b.id}].parts[${p.id}]: patternSet 길이 ${ps.length} ≠ 페이즈 수 ${phases} (§9.8)`);
      }
      ps.forEach((e, i) => {
        const n = Array.isArray(e.emitterIds) ? e.emitterIds.length : -1;
        if (n !== 1) V('S16', `bosses[${b.id}].parts[${p.id}].patternSet[${i}].emitterIds: 길이 ${n} ≠ 1 — 보스 부위는 1 강제 (§8.9-R8/S16)`);
      });
    }
  }
}

// ===========================================================================
//  S17 — 소환 (§8.9-R9)
//  summon != null ⟺ (tier == "mid" 그리고 id ∈ boss.midBossSummonsAllowed)
// ===========================================================================
function S17_summon() {
  const allowed = new Set((D.rules.boss && D.rules.boss.midBossSummonsAllowed) || []);
  for (const b of D.bosses.bosses || []) {
    if (!isObj(b)) continue;
    const lhs = has(b, 'summon') && b.summon !== null && !isAmb(b.summon);
    const rhs = b.tier === 'mid' && allowed.has(b.id);
    if (lhs !== rhs) {
      V('S17', `bosses[${b.id}]: (summon != null)=${lhs} ≠ (tier=="mid" ∧ id ∈ midBossSummonsAllowed)=${rhs} `
        + `— tier=${b.tier}, midBossSummonsAllowed=[${[...allowed].join(', ')}] (§8.9-R9/S17)`);
    }
  }
  // §8.11: 스테이지 보스는 소환하지 않는다
  if (D.rules.boss && D.rules.boss.summonsAllowed !== false) {
    V('S17', `rules.boss.summonsAllowed = ${D.rules.boss.summonsAllowed} ≠ false (§8.11)`);
  }
}

// ===========================================================================
//  S18 — mobility 의 진실성 (§8.12.1)
//  movePattern == "holdCenter" 인 보스는 mobility 부위 금지
// ===========================================================================
function S18_mobilityTruth() {
  for (const b of D.bosses.bosses || []) {
    if (!isObj(b) || b.tier === 'mid') continue;
    if (b.movePattern !== 'holdCenter') continue;
    const mob = (b.parts || []).filter((p) => p.partType === 'mobility');
    if (mob.length) {
      V('S18', `bosses[${b.id}]: movePattern=="holdCenter" 인데 mobility 부위 [${mob.map((p) => p.id).join(', ')}] `
        + `— ampPx 가 애초에 작아 파괴 효과가 무의미하다 = 거짓 트레이드오프 (§8.12.1)`);
    }
  }
}

// ===========================================================================
//  S19 — zone 의 탄 (§9.7, 04-R5) : emitterType == "zone" ⟺ bulletId == null
// ===========================================================================
function S19_zoneBullet() {
  for (const e of D.enemies.emitters || []) {
    if (!isObj(e) || isAmb(e.type)) continue;
    const lhs = e.type === 'zone';
    const rhs = has(e, 'bulletId') && e.bulletId === null;
    if (lhs !== rhs) {
      V('S19', `enemies.emitters[${e.id}]: (type=="zone")=${lhs} ≠ (bulletId==null)=${rhs} `
        + `— zone 은 dmg 를 직접 갖는다 (§9.7/S19)`);
    }
    if (lhs && !num(e.dmg)) V('S19', `enemies.emitters[${e.id}]: zone 인데 dmg 가 없다 (§3.2 피해원 목록)`);
  }
}

// ===========================================================================
//  S20 — 편대 전용성 (§9.9.2)
//  pincer ⟺ moveId == "strafe" / columnV ⟺ moveId == "column"
// ===========================================================================
function S20_formationExclusivity() {
  const archById = new Map((D.enemies.archetypes || []).map((a) => [a.id, a]));
  const pairs = [['pincer', 'strafe'], ['columnV', 'column']];
  const check = (formationId, archetypeId, tag) => {
    const a = archById.get(archetypeId);
    if (!a || isAmb(a.moveId) || isAmb(formationId)) return;
    for (const [form, move] of pairs) {
      const lhs = formationId === form, rhs = a.moveId === move;
      if (lhs !== rhs) {
        V('S20', `${tag}: (formationId=="${form}")=${lhs} ≠ (moveId=="${move}")=${rhs} `
          + `— 실제 formationId="${formationId}", archetype="${archetypeId}", moveId="${a.moveId}" (§9.9.2/S20)`);
      }
    }
  };
  for (const t of D.stages.themes || []) {
    (t.waves || []).forEach((w, i) => check(w.formationId, w.archetypeId, `stages.themes[${t.id}].waves[${i}]`));
  }
  for (const b of D.bosses.bosses || []) {
    if (isObj(b.summon)) check(b.summon.formationId, b.summon.archetypeId, `bosses[${b.id}].summon`);
  }
}

// ===========================================================================
//  S21 — 드래프트 보장 상한 (§11.1)
//  guarantee* 키의 동시 발동 최대 개수 ≤ optionCount − 1
// ===========================================================================
function S21_draftGuarantees() {
  const d = D.meta.draft;
  if (!isObj(d)) return;
  const gk = Object.keys(d).filter((k) => k.startsWith('guarantee'));
  const cap = num(d.optionCount) ? d.optionCount - 1 : null;
  if (cap === null) return;
  if (gk.length > cap) {
    V('S21', `meta.draft: guarantee* 키 ${gk.length}개 [${gk.join(', ')}] > optionCount − 1 = ${cap} `
      + `— 보장이 optionCount 가 되면 3장 전부 카테고리 고정 = 선택 0 (§11.1/S21)`);
  }
}

// ===========================================================================
//  S22 — 새떼 XP 상한 (§8.10 · §13.4)
//  (crisisTotal × swarmTotalScale[i] × swarmXp) ÷ (스테이지 i 저작 리스트의 Σ XP) ≤ 0.30
// ===========================================================================
const SWARM_XP_CAP = 0.30;

function S22_swarmXpShare() {
  const ph = D.stages.phase || {}, curve = D.stages.curve || {};
  const xpRef = D.enemies.bands && D.enemies.bands.chaff && D.enemies.bands.chaff.xpRef;
  if (!num(xpRef)) { A('enemies.bands.chaff.xpRef', 'S22 의 swarmXp 파생식(= xpRef × 0.5)이 값을 갖지 못한다'); return; }
  const swarmXp = xpRef * 0.5;                       // §8.10 파생식
  const archById = new Map((D.enemies.archetypes || []).map((a) => [a.id, a]));
  const scale = curve.swarmTotalScale || [];
  const FINAL_ID = D.stages.themeDraw ? D.stages.themeDraw.finalStageId : 'finale';

  // ★ 테마 순서는 셔플된다(§8.1) → 어떤 테마가 어느 스테이지에 와도 성립해야 한다
  for (const t of D.stages.themes || []) {
    const unlock = new Map((t.roster || []).map((r) => [r.archetypeId, r.unlockStageMin]));
    const stages = t.id === FINAL_ID ? [6] : [1, 2, 3, 4, 5];
    for (const st of stages) {
      let waveXp = 0, unresolved = false;
      for (const w of t.waves || []) {
        const u = unlock.get(w.archetypeId);
        if (isAmb(u) || !num(u)) { unresolved = true; continue; }
        if (u > st) continue;
        const a = archById.get(w.archetypeId);
        if (!a || !num(a.xp) || !num(w.count)) { unresolved = true; continue; }
        waveXp += w.count * a.xp;
      }
      if (unresolved) {
        A(`stages.themes[${t.id}].roster[].unlockStageMin`,
          `S22 의 분모(스테이지 ${st} 저작 리스트 Σ XP)를 확정할 수 없다`);
        break;
      }
      if (waveXp <= 0) { V('S22', `stages.themes[${t.id}] @ s${st}: 저작 리스트 Σ XP = 0 → 분모 없음`); continue; }
      const s = num(scale[st - 1]) ? scale[st - 1] : null;
      if (s === null) { A(`stages.curve.swarmTotalScale[${st - 1}]`, 'S22 를 평가할 수 없다'); continue; }
      const crisisXp = (ph.crisisTotal || 0) * s * swarmXp;
      const ratio = crisisXp / waveXp;
      if (ratio > SWARM_XP_CAP + 1e-9) {
        V('S22', `stages.themes[${t.id}] @ 스테이지 ${st}: 새떼 XP 지분 ${ratio.toFixed(3)} > ${SWARM_XP_CAP} `
          + `(위기 ${crisisXp} / 웨이브 ${waveXp}) — §13.4-S22 (상한이지 목표가 아니다, §8.10)`);
      }
    }
  }
}

// ===========================================================================
//  S23 — 코인원 균질성 (§13.4)
//  모든 테마의 roster 4종 중 turret + bruiser 밴드가 1~2종
// ===========================================================================
function S23_coinSourceHomogeneity() {
  const archById = new Map((D.enemies.archetypes || []).map((a) => [a.id, a]));
  const FINAL_ID = D.stages.themeDraw ? D.stages.themeDraw.finalStageId : 'finale';
  const pool = new Set((D.stages.themeDraw && D.stages.themeDraw.pool) || []);

  for (const t of D.stages.themes || []) {
    const roster = t.roster || [];
    const bands = roster.map((r) => (archById.get(r.archetypeId) || {}).band).filter(Boolean);
    const n = bands.filter((b) => b === 'turret' || b === 'bruiser').length;

    if (t.id === FINAL_ID) {
      // §8.16 이 finale 로스터를 "공용 9 + 시그니처 6 = 15종 총출동"으로 확정했다
      // → S23 의 문면("모든 테마의 roster 4종")을 finale 에 적용하면 정본이 자기 콘텐츠를 실패시킨다
      C('S23', `S23 의 정의역이 닫혀 있지 않다 — 문면은 "모든 테마의 roster 4종 중 turret+bruiser 가 1~2종"인데 `
        + `stages.themes[] 는 finale 을 포함하고 §8.16 은 finale 로스터를 15종으로 확정했다 `
        + `(실측: roster ${roster.length}종, turret+bruiser ${n}종 → 문면대로면 즉시 실패). `
        + `권고: S23 의 정의역 = themeDraw.pool 의 6테마로 명문화. 근거: 코인원 균질성이 재는 것은 `
        + `"셔플되는 5테마 사이의 코인 수급 편차"이고 finale 은 셔플 대상이 아니다(§8.1)`);
      continue;
    }
    if (!pool.has(t.id)) continue;
    // §8.6 "테마당 정확히 4종 (공용 3 + 시그니처 1)"
    if (roster.length !== 4) V('S23', `stages.themes[${t.id}].roster: ${roster.length}종 ≠ 4 (§8.6)`);
    if (n < 1 || n > 2) {
      V('S23', `stages.themes[${t.id}]: turret+bruiser 밴드 ${n}종 ∉ [1, 2] — 코인원 균질성 (§13.4-S23). `
        + `roster 밴드 = [${bands.join(', ')}]`);
    }
  }
  // 코인 드랍 주체 확정 (§8.6): chaff·line 은 coin 0 / coinDropChance 0
  for (const [bn, bv] of Object.entries(D.enemies.bands || {})) {
    if (!isObj(bv)) continue;
    if ((bn === 'chaff' || bn === 'line') && (bv.coin !== 0 || bv.coinDropChance !== 0.0)) {
      V('S23', `enemies.bands.${bn}: coin=${bv.coin}, coinDropChance=${bv.coinDropChance} — §8.6 "chaff·line 은 코인 0"`);
    }
  }
}

// ===========================================================================
//  S24 — HP 배분 (§13.6.4)
//  Σ(armor 부위 hp) == core.hp × armorCoreRatio  (±1%)
//  선택 부위 hp == armor 부위 1개 hp × boss.optionalPartArmorRatio  (±5%)
// ===========================================================================
function S24_hpDistribution() {
  const ratio = D.rules.boss && D.rules.boss.optionalPartArmorRatio;
  for (const b of D.bosses.bosses || []) {
    if (!isObj(b) || b.tier === 'mid') continue;
    const parts = b.parts || [];
    const armor = parts.filter((p) => p.partType === 'armor');
    const optional = parts.filter((p) => p.partType === 'mobility' || p.partType === 'armament');
    if (!armor.length || !isObj(b.core) || !num(b.core.hp) || !num(b.armorCoreRatio)) continue;

    // (1) Σ(armor hp) == core.hp × φ (±1%)
    const sum = armor.reduce((s, p) => s + (num(p.hp) ? p.hp : NaN), 0);
    const target = b.core.hp * b.armorCoreRatio;
    if (!Number.isFinite(sum)) { A(`bosses[${b.id}].parts[].hp`, 'S24 를 평가할 수 없다'); continue; }
    if (!withinPct(sum, target, 1)) {
      V('S24', `bosses[${b.id}]: Σ(armor hp) = ${sum} vs core.hp(${b.core.hp}) × armorCoreRatio(${b.armorCoreRatio}) = ${target.toFixed(1)} `
        + `— 오차 ${(((sum - target) / target) * 100).toFixed(3)}% > ±1% (§13.4-S24)`);
    }
    // §13.6.4: armor 부위끼리 균등 (비대칭이면 특화 짝 6종의 대칭이 깨진다)
    const hps = armor.map((p) => p.hp);
    if (new Set(hps).size > 1) {
      V('S24', `bosses[${b.id}]: armor 부위 hp 가 균등하지 않다 [${hps.join(', ')}] — §13.6.4 "armor 부위끼리 균등"`);
    }
    // (2) 선택 부위 hp == armor 1개 × optionalPartArmorRatio (±5%)
    if (!num(ratio)) continue;
    const t2 = armor[0].hp * ratio;
    for (const p of optional) {
      if (!num(p.hp)) continue;
      if (!withinPct(p.hp, t2, 5)) {
        V('S24', `bosses[${b.id}].parts[${p.id}] (${p.partType}): hp = ${p.hp} vs armor 1개(${armor[0].hp}) × optionalPartArmorRatio(${ratio}) = ${t2.toFixed(1)} `
          + `— 오차 ${(((p.hp - t2) / t2) * 100).toFixed(3)}% > ±5% (§13.4-S24)`);
      }
    }
  }
}

// ===========================================================================
//  S25 — 상성 매트릭스의 무결성 (§9.4.4)
//  ★ 이 게임의 중심 축이 데이터 오타 하나로 뒤집히는 것을 막는 유일한 문
// ===========================================================================
function S25_elementMatrix() {
  const e = D.elements;
  if (!isObj(e)) return;
  const { order, investable, matrix } = e;

  if (!Array.isArray(order) || order.length !== 4) {
    V('S25', `elements.order: ${JSON.stringify(order)} — 4원소 동결 배열이어야 한다 (§9.4.4)`);
  } else if (order.join(',') !== 'normal,fire,water,grass') {
    // §4.1 "표 순서 = 키 순서 = Q(노말) W(불) E(물) R(풀). 전 문서·전 UI·전 카드·전 패널"
    V('S25', `elements.order = [${order.join(', ')}] ≠ [normal, fire, water, grass] — §4.1 "Q W E R 순"이 전역 순서 규칙이고 §9.9.1 의 타이브레이크가 이것을 직접 읽는다`);
  }
  if (!isObj(matrix)) { V('S25', 'elements.matrix: 중첩 맵이 아니다 (§9.4.4)'); return; }

  const keys = Object.keys(matrix);
  if (!Array.isArray(order) || new Set(keys).size !== new Set(order).size
      || !order.every((k) => keys.includes(k))) {
    V('S25', `elements.matrix 의 키 집합 [${keys.join(', ')}] 이 order [${(order || []).join(', ')}] 와 정합하지 않다 (§9.4.4)`);
  }
  // investable = 3종, normal 불포함, matrix 키의 부분집합
  if (!Array.isArray(investable) || investable.length !== 3) {
    V('S25', `elements.investable: ${JSON.stringify(investable)} — 3종이어야 한다 (§4.2)`);
  } else {
    if (investable.includes('normal')) V('S25', 'elements.investable 에 "normal" — §4.2 "노말은 투자축이 아니다"');
    for (const el of investable) if (!keys.includes(el)) V('S25', `elements.investable 의 "${el}" 이 matrix 키에 없다`);
  }
  // 16셀 전부 + 값 ∈ {0.5, 1.0, 2.0}
  for (const a of keys) {
    const row = matrix[a];
    if (!isObj(row)) { V('S25', `elements.matrix.${a}: 객체가 아니다`); continue; }
    closedKeys('S25', row, keys, `elements.matrix.${a}`);
    for (const b of keys) {
      const v = row[b];
      if (![0.5, 1.0, 2.0].includes(v)) {
        V('S25', `elements.matrix.${a}.${b} = ${JSON.stringify(v)} ∉ {0.5, 1.0, 2.0} (§9.4.4/S25)`);
      }
    }
  }
  // normal 행/열 전부 1.0
  for (const b of keys) {
    if (matrix.normal && matrix.normal[b] !== 1.0) V('S25', `elements.matrix.normal.${b} = ${matrix.normal[b]} ≠ 1.0 — §4.1 "노말은 항상 ×1"`);
    if (matrix[b] && matrix[b].normal !== 1.0) V('S25', `elements.matrix.${b}.normal = ${matrix[b].normal} ≠ 1.0 — §4.1 "노말은 항상 ×1"`);
  }
  // 순환: matrix[x][y] == 2.0 ⟺ matrix[y][x] == 0.5
  for (const x of keys) for (const y of keys) {
    if (!matrix[x] || !matrix[y]) continue;
    const l = matrix[x][y] === 2.0, r = matrix[y][x] === 0.5;
    if (l !== r) {
      V('S25', `elements.matrix: 순환 위반 — (matrix.${x}.${y}==2.0)=${l} ≠ (matrix.${y}.${x}==0.5)=${r} `
        + `(값 = ${matrix[x][y]} / ${matrix[y][x]}) (§9.4.4/S25)`);
    }
    // 대각 = 1.0 (자기 자신에는 상성 없음, §4.1 표)
    if (x === y && matrix[x][y] !== 1.0) V('S25', `elements.matrix.${x}.${x} = ${matrix[x][y]} ≠ 1.0 (§4.1 표)`);
  }
  // 투자 가능 3종이 정확히 하나의 순환(물 > 불 > 풀 > 물)을 이룬다
  if (Array.isArray(investable) && investable.length === 3) {
    for (const el of investable) {
      const cnt2 = investable.filter((o) => matrix[el] && matrix[el][o] === 2.0).length;
      const cntH = investable.filter((o) => matrix[el] && matrix[el][o] === 0.5).length;
      if (cnt2 !== 1 || cntH !== 1) {
        V('S25', `elements.matrix.${el}: 투자 3종에 대해 ×2 가 ${cnt2}개, ×0.5 가 ${cntH}개 — 각 1개여야 순환이다 (§4.1 "물 > 불 > 풀 > 물")`);
      }
    }
  }
}

// ===========================================================================
//  S26 — 동시 개체 예산의 정적 하한 (§12.1)
//  웨이브 1개의 Σ count ≤ enemyConcurrentMax(40)
//  위기 서브웨이브 1파의 Σ count ≤ swarmConcurrentMax(70)
//  ★ 동시 개체 수 자체는 처치율의 함수라 정적 검사 불가 → "어떤 처치율에서도 깨지는 편성"만 잡는다
// ===========================================================================
function S26_concurrentBudget() {
  const f = D.rules.fairness || {}, ph = D.stages.phase || {}, curve = D.stages.curve || {};

  // (1) 웨이브 1개 = waves[] 의 레코드 1개 (§8.7 "waves: [{formationId, archetypeId, count, ...}]")
  for (const t of D.stages.themes || []) {
    (t.waves || []).forEach((w, i) => {
      if (!num(w.count) || !num(f.enemyConcurrentMax)) return;
      if (w.count > f.enemyConcurrentMax) {
        V('S26', `stages.themes[${t.id}].waves[${i}]: count ${w.count} > enemyConcurrentMax(${f.enemyConcurrentMax}) `
          + `— 이 웨이브는 어떤 처치율에서도 A층 예산을 깬다 = 편성 버그 (§12.1/S26)`);
      }
    });
  }

  // (2) 위기 서브웨이브 1파 = crisisTotal × swarmTotalScale[i] ÷ crisisSubWaves
  if (num(ph.crisisTotal) && num(ph.crisisSubWaves) && ph.crisisSubWaves > 0 && num(f.swarmConcurrentMax)) {
    (curve.swarmTotalScale || []).forEach((s, i) => {
      if (!num(s)) return;
      const total = ph.crisisTotal * s;
      const perSub = total / ph.crisisSubWaves;
      if (perSub > f.swarmConcurrentMax) {
        V('S26', `위기 서브웨이브 @ 스테이지 ${i + 1}: 1파 ${perSub} > swarmConcurrentMax(${f.swarmConcurrentMax}) (§12.1/S26)`);
      }
      // 새떼 전원이 동시에 살아있는 최악(= 아무도 안 죽는다)도 예산 안이어야 한다
      if (total > f.swarmConcurrentMax) {
        V('S26', `위기 총량 @ 스테이지 ${i + 1}: crisisTotal(${ph.crisisTotal}) × swarmTotalScale(${s}) = ${total} `
          + `> swarmConcurrentMax(${f.swarmConcurrentMax}) (§12.1/S26)`);
      }
    });
  }
}

// ===========================================================================
//  §13.1 certify 게이트 — 정적으로 검사 가능한 것
// ===========================================================================
function certifyStatic() {
  const c = D.meta.certify;
  if (!isObj(c)) return;

  // (1) 인쇄된 게이트 세트의 필드 집합 (§13.1 = certify 를 인쇄하는 유일한 절, C-10)
  closedKeys('CERT', c, ['runs', 'dpsRef', 'runFarmDpsRatio', 'runMode', 'dpsProbe', 'static'], 'meta.certify');
  if (isObj(c.runMode)) {
    closedKeys('CERT', c.runMode, ['runClearRate', 'bossTimeoutRate', 'noDeadLuck', 'stanceValue',
      'difficultySpread', 'dominance', 'coinScarcity', 'farmXpRatio', 'crisisKillShareWithoutCapstone'], 'meta.certify.runMode');
  }
  if (isObj(c.dpsProbe)) {
    closedKeys('CERT', c.dpsProbe, ['runsPerCell', 'difficulty', 'farm', 'uptimeRef', 'balancedPass',
      'specialistPass', 'noElementPass', 'killTimeMedianBalanced'], 'meta.certify.dpsProbe');
  }
  if (isObj(c.static)) {
    closedKeys('CERT', c.static, ['growthBudget', 'capHits', 'fairnessViolations'], 'meta.certify.static');
  }

  // (2) 밴드의 내적 정합 — min ≤ max
  const bands = [];
  const walkBands = (node, path) => {
    if (!isObj(node)) return;
    if (num(node.min) && num(node.max)) bands.push([path, node.min, node.max]);
    for (const k of Object.keys(node)) walkBands(node[k], `${path}.${k}`);
  };
  walkBands(c, 'meta.certify');
  for (const [path, lo, hi] of bands) {
    if (lo > hi) V('CERT', `${path}: min(${lo}) > max(${hi}) — 도달 불가능한 밴드`);
  }
  // 배열 밴드 (noElementPass)
  const nep = c.dpsProbe && c.dpsProbe.noElementPass;
  if (isObj(nep) && Array.isArray(nep.min) && Array.isArray(nep.max)) {
    if (nep.min.length !== nep.max.length) V('CERT', 'certify.dpsProbe.noElementPass: min/max 길이 불일치');
    nep.min.forEach((lo, i) => {
      const hi = nep.max[i];
      if (num(lo) && num(hi) && lo > hi) V('CERT', `certify.dpsProbe.noElementPass[${i}]: min(${lo}) > max(${hi})`);
    });
  }

  // (3) 길이 6 배열의 정합 (스테이지 축)
  const six = [
    ['certify.dpsRef', c.dpsRef],
    ['stages.curve.enemyHpScale', D.stages.curve && D.stages.curve.enemyHpScale],
    ['stages.curve.xpScale', D.stages.curve && D.stages.curve.xpScale],
    ['stages.curve.bossHpScale', D.stages.curve && D.stages.curve.bossHpScale],
    ['stages.curve.spawnDensityScale', D.stages.curve && D.stages.curve.spawnDensityScale],
    ['stages.curve.midBossCount', D.stages.curve && D.stages.curve.midBossCount],
    ['stages.curve.elitePerWaveChance', D.stages.curve && D.stages.curve.elitePerWaveChance],
    ['stages.curve.swarmTotalScale', D.stages.curve && D.stages.curve.swarmTotalScale],
    ['stages.curve.rearSpawnAllowed', D.stages.curve && D.stages.curve.rearSpawnAllowed],
    ['meta.flow.stagePar', D.meta.flow && D.meta.flow.stagePar],
    ['certify.dpsProbe.balancedPass.min', c.dpsProbe && c.dpsProbe.balancedPass && c.dpsProbe.balancedPass.min],
    ['certify.dpsProbe.specialistPass.min', c.dpsProbe && c.dpsProbe.specialistPass && c.dpsProbe.specialistPass.min],
    ['certify.dpsProbe.noElementPass.min', nep && nep.min],
  ];
  for (const [path, arr] of six) {
    if (Array.isArray(arr) && arr.length !== 6) V('CERT', `${path}: 길이 ${arr.length} ≠ 6 (스테이지 축)`);
  }

  // (4) §13.5.1 runFarmDpsRatio — probe farm 정책과 dpsProbe.farm 이 같아야 한다
  if (c.dpsProbe && c.dpsProbe.farm !== 'maxFarm') {
    V('CERT', `certify.dpsProbe.farm = "${c.dpsProbe.farm}" ≠ "maxFarm" — §13.5 "dpsRef 의 farm 정책 = dpsProbe.farm 과 같다"`);
  }
  // bot.policies 안의 값이어야 한다
  const pol = D.meta.bot && D.meta.bot.policies;
  if (isObj(pol)) {
    if (c.dpsProbe && Array.isArray(pol.farm) && !pol.farm.includes(c.dpsProbe.farm)) {
      V('CERT', `certify.dpsProbe.farm = "${c.dpsProbe.farm}" 이 bot.policies.farm 에 없다 (§10.4.1)`);
    }
    const bl = D.meta.bot.baseline;
    if (isObj(bl)) for (const ax of ['draft', 'farm', 'stance', 'shop']) {
      if (Array.isArray(pol[ax]) && !pol[ax].includes(bl[ax])) {
        V('CERT', `meta.bot.baseline.${ax} = "${bl[ax]}" 이 policies.${ax} 에 없다 (§10.4.1)`);
      }
    }
  }
  // §13.5.1: runFarmDpsRatio 는 baseline(balanced) 명목 ÷ dpsRef(maxFarm) → (0, 1]
  if (num(c.runFarmDpsRatio) && !(c.runFarmDpsRatio > 0 && c.runFarmDpsRatio <= 1)) {
    V('CERT', `certify.runFarmDpsRatio = ${c.runFarmDpsRatio} ∉ (0, 1] — balanced ÷ maxFarm (§13.5.1)`);
  }
  // §13.5: dpsRef 는 단조 비감소이고 [5] == [6] (제너럴리스트 화력 천장)
  if (Array.isArray(c.dpsRef) && c.dpsRef.every(num)) {
    for (let i = 1; i < c.dpsRef.length; i += 1) {
      if (c.dpsRef[i] < c.dpsRef[i - 1]) V('CERT', `certify.dpsRef: 스테이지 ${i + 1} 에서 감소 (${c.dpsRef[i - 1]} → ${c.dpsRef[i]}) — §13.5`);
    }
  }
  // §13.6.1: bossHpScale[i] = dpsRef 대비 배수 × bossRamp[i]. 최종은 미적용
  //          bossHpScale[5] == bossHpScale[6] (dpsRef[5]==dpsRef[6] 이므로)
  const bhs = D.stages.curve && D.stages.curve.bossHpScale;
  if (Array.isArray(bhs) && Array.isArray(c.dpsRef) && bhs.length === 6 && c.dpsRef.length === 6) {
    if (c.dpsRef[4] === c.dpsRef[5] && bhs[4] !== bhs[5]) {
      V('CERT', `stages.curve.bossHpScale[5](${bhs[4]}) ≠ [6](${bhs[5]}) 인데 dpsRef[5]==dpsRef[6]==${c.dpsRef[4]} — §13.6.1 "화력이 같은 두 스테이지의 보스 HP가 같은 것이 정합"`);
    }
    if (bhs[0] !== 1.0) V('CERT', `stages.curve.bossHpScale[1] = ${bhs[0]} ≠ 1.00 — 저작값은 스테이지 1 기준 base (§13.6.1)`);
  }

  // (5) §12.1 "캡에 닿는 콘텐츠는 콘텐츠 버그다" — 인증은 capHits == 0
  const stat = c.static || {};
  if (isObj(stat.capHits) && stat.capHits.max !== 0) {
    V('CERT', `certify.static.capHits.max = ${stat.capHits.max} ≠ 0 — §12.1 "캡에 닿는 콘텐츠는 콘텐츠 버그다"`);
  }
  if (isObj(stat.fairnessViolations) && stat.fairnessViolations.max !== 0) {
    V('CERT', `certify.static.fairnessViolations.max = ${stat.fairnessViolations.max} ≠ 0 — §9.3 "위반 → 로드 실패"`);
  }

  // (6) §11.3/§11.4 상호 정합 — 컨티뉴 비용 vs coinScarcity p90
  const p90 = c.runMode && c.runMode.coinScarcity && c.runMode.coinScarcity.p90EndCoins;
  const cc = D.meta.flow && D.meta.flow.continueCost;
  if (isObj(p90) && num(p90.max) && num(cc) && p90.max < cc) {
    V('CERT', `coinScarcity.p90EndCoins.max(${p90.max}) < flow.continueCost(${cc}) — §13.1.1 "p90 ≤ 컨티뉴 1회(${cc}) + 여유"가 성립 불가`);
  }
}

// ===========================================================================
//  동적 게이트 (시뮬 필요) — 스텁 + TODO. 인터페이스는 정본대로 (§13.1 · §10.4)
// ===========================================================================

/**
 * ★ 인터페이스 계약 (§10.4 · §13.1)
 *   tools/sim.mjs 가 아래 서명을 구현하고 report/summary.json 을 낸다.
 *   check.mjs 는 그 파일이 있으면 게이트를 채점하고, 없으면 STUB 으로 남긴다.
 *
 *   runCertify({ runs, difficulty, policy }) -> {
 *     runClearRate, bossTimeoutRate, noDeadLuck: {...}, stanceValue, difficultySpread,
 *     dominance: {...}, coinScarcity: {...}, farmXpRatio, crisisKillShareWithoutCapstone,
 *     capHits: { enemyConcurrentMax, swarmConcurrentMax, crisisWaveResidualMax,
 *                telegraphConcurrentMaxGlobal, capsOverflow },   // §13.1.1 — 4축 분리 출력
 *     fairnessViolations
 *   }
 *   runDpsProbe({ runsPerCell, difficulty, farm, uptimeRef }) -> {
 *     balancedPass[6], specialistPass[6], noElementPass[6], killTimeMedianBalanced
 *   }
 */
function dynamicGateStubs() {
  const c = D.meta.certify || {};
  const dp = c.dpsProbe || {};
  const todo = [
    ['runClearRate', `분모 = runs(${c.runs}) 전체, 분자 = 스테이지 6 보스 격파. policy=baseline, difficulty=normal (§13.1.1)`],
    ['bossTimeoutRate', `분모 = runs 전체. 분자 = deaths.csv 사인이 "시간 초과"인 런. 컨티뉴는 사인을 리셋 (§13.1.1)`],
    ['noDeadLuck', `테마 순서 720개를 스테이지 5 테마별 6군집(각 ≥1300런)으로 집계 + draft 축 6정책 각각의 runClearRate 최솟값 (§13.1.1)`],
    ['stanceValue', `runClearRate(baseline) − runClearRate(stance="static", 나머지 3축 baseline) (§13.1.1)`],
    ['difficultySpread', `disaster(speed 3.0) 의 runClearRate ∈ [0.02, 0.12] (§13.1)`],
    ['dominance.maxWeaponPickShare', `분모 = runs × 3 = ${(c.runs || 0) * 3}. forward 제외 후 11종 재정규화 (§13.1.1)`],
    ['dominance.maxWeaponWinShare', `★ 피해 지분의 평균. 분모 = 클리어 런 수. forward 제외 후 11종 재정규화 (§13.1.1)`],
    ['dominance.startWeaponDamageShare', `forward 전용. 분모 = 클리어 런의 4무기 총 피해 (§13.1.1)`],
    ['dominance.maxElementWinShare', `분모 = 클리어 런의 총 속성 투자 픽 수(런당 ≤ 6). 3종 재정규화 (§13.1.1)`],
    ['dominance.maxArchetypeLethalityShare', `분모 = 전 런에서 플레이어가 입은 총 피해(실드 흡수 제외). 대상 = 잡몹 15 + 새떼 2 = 17종. 엘리트는 원 아키타입 귀속, 중간보스·보스·부위는 분모에서도 제외 (§13.1.1)`],
    ['dominance.maxThemeClearStddev', `테마 t별 clearRate 6개 값의 표본 표준편차. finale 제외 (§13.1.1)`],
    ['coinScarcity', `medianEndCoins / medianPurchasesPerVisit(분모 = 상점 방문 수, 런당 5) / p90EndCoins (§13.1.1)`],
    ['farmXpRatio', `(maxFarm 스테이지 평균 XP) ÷ (passive 스테이지 평균 XP). 나머지 3축 baseline (§13.1.1)`],
    ['crisisKillShareWithoutCapstone', `capstone = 보유 무기에 nova 또는 aura. 대상 = capstone 미보유 ∧ 그 세션 폭탄 미사용. killShare = 처치 새떼 수 ÷ (crisisTotal × swarmTotalScale[stage]) 의 중앙값 (§13.1.1)`],
  ];
  for (const [k, why] of todo) {
    S('CERT-DYN', `${k}: 시뮬(run 모드) 필요 → TODO: tools/sim.mjs + report/summary.json. ${why}`);
  }
  S('CERT-DYN', `dpsProbe (balancedPass/specialistPass/noElementPass/killTimeMedianBalanced): `
    + `셀 = (보스, 스테이지) 쌍 = 3 + 24 + 1 = 28 셀 × runsPerCell(${dp.runsPerCell}), farm="${dp.farm}", uptimeRef=${dp.uptimeRef} (§10.4.2)`);
  S('CERT-DYN', `capHits: ★ A층(enemyConcurrentMax·swarmConcurrentMax·crisisWaveResidualMax·telegraphConcurrentMaxGlobal 의 defer) `
    + `+ B층(caps.* overflow, 순수 FX 3종 제외) 를 4축 분리 출력. 상한 ${(c.static && c.static.capHits && c.static.capHits.max)} (§13.1.1)`);
  S('CERT-DYN', `fairnessViolations: 런타임 어서션(특히 minSpawnRadiusPx). 상한 ${(c.static && c.static.fairnessViolations && c.static.fairnessViolations.max)} (§13.1)`);
}

// ===========================================================================
//  정본 결함 — 검사를 쓰면서 드러난 것 (하드코딩 신고)
//  ★ 발명하지 않는다: 아래는 전부 "정본이 답하지 않아 검사를 완성할 수 없는 자리"다
// ===========================================================================
function canonDefects() {
  // C-1: §13.4 의 S 번호
  C('META', `§13.4 표는 S1~S26 을 담는데 인쇄 순서가 S1..S23, S25, S26, S24 다 (S24 가 표 맨 끝). `
    + `그리고 §18.2 는 "S1~S24", §20 은 "S1~S26" 이라 부른다 → 이 파일은 §13.4 표의 실제 26행을 구현했다. `
    + `권고: 표를 번호순으로 재배열하고 인용을 "S1~S26" 으로 통일`);

  // C-2: weapons 계약의 "누락 키" 절반이 평가 불가
  C('S2', `§9.5 의 패밀리 계약이 "base 의 필수 키 집합"을 확정하지 못한다 — "공통: dmg cooldownSec count projSpeed `
    + `projRadius lifetimeSec pierce hitCooldownSec targetMode" 를 인쇄하고 "계약 = 공통 9 + 고유 n" 이라 한 뒤, `
    + `같은 절이 "orbit 이 cooldownSec·projSpeed·lifetimeSec·pierce·count 를 전부 0으로 선언해야 한다 = 죽은 필드" 라며 `
    + `그것을 부정한다. targetMode 의 부재만 5패밀리에 명시됐고, 나머지 공통 8 중 어느 것이 어느 패밀리에서 빠지는지는 `
    + `어디에도 없다. → §9.3 의 "누락 키 = 에러" 가 weapons.json 에서 평가 불가. `
    + `이 파일은 "미지 키" 절반(계약 상위집합 밖 = 에러)만 구현했다. `
    + `권고: §9.5 의 패밀리 표에 "base 필수 키" 열을 추가해 12행 전부를 인쇄한다(passiveHooks 의 rateKey/countKey 가 `
    + `이미 절반을 함의하므로 새 결정은 거의 없다)`);

  // C-3: 이미터 공통 파라미터 3종의 어휘·의미 부재
  C('S2/S6', `enemies.emitters 의 공통 파라미터 3개가 정의되지 않았다 — ① from: §9.7 이 "self" 를 예시로 인쇄했을 뿐 `
    + `어휘가 닫히지 않았고 S3 의 동결 어휘 목록(10종)에도 없다 ② repeat / restSec: §8.5 가 공통 파라미터로 열거만 하고 `
    + `의미(발사 반복 수? 버스트 후 휴지?)를 어디서도 정의하지 않는다 → S7(동시 텔레그래프)의 전개 모델이 `
    + `"t_k = offsetSec + k·everySec" 단순 주기밖에 될 수 없고, repeat > 1 이 생기는 순간 S7 이 조용히 과소평가한다. `
    + `권고: from 의 어휘를 S3 에 추가 + repeat/restSec 의 시간 모델을 §8.5 에 한 줄로 인쇄(또는 두 키를 삭제)`);

  // C-4: §7.4 "보스 대형 패턴"이 데이터 개념이 아니다
  C('S6', `§7.4 의 텔레그래프 하한 표에 "보스 대형 패턴 → 1.50" 행이 있으나 "대형"은 데이터에 존재하지 않는 구분이다 `
    + `(bosses[].parts[].patternSet[].emitterIds 는 이미터 id 만 갖는다). → check.mjs 가 어느 보스 이미터에 1.50 을 `
    + `걸어야 하는지 결정할 수 없다. 이 파일은 "중간보스 패턴 → 1.20"(소유 개체로 역인덱스 가능)만 구현하고 보스 행은 `
    + `구현하지 않았다. 권고: ⓐ 보스 부위 이미터 전부에 1.50 을 걸거나 ⓑ 이미터에 판별 필드를 두거나 `
    + `ⓒ 그 행을 렌더 규격으로 강등(= 하한이 아님을 명시)`);

  // C-5: 탄 속도의 거처가 둘
  C('S6', `탄 속도의 거처가 둘이다 (C-2 위반) — bullets[].speed(§9.7 인쇄)와 emitters[].speed(§8.5 의 `
    + `straight/fan/aimed/ring/spiral/wall 공통 파라미터). 어느 쪽이 진실이고 다른 쪽이 무엇인지(오버라이드? 초기속도?) `
    + `정본이 말하지 않는다. §9.3 은 폴백을 금지하므로 "이미터가 있으면 그것, 없으면 탄" 도 불가능하다. `
    + `이 파일은 보수적으로 양쪽 모두에 fairness.maxBulletSpeed 를 건다. `
    + `권고: 한쪽을 유일 소유자로 확정(§9.7 이 "적 탄 데미지의 소유자 = bullets[].dmg" 를 확정한 것과 같은 처방)`);

  // C-6: visual.statusBulletSpeedMul 의 적용 주체
  C('S6', `visual.statusBulletSpeedMul(0.6) 이 §12.4 의 "공정성 하한" 표에 "상태이상 탄 속도 배율"로 실려 있으나 `
    + `거처는 visual 스코프다. §9.4.3 이 "visual = 무엇을 그리는가의 규격, render = 예산" 이라 경계를 명문화했으므로 `
    + `visual 키가 게임플레이 속도를 바꾸면 그 경계가 깨진다. 그리고 엔진이 이것을 곱하는지 `
    + `(hexBolt.speed 95 → 57) 저작값에 이미 반영된 것인지(150 × 0.6 = 90 ≈ 95) 정본이 말하지 않는다 `
    + `→ S6 의 maxBulletSpeed 검사가 어느 수를 봐야 하는지 결정 불가. `
    + `권고: 엔진이 곱한다면 거처를 rules.status 로 옮기고, 저작 가이드라면 §12.4 표에서 빼고 백틱을 벗긴다`);

  // C-7: bosses[].themeId 가 mid/final 에서 의미를 갖지 않는다
  C('S2/S5', `bosses[].themeId 는 §9.8 의 스키마 행이며 §9.3 은 "누락 키 = 에러" 인데, tier=="mid" 는 §8.9 가 `
    + `"3종 전 테마 공용" 으로 확정했고 tier=="final" 은 §8.16 이 "테마 속성 없음" 으로 확정했다 `
    + `→ 두 tier 에서 이 키가 가리킬 대상이 없다(현재 data 는 3중간보스 + tetrarch 에서 "__AMBIGUOUS__"). `
    + `권고: themeId 를 tier=="stage" 전용 필드로 명문화하고, mid/final 에는 키 자체를 두지 않는다 `
    + `(§9.5 의 "targetMode 가 없는 패밀리는 null 이 아니라 키 자체가 없다" 와 같은 처방)`);

  // C-8: finale roster 의 unlockStageMin
  C('S22', `stages.themes[finale].roster[].unlockStageMin 의 값을 정본이 주지 않는다 — §9.7(04-R6)이 `
    + `unlockStageMin 의 유일한 거처를 로스터 엔트리로 확정했고 §9.3 은 폴백을 금지하는데, §8.16 은 finale 로스터를 `
    + `"15종 총출동"이라고만 한다. finale 은 항상 스테이지 6 이므로 값이 무엇이든 필터가 통과하지만, `
    + `"무엇이든"은 데이터가 아니다 → S22 의 분모(스테이지 6 저작 리스트 Σ XP)가 확정되지 않는다. `
    + `권고: §8.16 에 "finale 로스터의 unlockStageMin = 전부 6" 을 인쇄(또는 1)`);

  // C-9: S22 의 "스테이지 i" 가 테마 셔플과 곱해진다
  C('S22', `S22 의 문면("스테이지 i 저작 리스트의 Σ XP")이 테마 셔플(§8.1)과 결합되지 않았다 — `
    + `저작 리스트는 테마의 것이고 swarmTotalScale[i] 는 스테이지의 것인데, 같은 테마가 스테이지 1~5 어디에도 온다. `
    + `이 파일은 보수적으로 전 (테마 × 스테이지) 조합을 검사한다(= 어느 배치에서도 상한을 지켜야 한다). `
    + `권고: S22 를 "모든 (theme, stage) 쌍에 대해" 로 명문화`);

  // C-10: S10 의 좌변
  C('S10', `S13.4-S10 의 문면("XP 곡선으로 계산한 최대 레벨업 횟수")이 정적으로 계산 불가능하다 — `
    + `레벨업 횟수는 XP 곡선 × 웨이브 편성 × 처치율 × farm 정책의 함수이고, 정본은 그것을 상수 `
    + `certify.static.growthBudget.maxLevelUps(60) 로 소유한다. 권고: S10 을 "maxLevelUps < minTotalSink `
    + `(둘 다 선언 상수) + minTotalSink 가 player.weaponSlots/elementCapTotal/passiveSlots × passives.maxLevel 에서 `
    + `유도된 값과 일치" 로 문면을 고친다(이 파일은 그렇게 구현했다)`);

  // C-11: S6 의 minSpawnRadiusPx
  C('S6', `S6 이 "minSpawnRadiusPx 140" 을 정적 검사 목록에 넣었으나 이 규칙("적 탄은 플레이어 반경 140px 이내에서 `
    + `생성 불가")은 발사 시점의 플레이어 위치에 의존한다 → 정적으로 검사할 대상이 존재하지 않는다. `
    + `§12.1 이 enemyConcurrentMax 에 대해 "검사할 대상이 존재하지 않는다 → 정적 검사 폐기, 런타임 defer + capHits" `
    + `라고 내린 것과 정확히 같은 판정이 필요하다. 권고: S6 에서 이 항목을 빼고 `
    + `certify.static.fairnessViolations 가 세는 런타임 어서션으로 이관`);

  // C-12: finale.armorCoreRatio 의 거처가 둘이고 인쇄 자리는 0이다
  C('S2/S5', `rules.boss.finale.armorCoreRatio 의 거처가 확정되지 않았다 — §8.16 이 "★ finale.armorCoreRatio = 13.58" 을 `
    + `확정 키로 인쇄했으나 §9.4 의 rules.json 인쇄 블록(C-7 = 필드 집합의 확정 권한)의 boss.finale 은 `
    + `{partCount, armorPartCount, exemptRules, allowNormalPeripheral} 4키뿐이다 → 넣으면 "미지 키 = 에러"(§9.3)로 `
    + `로드 실패, 안 넣으면 §8.16 이 거짓. 그리고 §9.8 은 bosses[].armorCoreRatio 를 "필수(tier ∈ {stage, final})" 로 `
    + `확정했고 data 의 tetrarch 가 이미 13.58 을 갖는다 → ★ 같은 값의 거처가 둘이다(C-2 위반). `
    + `이 파일은 §9.4 의 인쇄 블록을 스키마로 삼았으므로 위 VIOLATION 목록에 미지 키로 뜬다. `
    + `권고: §8.16 에서 이 키를 삭제하고 bosses[].armorCoreRatio(§9.8) 를 유일 소유자로 확정한다 — `
    + `R7 이 이미 "a 를 알면 답이 나온다"며 개수 규칙(R6, 면제 가능)과 비율 규칙(R7, 면제 불필요)을 갈라놓았고, `
    + `φ 는 보스 개체의 속성이지 rules 의 상수가 아니다(테마 보스 6종도 rules 가 아니라 개체가 4.90 을 갖는다)`);

  // C-13: rules.audio.bgm 의 값이 존재하지 않는다
  C('S2', `rules.audio.bgm 은 §9.4 에서 "...트랙 8종..." 으로 축약됐고 §7.10 이 트랙 1개의 필드 집합만 `
    + `(rootNote, scale, bpm, patternIdx, layers) 인쇄한다 — 그러나 ⓐ 8개 trackId 의 어휘가 산문("테마 6 + 보스 1 + `
    + `타이틀 1")에만 있고 ⓑ 40개 값(8 × 5필드)이 전 코퍼스에 0개 있다. C-10 은 축약이 "필드 집합을 인쇄하는 절을 `
    + `정확히 하나" 가리킬 것을 요구하는데, §7.10 은 트랙의 필드 집합만 주고 맵의 키 집합을 주지 않는다. `
    + `권고: §7.10 에 trackId 8종 어휘를 인쇄(값은 저작 위임임을 명시)`);
}

// ===========================================================================
//  출력
// ===========================================================================
function print() {
  const line = (s = '') => { if (!QUIET) console.log(s); };
  const bar = '─'.repeat(78);

  line();
  line(`NAN 2026 — check.mjs   (정본 §13.4 S1~S26 + §9.3 로더 규칙)`);
  line(`data: ${relative(process.cwd(), DATA_DIR) || DATA_DIR}   (${MANIFEST.length}파일)`);
  line(bar);

  const sect = (title, items, fmt) => {
    if (!items.length) return;
    line();
    line(`${title}  (${items.length})`);
    line(bar);
    items.forEach((it, i) => line(`${String(i + 1).padStart(3)}. ${fmt(it)}`));
  };

  sect('[VIOLATION] 데이터가 정본을 위반한다 — 고칠 곳 = data/*.json',
    report.violation, (v) => `${v.check.padEnd(8)} ${v.msg}`);

  sect('[CANON] 정본 자신의 결함 — 검사를 완성할 수 없다. 고칠 곳 = design/CANON.md',
    report.canon, (v) => `${v.check.padEnd(8)} ${v.msg}`);

  sect('[AMBIGUOUS] 정본이 아직 답하지 않은 자리 ("__AMBIGUOUS__")',
    report.ambiguous, (a) => `${a.path}${a.note ? `  — ${a.note}` : ''}`);

  sect('[STUB] 시뮬 필요 — 정적 검사 불가 (인터페이스는 정본대로)',
    report.stub, (s) => `${s.check.padEnd(9)} ${s.msg}`);

  sect('[SKIP] 검사 대상이 아직 없다',
    report.skip, (s) => `${s.check.padEnd(8)} ${s.msg}`);

  line();
  line(bar);
  line(`요약  VIOLATION ${report.violation.length} · CANON ${report.canon.length} `
    + `· AMBIGUOUS ${report.ambiguous.length} · STUB ${report.stub.length} · SKIP ${report.skip.length}`);
  line(bar);

  const hardFail = report.violation.length > 0 || report.canon.length > 0;
  const ambFail = report.ambiguous.length > 0 && !ALLOW_AMBIGUOUS;

  if (hardFail || ambFail) {
    if (!QUIET) {
      line();
      if (report.violation.length) line(`✗ 데이터 위반 ${report.violation.length}건`);
      if (report.canon.length) line(`✗ 정본 결함 ${report.canon.length}건 — 정본이 답해야 검사가 완성된다`);
      if (ambFail) line(`✗ 미해결 모호값 ${report.ambiguous.length}건 (--allow-ambiguous 로 무시 가능)`);
      line();
    }
    return 1;
  }
  line();
  line('✓ 전 정적 게이트 통과');
  line();
  return 0;
}

// ===========================================================================
//  main
// ===========================================================================
function main() {
  loadAll();

  S2_schema();          // §9.3 로더 + §9.4~§9.9 스키마 + 참조 무결성
  S3_vocab();           // §13.4 S3
  S1_corePurity();      // §9.1
  S4_archetypeOverlap();
  S5_bossRules();       // R1~R7
  S6_fairness();
  S7_concurrentTelegraphs();
  S8_mix();
  S9_structure();
  S10_growthBudget();
  S11_rngStreams();
  S12_twoLayerCaps();
  S13_stunHome();
  S14_shapeStatusEquiv();
  S15_midBossElement();
  S16_patternSetLen();
  S17_summon();
  S18_mobilityTruth();
  S19_zoneBullet();
  S20_formationExclusivity();
  S21_draftGuarantees();
  S22_swarmXpShare();
  S23_coinSourceHomogeneity();
  S24_hpDistribution();
  S25_elementMatrix();
  S26_concurrentBudget();

  certifyStatic();      // §13.1 중 정적으로 검사 가능한 것
  dynamicGateStubs();   // 시뮬 필요 → TODO
  canonDefects();       // 검사를 쓰면서 드러난 정본 결함

  process.exit(print());
}

main();
