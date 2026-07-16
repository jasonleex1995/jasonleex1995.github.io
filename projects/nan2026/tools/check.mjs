#!/usr/bin/env node
/**
 * ============================================================================
 *  NAN 2026 — check.mjs   (정본 v1.4 §13.4 정적 게이트 S1~S40 + §9.3 로더 규칙)
 * ============================================================================
 *
 *  사용법
 *  ------
 *    node tools/check.mjs                 # 전 검사 실행. 위반 있으면 exit 1
 *    node tools/check.mjs --allow-ambiguous
 *                                         # __AMBIGUOUS__ 만 남았으면 exit 0
 *    node tools/check.mjs --quiet         # 요약만 출력
 *
 *  ★ Node.js 가 없다면
 *  -------------------
 *    이 파일은 Node.js >= 16 (ESM + node: 프로토콜 import) 을 요구한다. 의존성 0, 빌드 스텝 0.
 *
 *      brew install node          # macOS (Homebrew). 그 뒤 `node -v` 로 확인
 *      # Homebrew 자체가 없다면:
 *      #   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
 *      # 또는 https://nodejs.org 의 LTS 설치 파일
 *
 *    ★ node 없이 문법만 확인하려면 macOS 내장 JavaScriptCore 로 파싱할 수 있다:
 *      JSC=/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc
 *      "$JSC" -m tools/check.mjs
 *      → "SyntaxError" 가 나오면 문법 오류다.
 *      → "Module specifier, 'node:fs' is not absolute..." (TypeError) 가 나오면
 *        ★ 파싱은 성공한 것이다 (jsc 는 node: 프로토콜을 모를 뿐이며, 파싱이 해석보다 먼저다).
 *
 *  종료 코드
 *  ---------
 *    0 = 통과   1 = 위반/정본결함/모호 존재   2 = 실행 오류(파일 없음·JSON 파싱 실패)
 *
 *  출력 카테고리 (5종)
 *  -------------------
 *    [VIOLATION]     데이터가 정본을 위반. 고칠 곳 = data/*.json
 *    [CANON]         정본 자신의 결함. 검사를 문면대로 돌리면 정본이 확정한 콘텐츠가
 *                    실패하거나, 검사가 읽을 값이 존재하지 않는다. 고칠 곳 = CANON.md
 *    [AMBIGUOUS]     data 안의 "__AMBIGUOUS__" = 정본이 아직 답하지 않은 자리
 *    [STUB]          시뮬이 필요해 정적으로 검사 불가. 인터페이스만 정본대로 선언
 *    [SKIP]          검사 대상이 아직 없다 (src/ 미존재 등)
 *
 *  ★ 이 파일은 값을 발명하지 않는다 (C-6). 정본이 값을 주지 않은 자리는 검사하지
 *    않고 [CANON] 또는 [STUB] 로 신고한다.
 *
 *  ★★ 공허 통과 방지 (v1.4 신설 — 이 파일이 과거에 실제로 저지른 결함)
 *    `themes` → `stages` 개명(§23.3) 후 참조 45곳이 undefined 가 되어 `|| []` 로
 *    빈 배열을 순회했다 → S8·S9·S20·S22·S23·S26 + 참조 무결성이 **0행에 대해 공허
 *    통과**했다. 콘텐츠를 인증하는 문이 아무것도 안 보고 초록불을 냈다.
 *    → ① `|| []` 를 전부 제거하고 `rows()` 가드로 대체 (0행 = VIOLATION)
 *       ② VACUOUS_WATCH: 게이트가 실제로 검사한 행 수를 세고 0이면 VIOLATION
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
// ★ 공허 통과 감시 — 게이트가 실제로 본 행 수
// ---------------------------------------------------------------------------
const examined = Object.create(null);
/** 게이트가 행 n개를 실제로 검사했음을 기록한다 */
const EX = (check, n) => { examined[check] = (examined[check] || 0) + n; };

/**
 * ★ 콘텐츠를 순회하는 게이트 = 0행이면 그 게이트는 아무것도 인증하지 않았다.
 * 이 목록의 게이트가 0행을 봤으면 VIOLATION 이다 (통과가 아니다).
 */
const VACUOUS_WATCH = [
  'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S13', 'S14', 'S16',
  'S19', 'S20', 'S22', 'S23', 'S24', 'S26', 'S27', 'S28', 'S29',
  'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S39', 'S40',
  'REF',
];

// ---------------------------------------------------------------------------
// 유틸
// ---------------------------------------------------------------------------
const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isAmb = (v) => v === AMB;
const has = (o, k) => isObj(o) && Object.prototype.hasOwnProperty.call(o, k);
const num = (v) => typeof v === 'number' && Number.isFinite(v);

/** 상대 오차 비교 (±pct %) */
function withinPct(actual, target, pct) {
  if (target === 0) return actual === 0;
  return Math.abs(actual - target) / Math.abs(target) <= pct / 100;
}

/**
 * ★★ 0행 가드 — `|| []` 의 대체물.
 * 배열이 아니거나 0행이면 VIOLATION 을 내고 빈 배열을 준다.
 * 「빈 배열을 순회해서 위반이 안 나왔다」는 통과가 아니다.
 */
function rows(check, arr, path, why) {
  if (arr === undefined) {
    V(check, `${path}: 존재하지 않는다 (undefined) — 순회가 0회면 이 게이트는 아무것도 인증하지 않는다. `
      + `개명(§23.3)이나 오타로 참조가 끊긴 자리다. ${why}`);
    return [];
  }
  if (!Array.isArray(arr)) {
    V(check, `${path}: 배열이 아니다 (${arr === null ? 'null' : typeof arr}). ${why}`);
    return [];
  }
  if (arr.length === 0) {
    V(check, `${path}: 0행 — ★ 빈 배열 순회는 통과가 아니라 에러다. 이 게이트가 인증할 콘텐츠가 없다. ${why}`);
    return [];
  }
  return arr;
}

/** rows() 의 조용한 판본 — 이미 다른 게이트가 0행을 신고했을 때 중복 신고를 막는다 */
function rowsQuiet(arr) {
  return Array.isArray(arr) ? arr : [];
}

/**
 * §9.3 로더 규칙: 미지 키 = 에러 / 누락 키 = 에러 / 기본값 폴백 금지.
 * allowed 와 required 가 같은 집합인 것이 정본의 기본값이다
 * (예외는 §9.3이 명시한 weapons/passives 의 levels[] 부분 오버라이드뿐).
 */
function closedKeys(check, obj, allowed, path, opts = {}) {
  const optional = new Set(opts.optional || []);
  if (!isObj(obj)) {
    V(check, `${path}: 객체가 아니다 (실제 ${obj === undefined ? 'undefined' : (Array.isArray(obj) ? 'array' : typeof obj)})`);
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
    V(check, `${path}: 동결 어휘 밖의 값 ${JSON.stringify(value)} — 허용 = [${allowedList.map((x) => JSON.stringify(x)).join(', ')}]`);
    return false;
  }
  return true;
}

/** §9.8.1 — parts[].id 의 첫 글자를 대문자로 (thruster → Thruster, finL → FinL) */
const pascal = (s) => (typeof s === 'string' && s.length ? s[0].toUpperCase() + s.slice(1) : s);

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
// ★ 콘텐츠 인구조사 (v1.4 신설) — 게이트가 읽을 배열이 실제로 존재하고 비어 있지 않은가
//   ★ `themes` → `stages` 개명이 45곳을 undefined 로 만들었을 때 아무도 안 짖었다.
//     이 함수가 그 클래스의 결함을 첫 줄에서 잡는다.
// ---------------------------------------------------------------------------
function census() {
  const need = [
    ['stages.stages', D.stages && D.stages.stages, 7,
      '§9.9 — stages[] = 6테마 + finale. ★ v1.3에서 `themes` → `stages` 로 개명됐다(§23.3)'],
    ['enemies.archetypes', D.enemies && D.enemies.archetypes, 1, '§9.7'],
    ['enemies.emitters', D.enemies && D.enemies.emitters, 1, '§9.7 · §9.8.1(보스 부위 이미터 66개도 여기 산다)'],
    ['bullets.bullets', D.bullets && D.bullets.bullets, 1, '§9.7'],
    ['bosses.bosses', D.bosses && D.bosses.bosses, 1, '§9.8'],
    ['weapons.weapons', D.weapons && D.weapons.weapons, 12, '§9.5 — 12 패밀리 1:1'],
    ['passives.passives', D.passives && D.passives.passives, 12, '§9.6 — 12종'],
    ['stages.phase.crisisWaves', D.stages && D.stages.phase && D.stages.phase.crisisWaves, 12, '§9.9 — 12행'],
  ];
  for (const [path, arr, minRows, why] of need) {
    if (!Array.isArray(arr)) {
      V('S2', `${path}: 배열이 아니다 (${arr === undefined ? '★ undefined — 참조가 끊겼다' : typeof arr}). ${why}`);
      continue;
    }
    if (arr.length < minRows) {
      V('S2', `${path}: ${arr.length}행 < 최소 ${minRows}행 — 게이트가 인증할 콘텐츠가 없다. ${why}`);
    }
  }
  // 테마별 waves/roster 가 비어 있으면 S8·S22·S26 이 공허 통과한다
  for (const t of rowsQuiet(D.stages && D.stages.stages)) {
    if (!isObj(t)) continue;
    rows('S2', t.waves, `stages.stages[${t.id}].waves`, '§8.7 — 웨이브는 순서 리스트다. 0행이면 그 테마는 플레이 불가');
    rows('S2', t.roster, `stages.stages[${t.id}].roster`, '§8.6 — 테마당 정확히 4종(finale 은 15종)');
  }
}

// ---------------------------------------------------------------------------
// 동결 어휘 (§13.4 S3)
// ---------------------------------------------------------------------------
const MOVE_IDS = ['dive', 'weave', 'column', 'strafe', 'anchor', 'orbitDrift', 'charge', 'rearIn'];              // §8.4 (8)
const EMITTER_TYPES = ['straight', 'fan', 'aimed', 'ring', 'spiral', 'laser', 'zone', 'wall'];                   // §8.5 (8)
const FORMATION_IDS = ['lineH', 'columnV', 'vWedge', 'arc', 'pincer', 'scatter'];                                // §8.7 · §9.9.2 (6)
const PART_TYPES = ['mobility', 'armament', 'armor', 'core'];                                                    // §8.12 (4)
const SHAPE_IDS = ['wedge', 'delta', 'hexPod', 'orb', 'cross', 'spike', 'ring', 'slab', 'fin', 'claw', 'dart', 'bulb']; // §9.10 (12)
const TARGET_MODES = ['forward', 'nearest', 'lowestHp', 'densest', 'randomInArena'];                             // §9.5 (5)
const FAMILIES = ['forward', 'fan', 'seeker', 'lance', 'orbit', 'aura', 'mine', 'boomerang', 'barrage', 'omni', 'drone', 'nova']; // §9.5 (12)
const PASSIVE_STATS = ['dmgMul', 'fireRateMul', 'areaMul', 'pierceAdd', 'projCountAdd', 'elementBonusMul',
  'ghostSecOnHit', 'hitBulletClearRadius', 'maxHpAdd', 'moveSpeedMul', 'xpGainMul', 'coinGainMul'];              // §9.6 (12)
const MOVE_PATTERNS = ['sway', 'orbitArc', 'holdCenter'];                                                        // §8.12.1 (3)
const BULLET_SHAPES = ['circle', 'hex'];                                                                         // §9.7 (2)
const BULLET_STATUS = [null, 'slow', 'stun'];                                                                    // §9.7
const SPAWN_EDGES = ['top', 'left', 'right', 'bottom'];                                                          // §8.7
const BOSS_TIERS = ['stage', 'mid', 'final'];                                                                    // §9.8
const RNG_STREAMS = ['theme', 'draft', 'spawn', 'elite', 'drop', 'pattern', 'boss', 'bot'];                      // §10.2 (8)
const BANDS = ['chaff', 'line', 'turret', 'bruiser'];                                                            // §8.6 (4)
// ★ v1.3 신설 어휘 (§13.4-S3)
const FROM_VALUES = ['self', 'part'];                                                                            // §8.5 (2)
const CRISIS_ELEMENT_RULES = ['themePure', 'finaleRotating'];                                                    // §8.10 · §8.16 (2)
const ELEMENTS4 = ['normal', 'fire', 'water', 'grass'];                                                          // §4.1

// §9.5 12행 표 — 패밀리별 base 필수 키 (S34 의 유일한 소유자)
//   ✔ = base 의 필수 키 / ✖ = 계약에 존재하지 않는 키 (선언하면 미지 = 에러)
const FAMILY_COMMON = ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius',
  'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'];
const FAMILY_COMMON_CHECK = {
  //           dmg  cool  count projSp projR life  pierce hitCd tMode
  forward:   ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'],
  fan:       ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'],
  seeker:    ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'],
  lance:     ['dmg', 'cooldownSec', 'count', 'pierce', 'hitCooldownSec', 'targetMode'],
  orbit:     ['dmg', 'projRadius', 'hitCooldownSec'],
  aura:      ['dmg'],
  mine:      ['dmg'],
  boomerang: ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'],
  barrage:   ['dmg', 'cooldownSec', 'targetMode'],
  omni:      ['dmg', 'cooldownSec', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec'],
  drone:     ['dmg', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'],
  nova:      ['dmg'],
};
// §9.5 고유 파라미터 — base 거처 (evo* 아닌 것)
const FAMILY_OWN_BASE = {
  forward:   ['spreadDeg', 'jitterDeg', 'burstCount', 'burstIntervalSec'],
  fan:       ['arcDeg'],
  seeker:    ['turnRateDegSec', 'acquireRadius', 'retargetSec'],
  lance:     ['beamWidthPx', 'chargeSec', 'rangePx'],
  orbit:     ['orbitRadius', 'angularSpeedDegSec', 'bodyCount'],
  aura:      ['radius', 'tickIntervalSec', 'falloff'],
  mine:      ['placeIntervalSec', 'armSec', 'triggerRadius', 'blastRadius', 'maxAlive'],
  boomerang: ['outRangePx', 'returnSpeed', 'canRehit'],
  barrage:   ['strikeIntervalSec', 'strikesPerVolley', 'blastRadius', 'telegraphSec'],
  omni:      ['dirCount', 'dirOffsetDeg', 'rearBias'],
  drone:     ['droneCount', 'anchorOffsets', 'droneFireSec', 'droneRangePx'],
  nova:      ['intervalSec', 'radius', 'expandSec', 'telegraphSec'],
};
// §9.5 고유 파라미터 — evolution.params 거처 (evo* 접두)
const FAMILY_OWN_EVO = {
  forward:   ['evoRampSec', 'evoRampFireRateMul'],
  fan:       ['evoBlastRadius', 'evoSecondaryDmgMul'],
  seeker:    ['evoDistinctTargets', 'evoRetargetOnKill'],
  lance:     ['evoFullHeight'],
  orbit:     ['evoBulletClearCooldownSec'],
  aura:      ['evoPullForce'],
  mine:      ['evoClusterCount', 'evoClusterRadius', 'evoSecondaryDmgMul'],
  boomerang: ['evoChainCount'],
  barrage:   ['evoRadiusMul'],
  omni:      ['evoRingRotDeg'],
  drone:     ['evoTrailDelaySec'],
  nova:      ['evoRing2Radius', 'evoClearBullets', 'evoSecondaryDmgMul'],
};
// §9.5 허용 targetMode (패밀리별). null = targetMode 키 자체가 없다
const FAMILY_TARGET_MODES = {
  forward: ['forward'], fan: ['forward'], seeker: ['nearest', 'lowestHp', 'randomInArena'],
  lance: ['forward', 'nearest'], orbit: null, aura: null, mine: null,
  boomerang: ['forward', 'nearest'], barrage: ['randomInArena', 'densest'],
  omni: null, drone: ['nearest', 'lowestHp', 'forward'], nova: null,
};

// §7.4 텔레그래프 하한 — 3축 (거동별 표 · 탄 상태 · 개체 클래스). ★ 겹치면 max
const TELEGRAPH_FLOOR_BY_TYPE = {              // §7.4 · §8.5 거동별 표
  straight: 0.55, fan: 0.60, aimed: 0.60, ring: 0.60,
  spiral: 0.60, wall: 0.80, zone: 0.90, laser: 1.20,
};
const TELEGRAPH_FLOOR_SLOW_BULLET = 0.80;   // §7.4 "상태이상(slow) 탄"
const TELEGRAPH_FLOOR_MIDBOSS = 1.20;       // §7.4 "중간보스 패턴" (개체 클래스)
const TELEGRAPH_FLOOR_BOSSPART = 1.50;      // §7.4 "보스 부위 패턴" (v1.3 — "대형" → "부위")

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
    SKIP('S1', 'src/core/ 가 아직 없다 → core 순수성 검사 건너뜀 (구현 시작 전)');
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
//  ★ rules.json 루트 키 = 17개 목록 (§9.4 — v1.3에서 render 가 편입돼 17로 정합)
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

  // ★ v1.3: player.hpSegment 삭제 (§23.3 — hud.hpBarSegCount 가 칸을 소유한다)
  closedKeys('S2', r.player, ['hpMax', 'spriteRadius', 'hitboxRadius', 'moveSpeed',
    'moveResponseTau', 'diagonalNormalize', 'iframeSec', 'defenseBase', 'damageFloorRatio',
    'lowHpThreshold', 'lowHpCriticalThreshold', 'magnetRadius', 'startWeaponId', 'startStance',
    'stanceSwitchCooldown', 'stancePersistAcrossStages', 'elementCapPerElement', 'elementCapTotal',
    'weaponSlots', 'passiveSlots', 'lives'], 'rules.player');
  if (has(r.player, 'hpSegment')) {
    V('S2', 'rules.player.hpSegment: 삭제된 키 (§23.3) — 칸당 = hpMax / hud.hpBarSegCount (§2.1)');
  }

  closedKeys('S2', r.status, ['slowMoveSpeedMul', 'stackMode', 'resistAffects'], 'rules.status');
  // §9.4: bomb.stockMax 는 rules.bomb 소유 (상한은 그것이 제한하는 상태와 함께 산다)
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
    // ★ v1.3: finale.armorCoreRatio 삭제 — 유일 소유자 = bosses[].armorCoreRatio (§23.3)
    closedKeys('S2', r.boss.finale, ['partCount', 'armorPartCount', 'exemptRules', 'allowNormalPeripheral'],
      'rules.boss.finale');
    if (has(r.boss.finale, 'armorCoreRatio')) {
      V('S2', 'rules.boss.finale.armorCoreRatio: 삭제된 키 (§23.3) — φ 는 보스 개체의 속성이다. 유일 소유자 = bosses[tetrarch].armorCoreRatio (§9.8)');
    }
  }

  // ★ v1.3: statusBulletSpeedMul 이 visual → fairness 로 이사했다 (§23.3 · §12.4)
  closedKeys('S2', r.fairness, ['minTelegraphSec', 'minStunTelegraphSec', 'maxStunSec', 'maxBulletSpeed',
    'maxAimedBulletSpeed', 'statusBulletSpeedMul', 'minBulletRadiusPx', 'minGapWidthPx', 'minSpawnRadiusPx',
    'maxSimultaneousEnemyBullets', 'enemyConcurrentMax', 'swarmConcurrentMax', 'crisisWaveResidualMax',
    'telegraphConcurrentMaxPerEntity', 'telegraphConcurrentMaxGlobal', 'playerWeaponsExempt'], 'rules.fairness');

  // ★ v1.3: hud.icons 9 → 14 (§9.4.1 — 상점 10항목을 전부 그릴 수 있어야 한다)
  closedKeys('S2', r.hud, ['hitboxAlwaysVisible', 'showElementBudget', 'fontHeroPx', 'fontLargePx',
    'fontMediumPx', 'fontBodyPx', 'fontSmallPx', 'panelPadPx', 'keycapBoxPx', 'bossHpBarH',
    'hpBarSegGapPx', 'xpBarH', 'hpBarSegCount', 'panelCacheDirtyOnly', 'parGhostEnabled',
    'elementMatrixInPanel', 'coinShowsScoreValue', 'noHitIndicator', 'tokenKeycapGatedDisplay',
    'stanceHintTargetsMajorityElement', 'icons'], 'rules.hud');
  if (isObj(r.hud) && Array.isArray(r.hud.icons) && r.hud.icons.length !== 14) {
    V('S2', `rules.hud.icons: ${r.hud.icons.length}종 ≠ 14 (§9.4.1 — v1.3에서 9 → 14. 상점 5항목의 아이콘이 어휘에 없었다)`);
  }
  // §9.4.1: 전 폰트 크기 ≥ visual.text.minPx(14)
  if (isObj(r.hud) && isObj(r.visual) && isObj(r.visual.text) && num(r.visual.text.minPx)) {
    for (const k of ['fontHeroPx', 'fontLargePx', 'fontMediumPx', 'fontBodyPx', 'fontSmallPx']) {
      if (num(r.hud[k]) && r.hud[k] < r.visual.text.minPx) {
        V('S2', `rules.hud.${k} = ${r.hud[k]} < visual.text.minPx(${r.visual.text.minPx}) — §9.4.1`);
      }
    }
  }

  // §9.6.1 — 중첩 맵 · ★ v1.3: pierce → pierceApplies 개명
  closedKeys('S2', r.passiveHooks, FAMILIES, 'rules.passiveHooks');
  if (isObj(r.passiveHooks)) {
    for (const f of FAMILIES) {
      if (!has(r.passiveHooks, f)) continue;
      closedKeys('S2', r.passiveHooks[f], ['rateKey', 'countKey', 'pierceApplies', 'areaKeys'], `rules.passiveHooks.${f}`);
      if (has(r.passiveHooks[f], 'pierce')) {
        V('S2', `rules.passiveHooks.${f}.pierce: 개명된 키 → pierceApplies (§9.6.1/§23.3) — 무기 파라미터 pierce(정수)와 이름이 충돌했다`);
      }
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
    closedKeys('S2', r.palette.element, ELEMENTS4, 'rules.palette.element');
    closedKeys('S2', r.palette.elementCvd, ELEMENTS4, 'rules.palette.elementCvd');
    closedKeys('S2', r.palette.threat, ['enemyBullet', 'telegraph', 'bulletCore', 'outline'], 'rules.palette.threat');
    closedKeys('S2', r.palette.status, ['band'], 'rules.palette.status');
    closedKeys('S2', r.palette.pickup, ['coin', 'xp'], 'rules.palette.pickup');
    closedKeys('S2', r.palette.hud, ['panelBg', 'panelRule', 'textPrimary', 'textDim', 'hpFill'], 'rules.palette.hud');
    closedKeys('S2', r.palette.bg, ['maxSaturation', 'maxLightness', 'cvdMaxLightness',
      'parallaxLayers', 'maxScrollSpeed'], 'rules.palette.bg');
  }

  // §9.4.3 — visual 전 키 인쇄. ★ v1.3: statusBulletSpeedMul 이 빠졌다(→ fairness)
  closedKeys('S2', r.visual, ['iframeBlinkHz', 'stance', 'playerBullet',
    'glyph', 'telegraph', 'band', 'zone', 'timer', 'trail', 'hitFx', 'a11y', 'text'], 'rules.visual');
  if (has(r.visual, 'statusBulletSpeedMul')) {
    V('S2', 'rules.visual.statusBulletSpeedMul: 이사한 키 → rules.fairness.statusBulletSpeedMul (§23.3) — visual 키가 게임플레이 속도를 바꾸면 §9.4.3의 경계가 깨진다');
  }
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
    // ★ v1.3: visual.text.outlineColor 삭제 — 색의 유일한 거처는 palette (§9.4.3)
    closedKeys('S2', r.visual.text, ['family', 'minPx', 'outlinePx'], 'rules.visual.text');
    if (has(r.visual.text, 'outlineColor')) {
      V('S2', 'rules.visual.text.outlineColor: 삭제된 키 (§9.4.3/§23.3) — 캔버스 텍스트 아웃라인 색 = palette.threat.outline');
    }
  }

  closedKeys('S2', r.render, ['playerFxCompositeAlpha', 'killFxCompositeAlpha', 'playerBulletMaxAlpha',
    'playerBulletMaxRadiusPx', 'particleMaxAlpha', 'particleMaxLifeSec', 'fxMinRealMs', 'targetFps',
    'degradeOnFrameMs', 'degradeRecoverFrames'], 'rules.render');

  // ★ v1.3: audio.bgm · busGain.bgm 삭제 (§7.10 BGM 스코프 아웃 — C-10의 마지막 위반)
  closedKeys('S2', r.audio, ['busGain', 'cueRateLimitPerSec'], 'rules.audio');
  if (has(r.audio, 'bgm')) {
    V('S2', 'rules.audio.bgm: 삭제된 키 (§9.4/§23.3) — BGM 스코프 아웃(무음). 40개 값이 전 코퍼스에 0개였다');
  }
  if (isObj(r.audio)) {
    closedKeys('S2', r.audio.busGain, ['sfx'], 'rules.audio.busGain');
    if (has(r.audio.busGain, 'bgm')) V('S2', 'rules.audio.busGain.bgm: 삭제된 키 (§23.3)');
  }

  // --- elements.json (§9.4.4) ---------------------------------------------
  closedKeys('S2', D.elements, ['schemaVersion', 'order', 'investable', 'matrix'], 'elements');
}

// ===========================================================================
//  S2 (계속) — 파일별 스키마
// ===========================================================================
function S2_files() {
  // --- passives.json (§9.6) — ★ v1.3: desc 12행 추가 ----------------------
  closedKeys('S2', D.passives, ['schemaVersion', 'maxLevel', 'stats', 'passives'], 'passives');
  for (const p of rowsQuiet(D.passives.passives)) {
    closedKeys('S2', p, ['id', 'name', 'desc', 'stat', 'values'], `passives[${p && p.id}]`);
  }
  if (Array.isArray(D.passives.passives)) {
    // §9.6 "폐쇄 스탯 어휘 12종, 12 패시브와 1:1"
    const stats = D.passives.passives.map((p) => p && p.stat);
    if (new Set(stats).size !== stats.length) V('S2', 'passives: stat 중복 — §9.6 "12훅 = 12 패시브 1:1"');
    if (D.passives.passives.length !== 12) V('S2', `passives: ${D.passives.passives.length}종 ≠ 12 (§9.6)`);
  }

  // --- bullets.json (§9.7) — ★ v1.3: speed 삭제 (탄 속도는 이미터가 소유) ---
  closedKeys('S2', D.bullets, ['schemaVersion', 'bullets'], 'bullets');
  for (const b of rowsQuiet(D.bullets.bullets)) {
    if (!isObj(b)) continue;
    closedKeys('S2', b, ['id', 'radius', 'hitboxScale', 'dmg', 'shape', 'status',
      'statusDurationSec', 'accel', 'turnRateDegSec', 'retargetSec'], `bullets[${b.id}]`);
    // §9.7 "element 키가 존재하지 않는다 — 스키마가 '적 공격에는 속성이 없다'를 강제한다"
    if (has(b, 'element')) {
      V('S2', `bullets[${b.id}].element: 존재해서는 안 되는 키 — §9.7/§4.1 "적의 공격에는 속성이 없다"`);
    }
    // ★ v1.3 blocker: 이중 거처였다. pelletS 가 4 이미터에서 3 속도로 발사됐다
    if (has(b, 'speed')) {
      V('S2', `bullets[${b.id}].speed: 삭제된 키 (§9.7/§23.3) — 탄 속도의 유일 소유자 = emitters[].speed. `
        + `엔진이 안 읽는 수를 S6이 인증하고 있었다`);
    }
  }

  // --- enemies.json (§9.7) -------------------------------------------------
  closedKeys('S2', D.enemies, ['schemaVersion', 'bands', 'archetypes', 'emitters'], 'enemies');
  if (isObj(D.enemies.bands)) {
    closedKeys('S2', D.enemies.bands, BANDS, 'enemies.bands');
    for (const [bn, bv] of Object.entries(D.enemies.bands)) {
      // §9.7: xpRef 는 chaff 밴드 전용 필드다 (v1.3)
      const allowed = bn === 'chaff'
        ? ['hpMult', 'coinDropChance', 'coin', 'xpRef']
        : ['hpMult', 'coinDropChance', 'coin'];
      closedKeys('S2', bv, allowed, `enemies.bands.${bn}`);
      if (bn !== 'chaff' && has(bv, 'xpRef')) {
        V('S2', `enemies.bands.${bn}.xpRef: chaff 전용 필드다 (§9.7/§23.3) — 두 파생식(swarmXp · 중간보스 xp)이 chaff만 참조한다`);
      }
      // §9.7: bands[].sizePx 는 삭제되었다 (04-R17)
      if (has(bv, 'sizePx')) {
        V('S2', `enemies.bands.${bn}.sizePx: 삭제된 키 — 크기의 단일 진실 = archetypes[].radius (§9.7, 04-R17)`);
      }
    }
  }
  for (const a of rowsQuiet(D.enemies.archetypes)) {
    if (!isObj(a)) continue;
    closedKeys('S2', a, ['id', 'name', 'desc', 'band', 'shapeId', 'radius', 'moveId', 'moveParams',
      'attack', 'contactDmg', 'hp', 'xp', 'score', 'themeOnly'], `enemies.archetypes[${a.id}]`);
    // §9.7: 삭제 확정된 필드들. ★ unlockStageMin 은 v1.3에서 실제로 삭제됐다(유일 거처 = roster[])
    for (const dead of ['tier', 'element', 'hpScalePerStage', 'spriteId', 'unlockStageMin']) {
      if (has(a, dead)) {
        V('S2', `enemies.archetypes[${a.id}].${dead}: 삭제된 키 (§9.7)`
          + (dead === 'unlockStageMin' ? ' — 유일한 거처 = stages[].roster[] (04-R6)' : ''));
      }
    }
    if (a.attack !== null && isObj(a.attack)) {
      closedKeys('S2', a.attack, ['emitterId', 'firstDelaySec'], `enemies.archetypes[${a.id}].attack`);
    }
  }
  // §8.5 이미터 — 공통 8 + 타입별 고유. ★ v1.3: laser.chargeSec 삭제 (충전이 곧 텔레그래프다)
  const EMIT_COMMON = ['id', 'type', 'bulletId', 'from', 'telegraphSec', 'everySec', 'offsetSec', 'repeat', 'restSec'];
  const EMIT_OWN = {
    straight: ['count', 'spreadDeg', 'speed'],
    fan: ['count', 'arcDeg', 'speed'],
    aimed: ['count', 'spreadDeg', 'speed', 'leadSec'],
    ring: ['count', 'speed', 'rotOffsetDeg'],
    spiral: ['count', 'speed', 'rotStepDeg', 'durationSec', 'rateSec'],
    laser: ['widthPx', 'activeSec', 'angleDeg', 'trackDuringCharge'],
    zone: ['radius', 'activeSec', 'dmg'],
    wall: ['count', 'gapCount', 'gapWidthPx', 'speed'],
  };
  for (const e of rowsQuiet(D.enemies.emitters)) {
    if (!isObj(e)) continue;
    if (!vocab('S3', e.type, EMITTER_TYPES, `enemies.emitters[${e.id}].type`)) continue;
    closedKeys('S2', e, [...EMIT_COMMON, ...(EMIT_OWN[e.type] || [])], `enemies.emitters[${e.id}]`);
    // §8.5 "zone 의 dps 는 존재하지 않는다 — 모든 피해는 적용 1회"
    if (has(e, 'dps')) V('S2', `enemies.emitters[${e.id}].dps: 존재하지 않는 키 — §8.5 "모든 피해는 적용 1회"`);
    if (e.type === 'laser' && has(e, 'chargeSec')) {
      V('S2', `enemies.emitters[${e.id}].chargeSec: 삭제된 키 (§8.5/§23.3) — 충전이 곧 텔레그래프다. telegraphSec 하나로 통일`);
    }
  }

  // --- weapons.json (§9.5) -------------------------------------------------
  closedKeys('S2', D.weapons, ['schemaVersion', 'weapons'], 'weapons');
  for (const w of rowsQuiet(D.weapons.weapons)) {
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

    // §4.4 elementStampMode — 구조 결정 = 잠금 키
    const stampLive = ['orbit', 'aura'];
    if (FAMILIES.includes(w.family)) {
      const wantStamp = stampLive.includes(w.family) ? 'live' : 'spawn';
      if (w.elementStampMode !== wantStamp) {
        V('S2', `weapons[${w.id}].elementStampMode = ${JSON.stringify(w.elementStampMode)} ≠ "${wantStamp}" — §4.4/§9.5 표`);
      }
    }
  }

  // --- bosses.json (§9.8 · §9.8.2) — union 타입, tier 로 갈라진다 -----------
  closedKeys('S2', D.bosses, ['schemaVersion', 'bosses'], 'bosses');
  for (const b of rowsQuiet(D.bosses.bosses)) {
    if (!isObj(b)) continue;
    vocab('S3', b.tier, BOSS_TIERS, `bosses[${b.id}].tier`);
    if (b.tier === 'mid') {
      // §9.8.2 — 중간보스 팔 (v1.3 신설). hp·element 는 루트 필드다 (core 가 없다)
      closedKeys('S2', b, ['id', 'name', 'tier', 'themeId', 'hp', 'element', 'radius', 'contactDmg',
        'shapeId', 'moveId', 'moveParams', 'patternSet', 'summon', 'parts',
        'xp', 'coin', 'healDropChance', 'score'], `bosses[${b.id}]`);
      if (has(b, 'core')) {
        V('S2', `bosses[${b.id}].core: 중간보스에는 core 가 없다 (§9.8.2-ⓐ) — parts: [] 이므로 "부위와 구별되는 몸통"이 정의되지 않는다`);
      }
      // §8.9 v1.2 정정: bosses[].leaveAfterSec 는 삭제되었다
      if (has(b, 'leaveAfterSec')) {
        V('S2', `bosses[${b.id}].leaveAfterSec: 삭제된 키 — 유일 소유자 = stages.phase.midBossLeaveAfterSec (§8.9)`);
      }
    } else {
      // §9.8 — 스테이지·최종 보스 팔
      closedKeys('S2', b, ['id', 'name', 'tier', 'themeId', 'armorCoreRatio', 'core', 'parts',
        'movePattern', 'movePatternParams', 'summon'], `bosses[${b.id}]`);
      // §9.8: xp 는 tier == "mid" 전용 필드다 (v1.3)
      if (has(b, 'xp')) {
        V('S2', `bosses[${b.id}].xp: tier=="mid" 전용 필드다 (§9.8/§23.3) — 스테이지·최종 보스의 보상은 rules.boss.coin/partCoin + core.score/parts[].score 가 소유한다`);
      }
      if (isObj(b.core)) {
        closedKeys('S2', b.core, ['element', 'hp', 'radius', 'contactDmg', 'shapeId', 'score'], `bosses[${b.id}].core`);
      }
      if (isObj(b.movePatternParams)) {
        closedKeys('S2', b.movePatternParams, ['speedPxSec', 'ampPx', 'yHoldPx'], `bosses[${b.id}].movePatternParams`);
      }
      for (const p of rowsQuiet(b.parts)) {
        if (!isObj(p)) continue;
        closedKeys('S2', p, ['id', 'name', 'partType', 'element', 'hp', 'radius', 'anchor',
          'contactDmg', 'shapeId', 'score', 'patternSet'], `bosses[${b.id}].parts[${p.id}]`);
        // §9.8: 존재하지 않는 키들
        for (const dead of ['regenSec', 'onDestroy', 'hpShare', 'xp']) {
          if (has(p, dead)) V('S2', `bosses[${b.id}].parts[${p.id}].${dead}: 존재하지 않는 키 (§9.8)`);
        }
        for (const ps of rowsQuiet(p.patternSet)) {
          closedKeys('S2', ps, ['emitterIds'], `bosses[${b.id}].parts[${p.id}].patternSet[]`);
          if (has(ps, 'emitterId')) {
            V('S2', `bosses[${b.id}].parts[${p.id}].patternSet[].emitterId: 개명된 키 → emitterIds: [...] (§9.8/§23.3)`);
          }
        }
      }
    }
    if (isObj(b.summon)) {
      closedKeys('S2', b.summon, ['archetypeId', 'count', 'everySec', 'formationId'], `bosses[${b.id}].summon`);
    }
    for (const ps of rowsQuiet(b.patternSet)) {
      closedKeys('S2', ps, ['emitterIds'], `bosses[${b.id}].patternSet[]`);
    }
  }

  // --- stages.json (§9.9) — ★ v1.3: themes → stages 개명 -------------------
  closedKeys('S2', D.stages, ['schemaVersion', 'themeDraw', 'curve', 'phase', 'stages', 'formations'], 'stages');
  if (has(D.stages, 'themes')) {
    V('S2', 'stages.themes: 개명된 키 → stages.stages (§9.9/§23.3) — 파일 이름이 stages.json 이고 게이트가 stages[] 라 부른다');
  }
  closedKeys('S2', D.stages.themeDraw, ['pool', 'count', 'allowRepeat', 'stage1RequiresIntroOk', 'finalStageId'], 'stages.themeDraw');
  closedKeys('S2', D.stages.curve, ['enemyHpScale', 'xpScale', 'bossHpScale', 'spawnDensityScale',
    'midBossCount', 'elitePerWaveChance', 'swarmTotalScale', 'rearSpawnAllowed'], 'stages.curve');
  // §9.9 v1.3: crisisPerStage · crisisWaves · midBossAtSec 신설 / bossEntrySec · crisisElementRule 삭제
  closedKeys('S2', D.stages.phase, ['mobPhaseSec', 'mobPhaseSkippable', 'mobPhaseMaxWaves', 'waveIntervalSec',
    'waveClearAdvance', 'mobPhaseExitFadeSec', 'mobPhaseExitClearBullets', 'phaseEndAutocollect',
    'enemyExitForfeitsReward', 'waveListExhausted', 'crisisPerStage', 'crisisStartSec', 'crisisDurationSec',
    'crisisWarnSec', 'crisisSuspendsWaves', 'crisisTotal', 'crisisSubWaves', 'crisisWaves',
    'midBossAtSec', 'midBossLeaveAfterSec', 'midBossElementRule', 'midBossForcedLeaveOnCrisis',
    'bossTimerSec', 'timerWarnSec', 'timerRedAlertSec', 'statusStunMaxPerStage'], 'stages.phase');
  if (has(D.stages.phase, 'bossEntrySec')) {
    V('S2', 'stages.phase.bossEntrySec: 삭제된 키 (§9.9/§23.3) — 유일 소유자 = rules.boss.introSec (§6.3)');
  }
  if (has(D.stages.phase, 'crisisElementRule')) {
    V('S2', 'stages.phase.crisisElementRule: 이사한 키 → stages[].crisisElementRule (2값 어휘, §23.3)');
  }
  // §9.9.3: crisisSubWaveIntervalSec 은 파생값이지 키가 아니다 (새 키 0)
  if (has(D.stages.phase, 'crisisSubWaveIntervalSec')) {
    V('S2', 'stages.phase.crisisSubWaveIntervalSec: 파생값이지 키가 아니다 — §9.9.3 (= crisisDurationSec / crisisSubWaves)');
  }
  for (const cw of rowsQuiet(D.stages.phase && D.stages.phase.crisisWaves)) {
    closedKeys('S2', cw, ['subWave', 'formationId', 'archetypeId', 'count', 'spawnEdge'], 'stages.phase.crisisWaves[]');
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
  // §9.9 stages[] — ★ skinId · elitesAtSec · midBossAtSec · finaleCrisisRotating 삭제/이사
  for (const t of rowsQuiet(D.stages.stages)) {
    if (!isObj(t)) continue;
    closedKeys('S2', t, ['id', 'name', 'element', 'introOk', 'bossId', 'crisisElementRule',
      'roster', 'mix', 'mixGranularity', 'waves'], `stages.stages[${t.id}]`);
    for (const [dead, why] of [
      ['skinId', 'id 와 같다 → 삭제 (§9.9-⑥)'],
      ['elitesAtSec', '죽은 키 → 삭제 (§8.7)'],
      ['midBossAtSec', '이사 → stages.phase.midBossAtSec (스테이지 인덱스 배열, §8.9)'],
      ['finaleCrisisRotating', '이사 → stages[].crisisElementRule = "finaleRotating" (2값 어휘, §8.10)'],
    ]) {
      if (has(t, dead)) V('S2', `stages.stages[${t.id}].${dead}: ${why} (§23.3)`);
    }
    for (const rEnt of rowsQuiet(t.roster)) {
      closedKeys('S2', rEnt, ['archetypeId', 'unlockStageMin'], `stages.stages[${t.id}].roster[${rEnt && rEnt.archetypeId}]`);
    }
    // ★ v1.3: waves[] = 7필드 (unlockStageMin 신설 — S8의 통과 여부가 미정이었다)
    rowsQuiet(t.waves).forEach((w, i) => {
      closedKeys('S2', w, ['formationId', 'archetypeId', 'count', 'element', 'spawnEdge', 'eliteIndex',
        'unlockStageMin'], `stages.stages[${t.id}].waves[${i}]`);
      // §8.7: atSec 절대 타임라인은 폐기 — 순서 리스트다
      if (has(w, 'atSec')) {
        V('S2', `stages.stages[${t.id}].waves[${i}].atSec: 폐기된 키 — 웨이브는 순서 리스트다 (§8.7)`);
      }
    });
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
  // §11.3 — 점수. ★ v1.3: difficultyMul → difficulty[].scoreMul (거처는 meta.difficulty)
  closedKeys('S2', D.meta.score, ['superEffectiveDamageShare', 'superEffectiveKillBonusRatio', 'attribution',
    'timeBonusPerGameSec', 'bossClearBonus', 'midBossClearBonus', 'runClearBonus', 'noHitScope',
    'stageNoHitBonus', 'perfectScope', 'perfectBonus', 'shieldPreservesNoHit',
    'timeTokenForfeitsTimeBonus', 'coinToScore', 'roundMode'], 'meta.score');
  if (has(D.meta.score, 'difficultyMul')) {
    V('S2', 'meta.score.difficultyMul: 개명된 키 → meta.difficulty[].scoreMul (§23.5-05)');
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
  // §10.4 — bot. ★ grazeTolerancePx 는 삭제됐다 (§2.3 "그레이즈 없음")
  if (isObj(D.meta.bot)) {
    closedKeys('S2', D.meta.bot, ['reactionMs', 'reactionJitterMs', 'stanceSwitchMs', 'dodgeLookaheadSec',
      'aimErrorPx', 'slotOrder', 'policies', 'baseline', 'probes'], 'meta.bot');
    closedKeys('S2', D.meta.bot.policies, ['draft', 'farm', 'stance', 'shop'], 'meta.bot.policies');
    closedKeys('S2', D.meta.bot.baseline, ['draft', 'farm', 'stance', 'shop'], 'meta.bot.baseline');
    closedKeys('S2', D.meta.bot.probes, ['dpsProbe', 'forceNoElement'], 'meta.bot.probes');
    if (has(D.meta.bot, 'grazeTolerancePx')) {
      V('S2', 'meta.bot.grazeTolerancePx: 삭제된 키 (§10.4) — §2.3 "그레이즈 없음 (확정)"');
    }
  }
}

// ---------------------------------------------------------------------------
//  참조 무결성 (§9.3 "모든 *Id 는 로드 시 대상 존재 확인")
//  ★ 이 함수가 `themes` → `stages` 개명 때 통째로 공허 통과했다 → rows() 가드
// ---------------------------------------------------------------------------
function refIntegrity() {
  const archIds = new Set(rowsQuiet(D.enemies.archetypes).map((a) => a && a.id));
  const emitIds = new Set(rowsQuiet(D.enemies.emitters).map((e) => e && e.id));
  const bulletIds = new Set(rowsQuiet(D.bullets.bullets).map((b) => b && b.id));
  const bossIds = new Set(rowsQuiet(D.bosses.bosses).map((b) => b && b.id));
  const weaponIds = new Set(rowsQuiet(D.weapons.weapons).map((w) => w && w.id));
  const stageIds = new Set(rowsQuiet(D.stages.stages).map((t) => t && t.id));
  const formIds = new Set(Object.keys(D.stages.formations || {}));

  let n = 0;
  const need = (set, id, where) => {
    if (id === null || id === undefined || isAmb(id)) return;
    n += 1;
    if (!set.has(id)) V('REF', `${where}: 참조 무결성 실패 — "${id}" 가 존재하지 않는다 (§9.3)`);
  };

  for (const a of rowsQuiet(D.enemies.archetypes)) {
    if (isObj(a && a.attack)) need(emitIds, a.attack.emitterId, `enemies.archetypes[${a.id}].attack.emitterId`);
    if (isObj(a) && a.themeOnly !== null) need(stageIds, a.themeOnly, `enemies.archetypes[${a.id}].themeOnly`);
  }
  for (const e of rowsQuiet(D.enemies.emitters)) {
    if (isObj(e) && e.bulletId !== null) need(bulletIds, e.bulletId, `enemies.emitters[${e.id}].bulletId`);
  }
  for (const b of rowsQuiet(D.bosses.bosses)) {
    if (!isObj(b)) continue;
    if (isObj(b.summon)) {
      need(archIds, b.summon.archetypeId, `bosses[${b.id}].summon.archetypeId`);
      need(formIds, b.summon.formationId, `bosses[${b.id}].summon.formationId`);
    }
    for (const ps of rowsQuiet(b.patternSet)) {
      for (const id of rowsQuiet(ps && ps.emitterIds)) need(emitIds, id, `bosses[${b.id}].patternSet.emitterIds`);
    }
    for (const p of rowsQuiet(b.parts)) {
      for (const ps of rowsQuiet(p && p.patternSet)) {
        for (const id of rowsQuiet(ps && ps.emitterIds)) {
          need(emitIds, id, `bosses[${b.id}].parts[${p.id}].patternSet.emitterIds`);
        }
      }
    }
    if (b.themeId !== null) need(stageIds, b.themeId, `bosses[${b.id}].themeId`);
  }
  // ★ 0행이면 여기가 통째로 공허 통과한다 — rows() 가 짖는다
  for (const t of rows('REF', D.stages.stages, 'stages.stages',
    '§9.9 — 참조 무결성이 0행에 대해 공허 통과하면 bossId·archetypeId·formationId 를 아무도 안 본다')) {
    if (!isObj(t)) continue;
    need(bossIds, t.bossId, `stages.stages[${t.id}].bossId`);
    for (const r of rowsQuiet(t.roster)) need(archIds, r && r.archetypeId, `stages.stages[${t.id}].roster.archetypeId`);
    rowsQuiet(t.waves).forEach((w, i) => {
      need(archIds, w && w.archetypeId, `stages.stages[${t.id}].waves[${i}].archetypeId`);
      need(formIds, w && w.formationId, `stages.stages[${t.id}].waves[${i}].formationId`);
    });
  }
  for (const cw of rowsQuiet(D.stages.phase && D.stages.phase.crisisWaves)) {
    need(archIds, cw && cw.archetypeId, 'stages.phase.crisisWaves[].archetypeId');
    need(formIds, cw && cw.formationId, 'stages.phase.crisisWaves[].formationId');
  }
  need(weaponIds, D.rules.player && D.rules.player.startWeaponId, 'rules.player.startWeaponId');
  for (const id of rowsQuiet(D.rules.boss && D.rules.boss.midBossSummonsAllowed)) {
    need(bossIds, id, 'rules.boss.midBossSummonsAllowed');
  }
  for (const t of rowsQuiet(D.stages.themeDraw && D.stages.themeDraw.pool)) {
    need(stageIds, t, 'stages.themeDraw.pool');
  }
  need(stageIds, D.stages.themeDraw && D.stages.themeDraw.finalStageId, 'stages.themeDraw.finalStageId');
  EX('REF', n);
}

// ---------------------------------------------------------------------------
//  공용 접근자
// ---------------------------------------------------------------------------
const FINAL = () => (isObj(D.stages.themeDraw) ? D.stages.themeDraw.finalStageId : undefined);
const STAGES = () => rowsQuiet(D.stages.stages);
const BOSSES = () => rowsQuiet(D.bosses.bosses);
const EMITTERS = () => rowsQuiet(D.enemies.emitters);
const ARCHETYPES = () => rowsQuiet(D.enemies.archetypes);

/** 보스 부위 patternSet 에서 참조되는 이미터 id 집합 (§9.8.1) */
function bossPartEmitterIds() {
  const s = new Set();
  for (const b of BOSSES()) {
    if (!isObj(b) || b.tier === 'mid') continue;
    for (const p of rowsQuiet(b.parts)) {
      for (const ps of rowsQuiet(p && p.patternSet)) {
        for (const id of rowsQuiet(ps && ps.emitterIds)) s.add(id);
      }
    }
  }
  return s;
}
/** 중간보스 patternSet 에서 참조되는 이미터 id 집합 */
function midBossEmitterIds() {
  const s = new Set();
  for (const b of BOSSES()) {
    if (!isObj(b) || b.tier !== 'mid') continue;
    for (const ps of rowsQuiet(b.patternSet)) {
      for (const id of rowsQuiet(ps && ps.emitterIds)) s.add(id);
    }
  }
  return s;
}
/** 잡몹 attack 에서 참조되는 이미터 id 집합 */
function mobEmitterIds() {
  const s = new Set();
  for (const a of ARCHETYPES()) {
    if (isObj(a) && isObj(a.attack)) s.add(a.attack.emitterId);
  }
  return s;
}

// ===========================================================================
//  S3 — 어휘 (동결 목록 밖의 값 = 실패)
//  §13.4-S3: moveId(8) · emitterType(8) · formationId(6) · partType(4) · shapeId(12)
//            · targetMode(5) · family(12) · passive.stat(12) · movePattern(3)
//            · bullets[].shape(2) · ★ emitters[].from(2) · ★ archetypes[].themeOnly(6+null)
//            · ★ stages[].crisisElementRule(2)
//            + ★ tier=="mid" 의 다중 동치 (§9.8.2)
// ===========================================================================
function S3_vocab() {
  let n = 0;
  for (const a of ARCHETYPES()) {
    if (!isObj(a)) continue;
    n += 1;
    vocab('S3', a.moveId, MOVE_IDS, `enemies.archetypes[${a.id}].moveId`);
    vocab('S3', a.shapeId, SHAPE_IDS, `enemies.archetypes[${a.id}].shapeId`);
    vocab('S3', a.band, BANDS, `enemies.archetypes[${a.id}].band`);
    // ★ v1.3: themeOnly 어휘 폐쇄 = stages[6] 중 하나 | null. finale 은 값이 될 수 없다
    if (a.themeOnly !== null && !isAmb(a.themeOnly)) {
      const pool = rowsQuiet(D.stages.themeDraw && D.stages.themeDraw.pool);
      if (!pool.includes(a.themeOnly)) {
        V('S3', `enemies.archetypes[${a.id}].themeOnly = ${JSON.stringify(a.themeOnly)} — 허용 = themeDraw.pool 6테마 | null (§9.7). `
          + `finale 은 값이 될 수 없다 (최종 전용 아키타입이 존재하지 않는다, §8.16)`);
      }
    }
  }
  for (const e of EMITTERS()) {
    if (!isObj(e)) continue;
    n += 1;
    vocab('S3', e.type, EMITTER_TYPES, `enemies.emitters[${e.id}].type`);
    vocab('S3', e.from, FROM_VALUES, `enemies.emitters[${e.id}].from`);   // ★ v1.3 신설 (§8.5)
  }
  for (const b of rowsQuiet(D.bullets.bullets)) {
    if (!isObj(b)) continue;
    n += 1;
    vocab('S3', b.shape, BULLET_SHAPES, `bullets[${b.id}].shape`);
    if (!isAmb(b.status) && !BULLET_STATUS.includes(b.status)) {
      V('S3', `bullets[${b.id}].status = ${JSON.stringify(b.status)} — 허용 = null | "slow" | "stun" (§9.7)`);
    }
  }
  for (const w of rowsQuiet(D.weapons.weapons)) {
    if (!isObj(w)) continue;
    n += 1;
    vocab('S3', w.family, FAMILIES, `weapons[${w.id}].family`);
    if (isObj(w.base) && has(w.base, 'targetMode')) {
      vocab('S3', w.base.targetMode, TARGET_MODES, `weapons[${w.id}].base.targetMode`);
    }
  }
  for (const p of rowsQuiet(D.passives.passives)) {
    if (!isObj(p)) continue;
    n += 1;
    vocab('S3', p.stat, PASSIVE_STATS, `passives[${p.id}].stat`);
  }
  for (const b of BOSSES()) {
    if (!isObj(b)) continue;
    n += 1;
    if (b.tier === 'mid') {
      vocab('S3', b.moveId, MOVE_IDS, `bosses[${b.id}].moveId`);
      vocab('S3', b.shapeId, SHAPE_IDS, `bosses[${b.id}].shapeId`);
    } else {
      vocab('S3', b.movePattern, MOVE_PATTERNS, `bosses[${b.id}].movePattern`);
      if (isObj(b.core)) vocab('S3', b.core.shapeId, SHAPE_IDS, `bosses[${b.id}].core.shapeId`);
    }
    for (const p of rowsQuiet(b.parts)) {
      if (!isObj(p)) continue;
      vocab('S3', p.partType, PART_TYPES, `bosses[${b.id}].parts[${p.id}].partType`);
      vocab('S3', p.shapeId, SHAPE_IDS, `bosses[${b.id}].parts[${p.id}].shapeId`);
    }
    // ★★ §9.8.2 의 다중 동치 — union 타입의 두 팔을 기계적으로 분리한다
    //    tier=="mid" ⟺ moveId 보유 ⟺ movePattern 부재 ⟺ parts==[] ⟺ armorCoreRatio 부재 ⟺ 보상 4필드 보유
    const isMid = b.tier === 'mid';
    const eq = [
      ['moveId 보유', has(b, 'moveId')],
      ['movePattern 부재', !has(b, 'movePattern')],
      ['parts == []', Array.isArray(b.parts) && b.parts.length === 0],
      ['armorCoreRatio 부재', !has(b, 'armorCoreRatio')],
      ['보상 4필드 보유', ['xp', 'coin', 'healDropChance', 'score'].every((k) => has(b, k))],
    ];
    for (const [label, val] of eq) {
      if (val !== isMid) {
        V('S3', `bosses[${b.id}]: §9.8.2 다중 동치 위반 — (tier=="mid")=${isMid} ≠ (${label})=${val}. `
          + `하나라도 어긋나면 로드 실패다 (S15·S32와 같은 클래스)`);
      }
    }
  }
  for (const t of STAGES()) {
    if (!isObj(t)) continue;
    n += 1;
    // ★ v1.3: crisisElementRule 이 불리언에서 2값 어휘가 됐다 (§8.10)
    vocab('S3', t.crisisElementRule, CRISIS_ELEMENT_RULES, `stages.stages[${t.id}].crisisElementRule`);
    rowsQuiet(t.waves).forEach((w, i) => {
      if (!isObj(w)) return;
      vocab('S3', w.formationId, FORMATION_IDS, `stages.stages[${t.id}].waves[${i}].formationId`);
      vocab('S3', w.spawnEdge, SPAWN_EDGES, `stages.stages[${t.id}].waves[${i}].spawnEdge`);
      vocab('S3', w.element, ELEMENTS4, `stages.stages[${t.id}].waves[${i}].element`);
    });
  }
  for (const cw of rowsQuiet(D.stages.phase && D.stages.phase.crisisWaves)) {
    if (!isObj(cw)) continue;
    vocab('S3', cw.formationId, FORMATION_IDS, 'stages.phase.crisisWaves[].formationId');
    vocab('S3', cw.spawnEdge, SPAWN_EDGES, 'stages.phase.crisisWaves[].spawnEdge');
  }
  // §9.6 stats 어휘 = 12종 폐쇄
  for (const s of rowsQuiet(D.passives.stats)) vocab('S3', s, PASSIVE_STATS, 'passives.stats');
  EX('S3', n);
}

// ===========================================================================
//  S4 — 아키타입 겹침 금지 (§8.6)
//  (moveId, emitterType) 쌍 중복 시 실패. band 가 다르면 허용
// ===========================================================================
function S4_archetypeOverlap() {
  const emitById = new Map(EMITTERS().map((e) => [e && e.id, e]));
  const seen = new Map();
  let n = 0;
  for (const a of ARCHETYPES()) {
    if (!isObj(a)) continue;
    n += 1;
    const et = isObj(a.attack) ? (emitById.get(a.attack.emitterId) || {}).type : null;
    const key = `${a.band}|${a.moveId}|${et === undefined ? 'UNRESOLVED' : et}`;
    if (seen.has(key)) {
      V('S4', `아키타입 겹침: "${a.id}" 와 "${seen.get(key)}" 가 같은 (band=${a.band}, moveId=${a.moveId}, emitterType=${et}) — §8.6/S4`);
    } else seen.set(key, a.id);
  }
  EX('S4', n);
}

// ===========================================================================
//  S5 — 보스 R1~R7 전부 + partCount + armor 수 + tier:"final" 3중 동치 (§8.14 · §8.16)
// ===========================================================================
function S5_bossRules() {
  const rb = D.rules.boss;
  if (!isObj(rb)) return;
  const stageById = new Map(STAGES().map((t) => [t && t.id, t]));
  const FINAL_ID = FINAL();
  const finaleStage = stageById.get(FINAL_ID);
  const exempt = new Set(rowsQuiet(rb.finale && rb.finale.exemptRules));
  let n = 0;

  for (const b of BOSSES()) {
    if (!isObj(b) || b.tier === 'mid') continue;
    n += 1;
    const isFinal = b.tier === 'final';
    const tag = `bosses[${b.id}]`;
    const parts = rowsQuiet(b.parts);
    const armor = parts.filter((p) => isObj(p) && p.partType === 'armor');
    const periph = parts;                              // 주변부 = core 를 뺀 전부
    const theme = (b.themeId === null || isAmb(b.themeId)) ? null : stageById.get(b.themeId);
    const themeEl = theme ? theme.element : null;

    // partCount — core 를 포함한다 (§13.6.2 "armor 2 + 선택 1 + core 1 = partCount 4")
    const wantPartCount = isFinal ? (rb.finale && rb.finale.partCount) : rb.partCount;
    if (num(wantPartCount) && parts.length + 1 !== wantPartCount) {
      V('S5', `${tag}: partCount = ${parts.length + 1}(부위 ${parts.length} + core) ≠ ${wantPartCount} (§8.11/§8.16)`);
    }

    // R1: core 속성 = 항상 노말
    if (isObj(b.core) && b.core.element !== rb.coreElement) {
      V('S5', `${tag}: R1 위반 — core.element = ${JSON.stringify(b.core.element)} ≠ ${JSON.stringify(rb.coreElement)} (§8.14)`);
    }
    // core 는 정확히 1개 (§8.12) — parts[] 에 core 가 있으면 안 된다
    if (parts.some((p) => isObj(p) && p.partType === 'core')) {
      V('S5', `${tag}: parts[] 에 partType "core" — core 는 bosses[].core 가 유일한 자리 (§9.8)`);
    }

    // R2: 주변부 속성에 노말 금지 (최종의 allowNormalPeripheral 은 왕좌에만)
    const allowNormalPeriph = isFinal && rb.finale && rb.finale.allowNormalPeripheral === true;
    const normalPeriph = periph.filter((p) => isObj(p) && p.element === 'normal');
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
    if (!(isFinal && exempt.has('R3'))) {
      const distinct = new Set(periph.map((p) => isObj(p) ? p.element : null).filter((e) => e !== null && !isAmb(e)));
      if (num(rb.partElementDistinctMin) && distinct.size < rb.partElementDistinctMin) {
        V('S5', `${tag}: R3 위반 — 주변부 distinct 속성 ${distinct.size} < ${rb.partElementDistinctMin} (§8.14)`);
      }
    }

    // R4: 테마 속성은 최대 1개 부위 (최종 면제 — exemptRules 에 R4)
    if (!(isFinal && exempt.has('R4'))) {
      if (themeEl === null || themeEl === undefined) {
        if (!isFinal) C('S5', `${tag}: R4 를 평가할 테마 속성이 없다 (themeId=${JSON.stringify(b.themeId)})`);
      } else if (num(rb.partThemeElementMax)) {
        const cnt = periph.filter((p) => isObj(p) && p.element === themeEl).length;
        if (cnt > rb.partThemeElementMax) {
          V('S5', `${tag}: R4 위반 — 테마 속성(${themeEl}) 부위 ${cnt} > ${rb.partThemeElementMax} (§8.14)`);
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

    // R7: φ ∈ [0.85·B, B), B = coreGateMul^-a − 1.  ★ R7 은 면제하지 않는다 (§8.16)
    const a = armor.length;
    const phi = b.armorCoreRatio;
    if (num(phi) && a > 0 && num(rb.coreGateMul) && Array.isArray(rb.armorCoreRatioBandPct)) {
      const B = Math.pow(rb.coreGateMul, -a) - 1;
      const lo = rb.armorCoreRatioBandPct[0] * B;
      const okLo = phi >= lo - 1e-9;
      const okHi = rb.armorCoreRatioBandPct[1] === 1.0
        ? phi < B
        : phi <= rb.armorCoreRatioBandPct[1] * B + 1e-9;
      if (!okLo || !okHi) {
        V('S5', `${tag}: R7 위반 — φ = ${phi} ∉ [${lo.toFixed(3)}, ${B.toFixed(3)}) (a=${a}, B=${rb.coreGateMul}^-${a}−1) (§8.13.1)`);
      }
    } else if (!num(phi) && !isAmb(phi)) {
      V('S5', `${tag}.armorCoreRatio: tier ∈ {stage, final} 에 필수 (§9.8)`);
    }

    // tier == "final" ⟺ finale 스테이지 전용 ⟺ bossHpScale 미적용 (3중 동치)
    if (isFinal) {
      if (!finaleStage) {
        C('S5', `tier:"final" 보스 "${b.id}" 가 있으나 stages.stages 에 finalStageId(${JSON.stringify(FINAL_ID)}) 엔트리가 없다`);
      } else if (finaleStage.bossId !== b.id) {
        V('S5', `${tag}: tier:"final" 인데 finale.bossId = ${JSON.stringify(finaleStage.bossId)} — 3중 동치 위반 (S5)`);
      }
    } else if (finaleStage && finaleStage.bossId === b.id) {
      V('S5', `${tag}: finale 의 보스인데 tier = ${JSON.stringify(b.tier)} ≠ "final" — 3중 동치 위반 (S5)`);
    }
    // 테마 보스가 finale 테마를 참조하면 안 된다
    if (!isFinal && b.themeId === FINAL_ID) {
      V('S5', `${tag}: tier:"stage" 인데 themeId = finale (S5)`);
    }
  }
  EX('S5', n);
  // §8.11: 스테이지 보스는 소환하지 않는다
  if (rb.summonsAllowed !== false) {
    V('S5', `rules.boss.summonsAllowed = ${rb.summonsAllowed} ≠ false (§8.11)`);
  }
}

// ===========================================================================
//  S6 — 공정성 (§12.4 · §7.4)
//  ★ enemies.json > emitters 만 검사한다 (fairness.playerWeaponsExempt, §9.5)
//  ★ telegraphSec 는 3축(거동별 표 · 탄 상태 · 개체 클래스)의 max 를 만족해야 한다 (§7.4)
//  ★ v1.3: minSpawnRadiusPx 는 S6에서 뺐다 — 발사 시점 플레이어 위치의 함수라 정적 검사 불가
//          → certify.static.fairnessViolations (런타임)
// ===========================================================================
function S6_fairness() {
  const f = D.rules.fairness;
  if (!isObj(f)) return;
  if (f.playerWeaponsExempt !== true) {
    V('S6', `rules.fairness.playerWeaponsExempt = ${f.playerWeaponsExempt} ≠ true — §9.5 "플레이어 무기는 fairness 의 대상이 아니다"`);
  }
  const bulletById = new Map(rowsQuiet(D.bullets.bullets).map((b) => [b && b.id, b]));
  const partEmit = bossPartEmitterIds();     // §7.4 "보스 부위 패턴" → 1.50
  const midEmit = midBossEmitterIds();       // §7.4 "중간보스 패턴"  → 1.20
  let n = 0;

  for (const e of EMITTERS()) {
    if (!isObj(e)) continue;
    n += 1;
    const tag = `enemies.emitters[${e.id}]`;
    const bul = e.bulletId === null ? null : bulletById.get(e.bulletId);

    // (1) telegraphSec — 3축의 max (§7.4 "두 하한이 겹치면 큰 쪽")
    if (num(e.telegraphSec)) {
      const floors = [];
      if (num(f.minTelegraphSec)) floors.push([f.minTelegraphSec, 'fairness.minTelegraphSec(절대 하한)']);
      if (TELEGRAPH_FLOOR_BY_TYPE[e.type] !== undefined) floors.push([TELEGRAPH_FLOOR_BY_TYPE[e.type], `§7.4 거동표(${e.type})`]);
      if (bul && bul.status === 'slow') floors.push([TELEGRAPH_FLOOR_SLOW_BULLET, '§7.4 상태이상(slow) 탄']);
      if (bul && bul.status === 'stun' && num(f.minStunTelegraphSec)) floors.push([f.minStunTelegraphSec, 'fairness.minStunTelegraphSec(stun 탄)']);
      if (midEmit.has(e.id)) floors.push([TELEGRAPH_FLOOR_MIDBOSS, '§7.4 개체 클래스(중간보스 패턴)']);
      if (partEmit.has(e.id)) floors.push([TELEGRAPH_FLOOR_BOSSPART, '§7.4 개체 클래스(보스 부위 패턴) — v1.3']);
      if (floors.length) {
        // ★ max 합성: 가장 큰 하한 하나만 신고한다 (3축이 겹치면 큰 쪽, §7.4)
        let best = floors[0];
        for (const fl of floors) if (fl[0] > best[0]) best = fl;
        if (e.telegraphSec < best[0] - 1e-9) {
          V('S6', `${tag}.telegraphSec = ${e.telegraphSec} < ${best[0]} (${best[1]}) — §7.4 3축 max. `
            + `적용된 하한 = [${floors.map(([v2, w2]) => `${v2}(${w2})`).join(', ')}]`);
        }
      }
    }

    // (2) 탄 속도 — ★ bullets[].speed 는 삭제됐다. 이미터가 유일 소유자 (§9.7)
    if (num(e.speed)) {
      if (num(f.maxBulletSpeed) && e.speed > f.maxBulletSpeed) {
        V('S6', `${tag}.speed = ${e.speed} > fairness.maxBulletSpeed(${f.maxBulletSpeed}) (§12.4)`);
      }
      if (e.type === 'aimed' && num(f.maxAimedBulletSpeed) && e.speed > f.maxAimedBulletSpeed) {
        V('S6', `${tag}: 조준탄 speed = ${e.speed} > fairness.maxAimedBulletSpeed(${f.maxAimedBulletSpeed}) (§12.4)`);
      }
      // ★ v1.3: 상태이상 탄 ≤ maxBulletSpeed × statusBulletSpeedMul (= 156). 거처가 fairness 로 이사했다
      if (bul && bul.status !== null && !isAmb(bul.status)
          && num(f.maxBulletSpeed) && num(f.statusBulletSpeedMul)) {
        const cap = f.maxBulletSpeed * f.statusBulletSpeedMul;
        if (e.speed > cap + 1e-9) {
          V('S6', `${tag}.speed = ${e.speed} > maxBulletSpeed(${f.maxBulletSpeed}) × statusBulletSpeedMul(${f.statusBulletSpeedMul}) = ${cap} `
            + `— 상태이상 탄(bullets[${bul.id}].status="${bul.status}")은 크고 느려야 한다 (§12.4/§13.4-S6)`);
        }
      }
    }

    // (3) 최소 탄 반경 (§12.4)
    if (bul && num(bul.radius) && num(f.minBulletRadiusPx) && bul.radius < f.minBulletRadiusPx) {
      V('S6', `${tag} → bullets[${bul.id}].radius = ${bul.radius} < fairness.minBulletRadiusPx(${f.minBulletRadiusPx}) (§12.4)`);
    }

    // (4) wall 의 통과 틈 (§12.4)
    if (e.type === 'wall') {
      if (num(e.gapWidthPx) && num(f.minGapWidthPx) && e.gapWidthPx < f.minGapWidthPx) {
        V('S6', `${tag}.gapWidthPx = ${e.gapWidthPx} < fairness.minGapWidthPx(${f.minGapWidthPx}) (§12.4)`);
      }
      if (num(e.gapCount) && e.gapCount < 1) V('S6', `${tag}.gapCount = ${e.gapCount} — 틈 없는 벽은 회피 불가 (§12.4)`);
    }

    // (5) 스턴 최대 지속 (§12.4)
    if (bul && bul.status === 'stun' && num(bul.statusDurationSec) && num(f.maxStunSec)
        && bul.statusDurationSec > f.maxStunSec) {
      V('S6', `${tag} → bullets[${bul.id}].statusDurationSec = ${bul.statusDurationSec} > fairness.maxStunSec(${f.maxStunSec}) (§12.4)`);
    }
  }
  EX('S6', n);

  // moveId: charge 의 windUpSec (§8.4 "charge.windUpSec ≥ fairness.minTelegraphSec")
  const checkWind = (mp, tag) => {
    if (isObj(mp) && num(mp.windUpSec) && num(f.minTelegraphSec) && mp.windUpSec < f.minTelegraphSec) {
      V('S6', `${tag}.windUpSec = ${mp.windUpSec} < fairness.minTelegraphSec(${f.minTelegraphSec}) — §8.4`);
    }
  };
  for (const a of ARCHETYPES()) if (isObj(a) && a.moveId === 'charge') checkWind(a.moveParams, `enemies.archetypes[${a.id}].moveParams`);
  for (const b of BOSSES()) if (isObj(b) && b.moveId === 'charge') checkWind(b.moveParams, `bosses[${b.id}].moveParams`);

  // ★ v1.3: minSpawnRadiusPx 는 S6의 항목이 아니다 → 런타임 어서션
  S('S6', `fairness.minSpawnRadiusPx(${f.minSpawnRadiusPx}) = "적 탄은 플레이어 반경 N px 이내에서 생성 불가"는 `
    + `발사 시점의 플레이어 위치의 함수라 정적으로 검사할 대상이 존재하지 않는다 → ★ v1.3이 S6에서 뺐다(§13.4-S6). `
    + `TODO: step.js 의 탄 생성 경로에 런타임 assert + sim 이 certify.static.fairnessViolations 로 카운트 (상한 0)`);
}

// ===========================================================================
//  S7 — 동시 텔레그래프 (개체당 ≤ telegraphConcurrentMaxPerEntity)
//  보스 patternSet 을 3페이즈 전부 전개해 정적 검사 (§12.4 · §8.9-R8)
//  ★ v1.3: 악절(phrase) 모델이 정본에 인쇄됐다 (§8.5) → 전개 모델이 정확해졌다
//     이미터는 everySec 간격으로 repeat 발을 쏜 뒤 restSec 만큼 쉰다.
//     유효 주기 = repeat × everySec + restSec
// ===========================================================================
function gcd(a, b) { while (b) { const t = a % b; a = b; b = t; } return a; }
function lcm(a, b) { return (a / gcd(a, b)) * b; }

/**
 * ★ 악절 모델 (§8.5) — 이미터 1개의 발사 시각 집합과 유효 주기.
 *   발사 시각(악절 내부) t_k = offsetSec + k·everySec,  k = 0 .. repeat−1
 *   악절 주기 P = repeat × everySec + restSec
 *   텔레그래프 창 = [t_k − telegraphSec, t_k)
 */
function phraseOf(e) {
  const P = e.repeat * e.everySec + e.restSec;
  const ts = [];
  for (let k = 0; k < e.repeat; k += 1) ts.push(e.offsetSec + k * e.everySec);
  return { P, ts };
}

function maxConcurrentTelegraphs(emitters) {
  // 값이 하나라도 확정되지 않았으면 전개하지 않는다 (발명 금지)
  for (const e of emitters) {
    if (!e || !num(e.everySec) || e.everySec <= 0 || !num(e.telegraphSec)
        || !num(e.offsetSec) || !num(e.repeat) || e.repeat < 1 || !num(e.restSec) || e.restSec < 0) {
      return { max: 0, unresolved: true };
    }
  }
  if (!emitters.length) return { max: 0, unresolved: false };
  const ph = emitters.map(phraseOf);
  if (ph.some((p) => !(p.P > 0))) return { max: 0, unresolved: true };
  // 주기 = 악절 주기들의 LCM (센티초 정수화). 상한 60 게임초에서 절단
  let P = ph.map((p) => Math.max(1, Math.round(p.P * 100))).reduce((x, y) => lcm(x, y), 1);
  if (P > 6000) P = 6000;
  let mx = 0;
  for (let t = 0; t < P; t += 1) {           // 0.01 게임초 스텝
    const now = t / 100;
    let cnt = 0;
    for (let i = 0; i < emitters.length; i += 1) {
      const e = emitters[i];
      const { P: per, ts } = ph[i];
      for (const t0 of ts) {
        // now 기준 그 발사구의 다음 발사까지 남은 시간 ∈ (0, per]
        let untilFire = (t0 - now) % per;
        if (untilFire <= 0) untilFire += per;
        if (untilFire <= e.telegraphSec) cnt += 1;
      }
    }
    if (cnt > mx) mx = cnt;
  }
  return { max: mx, unresolved: false };
}

function S7_concurrentTelegraphs() {
  const cap = D.rules.fairness && D.rules.fairness.telegraphConcurrentMaxPerEntity;
  if (!num(cap)) return;
  const emitById = new Map(EMITTERS().map((e) => [e && e.id, e]));
  const resolveIds = (ids) => rowsQuiet(ids).map((id) => (isAmb(id) ? null : emitById.get(id)));
  let n = 0;

  for (const b of BOSSES()) {
    if (!isObj(b)) continue;
    if (b.tier === 'mid') {
      rowsQuiet(b.patternSet).forEach((ps, i) => {
        const es = resolveIds(ps && ps.emitterIds);
        if (!es.length || es.some((e) => !e)) return;   // 모호/미해결 → 다른 게이트가 잡는다
        const { max, unresolved } = maxConcurrentTelegraphs(es);
        if (unresolved) return;
        n += 1;
        if (max > cap) V('S7', `bosses[${b.id}].patternSet[${i}] (중간보스): 동시 텔레그래프 ${max} > ${cap} (§12.4/§8.9-R8)`);
      });
      continue;
    }
    for (const p of rowsQuiet(b.parts)) {
      if (!isObj(p)) continue;
      rowsQuiet(p.patternSet).forEach((ps, phIdx) => {
        const es = resolveIds(ps && ps.emitterIds);
        if (!es.length || es.some((e) => !e)) return;
        const { max, unresolved } = maxConcurrentTelegraphs(es);
        if (unresolved) return;
        n += 1;
        if (max > cap) {
          V('S7', `bosses[${b.id}].parts[${p.id}].patternSet[${phIdx}] (페이즈 ${phIdx + 1}): 동시 텔레그래프 ${max} > ${cap} (§12.4)`);
        }
      });
    }
  }
  // 잡몹 = 이미터 1개 → 자기 자신과의 중첩만 가능 (telegraphSec > everySec 이면 발화)
  for (const a of ARCHETYPES()) {
    if (!isObj(a) || !isObj(a.attack)) continue;
    const e = emitById.get(a.attack.emitterId);
    if (!e) continue;
    const { max, unresolved } = maxConcurrentTelegraphs([e]);
    if (unresolved) continue;
    n += 1;
    if (max > cap) {
      V('S7', `enemies.archetypes[${a.id}] → ${e.id}: 동시 텔레그래프 ${max} > ${cap} (telegraphSec ${e.telegraphSec} vs everySec ${e.everySec})`);
    }
  }
  EX('S7', n);
}

// ===========================================================================
//  S8 — 혼합 비율 (§8.2 · §8.2.1)
//  저작 리스트(= stages[].waves[] 중 unlockStageMin ≤ s 인 레코드, v1.3)의
//  ★ 원시 개체 수(= count 그대로. v1.4: 엘리트를 빼지 않는다) 기준 속성 비율이 mix 에 ±3%p
//  제외 = 중간보스 · 새떼 · 보스 (셋 다 waves[] 밖이라 동어반복이다)
//  + mix 가 counter/prey 규칙(70/10/10/10)을 따르는지
//  ★ 정의역 = 31셀 (pool 6테마 × 스테이지 1~5 + finale × 스테이지 6). 실측 전 셀 0.0000%p
// ===========================================================================
const MIX_TOL_PP = 3.0;   // §8.2.1 "허용 오차 ±3%p (저작 리스트 대비)"

function counterOf(el) {   // matrix[c][el] == 2.0 인 c
  const m = (D.elements && D.elements.matrix) || {};
  for (const c of Object.keys(m)) if (m[c] && m[c][el] === 2.0) return c;
  return null;
}
function preyOf(el) {      // matrix[el][p] == 2.0 인 p
  const m = (D.elements && D.elements.matrix) || {};
  const row = m[el] || {};
  for (const p of Object.keys(row)) if (row[p] === 2.0) return p;
  return null;
}

function S8_mix() {
  const FINAL_ID = FINAL();
  let cells = 0;
  for (const t of rows('S8', D.stages.stages, 'stages.stages',
    '§8.2 — 혼합 비율 게이트가 0행을 보면 70/10/10/10 을 아무도 검사하지 않는다')) {
    if (!isObj(t) || !isObj(t.mix)) continue;
    const tag = `stages.stages[${t.id}]`;

    // mix 는 4속성 실제 키로 전개된 가중치 맵 하나 (§8.2)
    closedKeys('S8', t.mix, ELEMENTS4, `${tag}.mix`);
    const sum = Object.values(t.mix).filter(num).reduce((x, y) => x + y, 0);
    if (Math.abs(sum - 1.0) > 1e-6) V('S8', `${tag}.mix: 합 ${sum} ≠ 1.0 (§8.2)`);

    // (1) 저작 리스트의 원시 개체 수 기준 비율 vs mix — ±3%p, 스테이지별
    const isFinale = t.id === FINAL_ID;
    const stageAxis = isFinale ? [6] : [1, 2, 3, 4, 5];
    for (const s of stageAxis) {
      const cnt = { normal: 0, fire: 0, water: 0, grass: 0 };
      let tot = 0;
      let skipped = 0;
      for (const w of rowsQuiet(t.waves)) {
        if (!isObj(w)) continue;
        if (!num(w.unlockStageMin) || isAmb(w.unlockStageMin)) { skipped += 1; continue; }
        if (w.unlockStageMin > s) continue;                    // ★ v1.3: 저작 리스트의 정의
        if (!num(w.count) || isAmb(w.element)) { skipped += 1; continue; }
        if (!has(cnt, w.element)) continue;
        cnt[w.element] += w.count; tot += w.count;             // ★ v1.4: 엘리트를 빼지 않는다
      }
      if (skipped) A(`${tag}.waves`, `${skipped}개 웨이브가 모호/미확정 값을 포함해 S8 @ s${s} 분모에서 빠졌다`);
      if (tot <= 0) {
        V('S8', `${tag} @ 스테이지 ${s}: 저작 리스트(unlockStageMin ≤ ${s})가 0개체 — 분모가 없다 = 이 셀은 검사되지 않는다`);
        continue;
      }
      cells += 1;
      for (const el of ELEMENTS4) {
        const actualPp = (cnt[el] / tot) * 100;
        const wantPp = (num(t.mix[el]) ? t.mix[el] : 0) * 100;
        if (Math.abs(actualPp - wantPp) > MIX_TOL_PP + 1e-9) {
          V('S8', `${tag} @ 스테이지 ${s}: 저작 리스트 ${el} = ${actualPp.toFixed(2)}%p vs mix ${wantPp.toFixed(2)}%p `
            + `— |Δ| ${Math.abs(actualPp - wantPp).toFixed(2)} > ${MIX_TOL_PP}%p (§8.2.1)`);
        }
      }
    }

    // (2) mix 가 counter/prey 규칙을 따르는지 — 테마 속성이 있는 테마만 (§8.2)
    if (isFinale || t.element === null) continue;   // §8.16: 최종은 테마 속성이 없다
    const T = t.element, c = counterOf(T), p = preyOf(T);
    if (!c || !p) { C('S8', `${tag}: counter/prey 를 elements.matrix 에서 유도할 수 없다 (element=${JSON.stringify(T)})`); continue; }
    const want = { [T]: 0.70, [c]: 0.10, [p]: 0.10, normal: 0.10 };
    for (const el of ELEMENTS4) {
      if (Math.abs((num(t.mix[el]) ? t.mix[el] : 0) - want[el]) > 1e-9) {
        V('S8', `${tag}.mix.${el} = ${t.mix[el]} ≠ ${want[el]} — 70/10/10/10 규칙 (T=${T}, counter=${c}, prey=${p}) (§8.2)`);
      }
    }
  }
  EX('S8', cells);
  if (cells && cells !== 31) {
    C('S8', `S8 의 정의역이 ${cells}셀 — §13.4-S8 의 실측 주석은 "전 31셀"(pool 6테마 × s1~s5 + finale × s6)이다. `
      + `셀 수가 다르면 그 주석이 가리키는 대상이 바뀐 것이다`);
  }
}

// ===========================================================================
//  S9 — 구조 (§13.4)
//  ★ v1.3: crisisElementRule == "finaleRotating" ⟺ stages[].id == "finale"
//          (불리언 finaleCrisisRotating 이 어휘값이 됐다)
// ===========================================================================
function S9_structure() {
  const st = D.stages, pl = D.rules.player;
  const FINAL_ID = FINAL();
  const archById = new Map(ARCHETYPES().map((a) => [a && a.id, a]));
  let n = 0;

  // (1) stages[].element ∈ {water, fire, grass, null} 이고 null 은 finale 만
  for (const t of rows('S9', st.stages, 'stages.stages',
    '§8.16 — 구조 게이트가 0행을 보면 element·finale 규칙을 아무도 검사하지 않는다')) {
    if (!isObj(t)) continue;
    n += 1;
    const ok = ['water', 'fire', 'grass'].includes(t.element) || t.element === null;
    if (!ok) V('S9', `stages.stages[${t.id}].element = ${JSON.stringify(t.element)} — 허용 = water|fire|grass|null (§8.16)`);
    if (t.element === null && t.id !== FINAL_ID) {
      V('S9', `stages.stages[${t.id}].element = null 인데 finale 가 아니다 (§8.16)`);
    }
    if (t.id === FINAL_ID && t.element !== null) {
      V('S9', `stages.stages[${FINAL_ID}].element = ${JSON.stringify(t.element)} ≠ null (§8.16 — 최종은 테마 속성이 없다)`);
    }
    // ★ v1.3: crisisElementRule == "finaleRotating" ⟺ id == finale
    const rot = t.crisisElementRule === 'finaleRotating';
    const isFin = t.id === FINAL_ID;
    if (rot !== isFin) {
      V('S9', `stages.stages[${t.id}]: (crisisElementRule=="finaleRotating")=${rot} ≠ (id=="${FINAL_ID}")=${isFin} `
        + `— 로테이션은 최종만 (§8.10/§8.16, S9 v1.3)`);
    }
  }

  // (2) 무기 levels 정확히 8행 (§9.5)
  for (const w of rowsQuiet(D.weapons.weapons)) {
    if (!isObj(w)) continue;
    if (!Array.isArray(w.levels)) { V('S9', `weapons[${w.id}].levels: 배열이 아니다 (§9.5)`); continue; }
    if (w.levels.length !== 8) V('S9', `weapons[${w.id}].levels: ${w.levels.length}행 ≠ 8행 (§9.5)`);
  }

  // (3) 4 ≤ elementCapTotal < 3 × elementCapPerElement (§4.2)
  if (isObj(pl) && num(pl.elementCapTotal) && num(pl.elementCapPerElement)) {
    if (!(pl.elementCapTotal >= 4 && pl.elementCapTotal < 3 * pl.elementCapPerElement)) {
      V('S9', `player.elementCapTotal(${pl.elementCapTotal}) ∉ [4, 3 × elementCapPerElement(${pl.elementCapPerElement}) = ${3 * pl.elementCapPerElement}) (§4.2)`);
    }
  }

  // (4) rearIn / spawnEdge:"bottom" 은 rearSpawnAllowed[stage] 일 때만 (§8.4 · §8.7)
  const rear = rowsQuiet(st.curve && st.curve.rearSpawnAllowed);
  const firstRearStage = rear.findIndex((v) => v === true) + 1;   // 1-indexed. 0 = 영영 불가
  for (const t of rowsQuiet(st.stages)) {
    if (!isObj(t)) continue;
    const isFinale = t.id === FINAL_ID;
    rowsQuiet(t.waves).forEach((w, i) => {
      if (!isObj(w)) return;
      const a = archById.get(w.archetypeId);
      if (!a) return;
      const needsRear = a.moveId === 'rearIn' || w.spawnEdge === 'bottom';
      if (!needsRear) return;
      const u = w.unlockStageMin;      // ★ v1.3: 웨이브 자신의 티어가 이 판정의 주체다
      const tag = `stages.stages[${t.id}].waves[${i}] (${w.archetypeId}, moveId=${a.moveId}, spawnEdge=${w.spawnEdge})`;
      if (isAmb(u) || !num(u)) {
        A(`${tag}.unlockStageMin`, 'rearSpawnAllowed 게이트를 평가할 unlockStageMin 이 없다');
        return;
      }
      // ★ 이 웨이브가 실제로 등장할 수 있는 가장 이른 스테이지 (S8·S22와 같은 스테이지 축)
      //   pool 테마: 셔플되어 s1~s5 어디에도 온다 → 최이른 = unlockStageMin
      //   finale:   ★ 항상 스테이지 6 이다 (§8.16). unlockStageMin 은 전부 1 이지만 그것은
      //             「단일 티어, 티어 필터링 없음」의 형식값이지 등장 스테이지가 아니다 (§23.1-D11)
      const earliest = isFinale ? 6 : Math.max(1, u);
      if (firstRearStage === 0) V('S9', `${tag}: rearSpawnAllowed 가 전 스테이지 false 인데 후방 진입 (§8.4)`);
      else if (earliest < firstRearStage) {
        V('S9', `${tag}: 최이른 등장 스테이지 ${earliest}(unlockStageMin ${u}) < 후방 스폰 최초 허용 스테이지 ${firstRearStage} `
          + `— rearSpawnAllowed = [${rear.join(', ')}] (§8.4/§8.7)`);
      }
    });
  }

  // (5) 새떼에 swarm* 외 아키타입 금지 (§8.10) — 역방향: 잡몹 로스터/웨이브에 swarm* 금지
  for (const t of rowsQuiet(st.stages)) {
    if (!isObj(t)) continue;
    for (const r of rowsQuiet(t.roster)) {
      if (isObj(r) && /^swarm/.test(r.archetypeId)) {
        V('S9', `stages.stages[${t.id}].roster: 새떼 전용 "${r.archetypeId}" 가 잡몹 로스터에 있다 (§8.10)`);
      }
    }
    rowsQuiet(t.waves).forEach((w, i) => {
      if (isObj(w) && /^swarm/.test(w.archetypeId)) {
        V('S9', `stages.stages[${t.id}].waves[${i}]: 새떼 전용 "${w.archetypeId}" 가 잡몹 웨이브에 있다 (§8.10)`);
      }
    });
  }
  const swarms = ARCHETYPES().filter((a) => isObj(a) && /^swarm/.test(a.id));
  if (swarms.length !== 2) V('S9', `enemies.archetypes: 새떼 전용 아키타입 ${swarms.length}종 ≠ 2 (swarmChaff, swarmLancer — §8.6)`);

  // (6) §8.16: 최종의 위기 = 6서브웨이브 물×2 → 불×2 → 풀×2
  const ph = st.phase || {};
  if (num(ph.crisisSubWaves) && ph.crisisSubWaves !== 6) {
    V('S9', `stages.phase.crisisSubWaves = ${ph.crisisSubWaves} ≠ 6 — §8.16 의 3속성 × 2 로테이션이 표현 불가`);
  }
  EX('S9', n);
}

// ===========================================================================
//  S10 — 성장 예산 (§11.1 · §13.1 static.growthBudget)
//  ★ v1.3 문면 수정: 선언 상수 비교 + 유도 검사
//     maxLevelUps(60) < minTotalSink(67) ∧ minTotalSink 가 실제 데이터 유도값과 일치
//     (= 3 신규 무기 + Σ(무기 maxLevel−1) 28 + elementCapTotal 6 + Σ(패시브 maxLevel) 30 = 67)
// ===========================================================================
function S10_growthBudget() {
  const g = D.meta.certify && D.meta.certify.static && D.meta.certify.static.growthBudget;
  if (!isObj(g)) return;
  closedKeys('S10', g, ['maxLevelUps', 'minTotalSink'], 'meta.certify.static.growthBudget');
  if (num(g.maxLevelUps) && num(g.minTotalSink) && !(g.maxLevelUps < g.minTotalSink)) {
    V('S10', `growthBudget: maxLevelUps(${g.maxLevelUps}) < minTotalSink(${g.minTotalSink}) 이 거짓 — §11.1 "전부 못 찍는다가 산술적으로 성립"`);
  }
  // totalSink 를 데이터에서 재유도한다 (§11.1 의 대차대조표 · §13.4-S10)
  const pl = D.rules.player;
  if (isObj(pl) && num(D.passives.maxLevel) && num(pl.weaponSlots)
      && num(pl.elementCapTotal) && num(pl.passiveSlots)) {
    const newWeapon = pl.weaponSlots - 1;                    // 시작 무기 1 지급 → 4칸 중 3칸
    const weaponLevel = pl.weaponSlots * 7;                  // Σ(무기 maxLevel−1) = 4무기 × (8−1)
    const elementLevel = pl.elementCapTotal;                 // elementCapTotal
    const passive = pl.passiveSlots * D.passives.maxLevel;   // Σ(패시브 maxLevel) = 6칸 × Lv5
    const derived = newWeapon + weaponLevel + elementLevel + passive;
    if (num(g.minTotalSink) && derived !== g.minTotalSink) {
      V('S10', `growthBudget.minTotalSink = ${g.minTotalSink} ≠ 데이터 유도값 ${derived} `
        + `(신규 무기 ${newWeapon} + 무기 레벨 ${weaponLevel} + 속성 ${elementLevel} + 패시브 ${passive}) `
        + `— §13.4-S10 "선언과 데이터가 갈라지면 실패"`);
    }
  }
  S('S10', `growthBudget.maxLevelUps(${g.maxLevelUps}) 의 실측: 스테이지별 총 XP는 웨이브 편성·처치율·farm 정책의 함수라 `
    + `정적으로 계산 불가하다(§13.4-S10 v1.3이 문면을 그렇게 고쳤다) → 정본이 상수로 소유한다. `
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
  const rowsAB = [
    // §12.1 정정: 위기 중 = 새떼 70 + 웨이브 잔존 10 = 80 이 A층 enemies 합
    ['enemies', (f.swarmConcurrentMax || 0) + (f.crisisWaveResidualMax || 0),
      `swarmConcurrentMax(${f.swarmConcurrentMax}) + crisisWaveResidualMax(${f.crisisWaveResidualMax})`, caps.enemies],
    ['enemyBullets', f.maxSimultaneousEnemyBullets, `maxSimultaneousEnemyBullets(${f.maxSimultaneousEnemyBullets})`, caps.enemyBullets],
    ['telegraphs', f.telegraphConcurrentMaxGlobal, `telegraphConcurrentMaxGlobal(${f.telegraphConcurrentMaxGlobal})`, caps.telegraphs],
  ];
  for (const [name, a, why, b] of rowsAB) {
    if (!num(a) || !num(b)) continue;
    if (!(a < b)) V('S12', `2층 캡 위반 — ${name}: A층 ${a} (= ${why}) < B층 caps.${name} ${b} 이 거짓 (§12.1)`);
  }
  // A층 enemyConcurrentMax 자체도 B층 아래여야 한다
  if (num(f.enemyConcurrentMax) && num(caps.enemies) && !(f.enemyConcurrentMax < caps.enemies)) {
    V('S12', `2층 캡 위반 — fairness.enemyConcurrentMax(${f.enemyConcurrentMax}) < caps.enemies(${caps.enemies}) 이 거짓 (§12.1)`);
  }
}

// ===========================================================================
//  S13 — 스턴의 거처 (§13.4 · §9.8.1-확정③)
//  status=="stun" 탄을 쓰는 이미터는 보스 부위의 patternSet[2](페이즈 3)에만.
//  = id 가 {bossId}{PartIdPascal}P3 형태여야 하고 스테이지당 최대 2개 부위
// ===========================================================================
function S13_stunHome() {
  const stunBullets = new Set(rowsQuiet(D.bullets.bullets).filter((b) => isObj(b) && b.status === 'stun').map((b) => b.id));
  if (!stunBullets.size) {
    V('S13', 'bullets: status=="stun" 인 탄이 0종 — §9.7·§12.4·§7.4·§7.10 이 스턴을 위해 설계한 것이 전부 도달 불가가 된다');
    return;
  }
  const stunEmitters = new Set(EMITTERS().filter((e) => isObj(e) && stunBullets.has(e.bulletId)).map((e) => e.id));
  if (!stunEmitters.size) {
    // ★ 이것이 v1.2의 상태였다 — 스턴 메커닉 전체가 도달 불가능한 콘텐츠였다
    V('S13', `status=="stun" 탄(${[...stunBullets].join(', ')})을 참조하는 이미터가 0개 — ★ 스턴 메커닉이 게임에 존재하지 않는다. `
      + `§9.8.1-확정③ 이 그 자리를 {bossId}{PartIdPascal}P3 로 지정했다 (§23.1-D4)`);
    return;
  }
  const maxPerStage = D.stages.phase && D.stages.phase.statusStunMaxPerStage;
  let n = 0;

  // 합법한 자리 = 보스 부위의 patternSet[2](페이즈 3) 뿐 → 그 밖의 사용처를 전수 신고한다
  for (const a of ARCHETYPES()) {
    if (isObj(a) && isObj(a.attack) && stunEmitters.has(a.attack.emitterId)) {
      V('S13', `enemies.archetypes[${a.id}]: 스턴 이미터 "${a.attack.emitterId}" — 스턴의 거처는 보스 부위 patternSet[2] 뿐 (§13.4-S13)`);
    }
  }
  for (const b of BOSSES()) {
    if (!isObj(b)) continue;
    if (b.tier === 'mid') {
      for (const ps of rowsQuiet(b.patternSet)) {
        for (const id of rowsQuiet(ps && ps.emitterIds)) {
          if (stunEmitters.has(id)) V('S13', `bosses[${b.id}] (중간보스): 스턴 이미터 "${id}" — 보스 부위 patternSet[2] 전용 (§13.4-S13)`);
        }
      }
      continue;
    }
    let stunPartsInBoss = 0;
    for (const p of rowsQuiet(b.parts)) {
      if (!isObj(p)) continue;
      rowsQuiet(p.patternSet).forEach((ps, phIdx) => {
        for (const id of rowsQuiet(ps && ps.emitterIds)) {
          if (!stunEmitters.has(id)) continue;
          n += 1;
          if (phIdx !== 2) {
            V('S13', `bosses[${b.id}].parts[${p.id}].patternSet[${phIdx}]: 스턴 이미터 "${id}" — 페이즈 3(index 2) 에만 (§13.4-S13)`);
          } else {
            stunPartsInBoss += 1;
            // §9.8.1-확정③: id 는 반드시 {bossId}{PartIdPascal}P3 형태
            const want = `${b.id}${pascal(p.id)}P3`;
            if (id !== want) {
              V('S13', `bosses[${b.id}].parts[${p.id}]: 스턴 이미터 id "${id}" ≠ "${want}" — §9.8.1-확정③ (S36과 같은 규칙)`);
            }
          }
        }
      });
    }
    if (num(maxPerStage) && stunPartsInBoss > maxPerStage) {
      V('S13', `bosses[${b.id}]: 스턴 이미터를 가진 부위 ${stunPartsInBoss} > statusStunMaxPerStage(${maxPerStage}) (§13.4-S13)`);
    }
  }
  EX('S13', n);
  // §9.7: 스턴은 difficulty.stunMinDifficulty 이상에서만
  const smd = D.meta.difficulty && D.meta.difficulty.stunMinDifficulty;
  if (smd && isObj(D.meta.difficulty) && !has(D.meta.difficulty, smd)) {
    V('S13', `meta.difficulty.stunMinDifficulty = ${JSON.stringify(smd)} 가 난이도 목록에 없다 (§9.7)`);
  }
}

// ===========================================================================
//  S14 — shape ↔ status 동치 (§9.7)
//  (bullets[].status === null) === (bullets[].shape === "circle")
// ===========================================================================
function S14_shapeStatusEquiv() {
  let n = 0;
  for (const b of rowsQuiet(D.bullets.bullets)) {
    if (!isObj(b) || isAmb(b.status) || isAmb(b.shape)) continue;
    n += 1;
    const l = b.status === null, r = b.shape === 'circle';
    if (l !== r) {
      V('S14', `bullets[${b.id}]: (status === null)=${l} ≠ (shape === "circle")=${r} `
        + `— status=${JSON.stringify(b.status)}, shape=${JSON.stringify(b.shape)}. §7.4 "육각 = 상태이상"이 거짓말이 된다`);
    }
  }
  EX('S14', n);
}

// ===========================================================================
//  S15 — 중간보스 속성 주입 (§8.9) : tier == "mid" ⟺ element == null
// ===========================================================================
function S15_midBossElement() {
  for (const b of BOSSES()) {
    if (!isObj(b)) continue;
    const isMid = b.tier === 'mid';
    const nullEl = has(b, 'element') && b.element === null;
    if (isMid && !nullEl) {
      V('S15', `bosses[${b.id}]: tier=="mid" 인데 element 가 null 이 아니다 (${JSON.stringify(b.element)}) — 주입 대상 표식 (§8.9)`);
    }
    if (!isMid && has(b, 'element')) {
      V('S15', `bosses[${b.id}]: tier=${JSON.stringify(b.tier)} 인데 최상위 element 키가 있다 — null 허용은 tier:"mid" 뿐 (§8.9/S15)`);
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
  let n = 0;
  for (const b of BOSSES()) {
    if (!isObj(b)) continue;
    if (b.tier === 'mid') {
      const ps = rowsQuiet(b.patternSet);
      n += 1;
      if (ps.length !== 1) V('S16', `bosses[${b.id}] (중간보스): patternSet 길이 ${ps.length} ≠ 1 (§9.8)`);
      ps.forEach((e, i) => {
        const cnt = Array.isArray(e && e.emitterIds) ? e.emitterIds.length : -1;
        if (cnt < 1 || cnt > 2) V('S16', `bosses[${b.id}].patternSet[${i}].emitterIds: 길이 ${cnt} ∉ [1, 2] (§8.9-R8)`);
      });
      // 중간보스는 parts: [] (§8.9 "단일 몸체. 부위 없음")
      if (!Array.isArray(b.parts) || b.parts.length !== 0) {
        V('S16', `bosses[${b.id}] (중간보스): parts 가 [] 가 아니다 — "단일 몸체. 부위 없음" (§8.9/§9.8.2-ⓓ)`);
      }
      continue;
    }
    // 스테이지·최종 보스는 루트 patternSet 을 갖지 않는다 (부위가 갖는다)
    if (has(b, 'patternSet')) {
      V('S16', `bosses[${b.id}]: tier=${JSON.stringify(b.tier)} 인데 루트 patternSet 이 있다 — 패턴의 거처는 parts[].patternSet 이다 (§9.8)`);
    }
    for (const p of rowsQuiet(b.parts)) {
      if (!isObj(p)) continue;
      const ps = rowsQuiet(p.patternSet);
      n += 1;
      if (ps.length !== phases) {
        V('S16', `bosses[${b.id}].parts[${p.id}]: patternSet 길이 ${ps.length} ≠ 페이즈 수 ${phases} (§9.8)`);
      }
      ps.forEach((e, i) => {
        const cnt = Array.isArray(e && e.emitterIds) ? e.emitterIds.length : -1;
        if (cnt !== 1) V('S16', `bosses[${b.id}].parts[${p.id}].patternSet[${i}].emitterIds: 길이 ${cnt} ≠ 1 — 보스 부위는 1 강제 (§8.9-R8/S16)`);
      });
    }
  }
  EX('S16', n);
}

// ===========================================================================
//  S17 — 소환 (§8.9-R9)
//  summon != null ⟺ (tier == "mid" 그리고 id ∈ boss.midBossSummonsAllowed)
// ===========================================================================
function S17_summon() {
  const allowed = new Set(rowsQuiet(D.rules.boss && D.rules.boss.midBossSummonsAllowed));
  for (const b of BOSSES()) {
    if (!isObj(b)) continue;
    const lhs = has(b, 'summon') && b.summon !== null && !isAmb(b.summon);
    const rhs = b.tier === 'mid' && allowed.has(b.id);
    if (lhs !== rhs) {
      V('S17', `bosses[${b.id}]: (summon != null)=${lhs} ≠ (tier=="mid" ∧ id ∈ midBossSummonsAllowed)=${rhs} `
        + `— tier=${b.tier}, midBossSummonsAllowed=[${[...allowed].join(', ')}] (§8.9-R9/S17)`);
    }
  }
}

// ===========================================================================
//  S18 — mobility 의 진실성 (§8.12.1)
//  movePattern == "holdCenter" 인 보스는 mobility 부위 금지
// ===========================================================================
function S18_mobilityTruth() {
  for (const b of BOSSES()) {
    if (!isObj(b) || b.tier === 'mid') continue;
    if (b.movePattern !== 'holdCenter') continue;
    const mob = rowsQuiet(b.parts).filter((p) => isObj(p) && p.partType === 'mobility');
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
  let n = 0;
  for (const e of EMITTERS()) {
    if (!isObj(e) || isAmb(e.type)) continue;
    n += 1;
    const lhs = e.type === 'zone';
    const rhs = has(e, 'bulletId') && e.bulletId === null;
    if (lhs !== rhs) {
      V('S19', `enemies.emitters[${e.id}]: (type=="zone")=${lhs} ≠ (bulletId==null)=${rhs} `
        + `— zone 은 dmg 를 직접 갖는다 (§9.7/S19)`);
    }
    if (lhs && !num(e.dmg)) V('S19', `enemies.emitters[${e.id}]: zone 인데 dmg 가 없다 (§3.2 피해원 목록)`);
  }
  EX('S19', n);
}

// ===========================================================================
//  S20 — 편대 전용성 (§9.9.2) — ★ 양방향 (⟺)
//  pincer ⟺ moveId == "strafe" / columnV ⟺ moveId == "column"
// ===========================================================================
function S20_formationExclusivity() {
  const archById = new Map(ARCHETYPES().map((a) => [a && a.id, a]));
  const pairs = [['pincer', 'strafe'], ['columnV', 'column']];
  let n = 0;
  const check = (formationId, archetypeId, tag) => {
    const a = archById.get(archetypeId);
    if (!a || isAmb(a.moveId) || isAmb(formationId)) return;
    n += 1;
    for (const [form, move] of pairs) {
      const lhs = formationId === form, rhs = a.moveId === move;
      if (lhs !== rhs) {
        V('S20', `${tag}: (formationId=="${form}")=${lhs} ≠ (moveId=="${move}")=${rhs} `
          + `— 실제 formationId="${formationId}", archetype="${archetypeId}", moveId="${a.moveId}" (§9.9.2/S20 — ⟺ 는 양방향이며 의도다)`);
      }
    }
  };
  for (const t of rows('S20', D.stages.stages, 'stages.stages',
    '§9.9.2 — 편대 전용성이 0행을 보면 strafe/pincer 짝을 아무도 검사하지 않는다')) {
    if (!isObj(t)) continue;
    rowsQuiet(t.waves).forEach((w, i) => {
      if (isObj(w)) check(w.formationId, w.archetypeId, `stages.stages[${t.id}].waves[${i}]`);
    });
  }
  for (const cw of rowsQuiet(D.stages.phase && D.stages.phase.crisisWaves)) {
    if (isObj(cw)) check(cw.formationId, cw.archetypeId, `stages.phase.crisisWaves[subWave ${cw.subWave}, ${cw.archetypeId}]`);
  }
  for (const b of BOSSES()) {
    if (isObj(b) && isObj(b.summon)) check(b.summon.formationId, b.summon.archetypeId, `bosses[${b.id}].summon`);
  }
  EX('S20', n);
}

// ===========================================================================
//  S21 — 드래프트 보장 상한 (§11.1)
//  guarantee* 키의 동시 발동 최대 개수 ≤ optionCount − 1 (= 2)
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
//  ★ v1.3 명문화: 정의역 = 모든 (theme, stage) 쌍. 최악 0.148
// ===========================================================================
const SWARM_XP_CAP = 0.30;

function S22_swarmXpShare() {
  const ph = D.stages.phase || {}, curve = D.stages.curve || {};
  const xpRef = D.enemies.bands && D.enemies.bands.chaff && D.enemies.bands.chaff.xpRef;
  if (!num(xpRef)) { A('enemies.bands.chaff.xpRef', 'S22 의 swarmXp 파생식(= xpRef × 0.5)이 값을 갖지 못한다'); return; }
  const swarmXp = xpRef * 0.5;                       // §8.10 파생식
  const archById = new Map(ARCHETYPES().map((a) => [a && a.id, a]));
  const scale = rowsQuiet(curve.swarmTotalScale);
  const FINAL_ID = FINAL();
  let n = 0;

  // ★ 테마 순서는 셔플된다(§8.1) → 어떤 테마가 어느 스테이지에 와도 성립해야 한다
  for (const t of rows('S22', D.stages.stages, 'stages.stages',
    '§8.10 — 새떼 XP 상한이 0행을 보면 분모가 없다')) {
    if (!isObj(t)) continue;
    const stageAxis = t.id === FINAL_ID ? [6] : [1, 2, 3, 4, 5];
    for (const st of stageAxis) {
      let waveXp = 0, unresolved = false;
      for (const w of rowsQuiet(t.waves)) {
        if (!isObj(w)) continue;
        if (!num(w.unlockStageMin) || isAmb(w.unlockStageMin)) { unresolved = true; continue; }
        if (w.unlockStageMin > st) continue;             // ★ v1.3: 저작 리스트의 정의 (S8과 같은 필터)
        const a = archById.get(w.archetypeId);
        if (!a || !num(a.xp) || !num(w.count)) { unresolved = true; continue; }
        waveXp += w.count * a.xp;
      }
      if (unresolved) {
        A(`stages.stages[${t.id}].waves[].unlockStageMin`,
          `S22 의 분모(스테이지 ${st} 저작 리스트 Σ XP)를 확정할 수 없다`);
        break;
      }
      if (waveXp <= 0) { V('S22', `stages.stages[${t.id}] @ s${st}: 저작 리스트 Σ XP = 0 → 분모 없음`); continue; }
      const s = num(scale[st - 1]) ? scale[st - 1] : null;
      if (s === null) { A(`stages.curve.swarmTotalScale[${st - 1}]`, 'S22 를 평가할 수 없다'); continue; }
      n += 1;
      const crisisXp = (num(ph.crisisTotal) ? ph.crisisTotal : 0) * s * swarmXp;
      const ratio = crisisXp / waveXp;
      if (ratio > SWARM_XP_CAP + 1e-9) {
        V('S22', `stages.stages[${t.id}] @ 스테이지 ${st}: 새떼 XP 지분 ${ratio.toFixed(3)} > ${SWARM_XP_CAP} `
          + `(위기 ${crisisXp} / 웨이브 ${waveXp}) — §13.4-S22 (상한이지 목표가 아니다, §8.10)`);
      }
    }
  }
  EX('S22', n);
}

// ===========================================================================
//  S23 — 코인원 균질성 (§13.4)
//  ★ v1.3: 정의역 = themeDraw.pool 에 속한 테마 (finale 은 15종이라 정의역 밖 — §13.2-⑪)
//  roster 4종 중 turret + bruiser 밴드가 1~2종
// ===========================================================================
function S23_coinSourceHomogeneity() {
  const archById = new Map(ARCHETYPES().map((a) => [a && a.id, a]));
  const pool = new Set(rowsQuiet(D.stages.themeDraw && D.stages.themeDraw.pool));
  let n = 0;

  for (const t of rows('S23', D.stages.stages, 'stages.stages',
    '§13.4-S23 — 코인원 균질성이 0행을 보면 셔플되는 테마 사이의 코인 수급 편차를 아무도 검사하지 않는다')) {
    if (!isObj(t)) continue;
    if (!pool.has(t.id)) continue;    // ★ v1.3: finale 은 정의역 밖 (셔플 대상이 아니다, §8.1)
    n += 1;
    const roster = rowsQuiet(t.roster);
    const bands = roster.map((r) => (archById.get(r && r.archetypeId) || {}).band).filter(Boolean);
    const cnt = bands.filter((b) => b === 'turret' || b === 'bruiser').length;
    // §8.6 "테마당 정확히 4종 (공용 3 + 시그니처 1)"
    if (roster.length !== 4) V('S23', `stages.stages[${t.id}].roster: ${roster.length}종 ≠ 4 (§8.6)`);
    if (cnt < 1 || cnt > 2) {
      V('S23', `stages.stages[${t.id}]: turret+bruiser 밴드 ${cnt}종 ∉ [1, 2] — 코인원 균질성 (§13.4-S23). `
        + `roster 밴드 = [${bands.join(', ')}]`);
    }
  }
  EX('S23', n);
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
  let n = 0;
  for (const b of BOSSES()) {
    if (!isObj(b) || b.tier === 'mid') continue;
    const parts = rowsQuiet(b.parts);
    const armor = parts.filter((p) => isObj(p) && p.partType === 'armor');
    const optional = parts.filter((p) => isObj(p) && (p.partType === 'mobility' || p.partType === 'armament'));
    if (!armor.length || !isObj(b.core) || !num(b.core.hp) || !num(b.armorCoreRatio)) continue;
    n += 1;

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
  EX('S24', n);
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
//  ★ v1.3: crisisWaves 의 subWave 1파의 Σ count ≤ swarmConcurrentMax(70)
//     (처음으로 읽을 데이터가 생겼다 — §9.9)
//  ★ 동시 개체 수 자체는 처치율의 함수라 정적 검사 불가 → "어떤 처치율에서도 깨지는 편성"만 잡는다
// ===========================================================================
function S26_concurrentBudget() {
  const f = D.rules.fairness || {}, ph = D.stages.phase || {}, curve = D.stages.curve || {};
  let n = 0;

  // (1) 웨이브 1개 = waves[] 의 레코드 1개 (§8.7)
  for (const t of rows('S26', D.stages.stages, 'stages.stages',
    '§12.1 — 동시 개체 예산이 0행을 보면 어떤 편성도 검사되지 않는다')) {
    if (!isObj(t)) continue;
    rowsQuiet(t.waves).forEach((w, i) => {
      if (!isObj(w) || !num(w.count) || !num(f.enemyConcurrentMax)) return;
      n += 1;
      if (w.count > f.enemyConcurrentMax) {
        V('S26', `stages.stages[${t.id}].waves[${i}]: count ${w.count} > enemyConcurrentMax(${f.enemyConcurrentMax}) `
          + `— 이 웨이브는 어떤 처치율에서도 A층 예산을 깬다 = 편성 버그 (§12.1/S26)`);
      }
    });
  }

  // (2) ★ v1.3: 위기 서브웨이브 1파 = crisisWaves 를 subWave 로 묶은 Σ count
  const bySub = new Map();
  for (const cw of rowsQuiet(ph.crisisWaves)) {
    if (!isObj(cw) || !num(cw.count)) continue;
    bySub.set(cw.subWave, (bySub.get(cw.subWave) || 0) + cw.count);
  }
  for (const [sub, total] of bySub) {
    n += 1;
    if (num(f.swarmConcurrentMax) && total > f.swarmConcurrentMax) {
      V('S26', `stages.phase.crisisWaves subWave ${sub}: Σ count ${total} > swarmConcurrentMax(${f.swarmConcurrentMax}) (§12.1/S26 v1.3)`);
    }
  }
  // 새떼 전원이 동시에 살아있는 최악(= 아무도 안 죽는다)도 예산 안이어야 한다
  if (num(ph.crisisTotal) && num(f.swarmConcurrentMax)) {
    rowsQuiet(curve.swarmTotalScale).forEach((s, i) => {
      if (!num(s)) return;
      const total = ph.crisisTotal * s;
      if (total > f.swarmConcurrentMax) {
        V('S26', `위기 총량 @ 스테이지 ${i + 1}: crisisTotal(${ph.crisisTotal}) × swarmTotalScale(${s}) = ${total} `
          + `> swarmConcurrentMax(${f.swarmConcurrentMax}) (§12.1/S26)`);
      }
    });
  }
  EX('S26', n);
}

// ###########################################################################
//  ★★ S27 ~ S40 — v1.3 신설 14개 (§13.4 · §23.4)
// ###########################################################################

// ===========================================================================
//  S27 ★ — 엘리트의 적법성 (§8.6 · §13.4-S27, v1.3)
//  eliteIndex != null ⟹ (archetypeId 의 band ∈ elite.bandAllowed)
//                     ∧ (element ∈ elite.elementAllowed)
//  §8.6이 두 규칙을 확정하고도 검사기를 주지 않아 04가 4건을 어겼다
// ===========================================================================
function S27_eliteLegality() {
  const el = D.rules.elite;
  if (!isObj(el)) return;
  const bandAllowed = rowsQuiet(el.bandAllowed);
  const elemAllowed = rowsQuiet(el.elementAllowed);
  const archById = new Map(ARCHETYPES().map((a) => [a && a.id, a]));
  let n = 0;

  for (const t of rows('S27', D.stages.stages, 'stages.stages',
    '§8.6 — 엘리트 적법성이 0행을 보면 bandAllowed·elementAllowed 를 아무도 검사하지 않는다')) {
    if (!isObj(t)) continue;
    rowsQuiet(t.waves).forEach((w, i) => {
      if (!isObj(w)) return;
      if (w.eliteIndex === null || w.eliteIndex === undefined || isAmb(w.eliteIndex)) return;
      n += 1;
      const tag = `stages.stages[${t.id}].waves[${i}] (${w.archetypeId}, element=${JSON.stringify(w.element)}, eliteIndex=${w.eliteIndex})`;
      const a = archById.get(w.archetypeId);
      if (a && !isAmb(a.band) && !bandAllowed.includes(a.band)) {
        V('S27', `${tag}: band "${a.band}" ∉ elite.bandAllowed [${bandAllowed.join(', ')}] (§8.6/S27)`);
      }
      if (!isAmb(w.element) && !elemAllowed.includes(w.element)) {
        V('S27', `${tag}: element ${JSON.stringify(w.element)} ∉ elite.elementAllowed [${elemAllowed.join(', ')}] (§8.6/S27). `
          + `★ 제거하려면 eliteIndex: null. 옮기려면 그 웨이브에 element != "normal" 인 개체가 있어야 한다 (§23.2-D7)`);
      }
      // eliteIndex 는 그 웨이브의 개체 인덱스여야 한다
      if (num(w.eliteIndex) && num(w.count) && (w.eliteIndex < 0 || w.eliteIndex >= w.count)) {
        V('S27', `${tag}: eliteIndex ${w.eliteIndex} ∉ [0, count(${w.count})) — 존재하지 않는 개체를 엘리트로 지정했다`);
      }
    });
  }
  EX('S27', n);
  // §8.6: 웨이브당 엘리트 최대 1 (perWaveMax)
  if (num(el.perWaveMax) && el.perWaveMax !== 1) {
    C('S27', `rules.elite.perWaveMax = ${el.perWaveMax} ≠ 1 — waves[].eliteIndex 는 스칼라라 웨이브당 2기 이상을 표현할 수 없다 (§8.6/§9.9)`);
  }
}

// ===========================================================================
//  S28 ★ — from 의 적법성 (§8.5 · §13.4-S28, v1.3)
//  from == "part" ⟺ 그 이미터가 bosses[].parts[].patternSet[i].emitterIds 에서만 참조된다
// ===========================================================================
function S28_fromLegality() {
  const partRef = bossPartEmitterIds();
  const midRef = midBossEmitterIds();
  const mobRef = mobEmitterIds();
  let n = 0;
  for (const e of EMITTERS()) {
    if (!isObj(e) || isAmb(e.from)) continue;
    n += 1;
    const lhs = e.from === 'part';
    const inPart = partRef.has(e.id);
    const inOther = midRef.has(e.id) || mobRef.has(e.id);
    const rhs = inPart && !inOther;     // "…에서만 참조된다"
    if (lhs !== rhs) {
      V('S28', `enemies.emitters[${e.id}]: (from=="part")=${lhs} ≠ (보스 부위 patternSet 에서만 참조)=${rhs} `
        + `— 부위참조=${inPart}, 중간보스/잡몹참조=${inOther} (§8.5/S28). `
        + `from="part" 는 참조한 부위의 anchor 에서 발사한다 = 보스 부위 이미터 전용 (§9.8.1)`);
    }
    // 어느 쪽에서도 참조되지 않는 이미터 = 죽은 정의
    if (!inPart && !inOther) {
      V('S28', `enemies.emitters[${e.id}]: 아무도 참조하지 않는다 — 죽은 이미터 정의는 AI가 의미를 발명하는 자리다 (§9.5의 죽은 필드 논거)`);
    }
  }
  EX('S28', n);
}

// ===========================================================================
//  S29 ★ — 중간보스 스케줄의 파생 무결성 (§8.9 · §13.4-S29, v1.3)
//  len(phase.midBossAtSec[i]) == curve.midBossCount[i]  (전 6스테이지)
// ===========================================================================
function S29_midBossSchedule() {
  const at = D.stages.phase && D.stages.phase.midBossAtSec;
  const cntArr = D.stages.curve && D.stages.curve.midBossCount;
  if (!Array.isArray(at) || !Array.isArray(cntArr)) {
    V('S29', `stages.phase.midBossAtSec / stages.curve.midBossCount: 배열이 아니다 — 파생 무결성을 평가할 수 없다 (§8.9/S29)`);
    return;
  }
  if (at.length !== 6) V('S29', `stages.phase.midBossAtSec: 길이 ${at.length} ≠ 6 (스테이지 축)`);
  if (cntArr.length !== 6) V('S29', `stages.curve.midBossCount: 길이 ${cntArr.length} ≠ 6 (스테이지 축)`);
  const lim = Math.min(at.length, cntArr.length);
  for (let i = 0; i < lim; i += 1) {
    if (!Array.isArray(at[i])) {
      V('S29', `stages.phase.midBossAtSec[${i}]: 배열이 아니다 (스테이지 ${i + 1}의 등장 시각 목록)`);
      continue;
    }
    if (at[i].length !== cntArr[i]) {
      V('S29', `스테이지 ${i + 1}: len(midBossAtSec[${i}]) = ${at[i].length} ≠ midBossCount[${i}] = ${cntArr[i]} `
        + `— 파생 무결성 위반. 등장 시각의 개수가 곧 등장 횟수다 (§8.9/S29)`);
    }
    // 등장 시각은 잡몹 페이즈 안이어야 하고 위기 세션과 겹치면 강제 이탈이다
    const mob = D.stages.phase.mobPhaseSec, cs = D.stages.phase.crisisStartSec;
    for (const t of at[i]) {
      if (!num(t)) continue;
      if (num(mob) && t >= mob) {
        V('S29', `stages.phase.midBossAtSec[${i}] = ${t} ≥ mobPhaseSec(${mob}) — 잡몹 페이즈 밖에서 등장할 수 없다 (§6.3)`);
      }
      if (num(cs) && t >= cs) {
        C('S29', `stages.phase.midBossAtSec[${i}] = ${t} ≥ crisisStartSec(${cs}) — 등장하자마자 midBossForcedLeaveOnCrisis 로 쫓겨난다 (§8.9)`);
      }
    }
  }
  EX('S29', lim);
}

// ===========================================================================
//  S30 ★ — 악절의 배타성 (§8.5 · §13.4-S30, v1.3)
//  이미터가 bosses[] 에서 참조된다 ⟺ (repeat ≥ 2 ∧ restSec > 0)
//  enemies.archetypes[].attack 에서 참조되는 이미터는 repeat == 1 ∧ restSec == 0
//  ★ "보스만 쓴다"를 문장이 아니라 기계로 만든다
// ===========================================================================
function S30_phraseExclusivity() {
  const bossRef = new Set([...bossPartEmitterIds(), ...midBossEmitterIds()]);
  const mobRef = mobEmitterIds();
  let n = 0;
  for (const e of EMITTERS()) {
    if (!isObj(e)) continue;
    if (isAmb(e.repeat) || isAmb(e.restSec) || !num(e.repeat) || !num(e.restSec)) continue;
    n += 1;
    const inBoss = bossRef.has(e.id);
    const isPhrase = e.repeat >= 2 && e.restSec > 0;
    if (inBoss !== isPhrase) {
      V('S30', `enemies.emitters[${e.id}]: (bosses[] 에서 참조)=${inBoss} ≠ (repeat ≥ 2 ∧ restSec > 0)=${isPhrase} `
        + `— repeat=${e.repeat}, restSec=${e.restSec} (§8.5/S30). `
        + `★ 잡몹은 메트로놈(everySec), 보스는 악절(몰아치고 → 쉰다). restSec 가 곧 플레이어가 화력을 넣는 창이다`);
    }
    if (mobRef.has(e.id) && !(e.repeat === 1 && e.restSec === 0)) {
      V('S30', `enemies.emitters[${e.id}]: 잡몹(archetypes[].attack)이 참조하는데 repeat=${e.repeat}, restSec=${e.restSec} `
        + `— 잡몹은 repeat == 1 ∧ restSec == 0 (= 메트로놈) (§8.5/S30)`);
    }
  }
  EX('S30', n);
}

// ===========================================================================
//  S31 ★ — 위기 편성의 무결성 (§8.10 · §13.4-S31, v1.3)
//  Σ(crisisWaves[].count) == crisisTotal(60)
//  ∧ distinct(crisisWaves[].subWave) == crisisSubWaves(6)
//  ∧ archetypeId 가 전부 swarm*
// ===========================================================================
function S31_crisisComposition() {
  const ph = D.stages.phase;
  if (!isObj(ph)) return;
  const cws = rows('S31', ph.crisisWaves, 'stages.phase.crisisWaves',
    '§8.10 — 위기 편성이 0행이면 새떼가 아예 등장하지 않는다 (v1.3이 처음 인쇄한 12행)');
  if (!cws.length) return;

  let sum = 0;
  const subs = new Set();
  for (const cw of cws) {
    if (!isObj(cw)) continue;
    if (num(cw.count)) sum += cw.count;
    subs.add(cw.subWave);
    if (!/^swarm/.test(String(cw.archetypeId))) {
      V('S31', `stages.phase.crisisWaves: archetypeId "${cw.archetypeId}" 가 swarm* 가 아니다 — 위기 세션은 새떼 전용 (§8.10/S31)`);
    }
  }
  if (num(ph.crisisTotal) && sum !== ph.crisisTotal) {
    V('S31', `stages.phase.crisisWaves: Σ count = ${sum} ≠ crisisTotal(${ph.crisisTotal}) (§8.10/S31)`);
  }
  if (num(ph.crisisSubWaves) && subs.size !== ph.crisisSubWaves) {
    V('S31', `stages.phase.crisisWaves: distinct(subWave) = ${subs.size} ≠ crisisSubWaves(${ph.crisisSubWaves}) (§8.10/S31)`);
  }
  // subWave 는 1..crisisSubWaves 의 연속 정수여야 한다
  if (num(ph.crisisSubWaves)) {
    for (let i = 1; i <= ph.crisisSubWaves; i += 1) {
      if (!subs.has(i)) V('S31', `stages.phase.crisisWaves: subWave ${i} 이 없다 — 1..${ph.crisisSubWaves} 연속이어야 한다 (§9.9.3의 간격 파생식이 이것을 전제한다)`);
    }
  }
  EX('S31', cws.length);
}

// ===========================================================================
//  S32 ★ — themeId 의 적법성 (§9.8 · §13.4-S32, v1.3)
//  tier == "stage" ⟺ themeId != null   (S15 와 대칭)
// ===========================================================================
function S32_themeIdLegality() {
  let n = 0;
  for (const b of BOSSES()) {
    if (!isObj(b) || isAmb(b.themeId)) continue;
    n += 1;
    const lhs = b.tier === 'stage';
    const rhs = b.themeId !== null;
    if (lhs !== rhs) {
      V('S32', `bosses[${b.id}]: (tier=="stage")=${lhs} ≠ (themeId != null)=${rhs} — themeId=${JSON.stringify(b.themeId)} (§9.8/S32). `
        + `mid 는 3종 전 테마 공용(§8.9), final 은 테마 속성 없음(§8.16) → 둘 다 null`);
    }
  }
  EX('S32', n);
}

// ===========================================================================
//  S33 ★ — 아이콘 어휘의 충분성 (§9.4.1 · §13.4-S33, v1.3)
//  shop 의 전 항목의 iconId ∈ hud.icons
//  ★ 어휘를 닫는 것과 그 어휘가 충분한지는 다른 일이다
// ===========================================================================
function S33_iconSufficiency() {
  const icons = rowsQuiet(D.rules.hud && D.rules.hud.icons);
  if (!icons.length) {
    V('S33', 'rules.hud.icons: 0종 — 상점 10항목의 아이콘을 하나도 그릴 수 없다 (§9.4.1)');
    return;
  }
  const shop = D.meta.shop;
  if (!isObj(shop)) {
    V('S33', `meta.shop: 객체가 아니다 (${isAmb(shop) ? '__AMBIGUOUS__ — §11.2.1의 블록을 그대로 넣어라 (§23.1-D12)' : typeof shop})`);
    return;
  }
  let n = 0;
  for (const [id, item] of Object.entries(shop)) {
    if (!isObj(item)) continue;
    n += 1;
    if (isAmb(item.iconId)) continue;
    if (!icons.includes(item.iconId)) {
      V('S33', `meta.shop.${id}.iconId = ${JSON.stringify(item.iconId)} ∉ hud.icons [${icons.join(', ')}] (§9.4.1/S33) `
        + `— 선언하면 미지 값, 안 그리면 §11.2의 "표시 필수" 위반 = 양방향 실패`);
    }
  }
  EX('S33', n);
}

// ===========================================================================
//  S34 ★ — 패밀리별 base 필수 키 집합 (§9.5 12행 표 · §13.4-S34, v1.3)
//  각 weapons[i].base 의 키 집합 == 그 family 의 §9.5 표가 ✔한 공통 키 ∪ 고유 파라미터
//  ★ 이 표가 없으면 S2의 "필수 키"가 무엇인지 검증기가 알 수 없다
//  ★ 고유 파라미터 중 evo* 접두는 base 가 아니라 evolution.params 에 산다 (§9.5) — 아래 CANON 참조
// ===========================================================================
function S34_familyBaseKeys() {
  let n = 0;
  for (const w of rowsQuiet(D.weapons.weapons)) {
    if (!isObj(w) || !FAMILIES.includes(w.family)) continue;
    n += 1;
    const want = new Set([...FAMILY_COMMON_CHECK[w.family], ...FAMILY_OWN_BASE[w.family]]);
    const tag = `weapons[${w.id}].base`;
    if (!isObj(w.base)) { V('S34', `${tag}: 객체가 아니다 (§9.5)`); continue; }
    const got = new Set(Object.keys(w.base));
    for (const k of got) {
      if (!want.has(k)) {
        const isCommon = FAMILY_COMMON.includes(k);
        V('S34', `${tag}.${k}: 패밀리 "${w.family}" 의 계약에 없는 키 = 미지 = 에러 (§9.5 12행 표)`
          + (isCommon ? ` — 이 공통 키는 "${w.family}" 행에서 ✖다. ★ null 로 선언하는 것이 아니라 키 자체가 없다` : ''));
      }
    }
    for (const k of want) {
      if (!got.has(k)) {
        V('S34', `${tag}.${k}: 누락 키 = 에러 (§9.5 12행 표가 "${w.family}" 행에서 ✔로 확정했다)`);
      }
    }
    // 진화 파라미터는 evolution.params 가 소유한다 (evo* 접두 규약, §9.5)
    const wantEvo = new Set(FAMILY_OWN_EVO[w.family]);
    const params = isObj(w.evolution) && isObj(w.evolution.params) ? w.evolution.params : null;
    if (!params) { V('S34', `weapons[${w.id}].evolution.params: 객체가 아니다 (§9.5)`); continue; }
    const gotEvo = new Set(Object.keys(params));
    for (const k of gotEvo) {
      if (!k.startsWith('evo')) {
        V('S34', `weapons[${w.id}].evolution.params.${k}: evo* 접두가 아니다 (§9.5 진화 계약)`);
      } else if (!wantEvo.has(k)) {
        V('S34', `weapons[${w.id}].evolution.params.${k}: 패밀리 "${w.family}" 계약 밖의 evo 키 (§9.5 12행 표)`);
      }
    }
    for (const k of wantEvo) {
      if (!gotEvo.has(k)) V('S34', `weapons[${w.id}].evolution.params.${k}: 누락 키 = 에러 (§9.5 12행 표의 고유 파라미터)`);
    }
    // levels[] 는 base 에 대한 부분 오버라이드 (§9.3의 유일한 예외) — 계약 밖 키는 금지
    rowsQuiet(w.levels).forEach((lv, i) => {
      if (!isObj(lv)) return;
      for (const k of Object.keys(lv)) {
        if (!want.has(k)) {
          V('S34', `weapons[${w.id}].levels[${i}].${k}: 계약 밖의 키 — levels[] 는 base 의 부분 오버라이드다 (§9.3/§9.5)`);
        }
      }
    });
    // §9.5 허용 targetMode (패밀리별)
    const tm = FAMILY_TARGET_MODES[w.family];
    if (isObj(w.base) && tm && has(w.base, 'targetMode') && !isAmb(w.base.targetMode)
        && !tm.includes(w.base.targetMode)) {
      V('S34', `${tag}.targetMode = ${JSON.stringify(w.base.targetMode)} — "${w.family}" 허용 = [${tm.join(', ')}] (§9.5 표)`);
    }
  }
  EX('S34', n);

  // §9.6.1 정합: pierceApplies ✔ 인 패밀리는 전부 base.pierce 를 갖는다 (§9.5 읽는 법 3)
  const hooks = D.rules.passiveHooks;
  if (isObj(hooks)) {
    for (const f of FAMILIES) {
      const h = hooks[f];
      if (!isObj(h)) continue;
      if (h.pierceApplies === true && !FAMILY_COMMON_CHECK[f].includes('pierce')) {
        V('S34', `rules.passiveHooks.${f}.pierceApplies = true 인데 §9.5 표가 "${f}" 행의 pierce 를 ✖로 확정했다 `
          + `— eff.pierce = base.pierce + v 가 undefined 를 읽는다 (§9.5 읽는 법 3 · §9.6.1)`);
      }
      // rateKey / countKey / areaKeys 가 그 패밀리의 유효 파라미터 공간 안에 있는가
      //   src = base ∪ (evolved ? evolution.params : {})  — §9.6.1 v1.3
      const space = new Set([...FAMILY_COMMON_CHECK[f], ...FAMILY_OWN_BASE[f], ...FAMILY_OWN_EVO[f]]);
      if (h.rateKey !== null && !space.has(h.rateKey)) {
        V('S34', `rules.passiveHooks.${f}.rateKey = ${JSON.stringify(h.rateKey)} 가 "${f}" 의 계약에 없다 (§9.6.1)`);
      }
      if (h.countKey !== null && !space.has(h.countKey)) {
        V('S34', `rules.passiveHooks.${f}.countKey = ${JSON.stringify(h.countKey)} 가 "${f}" 의 계약에 없다 (§9.6.1)`);
      }
      for (const k of rowsQuiet(h.areaKeys)) {
        if (!space.has(k)) {
          V('S34', `rules.passiveHooks.${f}.areaKeys 의 ${JSON.stringify(k)} 가 "${f}" 의 계약에 없다 `
            + `(§9.6.1 — src = base ∪ evolution.params 가 유효 파라미터 공간이다)`);
        }
      }
    }
  }
}

// ===========================================================================
//  S35 ★ — values 의 길이 (§9.6 · §13.4-S35, v1.3)
//  passives[] 12행 전부 len(values) == maxLevel(5)
// ===========================================================================
function S35_passiveValuesLen() {
  const maxLevel = D.passives.maxLevel;
  let n = 0;
  for (const p of rows('S35', D.passives.passives, 'passives.passives', '§9.6 — 12종 1:1')) {
    if (!isObj(p)) continue;
    n += 1;
    if (!Array.isArray(p.values)) { V('S35', `passives[${p.id}].values: 배열이 아니다 (§9.6)`); continue; }
    if (num(maxLevel) && p.values.length !== maxLevel) {
      V('S35', `passives[${p.id}].values: 길이 ${p.values.length} ≠ maxLevel(${maxLevel}) (§9.6/S35)`);
    }
  }
  EX('S35', n);
}

// ===========================================================================
//  S36 ★ — 보스 이미터 id 규칙 (§9.8.1 · §13.4-S36, v1.3)
//  bosses[].parts[i].patternSet[j].emitterIds[0] == {bossId}{PartIdPascal}P{j+1}
//  ★ id 가 규칙에서 벗어나면 로드 실패. 66개가 규칙의 인스턴스이므로 저작할 것이 하나도 없다
// ===========================================================================
function S36_bossEmitterIdRule() {
  let n = 0;
  for (const b of BOSSES()) {
    if (!isObj(b) || b.tier === 'mid') continue;
    for (const p of rowsQuiet(b.parts)) {
      if (!isObj(p)) continue;
      rowsQuiet(p.patternSet).forEach((ps, j) => {
        const ids = rowsQuiet(ps && ps.emitterIds);
        if (!ids.length) return;
        n += 1;
        const want = `${b.id}${pascal(p.id)}P${j + 1}`;
        if (ids[0] !== want) {
          V('S36', `bosses[${b.id}].parts[${p.id}].patternSet[${j}].emitterIds[0] = ${JSON.stringify(ids[0])} ≠ "${want}" `
            + `— §9.8.1-확정② id 명명 규칙 = {bossId}{PartIdPascal}P{phase}. `
            + `★ 이름이 부위를 가리켜야 §8.12의 "부위 파괴 = 그 부위의 이미터 정지"가 id 에 내장된다`);
        }
      });
    }
  }
  EX('S36', n);
  if (n && n !== 66) {
    C('S36', `보스 부위 이미터 슬롯이 ${n}개 — §9.8.1 은 66개(보스 6종 × 부위 3~4 × 페이즈 3 + 최종)라 인쇄했다. `
      + `개수가 다르면 §23.1-D4 의 저작 범위가 바뀐 것이다`);
  }
}

// ===========================================================================
//  S37 ★ — 보스 이미터의 존재 (§9.8.1 · §13.4-S37, v1.3)
//  위 66개가 enemies.json > emitters 에 전부 존재 (참조 무결성의 정적 판본)
//  ★ 이 66칸이 비어 있으면 stunMark 참조 0 → 스턴 메커닉이 게임에 존재하지 않는다
// ===========================================================================
function S37_bossEmitterExists() {
  const emitIds = new Set(EMITTERS().map((e) => e && e.id));
  let n = 0, missing = 0;
  for (const b of BOSSES()) {
    if (!isObj(b) || b.tier === 'mid') continue;
    for (const p of rowsQuiet(b.parts)) {
      if (!isObj(p)) continue;
      rowsQuiet(p.patternSet).forEach((ps, j) => {
        // ★ 규칙이 id 를 확정하므로 저작 여부와 무관하게 "있어야 할 id" 를 계산할 수 있다
        const want = `${b.id}${pascal(p.id)}P${j + 1}`;
        n += 1;
        if (!emitIds.has(want)) {
          missing += 1;
          V('S37', `enemies.emitters: 보스 부위 이미터 "${want}" 가 존재하지 않는다 `
            + `(bosses[${b.id}].parts[${p.id}] 페이즈 ${j + 1}) — §9.8.1/§23.1-D4. `
            + `id·거처·개수는 규칙이 확정했다. 저작이 남은 것은 내용뿐이다`);
        }
      });
    }
  }
  EX('S37', n);
  if (missing) {
    V('S37', `★ 보스 부위 이미터 ${missing}/${n} 개가 없다 — §9.3 참조 무결성 ${missing}건 실패 + S7·S13·S16 이 읽을 대상이 없다. `
      + `그리고 stunMark 를 참조하는 이미터가 0개면 스턴 메커닉 전체가 도달 불가능한 콘텐츠다 (§9.8.1)`);
  }
}

// ===========================================================================
//  S38 ★ — 중간보스 이탈의 단일 소유자 (§9.8.2 · §13.4-S38, v1.3)
//  tier == "mid" ⟹ moveParams 에 leaveAfterSec 부재
//  ★ anchor 의 leaveAfterSec 는 잡몹 전용 파라미터다
// ===========================================================================
function S38_midBossLeave() {
  let n = 0;
  for (const b of BOSSES()) {
    if (!isObj(b) || b.tier !== 'mid') continue;
    n += 1;
    if (isObj(b.moveParams) && has(b.moveParams, 'leaveAfterSec')) {
      V('S38', `bosses[${b.id}].moveParams.leaveAfterSec: 중간보스에는 없다 (§9.8.2/S38) `
        + `— 이탈의 유일 소유자 = stages.phase.midBossLeaveAfterSec(${D.stages.phase && D.stages.phase.midBossLeaveAfterSec}). `
        + `★ §21-A12가 삭제한 bosses[].leaveAfterSec 가 한 단계 아래에서 부활한 것이다`);
    }
  }
  EX('S38', n);
}

// ===========================================================================
//  S39 ★ — 웨이브 해금의 정합 (§9.9 · §13.4-S39, v1.3)
//  waves[i].unlockStageMin ≥ roster[waves[i].archetypeId].unlockStageMin
//  ★ 해금 안 된 적이 나오는 웨이브 금지
//  ★ 두 unlockStageMin 은 다른 축이다: 로스터 = 아키타입 해금 / 웨이브 = 블록 티어
// ===========================================================================
function S39_waveUnlockCoherence() {
  let n = 0;
  for (const t of rows('S39', D.stages.stages, 'stages.stages',
    '§9.9 — 웨이브 해금 정합이 0행을 보면 해금 안 된 적이 나오는 웨이브를 아무도 못 잡는다')) {
    if (!isObj(t)) continue;
    const ros = new Map(rowsQuiet(t.roster).map((r) => [r && r.archetypeId, r && r.unlockStageMin]));
    rowsQuiet(t.waves).forEach((w, i) => {
      if (!isObj(w)) return;
      const ru = ros.get(w.archetypeId);
      if (!num(w.unlockStageMin) || !num(ru)) return;
      n += 1;
      if (w.unlockStageMin < ru) {
        V('S39', `stages.stages[${t.id}].waves[${i}] (${w.archetypeId}): waves.unlockStageMin ${w.unlockStageMin} < roster.unlockStageMin ${ru} `
          + `— 해금 안 된 적이 나오는 웨이브다 (§9.9/S39)`);
      }
    });
  }
  EX('S39', n);
}

// ===========================================================================
//  S40 ★ — 상점 스키마 (§11.2.1 · §13.4-S40, v1.3)
//  shop 의 키 집합 == §11.2 표의 id 10종
//  ∧ (stockMax 보유 ⟺ id ∈ {reroll, shield, timeToken})
//  ★ 폭탄의 상한은 rules.bomb.stockMax 가 소유한다 (상한은 그것이 제한하는 상태와 함께 산다)
// ===========================================================================
const SHOP_IDS = ['reroll', 'potion', 'bomb', 'shield', 'timeToken',
  'defense', 'maxhp', 'movespeed', 'magnet', 'resist'];
const SHOP_STOCKMAX_IDS = ['reroll', 'shield', 'timeToken'];
// §11.2.1 — 항목별 효과 파라미터 (전부 기존 산문의 직역. 새 모델 0)
const SHOP_EFFECT_KEYS = {
  reroll: ['addStock'],
  potion: ['healPct'],
  bomb: ['addStock'],
  shield: ['addStock'],
  timeToken: ['addStock', 'addSec'],
  defense: ['addDefense'],
  maxhp: ['addHpMax', 'healsSameAmount'],
  movespeed: ['addMoveSpeedPct'],
  magnet: ['addMagnetPct'],
  resist: ['statusDurationPct'],
};

function S40_shopSchema() {
  const shop = D.meta.shop;
  if (!isObj(shop)) {
    V('S40', `meta.shop: 객체가 아니다 — §11.2.1의 인쇄 블록을 그대로 넣어라 (§23.1-D12: "shop": "__AMBIGUOUS__" 1칸이 ~50값으로 풀린다)`);
    return;
  }
  closedKeys('S40', shop, SHOP_IDS, 'meta.shop');
  let n = 0;
  for (const id of SHOP_IDS) {
    const item = shop[id];
    if (!isObj(item)) continue;
    n += 1;
    const wantStock = SHOP_STOCKMAX_IDS.includes(id);
    const allowed = ['basePrice', 'growth', 'maxPurchases', 'iconId',
      ...(wantStock ? ['stockMax'] : []), ...(SHOP_EFFECT_KEYS[id] || [])];
    closedKeys('S40', item, allowed, `meta.shop.${id}`);
    // ★ stockMax 보유 ⟺ id ∈ {reroll, shield, timeToken}
    const hasStock = has(item, 'stockMax');
    if (hasStock !== wantStock) {
      V('S40', `meta.shop.${id}: (stockMax 보유)=${hasStock} ≠ (id ∈ {${SHOP_STOCKMAX_IDS.join(', ')}})=${wantStock} (§11.2.1/S40)`
        + (id === 'bomb' ? ' — ★ 폭탄의 상한은 rules.bomb.stockMax 가 소유한다 (§9.4 "상한은 그것이 제한하는 상태와 함께 산다")' : ''));
    }
    // §11.2: 모든 스탯 항목에 maxPurchases (무한 스택으로 코인이 빌드를 사는 것을 구조적으로 차단)
    if (!num(item.maxPurchases) || item.maxPurchases < 1) {
      V('S40', `meta.shop.${id}.maxPurchases = ${JSON.stringify(item.maxPurchases)} — 양의 정수여야 한다 (§11.2)`);
    }
    if (num(item.growth) && item.growth < 1.0) {
      V('S40', `meta.shop.${id}.growth = ${item.growth} < 1.0 — 가격이 내려가면 희소성 설계가 뒤집힌다 (§11.2)`);
    }
  }
  // §11.2.1: bombStockMax 는 rules.bomb.stockMax 의 인용으로 강등됐다
  if (has(shop, 'bombStockMax')) {
    V('S40', 'meta.shop.bombStockMax: 삭제된 이름 (§9.4/§23.3) — rules.bomb.stockMax 를 인용하라. 02는 전자를, 03·05는 후자를 쓰고 있었다');
  }
  EX('S40', n);
}

// ===========================================================================
//  §13.1 certify 게이트 — 정적으로 검사 가능한 것
//  ★ v1.3: certify.m 이 인쇄됐다 (§13.1.0) → 19개 스텁이 읽을 값을 갖는다
// ===========================================================================
function certifyStatic() {
  const c = D.meta.certify;
  if (!isObj(c)) return;

  // (1) 인쇄된 게이트 세트의 필드 집합 (§13.1 = certify 를 인쇄하는 유일한 절, C-10)
  //     ★ v1.3: m 추가. v1.4 정정: 거처는 meta.json 이다 (rules.json 이 아니다)
  closedKeys('CERT', c, ['runs', 'dpsRef', 'runFarmDpsRatio', 'm', 'runMode', 'dpsProbe', 'static'], 'meta.certify');
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
    ['certify.m', c.m],                                    // ★ v1.3 신설 (§13.1.0)
    ['stages.curve.enemyHpScale', D.stages.curve && D.stages.curve.enemyHpScale],
    ['stages.curve.xpScale', D.stages.curve && D.stages.curve.xpScale],
    ['stages.curve.bossHpScale', D.stages.curve && D.stages.curve.bossHpScale],
    ['stages.curve.spawnDensityScale', D.stages.curve && D.stages.curve.spawnDensityScale],
    ['stages.curve.midBossCount', D.stages.curve && D.stages.curve.midBossCount],
    ['stages.curve.elitePerWaveChance', D.stages.curve && D.stages.curve.elitePerWaveChance],
    ['stages.curve.swarmTotalScale', D.stages.curve && D.stages.curve.swarmTotalScale],
    ['stages.curve.rearSpawnAllowed', D.stages.curve && D.stages.curve.rearSpawnAllowed],
    ['stages.phase.midBossAtSec', D.stages.phase && D.stages.phase.midBossAtSec],
    ['meta.flow.stagePar', D.meta.flow && D.meta.flow.stagePar],
    ['certify.dpsProbe.balancedPass.min', c.dpsProbe && c.dpsProbe.balancedPass && c.dpsProbe.balancedPass.min],
    ['certify.dpsProbe.specialistPass.min', c.dpsProbe && c.dpsProbe.specialistPass && c.dpsProbe.specialistPass.min],
    ['certify.dpsProbe.noElementPass.min', nep && nep.min],
  ];
  for (const [path, arr] of six) {
    if (Array.isArray(arr) && arr.length !== 6) V('CERT', `${path}: 길이 ${arr.length} ≠ 6 (스테이지 축)`);
  }
  // ★ §13.1.0: m 은 스탠스 배율이므로 ≥ 1.0 이고 단조 비감소다 (투자는 이월된다 — stancePersistAcrossStages)
  if (Array.isArray(c.m) && c.m.every(num)) {
    for (let i = 0; i < c.m.length; i += 1) {
      if (c.m[i] < 1.0) V('CERT', `certify.m[${i}] = ${c.m[i]} < 1.0 — 스탠스 배율은 1 미만일 수 없다 (§13.1.0)`);
      if (i > 0 && c.m[i] < c.m[i - 1]) {
        V('CERT', `certify.m: 스테이지 ${i + 1} 에서 감소 (${c.m[i - 1]} → ${c.m[i]}) — stancePersistAcrossStages=true 이므로 투자는 이월된다 (§13.1.0-④)`);
      }
    }
  }

  // (4) §13.5.1 runFarmDpsRatio — probe farm 정책과 dpsProbe.farm 이 같아야 한다
  if (c.dpsProbe && c.dpsProbe.farm !== 'maxFarm') {
    V('CERT', `certify.dpsProbe.farm = ${JSON.stringify(c.dpsProbe.farm)} ≠ "maxFarm" — §13.5 "dpsRef 의 farm 정책 = dpsProbe.farm 과 같다"`);
  }
  // bot.policies 안의 값이어야 한다
  const pol = D.meta.bot && D.meta.bot.policies;
  if (isObj(pol)) {
    if (c.dpsProbe && Array.isArray(pol.farm) && !pol.farm.includes(c.dpsProbe.farm)) {
      V('CERT', `certify.dpsProbe.farm = ${JSON.stringify(c.dpsProbe.farm)} 이 bot.policies.farm 에 없다 (§10.4.1)`);
    }
    const bl = D.meta.bot.baseline;
    if (isObj(bl)) for (const ax of ['draft', 'farm', 'stance', 'shop']) {
      if (Array.isArray(pol[ax]) && !pol[ax].includes(bl[ax])) {
        V('CERT', `meta.bot.baseline.${ax} = ${JSON.stringify(bl[ax])} 이 policies.${ax} 에 없다 (§10.4.1)`);
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
  // §13.6.1: bossHpScale[5] == bossHpScale[6] (dpsRef[5]==dpsRef[6] 이므로)
  const bhs = D.stages.curve && D.stages.curve.bossHpScale;
  if (Array.isArray(bhs) && Array.isArray(c.dpsRef) && bhs.length === 6 && c.dpsRef.length === 6) {
    if (c.dpsRef[4] === c.dpsRef[5] && bhs[4] !== bhs[5]) {
      V('CERT', `stages.curve.bossHpScale[5](${bhs[4]}) ≠ [6](${bhs[5]}) 인데 dpsRef[5]==dpsRef[6]==${c.dpsRef[4]} — §13.6.1`);
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
 *     m: [6],                                                     // ★ §13.1.0 교정 프로토콜
 *     capHits: { enemyConcurrentMax, swarmConcurrentMax, crisisWaveResidualMax,
 *                telegraphConcurrentMaxGlobal, capsOverflow },    // §13.1.1 — 4축 분리 출력
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
    ['bossTimeoutRate', '분모 = runs 전체. 분자 = deaths.csv 사인이 "시간 초과"인 런. 컨티뉴는 사인을 리셋 (§13.1.1)'],
    ['noDeadLuck', '테마 순서 720개를 스테이지 5 테마별 6군집(각 ≥1300런)으로 집계 + draft 축 6정책 각각의 runClearRate 최솟값 (§13.1.1)'],
    ['stanceValue', 'runClearRate(baseline) − runClearRate(stance="static", 나머지 3축 baseline) (§13.1.1)'],
    ['difficultySpread', 'disaster(speed 3.0) 의 runClearRate ∈ [0.02, 0.12] (§13.1)'],
    ['dominance.maxWeaponPickShare', `분모 = runs × 3 = ${(c.runs || 0) * 3}. forward 제외 후 11종 재정규화 (§13.1.1)`],
    ['dominance.maxWeaponWinShare', '★ 피해 지분의 평균. 분모 = 클리어 런 수. forward 제외 후 11종 재정규화 (§13.1.1)'],
    ['dominance.startWeaponDamageShare', 'forward 전용. 분모 = 클리어 런의 4무기 총 피해 (§13.1.1)'],
    ['dominance.maxElementWinShare', '분모 = 클리어 런의 총 속성 투자 픽 수(런당 ≤ 6). 3종 재정규화 (§13.1.1)'],
    ['dominance.maxArchetypeLethalityShare', '분모 = 전 런에서 플레이어가 입은 총 피해(실드 흡수 제외). 대상 = 잡몹 15 + 새떼 2 = 17종. 엘리트는 원 아키타입 귀속, 중간보스·보스·부위는 분모에서도 제외 (§13.1.1)'],
    ['dominance.maxThemeClearStddev', '테마 t별 clearRate 6개 값의 표본 표준편차. finale 제외 (§13.1.1)'],
    ['coinScarcity', 'medianEndCoins / medianPurchasesPerVisit(분모 = 상점 방문 수, 런당 5) / p90EndCoins (§13.1.1)'],
    ['farmXpRatio', '(maxFarm 스테이지 평균 XP) ÷ (passive 스테이지 평균 XP). 나머지 3축 baseline (§13.1.1)'],
    ['crisisKillShareWithoutCapstone', 'capstone = 보유 무기에 nova 또는 aura. 대상 = capstone 미보유 ∧ 그 세션 폭탄 미사용. killShare = 처치 새떼 수 ÷ (crisisTotal × swarmTotalScale[stage]) 의 중앙값 (§13.1.1)'],
  ];
  for (const [k, why] of todo) {
    S('CERT-DYN', `${k}: 시뮬(run 모드) 필요 → TODO: tools/sim.mjs + report/summary.json. ${why}`);
  }
  S('CERT-DYN', `dpsProbe (balancedPass/specialistPass/noElementPass/killTimeMedianBalanced): `
    + `셀 = (보스, 스테이지) 쌍 = 3 + 24 + 1 = 28 셀 × runsPerCell(${dp.runsPerCell}), farm="${dp.farm}", uptimeRef=${dp.uptimeRef} (§10.4.2)`);
  S('CERT-DYN', `capHits: ★ A층(enemyConcurrentMax·swarmConcurrentMax·crisisWaveResidualMax·telegraphConcurrentMaxGlobal 의 defer) `
    + `+ B층(caps.* overflow, 순수 FX 3종 제외) 를 4축 분리 출력. 상한 ${(c.static && c.static.capHits && c.static.capHits.max)} (§13.1.1)`);
  S('CERT-DYN', `fairnessViolations: 런타임 어서션(특히 minSpawnRadiusPx — v1.3이 S6에서 여기로 옮겼다). `
    + `상한 ${(c.static && c.static.fairnessViolations && c.static.fairnessViolations.max)} (§13.1/§13.4-S6)`);
  // ★ v1.3 신설 — certify.m 의 교정 프로토콜 (§13.1.0)
  S('CERT-DYN', `certify.m = [${rowsQuiet(c.m).join(', ')}]: 스테이지별 실측 스탠스 배율을 6항 배열로 report/summary.json 에 출력하라. `
    + `허용오차 ★ ±0.04 — 벗어나면 certify.m 을 고치고 §13.2-①③④⑨⑩ 과 §13.6.2 를 재검산한다(값 변경, 구조 불변). `
    + `uptimeRef(±0.05) · runFarmDpsRatio 와 완전히 같은 처방의 세 번째 사례다 (§13.1.0)`);
}

// ===========================================================================
//  정본 결함 — 검사를 쓰면서 드러난 것 (하드코딩 신고)
//  ★ 발명하지 않는다: 아래는 전부 "정본이 답하지 않아 검사를 완성할 수 없는 자리"다
//
//  ★★ v1.4 기준 — v1.2 판본이 신고하던 C-1 ~ C-13 은 전부 해소됐다:
//    C-1(S 번호)        → v1.3 표를 번호순 재배열 + 인용 "S1~S40" 통일 (§13.4)
//    C-2(계약 필수 키)  → §9.5 12행 표 신설 → S34 가 검사한다
//    C-3(from/repeat/restSec) → §8.5 from 2종 어휘 + 악절 규칙 → S3·S28·S30 + S7 전개 모델
//    C-4(보스 "대형")   → §7.4 "보스 부위 패턴" 1.50 (참조 경로로 기계 판별) → S6
//    C-5(탄 속도 거처)  → bullets[].speed 삭제, emitters[].speed 유일 소유 (§9.7)
//    C-6(statusBulletSpeedMul) → visual → fairness 이사 (§23.3) → S6 이 읽는다
//    C-7(themeId)       → string|null 확정 → S32
//    C-8(finale roster) → waves[].unlockStageMin 신설, finale 전부 1 (§9.9)
//    C-9(S22 정의역)    → "모든 (theme, stage) 쌍" 명문화 (§13.4-S22)
//    C-10(S10 좌변)     → 선언 상수 비교 + 유도 검사로 문면 수정 (§13.4-S10)
//    C-11(minSpawnRadiusPx) → S6 에서 제거 → certify.static.fairnessViolations (런타임)
//    C-12(finale.armorCoreRatio) → 삭제, bosses[].armorCoreRatio 유일 소유 (§23.3)
//    C-13(rules.audio.bgm) → 삭제 (BGM 스코프 아웃, §7.10)
// ===========================================================================
function canonDefects() {
  // N-1(S34): CANON v1.4 에서 닫힘 — §13.4-S34 의 문면이 두 검사로 쪼개졌고,
  //   §9.5 「읽는 법」 4번째 규칙이 표의 「+」를 거처 구분자로 명시한다.
  //   이 파일의 S34_familyBaseKeys() 구현이 곧 그 문면이다. 신고할 결함 없음.
  //
  // 새 결함을 발견하면 여기에 C('Sxx', '...') 로 추가한다.
  // hardFail 조건이 (violations > 0 || canonDefects > 0) 이므로,
  // 정본 결함 1건이면 데이터가 완전해도 exit 1 이다 — 의도된 동작이다(C-6).
}

// ===========================================================================
//  ★ 공허 통과 감시 — 게이트가 0행을 봤으면 그것은 통과가 아니다
// ===========================================================================
function vacuousWatch() {
  for (const check of VACUOUS_WATCH) {
    const n = examined[check] || 0;
    if (n === 0) {
      V('VACUOUS', `${check}: ★ 검사한 행 0개 — 이 게이트는 아무것도 인증하지 않았다. `
        + `"위반이 안 나왔다"가 아니라 "볼 것이 없었다"이며 이것은 통과가 아니다. `
        + `참조가 끊겼거나(개명·오타) 콘텐츠가 비었다`);
    }
  }
}

// ===========================================================================
//  출력
// ===========================================================================
function print() {
  const line = (s = '') => { if (!QUIET) console.log(s); };
  const bar = '─'.repeat(78);

  line();
  line('NAN 2026 — check.mjs   (정본 v1.4 §13.4 S1~S40 + §9.3 로더 규칙)');
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

  // ★ 게이트가 실제로 본 행 수 — 공허 통과가 아님의 증거
  if (!QUIET) {
    line();
    line(`검사한 행 수 (0 = 공허 통과 = 위반)`);
    line(bar);
    const cells = VACUOUS_WATCH.map((k) => `${k}:${examined[k] || 0}`);
    for (let i = 0; i < cells.length; i += 8) line(`  ${cells.slice(i, i + 8).join('  ')}`);
  }

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
  line('✓ 전 정적 게이트 통과 (S1~S40)');
  line();
  return 0;
}

// ===========================================================================
//  main — ★ S 번호는 §13.4 의 표와 1:1 (v1.3: 표를 번호순 재배열, S1~S40)
// ===========================================================================
function main() {
  loadAll();
  census();             // ★ 참조가 끊겼는가 / 콘텐츠가 비었는가 — 첫 줄에서 잡는다

  S1_corePurity();      // §9.1  core 순수성
  S2_schema();          // §9.3 로더 + §9.4 rules
  S2_files();           // §9.5~§9.9 파일별 스키마
  refIntegrity();       // §9.3 참조 무결성
  S3_vocab();           // §13.4-S3 어휘 + §9.8.2 다중 동치
  S4_archetypeOverlap();// §8.6
  S5_bossRules();       // §8.14 R1~R7 + §8.16
  S6_fairness();        // §12.4 · §7.4
  S7_concurrentTelegraphs(); // §12.4 · §8.9-R8 (악절 모델, §8.5)
  S8_mix();             // §8.2 · §8.2.1
  S9_structure();       // §13.4-S9
  S10_growthBudget();   // §11.1 · §13.1
  S11_rngStreams();     // §10.2
  S12_twoLayerCaps();   // §12.1
  S13_stunHome();       // §13.4-S13 · §9.8.1-③
  S14_shapeStatusEquiv();    // §9.7
  S15_midBossElement();      // §8.9
  S16_patternSetLen();       // §8.9-R8 · §9.8
  S17_summon();              // §8.9-R9
  S18_mobilityTruth();       // §8.12.1
  S19_zoneBullet();          // §9.7
  S20_formationExclusivity();// §9.9.2
  S21_draftGuarantees();     // §11.1
  S22_swarmXpShare();        // §8.10
  S23_coinSourceHomogeneity();// §13.4-S23
  S24_hpDistribution();      // §13.6.4
  S25_elementMatrix();       // §9.4.4
  S26_concurrentBudget();    // §12.1
  // ★ v1.3 신설 14개
  S27_eliteLegality();       // §8.6
  S28_fromLegality();        // §8.5
  S29_midBossSchedule();     // §8.9
  S30_phraseExclusivity();   // §8.5
  S31_crisisComposition();   // §8.10
  S32_themeIdLegality();     // §9.8
  S33_iconSufficiency();     // §9.4.1
  S34_familyBaseKeys();      // §9.5 12행 표
  S35_passiveValuesLen();    // §9.6
  S36_bossEmitterIdRule();   // §9.8.1
  S37_bossEmitterExists();   // §9.8.1
  S38_midBossLeave();        // §9.8.2
  S39_waveUnlockCoherence(); // §9.9
  S40_shopSchema();          // §11.2.1

  certifyStatic();      // §13.1 중 정적으로 검사 가능한 것
  dynamicGateStubs();   // 시뮬 필요 → TODO
  canonDefects();       // 검사를 쓰면서 드러난 정본 결함
  vacuousWatch();       // ★ 0행 게이트 = 공허 통과 = 위반

  process.exit(print());
}

main();
