#!/usr/bin/env node
/**
 * ============================================================================
 *  PRISM WING — check.mjs   (정본 v1.5 §13.4 정적 게이트 S1~S47 + §9.3 로더 규칙)
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
const report = { violation: [], canon: [], ambiguous: [], stub: [], skip: [], dynamic: [] };

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
  'S30', 'S31', 'S32', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S41',
  'S47', 'S49', 'S50', 'S51', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59', 'REF',
];
// ★ S38(중간보스 이탈)은 v1.3 콘텐츠 게이트(S27~S40) 중 유일하게 VACUOUS_WATCH 에서
//   빠져 있어, 중간보스 0행이면 EX('S38',0)이 공허 통과했다. §8.9/curve.midBossCount 가
//   중간보스를 필수로 요구하므로(현재 mbHammer·mbLancer·mbNest 3기) 감시 대상에 편입한다.

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
// §9.2 파일 매니페스트 — 정확히 11개, 닫힘 (v1.10 ⑲ traits)
// ---------------------------------------------------------------------------
const MANIFEST = [
  'rules', 'elements', 'weapons', 'passives', 'bullets',
  'enemies', 'bosses', 'stages', 'meta', 'traits', 'tutorial',   // ㊴ §6.7 튜토리얼
];
const SCHEMA_VERSION = 1;   // §9.4~§9.9 의 전 인쇄 블록이 1을 인쇄한다

const D = {};
function loadAll() {
  if (!existsSync(DATA_DIR)) {
    console.error(`FATAL: data 디렉터리가 없다: ${DATA_DIR}`);
    process.exit(2);
  }
  // 매니페스트가 닫혀 있으므로 여분 파일도 에러다 (§9.2 "정확히 11개, 닫힘")
  const present = readdirSync(DATA_DIR).filter((f) => extname(f) === '.json');
  const expect = new Set(MANIFEST.map((n) => `${n}.json`));
  for (const f of present) {
    if (!expect.has(f)) V('S2', `data/${f}: §9.2 매니페스트(정확히 11개, 닫힘) 밖의 파일`);
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
    ['weapons.weapons', D.weapons && D.weapons.weapons, 14, '§9.5 — 14 패밀리 1:1 (㉟ 5종 신설 · ㊵ 스파이럴 삭제)'],
    ['passives.passives', D.passives && D.passives.passives, 13, '§9.6 — 13종 (㉟ 추진기·장기 배터리)'],
    ['stages.phase.crisisWaves', D.stages && D.stages.phase && D.stages.phase.crisisWaves, 6, '§9.9 — 6행 (v1.10 ⑥ 서브웨이브당 1행)'],
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
const MOVE_IDS = ['dive', 'weave', 'column', 'strafe', 'anchor', 'orbitDrift', 'charge', 'bounce'];              // §8.4 (8 — v1.7: bounce 신설 · rearIn 폐지)
const EMITTER_TYPES = ['straight', 'fan', 'aimed', 'ring', 'spiral', 'laser', 'zone', 'wall', 'mortar', 'sweep']; // §8.5 (10, v1.5 mortar·sweep)
const FORMATION_IDS = ['lineH', 'columnV', 'vWedge', 'arc', 'pincer', 'scatter', 'wall'];                        // §8.7 · §9.9.2 (7 — v1.8 wall)
const PART_TYPES = ['mobility', 'armament', 'armor', 'core'];                                                    // §8.12 (4)
const SHAPE_IDS = ['wedge', 'delta', 'hexPod', 'orb', 'cross', 'spike', 'ring', 'slab', 'fin', 'claw', 'dart', 'bulb']; // §9.10 (12)
const TARGET_MODES = ['forward', 'nearest', 'lowestHp', 'densest', 'randomInArena', 'sweep'];                    // §9.5 (6 — ㉚ sweep)
const FAMILIES = ['forward', 'fan', 'seeker', 'lance', 'orbit', 'aura', 'boomerang', 'barrage', 'drone', 'nova', 'missile', 'chain', 'beam', 'pinball']; // §9.5 (㉟ 15 → ㊵ 14, 스파이럴 삭제)
const WEAPON_CLASSES = ['bullet', 'beam', 'area', 'orbital'];   // §9.5 ㊲ 분류 어휘 · ㊵ 값의 소유자는 weapons.json 의 class 다
/** family → class. ★ ㊵: 표를 여기서 «만들지» 않는다 — 데이터를 읽는다(중복 표는 조용히 어긋난다). D 는 로드 뒤에 찬다 → 호출 시 조회. */
function weaponClassOf(family) {
  const list = D.weapons && Array.isArray(D.weapons.weapons) ? D.weapons.weapons : [];
  for (const w of list) if (isObj(w) && w.family === family) return w.class;
  return undefined;
}
const PASSIVE_STATS = ['fireRateMul', 'projCountAdd', 'pierceAdd', 'projSpeedMul', 'durationMul',
  'beamDmgMul', 'beamAreaMul', 'areaMul', 'areaDmgMul', 'orbitMul',
  'maxHpAdd', 'terrainResist', 'xpGainMul', 'elementBonusMul'];   // §9.6 (14 — ㊲ 공용 1·탄 4·빔 2·범위 2·궤도 1·기체 4)
const BODY_STATS = ['maxHpAdd', 'terrainResist', 'xpGainMul', 'elementBonusMul'];                                 // §9.6 ㊲ 기체 4(무기 짝이 될 수 없다)
const DMG_STATS = ['beamDmgMul', 'areaDmgMul', 'orbitMul'];                                                                  // §9.6.1 ㊲ dmgStat 어휘
const HOOK_KEYS = ['rateKey', 'countKey', 'pierceApplies', 'speedKeys', 'durationKeys', 'areaKeys', 'beamKeys', 'orbitKeys', 'dmgStat'];
/** §11.1 ㊲ — 패시브 stat 이 패밀리에 기계적으로 유효한가(state.js passiveAppliesTo 와 같은 표 — check 는 독립 사본) */
function passiveAppliesTo(h, base, stat) {
  switch (stat) {
    case 'fireRateMul': return h.rateKey !== null;
    case 'projCountAdd': return h.countKey !== null;
    case 'pierceAdd': return h.pierceApplies === true && base.pierce !== -1;
    case 'projSpeedMul': return Array.isArray(h.speedKeys) && h.speedKeys.length > 0;
    case 'durationMul': return Array.isArray(h.durationKeys) && h.durationKeys.length > 0;
    case 'beamDmgMul': return h.dmgStat === 'beamDmgMul';
    case 'beamAreaMul': return Array.isArray(h.beamKeys) && h.beamKeys.length > 0;
    case 'areaMul': return Array.isArray(h.areaKeys) && h.areaKeys.length > 0;
    case 'areaDmgMul': return h.dmgStat === 'areaDmgMul';
    case 'orbitMul': return (Array.isArray(h.orbitKeys) && h.orbitKeys.length > 0) || h.dmgStat === 'orbitMul';
    default: return true;
  }
}
const MOVE_PATTERNS = ['sway', 'orbitArc', 'holdCenter'];                                                        // §8.12.1 (3)
const BULLET_SHAPES = ['circle', 'hex'];                                                                         // §9.7 (2)
const BULLET_STATUS = [null, 'slow', 'stun'];                                                                    // §9.7
const SPAWN_EDGES = ['top', 'left', 'right', 'bottom'];                                                          // §8.7
const BOSS_TIERS = ['stage', 'mid', 'final'];                                                                    // §9.8
const RNG_STREAMS = ['theme', 'draft', 'spawn', 'elite', 'drop', 'pattern', 'boss', 'bot', 'terrain'];           // §10.2 (9 — v1.10 ⑦ terrain)
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
  aura:      ['radius'],
  boomerang: ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'],
  barrage:   ['dmg', 'cooldownSec', 'targetMode'],
  drone:     ['dmg', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'],
  nova:      ['dmg'],
  missile:   ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'],
  chain:     ['dmg', 'cooldownSec', 'count', 'hitCooldownSec', 'targetMode'],
  beam:      ['dmg', 'count', 'pierce', 'hitCooldownSec', 'targetMode'],
  pinball:   ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'],
  spiral:    ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode'],
};
// §9.5 고유 파라미터 — base 거처 (evo* 아닌 것)
const FAMILY_OWN_BASE = {
  forward:   ['spreadDeg', 'jitterDeg', 'burstCount', 'burstIntervalSec'],
  fan:       ['arcDeg'],
  seeker:    ['turnRateDegSec', 'acquireRadius', 'retargetSec'],
  lance:     ['beamWidthPx', 'chargeSec', 'rangePx'],
  orbit:     ['orbitRadius', 'angularSpeedDegSec', 'bodyCount'],
  aura:      ['slowMul'],
  boomerang: ['outRangePx', 'returnSpeed', 'canRehit', 'bounceLeft', 'spacingDeg', 'sweepDegSec'],
  barrage:   ['strikeIntervalSec', 'strikesPerVolley', 'blastRadius', 'telegraphSec', 'slowSec', 'impactFlashSec'],
  drone:     ['droneCount', 'anchorOffsets', 'droneFireSec', 'droneRangePx'],
  nova:      ['intervalSec', 'radius', 'expandSec', 'telegraphSec', 'actionSlowSec'],
  missile:   ['blastRadius', 'spreadDeg'],
  chain:     ['acquireRadius', 'chainRangePx', 'chainCount', 'chainDmgMul'],
  beam:      ['rangePx', 'beamWidthPx'],
  pinball:   ['bounceLeft', 'launchDeg'],
  spiral:    ['ampPx', 'freqHz'],
};
// §9.5 고유 파라미터 — evolution.params 거처 (evo* 접두)
const FAMILY_OWN_EVO = {
  forward:   ['evoRampSec', 'evoRampFireRateMul'],
  fan:       ['evoBlastRadius', 'evoSecondaryDmgMul'],
  seeker:    ['evoDistinctTargets', 'evoRetargetOnKill'],
  lance:     ['evoFullHeight'],
  orbit:     ['evoBulletClearCooldownSec'],
  aura:      ['evoPullForce'],
  boomerang: ['evoChainCount'],
  barrage:   ['evoRadiusMul'],
  drone:     ['evoTrailDelaySec'],
  nova:      ['evoRing2Radius', 'evoSecondaryDmgMul', 'evoActionSlowSec'],
  missile:   ['evoClusterCount', 'evoClusterDmgMul'],
  chain:     ['evoForkOnSuper', 'evoChainCountMul'],
  beam:      ['evoSplitCount', 'evoSplitDmgMul', 'evoSplitRangePx'],
  pinball:   ['evoSplitOnBounce', 'evoMaxBalls'],
  spiral:    ['evoAmpMul', 'evoLifetimeMul'],
};
// §9.5 허용 targetMode (패밀리별). null = targetMode 키 자체가 없다
const FAMILY_TARGET_MODES = {
  forward: ['forward'], fan: ['forward'], seeker: ['nearest', 'lowestHp', 'randomInArena'],
  lance: ['forward', 'nearest'], orbit: null, aura: null,
  boomerang: ['forward', 'sweep'], barrage: ['randomInArena', 'densest'],   // ㉚ 리턴 sweep(회전 조준) · nearest 는 구현이 없어 어휘에서 뺐다
  missile: ['forward'], chain: ['nearest'], beam: ['nearest'], pinball: ['forward'], spiral: ['forward'],   // ㉟
  drone: ['nearest', 'lowestHp', 'forward'], nova: null,
};

// §7.4 텔레그래프 하한 — 3축 (거동별 표 · 탄 상태 · 개체 클래스). ★ 겹치면 max
const TELEGRAPH_FLOOR_BY_TYPE = {              // §7.4 · §8.5 거동별 표
  straight: 0.55, fan: 0.60, aimed: 0.60, ring: 0.60,
  spiral: 0.60, wall: 0.80, zone: 0.90, laser: 1.20, mortar: 0.60, sweep: 1.20,
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
    //   ★ 원문(raw)을 통짜로 훑던 옛 스캐너는 (a) 문자열 리터럴 'from'(예: schema.mjs
    //     EMIT_COMMON 의 이미터 발사원점 키)·(b) 주석에 인쇄된 import 예시를 core 밖
    //     import 로 오인해 정상 코드에서 위반을 낼 수 있는 오탐이었다.
    //   → import/export **문**에 앵커한다: 주석만 지운(문자열은 보존해 스펙시파이어를
    //     살린) 원문에서, from 앞에 import/export 키워드가 따옴표·세미콜론을 건너뛰지
    //     않고 실재할 때만 매칭한다. 실제 import(명명·기본·다중행·재export)는 전부 계속
    //     검사되고, 문자열 키 'from' 은 앞에 키워드가 없어 걸리지 않는다.
    //   ★ 이 수정으로 schema.mjs 의 `from` 백틱 우회(§8.5 소유 키)가 불필요해진다.
    const codeForImports = raw
      .replace(/\/\*[\s\S]*?\*\//g, ' ')
      .replace(/(^|[^:])\/\/[^\n]*/g, '$1 ');
    for (const m of codeForImports.matchAll(/\b(?:import|export)\b[^;'"]*?\bfrom\s*(['"])([^'"]+)\1/g)) {
      const spec = m[2];
      const target = spec.startsWith('.') ? resolve(dirname(f), spec) : spec;
      const inCore = spec.startsWith('.') && !relative(coreDir, target).startsWith('..');
      if (!inCore) V('S1', `${rel}: core 밖 import "${spec}" — src/core/** 는 src/core/** 외 import 금지 (§9.1)`);
    }

    // (3) src/core/weapons/** 숫자 리터럴 — 0, 1, -1, 0.5, 2 만
    //   ★ 식별자 경계 추가: (?<![\w$.]) 로 식별자 속 숫자(스크래치 슬롯 a3 의 3,
    //     evo* 키 등)를 리터럴로 오인하지 않는다. §9.1 은 튜닝 리터럴을 막는 것이지
    //     식별자 문자가 아니다. 실제 리터럴(3, 180 등)은 계속 잡힌다.
    if (!relative(join(coreDir, 'weapons'), f).startsWith('..')) {
      for (const m of code.matchAll(/(?<![\w$.])-?\d+(?:\.\d+)?\b/g)) {
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
  // ★ 코어-디렉터리 SKIP 과 대칭: weapons/ 디렉터리가 있으나 .js 가 0개면 리터럴
  //   게이트가 인증 대상 0으로 조용히 통과한다 → 명시적으로 SKIP 발화한다.
  const weaponsDir = join(coreDir, 'weapons');
  if (existsSync(weaponsDir)
    && !files.some((f) => !relative(weaponsDir, f).startsWith('..'))) {
    SKIP('S1', 'src/core/weapons/ 디렉터리는 있으나 .js 파일이 0개 — 리터럴 게이트가 인증할 무기 모듈이 없다');
  }
}

// ===========================================================================
//  S2 — 스키마 (§9.3 · §9.4~§9.9)
//  타입 · 필수 키 · 미지 키 거부 · 참조 무결성
//  ★ rules.json 루트 키 = 17개 목록 (§9.4 — v1.5에서 bomb 제거 = 경제·소비아이템 폐지 · v1.10 ⑦ terrain 추가)
// ===========================================================================
const RULES_ROOT_17 = ['loop', 'view', 'collide', 'caps', 'player', 'status', 'elite',
  'boss', 'fairness', 'terrain', 'hud', 'passiveHooks', 'input', 'palette', 'visual', 'render', 'audio'];   // v1.10 ⑦ terrain
const TERRAIN_KINDS = ['slow', 'inertia', 'heat'];
const TERRAIN_KIND_ELEMENT = { slow: 'grass', inertia: 'water', heat: 'fire' };   // §8.21 ② (schema.mjs 와 같은 사전 — check 는 독립 사본)
const TERRAIN_MIXED = 'mixed';                                                     // §8.21 ③ finale 순환
const TRAIT_EFFECT_KINDS = ['regenHpPerSec', 'lifestealPct', 'shieldEverySec'];   // §11.6 ㉒ (schema.mjs 와 같은 어휘 — check 는 독립 사본)
const SECTIONS = ['early', 'midboss', 'crisis', 'boss'];   // §8.19 구간 어휘

function S2_schema() {
  const r = D.rules;

  // --- rules.json 루트 = schemaVersion + 정확히 16 블록 (§9.4) --------------
  closedKeys('S2', r, ['schemaVersion', ...RULES_ROOT_17], 'rules');
  if (RULES_ROOT_17.length !== 17) C('S2', `내부 오류: 루트 목록이 ${RULES_ROOT_17.length}개 (정본은 17 — v1.10 ⑦ terrain)`);

  closedKeys('S2', r.loop, ['tickHz', 'maxStepsPerFrame', 'maxFrameGapMs', 'interpolate'], 'rules.loop');
  closedKeys('S2', r.view, ['logicalW', 'logicalH', 'arena', 'panelLeftW', 'panelRightW', 'bandTopH',
    'bandHpH', 'bandXpH', 'playerBoundsInset', 'spawnLineY', 'spawnPadPx', 'minViewportW', 'minViewportH', 'maxDpr'], 'rules.view');
  if (isObj(r.view)) {
    closedKeys('S2', r.view.arena, ['x', 'y', 'w', 'h'], 'rules.view.arena');
    closedKeys('S2', r.view.playerBoundsInset, ['top', 'bottom', 'left', 'right'], 'rules.view.playerBoundsInset');
  }
  closedKeys('S2', r.collide, ['gridCellPx'], 'rules.collide');

  closedKeys('S2', r.caps, ['playerBullets', 'enemyBullets', 'enemies', 'pickups', 'zones', 'drones',
    'particles', 'telegraphs', 'damageNumbers', 'effectMarkers', 'terrain', 'overflow'], 'rules.caps');
  if (isObj(r.caps)) {
    closedKeys('S2', r.caps.overflow, ['playerBullet', 'enemyBullet', 'enemy', 'pickup', 'zone',
      'drone', 'telegraph', 'particle', 'damageNumber', 'effectMarker', 'terrain'], 'rules.caps.overflow');
    // §12.1 "모든 캡에 정책이 있다" — 10 캡 ⟺ 10 정책
    const capNames = Object.keys(r.caps).filter((k) => k !== 'overflow');
    if (isObj(r.caps.overflow) && capNames.length !== Object.keys(r.caps.overflow).length) {
      V('S2', `rules.caps: 캡 ${capNames.length}개 ↔ overflow 정책 ${Object.keys(r.caps.overflow).length}개 — §12.1 "모든 캡에 정책이 있다"`);
    }
  }

  // ★ v1.3: player.hpSegment 삭제 (§23.3 — hud.hpBarSegCount 가 칸을 소유한다)
  // ★ §2.1 healPickupPct(회복 드랍량의 유일한 거처) — data/rules.json + src/core(killEnemy) +
  //   이 허용목록이 동시 착지하는 교차그룹 계약. data 에 0.35 로 착지됐으므로 required
  //   (누락=에러, §9.3 폴백 금지)로 잠근다 — 향후 실수로 빠지면 게이트가 짖는다.
  closedKeys('S2', r.player, ['hpMax', 'spriteRadius', 'hitboxRadius', 'moveSpeed',
    'moveResponseTau', 'diagonalNormalize', 'iframeSec', 'defenseBase', 'damageFloorRatio',
    'lowHpThreshold', 'lowHpCriticalThreshold', 'magnetRadius', 'xpDriftPxSec', 'elementSlots',
    'startStance', 'stanceSwitchCooldown', 'stancePersistAcrossStages', 'elementCapPerElement',
    'elementCapTotal', 'weaponSlots', 'passiveSlots', 'lives'], 'rules.player');
  if (has(r.player, 'hpSegment')) {
    V('S2', 'rules.player.hpSegment: 삭제된 키 (§23.3) — 칸당 = hpMax / hud.hpBarSegCount (§2.1)');
  }

  closedKeys('S2', r.status, ['slowMoveSpeedMul', 'actionSlowMul', 'stackMode', 'resistAffects'], 'rules.status');
  closedKeys('S2', r.elite, ['perWaveMax', 'hpMult', 'sizeMult', 'contactDmgMul', 'xpMult',
    'bandAllowed', 'elementAllowed'], 'rules.elite');

  // §9.4 인쇄 블록이 boss 스코프의 필드 집합을 확정한다 (C-7)
  closedKeys('S2', r.boss, ['partCount', 'partRegen', 'partHitPriority',
    'phaseThresholds', 'phaseTransitionSec', 'timerPausesOnPhaseTransition', 'introSec', 'entryWipeSec',
    'timerStartsAfterIntro', 'timerExpire', 'mobilityPenalty', 'partXpRatio', 'escalateFireRateMul', 'escalateFireRateMax', 'coreElement', 'coreEmitterId',
    'partNormalForbidden', 'partElementDistinctMin', 'partThemeElementMax', 'armorElementNotTheme',
    'armorPartCountRange', 'armorCoreRatioMax', 'optionalPartArmorRatio', 'partReachMinPx',
    'midBossSummonsAllowed', 'bossSummonsAllowed', 'finale'], 'rules.boss');
  if (isObj(r.boss)) {
    // ★ v1.3: finale.armorCoreRatio 삭제 — 유일 소유자 = bosses[].armorCoreRatio (§23.3)
    closedKeys('S2', r.boss.finale, ['partCount', 'armorPartCount', 'exemptRules', 'allowNormalPeripheral'],
      'rules.boss.finale');
    if (has(r.boss.finale, 'armorCoreRatio')) {
      V('S2', 'rules.boss.finale.armorCoreRatio: 삭제된 키 (§23.3) — φ 는 보스 개체의 속성이다. 유일 소유자 = bosses[tetrarch].armorCoreRatio (§9.8)');
    }
  }

  // ★ v1.3: statusBulletSpeedMul 이 visual → fairness 로 이사했다 (§23.3 · §12.4)
  // §8.21(v1.10 ⑦) 지형 장판
  if (isObj(D.stages && D.stages.phase) && isObj(D.stages.phase.sectionSpeedMul)) closedKeys('S2', D.stages.phase.sectionSpeedMul, ['early', 'mid', 'crisis'], 'stages.phase.sectionSpeedMul');   // v1.10 ⑪ drain
  closedKeys('S2', r.terrain, ['radiusPx', 'scrollSpeedPx', 'everySec', 'maxOnScreen', 'spawnIn', 'bossEntryCount', 'fadeSec', 'inertia', 'heat'], 'rules.terrain');
  if (isObj(r.terrain)) {
    closedKeys('S2', r.terrain.inertia, ['responseTauSec'], 'rules.terrain.inertia');
    closedKeys('S2', r.terrain.heat, ['fullSec', 'stallSec', 'coolSec'], 'rules.terrain.heat');
  }
  closedKeys('S2', r.fairness, ['minTelegraphSec', 'beamLockSec', 'beamBlockRadiusPx', 'beamBlockRatio', 'minStunTelegraphSec', 'maxStunSec', 'maxBulletSpeed',
    'maxAimedBulletSpeed', 'statusBulletSpeedMul', 'minBulletRadiusPx', 'minGapWidthPx', 'minSpawnRadiusPx',
    'maxSimultaneousEnemyBullets', 'maxBulletAgeSec', 'enemyConcurrentMax', 'introConcurrentMax', 'swarmConcurrentMax',
    'telegraphConcurrentMaxPerEntity', 'telegraphConcurrentMaxGlobal', 'playerWeaponsExempt'], 'rules.fairness');

  // ★ v1.5: hud.icons 14 → 3 (§9.4.1 — 상점·소비아이템 폐지로 살아있는 어휘 = xp + 상태이상 2종)
  closedKeys('S2', r.hud, ['hitboxAlwaysVisible', 'showElementBudget', 'fontHeroPx', 'fontLargePx',
    'fontMediumPx', 'fontBodyPx', 'fontSmallPx', 'panelPadPx', 'keycapBoxPx', 'bossHpBarH',
    'hpBarSegGapPx', 'xpBarH', 'hpBarSegCount', 'panelCacheDirtyOnly', 'parGhostEnabled',
    'elementMatrixInPanel', 'noHitIndicator',
    'stanceHintTargetsMajorityElement', 'icons'], 'rules.hud');
  if (isObj(r.hud) && Array.isArray(r.hud.icons) && r.hud.icons.length !== 3) {
    V('S2', `rules.hud.icons: ${r.hud.icons.length}종 ≠ 3 (§9.4.1 — v1.5 경제 폐지 후 xpDiamond + statusSlow + statusStun)`);
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
      closedKeys('S2', r.passiveHooks[f], HOOK_KEYS, `rules.passiveHooks.${f}`);
      if (has(r.passiveHooks[f], 'dmgStat') && r.passiveHooks[f].dmgStat !== null && !DMG_STATS.includes(r.passiveHooks[f].dmgStat)) {
        V('S2', `rules.passiveHooks.${f}.dmgStat = ${JSON.stringify(r.passiveHooks[f].dmgStat)} — 어휘 = null | ${DMG_STATS.join(' | ')} (§9.6.1 ㊲)`);
      }
      if (has(r.passiveHooks[f], 'pierce')) {
        V('S2', `rules.passiveHooks.${f}.pierce: 개명된 키 → pierceApplies (§9.6.1/§23.3) — 무기 파라미터 pierce(정수)와 이름이 충돌했다`);
      }
    }
  }

  closedKeys('S2', r.input, ['layout', 'socd', 'pauseOnBlur', 'bindings'], 'rules.input');
  if (isObj(r.input)) {
    closedKeys('S2', r.input.bindings, ['move', 'stanceNormal', 'stanceFire', 'stanceWater', 'stanceGrass',
      'pause', 'options', 'draftPick', 'reorderToggle', 'grab',
      'confirm', 'mute', 'cursor'], 'rules.input.bindings');
  }

  closedKeys('S2', r.palette, ['element', 'elementCvd', 'threat', 'status', 'pickup', 'enemyBody',
    'partDestroyed', 'neutralGray', 'hud', 'bg'], 'rules.palette');
  if (isObj(r.palette)) {
    closedKeys('S2', r.palette.element, ELEMENTS4, 'rules.palette.element');
    closedKeys('S2', r.palette.elementCvd, ELEMENTS4, 'rules.palette.elementCvd');
    closedKeys('S2', r.palette.threat, ['enemyBullet', 'telegraph', 'bulletCore', 'outline'], 'rules.palette.threat');
    closedKeys('S2', r.palette.status, ['band'], 'rules.palette.status');
    closedKeys('S2', r.palette.pickup, ['xp', 'trait'], 'rules.palette.pickup');   // v1.10 ⑲ 특성 구슬
    closedKeys('S2', r.palette.hud, ['panelBg', 'panelRule', 'textPrimary', 'textDim', 'hpFill', 'accent'], 'rules.palette.hud');
    closedKeys('S2', r.palette.bg, ['maxSaturation', 'maxLightness', 'cvdMaxLightness',
      'parallaxLayers', 'maxScrollSpeed'], 'rules.palette.bg');
  }

  // §9.4.3 — visual 전 키 인쇄. ★ v1.3: statusBulletSpeedMul 이 빠졌다(→ fairness)
  closedKeys('S2', r.visual, ['iframeBlinkHz', 'hpBar', 'stance', 'playerBullet',
    'glyph', 'telegraph', 'band', 'zone', 'terrain', 'wipe', 'timer', 'trail', 'hitFx', 'a11y', 'text'], 'rules.visual');
  if (isObj(r.visual)) closedKeys('S2', r.visual.terrain, ['fillAlpha', 'patternAlpha', 'iconAlpha', 'iconPx', 'heatPulseHz', 'heatWarnAt'], 'rules.visual.terrain');   // §7.13(v1.10 ⑦)
  if (isObj(r.visual)) closedKeys('S2', r.visual.wipe, ['bandPx', 'flashAlpha'], 'rules.visual.wipe');                                        // §8.22(v1.10 ⑧)
  if (has(r.visual, 'statusBulletSpeedMul')) {
    V('S2', 'rules.visual.statusBulletSpeedMul: 이사한 키 → rules.fairness.statusBulletSpeedMul (§23.3) — visual 키가 게임플레이 속도를 바꾸면 §9.4.3의 경계가 깨진다');
  }
  if (isObj(r.visual)) {
    closedKeys('S2', r.visual.hpBar, ['hPx', 'wPx', 'gapPx', 'trackAlpha', 'gatePostWPx',
      'gatePostOverhangPx'], 'rules.visual.hpBar');
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
    'playerBulletMaxRadiusPx', 'playerBulletDensityRef', 'playerBulletMinAlpha', 'particleMaxAlpha', 'particleMaxLifeSec', 'fxMinRealMs', 'targetFps',
    'degradeOnFrameMs', 'degradeRecoverFrames'], 'rules.render');

  // ★ v1.5: busGain.bgm 부활 (§7.10 BGM — 절차적 신스로 재도입). audio.bgm(데이터 40값 스펙)은
  //   여전히 스코프아웃: BGM 은 코드 신스(SFX 신스와 동류)라 데이터 표가 불필요하고, 게인만 busGain.bgm.
  closedKeys('S2', r.audio, ['busGain', 'cueRateLimitPerSec'], 'rules.audio');
  if (has(r.audio, 'bgm')) {
    V('S2', 'rules.audio.bgm: 데이터 40값 스펙은 스코프아웃 — BGM 은 코드 신스다(§7.10 v1.5). 게인만 busGain.bgm');
  }
  if (isObj(r.audio)) {
    closedKeys('S2', r.audio.busGain, ['sfx', 'bgm'], 'rules.audio.busGain');
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
    // §9.6 "폐쇄 스탯 어휘 11종, 11 패시브와 1:1" (v1.5 salvage 제거)
    const stats = D.passives.passives.map((p) => p && p.stat);
    if (new Set(stats).size !== stats.length) V('S2', 'passives: stat 중복 — §9.6 "11훅 = 11 패시브 1:1"');
    if (D.passives.passives.length !== PASSIVE_STATS.length) V('S2', `passives: ${D.passives.passives.length}종 ≠ ${PASSIVE_STATS.length} (§9.6 ㊲)`);
  }

  // --- bullets.json (§9.7) — ★ v1.3: speed 삭제 (탄 속도는 이미터가 소유) ---
  closedKeys('S2', D.bullets, ['schemaVersion', 'bullets'], 'bullets');
  for (const b of rowsQuiet(D.bullets.bullets)) {
    if (!isObj(b)) continue;
    closedKeys('S2', b, ['id', 'radius', 'hitboxScale', 'dmg', 'shape', 'status',
      'statusDurationSec', 'accel', 'turnRateDegSec', 'retargetSec', 'waveAmp', 'waveHz',
      'bounceLeft', 'homingSec'], `bullets[${b.id}]`, { optional: ['bounceLeft', 'homingSec'] });   // §8.5(v1.7)
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
        ? ['hpMult', 'xpRef', 'minPerWave']
        : ['hpMult', 'minPerWave'];
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
    // §8.17(v1.7) 적 개성 3종은 «선택 키»다 (§8.11 sealLayer 와 같은 규약) — 미선언 = 기본값 = 현행 동작.
    closedKeys('S2', a, ['id', 'name', 'desc', 'band', 'shapeId', 'radius', 'moveId', 'moveParams',
      'attack', 'contactDmg', 'hp', 'xp', 'score', 'themeOnly',
      'hitFloorSec', 'pierceCost', 'ccImmune'], `enemies.archetypes[${a.id}]`,
      { optional: ['hitFloorSec', 'pierceCost', 'ccImmune'] });
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
    mortar: ['radius', 'activeSec', 'dmg', 'fuseSec', 'leadSec'],
    sweep: ['widthPx', 'activeSec', 'angleStartDeg', 'angleEndDeg'],
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
    closedKeys('S2', w, ['id', 'family', 'name', 'desc', 'elementStampMode', 'slotClass', 'class', 'base', 'levels', 'evolution'],
      `weapons[${w.id}]`);
    if (isObj(w.evolution)) {
      closedKeys('S2', w.evolution, ['name', 'desc', 'params', 'requiresPassive'], `weapons[${w.id}].evolution`);
      // §9.5 v1.5 — 진화 짝 패시브 (뱀서식). 의미 검증은 S41.
      if (isObj(w.evolution.requiresPassive)) {
        closedKeys('S2', w.evolution.requiresPassive, ['id', 'level'], `weapons[${w.id}].evolution.requiresPassive`);
      }
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
        'xp', 'score'], `bosses[${b.id}]`);
      if (has(b, 'core')) {
        V('S2', `bosses[${b.id}].core: 중간보스에는 core 가 없다 (§9.8.2-ⓐ) — parts: [] 이므로 "부위와 구별되는 몸통"이 정의되지 않는다`);
      }
      // §8.9 v1.2 정정: bosses[].leaveAfterSec 는 삭제되었다
      if (has(b, 'leaveAfterSec')) {
        V('S2', `bosses[${b.id}].leaveAfterSec: 삭제된 키 — 타이머 이탈은 폐지, 이탈은 위기(midBossForcedLeaveOnCrisis)만이 부른다 (§8.9 v1.10)`);
      }
    } else {
      // §9.8 — 스테이지·최종 보스 팔
      closedKeys('S2', b, ['id', 'name', 'tier', 'themeId', 'armorCoreRatio', 'core', 'parts',
        'movePattern', 'movePatternParams', 'summon'], `bosses[${b.id}]`);
      // §9.8: xp 는 tier == "mid" 전용 필드다 (v1.3)
      if (has(b, 'xp')) {
        V('S2', `bosses[${b.id}].xp: tier=="mid" 전용 필드다 (§9.8/§23.3) — 스테이지·최종 보스의 보상은 core.score/parts[].score 가 소유한다 (v1.5: 코인 폐지)`);
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
          'contactDmg', 'shapeId', 'score', 'patternSet', 'extra', 'sealLayer'], `bosses[${b.id}].parts[${p.id}]`,
          { optional: ['extra', 'sealLayer'] });
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
      closedKeys('S2', b.summon, ['archetypeId', 'count', 'everySec', 'formationId', 'ghost'],
        `bosses[${b.id}].summon`, { optional: ['ghost'] });
    }
    for (const ps of rowsQuiet(b.patternSet)) {
      closedKeys('S2', ps, ['emitterIds'], `bosses[${b.id}].patternSet[]`);
    }
  }

  // --- stages.json (§9.9) — ★ v1.3: themes → stages 개명 -------------------
  closedKeys('S2', D.stages, ['schemaVersion', 'theme', 'themeDraw', 'curve', 'phase', 'stages', 'formations'], 'stages');
  if (has(D.stages, 'themes')) {
    V('S2', 'stages.themes: 개명된 키 → stages.stages (§9.9/§23.3) — 파일 이름이 stages.json 이고 게이트가 stages[] 라 부른다');
  }
  closedKeys('S2', D.stages.themeDraw, ['pool', 'count', 'allowRepeat', 'stage1RequiresIntroOk', 'finalStageId'], 'stages.themeDraw');
  closedKeys('S2', D.stages.curve, ['enemyHpScale', 'xpScale', 'bossHpScale', 'midBossHpScale', 'bossBulletScale', 'firingPartsPerStage',
    'spawnDensityScale', 'mobFireRateScale', 'mobBulletDmgScale', 'midBossCount', 'elitePerWaveChance', 'swarmTotalScale', 'crisisHpScale', 'rearSpawnAllowed',
    'shooterRatio', 'threatBudgetScale'], 'stages.curve');
  // §9.9 v1.3: crisisPerStage · crisisWaves · midBossAtSec 신설 / bossEntrySec · crisisElementRule 삭제
  closedKeys('S2', D.stages.phase, ['mobPhaseSec', 'mobPhaseSkippable', 'mobPhaseMaxWaves', 'waveIntervalSec',
    'waveClearAdvance', 'phaseEndAutocollect',
    'enemyExitForfeitsReward', 'waveListExhausted', 'crisisPerStage', 'crisisStartSec', 'crisisCycleSec', 'crisisSwarmLoop', 'crisisShooterId',
    'crisisSuspendsWaves', 'crisisOnMidBossClear', 'crisisTotal', 'crisisSubWaves', 'crisisWaves',
    'introFormationId', 'sectionSpeedMul', 'earlyWaveIntervalSec', 'earlyDrainSec', 'midBossSuspendsWaves',
    'midBossAtSec', 'midBossFirstId', 'midBossElementRule', 'midBossForcedLeaveOnCrisis',
    'bossTimerSec', 'timerWarnSec', 'timerRedAlertSec', 'statusStunMaxPerStage'], 'stages.phase');
  if (has(D.stages.phase, 'bossEntrySec')) {
    V('S2', 'stages.phase.bossEntrySec: 삭제된 키 (§9.9/§23.3) — 유일 소유자 = rules.boss.introSec (§6.3)');
  }
  if (has(D.stages.phase, 'crisisElementRule')) {
    V('S2', 'stages.phase.crisisElementRule: 이사한 키 → stages[].crisisElementRule (2값 어휘, §23.3)');
  }
  // §9.9.3: crisisSubWaveIntervalSec 은 파생값이지 키가 아니다 (새 키 0)
  if (has(D.stages.phase, 'crisisSubWaveIntervalSec')) {
    V('S2', 'stages.phase.crisisSubWaveIntervalSec: 파생값이지 키가 아니다 — §9.9.3 (= crisisCycleSec / crisisSubWaves)');
  }
  for (const cw of rowsQuiet(D.stages.phase && D.stages.phase.crisisWaves)) {
    closedKeys('S2', cw, ['subWave', 'formationId', 'bodyId', 'count', 'spawnEdge'], 'stages.phase.crisisWaves[]');   // v1.10 ⑫ bodyId
  }
  // §9.9.2 formations — 6종 + 파라미터
  closedKeys('S2', D.stages.formations, FORMATION_IDS, 'stages.formations');
  const FORM_PARAMS = {
    lineH: ['gapPx'], columnV: ['gapSec'], vWedge: ['gapPx', 'angleDeg'],
    arc: ['radiusPx', 'spanDeg', 'flatten', 'minSepPx'], pincer: ['yStartPx', 'yStepPx'], scatter: ['jitterPx', 'minSepPx'],
    wall: ['gapPx', 'rowGapPx', 'perRow', 'laneSlots', 'laneStrideCols', 'jitterY'],
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
      'introArchetypeId', 'terrainKind', 'roster', 'mix', 'waves'], `stages.stages[${t.id}]`);
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
      closedKeys('S2', w, ['formationId', 'archetypeId', 'count', 'spawnEdge', 'eliteIndex',
        'unlockStageMin'], `stages.stages[${t.id}].waves[${i}]`);
      // §8.7: atSec 절대 타임라인은 폐기 — 순서 리스트다
      if (has(w, 'atSec')) {
        V('S2', `stages.stages[${t.id}].waves[${i}].atSec: 폐기된 키 — 웨이브는 순서 리스트다 (§8.7)`);
      }
    });
  }

  // --- meta.json (§9.9 · §11 · §10.4 · §13.1) -----------------------------
  closedKeys('S2', D.meta, ['schemaVersion', 'xp', 'draft', 'score', 'flow', 'onboarding',
    'difficulty', 'bot', 'certify'], 'meta');
  closedKeys('S2', D.meta.xp, ['curve', 'base', 'exp', 'levelUpsPerRunTarget', 'levelUpQueueMode'], 'meta.xp');
  if (isObj(D.meta.draft)) {
    closedKeys('S2', D.meta.draft, ['optionCount', 'slotAssign', 'categoryWeights', 'newWeaponSlotScale',
      'weaponLevelEvolutionBonus', 'elementFirstLevelBonus', 'passiveNewBonus', 'distinctItemsPerDraft',
      'filterInvalid', 'newWeaponWhenSlotsFull', 'elementLevelOfferRequiresWeaponCount',
      'guaranteeElementCardOnFirstDraft', 'guaranteeNewWeaponUntilSlots', 'elementCardPity',
      'fallback', 'pauseGame'], 'meta.draft');
    closedKeys('S2', D.meta.draft.categoryWeights, ['newWeapon', 'weaponLevel', 'elementLevel', 'passive'], 'meta.draft.categoryWeights');
    closedKeys('S2', D.meta.draft.fallback, ['id', 'name', 'healPct'], 'meta.draft.fallback');
  }
  // §11.3 — 점수. ★ v1.3: difficultyMul → difficulty[].scoreMul (거처는 meta.difficulty)
  closedKeys('S2', D.meta.score, ['superEffectiveDamageShare', 'superEffectiveKillBonusRatio', 'attribution',
    'timeBonusPerGameSec', 'bossClearBonus', 'midBossClearBonus', 'runClearBonus', 'noHitScope',
    'stageNoHitBonus', 'perfectScope', 'perfectBonus', 'roundMode'], 'meta.score');
  if (has(D.meta.score, 'difficultyMul')) {
    V('S2', 'meta.score.difficultyMul: 개명된 키 → meta.difficulty[].scoreMul (§23.5-05)');
  }
  closedKeys('S2', D.meta.onboarding, ['autoEquipFirstElement', 'stanceHintPulse', 'stanceHintPulseStageMax'], 'meta.onboarding');
  if (isObj(D.meta.flow)) {
    closedKeys('S2', D.meta.flow, ['themeBannerSec', 'stageClearSec', 'healSec', 'stageClearHealPct',
      'pauseResumeCountdownSec', 'attractIdleSec', 'menuSpeed', 'deathAnimSec',
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
      'dodgePerceptionMs', 'aimErrorPx', 'slotOrder', 'policies', 'baseline', 'probes'], 'meta.bot');
    closedKeys('S2', D.meta.bot.policies, ['draft', 'farm', 'stance'], 'meta.bot.policies');
    closedKeys('S2', D.meta.bot.baseline, ['draft', 'farm', 'stance'], 'meta.bot.baseline');
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
    need(archIds, t.introArchetypeId, `stages.stages[${t.id}].introArchetypeId`);   // §8.19
    for (const r of rowsQuiet(t.roster)) need(archIds, r && r.archetypeId, `stages.stages[${t.id}].roster.archetypeId`);
    rowsQuiet(t.waves).forEach((w, i) => {
      need(archIds, w && w.archetypeId, `stages.stages[${t.id}].waves[${i}].archetypeId`);
      need(formIds, w && w.formationId, `stages.stages[${t.id}].waves[${i}].formationId`);
    });
  }
  for (const cw of rowsQuiet(D.stages.phase && D.stages.phase.crisisWaves)) {
    need(formIds, cw && cw.formationId, 'stages.phase.crisisWaves[].formationId');
  }
  for (const cw of rowsQuiet(D.stages.phase && D.stages.phase.crisisWaves)) need(archIds, cw && cw.bodyId, 'stages.phase.crisisWaves[].bodyId');   // v1.10 ⑫
  need(archIds, D.stages.phase && D.stages.phase.crisisShooterId, 'stages.phase.crisisShooterId');
  need(emitIds, D.rules.boss && D.rules.boss.coreEmitterId, 'rules.boss.coreEmitterId');   // §9.8.1 v1.5
  for (const id of rowsQuiet(D.rules.boss && D.rules.boss.midBossSummonsAllowed)) {
    need(bossIds, id, 'rules.boss.midBossSummonsAllowed');
  }
  for (const id of rowsQuiet(D.rules.boss && D.rules.boss.bossSummonsAllowed)) {
    need(bossIds, id, 'rules.boss.bossSummonsAllowed');
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
      ['보상 2필드 보유', ['xp', 'score'].every((k) => has(b, k))],
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
    });
  }
  for (const cw of rowsQuiet(D.stages.phase && D.stages.phase.crisisWaves)) {
    if (!isObj(cw)) continue;
    vocab('S3', cw.formationId, FORMATION_IDS, 'stages.phase.crisisWaves[].formationId');
    vocab('S3', cw.spawnEdge, SPAWN_EDGES, 'stages.phase.crisisWaves[].spawnEdge');
  }
  // §9.6 stats 어휘 = 11종 폐쇄 (v1.5 salvage 제거)
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

    // partCount — core 를 포함한다 (§13.6.2 "armor 2 + 선택 1 + core 1 = partCount 4"). ★ v1.10 ⑩: «기본» 부위만 센다 —
    //   extra 부위(선택 무장)는 포지션 곡선 firingPartsPerStage 가 몇 개를 세울지 정한다(§8.9.1). 곡선의 최댓값 − 기본 수 =
    //   그 보스가 «저작해야 하는» extra 수. 더 있으면 영영 안 서는 죽은 부위, 덜 있으면 후반이 약속을 못 지킨다.
    const baseParts = parts.filter((p) => isObj(p) && p.extra !== true);
    const extraParts = parts.filter((p) => isObj(p) && p.extra === true);
    const wantPartCount = isFinal ? (rb.finale && rb.finale.partCount) : rb.partCount;
    if (num(wantPartCount) && baseParts.length + 1 !== wantPartCount) {
      V('S5', `${tag}: partCount = ${baseParts.length + 1}(기본 부위 ${baseParts.length} + core) ≠ ${wantPartCount} (§8.11/§8.16)`);
    }
    const fps = D.stages && D.stages.curve && D.stages.curve.firingPartsPerStage;
    if (Array.isArray(fps) && fps.length === 6 && num(wantPartCount)) {
      let maxFiring = -Infinity;
      if (isFinal) maxFiring = fps[5];
      else for (let i = 0; i < 5; i += 1) if (num(fps[i]) && fps[i] > maxFiring) maxFiring = fps[i];
      const wantExtra = maxFiring - (wantPartCount - 1);
      if (extraParts.length !== wantExtra) {
        V('S5', `${tag}: extra 부위 ${extraParts.length}개 ≠ ${wantExtra} (= ${isFinal ? 'firingPartsPerStage[5]' : 'max(firingPartsPerStage[0..4])'} ${maxFiring} − 기본 ${wantPartCount - 1}) — §8.9.1 «발사 파트 수가 포지션으로 성장한다»의 저작 범위`);
      }
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

    // R8 (v1.10 ㉞): 주변부 속성 ⊆ 그 스테이지의 «테마 + 먹이»(stages[].mix > 0) — 사용자(2026-09-05): 「늪은 풀·물만 나오니
    //   보스의 공격 모듈도 주로 풀·물로」. 최종은 mix 가 3속성이라 자동 충족. 파트 속성이 잡몹 속성 밖이면 스탠스 문법이 보스에서 깨진다.
    {
      const stg = STAGES().find((t) => isObj(t) && t.bossId === b.id);
      if (stg && isObj(stg.mix)) {
        const allowed = Object.keys(stg.mix).filter((el) => num(stg.mix[el]) && stg.mix[el] > 0);
        const bad = periph.filter((p) => isObj(p) && p.element !== 'normal' && !isAmb(p.element) && allowed.indexOf(p.element) < 0);
        if (bad.length > 0) V('S5', `${tag}: R8 위반 — 주변부 [${bad.map((p) => `${p.id}:${p.element}`).join(', ')}] ∉ 스테이지 «테마+먹이» [${allowed.join(', ')}] (§8.14 ㉞)`);
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

    // R7(v1.10 ㉘): φ ∈ (0, boss.armorCoreRatioMax] — 하드 게이트에서 φ 는 «HP 배분»(armor 총합 = 코어 × φ)일 뿐, 소프트 게이트의
    //   산술(B = coreGateMul^-a − 1)은 사라졌다. 상한은 «코어 대비 모듈이 터무니없이 두꺼워 타이머 안에 못 여는» 저작을 막는다.
    //   ★ R7 은 면제하지 않는다 (§8.16). 값이 없거나 상한 키가 없으면 «공허 통과»가 아니라 위반이다.
    const a = armor.length;
    const phi = b.armorCoreRatio;
    if (!num(rb.armorCoreRatioMax) || rb.armorCoreRatioMax <= 0) V('S5', 'rules.boss.armorCoreRatioMax 가 없다 — R7 의 상한 (§8.13.1 ㉘)');
    else if (num(phi)) {
      if (a === 0) V('S5', `${tag}: armor 가 0 인데 φ = ${phi} — 배분할 부위가 없다 (§8.13.1)`);
      else if (!(phi > 0) || phi > rb.armorCoreRatioMax + 1e-9) V('S5', `${tag}: R7 위반 — φ = ${phi} ∉ (0, armorCoreRatioMax ${rb.armorCoreRatioMax}] (§8.13.1 ㉘)`);
    } else if (!isAmb(phi)) {
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
  // ★ v1.5 — 「스테이지 보스는 소환하지 않는다」는 폐기됐다(§8.9-R9 확장: bossSummonsAllowed).
  //   그 규칙을 강제하던 rules.boss.summonsAllowed 는 src/ 어디서도 읽히지 않는 화석이라 삭제했다.
  //   살아있는 규칙은 S17(mid ∨ stage 소환 허용 목록)이 강제한다.
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
  // ★ v1.10 ⑬ — 사용자 결정(2026-09-04): 「한 스테이지에 나오는 속성은 2개로만」. 늪이면 풀·물만 — 화염까지 나오면 스탠스
  //   선택이 어렵다. mix 는 이제 «런타임의 유일한 출처»(스포너의 속성 봉지가 읽는다, §8.2)이고 waves[].element 는 삭제됐다.
  //   규칙: 테마 T 의 mix 는 정확히 두 속성 — T(≥ 0.5) + prey(T)(T 가 이기는 속성), 나머지 0. 노말 0. 최종은 물·불·풀 3속성, 노말 0.
  //   왜 prey 인가: 사용자 예 「늪 = 풀 + 물」(풀이 물을 이긴다). 정답 스탠스(counter) 하나로 다 갈면 재미가 없고, 「풀로 다
  //   쏘다가 물이 없으면 불로」 — 두 스탠스를 오가는 선택이 생긴다(§4.2).
  let cells = 0;
  const FINAL_ID = FINAL();
  for (const t of rows('S8', D.stages.stages, 'stages.stages',
    '§8.2 — 속성 규칙이 0행을 보면 «2속성 한정»을 아무도 검사하지 않는다')) {
    if (!isObj(t) || !isObj(t.mix)) continue;
    const tag = `stages.stages[${t.id}]`;
    closedKeys('S8', t.mix, ELEMENTS4, `${tag}.mix`);
    const sum = Object.values(t.mix).filter(num).reduce((x, y) => x + y, 0);
    if (Math.abs(sum - 1.0) > 1e-6) V('S8', `${tag}.mix: 합 ${sum} ≠ 1.0 (§8.2)`);
    cells += 1;
    if (num(t.mix.normal) && t.mix.normal !== 0) V('S8', `${tag}.mix.normal = ${t.mix.normal} ≠ 0 — 잡몹에 노말은 없다 (§8.2 v1.10 ⑬)`);
    const isFin = t.id === FINAL_ID || t.element === null;
    if (isFin) {
      for (const el of ['water', 'fire', 'grass']) if (!num(t.mix[el]) || t.mix[el] <= 0) V('S8', `${tag}.mix.${el} = ${t.mix[el]} — 최종은 물·불·풀 셋 다 나온다 (§8.16)`);
      continue;
    }
    const T = t.element, p = preyOf(T);
    if (!p) { C('S8', `${tag}: prey 를 elements.matrix 에서 유도할 수 없다 (element=${JSON.stringify(T)})`); continue; }
    const nonZero = ELEMENTS4.filter((el) => num(t.mix[el]) && t.mix[el] > 0);
    if (nonZero.length !== 2 || !nonZero.includes(T) || !nonZero.includes(p)) {
      V('S8', `${tag}.mix: 0 이 아닌 속성 = [${nonZero.join(', ')}] ≠ [테마 ${T}, 먹이 ${p}] — 한 스테이지는 «테마 + 먹이» 2속성 (§8.2 v1.10 ⑬)`);
    }
    if (num(t.mix[T]) && t.mix[T] < 0.5) V('S8', `${tag}.mix.${T} = ${t.mix[T]} < 0.5 — 테마가 다수여야 한다 (§8.2)`);
  }
  // ③ (v1.10 ㉔) 테마 밖 속성은 약하다 — stages.theme.offThemeHpMul ∈ (0, 1). 사용자(2026-09-05): 「풀 몹은 3방, 다른 속성은 2방」.
  //    1 이면 «항만 있고 효과 없음»(죽은 키) → 위반. 하한 0.5: 절반 아래면 먹이 속성이 «있으나 마나»가 돼 2속성 규칙의 뜻이 사라진다.
  cells += 1;
  const th = D.stages.theme;
  if (!isObj(th) || !num(th.offThemeHpMul)) V('S8', 'stages.theme.offThemeHpMul 이 없다 (§8.2 ③)');
  else if (th.offThemeHpMul >= 1 || th.offThemeHpMul < 0.5) V('S8', `stages.theme.offThemeHpMul = ${th.offThemeHpMul} ∉ [0.5, 1) — 1 은 죽은 키, 0.5 아래는 먹이가 유령이 된다 (§8.2 ③)`);
  EX('S8', cells);
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
    if (w.levels.length !== 10) V('S9', `weapons[${w.id}].levels: ${w.levels.length}행 ≠ 10행 (§9.5 v1.10 ⑱ — Lv8 진화 + Lv9·10 강화)`);
  }

  // (3) 4 ≤ elementCapTotal < 3 × elementCapPerElement (§4.2)
  if (isObj(pl) && num(pl.elementCapTotal) && num(pl.elementCapPerElement)) {
    if (!(pl.elementCapTotal >= 4 && pl.elementCapTotal <= 3 * pl.elementCapPerElement)) {
      // v1.10 ⑱ — 사용자 결정 「속성은 모두 다 채울 수 있게」: 합계 상한이 3 × 개별 상한과 «같을 수» 있다(전 속성 만렙 허용). ~~<~~
      V('S9', `player.elementCapTotal(${pl.elementCapTotal}) ∉ [4, 3 × elementCapPerElement(${pl.elementCapPerElement}) = ${3 * pl.elementCapPerElement}] (§4.2)`);
    }
  }

  // (4) §8.4(v1.7) 진입 방향 — 사용자 판정으로 어휘가 바뀌었다:
  //     「적은 위에서 아래로, 혹은 «옆»에서 들어온다. 아래에서 위로 올라오는 적은 없다.
  //      옆 진입은 화면 상단 절반이어야 한다」
  //   근거: 아래에서 올라오는 적은 플레이어의 «시선 뒤»라 예고 없이 닿는다 — 회피 판단의 근거가 없다.
  //   (v1.7 이전의 rearIn / rearSpawnAllowed 게이트를 이것으로 대체했다 — 그 어휘는 폐지됐다)
  // ★ st 는 D.stages 다 — st.rules 는 없다. 이걸 틀리면 arenaH 가 0 이 되어 아래 검사가
  //   통째로 건너뛰어진다(«죽은 게이트»). 실제로 한 번 그렇게 넣었고 주입 테스트가 잡았다.
  const arenaH = (D.rules && D.rules.view && D.rules.view.arena) ? D.rules.view.arena.h : 0;
  if (!(arenaH > 0)) V('S9', 'rules.view.arena.h 를 못 읽었다 — 진입 높이 검사가 죽는다');
  // ★ 위기 웨이브(phase.crisisWaves)도 spawnEdge 를 «선언»한다 — 여기를 빠뜨리면 게이트가
  //   전체의 절반만 본다. (실제로 처음엔 빠뜨렸고, 주입 테스트가 「웨이브 156, 전부 top」이라
  //   말해 줘서 알았다 — 내가 주입한 bottom 이 crisisWaves 쪽이라 보이지 않았다)
  const edgeSources = [];
  for (const t of rowsQuiet(st.stages)) {
    if (!isObj(t)) continue;
    rowsQuiet(t.waves).forEach((w, wi) => edgeSources.push([`stages.stages[${t.id}].waves[${wi}]`, w]));
  }
  rowsQuiet(st.phase && st.phase.crisisWaves).forEach((w, wi) => edgeSources.push([`stages.phase.crisisWaves[${wi}]`, w]));
  for (const [tag, w] of edgeSources) {
    if (!isObj(w)) continue;
    if (w.spawnEdge !== undefined && w.spawnEdge !== 'top') {
      V('S9', `${tag}.spawnEdge = "${w.spawnEdge}" — 진입은 상단(top)뿐이다. 아래에서 올라오는 적은 시선 뒤라 회피 판단의 근거가 없다 (§8.4 v1.7)`);
    }
  }
  for (const a of ARCHETYPES()) {
    if (!isObj(a) || a.moveId !== 'strafe') continue;
    const y = (a.moveParams && a.moveParams.yPx);
    if (typeof y !== 'number') {
      V('S9', `enemies.archetypes[${a.id}]: strafe 는 moveParams.yPx 가 필요하다 — 없으면 스폰 라인에 머물러 화면에 서지 못한다 (§8.4)`);
    } else if (arenaH > 0 && y > arenaH * 0.5) {
      V('S9', `enemies.archetypes[${a.id}].moveParams.yPx(${y}) — 옆 진입은 «화면 상단 절반»(≤ ${arenaH * 0.5})이어야 한다 (§8.4 v1.7)`);
    }
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
  if (swarms.length !== 3) V('S9', `enemies.archetypes: 새떼 전용 아키타입 ${swarms.length}종 ≠ 3 (swarmChaff·swarmDart·swarmLancer — §8.6 v1.10 ⑫)`);

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
//     (v1.10 ⑱: 5 신규 무기 + Σ(무기 maxLevel−1) 54 + elementCapTotal 12 + Σ(패시브 maxLevel) 60 = 131)
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
    const newWeapon = pl.weaponSlots - 1;                    // 시작 무기 1 지급 → 6칸 중 5칸
    const weaponLevel = pl.weaponSlots * 9;                  // Σ(무기 maxLevel−1) = 6무기 × (10−1) (v1.10 ⑱)
    const elementLevel = pl.elementCapTotal;                 // elementCapTotal
    const passive = pl.passiveSlots * D.passives.maxLevel - 1;   // Σ(패시브 maxLevel) = 6칸 × Lv10 − 시작 짝 패시브 1(공짜, ㉚)
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
        V('S11', `${rel}: 미등록 RNG 스트림 "${name}" — 동결 9종 = [${RNG_STREAMS.join(', ')}] (§10.2)`);
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
function maxThreatScale() {
  const c = D.stages && D.stages.curve && D.stages.curve.threatBudgetScale;
  if (!Array.isArray(c) || c.length === 0) return 1;
  let m = 1;
  for (const v of c) if (num(v) && v > m) m = v;
  return m;
}
function maxMidBoss() {
  const mc = D.stages && D.stages.curve && D.stages.curve.midBossCount;
  if (!Array.isArray(mc) || mc.length === 0) return 0;
  let m = 0;
  for (let i = 0; i < mc.length; i += 1) if (num(mc[i]) && mc[i] > m) m = mc[i];
  return m;
}

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
    // v1.10 ⑥ — 위기 중 무대 = 새떼 + 남은 유령(Nest 소환, 자기 몫 enemyConcurrentMax). crisisWaveResidualMax 는 폐지(읽는 곳 0).
    ['enemies-crisis', (f.swarmConcurrentMax || 0) + (f.enemyConcurrentMax || 0),
      `swarmConcurrentMax(${f.swarmConcurrentMax}) + enemyConcurrentMax(${f.enemyConcurrentMax}) 유령`, caps.enemies],
    // §12.1(v1.8) — 잡몹 페이즈의 A층 enemies 합. 웨이브 예산과 유령 예산은 «다른 몫»이고
    //   유령은 새 키 없이 같은 enemyConcurrentMax 를 자기 상한으로 재사용한다(midboss.summon).
    // §12.1(v1.9) — 도입 구간의 몸(introBody)은 «위협» 예산에서 빠지고 자기 몫을 쓴다.
    //   그 몫은 웨이브 예산과 «배타가 아니다»: 벽이 창을 넘어 내려오는 동안 정상 웨이브가 함께 선다.
    //   그래서 A층 합은 max 가 아니라 **덧셈**이고, 이 행이 그 덧셈을 B층 아래로 묶는다.
    // §8.19(v1.10) 웨이브 몫은 포지션 곡선 threatBudgetScale 을 탄다 — 최댓값으로 센다(유령 몫은 배율 없음, midboss.js).
    ['enemies-mobPhase',
      Math.round((f.enemyConcurrentMax || 0) * maxThreatScale()) + (f.enemyConcurrentMax || 0) + (f.introConcurrentMax || 0) + maxMidBoss(),
      `enemyConcurrentMax(${f.enemyConcurrentMax}) × max(threatBudgetScale)(${maxThreatScale()}) 웨이브 + enemyConcurrentMax 유령 + introConcurrentMax(${f.introConcurrentMax}) 도입 + max(midBossCount)(${maxMidBoss()})`,
      caps.enemies],
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
  if (rule !== 'themeElseNonTheme') {
    V('S15', `stages.phase.midBossElementRule = ${JSON.stringify(rule)} ≠ "themeElseNonTheme" (§8.9 v1.5)`);
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
  const midAllowed = new Set(rowsQuiet(D.rules.boss && D.rules.boss.midBossSummonsAllowed));
  const bossAllowed = new Set(rowsQuiet(D.rules.boss && D.rules.boss.bossSummonsAllowed));   // §8.9 v1.5 — 유령 방패
  for (const b of BOSSES()) {
    if (!isObj(b)) continue;
    const lhs = has(b, 'summon') && b.summon !== null && !isAmb(b.summon);
    const rhs = (b.tier === 'mid' && midAllowed.has(b.id)) || (b.tier !== 'mid' && bossAllowed.has(b.id));
    if (lhs !== rhs) {
      V('S17', `bosses[${b.id}]: (summon != null)=${lhs} ≠ (mid ∧ midBossSummonsAllowed) ∨ (stage ∧ bossSummonsAllowed)=${rhs} `
        + `— tier=${b.tier}, mid=[${[...midAllowed].join(', ')}], boss=[${[...bossAllowed].join(', ')}] (§8.9-R9/S17 v1.5)`);
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
    const lhs = e.type === 'zone' || e.type === 'mortar';   // §9.7 v1.5 — mortar 도 탄 없이 dmg 직접
    const rhs = has(e, 'bulletId') && e.bulletId === null;
    if (lhs !== rhs) {
      V('S19', `enemies.emitters[${e.id}]: (type∈{zone,mortar})=${lhs} ≠ (bulletId==null)=${rhs} `
        + `— zone·mortar 은 dmg 를 직접 갖는다 (§9.7/S19)`);
    }
    if (lhs && !num(e.dmg)) V('S19', `enemies.emitters[${e.id}]: ${e.type} 인데 dmg 가 없다 (§3.2 피해원 목록)`);
    // §7.4 v1.5 — mortar 의 «퓨즈»(착탄 후 폭발까지)가 곧 회피 창 → 절대 하한 강제(공정성)
    if (e.type === 'mortar' && num(e.fuseSec) && num(D.rules.fairness.minTelegraphSec)
        && e.fuseSec < D.rules.fairness.minTelegraphSec) {
      V('S19', `enemies.emitters[${e.id}]: mortar fuseSec ${e.fuseSec} < fairness.minTelegraphSec(${D.rules.fairness.minTelegraphSec}) `
        + `— 착탄 후 폭발까지가 회피 창이다 (§7.4)`);
    }
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
    if (!isObj(cw)) continue;
    // v1.10 ⑫ — 서브웨이브 하나에 몸(레코드 bodyId)과 공격형(phase.crisisShooterId)이 봉지로 섞인다: 둘 다 그 편대에 맞아야 한다
    for (const id of [cw.bodyId, D.stages.phase.crisisShooterId]) {
      check(cw.formationId, id, `stages.phase.crisisWaves[subWave ${cw.subWave}, ${id}]`);
    }
  }
  for (const b of BOSSES()) {
    if (isObj(b) && isObj(b.summon)) check(b.summon.formationId, b.summon.archetypeId, `bosses[${b.id}].summon`);
  }
  EX('S20', n);
}

// ===========================================================================
//  S21 — 드래프트 보장 상한 (§11.1)
//  ① guarantee* 키의 동시 발동 최대 개수 ≤ optionCount − 1 (= 2)
//  ② (v1.10 ㉓) newWeaponSlotScale 길이 = player.weaponSlots, 전부 양수 — 인덱스 = 보유 무기 수 − 1 이라 슬롯 수만큼 있어야 한다.
//     ⑱ 이 슬롯을 4→6 으로 늘리며 이 배열을 안 늘렸고, 무기 5개째부터 가중치 NaN → weighted() 합이 NaN → 유효 후보가 있는데도
//     폴백 3장(플레이테스트 2026-09-05 «Lv13 에 보급 3장»). 값의 «존재»는 게이트가 지킨다.
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
  const ws = D.rules && D.rules.player && D.rules.player.weaponSlots;
  const sc = d.newWeaponSlotScale;
  if (!Array.isArray(sc) || !num(ws) || sc.length !== ws) {
    V('S21', `meta.draft.newWeaponSlotScale 길이 ${Array.isArray(sc) ? sc.length : '?'} ≠ player.weaponSlots ${ws} — 인덱스 = 보유 무기 수 − 1, 모자라면 NaN 가중치가 드래프트 전체를 폴백으로 만든다 (§11.1 ②)`);
  } else {
    for (let i = 0; i < sc.length; i += 1) if (!num(sc[i]) || sc[i] <= 0) V('S21', `meta.draft.newWeaponSlotScale[${i}] = ${sc[i]} — 양수여야 한다 (§11.1 ②)`);
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
  const minPerWave = D.enemies.bands && D.enemies.bands.chaff && D.enemies.bands.chaff.minPerWave;
  let n = 0;

  // ★ v1.10 ⑥ — «지분»이 아니라 «율»이다. 새떼는 위기 내내 반복되고(crisisSwarmLoop) 위기 길이는 격파 시각의 함수라
  //   총량을 정적으로 못 센다. 그리고 비율 모델(§8.19)에서 실제 웨이브 몸 수는 저작 count 가 아니라 chaff.minPerWave 다.
  //   → 초당 XP 로 견준다: 새떼 = crisisTotal × scale × swarmXp ÷ crisisCycleSec,
  //                       초기 = chaff.minPerWave × introXp ÷ earlyWaveIntervalSec (몸 수 하한 = 실측 스폰, §8.7.1).
  //   새떼율 ÷ 초기율 ≤ SWARM_XP_CAP — 위기가 초기 구간보다 빨리 파밍되면 «위기»가 아니라 «수확»이다(§8.10).
  if (!num(minPerWave) || !num(ph.earlyWaveIntervalSec) || !num(ph.crisisCycleSec) || ph.earlyWaveIntervalSec <= 0 || ph.crisisCycleSec <= 0) {
    A('stages.phase.earlyWaveIntervalSec / crisisCycleSec / enemies.bands.chaff.minPerWave', 'S22 의 두 율을 확정할 수 없다');
    return;
  }
  for (const t of rows('S22', D.stages.stages, 'stages.stages',
    '§8.10 — 새떼 XP 상한이 0행을 보면 분모가 없다')) {
    if (!isObj(t)) continue;
    const intro = archById.get(t.introArchetypeId);
    if (!intro || !num(intro.xp)) { A(`stages.stages[${t.id}].introArchetypeId`, 'S22 의 초기율(도입종 xp)을 확정할 수 없다'); continue; }
    const earlyRate = minPerWave * intro.xp / ph.earlyWaveIntervalSec;
    if (earlyRate <= 0) { V('S22', `stages.stages[${t.id}]: 초기 XP 율 = 0 → 분모 없음`); continue; }
    const stageAxis = t.id === FINAL_ID ? [6] : [1, 2, 3, 4, 5];
    for (const st of stageAxis) {
      const sc = num(scale[st - 1]) ? scale[st - 1] : null;
      if (sc === null) { A(`stages.curve.swarmTotalScale[${st - 1}]`, 'S22 를 평가할 수 없다'); continue; }
      n += 1;
      const crisisRate = (num(ph.crisisTotal) ? ph.crisisTotal : 0) * sc * swarmXp / ph.crisisCycleSec;
      const ratio = crisisRate / earlyRate;
      if (ratio > SWARM_XP_CAP + 1e-9) {
        V('S22', `stages.stages[${t.id}] @ 스테이지 ${st}: 새떼 XP 율 ${crisisRate.toFixed(1)}/s ÷ 초기 ${earlyRate.toFixed(1)}/s = ${ratio.toFixed(3)} > ${SWARM_XP_CAP} `
          + '— §13.4-S22 (v1.10 ⑥ 율 기준 · 상한이지 목표가 아니다, §8.10)');
      }
    }
  }
  EX('S22', n);
}

// ===========================================================================
//  S23 — roster 편성 밸런스 (§8.6)
//  ★ v1.5: 「코인원 균질성」 프레이밍 폐기(경제 제거). 편성 불변식만 유지 —
//     테마당 정확히 4종 · turret+bruiser 밴드 1~2종(테마별 탱킹 적 확보).
//  정의역 = themeDraw.pool 에 속한 테마 (finale 은 셔플 대상 밖 — §8.1)
// ===========================================================================
function S23_rosterComposition() {
  const archById = new Map(ARCHETYPES().map((a) => [a && a.id, a]));
  const pool = new Set(rowsQuiet(D.stages.themeDraw && D.stages.themeDraw.pool));
  let n = 0;

  for (const t of rows('S23', D.stages.stages, 'stages.stages',
    '§8.6 — roster 편성이 0행을 보면 테마별 적 구성 밸런스를 아무도 검사하지 않는다')) {
    if (!isObj(t)) continue;
    if (!pool.has(t.id)) continue;    // finale 은 정의역 밖 (셔플 대상이 아니다, §8.1)
    n += 1;
    const roster = rowsQuiet(t.roster);
    const bands = roster.map((r) => (archById.get(r && r.archetypeId) || {}).band).filter(Boolean);
    const cnt = bands.filter((b) => b === 'turret' || b === 'bruiser').length;
    // §8.6 "테마당 정확히 4종 (공용 3 + 시그니처 1)"
    if (roster.length !== 4) V('S23', `stages.stages[${t.id}].roster: ${roster.length}종 ≠ 4 (§8.6)`);
    if (cnt < 1 || cnt > 2) {
      V('S23', `stages.stages[${t.id}]: turret+bruiser 밴드 ${cnt}종 ∉ [1, 2] — 테마별 탱킹 적 밸런스 (§8.6). `
        + `roster 밴드 = [${bands.join(', ')}]`);
    }
  }
  EX('S23', n);
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
    // v1.10 ⑥ — 한 사이클(crisisCycleSec)이 통째로 무대에 서지는 않는다: 몸은 arena.h ÷ (최저 새떼 속도 × sectionSpeedMul.crisis)
    //   초 만에 빠져나간다. 무대 상한 = 사이클 총량 × min(1, 체류 ÷ 사이클). 체류가 사이클보다 길면 옛 식(전량)으로 돌아간다.
    const arch = {}; for (const a of ARCHETYPES()) if (isObj(a)) arch[a.id] = a;
    let vMin = Infinity;
    const swarmIds = [ph.crisisShooterId]; for (const cw of rowsQuiet(ph.crisisWaves)) if (isObj(cw)) swarmIds.push(cw.bodyId);
    for (const id of swarmIds) {
      const a = arch[id]; const v = a && a.moveParams && a.moveParams.speed;
      if (num(v) && v > 0 && v < vMin) vMin = v;
    }
    const mul = ph.sectionSpeedMul && num(ph.sectionSpeedMul.crisis) ? ph.sectionSpeedMul.crisis : 1;
    const ah = D.rules.view && D.rules.view.arena && D.rules.view.arena.h;
    const dwell = (num(ah) && vMin < Infinity) ? ah / (vMin * mul) : Infinity;
    const share = (num(ph.crisisCycleSec) && ph.crisisCycleSec > 0) ? Math.min(1, dwell / ph.crisisCycleSec) : 1;
    rowsQuiet(curve.swarmTotalScale).forEach((s, i) => {
      if (!num(s)) return;
      const total = ph.crisisTotal * s * share;
      if (total > f.swarmConcurrentMax) {
        V('S26', `위기 무대 상한 @ 스테이지 ${i + 1}: crisisTotal(${ph.crisisTotal}) × swarmTotalScale(${s}) × 체류비(${share.toFixed(2)}) = ${total.toFixed(1)} `
          + `> swarmConcurrentMax(${f.swarmConcurrentMax}) (§12.1/S26 v1.10 ⑥)`);
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
      const tag = `stages.stages[${t.id}].waves[${i}] (${w.archetypeId}, eliteIndex=${w.eliteIndex})`;
      const a = archById.get(w.archetypeId);
      if (a && !isAmb(a.band) && !bandAllowed.includes(a.band)) {
        V('S27', `${tag}: band "${a.band}" ∉ elite.bandAllowed [${bandAllowed.join(', ')}] (§8.6/S27)`);
      }
      // v1.10 ⑬ — 웨이브에 element 가 없다(몸마다 mix 봉지). 속성 자격은 «그 스테이지의 mix 에 elementAllowed 밖 속성이 없다»로 본다
      //   (S8 이 mix.normal == 0 을 지키므로 사실상 항상 참 — 여기서는 mix 의 0 이 아닌 키만 확인한다).
      if (isObj(t.mix)) {
        for (const el of Object.keys(t.mix)) {
          if (num(t.mix[el]) && t.mix[el] > 0 && !elemAllowed.includes(el)) {
            V('S27', `${tag}: 이 스테이지 mix 에 elite.elementAllowed 밖 속성 "${el}" (${t.mix[el]}) — 베이크된 엘리트가 그 속성으로 설 수 있다 (§8.6/S27)`);
          }
        }
      }
      // eliteIndex 는 그 웨이브의 개체 인덱스여야 한다
      if (num(w.eliteIndex) && num(w.count) && (w.eliteIndex < 0 || w.eliteIndex >= w.count)) {
        V('S27', `${tag}: eliteIndex ${w.eliteIndex} ∉ [0, count(${w.count})) — 존재하지 않는 개체를 엘리트로 지정했다`);
      }
    });
  }
  EX('S27', n);
  // §8.6: perWaveMax = 베이크된 스포트라이트(eliteIndex) 상한이다. eliteIndex 는 스칼라라
  //   웨이브당 2기 이상을 표현할 수 없다 → 반드시 1. (v1.5: 「엘리트 재롤」이 이 위에 확률로 더 얹는다.)
  if (num(el.perWaveMax) && el.perWaveMax !== 1) {
    C('S27', `rules.elite.perWaveMax = ${el.perWaveMax} ≠ 1 — waves[].eliteIndex 는 스칼라라 웨이브당 2기 이상을 표현할 수 없다 (§8.6/§9.9). 재롤은 elitePerWaveChance 소관`);
  }
  // §8.6(v1.5) — elitePerWaveChance = 라이브 «엘리트 재롤» 확률(런 포지션별). enemies.spawnWave 가
  //   자격 개체(band∈bandAllowed ∧ element∈elementAllowed)를 이 확률로 엘리트화한다(rng.elite, §10.2).
  //   ∴ 확률이므로 ∈[0,1] 이고, 「초반 자유·후반 전면 엘리트」를 인쇄하려면 단조 비감소여야 한다.
  const epc = D.stages.curve && D.stages.curve.elitePerWaveChance;
  if (Array.isArray(epc)) {
    for (let i = 0; i < epc.length; i += 1) {
      if (num(epc[i]) && (epc[i] < 0 || epc[i] > 1)) {
        V('S27', `stages.curve.elitePerWaveChance[${i}] = ${epc[i]} ∉ [0, 1] — 라이브 엘리트 재롤 확률이다 (§8.6/S27)`);
      }
      if (i > 0 && num(epc[i]) && num(epc[i - 1]) && epc[i] < epc[i - 1]) {
        V('S27', `stages.curve.elitePerWaveChance[${i}] = ${epc[i]} < [${i - 1}]=${epc[i - 1]} — 「초반 자유·후반 전면 엘리트」는 단조 비감소를 요구한다 (§8.6/S27)`);
      }
    }
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
  const coreRef = new Set();   // §9.8.1(v1.5) — 코어 이미터(rules.boss.coreEmitterId)도 «참조됨»으로 친다
  if (D.rules.boss && D.rules.boss.coreEmitterId) coreRef.add(D.rules.boss.coreEmitterId);
  let n = 0;
  for (const e of EMITTERS()) {
    if (!isObj(e) || isAmb(e.from)) continue;
    n += 1;
    const lhs = e.from === 'part';
    const inPart = partRef.has(e.id);
    const inOther = midRef.has(e.id) || mobRef.has(e.id) || coreRef.has(e.id);
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
    '§8.10 — 위기 편성이 0행이면 새떼가 아예 등장하지 않는다 (v1.10 ⑥ 서브웨이브당 1행 = 6행)');
  if (!cws.length) return;

  let sum = 0;
  const subs = new Set();
  for (const cw of cws) {
    if (!isObj(cw)) continue;
    if (num(cw.count)) sum += cw.count;
    if (subs.has(cw.subWave)) V('S31', `stages.phase.crisisWaves: subWave ${cw.subWave} 레코드가 둘 — v1.10 ⑥ 은 서브웨이브당 레코드 하나(몸/공격형은 봉지가 가른다)`);
    subs.add(cw.subWave);
  }
  // v1.10 ⑫ — 몸은 레코드가(bodyId, 서브웨이브마다 달라도 된다: arc = 흔들며 오는 새떼, vWedge = 직선 화살), 공격형은 phase 가 지명한다.
  //   둘 다 swarm* · 몸은 attack null · 공격형은 attack ≠ null · 몸 ≠ 공격형. 「좌우로 흔들며 빨리 오는 애도, 직선으로 쭉 오는 애도」(사용자 2026-09-04).
  {
    const arch = {}; for (const a of ARCHETYPES()) if (isObj(a)) arch[a.id] = a;
    const shooter = arch[ph.crisisShooterId];
    if (!shooter) V('S31', `stages.phase.crisisShooterId "${ph.crisisShooterId}" 미지`);
    else {
      if (!/^swarm/.test(shooter.id)) V('S31', `crisisShooterId "${shooter.id}" 가 swarm* 가 아니다 (§8.10)`);
      if (!isObj(shooter.attack)) V('S31', `crisisShooterId "${shooter.id}" 가 안 쏜다 — 새떼의 «공격형»은 attack ≠ null (§8.10 v1.10 ⑥)`);
    }
    const moves = new Set();
    for (const cw of cws) {
      if (!isObj(cw)) continue;
      const body = arch[cw.bodyId];
      if (!body) { V('S31', `crisisWaves subWave ${cw.subWave}: bodyId "${cw.bodyId}" 미지`); continue; }
      if (!/^swarm/.test(body.id)) V('S31', `crisisWaves subWave ${cw.subWave}: bodyId "${body.id}" 가 swarm* 가 아니다 — 위기 세션은 새떼 전용 (§8.10)`);
      if (body.attack !== null) V('S31', `crisisWaves subWave ${cw.subWave}: bodyId "${body.id}" 가 쏜다 — 새떼의 «몸»은 무공격 (§8.10)`);
      if (shooter && body.id === shooter.id) V('S31', `crisisWaves subWave ${cw.subWave}: bodyId == crisisShooterId — 봉지가 가를 두 종이 하나다`);
      moves.add(body.moveId);
    }
    if (moves.size < 2) V('S31', `crisisWaves: 몸의 이동 동사가 ${moves.size}종 — «흔들며 오는 새떼 + 직선 화살» 둘은 있어야 위기가 읽힌다 (§8.10 v1.10 ⑫)`);
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
      for (const arr of ['areaKeys', 'speedKeys', 'durationKeys', 'beamKeys', 'orbitKeys']) {   // ㊲ H2·H5·H6·H7·H8 전부 같은 규약
        for (const k of rowsQuiet(h[arr])) {
          if (!space.has(k)) {
            V('S34', `rules.passiveHooks.${f}.${arr} 의 ${JSON.stringify(k)} 가 "${f}" 의 계약에 없다 `
              + `(§9.6.1 — src = base ∪ evolution.params 가 유효 파라미터 공간이다)`);
          }
        }
      }
      // ㊲ 분류 순수성 — 탄 특화 훅(countKey·pierceApplies·speedKeys·durationKeys)은 탄 패밀리에만, 빔 훅(beamKeys·beamDmgMul)은 빔에만,
      //   범위 훅(areaKeys·areaDmgMul)은 범위 + 미사일 폭발에만, 궤도 훅(orbitKeys)은 궤도에만. 표는 §9.6.1.
      const cls = weaponClassOf(f);
      const bulletHook = h.countKey !== null || h.pierceApplies === true || (h.speedKeys || []).length > 0 || (h.durationKeys || []).length > 0;
      if (bulletHook && cls !== 'bullet') V('S34', `rules.passiveHooks.${f}: 탄 특화 훅(countKey/pierceApplies/speedKeys/durationKeys)이 «${cls}» 분류 무기에 붙었다 (§9.6.1 ㊲)`);
      if (((h.beamKeys || []).length > 0 || h.dmgStat === 'beamDmgMul') && cls !== 'beam') V('S34', `rules.passiveHooks.${f}: 빔 훅이 «${cls}» 분류 무기에 붙었다 (§9.6.1 ㊲)`);
      if (((h.areaKeys || []).length > 0 || h.dmgStat === 'areaDmgMul') && cls !== 'area' && f !== 'missile') V('S34', `rules.passiveHooks.${f}: 범위 훅이 «${cls}» 분류 무기에 붙었다 (§9.6.1 ㊲ — 예외는 미사일 폭발뿐)`);
      if (((h.orbitKeys || []).length > 0 || h.dmgStat === 'orbitMul') && cls !== 'orbital') V('S34', `rules.passiveHooks.${f}: 궤도 훅이 «${cls}» 분류 무기에 붙었다 (§9.6.1 ㊲)`);
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
  if (n && n !== 126) {
    C('S36', `보스 부위 이미터 슬롯이 ${n}개 — §9.8.1(v1.10 ⑩) 은 126개(테마 6종 × (기본 3 + extra 3) + 최종 (4 + 2), × 페이즈 3)라 인쇄했다. `
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
        + `— v1.10: 중간보스 타이머 이탈은 폐지됐다(격파 아니면 위기가 부른다, §8.19). `
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

// §9.5 v1.5 — 진화 짝 패시브 (뱀서식). 10 무기 전부 requiresPassive{id,level} 를 갖고,
//   짝은 실재 패시브 · level∈[1,maxLevel] · 그 무기에 기계적으로 유효(무효 패시브 아님).
/**
 * §13.4-S42 (v1.7) — 적 개성 3종의 «값»을 강제한다.
 *   닫힌-키 검사는 「그 키가 있어도 되는가」만 본다. 타입·범위는 아무도 안 봤다 —
 *   pierceCost 0 이면 관통이 영원히 안 닳아 탄이 아레나를 무한 관통하고, 음수면 관통이 «늘어난다».
 *   개성이 하나도 없으면 §8.17 자체가 죽은 어휘이므로, «적어도 1종은 갖는다»도 함께 강제한다.
 */
/**
 * §13.4-S43 (v1.7) — 적 탄의 반사 예산은 «무제한(-1)»이거나 «없음(0)»이어야 한다.
 *   봇(src/core/bot.js)은 반사탄을 삼각파 접기의 «닫힌 형태»로 외삽한다 — 무한 반사여야
 *   그 수식이 정확하다. 유한 반사는 예산 소진 후 직선이 되어 봇의 예측이 빗나가고,
 *   그러면 회피율이 떨어진 채로 잰 시뮬 수치 전체가 밸런스 판단의 근거로 썩는다.
 *   ★ 플레이어 탄은 봇의 위협 모델에 없으므로 유한 반사가 허용된다(리턴 = 2).
 */
/**
 * §13.4-S44 (v1.7) — 무기 레벨 곡선의 «단조성». 레벨업이 무기를 약하게 만들면 안 된다.
 *   대용치 DPS = dmg × count ÷ rate 이며, count·rate 의 «이름»은 정본이 이미 소유한다
 *   (rules.passiveHooks[family].countKey / rateKey). 그래서 게이트가 어휘를 새로 만들지 않는다.
 *   ★ dmg 단독으로 검사하면 안 된다 — 설계자는 피해를 주기·발수와 맞바꾼다(시커 Lv6 dmg 12→10 이지만
 *     쿨다운 1.05→0.75 · count 3 이라 실제로는 강해진다). 세 항을 함께 봐야 «약해졌는가»가 나온다.
 *   ★ 이 게이트가 없어서 실제로 두 건이 살아 있었다: 바라지 Lv5→6(실측 47→40 DPS)과
 *     리턴의 짝수 count 정면 사각(Lv2 11 → Lv3 0). 둘 다 «레벨업이 약화»다.
 */
/**
 * §13.4-S45 (v1.7) — 드래프트 카드의 «증분 줄»이 원시 키를 노출하지 않는다.
 *   카드는 레벨업이 무엇을 얼마나 바꾸는지 보여준다(§11.1). 그 이름표는 src/render/hud.js 의
 *   PARAM_KO 가 소유하는데, 새 무기 파라미터를 데이터에 넣고 이름표를 안 만들면
 *   화면에 `evoRampFireRateMul 1.55` 같은 **코드 식별자가 그대로 뜬다**.
 *   테스트로는 안 잡힌다(렌더는 던지지 않는다) — 그래서 정적으로 강제한다.
 *   ★ 불리언 진화 파라미터는 표기에서 빠지므로 이름표가 필요 없다.
 */

// ===========================================================================
//  S49 — ★ 부위 도달 가능성 (§8.11 · v1.8)
//  플레이어는 화면 «아래»에서 위로만 쏜다. 부위가 코어보다 위에 있으면 그 부위로 가는
//  모든 사선이 코어에 먼저 막힌다 — partHitPriority:"outermostFirst" 는 「겹치면 부위 우선」
//  이지 「뒤를 뚫어준다」가 아니다.
//  노출폭 = dx 1px 격자에서 yFirst(P,dx) > yFirst(core,dx) 인 dx 의 개수.
//           yFirst(E,dx) = E.ay + √(E.r² − (E.ax−dx)²)     (+y = 아래 = 플레이어 쪽)
//  ★ 정직하게 — 이 게이트가 재는 것은 «코어에 가리는가»뿐이다. 형제 부위 차폐도 sealedNow 도
//    세지 않는다(실제 도달폭은 이 값 이하일 수 있다). 그래도 v1.8 이전의 위반 5건은 전부 0px
//    이었고 통과 부위의 최소는 41px 이라, 이 눈금만으로 그 사고를 영구히 막는다.
// ===========================================================================
function S49_partReach() {
  const minPx = D.rules.boss && D.rules.boss.partReachMinPx;
  if (!num(minPx)) { V('S49', 'rules.boss.partReachMinPx 가 없다 — §8.11 이 요구하는 문턱 (§9.4)'); return; }
  let n = 0;
  for (const b of rows('S49', D.bosses.bosses, 'bosses.bosses',
    '§8.11 — 부위가 0행이면 「보이는 것은 때릴 수 있다」를 아무도 검사하지 않는다')) {
    if (!isObj(b)) continue;
    if (b.tier !== 'stage' && b.tier !== 'final') continue;
    const core = b.core;
    if (!isObj(core) || !num(core.radius)) continue;
    const rc = core.radius;
    for (const p of rowsQuiet(b.parts)) {
      if (!isObj(p) || !Array.isArray(p.anchor) || !num(p.radius)) continue;
      const ax = p.anchor[0], ay = p.anchor[1], r = p.radius;
      if (!num(ax) || !num(ay)) continue;
      n += 1;
      let w = 0;
      for (let dx = Math.ceil(ax - r); dx <= Math.floor(ax + r); dx += 1) {
        const tp = r * r - (ax - dx) * (ax - dx);
        if (tp < 0) continue;
        const yPart = ay + Math.sqrt(tp);
        const tc = rc * rc - dx * dx;
        const yCore = tc < 0 ? -Infinity : Math.sqrt(tc);
        if (yPart > yCore) w += 1;
      }
      if (w < minPx) {
        V('S49', `bosses[${b.id}].parts[${p.id}]: 노출폭 ${w}px < boss.partReachMinPx ${minPx} `
          + `(anchor [${ax}, ${ay}] · r ${r} · core r ${rc}) — 아래에서 위로 쏘는 탄이 코어에 먼저 막힌다. `
          + '부위는 코어보다 «아래»(anchor[1] > 0 쪽)에 있어야 한다 (§8.11)');
      }
    }
  }
  EX('S49', n);
}

// ===========================================================================
//  S50 — 웨이브 몸 수 하한의 정합 (§8.7.1 · v1.8)
//  ① 4밴드 전부 선언 · 정수 ≥ 1
//  ② hpMult 오름차순으로 minPerWave 단조 «비증가» (총 HP 예산 보존과 같은 방향)
//  ③ minPerWave ≤ enemyConcurrentMax ÷ 2 (한 웨이브가 A층 예산 절반을 혼자 먹지 않는다)
//  ★ 이 게이트는 «값의 정합»만 본다 — effHP 가 감당 가능한가는 계측이 답할 몫이다.
// ===========================================================================
function S50_minPerWave() {
  const bands = D.enemies && D.enemies.bands;
  if (!isObj(bands)) { V('S50', 'enemies.bands 가 없다 (§9.7)'); return; }
  const cap = D.rules.fairness && D.rules.fairness.enemyConcurrentMax;
  let n = 0;
  const seen = [];
  for (const bn of BANDS) {
    const bv = bands[bn];
    if (!isObj(bv)) { V('S50', `enemies.bands.${bn} 가 없다`); continue; }
    n += 1;
    const v = bv.minPerWave;
    if (!Number.isInteger(v) || v < 1) {
      V('S50', `enemies.bands.${bn}.minPerWave = ${v}: 정수 ≥ 1 이어야 한다 — `
        + '하드코딩 2 를 대체한 «값»이고 값은 데이터가 소유한다 (C-4 · §8.7.1)');
      continue;
    }
    if (!num(bv.hpMult)) { V('S50', `enemies.bands.${bn}.hpMult 가 수가 아니다`); continue; }
    // §8.19(v1.10) 몸 수의 예산은 «자기 몫»이다 — 무공격 몸(chaff = 도입 밴드)은 introConcurrentMax,
    //   나머지 밴드는 enemyConcurrentMax. 비율 모델에서 웨이브의 몸 수는 chaff 하한이 사실상 정한다
    //   (저작 count 가 3~16 이라 전부 하한에 걸린다, 실측) — 그래서 이 하한이 곧 «초기 구간의 밀도 손잡이»다.
    const introCap = D.rules.fairness && D.rules.fairness.introConcurrentMax;
    const myCap = bn === 'chaff' ? introCap : cap;
    if (myCap !== undefined && num(myCap) && v > myCap / 2) {
      V('S50', `enemies.bands.${bn}.minPerWave = ${v} > ${bn === 'chaff' ? 'introConcurrentMax' : 'enemyConcurrentMax'}(${myCap}) ÷ 2 — `
        + '한 웨이브가 자기 몫 예산의 절반을 혼자 먹는다 (§12.1)');
    }
    seen.push([bn, bv.hpMult, v]);
  }
  seen.sort((a, b) => a[1] - b[1]);
  for (let i = 1; i < seen.length; i += 1) {
    if (seen[i][2] > seen[i - 1][2]) {
      V('S50', `enemies.bands: hpMult 가 큰 밴드의 몸 수 하한이 더 클 수 없다 — `
        + `${seen[i - 1][0]}(hpMult ${seen[i - 1][1]}) minPerWave ${seen[i - 1][2]} → `
        + `${seen[i][0]}(hpMult ${seen[i][1]}) minPerWave ${seen[i][2]} (§8.7.1)`);
    }
  }
  EX('S50', n);
}

// ===========================================================================
//  S51 — ★ 가시 피해 (§8.20 · v1.8)
//  ① src/core 에서 hp 를 «깎는» 자리는 정확히 셋이고 그 주소가 정본이다.
//     ★ 정직하게 — 이것은 증명이 아니라 관용구(`X.hp -=` · `X.hp = X.hp - …`)에 대한 철사다.
//  ② hitEnemy · collide 의 «함수 본문 안»에 onScreen( 이 있다.
//     ★ 파일 단위로 세면 안 된다 — damage.js 는 술어를 «선언»하는 파일이라 선언 자체가
//       토큰을 만족시켜 게이트가 공허해진다.
//  ③ world.enemies.items 를 순회하는 무기는 onScreen( 을 부르거나 이유와 함께 AIM_EXEMPT 에 오른다.
//  ④ min(view.playerBoundsInset) > player.hitboxRadius — 대칭(맞지 않는 적은 때리지도 못한다)의 부등식.
// ===========================================================================
const S51_HP_SITES = ['src/core/damage.js', 'src/core/step.js', 'src/core/step.js'];
const AIM_EXEMPT = {
  'aura.js': '피해가 없다(슬로우+끌어당김) — 오히려 화면 밖 chaff 를 «안»으로 데려온다',
  'nova.js': '조준하지 않는다(플레이어 중심 반경 전체) — 피해는 hitEnemy 가 게이트한다',
  'fan.js': '조준하지 않는다(정면 부채) — 피해는 hitEnemy·collide 가 게이트한다',
  'missile.js': '조준하지 않는다(정면 로켓, 폭발은 탄 자리 반경) — 피해는 hitEnemy 가 게이트한다 (㉟)',
};
function S51_visibleDamage() {
  let n = 0;
  // ① hp 감산 자리
  const found = [];
  const walk = (dir) => {
    for (const f of readdirSync(dir)) {
      const p = join(dir, f);
      if (statSync(p).isDirectory()) { walk(p); continue; }
      if (!f.endsWith('.js') && !f.endsWith('.mjs')) continue;
      const lines = readFileSync(p, 'utf8').split('\n');
      for (let i = 0; i < lines.length; i += 1) {
        const L = lines[i];
        if (/\.hp\s*-=/.test(L) || /\.hp\s*=[^;]*\.hp\s*-/.test(L)) {
          found.push(`${relative(ROOT, p).split('\\').join('/')}:${i + 1}`);
        }
      }
    }
  };
  walk(join(SRC_DIR, 'core'));
  n += found.length;
  const files = found.map((s) => s.split(':')[0]).sort();
  const want = S51_HP_SITES.slice().sort();
  if (files.length !== want.length || files.some((f, i) => f !== want[i])) {
    V('S51', `src/core 의 hp 감산 자리가 [${found.join(' · ')}] — 정본은 `
      + `[${S51_HP_SITES.join(' · ')}] 셋이다. 새 피해원은 §8.20 가시 게이트를 지나야 한다 (§13.4-S51)`);
  }
  // ② 두 함수 «본문 안»의 onScreen
  const body = (src, sig) => {
    const at = src.indexOf(sig);
    if (at < 0) return null;
    let i = src.indexOf('{', at);
    if (i < 0) return null;
    let depth = 0;
    for (let k = i; k < src.length; k += 1) {
      if (src[k] === '{') depth += 1;
      else if (src[k] === '}') { depth -= 1; if (depth === 0) return src.slice(i, k + 1); }
    }
    return null;
  };
  const pairs = [
    ['src/core/damage.js', 'export function hitEnemy('],
    ['src/core/step.js', 'function collide('],
  ];
  for (const [rel, sig] of pairs) {
    const p = join(ROOT, rel);
    if (!existsSync(p)) { V('S51', `${rel} 가 없다`); continue; }
    const bd = body(readFileSync(p, 'utf8'), sig);
    if (bd === null) {
      V('S51', `${rel} 에서 ${sig} 의 본문을 찾지 못했다 — 개명했으면 이 게이트도 함께 고쳐라 (§13.4-S51)`);
      continue;
    }
    n += 1;
    if (bd.indexOf('onScreen(') < 0) {
      V('S51', `${rel} 의 ${sig} 본문에 onScreen( 이 없다 — §8.20 가시 게이트가 «두 경로 모두»에 있어야 한다. `
        + 'v1.5 봉인·v1.7 장갑이 정확히 이 자리에서 한쪽만 막는 사고를 두 번 냈다');
    }
  }
  // ③ 조준 스캔
  const wdir = join(SRC_DIR, 'core', 'weapons');
  if (existsSync(wdir)) {
    for (const f of readdirSync(wdir)) {
      if (!f.endsWith('.js')) continue;
      const src = readFileSync(join(wdir, f), 'utf8');
      if (src.indexOf('enemies.items') < 0) continue;
      n += 1;
      if (src.indexOf('onScreen(') >= 0) continue;
      if (Object.prototype.hasOwnProperty.call(AIM_EXEMPT, f)) continue;
      V('S51', `src/core/weapons/${f}: world.enemies.items 를 순회하면서 onScreen( 을 부르지 않는다 — `
        + '§8.20 조준 필터. 조준하지 않는 무기라면 check.mjs 의 AIM_EXEMPT 에 «이유와 함께» 올려라');
    }
  }
  // ④ 대칭 부등식
  const v = D.rules.view, pl = D.rules.player;
  if (isObj(v) && isObj(v.playerBoundsInset) && isObj(pl) && num(pl.hitboxRadius)) {
    const ins = Object.values(v.playerBoundsInset).filter(num);
    if (ins.length > 0) {
      n += 1;
      const mn = Math.min(...ins);
      if (!(mn > pl.hitboxRadius)) {
        V('S51', `min(view.playerBoundsInset) = ${mn} ≤ player.hitboxRadius ${pl.hitboxRadius} — `
          + '§8.20 대칭(아레나와 겹치지 않는 적은 몸통 충돌도 못 준다)의 증명이 «적 반지름과 무관하게» 성립하려면 '
          + '이 부등식이 필요하다');
      }
    }
  }
  EX('S51', n);
}





/** §8.4 — «화면을 세로로 지나가는» 이동 동사. strafe(yPx 고정)·anchor(정지)·orbitDrift(추적)는 안 지나간다. */
const DESCENT_MOVES = ['dive', 'weave', 'column', 'bounce'];

/**
 * §13.4-S54 (v1.10) — 구간과 비율의 강제 (§8.19).
 *   사용자 확정 사양: 스테이지 진행의 축은 «쏘는 적의 비율»이고, 총량은 대체로 그대로다.
 *   v1.9 의 도입 침묵(S48)·초입 위기(S52)·벽시계(S53)는 이 비율 모델이 «흡수»했다 — 지킬 대상이
 *   사라진 게이트는 남기지 않고, 아직 참인 조항만 여기로 옮겼다(②·③·④).
 *   ① 비율 곡선: 길이 6 · [0,1] · 포지션 단조 비감소 («갈수록 탄이 많아진다») · [0] < 1 (초반은 섞인다)
 *   ② 겹침(구 S53-②): 2 × waveIntervalSec ≤ 정상 로스터 최속 «하강»종의 화면 통과 시간
 *      — 「한 벌이 다 지나간 뒤 다음이 온다 = 끊긴다」를 산술로 막는다
 *   ③ 공급(구 S53-①, v1.10 개정): 웨이브가 흐르는 구간은 초기 + 위기뿐이다(배수·중간보스 구간은 정지).
 *      위기는 격파로 앞당겨질 수 있어 최장 = 마지막 중간보스 등장 직후 ~ mobPhaseSec. 포지션마다
 *      ceil((midBossAtSec[0] − earlyDrainSec) ÷ earlyWaveIntervalSec) + ceil((mobPhaseSec − midBossAtSec[last]) ÷ waveIntervalSec) ≤ mobPhaseMaxWaves
 *   ④ 벽의 차선(구 S52): wall 편대의 차선 순틈 ≥ fairness.minGapWidthPx — 못 지나가는 벽은 ①(완벽하면 안 맞는다) 위반
 *   ⑤ 무공격 칸: stages[].introArchetypeId 가 실재 ∧ attack == null ∧ 위기 전용 아님
 *   ⑥ 속성 3종 보장: themeDraw.count 개를 pool 에서 어떻게 뽑아도 물·불·풀이 전부 나온다
 *      — 2×3 구조가 «우연히» 보장하던 것을 못박는다(테마를 늘리면 조용히 깨진다)
 *   ⑨(㊱) 호 편대 최소 간격: formations.arc.minSepPx ≥ 2 × (arc 로 서는 모든 몸의 반지름) — 몸이 겹치지 않는다(§9.9.2)
 */
function S54_sectionsAndRatio() {
  const st = D.stages; const cu = st && st.curve; const ph = st && st.phase;
  if (!isObj(cu) || !isObj(ph)) { V('S54', 'stages.curve / phase 가 없다'); return; }
  let n = 0;
  // ① 비율 곡선
  // ⑧ (v1.10 ㉜) crisisHpScale — 길이 6 · [0]=1(스테이지 1 의 새떼는 저작 HP 그대로) · ≥ 1 · 단조 비감소 (§8.10)
  n += 1;
  const chs = cu.crisisHpScale;
  if (!Array.isArray(chs) || chs.length !== 6) V('S54', 'stages.curve.crisisHpScale: 길이 6 배열이어야 한다 (§8.10 ㉜)');
  else {
    if (chs[0] !== 1) V('S54', `crisisHpScale[0] = ${chs[0]} ≠ 1 — 스테이지 1 의 새떼는 저작 HP 그대로 (§8.10)`);
    for (let i = 0; i < 6; i += 1) {
      if (typeof chs[i] !== 'number' || chs[i] < 1) V('S54', `crisisHpScale[${i}] = ${chs[i]} < 1`);
      if (i > 0 && chs[i] < chs[i - 1]) V('S54', `crisisHpScale: [${i - 1}]=${chs[i - 1]} → [${i}]=${chs[i]} 로 내려갔다 — 포지션 단조`);
    }
  }
  const r = cu.shooterRatio;
  if (!Array.isArray(r) || r.length !== 6) { V('S54', `stages.curve.shooterRatio: 길이 6 배열이어야 한다 (§8.19)`); }
  else {
    for (let i = 0; i < 6; i += 1) {
      n += 1;
      if (typeof r[i] !== 'number' || r[i] < 0 || r[i] > 1) V('S54', `shooterRatio[${i}] = ${r[i]}: [0,1] 이어야 한다`);
      if (i > 0 && r[i] < r[i - 1]) V('S54', `shooterRatio: [${i - 1}]=${r[i - 1]} → [${i}]=${r[i]} 로 «내려갔다» — 갈수록 탄이 많아져야 한다 (§2.1 ③)`);
    }
    if (typeof r[0] === 'number' && r[0] >= 1) V('S54', `shooterRatio[0] = ${r[0]}: 초반은 무공격이 섞여야 한다 (< 1)`);
  }
  // ② 겹침 — 정상 로스터에 서는 «하강»종의 최속 통과 시간
  const a = D.rules.view.arena; const iv = ph.waveIntervalSec;
  const arch = {}; for (const x of D.enemies.archetypes) arch[x.id] = x;
  const crisisOnly = {}; crisisOnly[ph.crisisShooterId] = 1; for (const c of (ph.crisisWaves || [])) crisisOnly[c.bodyId] = 1;   // v1.10 ⑫
  let fastest = Infinity; let who = '';
  for (const s2 of st.stages) for (const ro of (s2.roster || [])) {
    const x = arch[ro.archetypeId]; if (!x || crisisOnly[x.id]) continue;
    if (DESCENT_MOVES.indexOf(x.moveId) < 0) continue;
    const sp = x.moveParams && x.moveParams.speed; if (typeof sp !== 'number' || sp <= 0) continue;
    const t = (a.h + 2 * x.radius) / sp;
    if (t < fastest) { fastest = t; who = x.id; }
  }
  n += 1;
  if (typeof iv === 'number' && fastest < Infinity && 2 * iv > fastest + 1e-9) {
    V('S54', `2 × waveIntervalSec = ${(2 * iv).toFixed(2)}초 > 최속 하강종 통과 ${fastest.toFixed(2)}초 [${who}] — 한 벌이 다 지나간 뒤 다음이 온다 = 끊긴다 (§8.7.3)`);
  }
  // ③ 공급 — 초기(earlyWaveIntervalSec) + 최장 위기(waveIntervalSec). crisisSuspendsWaves 가 true 면 위기 항은 0.
  n += 1;
  if (Array.isArray(ph.midBossAtSec) && num(ph.earlyWaveIntervalSec) && num(ph.earlyDrainSec) && num(ph.mobPhaseSec)) {
    for (let i = 0; i < ph.midBossAtSec.length; i += 1) {
      const at = ph.midBossAtSec[i];
      if (!Array.isArray(at) || at.length === 0) continue;
      const early = Math.ceil(Math.max(0, at[0] - ph.earlyDrainSec) / ph.earlyWaveIntervalSec);
      const crisis = ph.crisisSuspendsWaves ? 0 : Math.ceil(Math.max(0, ph.mobPhaseSec - at[at.length - 1]) / iv);
      const need = early + crisis;
      if (need > ph.mobPhaseMaxWaves) V('S54', `포지션 ${i + 1}: 초기 ${early} + 최장 위기 ${crisis} = ${need}웨이브가 필요한데 mobPhaseMaxWaves = ${ph.mobPhaseMaxWaves} — 재고가 마르면 스폰이 0 이 된다 (§8.7.3)`);
    }
  }
  // ④ 벽의 차선
  const w = st.formations && st.formations.wall;
  const chaff = arch[(st.stages[0] || {}).introArchetypeId];
  if (isObj(w) && chaff) {
    n += 1;
    const lane = (w.laneSlots + 1) * w.gapPx - 2 * chaff.radius;
    const minGap = D.rules.fairness.minGapWidthPx;
    if (lane < minGap) V('S54', `wall 차선 순틈 ${lane.toFixed(1)}px < fairness.minGapWidthPx ${minGap} — 못 지나가는 벽은 §2.1 ① 위반`);
  }
  // ⑨(㊱) 호 편대의 몸이 겹치지 않는다: 레코드마다 실제 간격 = max(minSepPx, radiusPx × I(span, flatten) ÷ (count−1)) ≥ 2 × 몸의 반지름.
  //   I = 반지름 1 타원 호의 길이(formations.js arcEllipseLength 와 같은 중점 적분 64 등분). 위기 새떼(crisisWaves, swarmTotalScale 상한 1.0
  //   → count 그대로)와 정상 웨이브(stages[].waves) 중 formationId == 'arc' 전부. 엘리트 배율은 새떼 몸에 안 붙고 웨이브 엘리트는 1기라 무시.
  const arcF = st.formations && st.formations.arc;
  if (isObj(arcF)) {
    n += 1;
    const ok = (k) => typeof arcF[k] === 'number' && arcF[k] > 0;
    if (!ok('minSepPx') || !ok('flatten') || !ok('radiusPx') || !ok('spanDeg')) V('S54', 'formations.arc: radiusPx·spanDeg·flatten·minSepPx 는 양수여야 한다 (§9.9.2 ㊱)');
    else {
      const span = arcF.spanDeg * Math.PI / 180; const SEG = 64; const h = span / SEG; let I = 0;
      for (let k = 0; k < SEG; k += 1) { const th = -span / 2 + (k + 0.5) * h; const c = Math.cos(th); const sn = Math.sin(th); I += Math.sqrt(c * c + arcF.flatten * arcF.flatten * sn * sn) * h; }
      const recs = [];
      for (const r of rowsQuiet(ph.crisisWaves)) if (isObj(r) && r.formationId === 'arc' && arch[r.bodyId]) recs.push([arch[r.bodyId], r.count, 'crisisWaves']);
      for (const s2 of rowsQuiet(st.stages)) for (const wv of rowsQuiet(s2.waves)) if (isObj(wv) && wv.formationId === 'arc' && arch[wv.archetypeId]) recs.push([arch[wv.archetypeId], wv.count, `stages[${s2.id}]`]);
      for (const [b, count, where] of recs) {
        const sep = count > 1 ? Math.max(arcF.minSepPx, arcF.radiusPx * I / (count - 1)) : Infinity;
        if (sep < 2 * b.radius) V('S54', `${where} arc ${b.id}×${count}: 간격 ${sep.toFixed(1)}px < 2 × 반지름 ${b.radius} — 호 편대의 몸이 겹쳐 «튜브»로 보인다 (§9.9.2 ㊱)`);
      }
    }
  }
  // ⑤ 무공격 칸
  for (const s2 of st.stages) {
    n += 1;
    const x = arch[s2.introArchetypeId];
    if (!x) { V('S54', `stages[${s2.id}].introArchetypeId "${s2.introArchetypeId}" 미지 (§9.9)`); continue; }
    if (x.attack !== null) V('S54', `stages[${s2.id}].introArchetypeId "${x.id}" 가 쏜다 — 무공격 칸이어야 한다 (§8.19)`);
    if (crisisOnly[x.id]) V('S54', `stages[${s2.id}].introArchetypeId "${x.id}" 은 위기 전용이다 (§8.10)`);
  }
  // ⑥ 속성 3종 보장 — 전수 조합
  const td = st.themeDraw; const el = {};
  for (const s2 of st.stages) if (s2.element) el[s2.id] = s2.element;
  const pool = (td.pool || []).filter((id) => el[id]);
  const k = td.count; const els = new Set(Object.values(el));
  const combos = (arr, m, from, cur, out) => {
    if (cur.length === m) { out.push(cur.slice()); return; }
    for (let i = from; i < arr.length; i += 1) { cur.push(arr[i]); combos(arr, m, i + 1, cur, out); cur.pop(); }
  };
  const all = []; combos(pool, k, 0, [], all);
  for (const c of all) {
    n += 1;
    const got = new Set(c.map((id) => el[id]));
    if (got.size < els.size) V('S54', `themeDraw ${k}/${pool.length} 조합 [${c.join(',')}] 에 속성 ${[...els].filter((e) => !got.has(e)).join('·')} 이 없다 — 스테이지 1~5 에서 물·불·풀을 다 겪어야 한다 (§8.1)`);
  }
  // ⑦ (v1.10 ⑪·⑮) 배수가 무리를 비운다: earlyDrainSec × 도입종 하강속도 × sectionSpeedMul.early ≥ arena.h + spawnPad + 벽 줄 높이 + 2r
  //    — 배수는 «속도를 올리지 않는다»(파밍 구간). 배수 시작 직전에 스폰된 벽(최대 줄 수 만큼 위에서 시작)이 중간보스 등장 전에
  //    아레나 아래로 «나간다». 안 그러면 중간보스 구간이 «중간보스 + 벽»이다.
  n += 1;
  {
    const sm = ph.sectionSpeedMul;
    const a2 = D.rules.view && D.rules.view.arena;
    const v2 = D.rules.view;
    const wall = st.formations && st.formations.wall;
    const arch2 = {}; for (const x of (D.enemies.archetypes || [])) arch2[x.id] = x;
    if (isObj(sm) && num(sm.early) && num(ph.earlyDrainSec) && isObj(a2) && isObj(wall)) {
      const bands2 = D.enemies.bands || {};
      for (const s2 of st.stages) {
        const x = arch2[s2.introArchetypeId]; if (!x) continue;
        const sp = x.moveParams && x.moveParams.speed; if (!num(sp)) continue;
        const perRow = wall.perRow - wall.laneSlots;
        const minPer = bands2[x.band] && bands2[x.band].minPerWave;
        const rows = num(minPer) && perRow > 0 ? Math.ceil(minPer / perRow) : 1;
        const above = (num(v2.spawnLineY) ? a2.y - v2.spawnLineY : 0) + (rows - 1) * wall.rowGapPx + (num(wall.jitterY) ? wall.jitterY * wall.rowGapPx : 0);
        const travel = ph.earlyDrainSec * sp * sm.early;
        const need = a2.h + above + 2 * x.radius;
        if (travel < need) V('S54', `stages[${s2.id}]: 배수 ${ph.earlyDrainSec}초 × ${x.id} ${sp}px/s × early ${sm.early} = ${travel.toFixed(0)}px < 필요 ${need.toFixed(0)}px(아레나 ${a2.h} + 위 ${above.toFixed(0)} + 2r) — 중간보스가 올 때 벽이 남는다 (§8.19 ① 배수)`);
      }
    }
  }
  EX('S54', n);
}

// ─────────────────────────────────────────────────────────────────────────────
//  S55 — 중간보스 «구간» (§8.19 v1.10 · §8.9 · §8.10)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * 사용자 결정: 「무조건 유령/몬스터를 소환하는 중간보스 하나 + 다른 형태의 중간보스들 — 마치 보스 구간처럼」.
 *   ① `phase.midBossFirstId` 가 tier "mid" 에 실재하고 summon ≠ null 이며 `rules.boss.midBossSummonsAllowed` 에 있다
 *      — 소환자가 아니면 중간보스 구간(웨이브 정지)에 «적당히 나올 몹»이 없다
 *   ② 소환자를 뺀 tier "mid" 종이 ≥ 1 — 둘째 마리부터 뽑을 «다른 형태»가 있어야 한다
 *   ③ 시계 일관성: 모든 포지션에서 midBossAtSec[last] < crisisStartSec (S29 가 이미 본다) 그리고
 *      crisisStartSec + crisisCycleSec ≤ mobPhaseSec — 새떼 한 사이클은 돌아야 한다
 *   ④ 위기가 격파로 앞당겨질 때(crisisOnMidBossClear) 무대가 비면 안 된다:
 *      crisisOnMidBossClear ⇒ (¬crisisSuspendsWaves ∨ crisisSwarmLoop) — v1.10 ⑥ 은 새떼 반복이 채운다
 */
function S55_midBossSection() {
  const ph = D.stages && D.stages.phase;
  const bs = D.bosses && D.bosses.bosses;
  const rb = D.rules && D.rules.boss;
  if (!isObj(ph) || !Array.isArray(bs) || !isObj(rb)) { V('S55', 'stages.phase / bosses / rules.boss 가 없다'); return; }
  let n = 0;
  // ① 첫 마리 = 소환자
  n += 1;
  const mids = bs.filter((b) => isObj(b) && b.tier === 'mid');
  const first = mids.find((b) => b.id === ph.midBossFirstId);
  if (!first) V('S55', `stages.phase.midBossFirstId "${ph.midBossFirstId}" 가 tier "mid" 에 없다 (§8.19)`);
  else {
    if (!isObj(first.summon)) V('S55', `midBossFirstId "${first.id}" 의 summon 이 null — 첫 중간보스는 소환자여야 중간보스 구간에 몹이 «적당히» 흐른다 (§8.19)`);
    if (!Array.isArray(rb.midBossSummonsAllowed) || !rb.midBossSummonsAllowed.includes(first.id)) V('S55', `midBossFirstId "${first.id}" 가 rules.boss.midBossSummonsAllowed 에 없다 (S17)`);
  }
  // ② 다른 형태 ≥ 1
  n += 1;
  if (mids.filter((b) => b.id !== ph.midBossFirstId).length === 0) V('S55', '소환자를 뺀 tier "mid" 종이 0 — 둘째 마리부터 뽑을 «다른 형태»가 없다 (§8.19)');
  // ③ 시계
  n += 1;
  if (num(ph.crisisStartSec) && num(ph.crisisCycleSec) && num(ph.mobPhaseSec)
    && ph.crisisStartSec + ph.crisisCycleSec > ph.mobPhaseSec) {
    V('S55', `crisisStartSec ${ph.crisisStartSec} + crisisCycleSec ${ph.crisisCycleSec} > mobPhaseSec ${ph.mobPhaseSec} — 새떼 한 사이클도 못 돈다 (§8.10)`);
  }
  // ④ 앞당김 ⇒ 무대가 비지 않는다: 웨이브가 계속되거나(¬crisisSuspendsWaves) 새떼가 반복된다(crisisSwarmLoop)
  n += 1;
  if (ph.crisisOnMidBossClear === true && ph.crisisSuspendsWaves === true && ph.crisisSwarmLoop !== true) {
    V('S55', 'crisisOnMidBossClear ∧ crisisSuspendsWaves ∧ ¬crisisSwarmLoop — 격파로 앞당긴 위기가 새떼 한 사이클 뒤 페이즈 끝까지 «공백»이 된다 (§8.19 v1.10)');
  }
  EX('S55', n);
}

// ─────────────────────────────────────────────────────────────────────────────
//  S56 — 지형 장판 (§8.21 v1.10 ⑦)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * 사용자 결정: 「공격이 아니라 유틸을 방해하는 지형 — 늪은 느리게, 빙원은 관성, 화산은 과열 정지」.
 *   ① 테마(finale 제외)마다 `terrainKind` ∈ TERRAIN_KINDS · finale 은 "mixed"(3종 순환, v1.10 ⑳) 또는 null(지형 없음)
 *   ② «속성당 하나»: kind == TERRAIN_KIND_ELEMENT 의 역(풀 slow · 물 inertia · 불 heat) — 그림의 색이 이 사전으로 종을 칠하므로
 *      데이터가 어긋나면 늪 위에 물색 장판이 뜬다 (기계는 3종, 테마는 겉모습만 다르다 — §8.21)
 *   ③ 3종이 전부 쓰인다 — 안 쓰이는 종은 죽은 어휘다
 *   ④ rules.terrain 의 값: radiusPx ∈ [24, arena.w ÷ 4] · scrollSpeedPx > 0 · everySec > 0 · 1 ≤ maxOnScreen ≤ caps.terrain
 *      · inertia.responseTauSec ∈ (0, 1] · heat.stallSec ∈ (0, fairness.maxStunSec] ∧ fullSec > stallSec ∧ coolSec > 0
 *      — 과열 정지는 스턴이므로 스턴 상한(§2.7)을 그대로 따른다. 지형은 피해 0 이라 텔레그래프 하한의 대상이 아니다
 *   ⑤ 지형이 화면을 «막지» 않는다: 2 × radiusPx < arena.w − 2 × radiusPx (한 장판이 서 있어도 좌우로 돌아갈 폭이 남는다)
 */
function S56_terrain() {
  const st = D.stages && D.stages.stages;
  const tr = D.rules && D.rules.terrain;
  const caps = D.rules && D.rules.caps;
  const fa = D.rules && D.rules.fairness;
  const a = D.rules && D.rules.view && D.rules.view.arena;
  if (!Array.isArray(st) || !isObj(tr) || !isObj(caps) || !isObj(a) || !isObj(fa)) { V('S56', 'stages / rules.terrain / caps / view.arena / fairness 가 없다'); return; }
  let n = 0;
  const FINAL_ID = FINAL();
  const used = new Set();
  for (const t of st) {
    if (!isObj(t)) continue;
    n += 1;
    if (t.id === FINAL_ID) {
      if (t.terrainKind !== TERRAIN_MIXED && t.terrainKind !== null) V('S56', `stages[${t.id}].terrainKind = "${t.terrainKind}" — 최종 스테이지는 테마가 없으니 "mixed"(3종 순환) 또는 null (§8.21 ③)`);
      continue;
    }
    if (TERRAIN_KINDS.indexOf(t.terrainKind) < 0) { V('S56', `stages[${t.id}].terrainKind = ${JSON.stringify(t.terrainKind)} ∉ ${JSON.stringify(TERRAIN_KINDS)} (§8.21)`); continue; }
    used.add(t.terrainKind);
    if (TERRAIN_KIND_ELEMENT[t.terrainKind] !== t.element) V('S56', `stages[${t.id}] (${t.element}): terrainKind "${t.terrainKind}" 는 ${TERRAIN_KIND_ELEMENT[t.terrainKind]} 의 기계 — 속성당 하나, 색도 그 사전으로 칠한다 (§8.21 ②)`);
  }
  for (const k of TERRAIN_KINDS) if (!used.has(k)) V('S56', `terrainKind "${k}" 를 쓰는 테마가 0 — 죽은 어휘 (§8.21 ③)`);
  // ④ 값
  n += 1;
  if (!num(tr.radiusPx) || tr.radiusPx < 24 || tr.radiusPx > a.w / 4) V('S56', `rules.terrain.radiusPx = ${tr.radiusPx} ∉ [24, arena.w/4 = ${a.w / 4}]`);
  if (!num(tr.scrollSpeedPx) || tr.scrollSpeedPx <= 0) V('S56', `rules.terrain.scrollSpeedPx = ${tr.scrollSpeedPx} — 양수여야 지형이 «흐른다»`);
  if (!num(tr.everySec) || tr.everySec <= 0) V('S56', `rules.terrain.everySec = ${tr.everySec} — 양수`);
  if (!Number.isInteger(tr.maxOnScreen) || tr.maxOnScreen < 1 || !num(caps.terrain) || tr.maxOnScreen > caps.terrain) V('S56', `rules.terrain.maxOnScreen = ${tr.maxOnScreen} ∉ [1, caps.terrain = ${caps.terrain}]`);
  const inr = tr.inertia, ht = tr.heat;
  if (!isObj(inr) || !num(inr.responseTauSec) || inr.responseTauSec <= 0 || inr.responseTauSec > 1) V('S56', `rules.terrain.inertia.responseTauSec = ${inr && inr.responseTauSec} ∉ (0, 1]`);
  if (!isObj(ht) || !num(ht.stallSec) || ht.stallSec <= 0 || (num(fa.maxStunSec) && ht.stallSec > fa.maxStunSec)) V('S56', `rules.terrain.heat.stallSec = ${ht && ht.stallSec} ∉ (0, fairness.maxStunSec = ${fa.maxStunSec}] — 과열 정지는 스턴이다 (§2.7)`);
  if (isObj(ht) && (!num(ht.fullSec) || !(ht.fullSec > ht.stallSec))) V('S56', `rules.terrain.heat.fullSec = ${ht.fullSec} ≤ stallSec ${ht.stallSec} — 정지보다 빨리 차면 연쇄 정지`);
  if (isObj(ht) && (!num(ht.coolSec) || ht.coolSec <= 0)) V('S56', `rules.terrain.heat.coolSec = ${ht.coolSec} — 양수`);
  // ⑤ 통로
  n += 1;
  if (num(tr.radiusPx) && !(2 * tr.radiusPx < a.w - 2 * tr.radiusPx)) V('S56', `rules.terrain.radiusPx = ${tr.radiusPx}: 장판 하나가 아레나 폭 ${a.w} 의 절반을 넘는다 — 돌아갈 폭이 없다 (§8.21 ⑤)`);
  // ⑥ (v1.10 ⑧) 구간·보스 등장 무리·페이드 — spawnIn ⊆ SECTIONS · 비어 있지 않다 · 중복 없음 · ★ 'crisis' 가 없다
  //    (186px/s 새떼 속의 둔화·정지는 확정 피격 = §2.1 ① 위반 — 사용자 결정 2026-09-04) · bossEntryCount ∈ [0, maxOnScreen] · fadeSec > 0
  n += 1;
  if (!Array.isArray(tr.spawnIn) || tr.spawnIn.length === 0) V('S56', 'rules.terrain.spawnIn: 비어 있으면 지형이 어디에도 안 나온다 — 죽은 기능 (§8.21 ④)');
  else {
    const seen = new Set();
    for (const sec of tr.spawnIn) {
      if (SECTIONS.indexOf(sec) < 0) V('S56', `rules.terrain.spawnIn: "${sec}" ∉ ${JSON.stringify(SECTIONS)}`);
      if (seen.has(sec)) V('S56', `rules.terrain.spawnIn: "${sec}" 중복`);
      seen.add(sec);
    }
    if (seen.has('crisis')) V('S56', "rules.terrain.spawnIn 에 'crisis' — 위기(새떼 186px/s) 속의 둔화·정지는 확정 피격이라 §2.1 ① 을 깬다 (§8.21 ④)");
  }
  if (!Number.isInteger(tr.bossEntryCount) || tr.bossEntryCount < 0 || (Number.isInteger(tr.maxOnScreen) && tr.bossEntryCount > tr.maxOnScreen)) V('S56', `rules.terrain.bossEntryCount = ${tr.bossEntryCount} ∉ [0, maxOnScreen = ${tr.maxOnScreen}] (§8.22)`);
  if (!num(tr.fadeSec) || tr.fadeSec <= 0) V('S56', `rules.terrain.fadeSec = ${tr.fadeSec} — 양수 (§8.21 ④)`);
  EX('S56', n);
}

// ─────────────────────────────────────────────────────────────────────────────
//  S57 — 보스 등장 쓸어내기 (§8.22 v1.10 ⑧)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * ① 0 < boss.entryWipeSec < boss.introSec — 쓸어내기는 강림 연출 «안»에서 끝난다(타이머는 강림 뒤 시작, §8.11)
 * ② 앞선 속도 = (arena.h + 40 − spawnLineY) ÷ entryWipeSec > fairness.maxBulletSpeed — 어떤 탄도 앞선을 앞지르지 못한다
 *    (안 그러면 쓸어내기 뒤에도 탄이 남아 «보스뿐인 무대»가 거짓이 된다)
 * ③ visual.wipe.bandPx > 0 · flashAlpha ∈ [0, 1]
 */
function S57_entryWipe() {
  const b = D.rules && D.rules.boss; const v = D.rules && D.rules.view; const fa = D.rules && D.rules.fairness;
  const vw = D.rules && D.rules.visual && D.rules.visual.wipe;
  if (!isObj(b) || !isObj(v) || !isObj(v.arena) || !isObj(fa) || !isObj(vw)) { V('S57', 'rules.boss / view / fairness / visual.wipe 가 없다'); return; }
  let n = 0;
  n += 1;
  if (!num(b.entryWipeSec) || b.entryWipeSec <= 0 || !(num(b.introSec) && b.entryWipeSec < b.introSec)) V('S57', `rules.boss.entryWipeSec = ${b.entryWipeSec} ∉ (0, introSec = ${b.introSec}) (§8.22 ①)`);
  n += 1;
  if (num(b.entryWipeSec) && b.entryWipeSec > 0 && num(v.spawnLineY) && num(fa.maxBulletSpeed)) {
    const speed = (v.arena.y + v.arena.h + 40 - v.spawnLineY) / b.entryWipeSec;
    if (!(speed > fa.maxBulletSpeed)) V('S57', `쓸어내기 앞선 ${speed.toFixed(0)}px/s ≤ fairness.maxBulletSpeed ${fa.maxBulletSpeed} — 탄이 앞선을 앞지른다 (§8.22 ②)`);
  }
  n += 1;
  if (!num(vw.bandPx) || vw.bandPx <= 0) V('S57', `rules.visual.wipe.bandPx = ${vw.bandPx} — 양수`);
  if (!num(vw.flashAlpha) || vw.flashAlpha < 0 || vw.flashAlpha > 1) V('S57', `rules.visual.wipe.flashAlpha = ${vw.flashAlpha} ∉ [0, 1]`);
  EX('S57', n);
}

// ─────────────────────────────────────────────────────────────────────────────
//  S58 — 오빗 반경 = 자석 점선 원 (§7.8 · §9.5, v1.10 ⑨)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * 사용자(2026-09-04): 「오빗 사거리가 너무 짧다 — 비행기 주변 점선(자석 반경)과 일치시켜야」. 화면에 상시 보이는
 *   원이 하나뿐이라(§7.8 자석 반경, 알파 0.12) 오빗의 공이 그 원 위를 돌아야 «내 영역»이 하나로 읽힌다.
 *   ① weapons.orbit.base.orbitRadius == player.magnetRadius ② 어느 레벨도 orbitRadius 를 바꾸지 않는다(항상 일치)
 *   ★ 면적 패시브(areaKeys)는 둘 다 안 건드린다 — 점선은 자석 반경 그대로, 공은 areaMul 을 탄다. 그 어긋남은 «업그레이드가
 *     보인다»는 뜻이라 허용한다(§9.6 areaKeys 의 의도).
 */
function S58_orbitRadius() {
  const ws = D.weapons && D.weapons.weapons;
  const rp = D.rules && D.rules.player;
  if (!Array.isArray(ws) || !isObj(rp)) { V('S58', 'weapons / rules.player 가 없다'); return; }
  const o = ws.find((w) => isObj(w) && w.id === 'orbit');
  if (!o) { V('S58', 'weapons: orbit 이 없다 (§9.5)'); return; }
  let n = 1;
  if (!isObj(o.base) || !num(o.base.orbitRadius) || o.base.orbitRadius !== rp.magnetRadius) {
    V('S58', `weapons.orbit.base.orbitRadius = ${o.base && o.base.orbitRadius} ≠ player.magnetRadius ${rp.magnetRadius} — 공이 점선 원 위를 돌아야 한다 (§7.8)`);
  }
  for (const lv of rowsQuiet(o.levels)) {
    n += 1;
    if (isObj(lv) && has(lv, 'orbitRadius')) V('S58', `weapons.orbit.levels: orbitRadius 를 바꾸는 레벨 — 점선 원과의 일치가 레벨에서 깨진다 (§7.8)`);
  }
  EX('S58', n);
}

// ─────────────────────────────────────────────────────────────────────────────
//  S59 — 특성 (§11.6 v1.10 ⑲ · ㉒)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * 사용자(2026-09-05): 「딱 3개 — 자연 재생·흡혈·쉴드 생성 — 중에서 선택하게 만들자. 선택하면 쿨타임이 줄거나 회복 폭이 늘어나는
 *   방식으로.」 로더(schema.checkTraits)가 형식(닫힌 키·kind 어휘·1:1·values 길이 = maxLevel·양수)을 지키고, 여기는 «설계»를 지킨다:
 *   ① maxLevel = 한 런의 구슬 수(스테이지 보스 5 — 최종은 구슬이 없다) — 다섯 번 고르면 정확히 다섯 레벨이 있다
 *   ② 효과 어휘 3종이 전부 쓰인다 — 안 쓰이는 kind 는 죽은 어휘 (특성 수 = 어휘 수 = 3)
 *   ②' (㉗) 흡혈은 «HP 비율 게이트»를 반드시 단다 — effect.hpRatio ∈ [0.3, 0.6] (사용자 「HP 50% 이하일 때만 — 페널티로 3택이 선택지가 되게」;
 *      1 이면 상시 회복 = ㉒ 의 «깡패» 로 되돌아간다)
 *   ③ 레벨은 «좋아지는 방향»으로 단조: 재생·흡혈은 증가, 쉴드 주기는 감소 — 같은 카드를 다시 골랐는데 나빠지면 안 된다
 *   ④ 값의 범위: regenHpPerSec ≤ 2.0(초당 2 = 100 HP 를 50초에 — 원데스 긴박함의 하한) · lifestealPct ≤ 0.02(HP 50% 이하에서만 듣는
 *      안전망 — 사람 DPS 150 기준 Lv5 1.4% = 2.1 HP/s, 50% 까지만)
 *      · shieldEverySec ≥ fairness.iframeSec(1.0) × 5 — 쉴드가 i-frame 보다 촘촘하면 «맞을 수 없는» 기체가 된다
 *   ⑤ palette.pickup.trait 가 있다 — 구슬은 «보상 그 자체»라 자기 색이 있다(hud.accent 채널). 쉴드 링도 이 색
 */
function S59_traits() {
  const td = D.traits;
  if (!isObj(td) || !Array.isArray(td.traits)) { V('S59', 'traits.json 이 없다 (§11.6)'); return; }
  let n = 0;
  const runBosses = 5;   // 한 런 = 테마 5 + 최종(구슬 없음)
  n += 1;
  if (td.maxLevel !== runBosses) V('S59', `traits.maxLevel = ${td.maxLevel} ≠ 한 런의 구슬 수 ${runBosses} — 다섯 번 고르면 정확히 다섯 레벨이어야 한다 (§11.6 ①)`);
  n += 1;
  const used = new Set(td.traits.filter(isObj).map((t) => isObj(t.effect) ? t.effect.kind : ''));
  for (const k of TRAIT_EFFECT_KINDS) if (!used.has(k)) V('S59', `특성 효과 "${k}" 를 쓰는 특성이 0 — 죽은 어휘 (§11.6 ②)`);
  if (td.traits.length !== TRAIT_EFFECT_KINDS.length) V('S59', `특성 ${td.traits.length}개 ≠ 효과 어휘 ${TRAIT_EFFECT_KINDS.length} — 1:1 (§11.6 ②)`);
  n += 1;
  for (const t of td.traits) {
    if (!isObj(t) || !isObj(t.effect) || t.effect.kind !== 'lifestealPct') continue;
    const hr = t.effect.hpRatio;
    if (!num(hr) || hr < 0.3 || hr > 0.6) V('S59', `traits[${t.id}].effect.hpRatio = ${hr} ∉ [0.3, 0.6] — 흡혈은 «위험할 때만» 듣는 안전망이다 (§11.6 ②')`);
  }
  for (const t of td.traits) {
    if (!isObj(t) || !isObj(t.effect) || !Array.isArray(t.effect.values)) continue;
    const v = t.effect.values; const tag = `traits[${t.id}].effect`;
    n += 1;
    const dec = t.effect.kind === 'shieldEverySec';
    for (let i = 1; i < v.length; i += 1) {
      if (dec ? !(v[i] < v[i - 1]) : !(v[i] > v[i - 1])) V('S59', `${tag}.values[${i}] = ${v[i]} — ${dec ? '주기는 레벨마다 줄어야' : '레벨마다 늘어야'} 한다 (§11.6 ③)`);
    }
    n += 1;
    const mx = Math.max(...v); const mn = Math.min(...v);
    switch (t.effect.kind) {
      case 'regenHpPerSec': {
        // ㊺ «몰빵 보상» — 마지막 레벨만 뛴다(사용자 2026-09-06). Lv5 는 구슬 5개를 한 특성에 다 넣어야 서고, 다섯 번째 구슬은
        //   스테이지 5 보스에서 나오므로 **그 레벨이 존재하는 구간은 최종 스테이지뿐**이다. 그래서 마지막만 상한을 연다.
        const last = v[v.length - 1];
        for (let i = 0; i < v.length - 1; i += 1) if (v[i] > 2.0) V('S59', `${tag}.values[${i}] = ${v[i]} > 2.0 HP/s — Lv1~4 는 기존 밴드다 (§11.6 ④)`);
        if (last > 3.0) V('S59', `${tag}: 마지막 레벨 ${last} > 3.0 HP/s (§11.6 ④ 몰빵 상한)`);
        if (last > v[v.length - 2] * 2) V('S59', `${tag}: 마지막 레벨 ${last} > 직전 ${v[v.length - 2]} × 2 — 몰빵 보상도 «두 배»까지다 (§11.6 ④)`);
        break;
      }
      case 'lifestealPct': if (mx > 0.02) V('S59', `${tag}: 최대 ${mx} > 0.02 (§11.6 ④)`); break;
      case 'shieldEverySec': {
        const lo = D.rules.player.iframeSec * 5;
        if (mn < lo) V('S59', `${tag}: 최소 ${mn}초 < i-frame × 5 = ${lo}초 — 쉴드가 무적보다 촘촘하다 (§11.6 ④)`);
        break;
      }
      default: break;   // 어휘는 로더(schema)가 지킨다
    }
  }
  n += 1;
  if (!(D.rules.palette && D.rules.palette.pickup && typeof D.rules.palette.pickup.trait === 'string')) V('S59', 'rules.palette.pickup.trait 가 없다 (§11.6 ⑤)');
  EX('S59', n);
}




/**
 * §13.4-S47 (v1.8) — 형태 ↔ 이미터 법칙 (§7.6.1).
 *   「모듈의 생김새로 무슨 무기를 쓰는지 알 수 있어야 한다」(플레이 피드백)의 정적 강제.
 *   v1.7 은 모양을 통일하지 않고 기호만 얹었고, 플레이테스트가 그 기호를 「희미한 무언가」라
 *   불렀다. 법칙을 산문으로 두면 다음 저작이 즉시 깬다 — 그래서 게이트로 못 박는다.
 *   ★ 새 키 0. 두 기존 필드(shapeId · 참조된 이미터의 type) 사이의 «관계»만 검사한다.
 *   ★ cross 는 보스 코어 전용이다 — grass 속성 글리프(✚)와 같은 실루엣이라, 부위·잡몹이
 *     cross 를 쓰면 §7.3 이 「100% 커버」라 부른 글리프 채널이 본체에 먹힌다. 코어는
 *     rules.boss.coreElement 가 항상 normal(● 원)이라 이 충돌이 구조적으로 불가능하다.
 */
const SHAPE_BY_TYPE = {          // §7.6.1 12행표 (straight 만 밴드로 갈린다)
  aimed: 'dart', fan: 'claw', ring: 'ring', spiral: 'ring', wall: 'slab',
  laser: 'fin', sweep: 'spike', zone: 'hexPod', mortar: 'bulb',
};
const SHAPE_UNARMED = 'orb';     // 사격하지 않는 적 — 그 «없음»도 정보다(§7.6.1)
const SHAPE_CORE = 'cross';      // 보스 코어 전용 = 「이것은 무기가 아니라 목표다」
const SHAPE_LAW_EXEMPT_MID = new Set(['mbHammer']);  // fan+mortar 두 가족 — §8.9 문서화된 예외

function shapeForType(type, band) {
  if (type === 'straight') return band === 'chaff' ? 'delta' : 'wedge';
  return SHAPE_BY_TYPE[type];
}

function S47_shapeLaw() {
  const emitById = new Map();
  for (const e of EMITTERS()) if (isObj(e)) emitById.set(e.id, e);
  const typeOf = (id) => { const e = emitById.get(id); return e === undefined ? '' : e.type; };
  let n = 0;

  // ① 잡몹 아키타입 — attack 이 곧 형태를 정한다
  for (const a of ARCHETYPES()) {
    if (!isObj(a)) continue;
    n += 1;
    let want;
    if (a.attack === null || a.attack === undefined) want = SHAPE_UNARMED;
    else want = shapeForType(typeOf(a.attack.emitterId), a.band);
    if (want === undefined) { V('S47', `archetypes[${a.id}]: 이미터 타입을 읽지 못했다`); continue; }
    if (a.shapeId !== want) {
      V('S47', `archetypes[${a.id}].shapeId = "${a.shapeId}" ≠ "${want}" — §7.6.1 형태↔이미터 법칙. `
        + `저작자는 shapeId 를 고를 수 없다: 이미터가 고른다`);
    }
  }

  // ②③④ 보스 — 코어는 cross 고정, 부위·중간보스는 P1 이 정한다
  for (const b of BOSSES()) {
    if (!isObj(b)) continue;
    if (b.tier === 'mid') {
      n += 1;
      if (SHAPE_LAW_EXEMPT_MID.has(b.id)) continue;
      const ids = rowsQuiet(b.patternSet && b.patternSet[0] && b.patternSet[0].emitterIds);
      if (!ids.length) continue;
      const want = shapeForType(typeOf(ids[0]), '');
      if (want !== undefined && b.shapeId !== want) {
        V('S47', `bosses[${b.id}].shapeId = "${b.shapeId}" ≠ "${want}" (§7.6.1 · 중간보스는 patternSet[0].emitterIds[0])`);
      }
      continue;
    }
    if (isObj(b.core)) {
      n += 1;
      if (b.core.shapeId !== SHAPE_CORE) {
        V('S47', `bosses[${b.id}].core.shapeId = "${b.core.shapeId}" ≠ "${SHAPE_CORE}" — §7.6.1 코어는 법칙의 유일한 예외이며 `
          + `전 보스가 같은 형태다(「가운데 그것이 이 판을 끝낸다」가 7보스에 걸쳐 한 문장)`);
      }
    }
    for (const p of rowsQuiet(b.parts)) {
      if (!isObj(p)) continue;
      n += 1;
      const ids = rowsQuiet(p.patternSet && p.patternSet[0] && p.patternSet[0].emitterIds);
      if (!ids.length) continue;
      const want = shapeForType(typeOf(ids[0]), '');
      if (want === undefined) continue;
      if (p.shapeId !== want) {
        V('S47', `bosses[${b.id}].parts[${p.id}].shapeId = "${p.shapeId}" ≠ "${want}" — §7.6.1. `
          + `형태는 P1 이 정한다(플레이어가 처음 만나는 페이즈). P2·P3 의 타입 전환은 기호가 말한다`);
      }
      if (p.shapeId === SHAPE_CORE) {
        V('S47', `bosses[${b.id}].parts[${p.id}].shapeId = "cross" — cross 는 코어 전용이다(§7.6.1 · §7.3 grass 글리프 ✚ 와 같은 실루엣)`);
      }
    }
  }
  EX('S47', n);
}

function S45_draftParamLabels() {
  const hudPath = join(ROOT, 'src', 'render', 'hud.js');
  if (!existsSync(hudPath)) { V('S45', 'src/render/hud.js 가 없다'); return; }
  const src = readFileSync(hudPath, 'utf8');
  const m = src.match(/const PARAM_KO = \{([\s\S]*?)\n\};/);
  if (m === null) { V('S45', 'src/render/hud.js 에서 PARAM_KO 표를 찾지 못했다 — 이름이 바뀌었으면 이 게이트도 함께 고쳐라'); return; }
  const known = new Set();
  const re = /(\w+)\s*:/g;
  let hit = re.exec(m[1]);
  while (hit !== null) { known.add(hit[1]); hit = re.exec(m[1]); }
  for (const w of rowsQuiet(D.weapons.weapons)) {
    if (!isObj(w)) continue;
    const seen = new Set();
    for (const lv of (Array.isArray(w.levels) ? w.levels : [])) {
      if (isObj(lv)) for (const k of Object.keys(lv)) seen.add(k);
    }
    if (isObj(w.evolution) && isObj(w.evolution.params)) {
      for (const k of Object.keys(w.evolution.params)) {
        if (typeof w.evolution.params[k] !== 'boolean') seen.add(k);
      }
    }
    for (const k of seen) {
      if (!known.has(k)) {
        V('S45', `weapons[${w.id}] 의 파라미터 "${k}" 에 한글 이름표가 없다 — 드래프트 카드에 코드 식별자가 그대로 뜬다 (src/render/hud.js PARAM_KO, §11.1)`);
      }
    }
  }
}

function S44_weaponCurveMonotonic() {
  const hooks = D.rules.passiveHooks;
  for (const w of rowsQuiet(D.weapons.weapons)) {
    if (!isObj(w) || !Array.isArray(w.levels)) continue;
    const h = hooks[w.family];
    if (!isObj(h) || typeof h.rateKey !== 'string') continue;      // 주기가 없는 무기는 대상 밖
    const cur = Object.assign({}, w.base);
    let prev = null;
    for (let i = 0; i < w.levels.length; i += 1) {
      // ★ «질적» 변화가 낀 스텝은 대용치가 볼 수 없다 — 조준 방식이 바뀌면 같은 수치라도
      //   실제 명중이 달라진다(바라지 Lv8 randomInArena → densest 는 실측 DPS 가 오히려 5배다).
      //   대용치로 판정할 수 없는 구간은 «통과»가 아니라 «측정 불가»로 두고 비교를 끊는다.
      const qualitative = isObj(w.levels[i]) && w.levels[i].targetMode !== undefined;
      Object.assign(cur, w.levels[i]);                              // 부분 오버라이드 누적
      if (qualitative) { prev = null; continue; }
      const rate = cur[h.rateKey];
      const dmg = cur.dmg;
      if (typeof rate !== 'number' || rate <= 0 || typeof dmg !== 'number') { prev = null; continue; }
      const cnt = (h.countKey && typeof cur[h.countKey] === 'number') ? cur[h.countKey] : 1;
      const proxy = (dmg * cnt) / rate;
      if (prev !== null && proxy < prev * 0.995) {
        V('S44', `weapons[${w.id}] Lv${i} → Lv${i + 1}: 대용치 DPS ${prev.toFixed(1)} → ${proxy.toFixed(1)} 로 «역행»한다 — 레벨업이 무기를 약하게 만든다 (dmg×${h.countKey || 1}÷${h.rateKey}, §9.5)`);
      }
      prev = proxy;
    }
  }
}

function S43_bulletBounce() {
  for (const b of rowsQuiet(D.bullets.bullets)) {
    if (!isObj(b) || b.bounceLeft === undefined) continue;
    if (!Number.isInteger(b.bounceLeft) || (b.bounceLeft !== -1 && b.bounceLeft !== 0)) {
      V('S43', `bullets[${b.id}].bounceLeft(${b.bounceLeft}): 적 탄은 -1(무제한) 또는 0(없음)만 허용된다 — 유한 반사는 bot.js 의 닫힌 형태 외삽을 빗나가게 해 시뮬 수치를 못 믿게 만든다 (§8.5 v1.7)`);
    }
  }
}

function S42_enemyTraits() {
  let withTrait = 0;
  for (const a of ARCHETYPES()) {
    if (!isObj(a)) continue;
    const p = `enemies.archetypes[${a.id}]`;
    let n = 0;
    if (a.hitFloorSec !== undefined) {
      n += 1;
      if (typeof a.hitFloorSec !== 'number' || !(a.hitFloorSec > 0) || a.hitFloorSec > 0.5) {
        V('S42', `${p}.hitFloorSec(${a.hitFloorSec}): 수 ∈ (0, 0.5] 여야 한다 — 0 은 «개성 없음»이라 선언 자체가 무의미하고, 큰 값은 무기 종류를 가리지 않는 DPS 벽이 된다 (§8.17)`);
      }
    }
    if (a.pierceCost !== undefined) {
      n += 1;
      if (!Number.isInteger(a.pierceCost) || a.pierceCost < 1 || a.pierceCost > 3) {
        V('S42', `${p}.pierceCost(${a.pierceCost}): 정수 ∈ [1, 3] 여야 한다 — 0·음수는 b.pierceLeft 를 안 깎거나 늘려 탄이 영원히 산다 (§8.17)`);
      }
    }
    if (a.ccImmune !== undefined) {
      n += 1;
      if (typeof a.ccImmune !== 'boolean') V('S42', `${p}.ccImmune(${a.ccImmune}): 불리언이어야 한다 (§8.17)`);
    }
    if (n > 1) {
      V('S42', `${p}: 개성이 ${n}개다 — 한 적은 «무엇이 안 통하는지» 하나만 말한다. 둘을 겹치면 플레이어가 원인을 분리할 수 없다 (§8.17)`);
    }
    if (n === 1) withTrait += 1;
  }
  if (withTrait === 0) {
    V('S42', '적 개성을 가진 아키타입이 0종이다 — §8.17 이 죽은 어휘가 된다. 최소 1종은 가져야 한다');
  }
}

function S41_evolutionPairing() {
  const byId = {};
  for (const p of rowsQuiet(D.passives.passives)) if (isObj(p)) byId[p.id] = p;
  const maxLv = D.passives.maxLevel;
  const hooks = D.rules.passiveHooks;
  let n = 0;
  for (const w of rowsQuiet(D.weapons.weapons)) {
    if (!isObj(w) || !isObj(w.evolution)) continue;
    const rp = w.evolution.requiresPassive;
    if (!isObj(rp)) {
      V('S41', `weapons[${w.id}].evolution.requiresPassive: 없음 — §9.5(v1.5) 진화는 짝 패시브를 요구한다`);
      continue;
    }
    n += 1;
    if (!byId[rp.id]) {
      V('S41', `weapons[${w.id}]: requiresPassive.id "${rp.id}" — 실재하지 않는 패시브 (§9.5)`);
      continue;
    }
    if (!num(rp.level) || rp.level < 1 || rp.level > maxLv) {
      V('S41', `weapons[${w.id}]: requiresPassive.level ${JSON.stringify(rp.level)} — [1, ${maxLv}] 밖 (§9.5)`);
    }
    // 기계적 유효성(㊲ 일반화) — 짝 패시브가 이 무기에 무효면 «투자해도 소용없는 진화 조건»이 된다. 표 = passiveAppliesTo(state.js 와 같은 표).
    //   ★ 짝은 «무기 분류 패시브»여야 한다 — 기체 4(강화 격벽·자세 안정기·학습 회로·상성 증폭)는 짝이 될 수 없다(사용자 2026-09-05: 「무기 짝을 맞추는 방향」).
    const h = hooks[w.family];
    const pdef = byId[rp.id];
    if (isObj(h) && isObj(pdef) && isObj(w.base)) {
      if (BODY_STATS.includes(pdef.stat)) {
        V('S41', `weapons[${w.id}]: 짝 ${rp.id}(${pdef.stat})은 기체 패시브다 — 진화 짝은 무기 분류 패시브여야 한다 (§9.5 ㊲)`);
      } else if (!passiveAppliesTo(h, w.base, pdef.stat)) {
        V('S41', `weapons[${w.id}]: 짝 ${rp.id}(${pdef.stat})은 이 무기에 무효 — §9.5 "기계적으로 유효해야 한다" (훅 표 §9.6.1)`);
      }
    }
  }
  EX('S41', n);
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
      'difficultySpread', 'dominance', 'farmXpRatio', 'crisisKillShareWithoutCapstone'], 'meta.certify.runMode');
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
    ['stages.curve.introQuietWaves', D.stages.curve && D.stages.curve.introQuietWaves],   // §8.19
    ['stages.curve.introSurgeWaves', D.stages.curve && D.stages.curve.introSurgeWaves],   // §8.19.1
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
    if (isObj(bl)) for (const ax of ['draft', 'farm', 'stance']) {
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
  // ★ v1.5: 컨티뉴·코인 상호 정합 검사는 경제 폐지로 제거됐다
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
 *     dominance: {...}, farmXpRatio, crisisKillShareWithoutCapstone,
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

/**
 * ★ D3 — 동적 게이트 채점 (§23 · §13.1). `tools/report/summary.json`(= `sim.mjs --certify` 산출)이
 *   있으면 그 안의 `certify.results` 를 읽어 PASS/FAIL/UNMEASURED 로 «채점»한다(없으면 STUB 유지).
 *   ★ 정적 게이트와 분리된 [DYNAMIC] 채널로 신고한다 — 정적 종료코드(정본 정합)를 흔들지 않는다.
 *   시뮬은 별도 종료코드(`sim --certify` 자체가 실패 시 exit 1)로 커밋을 막는다(§23 개발 흐름).
 */
function dynamicGateGrade() {
  const reportPath = join(ROOT, 'tools', 'report', 'summary.json');
  if (!existsSync(reportPath)) { dynamicGateStubs(); return; }
  let sum;
  try { sum = JSON.parse(readFileSync(reportPath, 'utf8')); }
  catch { S('CERT-DYN', 'report/summary.json 파싱 실패 → 동적 게이트 STUB 유지'); dynamicGateStubs(); return; }
  const cert = sum.certify;
  if (cert === undefined || !Array.isArray(cert.results)) {
    S('CERT-DYN', 'report/summary.json 에 certify.results 없음 (sim --certify 미실행) → 동적 게이트 STUB');
    dynamicGateStubs();
    return;
  }
  for (const r of cert.results) {
    const v = (r.value === null || r.value === undefined)
      ? '   —   '
      : (typeof r.value === 'number' ? r.value.toFixed(4) : String(r.value));
    const band = `${r.min === undefined ? '' : `≥${r.min}`}${r.max === undefined ? '' : ` ≤${r.max}`}`.trim();
    const status = r.status === 'PASS' ? 'PASS' : (r.status === 'FAIL' ? 'FAIL' : 'UNMEASURED');
    report.dynamic.push({ status, name: r.name, v, band });
  }
  report.dynamicMeta = { runs: (sum.run && sum.run.runs) || null, path: relative(ROOT, reportPath) };
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
  line('PRISM WING — check.mjs   (정본 v1.5 §13.4 S1~S47 + §9.3 로더 규칙)');
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

  // ★ D3 — 동적 게이트 채점 결과(report/summary.json 이 있을 때만). 정적 종료코드와 분리.
  if (report.dynamic.length > 0) {
    const src = report.dynamicMeta ? ` (${report.dynamicMeta.path} · runs ${report.dynamicMeta.runs})` : '';
    sect(`[DYNAMIC] 동적 게이트 — sim --certify 채점${src}`,
      report.dynamic, (d) => `${d.status.padEnd(11)} ${d.name.padEnd(38)} ${d.v}   ${d.band}`);
  }

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
  if (report.dynamic.length > 0) {
    const dp = report.dynamic.filter((d) => d.status === 'PASS').length;
    const df = report.dynamic.filter((d) => d.status === 'FAIL').length;
    const du = report.dynamic.filter((d) => d.status === 'UNMEASURED').length;
    line(`동적  PASS ${dp} · FAIL ${df} · UNMEASURED ${du}  (정적 종료코드와 분리 — sim --certify 가 커밋 게이트)`);
  }
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
  line('✓ 전 정적 게이트 통과 (S1~S59 · S33·S40·S46·S48·S52·S53 은 삭제)');
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
  S23_rosterComposition();   // §8.6 roster 편성 밸런스
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
  S34_familyBaseKeys();      // §9.5 12행 표
  S35_passiveValuesLen();    // §9.6
  S36_bossEmitterIdRule();   // §9.8.1
  S37_bossEmitterExists();   // §9.8.1
  S38_midBossLeave();        // §9.8.2
  S39_waveUnlockCoherence(); // §9.9
  S41_evolutionPairing();    // §9.5 v1.5 진화 짝 패시브
  S42_enemyTraits();         // §8.17 v1.7 적 개성 값 범위
  S43_bulletBounce();        // §8.5 v1.7 적 탄 반사 예산
  S44_weaponCurveMonotonic();// §9.5 v1.7 무기 레벨 곡선 단조성
  S45_draftParamLabels();    // §11.1 v1.7 드래프트 카드 이름표
  S47_shapeLaw();            // §7.6.1 v1.8 형태 ↔ 이미터 법칙
  S49_partReach();           // §8.11 v1.8 부위 도달 가능성
  S50_minPerWave();          // §8.7.1 v1.8 웨이브 몸 수 하한
  S54_sectionsAndRatio();    // §8.19 v1.10 구간·비율·겹침·차선·속성3종
  S55_midBossSection();      // §8.19 v1.10 중간보스 구간 — 첫 마리 소환자 · 시계 · 앞당김⇒웨이브 계속
  S56_terrain();             // §8.21 v1.10 ⑦ 지형 장판 — 종·속성당 하나·값·통로·구간
  S57_entryWipe();           // §8.22 v1.10 ⑧ 보스 등장 쓸어내기 — 강림 안·탄보다 빠름·시각값
  S58_orbitRadius();         // §7.8 v1.10 ⑨ 오빗 반경 = 자석 점선 원
  S59_traits();              // §11.6 v1.10 ⑲ 특성 — 회복 묶음·묶음 수·수·값 범위·구슬 색
  S51_visibleDamage();       // §8.20 v1.8 가시 피해

  certifyStatic();      // §13.1 중 정적으로 검사 가능한 것
  dynamicGateGrade();   // ★ D3 — report/summary.json 있으면 채점, 없으면 STUB
  canonDefects();       // 검사를 쓰면서 드러난 정본 결함
  vacuousWatch();       // ★ 0행 게이트 = 공허 통과 = 위반

  process.exit(print());
}

main();
