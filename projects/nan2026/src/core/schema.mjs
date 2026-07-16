/**
 * src/core/schema.mjs
 *
 * 정본 v1.4 구현 절:
 *   §9.2  파일 매니페스트 — 정확히 9개, 닫힘
 *   §9.3  로더 규칙 — schemaVersion 필수 / **미지 키 = 에러** / **누락 키 = 에러, 폴백 금지**
 *         / 참조 무결성 / 모든 이미터가 rules.fairness 통과 필수 / 게임·시뮬 양쪽에서 동일 실행
 *   §9.4~§9.9  각 파일의 인쇄 블록 = 필드 집합의 소유자 (C-7)
 *   §9.5  패밀리별 base 필수 키 집합 12행 + evolution.params 키 집합
 *   §12.4 · §7.4  공정성 하한 (이미터)
 *   §9.1  core 순수성 — 파일을 읽지 않는다. **파싱된 객체를 주입받는다**
 *
 * ★ 이 파일은 fetch 도 fs 도 모른다. 호출자(main.js / tools/sim.mjs)가 9개 JSON 을
 *   파싱해서 넘긴다 → 같은 코드가 브라우저와 node 에서 문자 그대로 동일하게 돈다 (§9.3).
 * ★ 폴백이 없다. 실패는 전부 throw 다 — 조용한 밸런스 드리프트의 근원을 차단한다.
 *
 * ★ tools/check.mjs (S1~S40) 가 전수 정적 게이트이며 이 파일은 **런타임 로더**다.
 *   둘은 §9.3의 같은 규칙을 집행하되 check.mjs 가 상위집합이다.
 */

/** §9.2 — 정확히 9개, 닫힘 */
export const MANIFEST = ['rules', 'elements', 'weapons', 'passives', 'bullets',
  'enemies', 'bosses', 'stages', 'meta'];

/** §9.3 — 모든 파일 루트에 필수. 불일치 → 로드 실패 */
export const SCHEMA_VERSION = 1;

// ---------------------------------------------------------------------------
// 동결 어휘 (§13.4-S3 · C-3)
// ---------------------------------------------------------------------------
const ELEMENTS4 = ['normal', 'fire', 'water', 'grass'];
const FAMILIES = ['forward', 'fan', 'seeker', 'lance', 'orbit', 'aura', 'mine',
  'boomerang', 'barrage', 'omni', 'drone', 'nova'];
const PASSIVE_STATS = ['dmgMul', 'fireRateMul', 'areaMul', 'pierceAdd', 'projCountAdd',
  'elementBonusMul', 'ghostSecOnHit', 'hitBulletClearRadius', 'maxHpAdd', 'moveSpeedMul',
  'xpGainMul', 'coinGainMul'];
const BANDS = ['chaff', 'line', 'turret', 'bruiser'];
const FORMATION_IDS = ['lineH', 'columnV', 'vWedge', 'arc', 'pincer', 'scatter'];
const EMITTER_TYPES = ['straight', 'fan', 'aimed', 'ring', 'spiral', 'laser', 'zone', 'wall'];
const BOSS_TIERS = ['stage', 'mid', 'final'];

/** §9.5 — 패밀리별 base 필수 키 집합 12행 (공통 ✔ + 고유 non-evo). 동결 */
const FAMILY_BASE_KEYS = {
  forward: ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce',
    'hitCooldownSec', 'targetMode', 'spreadDeg', 'jitterDeg', 'burstCount', 'burstIntervalSec'],
  fan: ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce',
    'hitCooldownSec', 'targetMode', 'arcDeg'],
  seeker: ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce',
    'hitCooldownSec', 'targetMode', 'turnRateDegSec', 'acquireRadius', 'retargetSec'],
  lance: ['dmg', 'cooldownSec', 'count', 'pierce', 'hitCooldownSec', 'targetMode',
    'beamWidthPx', 'chargeSec', 'rangePx'],
  orbit: ['dmg', 'projRadius', 'hitCooldownSec', 'orbitRadius', 'angularSpeedDegSec', 'bodyCount'],
  aura: ['dmg', 'radius', 'tickIntervalSec', 'falloff'],
  mine: ['dmg', 'placeIntervalSec', 'armSec', 'triggerRadius', 'blastRadius', 'maxAlive'],
  boomerang: ['dmg', 'cooldownSec', 'count', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce',
    'hitCooldownSec', 'targetMode', 'outRangePx', 'returnSpeed', 'canRehit'],
  barrage: ['dmg', 'cooldownSec', 'targetMode', 'strikeIntervalSec', 'strikesPerVolley',
    'blastRadius', 'telegraphSec'],
  omni: ['dmg', 'cooldownSec', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec',
    'dirCount', 'dirOffsetDeg', 'rearBias'],
  drone: ['dmg', 'projSpeed', 'projRadius', 'lifetimeSec', 'pierce', 'hitCooldownSec', 'targetMode',
    'droneCount', 'anchorOffsets', 'droneFireSec', 'droneRangePx'],
  nova: ['dmg', 'intervalSec', 'radius', 'expandSec', 'telegraphSec'],
};

/** §9.5 — evolution.params 키 집합 (evo* 접두). 동결 */
const FAMILY_EVO_KEYS = {
  forward: ['evoRampSec', 'evoRampFireRateMul'],
  fan: ['evoBlastRadius', 'evoSecondaryDmgMul'],
  seeker: ['evoDistinctTargets', 'evoRetargetOnKill'],
  lance: ['evoFullHeight'],
  orbit: ['evoBulletClearCooldownSec'],
  aura: ['evoPullForce'],
  mine: ['evoClusterCount', 'evoClusterRadius', 'evoSecondaryDmgMul'],
  boomerang: ['evoChainCount'],
  barrage: ['evoRadiusMul'],
  omni: ['evoRingRotDeg'],
  drone: ['evoTrailDelaySec'],
  nova: ['evoRing2Radius', 'evoClearBullets', 'evoSecondaryDmgMul'],
};

/** §9.5 — 허용 targetMode. null = 그 패밀리 계약에 targetMode 키가 없다 */
const FAMILY_TARGET_MODES = {
  forward: ['forward'], fan: ['forward'], seeker: ['nearest', 'lowestHp', 'randomInArena'],
  lance: ['forward', 'nearest'], orbit: null, aura: null, mine: null,
  boomerang: ['forward', 'nearest'], barrage: ['randomInArena', 'densest'],
  omni: null, drone: ['nearest', 'lowestHp', 'forward'], nova: null,
};

/** §4.4 — elementStampMode. 구조 결정 = 잠금 키 */
const STAMP_LIVE = ['orbit', 'aura'];

/** §8.5 — 이미터 공통 9 + 타입별 고유 (발사 원점 키 from, 값 = self | part) */
const EMIT_COMMON = ['id', 'type', 'bulletId', 'from', 'telegraphSec', 'everySec', 'offsetSec',
  'repeat', 'restSec'];
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

/** §7.4 — 텔레그래프 하한 3축. 겹치면 max */
const TELEGRAPH_FLOOR_BY_TYPE = {
  straight: 0.55, fan: 0.60, aimed: 0.60, ring: 0.60,
  spiral: 0.60, wall: 0.80, zone: 0.90, laser: 1.20,
};

/** §9.4 — rules.json 루트 = schemaVersion + 정확히 17 블록 */
const RULES_ROOT_17 = ['loop', 'view', 'collide', 'caps', 'player', 'status', 'bomb', 'elite',
  'boss', 'fairness', 'hud', 'passiveHooks', 'input', 'palette', 'visual', 'render', 'audio'];

// ---------------------------------------------------------------------------
// 검증 원시 함수
// ---------------------------------------------------------------------------
const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

/** 위반 수집기. §9.3의 어떤 위반도 조용히 통과하지 않는다 */
function collector() {
  const errs = [];
  return {
    errs,
    fail(path, msg) { errs.push(`${path}: ${msg}`); },
    /**
     * §9.3 — 미지 키 = 에러 / 누락 키 = 에러 / 폴백 금지.
     * allowed 와 required 가 같은 집합인 것이 정본의 기본값이다.
     */
    closed(path, node, allowed) {
      if (!isObj(node)) {
        this.fail(path, `객체가 아니다 (${node === undefined ? 'undefined' : typeof node})`);
        return false;
      }
      let ok = true;
      const keys = Object.keys(node);
      for (let i = 0; i < keys.length; i += 1) {
        if (allowed.indexOf(keys[i]) < 0) {
          this.fail(`${path}.${keys[i]}`, '미지 키 = 에러 (§9.3) — 정본의 인쇄 블록에 이 필드의 자리가 없다');
          ok = false;
        }
      }
      for (let i = 0; i < allowed.length; i += 1) {
        if (!own(node, allowed[i])) {
          this.fail(`${path}.${allowed[i]}`, '누락 키 = 에러, 기본값 폴백 금지 (§9.3)');
          ok = false;
        }
      }
      return ok;
    },
    vocab(path, v, list) {
      if (list.indexOf(v) < 0) {
        this.fail(path, `동결 어휘 밖의 값 ${JSON.stringify(v)} — 허용 = [${list.join(', ')}] (C-3)`);
        return false;
      }
      return true;
    },
    arr(path, v, len) {
      if (!Array.isArray(v)) { this.fail(path, '배열이 아니다'); return false; }
      if (len !== undefined && v.length !== len) {
        this.fail(path, `${v.length}행 ≠ ${len}행`);
        return false;
      }
      if (len === undefined && v.length === 0) {
        this.fail(path, '0행 — 빈 배열은 통과가 아니다 (인증할 콘텐츠가 없다)');
        return false;
      }
      return true;
    },
  };
}

// ---------------------------------------------------------------------------
// 파일별 검증
// ---------------------------------------------------------------------------
function checkRules(c, r) {
  c.closed('rules', r, ['schemaVersion', ...RULES_ROOT_17]);
  c.closed('rules.loop', r.loop, ['tickHz', 'maxStepsPerFrame', 'maxFrameGapMs', 'interpolate']);
  c.closed('rules.view', r.view, ['logicalW', 'logicalH', 'arena', 'panelLeftW', 'panelRightW',
    'bandTopH', 'bandHpH', 'bandXpH', 'playerBoundsInset', 'spawnLineY', 'minViewportW',
    'minViewportH', 'maxDpr']);
  if (isObj(r.view)) {
    c.closed('rules.view.arena', r.view.arena, ['x', 'y', 'w', 'h']);
    c.closed('rules.view.playerBoundsInset', r.view.playerBoundsInset, ['top', 'bottom', 'left', 'right']);
  }
  c.closed('rules.collide', r.collide, ['gridCellPx']);
  c.closed('rules.caps', r.caps, ['playerBullets', 'enemyBullets', 'enemies', 'pickups', 'zones',
    'drones', 'particles', 'telegraphs', 'damageNumbers', 'effectMarkers', 'overflow']);
  if (isObj(r.caps)) {
    c.closed('rules.caps.overflow', r.caps.overflow, ['playerBullet', 'enemyBullet', 'enemy',
      'pickup', 'zone', 'drone', 'telegraph', 'particle', 'damageNumber', 'effectMarker']);
  }
  // ★ §2.1 healPickupPct — 회복 드랍량의 유일한 거처. data 에 0.35 로 착지됨(required).
  c.closed('rules.player', r.player, ['hpMax', 'spriteRadius', 'hitboxRadius', 'moveSpeed',
    'moveResponseTau', 'diagonalNormalize', 'iframeSec', 'defenseBase', 'damageFloorRatio',
    'lowHpThreshold', 'lowHpCriticalThreshold', 'magnetRadius', 'healPickupPct', 'startWeaponId',
    'startStance', 'stanceSwitchCooldown', 'stancePersistAcrossStages', 'elementCapPerElement',
    'elementCapTotal', 'weaponSlots', 'passiveSlots', 'lives']);
  c.closed('rules.status', r.status, ['slowMoveSpeedMul', 'stackMode', 'resistAffects']);
  c.closed('rules.bomb', r.bomb, ['stockStart', 'stockMax', 'iframeSec', 'mobDmg',
    'clearsEnemyBullets', 'clearsDuringBoss', 'bossDmgRatio', 'bossDmgCap']);
  c.closed('rules.elite', r.elite, ['perWaveMax', 'hpMult', 'sizeMult', 'contactDmgMul', 'xpMult',
    'coin', 'healDropChance', 'bandAllowed', 'elementAllowed']);
  c.closed('rules.boss', r.boss, ['partCount', 'partRegen', 'summonsAllowed', 'partHitPriority',
    'phaseThresholds', 'phaseTransitionSec', 'timerPausesOnPhaseTransition', 'introSec',
    'timerStartsAfterIntro', 'timerExpire', 'coreGateMul', 'mobilityPenalty', 'coreElement',
    'partNormalForbidden', 'partElementDistinctMin', 'partThemeElementMax', 'armorElementNotTheme',
    'armorPartCountRange', 'armorCoreRatioBandPct', 'coin', 'partCoin', 'optionalPartArmorRatio',
    'midBossSummonsAllowed', 'finale']);
  if (isObj(r.boss)) {
    c.closed('rules.boss.finale', r.boss.finale, ['partCount', 'armorPartCount', 'exemptRules',
      'allowNormalPeripheral']);
  }
  c.closed('rules.fairness', r.fairness, ['minTelegraphSec', 'minStunTelegraphSec', 'maxStunSec',
    'maxBulletSpeed', 'maxAimedBulletSpeed', 'statusBulletSpeedMul', 'minBulletRadiusPx',
    'minGapWidthPx', 'minSpawnRadiusPx', 'maxSimultaneousEnemyBullets', 'enemyConcurrentMax',
    'swarmConcurrentMax', 'crisisWaveResidualMax', 'telegraphConcurrentMaxPerEntity',
    'telegraphConcurrentMaxGlobal', 'playerWeaponsExempt']);
  c.closed('rules.hud', r.hud, ['hitboxAlwaysVisible', 'showElementBudget', 'fontHeroPx',
    'fontLargePx', 'fontMediumPx', 'fontBodyPx', 'fontSmallPx', 'panelPadPx', 'keycapBoxPx',
    'bossHpBarH', 'hpBarSegGapPx', 'xpBarH', 'hpBarSegCount', 'panelCacheDirtyOnly',
    'parGhostEnabled', 'elementMatrixInPanel', 'coinShowsScoreValue', 'noHitIndicator',
    'tokenKeycapGatedDisplay', 'stanceHintTargetsMajorityElement', 'icons']);

  // §9.6.1 — family 를 키로 하는 중첩 맵. 12행 동결
  c.closed('rules.passiveHooks', r.passiveHooks, FAMILIES);
  if (isObj(r.passiveHooks)) {
    for (let i = 0; i < FAMILIES.length; i += 1) {
      const f = FAMILIES[i];
      if (!own(r.passiveHooks, f)) continue;
      c.closed(`rules.passiveHooks.${f}`, r.passiveHooks[f],
        ['rateKey', 'countKey', 'pierceApplies', 'areaKeys']);
    }
  }

  c.closed('rules.input', r.input, ['layout', 'socd', 'pauseOnBlur', 'bindings']);
  if (isObj(r.input)) {
    c.closed('rules.input.bindings', r.input.bindings, ['move', 'stanceNormal', 'stanceFire',
      'stanceWater', 'stanceGrass', 'bomb', 'timeToken', 'pause', 'options', 'draftPick', 'reroll',
      'reorderToggle', 'grab', 'confirm', 'cursor']);
  }

  c.closed('rules.palette', r.palette, ['element', 'elementCvd', 'threat', 'status', 'pickup',
    'enemyBody', 'partDestroyed', 'neutralGray', 'hud', 'bg']);
  if (isObj(r.palette)) {
    c.closed('rules.palette.element', r.palette.element, ELEMENTS4);
    c.closed('rules.palette.elementCvd', r.palette.elementCvd, ELEMENTS4);
    c.closed('rules.palette.threat', r.palette.threat, ['enemyBullet', 'telegraph', 'bulletCore', 'outline']);
    c.closed('rules.palette.status', r.palette.status, ['band']);
    c.closed('rules.palette.pickup', r.palette.pickup, ['coin', 'xp']);
    c.closed('rules.palette.hud', r.palette.hud, ['panelBg', 'panelRule', 'textPrimary', 'textDim', 'hpFill']);
    c.closed('rules.palette.bg', r.palette.bg, ['maxSaturation', 'maxLightness', 'cvdMaxLightness',
      'parallaxLayers', 'maxScrollSpeed']);
  }

  c.closed('rules.visual', r.visual, ['iframeBlinkHz', 'stance', 'playerBullet', 'glyph',
    'telegraph', 'band', 'zone', 'timer', 'trail', 'hitFx', 'a11y', 'text']);
  if (isObj(r.visual)) {
    c.closed('rules.visual.stance', r.visual.stance, ['ringExpandSec', 'ringMaxRadiusPx',
      'ringStrokePx', 'emptyDesatSec', 'dotRadiusPx', 'dotRingPx', 'auraAlpha', 'pipPx', 'pipPxCvd',
      'pipGapPx', 'pipOffsetYPx', 'hintPulseHz', 'hintPulseAlpha']);
    c.closed('rules.visual.playerBullet', r.visual.playerBullet, ['coreRadiusRatio', 'coreLightnessAdd']);
    c.closed('rules.visual.glyph', r.visual.glyph, ['bodyRatio', 'maxPx', 'occludedSkip',
      'lodMinBodyPx', 'lodDegradedBodyPx']);
    c.closed('rules.visual.telegraph', r.visual.telegraph, ['strokePx', 'dashPx', 'dashTightenAtPct',
      'dashTightenMul', 'airAlpha', 'fillAlpha', 'emphasisBySpeed']);
    c.closed('rules.visual.band', r.visual.band, ['plateAlpha', 'contentOpaque']);
    c.closed('rules.visual.zone', r.visual.zone, ['fillAlpha', 'pulseHz']);
    c.closed('rules.visual.timer', r.visual.timer, ['warnScale', 'warnPulseHz', 'alertScale', 'alertPulseHz']);
    c.closed('rules.visual.trail', r.visual.trail, ['ghostCount', 'ghostAlpha']);
    c.closed('rules.visual.hitFx', r.visual.hitFx, ['numberTargets', 'numberAggregateSec',
      'numberMinPx', 'numberOutlinePx', 'numberLifeSec', 'numberDriftPx', 'markerPolicy',
      'markerCooldownSecPerEntity', 'superFreezeSec', 'superFreezeScale', 'resistArcSweepDeg',
      'resistArcLifeSec', 'resistArcStrokePx', 'particles']);
    if (isObj(r.visual.hitFx)) {
      c.closed('rules.visual.hitFx.particles', r.visual.hitFx.particles, ['super', 'neutral', 'resist']);
    }
    c.closed('rules.visual.a11y', r.visual.a11y, ['cbMode', 'reduceFlash', 'screenShake',
      'shakeMaxPx', 'fullscreenFlashMaxPerSec', 'fullscreenFlashMaxAlpha']);
    c.closed('rules.visual.text', r.visual.text, ['family', 'minPx', 'outlinePx']);
  }

  c.closed('rules.render', r.render, ['playerFxCompositeAlpha', 'killFxCompositeAlpha',
    'playerBulletMaxAlpha', 'playerBulletMaxRadiusPx', 'particleMaxAlpha', 'particleMaxLifeSec',
    'fxMinRealMs', 'targetFps', 'degradeOnFrameMs', 'degradeRecoverFrames']);
  c.closed('rules.audio', r.audio, ['busGain', 'cueRateLimitPerSec']);
  if (isObj(r.audio)) c.closed('rules.audio.busGain', r.audio.busGain, ['sfx']);

  // §4.3 · §2.6 — 어휘 검사
  if (isObj(r.player)) {
    c.vocab('rules.player.startStance', r.player.startStance, ELEMENTS4);
  }
}

function checkElements(c, e) {
  // §9.4.4
  c.closed('elements', e, ['schemaVersion', 'order', 'investable', 'matrix']);
  if (!c.arr('elements.order', e.order, 4)) return;
  for (let i = 0; i < ELEMENTS4.length; i += 1) {
    if (e.order[i] !== ELEMENTS4[i]) {
      c.fail(`elements.order[${i}]`,
        `${JSON.stringify(e.order[i])} ≠ "${ELEMENTS4[i]}" — §4.1 "표 순서 = 키 순서 = Q(노말) W(불) E(물) R(풀)"`);
    }
  }
  // §4.2 — 노말은 투자축이 아니다
  if (Array.isArray(e.investable) && e.investable.indexOf('normal') >= 0) {
    c.fail('elements.investable', '"normal" 이 들어 있다 — §4.2 "노말은 투자축이 아니다"');
  }
  // §4.1 — 4×4 완전 행렬
  c.closed('elements.matrix', e.matrix, ELEMENTS4);
  if (isObj(e.matrix)) {
    for (let i = 0; i < ELEMENTS4.length; i += 1) {
      c.closed(`elements.matrix.${ELEMENTS4[i]}`, e.matrix[ELEMENTS4[i]], ELEMENTS4);
    }
  }
}

function checkWeapons(c, w) {
  c.closed('weapons', w, ['schemaVersion', 'weapons']);
  if (!c.arr('weapons.weapons', w.weapons, 12)) return;
  for (let i = 0; i < w.weapons.length; i += 1) {
    const it = w.weapons[i];
    const p = `weapons[${it && it.id}]`;
    if (!c.closed(p, it, ['id', 'family', 'name', 'desc', 'elementStampMode', 'base', 'levels', 'evolution'])) continue;
    if (!c.vocab(`${p}.family`, it.family, FAMILIES)) continue;
    // §9.5 — id == family (12종 1:1)
    if (it.id !== it.family) c.fail(p, `id ≠ family(${it.family}) — §9.5 "id == family"`);
    // §4.4 — 구조 결정 = 잠금 키
    const wantStamp = STAMP_LIVE.indexOf(it.family) >= 0 ? 'live' : 'spawn';
    if (it.elementStampMode !== wantStamp) {
      c.fail(`${p}.elementStampMode`, `${JSON.stringify(it.elementStampMode)} ≠ "${wantStamp}" — §4.4/§9.5 표`);
    }
    // §9.5 — base 의 키 집합 == 그 family 의 계약 (미지 = 에러 / 누락 = 에러)
    const baseKeys = FAMILY_BASE_KEYS[it.family];
    c.closed(`${p}.base`, it.base, baseKeys);
    // §9.5 — targetMode 어휘
    const modes = FAMILY_TARGET_MODES[it.family];
    if (modes !== null && isObj(it.base) && own(it.base, 'targetMode')) {
      c.vocab(`${p}.base.targetMode`, it.base.targetMode, modes);
    }
    // §9.3 — levels 는 정확히 8행. 유일한 부분 오버라이드 예외
    if (c.arr(`${p}.levels`, it.levels, 8)) {
      for (let j = 0; j < 8; j += 1) {
        const row = it.levels[j];
        if (!isObj(row)) { c.fail(`${p}.levels[${j}]`, '객체가 아니다 (빈 객체 {} 허용)'); continue; }
        const rk = Object.keys(row);
        for (let k = 0; k < rk.length; k += 1) {
          // 행에 등장하는 키는 반드시 그 패밀리의 계약 안이어야 한다 (§9.3)
          if (baseKeys.indexOf(rk[k]) < 0) {
            c.fail(`${p}.levels[${j}].${rk[k]}`,
              '계약 밖의 키 = 미지 키 = 에러 (§9.3 — levels[i] 는 base 에 대한 부분 오버라이드다)');
          }
        }
      }
    }
    // §9.5 — evolution.params 의 키 집합 == 그 family 의 evo* 목록
    if (c.closed(`${p}.evolution`, it.evolution, ['name', 'desc', 'params'])) {
      c.closed(`${p}.evolution.params`, it.evolution.params, FAMILY_EVO_KEYS[it.family]);
    }
  }
}

function checkPassives(c, ps) {
  c.closed('passives', ps, ['schemaVersion', 'maxLevel', 'stats', 'passives']);
  // §9.6 — 폐쇄 스탯 어휘 12종. 전수 일치 (순서는 정본의 인쇄 순서를 따르지 않아도 된다)
  if (c.arr('passives.stats', ps.stats, 12)) {
    for (let i = 0; i < ps.stats.length; i += 1) c.vocab(`passives.stats[${i}]`, ps.stats[i], PASSIVE_STATS);
    for (let i = 0; i < PASSIVE_STATS.length; i += 1) {
      if (ps.stats.indexOf(PASSIVE_STATS[i]) < 0) {
        c.fail('passives.stats', `"${PASSIVE_STATS[i]}" 누락 — §9.6 폐쇄 어휘 12종`);
      }
    }
  }
  if (!c.arr('passives.passives', ps.passives, 12)) return;
  const seen = [];
  for (let i = 0; i < ps.passives.length; i += 1) {
    const it = ps.passives[i];
    const p = `passives[${it && it.id}]`;
    if (!c.closed(p, it, ['id', 'name', 'desc', 'stat', 'values'])) continue;
    // §9.6 — 각 패시브 = 엔진 훅 정확히 1개. 12훅 = 12 패시브 1:1
    if (!c.vocab(`${p}.stat`, it.stat, PASSIVE_STATS)) continue;
    if (seen.indexOf(it.stat) >= 0) c.fail(p, `stat "${it.stat}" 중복 — §9.6 "12훅 = 12 패시브 1:1"`);
    seen.push(it.stat);
    // §9.6 — values 는 정확히 maxLevel 행. 각 레벨의 **절대 총량**이지 증분이 아니다
    c.arr(`${p}.values`, it.values, ps.maxLevel);
  }
}

function checkBullets(c, b) {
  c.closed('bullets', b, ['schemaVersion', 'bullets']);
  if (!c.arr('bullets.bullets', b.bullets)) return;
  for (let i = 0; i < b.bullets.length; i += 1) {
    const it = b.bullets[i];
    // §9.7 — element 키가 존재하지 않는다. 스키마가 "적의 공격에는 속성이 없다"를 강제한다 (§4.1)
    c.closed(`bullets[${it && it.id}]`, it, ['id', 'radius', 'hitboxScale', 'dmg', 'shape',
      'status', 'statusDurationSec', 'accel', 'turnRateDegSec', 'retargetSec']);
  }
}

function checkEnemies(c, e) {
  c.closed('enemies', e, ['schemaVersion', 'bands', 'archetypes', 'emitters']);
  c.closed('enemies.bands', e.bands, BANDS);
  if (isObj(e.bands)) {
    for (let i = 0; i < BANDS.length; i += 1) {
      const bn = BANDS[i];
      if (!own(e.bands, bn)) continue;
      // §9.7 — xpRef 는 chaff 밴드 전용 필드다
      c.closed(`enemies.bands.${bn}`, e.bands[bn],
        bn === 'chaff' ? ['hpMult', 'coinDropChance', 'coin', 'xpRef'] : ['hpMult', 'coinDropChance', 'coin']);
    }
  }
  if (c.arr('enemies.archetypes', e.archetypes)) {
    for (let i = 0; i < e.archetypes.length; i += 1) {
      const a = e.archetypes[i];
      const p = `enemies.archetypes[${a && a.id}]`;
      // §8.6 — element 는 아키타입 필드가 아니다. 웨이브 편성이 주입한다
      if (!c.closed(p, a, ['id', 'name', 'desc', 'band', 'shapeId', 'radius', 'moveId',
        'moveParams', 'attack', 'contactDmg', 'hp', 'xp', 'score', 'themeOnly'])) continue;
      c.vocab(`${p}.band`, a.band, BANDS);
      if (a.attack !== null) c.closed(`${p}.attack`, a.attack, ['emitterId', 'firstDelaySec']);
    }
  }
  if (c.arr('enemies.emitters', e.emitters)) {
    for (let i = 0; i < e.emitters.length; i += 1) {
      const em = e.emitters[i];
      const p = `enemies.emitters[${em && em.id}]`;
      if (!isObj(em)) { c.fail(p, '객체가 아니다'); continue; }
      if (!c.vocab(`${p}.type`, em.type, EMITTER_TYPES)) continue;
      c.closed(p, em, [...EMIT_COMMON, ...EMIT_OWN[em.type]]);
    }
  }
}

function checkBosses(c, b) {
  c.closed('bosses', b, ['schemaVersion', 'bosses']);
  if (!c.arr('bosses.bosses', b.bosses)) return;
  for (let i = 0; i < b.bosses.length; i += 1) {
    const it = b.bosses[i];
    const p = `bosses[${it && it.id}]`;
    if (!isObj(it)) { c.fail(p, '객체가 아니다'); continue; }
    if (!c.vocab(`${p}.tier`, it.tier, BOSS_TIERS)) continue;
    if (it.tier === 'mid') {
      // §9.8.2 — 중간보스에는 core 가 없다. hp·element 가 루트 필드다
      c.closed(p, it, ['id', 'name', 'tier', 'themeId', 'hp', 'element', 'radius', 'contactDmg',
        'shapeId', 'moveId', 'moveParams', 'patternSet', 'summon', 'parts', 'xp', 'coin',
        'healDropChance', 'score']);
    } else {
      // §9.8 — 스테이지·최종 보스
      c.closed(p, it, ['id', 'name', 'tier', 'themeId', 'armorCoreRatio', 'core', 'parts',
        'movePattern', 'movePatternParams', 'summon']);
      if (isObj(it.core)) {
        c.closed(`${p}.core`, it.core, ['element', 'hp', 'radius', 'contactDmg', 'shapeId', 'score']);
      }
      if (isObj(it.movePatternParams)) {
        c.closed(`${p}.movePatternParams`, it.movePatternParams, ['speedPxSec', 'ampPx', 'yHoldPx']);
      }
      if (Array.isArray(it.parts)) {
        for (let j = 0; j < it.parts.length; j += 1) {
          const pt = it.parts[j];
          c.closed(`${p}.parts[${pt && pt.id}]`, pt, ['id', 'name', 'partType', 'element', 'hp',
            'radius', 'anchor', 'contactDmg', 'shapeId', 'score', 'patternSet']);
        }
      }
    }
    if (isObj(it.summon)) {
      c.closed(`${p}.summon`, it.summon, ['archetypeId', 'count', 'everySec', 'formationId']);
    }
  }
}

function checkStages(c, s) {
  c.closed('stages', s, ['schemaVersion', 'themeDraw', 'curve', 'phase', 'stages', 'formations']);
  c.closed('stages.themeDraw', s.themeDraw, ['pool', 'count', 'allowRepeat', 'stage1RequiresIntroOk', 'finalStageId']);
  c.closed('stages.curve', s.curve, ['enemyHpScale', 'xpScale', 'bossHpScale', 'spawnDensityScale',
    'midBossCount', 'elitePerWaveChance', 'swarmTotalScale', 'rearSpawnAllowed']);
  c.closed('stages.phase', s.phase, ['mobPhaseSec', 'mobPhaseSkippable', 'mobPhaseMaxWaves',
    'waveIntervalSec', 'waveClearAdvance', 'mobPhaseExitFadeSec', 'mobPhaseExitClearBullets',
    'phaseEndAutocollect', 'enemyExitForfeitsReward', 'waveListExhausted', 'crisisPerStage',
    'crisisStartSec', 'crisisDurationSec', 'crisisWarnSec', 'crisisSuspendsWaves', 'crisisTotal',
    'crisisSubWaves', 'crisisWaves', 'midBossAtSec', 'midBossLeaveAfterSec', 'midBossElementRule',
    'midBossForcedLeaveOnCrisis', 'bossTimerSec', 'timerWarnSec', 'timerRedAlertSec',
    'statusStunMaxPerStage']);
  if (isObj(s.phase) && Array.isArray(s.phase.crisisWaves)) {
    for (let i = 0; i < s.phase.crisisWaves.length; i += 1) {
      c.closed(`stages.phase.crisisWaves[${i}]`, s.phase.crisisWaves[i],
        ['subWave', 'formationId', 'archetypeId', 'count', 'spawnEdge']);
    }
  }
  // §9.9.2 — 편대 6종 + 파라미터
  c.closed('stages.formations', s.formations, FORMATION_IDS);
  const FORM_PARAMS = {
    lineH: ['gapPx'], columnV: ['gapSec'], vWedge: ['gapPx', 'angleDeg'],
    arc: ['radiusPx', 'spanDeg'], pincer: ['yStartPx', 'yStepPx'], scatter: ['jitterPx', 'minSepPx'],
  };
  if (isObj(s.formations)) {
    for (let i = 0; i < FORMATION_IDS.length; i += 1) {
      const f = FORMATION_IDS[i];
      if (own(s.formations, f)) c.closed(`stages.formations.${f}`, s.formations[f], FORM_PARAMS[f]);
    }
  }
  if (!c.arr('stages.stages', s.stages)) return;
  for (let i = 0; i < s.stages.length; i += 1) {
    const t = s.stages[i];
    const p = `stages.stages[${t && t.id}]`;
    if (!c.closed(p, t, ['id', 'name', 'element', 'introOk', 'bossId', 'crisisElementRule',
      'roster', 'mix', 'mixGranularity', 'waves'])) continue;
    if (c.arr(`${p}.roster`, t.roster)) {
      for (let j = 0; j < t.roster.length; j += 1) {
        c.closed(`${p}.roster[${j}]`, t.roster[j], ['archetypeId', 'unlockStageMin']);
      }
    }
    if (c.arr(`${p}.waves`, t.waves)) {
      for (let j = 0; j < t.waves.length; j += 1) {
        c.closed(`${p}.waves[${j}]`, t.waves[j], ['formationId', 'archetypeId', 'count', 'element',
          'spawnEdge', 'eliteIndex', 'unlockStageMin']);
      }
    }
  }
}

function checkMeta(c, m) {
  c.closed('meta', m, ['schemaVersion', 'xp', 'draft', 'shop', 'score', 'flow', 'onboarding',
    'difficulty', 'bot', 'certify']);
  c.closed('meta.xp', m.xp, ['curve', 'base', 'exp', 'levelUpsPerRunTarget', 'levelUpQueueMode']);
  if (isObj(m.draft)) {
    c.closed('meta.draft', m.draft, ['optionCount', 'slotAssign', 'categoryWeights',
      'newWeaponSlotScale', 'weaponLevelEvolutionBonus', 'elementFirstLevelBonus', 'passiveNewBonus',
      'distinctItemsPerDraft', 'filterInvalid', 'newWeaponWhenSlotsFull',
      'elementLevelOfferRequiresWeaponCount', 'guaranteeElementCardOnFirstDraft',
      'guaranteeNewWeaponUntilSlots', 'elementCardPity', 'reroll', 'fallback', 'pauseGame']);
    c.closed('meta.draft.categoryWeights', m.draft.categoryWeights,
      ['newWeapon', 'weaponLevel', 'elementLevel', 'passive']);
    c.closed('meta.draft.reroll', m.draft.reroll, ['granularity', 'canRepeatPrevious', 'maxPerDraft']);
    c.closed('meta.draft.fallback', m.draft.fallback, ['id', 'name', 'coins']);
  }
  c.closed('meta.score', m.score, ['superEffectiveDamageShare', 'superEffectiveKillBonusRatio',
    'attribution', 'timeBonusPerGameSec', 'bossClearBonus', 'midBossClearBonus', 'runClearBonus',
    'noHitScope', 'stageNoHitBonus', 'perfectScope', 'perfectBonus', 'shieldPreservesNoHit',
    'timeTokenForfeitsTimeBonus', 'coinToScore', 'roundMode']);
  c.closed('meta.onboarding', m.onboarding, ['autoEquipFirstElement', 'stanceHintPulse', 'stanceHintPulseStageMax']);
  if (isObj(m.flow)) {
    c.closed('meta.flow', m.flow, ['themeBannerSec', 'stageClearSec', 'healSec', 'stageClearHealPct',
      'pauseResumeCountdownSec', 'attractIdleSec', 'continueCost', 'continueTimerRestoreSec',
      'continueIframeSec', 'continueHealToFull', 'continueMaxPerRun', 'menuSpeed', 'deathAnimSec',
      'edgeTriggerOnStateEnter', 'pauseAllowsAbandon', 'attract', 'stagePar']);
    c.closed('meta.flow.attract', m.flow.attract, ['difficulty', 'draftDwellSec', 'endAfterMobPhase']);
  }
  if (isObj(m.difficulty)) {
    c.closed('meta.difficulty', m.difficulty, ['normal', 'hard', 'hell', 'disaster', 'stunMinDifficulty']);
    const ds = ['normal', 'hard', 'hell', 'disaster'];
    for (let i = 0; i < ds.length; i += 1) {
      if (own(m.difficulty, ds[i])) c.closed(`meta.difficulty.${ds[i]}`, m.difficulty[ds[i]], ['speed', 'scoreMul']);
    }
  }
  if (isObj(m.bot)) {
    // §10.4 — grazeTolerancePx 는 삭제됐다 (§2.3 "그레이즈 없음")
    c.closed('meta.bot', m.bot, ['reactionMs', 'reactionJitterMs', 'stanceSwitchMs',
      'dodgeLookaheadSec', 'aimErrorPx', 'slotOrder', 'policies', 'baseline', 'probes']);
    c.closed('meta.bot.policies', m.bot.policies, ['draft', 'farm', 'stance', 'shop']);
    c.closed('meta.bot.baseline', m.bot.baseline, ['draft', 'farm', 'stance', 'shop']);
    c.closed('meta.bot.probes', m.bot.probes, ['dpsProbe', 'forceNoElement']);
  }
  // certify 의 하위 트리는 tools/sim.mjs 의 소유다 (§13.1). 로더는 존재만 요구한다.
  if (!isObj(m.certify)) c.fail('meta.certify', '객체가 아니다 (§13.1)');
}

// ---------------------------------------------------------------------------
// §9.3 — 참조 무결성: 모든 *Id 는 로드 시 대상 존재 확인
// ---------------------------------------------------------------------------
function idList(arr) {
  const out = [];
  if (!Array.isArray(arr)) return out;
  for (let i = 0; i < arr.length; i += 1) if (isObj(arr[i])) out.push(arr[i].id);
  return out;
}

function checkRefs(c, d) {
  const archIds = idList(d.enemies.archetypes);
  const emitIds = idList(d.enemies.emitters);
  const bulletIds = idList(d.bullets.bullets);
  const bossIds = idList(d.bosses.bosses);
  const weaponIds = idList(d.weapons.weapons);
  const stageIds = idList(d.stages.stages);
  const formIds = isObj(d.stages.formations) ? Object.keys(d.stages.formations) : [];

  const need = (list, id, where) => {
    if (id === null || id === undefined) return;
    if (list.indexOf(id) < 0) c.fail(where, `참조 무결성 실패 — "${id}" 가 존재하지 않는다 (§9.3)`);
  };

  for (let i = 0; i < d.enemies.archetypes.length; i += 1) {
    const a = d.enemies.archetypes[i];
    if (!isObj(a)) continue;
    if (isObj(a.attack)) need(emitIds, a.attack.emitterId, `enemies.archetypes[${a.id}].attack.emitterId`);
    if (a.themeOnly !== null) need(stageIds, a.themeOnly, `enemies.archetypes[${a.id}].themeOnly`);
  }
  for (let i = 0; i < d.enemies.emitters.length; i += 1) {
    const e = d.enemies.emitters[i];
    if (isObj(e) && e.bulletId !== null) need(bulletIds, e.bulletId, `enemies.emitters[${e.id}].bulletId`);
  }
  for (let i = 0; i < d.stages.stages.length; i += 1) {
    const t = d.stages.stages[i];
    if (!isObj(t)) continue;
    need(bossIds, t.bossId, `stages.stages[${t.id}].bossId`);
    for (let j = 0; Array.isArray(t.roster) && j < t.roster.length; j += 1) {
      need(archIds, t.roster[j].archetypeId, `stages.stages[${t.id}].roster[${j}].archetypeId`);
    }
    for (let j = 0; Array.isArray(t.waves) && j < t.waves.length; j += 1) {
      need(archIds, t.waves[j].archetypeId, `stages.stages[${t.id}].waves[${j}].archetypeId`);
      need(formIds, t.waves[j].formationId, `stages.stages[${t.id}].waves[${j}].formationId`);
      // §4.1 — 웨이브가 주입하는 속성은 4속성 어휘 안이어야 한다
      c.vocab(`stages.stages[${t.id}].waves[${j}].element`, t.waves[j].element, ELEMENTS4);
    }
  }
  need(weaponIds, d.rules.player.startWeaponId, 'rules.player.startWeaponId');
  need(stageIds, d.stages.themeDraw.finalStageId, 'stages.themeDraw.finalStageId');
}

// ---------------------------------------------------------------------------
// §9.3 — "모든 이미터가 rules.fairness 통과 필수. 위반 → 로드 실패"
//   §12.4 (각 하나의 값) + §7.4 (텔레그래프 하한 3축, 겹치면 max)
//   ★ 플레이어 무기는 대상이 아니다 (fairness.playerWeaponsExempt, §9.5)
// ---------------------------------------------------------------------------
function checkFairness(c, d) {
  const f = d.rules.fairness;
  if (f.playerWeaponsExempt !== true) {
    c.fail('rules.fairness.playerWeaponsExempt',
      `${f.playerWeaponsExempt} ≠ true — §9.5 "플레이어 무기는 fairness 의 대상이 아니다"`);
  }
  const bullets = d.bullets.bullets;
  const findBullet = (id) => {
    for (let i = 0; i < bullets.length; i += 1) if (bullets[i].id === id) return bullets[i];
    return null;
  };
  for (let i = 0; i < d.enemies.emitters.length; i += 1) {
    const e = d.enemies.emitters[i];
    if (!isObj(e)) continue;
    const p = `enemies.emitters[${e.id}]`;
    const b = e.bulletId === null ? null : findBullet(e.bulletId);

    // (1) §7.4 — 텔레그래프 하한. 절대 하한 + 거동별 + 탄 상태. 겹치면 max
    //     ★ 개체 클래스 축(중간보스 1.20 · 보스 부위 1.50)은 이미터가 아니라 **참조자**가
    //       결정한다 → 전수 검사는 tools/check.mjs S6 이 소유한다 (로더는 이미터 로컬 축만).
    let floor = f.minTelegraphSec;
    const byType = TELEGRAPH_FLOOR_BY_TYPE[e.type];
    if (byType !== undefined && byType > floor) floor = byType;
    if (b !== null && b.status === 'slow' && 0.80 > floor) floor = 0.80;
    if (b !== null && b.status === 'stun' && f.minStunTelegraphSec > floor) floor = f.minStunTelegraphSec;
    if (typeof e.telegraphSec === 'number' && e.telegraphSec < floor - 1e-9) {
      c.fail(`${p}.telegraphSec`, `${e.telegraphSec} < ${floor} — §7.4 3축 max`);
    }

    // (2) §12.4 — 탄 속도. 이미터가 유일 소유자 (§9.7: bullets[].speed 는 삭제됐다)
    if (typeof e.speed === 'number') {
      if (e.speed > f.maxBulletSpeed) {
        c.fail(`${p}.speed`, `${e.speed} > fairness.maxBulletSpeed(${f.maxBulletSpeed}) (§12.4)`);
      }
      if (e.type === 'aimed' && e.speed > f.maxAimedBulletSpeed) {
        c.fail(`${p}.speed`, `조준탄 ${e.speed} > fairness.maxAimedBulletSpeed(${f.maxAimedBulletSpeed}) (§12.4)`);
      }
      if (b !== null && b.status !== null) {
        const cap = f.maxBulletSpeed * f.statusBulletSpeedMul;
        if (e.speed > cap + 1e-9) {
          c.fail(`${p}.speed`, `${e.speed} > ${cap} — 상태이상 탄은 크고 느려야 한다 (§12.4)`);
        }
      }
    }
    // (3) §12.4 — 최소 탄 반경
    if (b !== null && b.radius < f.minBulletRadiusPx) {
      c.fail(`${p} → bullets[${b.id}].radius`, `${b.radius} < fairness.minBulletRadiusPx(${f.minBulletRadiusPx}) (§12.4)`);
    }
    // (4) §12.4 — wall 의 통과 틈
    if (e.type === 'wall') {
      if (e.gapWidthPx < f.minGapWidthPx) {
        c.fail(`${p}.gapWidthPx`, `${e.gapWidthPx} < fairness.minGapWidthPx(${f.minGapWidthPx}) (§12.4)`);
      }
      if (e.gapCount < 1) c.fail(`${p}.gapCount`, `${e.gapCount} — 틈 없는 벽은 회피 불가 (§12.4)`);
    }
    // (5) §12.4 — 스턴 최대 지속
    if (b !== null && b.status === 'stun' && b.statusDurationSec > f.maxStunSec) {
      c.fail(`${p} → bullets[${b.id}].statusDurationSec`,
        `${b.statusDurationSec} > fairness.maxStunSec(${f.maxStunSec}) (§12.4)`);
    }
  }
}

// ---------------------------------------------------------------------------
// 공개 API
// ---------------------------------------------------------------------------
/**
 * §9.3 — 9개 파일의 **이미 파싱된 객체**를 검증하고 그대로 돌려준다.
 * 위반이 하나라도 있으면 throw. 부분 로드도 폴백도 없다.
 *
 * @param raw { rules, elements, weapons, passives, bullets, enemies, bosses, stages, meta }
 * @returns 같은 객체 (동결됨)
 */
export function validate(raw) {
  const c = collector();

  // §9.2 — 매니페스트는 닫혀 있다. 여분 파일도 에러다
  const given = Object.keys(raw);
  for (let i = 0; i < given.length; i += 1) {
    if (MANIFEST.indexOf(given[i]) < 0) {
      c.fail(`data/${given[i]}.json`, '§9.2 매니페스트(정확히 9개, 닫힘) 밖의 파일');
    }
  }
  for (let i = 0; i < MANIFEST.length; i += 1) {
    const n = MANIFEST[i];
    if (!own(raw, n)) { c.fail(`data/${n}.json`, '매니페스트 파일 누락 (§9.2)'); continue; }
    if (!isObj(raw[n])) { c.fail(`data/${n}.json`, '루트가 객체가 아니다'); continue; }
    // §9.3 — schemaVersion 은 모든 파일 루트에 필수. 불일치 → 로드 실패
    if (!own(raw[n], 'schemaVersion')) {
      c.fail(`${n}.schemaVersion`, '누락 (§9.3 — 모든 파일 루트에 필수)');
    } else if (raw[n].schemaVersion !== SCHEMA_VERSION) {
      c.fail(`${n}.schemaVersion`, `${raw[n].schemaVersion} ≠ ${SCHEMA_VERSION} → 로드 실패 (§9.3)`);
    }
  }
  if (c.errs.length > 0) throwAll(c.errs);

  checkRules(c, raw.rules);
  checkElements(c, raw.elements);
  checkWeapons(c, raw.weapons);
  checkPassives(c, raw.passives);
  checkBullets(c, raw.bullets);
  checkEnemies(c, raw.enemies);
  checkBosses(c, raw.bosses);
  checkStages(c, raw.stages);
  checkMeta(c, raw.meta);
  if (c.errs.length > 0) throwAll(c.errs);

  // 구조가 성립한 뒤에만 참조·공정성을 본다 (undefined 를 훑지 않기 위해)
  checkRefs(c, raw);
  checkFairness(c, raw);
  if (c.errs.length > 0) throwAll(c.errs);

  return raw;
}

function throwAll(errs) {
  const head = `schema.mjs: §9.3 로더 규칙 위반 ${errs.length}건 — 로드 실패 (폴백 금지)`;
  throw new Error(`${head}\n  ${errs.join('\n  ')}`);
}
