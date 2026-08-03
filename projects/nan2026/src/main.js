/**
 * src/main.js — 부트스트랩 (브라우저 전용, **순수하지 않다**)
 *
 * 정본 v1.4 구현 절:
 *   §9.1   main.js = 캔버스·rAF·입력·화면 전환. **게임 로직 없음.** ★배속이 존재하는 유일한 곳★
 *   §9.2   data/*.json 9개를 Promise.all(fetch) 병렬 로드
 *   §9.3   schema.mjs 가 게임·시뮬 양쪽에서 동일 실행. 위반 = 로드 실패 + 에러 화면 (폴백 금지)
 *   §1.1   s = min(vw/1280, vh/720) · CSS 중앙 정렬 레터박스 · 백킹스토어 1280·min(dpr,2)
 *          · 뷰포트 < minViewportW/H → 플레이 차단 + 안내 (입력 무시)
 *   §5.1   키맵 — 이동 = 방향키 / Q W E R = 스탠스. `event.code` (물리 키 위치)
 *   §5.2   드래프트 — Digit1~3 즉시 선택 · ←→ 커서 · Enter 확정 · F 리롤 · Escape 무시
 *   §5.5   flow.edgeTriggerOnStateEnter — 상태 진입 프레임에 이미 눌려 있던 키는 무시
 *   §5.7   매 고정 틱마다 키 상태를 **폴링**한다 (이벤트 큐 아님) · 게임 키 전부 preventDefault
 *          · blur → 자동 일시정지 + acc = 0 (input.pauseOnBlur)
 *   §6.1   ★ tickDur = 1000 / (TICK_HZ × speed) — **배속이 코드에 존재하는 정확히 1곳**
 *   §6.4   레벨업 드래프트 — draft.pauseGame: 게임 클럭을 멈춘다
 *   §10.1  고정 타임스텝 · 나선형 죽음 방지(남은 시간 폐기) · render(alpha) 는 위치 보간만
 *   §10.2  마스터 시드 = (Date.now() ^ (performance.now()×1000)) >>> 0
 *          — ★ 비결정성이 들어오는 유일한 지점이며 core 바깥에서 생성해 주입한다
 *
 * ★ 단방향: 이 파일은 core 를 import 한다. core 는 이 파일을 모른다.
 */

import { validate, MANIFEST } from './core/schema.mjs';
import { createWorld } from './core/state.js';
import { step, makeInput, TICK_HZ } from './core/step.js';
import { buildDraft, applyCard, candidates } from './core/draft.js';
import { weapons } from './core/weapons/index.js';
import { enemies } from './core/enemies.js';
import { emitters } from './core/emitters.js';
import { bossHook } from './core/boss.js';
import { initRun, tickRun, advanceStage, applyStageClearHeal, stageEntry, PHASE } from './core/stage.js';
import { tally } from './core/score.js';
import { seedHex } from './core/rng.js';
import { resolvePalette, drawWorld, makeInterp, captureInterp, makeFx, updateFx, rgba } from './render/draw.js';
import { drawPanels, drawDraft, drawResults } from './render/hud.js';

// ---------------------------------------------------------------------------
// 에러 화면 (§9.3 — 로드 실패는 조용히 지나가지 않는다)
// ---------------------------------------------------------------------------
function fatal(err) {
  const el = document.getElementById('fatal');
  el.textContent = String(err && err.message ? err.message : err);
  el.hidden = false;
  document.getElementById('game').hidden = true;
}

// ---------------------------------------------------------------------------
// 데이터 (§9.2 · §9.3)
// ---------------------------------------------------------------------------
async function loadData() {
  const jobs = MANIFEST.map(async (name) => {
    const url = new URL(`../data/${name}.json`, import.meta.url);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`data/${name}.json 로드 실패 (HTTP ${res.status}) — §9.2 매니페스트 9개는 닫혀 있다`);
    return [name, await res.json()];
  });
  const pairs = await Promise.all(jobs);
  const raw = {};
  for (let i = 0; i < pairs.length; i += 1) raw[pairs[i][0]] = pairs[i][1];
  return validate(raw);                       // §9.3 — 미지 키 = 에러 / 누락 키 = 에러 / 폴백 금지
}

/**
 * ★ 1주차 시임 — 드래프트 후보 × 구현된 패밀리의 교집합 (`src/core/weapons/index.js` 가 이 일을
 *   **명시적으로 main.js 에 위임**했다):
 *
 *     draft.js 의 newWeapon 후보는 data/weapons.json 의 **12 패밀리 전부**에서 나온다. 그런데
 *     1주차 레지스트리는 3개다 → 미구현 패밀리 카드를 확정하면 step.fireWeapons 가 던진다
 *     (§9.3 폴백 금지의 올바른 동작이다 — 조용히 안 쏘는 무기가 밸런스를 드리프트시키는 것보다 낫다).
 *
 * ★ 보고 대상: 정본 §11.1 에는 「구현된 무기」라는 개념이 없다 (당연히 — 완성본에는 12개가 다 있다).
 *   그러므로 이것은 **정본의 규칙이 아니라 1주차의 발판**이며, 12 패밀리가 다 서면 통째로 삭제된다.
 *   그때 이 함수를 지우는 것 말고 되돌릴 것이 없도록 **core 를 한 줄도 건드리지 않는다.**
 *
 * ★ 지운 자리를 곧바로 resupply 로 메우지 않는 이유: `guaranteeNewWeaponUntilSlots`(2) 때문에
 *   초반 드래프트는 newWeapon 을 1장 보장하는데, 1주차엔 11 후보 중 9가 미구현이라
 *   **보장 카드가 82% 확률로 「보급」이 되어** 3택이 2택으로 쪼그라든다. 같은 유효 후보 풀에서
 *   대체를 뽑아 「3장 = 실제 선택 3개」를 지킨다. resupply 는 draft.js 와 같은 **최후** 폴백이다.
 */
function implemented(c) {
  return c.category !== 'newWeapon' || Object.prototype.hasOwnProperty.call(weapons, c.weaponId);
}

function dropUnimplementedWeapons(world, draft) {
  const d = world.data.meta.draft;
  const taken = new Set(draft.excluded);
  for (let i = 0; i < draft.cards.length; i += 1) taken.add(draft.cards[i].key);

  for (let i = draft.cards.length - 1; i >= 0; i -= 1) {
    if (implemented(draft.cards[i])) continue;
    taken.delete(draft.cards[i].key);
    draft.cards.splice(i, 1);
  }

  // 대체 추첨 — draft.js 와 같은 가중치·같은 비복원·같은 rng.draft 스트림
  while (draft.cards.length < d.optionCount) {
    const pool = candidates(world).filter((c) => implemented(c) && !taken.has(c.key));
    if (pool.length === 0) break;
    const k = world.rng.draft.weighted(pool.map((c) => c.weight));
    if (k < 0) break;
    draft.cards.push(pool[k]);
    taken.add(pool[k].key);
  }

  // 최후 폴백 — 빈 드래프트 화면이 물리적으로 불가능해진다 (§11.1). v1.5: 코인 폐지 → 회복.
  while (draft.cards.length < d.optionCount) {
    draft.cards.push({ category: 'resupply', key: `resupply:${draft.cards.length}`,
      id: d.fallback.id, name: d.fallback.name, healPct: d.fallback.healPct, weight: 0 });
  }

  // ★ 피티 재계산 (§11.1 elementCardPity) — fill() 은 시임 **전** 카드로 elementPity 를
  //   정했는데, 위 대체 추첨이 미구현 newWeapon 자리에 elementLevel 카드를 넣을 수 있다. 그러면 최종
  //   카드셋엔 속성 카드가 있는데 피티가 리셋되지 않아 과다 계상된다(속성 카드 강제 빈발). draft.js 와
  //   같은 규칙으로 **최종 카드셋** 기준 다시 판정한다. 이 시임(12패밀리 완성 시 삭제)의 결정성 이탈 봉합.
  let sawElement = false;
  for (let i = 0; i < draft.cards.length; i += 1) {
    if (draft.cards[i].category === 'elementLevel') { sawElement = true; break; }
  }
  world.elementPity = sawElement ? 0 : draft.pityBefore + 1;
}

// ---------------------------------------------------------------------------
// 입력 (§5.7 — 폴링. 이벤트는 **키 상태를 갱신할 뿐** 게임에 직접 도달하지 않는다)
// ---------------------------------------------------------------------------
function makeKeyboard(rules) {
  const b = rules.input.bindings;
  // §5.7 — 게임이 사용하는 모든 키에 preventDefault (특히 Space, Tab, 방향키)
  const owned = new Set();
  const add = (v) => { if (Array.isArray(v)) v.forEach((k) => owned.add(k)); else owned.add(v); };
  Object.keys(b).forEach((k) => add(b[k]));

  const down = new Set();
  const masked = new Set();                   // §5.5 edgeTriggerOnStateEnter 로 무효화된 키
  const kb = {
    /** §5.1 — layout "code": event.code = **물리 키 위치**. 자판 배열이 바뀌어도 손가락이 안 바뀐다 */
    held(code) { return down.has(code) && !masked.has(code); },
    /** 상태 진입 시 이미 눌려 있던 키를 그 상태에서 무효화한다 (§5.5 — 죽는 순간 Space 연타 방어) */
    maskHeld() {
      masked.clear();
      down.forEach((c) => masked.add(c));
    },
    clear() { down.clear(); masked.clear(); },
  };

  window.addEventListener('keydown', (e) => {
    if (rules.input.layout !== 'code') throw new Error(`main: 미지의 input.layout "${rules.input.layout}" (§5.1)`);
    if (owned.has(e.code)) e.preventDefault();
    if (e.repeat) return;
    down.add(e.code);
  });
  window.addEventListener('keyup', (e) => {
    if (owned.has(e.code)) e.preventDefault();
    down.delete(e.code);
    masked.delete(e.code);                    // 떼면 마스크가 풀린다 = "다시 눌러야 유효"
  });
  return kb;
}

/** §5.7 — 이번 틱의 키 상태 스냅샷. ★ 매 고정 틱마다 새로 폴링한다 (헤드리스 재현성의 전제) */
function pollInput(kb, bindings, input) {
  const mv = bindings.move;                   // ["ArrowLeft","ArrowUp","ArrowRight","ArrowDown"]
  input.left = kb.held(mv[0]);
  input.up = kb.held(mv[1]);
  input.right = kb.held(mv[2]);
  input.down = kb.held(mv[3]);
  input.stanceNormal = kb.held(bindings.stanceNormal);
  input.stanceFire = kb.held(bindings.stanceFire);
  input.stanceWater = kb.held(bindings.stanceWater);
  input.stanceGrass = kb.held(bindings.stanceGrass);
  return input;
}

/** 상승 엣지 — 폴링 모델에서 "눌린 순간" = 직전 프레임 대비 변화 (§5.5 · §5.7) */
function makeEdge(kb) {
  const prev = new Set();
  return {
    pressed(code) {
      const now = kb.held(code);
      const was = prev.has(code);
      if (now) prev.add(code); else prev.delete(code);
      return now && !was;
    },
  };
}

// ---------------------------------------------------------------------------
// 캔버스 (§1.1 — 레터박스. **모든 게임 좌표는 논리 픽셀이며 스케일은 렌더에만 존재한다**)
// ---------------------------------------------------------------------------
function fitCanvas(canvas, view) {
  // ★ 정본 §1.1 은 CSS 스케일 s = min(vw/1280, vh/720) 와 setTransform 의 배율을 **같은 글자 s**
  //   로 적었으나, 백킹스토어가 1280·min(dpr,2) 이므로 setTransform 의 배율은 min(dpr,2) 여야 한다
  //   (CSS s 를 넣으면 논리 좌표가 두 번 축소된다). 표기 겹침이며 구조는 자명하다 — 보고 대상.
  const dpr = Math.min(window.devicePixelRatio || 1, view.maxDpr);
  const bw = Math.round(view.logicalW * dpr);
  const bh = Math.round(view.logicalH * dpr);
  if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }

  const s = Math.min(window.innerWidth / view.logicalW, window.innerHeight / view.logicalH);
  canvas.style.width = `${view.logicalW * s}px`;
  canvas.style.height = `${view.logicalH * s}px`;
  return dpr;
}

/** §1.1 — 뷰포트 < 최소 → 플레이 차단 + 안내(입력 무시). 데스크톱 키보드 전용 */
function viewportTooSmall(view) {
  return window.innerWidth < view.minViewportW || window.innerHeight < view.minViewportH;
}

// ---------------------------------------------------------------------------
// 오디오 (§7.10 — WebAudio 절차적 합성, 에셋 0KB · SFX 버스 하나 · 배속 무관)
//   ★ main.js 는 비순수이므로 WebAudio 사용 가능(core 는 소리를 모른다).
//   ★ 모든 큐는 시각 짝을 갖는다(§7.10) → 오디오 실패/무음이어도 게임은 100% 성립한다.
//     그러므로 여기의 모든 접근은 try/catch 로 감싸 소리가 게임을 절대 막지 않게 한다.
//   §7.7 히트 SFX: ×2 저역 크런치 / ×1 얇은 틱 / ×0.5 금속 "틴".
// ---------------------------------------------------------------------------
function makeAudio(rules) {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return null;                       // 오디오 인프라 없음 → 시각만 (호출부 null 가드)
  let ctx = null;
  let master = null;
  const sfxGain = rules.audio.busGain.sfx;                       // §7.10 busGain.sfx = 0.8
  const minInterval = 1 / rules.audio.cueRateLimitPerSec;        // §7.10 동일 큐 초당 상한
  const lastAt = { super: -1, neutral: -1, resist: -1, levelup: -1, hurt: -1, stance: -1 };
  let muted = false;                          // §5.5 OPTIONS — SFX 버스 뮤트/볼륨
  let vol = 1;
  function applyGain() { if (master !== null) master.gain.value = muted ? 0 : sfxGain * vol; }
  function ensure() {
    if (ctx === null) {
      ctx = new AC();
      master = ctx.createGain();
      master.gain.value = muted ? 0 : sfxGain * vol;
      master.connect(ctx.destination);
    }
    return ctx;
  }
  function env(o, g, t0, peak, dur) {                            // 공통 감쇠 엔벨로프
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(peak, t0 + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    o.connect(g); g.connect(master);
    o.start(t0); o.stop(t0 + dur + 0.02);
  }
  function crunch(t0) {                                          // ×2 — 짧고 두꺼운 저음
    const o = ctx.createOscillator();
    o.type = 'square';
    o.frequency.setValueAtTime(150, t0);
    o.frequency.exponentialRampToValueAtTime(58, t0 + 0.09);
    env(o, ctx.createGain(), t0, 0.9, 0.12);
  }
  function tick(t0) {                                            // ×1 — 얇은 틱
    const o = ctx.createOscillator();
    o.type = 'triangle';
    o.frequency.setValueAtTime(1900, t0);
    env(o, ctx.createGain(), t0, 0.22, 0.04);
  }
  function tin(t0) {                                             // ×0.5 — 비조화 3부분음 = 금속 튕김("틴")
    // ★ 시각 강화(두꺼운 회색 방패 + 본체 차폐 플래시)에 맞춰 살짝 더 또렷하게: 밝은 어택 트랜지언트
    //   1개 추가 + 기본 부분음 게인 소폭 상향. 여전히 비조화·고역·저역 없음 = "안 통함"(super 저역 크런치와 대비).
    const parts = [[2100, 0.38, 0.20], [3170, 0.20, 0.15], [4600, 0.12, 0.05]];
    for (let i = 0; i < parts.length; i += 1) {
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.setValueAtTime(parts[i][0], t0);
      env(o, ctx.createGain(), t0, parts[i][1], parts[i][2]);
    }
  }
  function chime(t0) {                                           // 레벨업 — 밝은 2음 상승(보상감)
    const freqs = [660, 990];
    for (let i = 0; i < freqs.length; i += 1) {
      const o = ctx.createOscillator();
      o.type = 'triangle';
      o.frequency.setValueAtTime(freqs[i], t0 + i * 0.06);
      env(o, ctx.createGain(), t0 + i * 0.06, 0.3, 0.12);
    }
  }
  function thud(t0) {                                            // 피격 — 낮고 둔탁한 경고
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.setValueAtTime(200, t0);
    o.frequency.exponentialRampToValueAtTime(70, t0 + 0.12);
    env(o, ctx.createGain(), t0, 0.5, 0.16);
  }
  function clickS(t0) {                                          // 스탠스 전환 — 얇은 클릭
    const o = ctx.createOscillator();
    o.type = 'square';
    o.frequency.setValueAtTime(880, t0);
    env(o, ctx.createGain(), t0, 0.12, 0.03);
  }
  return {
    resume() { try { ensure(); if (ctx.state === 'suspended') ctx.resume(); } catch (e) { /* 무음 폴백 */ } },
    setMuted(m) { muted = m; applyGain(); },
    toggleMuted() { muted = !muted; applyGain(); return muted; },
    isMuted() { return muted; },
    setVolume(v) { vol = v < 0 ? 0 : (v > 1 ? 1 : v); applyGain(); return vol; },
    getVolume() { return vol; },
    cue(tier) {
      try {
        if (muted) return;                                      // §5.5 뮤트
        ensure();
        if (ctx.state !== 'running') return;                    // 사용자 제스처 전 → 무음
        const now = ctx.currentTime;
        if (now - lastAt[tier] < minInterval) return;           // §7.10 rate limit
        lastAt[tier] = now;
        if (tier === 'super') crunch(now);
        else if (tier === 'resist') tin(now);
        else if (tier === 'levelup') chime(now);
        else if (tier === 'hurt') thud(now);
        else if (tier === 'stance') clickS(now);
        else tick(now);
      } catch (e) { /* 오디오 실패는 게임을 막지 않는다 */ }
    },
  };
}

/** §7.7 — 이번 스텝의 히트 이벤트에서 존재하는 tier 별로 큐를 1회 발화한다(각기 rate-limited). */
function playHitCues(audio, world) {
  if (audio === null) return;
  const h = world.hitFx;
  let sup = false; let neu = false; let res = false;
  for (let i = 0; i < h.count; i += 1) {
    const t = h.buf[i].tier;
    if (t === 'super') sup = true; else if (t === 'resist') res = true; else neu = true;
  }
  if (sup) audio.cue('super');
  if (res) audio.cue('resist');
  if (neu) audio.cue('neutral');
}

/** §7.10 — 플레이어 상태 변화를 SFX 로: 레벨업·피격·스탠스 전환. prev 는 프레임 간 유지. (v1.5: 코인 폐지) */
function playEventCues(audio, world, prev) {
  if (audio === null) return;
  const p = world.player;
  if (p.level > prev.level) audio.cue('levelup');
  if (p.hp < prev.hp) audio.cue('hurt');
  if (p.stance !== prev.stance) audio.cue('stance');
  prev.level = p.level; prev.hp = p.hp; prev.stance = p.stance;
}
const audioPrev = { level: 1, hp: 0, stance: '' };

// ---------------------------------------------------------------------------
// 부트
// ---------------------------------------------------------------------------
async function boot() {
  const data = await loadData();
  const rules = data.rules;
  const view = rules.view;

  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d', { alpha: false });
  const pal = resolvePalette(rules);
  const kb = makeKeyboard(rules);
  const edge = makeEdge(kb);
  const input = makeInput();
  // §7.10 — SFX. AudioContext 는 사용자 제스처 후에만 소리를 낸다 → 첫 키 입력에서 resume.
  const audio = makeAudio(rules);
  if (audio !== null) window.addEventListener('keydown', () => audio.resume());
  const baseTitle = document.title;

  // §9.1 — enemies.js · emitters.js 의 합성 계약을 정본이 인쇄하지 않았다 → state.js 가 주입으로 뒀다.
  //   적 스포너/이동 훅 + 적-공격 훅 + 런 디렉터 + 보스 훅을 주입한다.
  const HOOKS = { enemies, emitters, run: tickRun, boss: bossHook };

  // ★★ 배속 — 정본이 코드에 허용한 **유일한 거처** (§6.1 · §10.1) ★★
  //    난이도는 dt 를 바꾸지 않는다. **초당 소비 틱 수**만 바꾼다(tickDur).
  //    §6.5 — 이제 난이도는 DIFFICULTY 화면에서 고른다 → world·seed·tickDur 은 런마다 새로 만든다.
  let seed = 0;
  let world = null;
  let interp = null;
  let fx = null;
  let difficultyId = 'normal';
  let tickDur = 1000 / TICK_HZ;
  let bannerT = 0;                            // THEME_BANNER 잔여(실시간 ms)

  /** §6.5 RUN_START — 고른 난이도로 새 런을 조립한다(시드·world·보간·FX 전부 신규). */
  function startRun(diffId) {
    difficultyId = diffId;
    tickDur = 1000 / (TICK_HZ * data.meta.difficulty[diffId].speed);
    // §10.2 — 마스터 시드 = uint32. **비결정성이 들어오는 유일한 지점**. core 밖에서 만들어 주입.
    seed = (Date.now() ^ Math.floor(performance.now() * 1000)) >>> 0;
    world = createWorld({ data, seed, weapons, hooks: HOOKS });
    world.difficultyId = diffId;             // §11.3 점수 배율(tally)이 읽는다
    initRun(world);
    interp = makeInterp(world);
    fx = makeFx(world);
    document.title = `${baseTitle} — ${seedHex(seed)}`;
    enterBanner();                           // 스테이지 1 테마 배너 → PLAY
  }

  let state = 'TITLE';   // TITLE | DIFFICULTY | OPTIONS | THEME_BANNER | PLAY | DRAFT | PAUSE | RESULTS | TOO_SMALL  (v1.5: SHOP·DEATH 폐지)
  const DIFFS = Object.keys(data.meta.difficulty).filter((k) => data.meta.difficulty[k].speed !== undefined);
  let diffCursor = 0;
  let optionsFrom = 'TITLE';                 // OPTIONS 를 어디서 들어왔는가(나갈 때 복귀)
  let tooSmallReturn = 'TITLE';              // TOO_SMALL 에서 복귀할 상태
  let draft = null;
  let cursor = 0;
  let acc = 0;
  let last = performance.now();

  /** §6.5 THEME_BANNER — 스테이지 시작마다 themeBannerSec 동안 테마를 알린다(Space 스킵). */
  function enterBanner() {
    bannerT = data.meta.flow.themeBannerSec * 1000;   // 메뉴 배속 1 → 게임초 = 실초
    enter('THEME_BANNER');
    acc = 0;
  }

  function enter(next) {
    state = next;
    if (data.meta.flow.edgeTriggerOnStateEnter) kb.maskHeld();
  }

  // §5.7 — blur → 자동 일시정지 + acc = 0. 세이브 없는 런이 알트탭으로 죽지 않는다
  if (rules.input.pauseOnBlur) {
    window.addEventListener('blur', () => {
      kb.clear();
      acc = 0;
      if (state === 'PLAY') enter('PAUSE');
    });
  }

  function openDraftIfQueued() {
    if (world.draftQueue <= 0) return false;
    if (!data.meta.draft.pauseGame) return false;   // §6.4 — 드래프트는 게임 클럭을 멈춘다
    draft = buildDraft(world);
    dropUnimplementedWeapons(world, draft);         // ★ 1주차 시임 (위)
    cursor = 0;
    acc = 0;
    enter('DRAFT');
    return true;
  }

  // ★ v1.5 — 상점 입력(tickShop)은 폐지됐다: 경제 제거.

  function tickDraft(confirmE) {
    const b = rules.input.bindings;
    for (let i = 0; i < b.draftPick.length; i += 1) {
      if (edge.pressed(b.draftPick[i]) && i < draft.cards.length) { pick(i); return; }
    }
    if (edge.pressed(b.cursor[0])) cursor = (cursor + draft.cards.length - 1) % draft.cards.length;
    if (edge.pressed(b.cursor[1])) cursor = (cursor + 1) % draft.cards.length;
    if (confirmE) { pick(cursor); return; }             // ★ 상단에서 소비한 Enter 를 넘겨받는다
    // §5.2 — Escape 는 무시한다 (드래프트는 스킵 불가, 리롤 폐지). Q W E R · Space 도 죽은 키다
  }

  function pick(i) {
    applyCard(world, draft.cards[i]);         // §6.4 — 큐를 하나 줄인다
    draft = null;
    if (!openDraftIfQueued()) { enter('PLAY'); last = performance.now(); }
  }

  function frame(now) {
    requestAnimationFrame(frame);

    if (viewportTooSmall(view)) {
      if (state !== 'TOO_SMALL') { tooSmallReturn = state; state = 'TOO_SMALL'; }
    } else if (state === 'TOO_SMALL') {
      // 창이 다시 커지면 직전 상태로 복귀(런 중이면 PAUSE, 메뉴면 그 메뉴)
      enter(tooSmallReturn === 'PLAY' ? 'PAUSE' : tooSmallReturn); last = now; acc = 0;
    }

    const dpr = fitCanvas(canvas, view);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);   // 이후 모든 좌표 = 논리 픽셀 (§1.1)

    const elapsed = now - last;
    last = now;

    // ★ 엣지는 매 프레임 소비해 prev 를 갱신한다(상태가 어긋나지 않게). Space=시작/스킵, Enter=확정,
    //   Escape=일시정지/뒤로, O=옵션.
    const pauseEdge = edge.pressed(rules.input.bindings.pause);
    const startEdge = edge.pressed(rules.input.bindings.grab);       // Space
    const confirmEdge = edge.pressed(rules.input.bindings.confirm);  // Enter
    const optionsEdge = edge.pressed(rules.input.bindings.options);  // O
    const upEdge = edge.pressed(rules.input.bindings.cursor[2]);
    const downEdge = edge.pressed(rules.input.bindings.cursor[3]);

    // ── §6.5 메뉴(런 없음) ─────────────────────────────────────────
    if (state === 'TITLE') {
      if (startEdge) enter('DIFFICULTY');
      else if (optionsEdge) { optionsFrom = 'TITLE'; enter('OPTIONS'); }
      renderFrame();
      return;
    }
    if (state === 'DIFFICULTY') {
      if (upEdge) diffCursor = (diffCursor + DIFFS.length - 1) % DIFFS.length;
      if (downEdge) diffCursor = (diffCursor + 1) % DIFFS.length;
      if (confirmEdge) startRun(DIFFS[diffCursor]);   // → THEME_BANNER
      else if (pauseEdge) enter('TITLE');
      renderFrame();
      return;
    }
    if (state === 'OPTIONS') {
      if (audio !== null) {
        if (startEdge) audio.toggleMuted();                    // Space = 뮤트 토글
        if (upEdge) audio.setVolume(audio.getVolume() + 0.1);
        if (downEdge) audio.setVolume(audio.getVolume() - 0.1);
      }
      if (pauseEdge || optionsEdge) enter(optionsFrom);
      renderFrame();
      return;
    }

    // ── §6.5 THEME_BANNER — 실시간으로 카운트다운, Space 로 스킵 → PLAY ──
    if (state === 'THEME_BANNER') {
      bannerT -= Math.min(elapsed, rules.loop.maxFrameGapMs);   // §10.1 프레임 갭 클램프(첫 프레임 폭주 방지)
      if (startEdge || bannerT <= 0) { enter('PLAY'); last = now; acc = 0; }
      renderFrame();
      return;
    }

    // §5.7 — Escape = 일시정지 토글 (PLAY ↔ PAUSE).
    if (pauseEdge && state === 'PLAY') { enter('PAUSE'); acc = 0; }
    else if (pauseEdge && state === 'PAUSE') { enter('PLAY'); acc = 0; }
    else if (pauseEdge && state === 'RESULTS') { enter('TITLE'); }
    // §5.5 — PAUSE 에서 O = OPTIONS
    else if (optionsEdge && state === 'PAUSE') { optionsFrom = 'PAUSE'; enter('OPTIONS'); }
    // §6.5 — RESULTS 에서 Space = 같은 난이도 즉시 재시작
    if (startEdge && state === 'RESULTS') { startRun(difficultyId); }

    if (state === 'PLAY') {
      // §10.1 — 고정 타임스텝. maxFrameGapMs 로 프레임 갭을 자른다
      acc += Math.min(elapsed, rules.loop.maxFrameGapMs);
      let steps = 0;
      while (acc >= tickDur && steps < rules.loop.maxStepsPerFrame) {
        captureInterp(interp, world);         // §10.1 — 보간용 직전 위치. 렌더가 자기 것으로 들고 있는다
        step(world, pollInput(kb, rules.input.bindings, input), 1 / TICK_HZ);   // ★ dt 는 상수. speed 를 곱하지 않는다
        updateFx(fx, world, 1 / TICK_HZ);
        playHitCues(audio, world);            // §7.7 — 이 스텝의 히트 tier 를 SFX 로 (시각 짝, §7.10)
        playEventCues(audio, world, audioPrev); // §7.10 — 레벨업·피격·스탠스 SFX
        acc -= tickDur;
        steps += 1;
        // §11.4(v1.5) — 사망 = 즉시 결과. 컨티뉴 폐지 = 원데스=게임오버.
        if (world.over) { enter('RESULTS'); break; }
        // §6.5(v1.5) STAGE_CLEAR → 회복 → 바로 다음 스테이지 배너 (상점 폐지).
        if (world.run.phase === PHASE.STAGE_CLEAR) {
          applyStageClearHeal(world);
          advanceStage(world);
          enterBanner();
          break;
        }
        if (world.draftQueue > 0 && openDraftIfQueued()) break;
      }
      // §10.1 — 나선형 죽음 방지: 남은 시간 폐기 (빨리감기 금지)
      if (acc >= tickDur) acc = 0;
    } else {
      acc = 0;
      if (state === 'DRAFT') tickDraft(confirmEdge);
      else if (state === 'TOO_SMALL') { /* 입력 무시 (§1.1) */ }
      // PAUSE 재개는 위의 Escape 토글이 처리한다 (§5.7)
    }

    renderFrame();
  }

  /** 모든 상태의 렌더. 메뉴 상태(TITLE/DIFFICULTY/OPTIONS)는 world 가 없다 → 메뉴 배경을 그린다. */
  function renderFrame() {
    if (state === 'TOO_SMALL') {
      menuBg();
      menuBanner('창이 너무 작습니다',
        `최소 ${view.minViewportW} × ${view.minViewportH} — 데스크톱 키보드 전용`);
      return;
    }
    if (world === null) {   // TITLE / DIFFICULTY / OPTIONS
      menuBg();
      if (state === 'TITLE') drawTitleScreen();
      else if (state === 'DIFFICULTY') drawDifficultyScreen();
      else if (state === 'OPTIONS') drawOptionsScreen();
      return;
    }

    const alpha = state === 'PLAY' ? acc / tickDur : 0;    // §10.1 — 위치 lerp 만. 로직 금지
    drawWorld(ctx, world, pal, fx, interp, alpha);
    drawPanels(ctx, world, pal);
    if (state === 'THEME_BANNER') drawThemeBanner();
    if (state === 'DRAFT') drawDraft(ctx, world, pal, draft, cursor);
    if (state === 'PAUSE') banner(ctx, data, pal, '일시정지', '[Esc] 재개   ·   [O] 옵션');
    if (state === 'OPTIONS') drawOptionsScreen();          // PAUSE→OPTIONS 는 world 위에 겹친다
    // §11.3 — 결과 화면(죽어도 집계된다). 내역 + 총점.
    if (state === 'RESULTS') drawResults(ctx, world, pal, tally(world), `시드 ${seedHex(seed)}`);
  }

  // ── §6.5 메뉴 렌더 (world 없이도 그린다) ──────────────────────────────
  function menuBg() {
    ctx.fillStyle = pal.hud.panelBg;
    ctx.fillRect(0, 0, view.logicalW, view.logicalH);
  }
  function mText(text, y, sizePx, color, weight, align) {
    ctx.textAlign = align || 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.font = `${weight || 400} ${sizePx}px ${rules.visual.text.family}`;
    ctx.fillText(text, view.logicalW / 2, y);
  }
  function menuBanner(title, sub) {
    const h = rules.hud;
    mText(title, view.logicalH / 2 - 16, h.fontHeroPx, pal.hud.textPrimary, 800);
    mText(sub, view.logicalH / 2 + 24, h.fontBodyPx, pal.hud.textDim, 400);
  }
  function drawTitleScreen() {
    const h = rules.hud;
    mText('NAN 2026', view.logicalH / 2 - 70, h.fontHeroPx, pal.hud.textPrimary, 800);
    mText('속성 스탠스 슈팅', view.logicalH / 2 - 24, h.fontLargePx, pal.hud.textPrimary, 700);
    mText('[Space] 시작        [O] 옵션', view.logicalH / 2 + 48, h.fontBodyPx, pal.hud.textDim, 400);
    mText('QWER 스탠스 · 상성 ×2 · 6 스테이지 · 엔드리스 없음',
      view.logicalH / 2 + 84, h.fontSmallPx, pal.hud.textDim, 400);
  }
  const DIFF_LABEL = { normal: '노멀', hard: '하드', hell: '헬', disaster: '디재스터' };
  function drawDifficultyScreen() {
    const h = rules.hud;
    mText('난이도 선택', view.logicalH / 2 - 110, h.fontLargePx, pal.hud.textPrimary, 800);
    for (let i = 0; i < DIFFS.length; i += 1) {
      const id = DIFFS[i];
      const d = data.meta.difficulty[id];
      const sel = i === diffCursor;
      const y = view.logicalH / 2 - 40 + i * 40;
      const label = `${sel ? '▶ ' : '   '}${DIFF_LABEL[id] || id}   ×${d.speed} 속도 · ×${d.scoreMul} 점수`;
      mText(label, y, h.fontBodyPx, sel ? pal.hud.textPrimary : pal.hud.textDim, sel ? 700 : 400);
    }
    mText('[↑↓] 선택   [Enter] 시작   [Esc] 뒤로',
      view.logicalH / 2 + 120, h.fontSmallPx, pal.hud.textDim, 400);
  }
  function drawOptionsScreen() {
    const h = rules.hud;
    if (world !== null) { ctx.save(); ctx.fillStyle = rgba(pal.threat.outline, 0.72); ctx.fillRect(view.arena.x, 0, view.arena.w, view.logicalH); ctx.restore(); }
    mText('옵션', view.logicalH / 2 - 70, h.fontLargePx, pal.hud.textPrimary, 800);
    if (audio === null) {
      mText('오디오 인프라 없음 (시각 전용)', view.logicalH / 2 - 12, h.fontBodyPx, pal.hud.textDim, 400);
    } else {
      const muted = audio.isMuted();
      const vol = Math.round(audio.getVolume() * 100);
      mText(`효과음   ${muted ? '음소거' : `${vol}%`}`, view.logicalH / 2 - 12,
        h.fontBodyPx, muted ? pal.hud.textDim : pal.hud.textPrimary, 700);
      mText('[Space] 음소거   [↑↓] 볼륨', view.logicalH / 2 + 28, h.fontSmallPx, pal.hud.textDim, 400);
    }
    mText('[Esc] 뒤로', view.logicalH / 2 + 60, h.fontSmallPx, pal.hud.textDim, 400);
  }
  function drawThemeBanner() {
    const h = rules.hud;
    const st = stageEntry(world);
    const n = world.run.stageIndex + 1;
    const total = world.run.order.length;
    ctx.save();
    ctx.fillStyle = rgba(pal.threat.outline, 0.55);
    ctx.fillRect(view.arena.x, view.logicalH / 2 - 60, view.arena.w, 120);
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = pal.hud.textDim;
    ctx.font = `700 ${h.fontBodyPx}px ${rules.visual.text.family}`;
    ctx.fillText(`스테이지 ${n} / ${total}`, view.arena.x + view.arena.w / 2, view.logicalH / 2 - 20);
    ctx.fillStyle = pal.hud.textPrimary;
    ctx.font = `800 ${h.fontHeroPx}px ${rules.visual.text.family}`;
    ctx.fillText(st.name, view.arena.x + view.arena.w / 2, view.logicalH / 2 + 18);
    ctx.restore();
  }

  requestAnimationFrame(frame);
}

function banner(ctx, data, pal, title, sub) {
  const v = data.rules.view;
  const h = data.rules.hud;
  ctx.save();
  ctx.fillStyle = rgba(pal.threat.outline, 0.72);      // §7.2 — 색의 유일한 거처는 palette 다
  ctx.fillRect(v.arena.x, 0, v.arena.w, v.logicalH);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = pal.hud.textPrimary;
  ctx.font = `800 ${h.fontHeroPx}px ${data.rules.visual.text.family}`;
  ctx.fillText(title, v.arena.x + v.arena.w / 2, v.logicalH / 2 - 16);
  ctx.fillStyle = pal.hud.textDim;
  ctx.font = `400 ${h.fontBodyPx}px ${data.rules.visual.text.family}`;
  ctx.fillText(sub, v.arena.x + v.arena.w / 2, v.logicalH / 2 + 24);
  ctx.restore();
}

boot().catch(fatal);
