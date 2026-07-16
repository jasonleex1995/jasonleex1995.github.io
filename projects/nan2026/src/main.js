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
import { buildDraft, rerollDraft, applyCard, candidates } from './core/draft.js';
import { weapons } from './core/weapons/index.js';
import { seedHex } from './core/rng.js';
import { resolvePalette, drawWorld, makeInterp, captureInterp, makeFx, updateFx, rgba } from './render/draw.js';
import { drawPanels, drawDraft } from './render/hud.js';

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

  // 최후 폴백 — 빈 드래프트 화면이 물리적으로 불가능해진다 (§11.1)
  while (draft.cards.length < d.optionCount) {
    draft.cards.push({ category: 'resupply', key: `resupply:${draft.cards.length}`,
      id: d.fallback.id, name: d.fallback.name, coins: d.fallback.coins, weight: 0 });
  }

  // ★ 피티 재계산 (§11.1 elementCardPity) — fill()/rerollDraft 는 시임 **전** 카드로 elementPity 를
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
// 부트
// ---------------------------------------------------------------------------
async function boot() {
  const data = await loadData();
  const rules = data.rules;
  const view = rules.view;

  // §10.2 — 마스터 시드 = uint32. **비결정성이 들어오는 유일한 지점**이며 core 바깥에서 만들어 주입한다
  const seed = (Date.now() ^ Math.floor(performance.now() * 1000)) >>> 0;

  // §9.1 — enemies.js · emitters.js 의 합성 계약을 정본이 인쇄하지 않았다 → state.js 가 주입으로 뒀다.
  //   1주차에는 스포너·이미터가 없다 (스테이지는 범위 밖) → null. step() 은 적의 등속 적분만 한다.
  const world = createWorld({ data, seed, weapons, hooks: { enemies: null, emitters: null } });

  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d', { alpha: false });
  const pal = resolvePalette(rules);
  const interp = makeInterp(world);
  const fx = makeFx(world);
  const kb = makeKeyboard(rules);
  const edge = makeEdge(kb);
  const input = makeInput();

  document.title = `${document.title} — ${seedHex(seed)}`;

  // ★★ 배속 — 정본이 코드에 허용한 **유일한 거처** (§6.1 · §10.1) ★★
  //    난이도는 dt 를 바꾸지 않는다. **초당 소비 틱 수**만 바꾼다.
  //    src/core/** 는 speed 를 모르며, 아래 tickDur 말고 speed 가 사는 줄이 이 저장소에 없다.
  const difficultyId = 'normal';              // §6.1 — 난이도 선택 화면은 1주차 범위 밖
  const speed = data.meta.difficulty[difficultyId].speed;
  const tickDur = 1000 / (TICK_HZ * speed);   // 실제 ms

  let state = 'PLAY';                         // PLAY | DRAFT | PAUSE | OVER | TOO_SMALL
  let draft = null;
  let cursor = 0;
  let acc = 0;
  let last = performance.now();

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

  function tickDraft() {
    const b = rules.input.bindings;
    for (let i = 0; i < b.draftPick.length; i += 1) {
      if (edge.pressed(b.draftPick[i]) && i < draft.cards.length) { pick(i); return; }
    }
    if (edge.pressed(b.cursor[0])) cursor = (cursor + draft.cards.length - 1) % draft.cards.length;
    if (edge.pressed(b.cursor[1])) cursor = (cursor + 1) % draft.cards.length;
    if (edge.pressed(b.confirm)) { pick(cursor); return; }
    if (edge.pressed(b.reroll)) {
      if (rerollDraft(world, draft)) { dropUnimplementedWeapons(world, draft); cursor = 0; }
    }
    // §5.2 — Escape 는 무시한다 (드래프트는 스킵 불가). Q W E R · Space · Shift 도 죽은 키다
  }

  function pick(i) {
    applyCard(world, draft.cards[i]);         // §6.4 — 큐를 하나 줄인다
    draft = null;
    if (!openDraftIfQueued()) { enter('PLAY'); last = performance.now(); }
  }

  function frame(now) {
    requestAnimationFrame(frame);

    if (viewportTooSmall(view)) { state = state === 'TOO_SMALL' ? state : 'TOO_SMALL'; }
    else if (state === 'TOO_SMALL') { enter('PAUSE'); last = now; acc = 0; }

    const dpr = fitCanvas(canvas, view);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);   // 이후 모든 좌표 = 논리 픽셀 (§1.1)

    const elapsed = now - last;
    last = now;

    if (state === 'PLAY') {
      // §10.1 — 고정 타임스텝. maxFrameGapMs 로 프레임 갭을 자른다
      acc += Math.min(elapsed, rules.loop.maxFrameGapMs);
      let steps = 0;
      while (acc >= tickDur && steps < rules.loop.maxStepsPerFrame) {
        captureInterp(interp, world);         // §10.1 — 보간용 직전 위치. 렌더가 자기 것으로 들고 있는다
        step(world, pollInput(kb, rules.input.bindings, input), 1 / TICK_HZ);   // ★ dt 는 상수. speed 를 곱하지 않는다
        updateFx(fx, world, 1 / TICK_HZ);
        acc -= tickDur;
        steps += 1;
        if (world.over) { enter('OVER'); break; }
        if (world.draftQueue > 0 && openDraftIfQueued()) break;
      }
      // §10.1 — 나선형 죽음 방지: 남은 시간 폐기 (빨리감기 금지)
      if (acc >= tickDur) acc = 0;
    } else {
      acc = 0;
      if (state === 'DRAFT') tickDraft();
      else if (state === 'PAUSE' && edge.pressed(rules.input.bindings.pause)) { enter('PLAY'); }
      else if (state === 'TOO_SMALL') { /* 입력 무시 (§1.1) */ }
    }

    // ---- 렌더 ------------------------------------------------------------
    const alpha = state === 'PLAY' ? acc / tickDur : 0;    // §10.1 — 위치 lerp 만. 로직 금지
    drawWorld(ctx, world, pal, fx, interp, alpha);
    drawPanels(ctx, world, pal);
    if (state === 'DRAFT') drawDraft(ctx, world, pal, draft, cursor);
    if (state === 'PAUSE') banner(ctx, world, pal, '일시정지', '[Escape] 재개');
    if (state === 'OVER') banner(ctx, world, pal, 'GAME OVER', `시드 ${seedHex(seed)}`);
    if (state === 'TOO_SMALL') {
      banner(ctx, world, pal, '창이 너무 작습니다',
        `최소 ${view.minViewportW} × ${view.minViewportH} — 데스크톱 키보드 전용`);
    }
  }

  requestAnimationFrame(frame);
}

function banner(ctx, world, pal, title, sub) {
  const v = world.data.rules.view;
  const h = world.data.rules.hud;
  ctx.save();
  ctx.fillStyle = rgba(pal.threat.outline, 0.72);      // §7.2 — 색의 유일한 거처는 palette 다
  ctx.fillRect(v.arena.x, 0, v.arena.w, v.logicalH);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = pal.hud.textPrimary;
  ctx.font = `800 ${h.fontHeroPx}px ${world.data.rules.visual.text.family}`;
  ctx.fillText(title, v.arena.x + v.arena.w / 2, v.logicalH / 2 - 16);
  ctx.fillStyle = pal.hud.textDim;
  ctx.font = `400 ${h.fontBodyPx}px ${world.data.rules.visual.text.family}`;
  ctx.fillText(sub, v.arena.x + v.arena.w / 2, v.logicalH / 2 + 24);
  ctx.restore();
}

boot().catch(fatal);
