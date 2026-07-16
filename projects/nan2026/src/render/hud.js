/**
 * src/render/hud.js — 최소 HUD (브라우저 전용)
 *
 * 정본 v1.4 구현 절:
 *   §1.2   아레나 오버레이 띠 — 하단 A(672~696) HP 세그먼트 · 하단 B(696~720) XP
 *          판 알파 visual.band.plateAlpha 0.30 / **내용 불투명** visual.band.contentOpaque
 *   §2.1   칸당 = hpMax / hud.hpBarSegCount (★ 칸 수는 5 고정, 칸당이 파생값)
 *   §4.2   투자 상한 2종 — elementCapPerElement 4 · elementCapTotal 6
 *   §4.3   부여 — 슬롯 1..N. ★ 이 규칙을 화면에서 읽을 수 있게 하는 것이 이 파일의 존재 이유
 *   §5.1   키맵 Q/W/E/R
 *   §6.4   LV UP ×N 배지 (드래프트 큐)
 *   §7.2   팔레트 = rules.palette. ★ 이 파일에 색 리터럴이 없다
 *   §7.12.4-① 저체력 HP 바의 호박 (1Hz → 위독 2Hz). 호박 5곳 목록 안
 *   §9.4.1 hud 스코프 — hpBarSegCount · xpBarH · font*Px · panelPadPx · keycapBoxPx
 *          · showElementBudget · elementMatrixInPanel
 *   §11.1  드래프트 3택 (카드 표기 — 속성 카드의 부여 프리뷰 포함)
 *   §12.3  레이어 1(띠) · 12(HUD 패널)
 *
 * ★ 정본 §17 은 **패널 내부 레이아웃**을 02 섹션에 위임했고 그 문서는 이 저장소에 없다
 *   → 패널 안의 px 좌표는 이 파일이 임시로 소유한다 (보고 대상). 띠(§1.2)·칸 수·상한은 정본 값이다.
 */

import { rgba, glyphPath } from './draw.js';

const KEYCAP = { normal: 'Q', fire: 'W', water: 'E', grass: 'R' };

function font(world, px, weight) {
  return `${weight} ${px}px ${world.data.rules.visual.text.family}`;
}

/** §9.4.3 visual.text.outlinePx + §7.2 — 텍스트 아웃라인 색 = palette.threat.outline (색의 유일한 거처) */
function text(ctx, world, pal, s, x, y, px, color, align, weight) {
  ctx.font = font(world, px, weight === undefined ? 400 : weight);
  ctx.textAlign = align === undefined ? 'left' : align;
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.lineWidth = world.data.rules.visual.text.outlinePx;
  ctx.strokeStyle = pal.threat.outline;
  ctx.strokeText(s, x, y);
  ctx.fillStyle = color;
  ctx.fillText(s, x, y);
}

// ---------------------------------------------------------------------------
// 레이어 1 — 아레나 오버레이 띠 (§1.2). draw.js 가 아레나 클립 안에서 부른다
// ---------------------------------------------------------------------------
/**
 * ★ 1주차에 그리는 띠 = 하단 A(HP) · 하단 B(XP) 둘.
 *   상단 띠(0~40 보스 타이머·배속 배지·개체 이름)와 코어 HP 바(40~48)는 **스테이지·보스가
 *   1주차 범위 밖**이므로 그리지 않는다 — §1.2 「없을 때: 바 자체를 그리지 않는다(빈 트랙 금지)」.
 */
export function drawArenaBands(ctx, world, pal) {
  const v = world.data.rules.view;
  const h = world.data.rules.hud;
  const vb = world.data.rules.visual.band;
  const rp = world.data.rules.player;
  const p = world.player;
  const a = v.arena;

  const hpY = a.y + a.h - v.bandHpH - v.bandXpH;      // 672
  const xpY = a.y + a.h - v.bandXpH;                  // 696
  const pad = 8;

  // 판 — 알파 0.30 (내용은 아래에서 불투명하게 그린다)
  ctx.fillStyle = rgba(pal.hud.panelBg, vb.plateAlpha);
  ctx.fillRect(a.x, hpY, a.w, v.bandHpH + v.bandXpH);

  ctx.save();
  if (vb.contentOpaque) ctx.globalAlpha = 1.0;

  // ---- 하단 A — HP 세그먼트 바 -------------------------------------------
  // §2.1 — 칸 수는 hpBarSegCount(5) 고정. **칸당 = hpMax / 5** 는 파생값이다
  const segs = h.hpBarSegCount;
  const perSeg = p.hpMax / segs;
  const labelW = 78;
  const trackW = a.w - pad * 2 - labelW;
  const segW = (trackW - h.hpBarSegGapPx * (segs - 1)) / segs;
  const barH = v.bandHpH - 10;
  const barY = hpY + 5;

  // §7.12.4-① — 저체력이면 호박. 위독(0.15)에서 주파수가 2배가 된다 (같은 사건의 두 번째 표면)
  const ratio = p.hp / p.hpMax;
  let fill = pal.hud.hpFill;
  if (ratio <= rp.lowHpThreshold) {
    const hz = ratio <= rp.lowHpCriticalThreshold ? 2 : 1;
    const pulse = 0.5 + 0.5 * Math.sin(world.time * hz * Math.PI * 2);
    fill = rgba(pal.status.band, 0.55 + 0.45 * pulse);
  }

  for (let i = 0; i < segs; i += 1) {
    const x = a.x + pad + i * (segW + h.hpBarSegGapPx);
    ctx.fillStyle = rgba(pal.hud.panelRule, 0.85);
    ctx.fillRect(x, barY, segW, barH);
    const inSeg = Math.max(0, Math.min(perSeg, p.hp - i * perSeg)) / perSeg;
    if (inSeg > 0) {
      ctx.fillStyle = fill;
      ctx.fillRect(x, barY, segW * inSeg, barH);
    }
  }
  text(ctx, world, pal, `${Math.ceil(p.hp)}/${Math.round(p.hpMax)}`,
    a.x + a.w - pad, hpY + v.bandHpH / 2, h.fontBodyPx, pal.hud.textPrimary, 'right', 600);

  // ---- 하단 B — XP 바 + Lv + LV UP ×N -----------------------------------
  const lvW = 62;
  const badgeW = world.draftQueue > 0 ? 86 : 0;
  const xTrack = a.x + pad + lvW;
  const wTrack = a.w - pad * 2 - lvW - badgeW;
  const xbY = xpY + (v.bandXpH - h.xpBarH) / 2;

  text(ctx, world, pal, `Lv.${p.level}`, a.x + pad, xpY + v.bandXpH / 2,
    h.fontSmallPx, pal.hud.textPrimary, 'left', 600);
  ctx.fillStyle = rgba(pal.hud.panelRule, 0.85);
  ctx.fillRect(xTrack, xbY, wTrack, h.xpBarH);
  ctx.fillStyle = pal.pickup.xp;                       // XP 바 = XP 픽업과 같은 색 (새 색 0)
  ctx.fillRect(xTrack, xbY, wTrack * Math.max(0, Math.min(1, p.xp / p.xpToNext)), h.xpBarH);

  // §6.4 — 동시 다중 레벨업은 순차 드래프트다. 큐가 화면에 있어야 "몇 번 남았나"가 읽힌다
  if (world.draftQueue > 0) {
    text(ctx, world, pal, `LV UP ×${world.draftQueue}`, a.x + a.w - pad, xpY + v.bandXpH / 2,
      h.fontSmallPx, pal.element.normal, 'right', 700);
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// 레이어 12 — HUD 패널 (§1.1 — 좌 0~350 · 우 930~1280. 아레나를 침범하지 않는다)
// ---------------------------------------------------------------------------
export function drawPanels(ctx, world, pal) {
  const v = world.data.rules.view;
  ctx.fillStyle = pal.hud.panelBg;
  ctx.fillRect(0, 0, v.panelLeftW, v.logicalH);
  ctx.fillRect(v.arena.x + v.arena.w, 0, v.panelRightW, v.logicalH);
  ctx.fillStyle = pal.hud.panelRule;
  ctx.fillRect(v.panelLeftW - 1, 0, 1, v.logicalH);
  ctx.fillRect(v.arena.x + v.arena.w, 0, 1, v.logicalH);

  drawLeftPanel(ctx, world, pal);
  drawRightPanel(ctx, world, pal);
}

/** 좌 패널 — §9.4.1 elementMatrixInPanel: 상성표 4×4 상시 = **정보 루프의 폐쇄** */
function drawLeftPanel(ctx, world, pal) {
  const h = world.data.rules.hud;
  const pad = h.panelPadPx;
  const order = world.data.elements.order;
  const matrix = world.data.elements.matrix;
  let y = pad + 10;

  text(ctx, world, pal, '상성표', pad, y, h.fontMediumPx, pal.hud.textPrimary, 'left', 700);
  y += 24;
  text(ctx, world, pal, '세로 = 내 스탠스 / 가로 = 적', pad, y, h.fontSmallPx, pal.hud.textDim, 'left');
  y += 22;

  if (!h.elementMatrixInPanel) return;

  const cell = 46;
  const x0 = pad + cell;
  const y0 = y + cell;

  for (let i = 0; i < order.length; i += 1) {                 // 헤더 (가로 = 방어 속성)
    const e = order[i];
    const cx = x0 + i * cell + cell / 2;
    glyphPath(ctx, e, cx, y + cell / 2, 8);
    ctx.fillStyle = pal.element[e];
    ctx.fill();
  }
  for (let r = 0; r < order.length; r += 1) {
    const atk = order[r];
    const cy = y0 + r * cell + cell / 2;
    glyphPath(ctx, atk, pad + cell / 2, cy, 8);              // 헤더 (세로 = 공격 속성 = 내 스탠스)
    ctx.fillStyle = pal.element[atk];
    ctx.fill();
    // ★ 현재 스탠스 행을 강조 — "지금 내가 어느 줄에 서 있는가"가 상성표의 유일한 사용법이다
    if (atk === world.player.stance) {
      ctx.fillStyle = rgba(pal.element[atk], 0.12);
      ctx.fillRect(pad, y0 + r * cell, cell * 5, cell);
    }
    for (let c = 0; c < order.length; c += 1) {
      const m = matrix[atk][order[c]];
      const cx = x0 + c * cell + cell / 2;
      const label = m > 1 ? '×2' : m < 1 ? '×½' : '×1';
      // §7.7 — ×2 는 속성색 / ×1 은 은색 / ×0.5 는 회색. 히트 피드백과 같은 3단 어휘
      const col = m > 1 ? pal.element[atk] : m < 1 ? pal.neutralGray : pal.element.normal;
      text(ctx, world, pal, label, cx, cy, m > 1 ? h.fontBodyPx : h.fontSmallPx, col, 'center', m > 1 ? 700 : 400);
    }
  }
  ctx.strokeStyle = pal.hud.panelRule;
  ctx.lineWidth = 1;
  for (let i = 0; i <= order.length; i += 1) {
    ctx.beginPath(); ctx.moveTo(x0, y0 + i * cell); ctx.lineTo(x0 + cell * 4, y0 + i * cell); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x0 + i * cell, y0); ctx.lineTo(x0 + i * cell, y0 + cell * 4); ctx.stroke();
  }

  // §5.6 — 상성 니모닉. 정본이 LOCKED 로 확정한 것을 화면이 말한다 (온보딩 비용 0)
  text(ctx, world, pal, '적 속성 키의 오른쪽 이웃이 정답 키',
    pad, y0 + cell * 4 + 24, h.fontSmallPx, pal.hud.textDim, 'left');
  text(ctx, world, pal, 'W(불) → E · E(물) → R · R(풀) → W',
    pad, y0 + cell * 4 + 44, h.fontSmallPx, pal.hud.textDim, 'left');
}

/** 우 패널 — 스탠스 키캡 · 속성 투자 pip · 무기 4슬롯 + 부여 상태 */
function drawRightPanel(ctx, world, pal) {
  const v = world.data.rules.view;
  const h = world.data.rules.hud;
  const rp = world.data.rules.player;
  const p = world.player;
  const x = v.arena.x + v.arena.w + h.panelPadPx;
  const w = v.panelRightW - h.panelPadPx * 2;
  let y = h.panelPadPx + 10;

  // ---- 스탠스 (§5.1 QWER) ------------------------------------------------
  text(ctx, world, pal, '스탠스', x, y, h.fontMediumPx, pal.hud.textPrimary, 'left', 700);
  y += 26;
  const order = world.data.elements.order;
  const box = h.keycapBoxPx;
  for (let i = 0; i < order.length; i += 1) {
    const e = order[i];
    const bx = x + i * (box + 10);
    const active = p.stance === e;
    // I-2 — 투자 0인 키는 은색. 이 키를 눌러도 아무것도 부여되지 않는 것이 참이다 (§7.12.7 과 같은 규칙)
    const invested = e === 'normal' ? true : p.invest[e] > 0;
    const col = invested ? pal.element[e] : pal.element.normal;
    ctx.fillStyle = active ? rgba(col, 0.22) : rgba(pal.hud.panelRule, 0.5);
    ctx.fillRect(bx, y, box, box);
    ctx.lineWidth = active ? 2 : 1;
    ctx.strokeStyle = active ? col : pal.hud.panelRule;
    ctx.strokeRect(bx, y, box, box);
    text(ctx, world, pal, KEYCAP[e], bx + box / 2, y + box / 2, h.fontBodyPx,
      active ? col : pal.hud.textDim, 'center', 700);
    glyphPath(ctx, e, bx + box / 2, y + box + 12, 5);
    ctx.fillStyle = invested ? pal.element[e] : rgba(pal.element.normal, 0.35);
    ctx.fill();
  }
  y += box + 30;

  // ---- 속성 투자 pip (§4.2 상한 2종) -------------------------------------
  let head = '속성 투자';
  if (h.showElementBudget) {
    let total = 0;
    for (let i = 0; i < world.data.elements.investable.length; i += 1) {
      total += p.invest[world.data.elements.investable[i]];
    }
    head += `  ${total}/${rp.elementCapTotal}`;
  }
  text(ctx, world, pal, head, x, y, h.fontMediumPx, pal.hud.textPrimary, 'left', 700);
  y += 24;

  const inv = world.data.elements.investable;
  for (let i = 0; i < inv.length; i += 1) {
    const e = inv[i];
    glyphPath(ctx, e, x + 7, y + 8, 6);
    ctx.fillStyle = pal.element[e];
    ctx.fill();
    for (let k = 0; k < rp.elementCapPerElement; k += 1) {
      const px = x + 30 + k * 16;
      const on = k < p.invest[e];
      ctx.beginPath();
      ctx.arc(px, y + 8, 5, 0, Math.PI * 2);
      ctx.fillStyle = on ? pal.element[e] : rgba(pal.hud.panelRule, 0.9);
      ctx.fill();
    }
    // ★ §4.3 을 문장으로 — "투자 N = 슬롯 1..N 부여". 투자 0 = 부여 없음 (I-2: 화면이 참을 말한다)
    const lv = p.invest[e];
    text(ctx, world, pal, lv === 0 ? '부여 없음' : lv === 1 ? '슬롯 1 부여' : `슬롯 1~${lv} 부여`,
      x + 30 + rp.elementCapPerElement * 16 + 8, y + 8,
      h.fontSmallPx, lv > 0 ? pal.hud.textDim : rgba(pal.hud.textDim, 0.4), 'left');
    y += 22;
  }
  y += 14;

  // ---- 무기 슬롯 4칸 + 부여 상태 (§4.3 — 좌→우 = 슬롯 1..4 = 부여 우선순위) --
  text(ctx, world, pal, '무기 슬롯', x, y, h.fontMediumPx, pal.hud.textPrimary, 'left', 700);
  y += 24;
  for (let i = 0; i < world.slots.length; i += 1) {
    const s = world.slots[i];
    const rowH = 34;
    const imbued = s.stampElement !== 'normal';
    ctx.fillStyle = imbued ? rgba(pal.element[s.stampElement], 0.14) : rgba(pal.hud.panelRule, 0.35);
    ctx.fillRect(x, y, w, rowH - 4);
    ctx.lineWidth = 1;
    ctx.strokeStyle = imbued ? pal.element[s.stampElement] : pal.hud.panelRule;
    ctx.strokeRect(x, y, w, rowH - 4);

    text(ctx, world, pal, `${i + 1}`, x + 10, y + (rowH - 4) / 2, h.fontSmallPx, pal.hud.textDim, 'left', 700);
    if (s.weaponId === null) {
      text(ctx, world, pal, '빈 슬롯', x + 26, y + (rowH - 4) / 2, h.fontBodyPx, rgba(pal.hud.textDim, 0.5), 'left');
    } else {
      const def = world.weaponDefs[s.family];
      const name = s.evolved ? def.evolution.name : def.name;
      text(ctx, world, pal, name, x + 26, y + (rowH - 4) / 2, h.fontBodyPx, pal.hud.textPrimary, 'left', 600);
      text(ctx, world, pal, s.evolved ? 'EVO' : `Lv.${s.level}`, x + w - 46, y + (rowH - 4) / 2,
        h.fontSmallPx, s.evolved ? pal.element.normal : pal.hud.textDim, 'right', 600);
    }
    // 부여 칩 — 기체의 슬롯 스트립(§7.5 ②)과 **같은 어휘**. 두 표면이 같은 것을 말한다
    const cx = x + w - 18;
    const cy = y + (rowH - 4) / 2;
    if (imbued) { glyphPath(ctx, s.stampElement, cx, cy, 6); ctx.fillStyle = pal.element[s.stampElement]; ctx.fill(); }
    else {
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fillStyle = s.weaponId === null ? rgba(pal.element.normal, 0.2) : pal.element.normal;
      ctx.fill();
    }
    y += rowH;
  }
}

// ---------------------------------------------------------------------------
// 드래프트 3택 (§11.1 · §5.2)
//   ★ 정본 §9.1 은 드래프트 **화면**의 거처를 `src/ui/` 로 확정했다. 1주차 최소 판본을
//     여기에 둔 것은 임시이며 2주차에 `src/ui/draft.js` 로 이사한다 (보고 대상).
// ---------------------------------------------------------------------------
export function drawDraft(ctx, world, pal, draft, cursor) {
  const v = world.data.rules.view;
  const h = world.data.rules.hud;
  const d = world.data.meta.draft;

  ctx.fillStyle = rgba(pal.threat.outline, 0.78);
  ctx.fillRect(0, 0, v.logicalW, v.logicalH);

  text(ctx, world, pal, 'LEVEL UP', v.logicalW / 2, 92, h.fontHeroPx, pal.hud.textPrimary, 'center', 800);
  text(ctx, world, pal, `Lv.${world.player.level}  ·  1 / 2 / 3 선택   ←→ 커서   Enter 확정`,
    v.logicalW / 2, 132, h.fontBodyPx, pal.hud.textDim, 'center');
  if (world.draftQueue > 1) {
    text(ctx, world, pal, `대기 중인 레벨업 ×${world.draftQueue - 1}`, v.logicalW / 2, 156,
      h.fontSmallPx, pal.element.normal, 'center', 600);
  }

  const cw = 300;
  const ch = 380;
  const gap = 28;
  const total = draft.cards.length * cw + (draft.cards.length - 1) * gap;
  const x0 = (v.logicalW - total) / 2;
  const y0 = 200;

  for (let i = 0; i < draft.cards.length; i += 1) {
    const c = draft.cards[i];
    const x = x0 + i * (cw + gap);
    const sel = i === cursor;
    const accent = cardAccent(world, pal, c);

    ctx.fillStyle = rgba(pal.hud.panelBg, 0.97);
    ctx.fillRect(x, y0, cw, ch);
    ctx.lineWidth = sel ? 3 : 1;
    ctx.strokeStyle = sel ? accent : pal.hud.panelRule;
    ctx.strokeRect(x, y0, cw, ch);
    ctx.fillStyle = accent;
    ctx.fillRect(x, y0, cw, 4);

    text(ctx, world, pal, `${i + 1}`, x + 16, y0 + 30, h.fontLargePx, pal.hud.textDim, 'left', 800);
    text(ctx, world, pal, categoryLabel(c.category), x + cw - 16, y0 + 28,
      h.fontSmallPx, accent, 'right', 700);

    const body = cardBody(world, c);
    // §7.3 — cvd/mono 에서 HUD·카드는 **텍스트 라벨 강제**. off 에서도 라벨은 손해가 없다
    if (body.glyph !== null) {
      glyphPath(ctx, body.glyph, x + cw / 2, y0 + 108, 26 * pal.glyphScale);
      ctx.fillStyle = accent;
      ctx.fill();
    }
    text(ctx, world, pal, body.title, x + cw / 2, y0 + 172, h.fontLargePx, pal.hud.textPrimary, 'center', 700);
    if (body.sub !== '') {
      text(ctx, world, pal, body.sub, x + cw / 2, y0 + 204, h.fontBodyPx, accent, 'center', 600);
    }
    wrap(ctx, world, pal, body.desc, x + 18, y0 + 244, cw - 36, h.fontSmallPx, pal.hud.textDim, 20);
  }

  // §11.1 리롤 — 스톡은 상점에서 산다. 1주차엔 상점이 없으므로 스톡 0이 정상이다
  const rr = `[F] 리롤  ${world.player.rerolls}회 보유 · 이 드래프트 ${draft.rerollsUsed}/${d.reroll.maxPerDraft}`;
  text(ctx, world, pal, rr, v.logicalW / 2, y0 + ch + 34, h.fontSmallPx,
    world.player.rerolls > 0 ? pal.hud.textPrimary : rgba(pal.hud.textDim, 0.5), 'center');
}

function categoryLabel(cat) {
  if (cat === 'newWeapon') return '새 무기';
  if (cat === 'weaponLevel') return '무기 레벨';
  if (cat === 'elementLevel') return '속성 투자';
  if (cat === 'passive') return '패시브';
  if (cat === 'resupply') return '보급';
  throw new Error(`hud: 미지의 드래프트 카테고리 "${cat}" (§11.1)`);
}

function cardAccent(world, pal, c) {
  if (c.category === 'elementLevel') return pal.element[c.element];
  if (c.category === 'weaponLevel' && c.isEvolution) return pal.element.normal;
  if (c.category === 'resupply') return pal.pickup.coin;
  return pal.element.normal;
}

function cardBody(world, c) {
  if (c.category === 'newWeapon') {
    const def = world.weaponDefs[c.weaponId];
    return { glyph: null, title: def.name, sub: `슬롯 ${c.slot + 1} 에 장착`, desc: def.desc };
  }
  if (c.category === 'weaponLevel') {
    const def = world.weaponDefs[c.weaponId];
    if (c.isEvolution) {
      return { glyph: null, title: def.evolution.name, sub: `${def.name} 진화 (Lv.7 → 8)`, desc: def.evolution.desc };
    }
    return { glyph: null, title: def.name, sub: `Lv.${c.from} → ${c.to}`, desc: def.desc };
  }
  if (c.category === 'elementLevel') {
    // ★ §11.1 — 부여 프리뷰. 이 한 줄이 "슬롯 순서" 규칙을 가르치는 유일한 지점이다
    return {
      glyph: c.element,
      title: `${KEYCAP[c.element]} 투자 ${c.from} → ${c.to}`,
      sub: `부여 슬롯 ${c.imbuedBefore} → ${c.imbuedAfter}`,
      desc: `${KEYCAP[c.element]} 스탠스에서 앞의 ${c.imbuedAfter}개 무기가 이 속성이 된다.`,
    };
  }
  if (c.category === 'passive') {
    const list = world.data.passives.passives;
    for (let i = 0; i < list.length; i += 1) {
      if (list[i].id !== c.passiveId) continue;
      return { glyph: null, title: list[i].name, sub: c.isNew ? '신규' : `Lv.${c.from} → ${c.to}`, desc: list[i].desc };
    }
    throw new Error(`hud: 미지의 패시브 "${c.passiveId}" (§9.6)`);
  }
  if (c.category === 'resupply') {
    return { glyph: null, title: c.name, sub: `+${c.coins} 코인`, desc: '유효한 후보가 부족할 때의 폴백 카드.' };
  }
  throw new Error(`hud: 미지의 드래프트 카테고리 "${c.category}" (§11.1)`);
}

function wrap(ctx, world, pal, s, x, y, maxW, px, color, lineH) {
  ctx.font = font(world, px, 400);
  const words = s.split(' ');
  let line = '';
  let cy = y;
  for (let i = 0; i < words.length; i += 1) {
    const t = line === '' ? words[i] : `${line} ${words[i]}`;
    if (ctx.measureText(t).width > maxW && line !== '') {
      text(ctx, world, pal, line, x, cy, px, color, 'left');
      line = words[i];
      cy += lineH;
    } else line = t;
  }
  if (line !== '') text(ctx, world, pal, line, x, cy, px, color, 'left');
}
// hudText 별칭은 importer 0 이었다 → 제거(text 는 이 파일 안에서 직접 쓰인다, 모듈-프라이빗)
