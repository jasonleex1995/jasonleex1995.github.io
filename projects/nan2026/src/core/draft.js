/**
 * src/core/draft.js
 *
 * 정본 v1.4 구현 절:
 *   §11.1  드래프트 (meta.json > draft) — 카테고리를 먼저 뽑지 않는다.
 *          **유효 후보 아이템 전체를 가중치 목록으로 만들고 비복원 3장 추첨.**
 *              weight(item) = categoryWeights[item.category] × modifier(item)
 *          유효성 필터 · 보장(≤ optionCount−1) · 피티 · 리롤 · 폴백(resupply)
 *   §11.1  슬롯 상한(무기 4칸 만석 → newWeapon 카테고리 전체 제외, 교체 제안 없음)
 *   §4.2   속성 상한 — elementCapPerElement 4 · elementCapTotal 6
 *   §9.9   onboarding.autoEquipFirstElement — 투자 0 → 1의 **최초 전이에서만** 자동 장착
 *   §10.2  rng.draft — 드래프트 후보 추첨 · 리롤 (다른 7 스트림과 공유하지 않는다)
 *   §9.1   core 순수성
 *
 * ★ 값(가중치·보너스·피티·폴백 코인)은 전부 meta.draft 에서 온다. 이 파일에 20도 40도 6도 없다.
 * ★ H4 (§9.6.1) — "훅이 무효인 패시브"는 **거르지 않는다**. 시스템이 지우면 드래프트는
 *   선택이 아니라 자동 최적화가 된다.
 */

import { giveWeapon, levelUpWeapon, givePassive } from './state.js';
import { investElement, investTotal, requestStance } from './stance.js';

const CAT_NEW_WEAPON = 'newWeapon';
const CAT_WEAPON_LEVEL = 'weaponLevel';
const CAT_ELEMENT_LEVEL = 'elementLevel';
const CAT_PASSIVE = 'passive';
const CAT_RESUPPLY = 'resupply';

const MAX_WEAPON_LEVEL = 8;   // §9.5 — Lv8 에서 종료. Lv9 없음 (구조이지 값이 아니다)

// ---------------------------------------------------------------------------
// 유효 후보 (§11.1 — 무효 카드는 애초에 풀에 없다)
// ---------------------------------------------------------------------------
function ownedWeaponCount(world) {
  let n = 0;
  for (let i = 0; i < world.slots.length; i += 1) if (world.slots[i].weaponId !== null) n += 1;
  return n;
}

function hasWeapon(world, id) {
  for (let i = 0; i < world.slots.length; i += 1) if (world.slots[i].weaponId === id) return true;
  return false;
}

function passiveLevel(world, id) {
  for (let i = 0; i < world.passives.length; i += 1) if (world.passives[i].id === id) return world.passives[i].level;
  return 0;
}

function passiveSlotsFull(world) {
  for (let i = 0; i < world.passives.length; i += 1) if (world.passives[i].id === null) return false;
  return true;
}

/**
 * §4.1 — 이 속성으로 부여된 무기가 ×2 를 넣는 상대(먹이).
 *   상성표를 이 파일에 리터럴로 복제하지 않고 elements.matrix 에서 유도한다.
 *   ★ 카드가 "키 문자(W/E/R)"가 아니라 **결과**(무엇에게 강해지는가)를 스스로 들고 다니게
 *     하는 필드 — 렌더는 이 element id 를 플레이어 언어로 옮기기만 하면 된다 (§11.1).
 */
function preyElement(world, el) {
  const row = world.data.elements.matrix[el];
  const keys = Object.keys(row);
  for (let i = 0; i < keys.length; i += 1) if (row[keys[i]] === 2) return keys[i];
  return null;
}

/**
 * §11.1 — 유효 후보 아이템 **전체**를 가중치와 함께 만든다.
 * 카테고리를 먼저 뽑지 않는다 — 이 목록 하나에서 비복원 추첨한다.
 */
export function candidates(world) {
  const d = world.data.meta.draft;
  const cw = d.categoryWeights;
  const rp = world.data.rules.player;
  const out = [];
  const nWeapons = ownedWeaponCount(world);

  // --- newWeapon ------------------------------------------------------------
  // §11.1 — 무기 4칸 만석 → 카테고리 **전체**를 풀에서 제외. 교체 제안 없음
  if (nWeapons < rp.weaponSlots) {
    // §11.1 — newWeaponSlotScale 은 **슬롯 상황**의 함수다.
    //   §13.5.1 이 "무기 1개 보유 시 newWeapon 실효 가중치 60"(= 20 × 3.0)이라 검산했다
    //   → 인덱스 = 보유 무기 수 − 1.
    const scale = d.newWeaponSlotScale[nWeapons - 1];
    const ws = world.data.weapons.weapons;
    for (let i = 0; i < ws.length; i += 1) {
      // §11.1 — 이미 보유한 무기는 newWeapon 후보에서 제외 (id == family 1:1)
      //   귀결: forward 는 시작 무기이므로 newWeapon 카드로 영원히 등장하지 않는다
      if (hasWeapon(world, ws[i].id)) continue;
      out.push({ category: CAT_NEW_WEAPON, key: `${CAT_NEW_WEAPON}:${ws[i].id}`,
        weaponId: ws[i].id, slot: nWeapons, weight: cw.newWeapon * scale });
    }
  }

  // --- weaponLevel ----------------------------------------------------------
  for (let i = 0; i < world.slots.length; i += 1) {
    const s = world.slots[i];
    if (s.weaponId === null) continue;
    if (s.level >= MAX_WEAPON_LEVEL) continue;              // §11.1 — Lv8 이면 그 카드 제외
    // §9.5 — Lv7 → Lv8 레벨업 카드 그 자체가 진화 카드다
    const isEvo = s.level + 1 === MAX_WEAPON_LEVEL;
    out.push({ category: CAT_WEAPON_LEVEL, key: `${CAT_WEAPON_LEVEL}:${s.weaponId}`,
      slot: i, weaponId: s.weaponId, from: s.level, to: s.level + 1, isEvolution: isEvo,
      weight: cw.weaponLevel * (isEvo ? d.weaponLevelEvolutionBonus : 1) });
  }

  // --- elementLevel ---------------------------------------------------------
  const total = investTotal(world);
  if (total < rp.elementCapTotal) {                         // §4.2 — 합계 도달 → 3속성 카드 전부 제외
    const inv = world.data.elements.investable;
    for (let i = 0; i < inv.length; i += 1) {
      const el = inv[i];
      const lv = world.player.invest[el];
      if (lv >= rp.elementCapPerElement) continue;          // §4.2 — 개별 상한 도달
      // §11.1 — 속성 레벨 ≥ 현재 무기 수 → 그 속성 카드 제외 (죽은 투자 차단)
      if (d.elementLevelOfferRequiresWeaponCount && lv >= nWeapons) continue;
      out.push({ category: CAT_ELEMENT_LEVEL, key: `${CAT_ELEMENT_LEVEL}:${el}`,
        element: el, from: lv, to: lv + 1,
        // ★ prey = 이 속성이 ×2 를 넣는 상대 (결과 프리뷰 — matrix 유도, §4.1)
        prey: preyElement(world, el),
        // ★ 부여 프리뷰 = 이 한 줄이 슬롯 순서 규칙을 가르치는 유일한 지점 (§11.1 카드 표기)
        imbuedBefore: Math.min(lv, nWeapons), imbuedAfter: Math.min(lv + 1, nWeapons),
        weight: cw.elementLevel * (lv === 0 ? d.elementFirstLevelBonus : 1) });
    }
  }

  // --- passive --------------------------------------------------------------
  const ps = world.data.passives.passives;
  const maxLv = world.data.passives.maxLevel;
  const full = passiveSlotsFull(world);
  for (let i = 0; i < ps.length; i += 1) {
    const lv = passiveLevel(world, ps[i].id);
    if (lv >= maxLv) continue;                               // §11.1 — Lv5 이면 그 카드 제외
    if (lv === 0 && full) continue;                          // §11.1 — 6칸 만석 → 미보유 카드 제외
    // ★ H4 — 훅이 무효인 패시브(autoload × aura/nova/drone 등)는 **제외하지 않는다**
    out.push({ category: CAT_PASSIVE, key: `${CAT_PASSIVE}:${ps[i].id}`,
      passiveId: ps[i].id, from: lv, to: lv + 1, isNew: lv === 0,
      weight: cw.passive * (lv === 0 ? d.passiveNewBonus : 1) });
  }

  return out;
}

// ---------------------------------------------------------------------------
// 추첨
// ---------------------------------------------------------------------------
/** 후보 목록에서 rng.draft 로 1장 뽑아 제거한다 (비복원) */
function drawFrom(world, pool, filterCategory) {
  const idx = [];
  const w = [];
  for (let i = 0; i < pool.length; i += 1) {
    if (filterCategory !== null && pool[i].category !== filterCategory) continue;
    idx.push(i);
    w.push(pool[i].weight);
  }
  if (idx.length === 0) return null;
  const k = world.rng.draft.weighted(w);
  if (k < 0) return null;
  const at = idx[k];
  const card = pool[at];
  pool.splice(at, 1);
  return card;
}

/**
 * §11.1 — 보장 목록. ★ 총 개수는 optionCount − 1 을 넘을 수 없다 (check.mjs S21).
 *   보장되는 것은 「카드의 존재」이지 「플레이어의 선택」이 아니다.
 * ★ §11.1(v1.4) — 두 보장이 동시에 걸릴 때의 추첨 순서 = [elementLevel, newWeapon] 확정
 *   (비복원이라 순서가 자유 1장 분포를 바꾼다 → 결정성 필수).
 */
function guarantees(world, pityBefore) {
  const d = world.data.meta.draft;
  const g = [];
  if (d.guaranteeElementCardOnFirstDraft && world.draftsSeen === 0) g.push(CAT_ELEMENT_LEVEL);
  // §11.1 · §13.2 — elementCardPity: 6 = "6드래프트마다 속성 카드의 **등장**을 보장"
  //   (연속 5회 미등장 → 6회차에 강제 → 42드래프트에 강제 7회, §13.2-① 의 검산과 일치)
  else if (pityBefore >= d.elementCardPity - 1) g.push(CAT_ELEMENT_LEVEL);
  if (ownedWeaponCount(world) < d.guaranteeNewWeaponUntilSlots) g.push(CAT_NEW_WEAPON);
  if (g.length > d.optionCount - 1) {
    throw new Error(`draft: 보장 ${g.length}개 > optionCount−1(${d.optionCount - 1}) — §11.1 "보장이 3개가 되면 선택이 0이 된다" (S21)`);
  }
  return g;
}

/**
 * ★ 레벨업 드래프트 1회를 생성한다. 게임 클럭은 이미 멈춰 있다 (§6.4 · draft.pauseGame).
 * @returns { cards, rerollsUsed, excluded }
 */
export function buildDraft(world) {
  // pityBefore 를 고정해 둔다 → 리롤을 몇 번 하든 피티 카운터가 한 번만 움직인다
  const draft = { cards: [], rerollsUsed: 0, excluded: [], pityBefore: world.elementPity };
  fill(world, draft);
  return draft;
}

/**
 * §11.1 리롤 — 3장 **전체** 재추첨. 이전 3장은 그 드래프트 동안 풀에서 제외.
 *   ★ 리롤이 재도박이면 운 완화가 아니다. 제외해야 리롤이 **확정적인 개선**이 된다.
 * @returns 리롤이 실제로 일어났는가
 */
export function rerollDraft(world, draft) {
  const d = world.data.meta.draft;
  if (draft.rerollsUsed >= d.reroll.maxPerDraft) return false;
  if (world.player.rerolls <= 0) return false;              // 스톡은 상점에서 산다
  world.player.rerolls -= 1;
  draft.rerollsUsed += 1;
  if (!d.reroll.canRepeatPrevious) {
    for (let i = 0; i < draft.cards.length; i += 1) draft.excluded.push(draft.cards[i].key);
  }
  draft.cards.length = 0;
  fill(world, draft);
  return true;
}

function fill(world, draft) {
  const d = world.data.meta.draft;
  const pool = candidates(world);

  // 리롤로 제외된 카드를 뺀다 (§11.1 — "방금 본 3장은 그 드래프트 동안 풀에서 제외")
  for (let i = pool.length - 1; i >= 0; i -= 1) {
    if (draft.excluded.indexOf(pool[i].key) >= 0) pool.splice(i, 1);
  }

  // ① 보장분 — "3장 = {속성 1, 무기 1, 자유 1}"
  const g = guarantees(world, draft.pityBefore);
  for (let i = 0; i < g.length && draft.cards.length < d.optionCount; i += 1) {
    const c = drawFrom(world, pool, g[i]);
    // ★ §11.1(v1.4) — 보장 카테고리에 유효 후보가 0이면 그 보장은 **소멸**한다
    //   (존재할 카드가 없으면 지어내지 않는다, §9.3 폴백 금지) → 건너뛴다.
    if (c !== null) draft.cards.push(c);
  }

  // ② 자유분 — 유효 후보 **전체**에서 비복원 추첨 (distinctItemsPerDraft: 비복원이 곧 그 보장)
  while (draft.cards.length < d.optionCount) {
    const c = drawFrom(world, pool, null);
    if (c === null) break;
    draft.cards.push(c);
  }

  // ③ 폴백 — 유효 후보가 optionCount 미만이면 부족분을 코인 카드로 채운다.
  //    빈 드래프트 화면이 **물리적으로 불가능**해진다 (§11.1)
  while (draft.cards.length < d.optionCount) {
    draft.cards.push({ category: CAT_RESUPPLY, key: `${CAT_RESUPPLY}:${draft.cards.length}`,
      id: d.fallback.id, name: d.fallback.name, coins: d.fallback.coins, weight: 0 });
  }

  // §11.1 피티 — 이번 드래프트에 속성 카드가 **등장**했는가로 카운터가 갈린다
  let sawElement = false;
  for (let i = 0; i < draft.cards.length; i += 1) {
    if (draft.cards[i].category === CAT_ELEMENT_LEVEL) { sawElement = true; break; }
  }
  world.elementPity = sawElement ? 0 : draft.pityBefore + 1;
}

// ---------------------------------------------------------------------------
// 확정
// ---------------------------------------------------------------------------
/**
 * 카드 1장을 확정한다. 이것이 드래프트를 소비하고 큐를 하나 줄인다 (§6.4).
 */
export function applyCard(world, card) {
  if (card.category === CAT_NEW_WEAPON) {
    // §11.1 slotAssign "append" — 가장 앞의 빈 슬롯에 자동 배치
    giveWeapon(world, card.weaponId);
  } else if (card.category === CAT_WEAPON_LEVEL) {
    // §5.3 · §9.5 family-키 — 슬롯 재정렬이 card.slot 을 stale 로 만들 수 있다(빈 슬롯 레벨업
    //   크래시). weaponId(==family, 1:1)로 world.slots 에서 현재 슬롯을 해소해 레벨업한다.
    let si = -1;
    for (let i = 0; i < world.slots.length; i += 1) {
      if (world.slots[i].weaponId === card.weaponId) { si = i; break; }
    }
    if (si >= 0) levelUpWeapon(world, si);
  } else if (card.category === CAT_ELEMENT_LEVEL) {
    const first = world.player.invest[card.element] === 0;
    investElement(world, card.element);
    // §9.9 onboarding — 속성 카드 확정 순간 그 스탠스 자동 장착. **투자 0 → 1의 최초 전이에서만**
    if (first && world.data.meta.onboarding.autoEquipFirstElement && !world.autoEquipDone) {
      world.autoEquipDone = true;
      requestStance(world, card.element);
    }
  } else if (card.category === CAT_PASSIVE) {
    givePassive(world, card.passiveId);
  } else if (card.category === CAT_RESUPPLY) {
    world.player.coins += card.coins;
  } else {
    throw new Error(`draft: 미지의 카테고리 "${card.category}" (§11.1)`);
  }
  world.draftsSeen += 1;
  if (world.draftQueue > 0) world.draftQueue -= 1;
}
