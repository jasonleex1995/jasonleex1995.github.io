/**
 * src/core/shop.js — 상점 구매 적용 (순수 core, §11.2)
 *
 * 정본 v1.4 구현 절:
 *   §11.2   가격 = ceil(basePrice × growth^n), n = **그 항목의 런 누적 구매 수**(스테이지 리셋 없음).
 *           전 항목 상시 노출·무작위 없음·재고/시간 제한은 maxPurchases 와 재고 상한뿐.
 *   §11.2.1 10항목의 값(meta.json > shop)이 유일한 거처. 이 파일에 밸런스 수치가 없다.
 *   §2.1    hpMax 증가는 **증가분만큼 즉시 회복**한다(healsSameAmount) — bulkhead 패시브와 같은 규칙.
 *   §9.1    순수성 — window/Date/Math.random 0. RNG 없음(상점에는 난수가 없다).
 *
 * ★ 상한의 두 거처 (§9.4 · §11.2.1 — 틀리면 S40 의 불변식이 런타임에서 조용히 깨진다):
 *     reroll · shield · timeToken → `meta.shop.<id>.stockMax`
 *     bomb                        → `rules.bomb.stockMax`  (상점이 아니라 폭탄 자신의 성질)
 *   defense/maxhp/movespeed/magnet/resist 는 **maxPurchases 가 곧 상한**이다
 *   (예: maxhp 10 × 4회 = +40 → hpMax 140 · resist 0.20 × 3회 = 0.60).
 *
 * ★ 코인은 실수다 — salvage(coinGainMul)가 곱해지므로 정수가 아니다. §11.3 이 「마지막에 한 번
 *   floor」를 확정했으므로 여기서 반올림하지 않는다(중간 반올림은 잔액을 조용히 흘린다).
 */

import { recomputeStats } from './state.js';

/** 항목 정의. 미지 항목은 소리내어 실패한다(§9.3 폴백 금지). */
function def(world, itemId) {
  const it = world.data.meta.shop[itemId];
  if (it === undefined) throw new Error(`shop: 미지의 항목 "${itemId}" (§11.2)`);
  return it;
}

/** §11.2 — 다음 구매 가격. n = 런 누적 구매 수. */
export function price(world, itemId) {
  const it = def(world, itemId);
  return Math.ceil(it.basePrice * Math.pow(it.growth, world.purchaseCounts[itemId]));
}

/** 재고형 항목의 현재 보유량과 상한. 재고형이 아니면 null. */
function stockOf(world, itemId, it) {
  const p = world.player;
  if (itemId === 'reroll') return { have: p.rerolls, max: it.stockMax };
  if (itemId === 'shield') return { have: p.shields, max: it.stockMax };
  if (itemId === 'timeToken') return { have: p.tokens, max: it.stockMax };
  if (itemId === 'bomb') return { have: p.bombs, max: world.data.rules.bomb.stockMax };
  return null;
}

/**
 * 지금 살 수 있는가. 세 관문: 구매 횟수 상한 · 재고 상한 · 코인.
 *   ★ 「살 수 없는 이유」를 UI 가 구분해 보여줄 수 있게 실패 사유를 문자열로 돌려준다(null = 구매 가능).
 */
export function buyBlockedBy(world, itemId) {
  const it = def(world, itemId);
  if (world.purchaseCounts[itemId] >= it.maxPurchases) return 'maxed';
  const st = stockOf(world, itemId, it);
  if (st !== null && st.have >= st.max) return 'stock';
  if (world.player.coins < price(world, itemId)) return 'coins';
  return null;
}

/** 살 수 있는가(§11.2). */
export function canBuy(world, itemId) { return buyBlockedBy(world, itemId) === null; }

/**
 * §11.2 — 구매를 적용한다. 살 수 없으면 아무것도 하지 않고 false.
 *   효과의 거처는 **이미 존재하는 누적 필드**들이다(core 는 상점을 몰라도 그 합에는 참여해 왔다).
 */
export function buy(world, itemId) {
  if (!canBuy(world, itemId)) return false;
  const it = def(world, itemId);
  const p = world.player;

  p.coins -= price(world, itemId);
  world.purchaseCounts[itemId] += 1;

  if (itemId === 'reroll') { p.rerolls += it.addStock; return true; }
  if (itemId === 'bomb') { p.bombs += it.addStock; return true; }
  if (itemId === 'shield') { p.shields += it.addStock; return true; }
  if (itemId === 'timeToken') { p.tokens += it.addStock; return true; }
  if (itemId === 'potion') {
    p.hp += it.healPct * p.hpMax;
    if (p.hp > p.hpMax) p.hp = p.hpMax;
    return true;
  }
  if (itemId === 'defense') { p.defense += it.addDefense; return true; }
  if (itemId === 'maxhp') {
    // ★ healsSameAmount(§2.1)는 여기서 처리하지 않는다 — recomputeStats 가 **모든** hpMax 증가에
    //   대해 델타 회복을 수행하는 공용 경로다(패시브 bulkhead 와 같은 규칙). 여기서 또 채우면 이중 회복.
    world.shopHpAdd += it.addHpMax;
    recomputeStats(world);
    return true;
  }
  if (itemId === 'movespeed') { world.shopMoveSpeedPct += it.addMoveSpeedPct; return true; }
  if (itemId === 'magnet') { world.shopMagnetPct += it.addMagnetPct; return true; }
  if (itemId === 'resist') {
    // statusDurationPct 는 **음수**(-0.20 = 지속 20% 감소)이고 statusResist 는 양수 저항이다
    p.statusResist -= it.statusDurationPct;
    return true;
  }
  throw new Error(`shop: 효과가 구현되지 않은 항목 "${itemId}" (§11.2 — 폴백 금지)`);
}
