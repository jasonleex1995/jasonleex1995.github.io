/**
 * tests/shop.test.mjs — §11.2 상점 구매 (src/core/shop.js).
 *
 * 정본 계약:
 *   가격 = ceil(basePrice × growth^n), n = 런 누적 구매 수(스테이지 리셋 없음).
 *   세 관문 = maxPurchases · 재고 상한 · 코인. 상한의 거처는 둘(shop.<id>.stockMax / rules.bomb.stockMax).
 *   maxhp 는 증가분만큼 즉시 회복(§2.1). resist 는 statusDurationPct(음수)를 저항(양수)으로 누적.
 */

import { suite, test, assert, loadData } from '../tools/test.mjs';
import { createWorld } from '../src/core/state.js';
import { weapons } from '../src/core/weapons/index.js';
import { price, canBuy, buy, buyBlockedBy } from '../src/core/shop.js';

function mkWorld(coins) {
  const w = createWorld({ data: loadData(), seed: 1, weapons, hooks: {} });
  w.player.coins = coins === undefined ? 100000 : coins;
  return w;
}
const SHOP = () => loadData().meta.shop;

suite('shop/가격 §11.2', () => {
  test('가격 = ceil(basePrice × growth^n) 이고 구매마다 오른다', () => {
    const w = mkWorld();
    const s = SHOP();
    for (const id of Object.keys(s)) {
      assert.eq(price(w, id), Math.ceil(s[id].basePrice), `${id} 첫 가격 = ceil(basePrice)`);
    }
    const id = 'potion';
    const p0 = price(w, id);
    buy(w, id);
    const p1 = price(w, id);
    assert.eq(p1, Math.ceil(s[id].basePrice * s[id].growth), '두 번째 = ceil(base × growth)');
    assert.gt(p1, p0, '구매하면 가격이 오른다');
    assert.eq(w.purchaseCounts[id], 1, '누적 구매 수 기록');
  });

  test('가격 누적은 런 전체 — 여러 번 사면 growth^n', () => {
    const w = mkWorld();
    const s = SHOP().defense;
    for (let n = 0; n < s.maxPurchases; n += 1) {
      assert.eq(price(w, 'defense'), Math.ceil(s.basePrice * Math.pow(s.growth, n)), `n=${n}`);
      assert.ok(buy(w, 'defense'), `n=${n} 구매 성공`);
    }
  });

  test('미지 항목은 소리내어 실패 (§9.3 폴백 금지)', () => {
    const w = mkWorld();
    assert.throws(() => price(w, 'nope'), '미지 항목 → throw');
  });
});

suite('shop/세 관문', () => {
  test('maxPurchases 초과 구매 불가', () => {
    const w = mkWorld();
    const s = SHOP().movespeed;
    for (let i = 0; i < s.maxPurchases; i += 1) assert.ok(buy(w, 'movespeed'), `${i + 1}회차 성공`);
    assert.eq(buyBlockedBy(w, 'movespeed'), 'maxed', '상한 도달');
    assert.ok(!buy(w, 'movespeed'), '초과 구매 실패');
    assert.eq(w.purchaseCounts.movespeed, s.maxPurchases, '카운트가 상한을 넘지 않는다');
  });

  test('재고 상한 — reroll/shield/timeToken 은 shop.stockMax, bomb 은 rules.bomb.stockMax', () => {
    const s = SHOP();
    for (const id of ['reroll', 'shield', 'timeToken']) {
      const w = mkWorld();
      const max = s[id].stockMax;
      while (canBuy(w, id)) buy(w, id);
      const have = id === 'reroll' ? w.player.rerolls : id === 'shield' ? w.player.shields : w.player.tokens;
      assert.lte(have, max, `${id} 재고 ≤ stockMax(${max})`);
      assert.ok(['stock', 'maxed'].includes(buyBlockedBy(w, id)), `${id} 는 재고/횟수로 막힌다`);
    }
    const w = mkWorld();
    const bombMax = w.data.rules.bomb.stockMax;                 // ★ 거처가 상점이 아니다
    while (canBuy(w, 'bomb')) buy(w, 'bomb');
    assert.lte(w.player.bombs, bombMax, `bomb 재고 ≤ rules.bomb.stockMax(${bombMax})`);
  });

  test('코인 부족이면 구매 불가 + 잔액은 정확히 차감된다', () => {
    const w = mkWorld(0);
    assert.eq(buyBlockedBy(w, 'potion'), 'coins', '코인 부족');
    assert.ok(!buy(w, 'potion'), '구매 실패');

    const cost = price(w, 'potion');
    w.player.coins = cost;
    w.player.hp = 1;
    assert.ok(buy(w, 'potion'), '정확히 가격만큼 있으면 구매 가능');
    assert.eq(w.player.coins, 0, '잔액 = 0 (정확 차감)');
  });
});

suite('shop/효과 10종', () => {
  test('potion — healPct × hpMax 회복, hpMax 클램프', () => {
    const w = mkWorld(); const p = w.player;
    const it = SHOP().potion;
    p.hp = 1;
    buy(w, 'potion');
    assert.near(p.hp, 1 + it.healPct * p.hpMax, 1e-9, 'healPct × hpMax 회복');
    p.hp = p.hpMax;
    buy(w, 'potion');
    assert.eq(p.hp, p.hpMax, 'hpMax 초과 없음');
  });

  test('maxhp — hpMax 증가 + 증가분만큼 즉시 회복 (§2.1)', () => {
    const w = mkWorld(); const p = w.player;
    const it = SHOP().maxhp;
    p.hp = 50;
    const hpMax0 = p.hpMax;
    buy(w, 'maxhp');
    assert.eq(p.hpMax, hpMax0 + it.addHpMax, 'hpMax + addHpMax');
    assert.eq(p.hp, 50 + it.addHpMax, '증가분만큼 즉시 회복');
  });

  test('maxhp 최대 구매 시 hpMax = 기본 + addHpMax × maxPurchases', () => {
    const w = mkWorld(); const p = w.player;
    const it = SHOP().maxhp;
    const hpMax0 = p.hpMax;
    for (let i = 0; i < it.maxPurchases; i += 1) buy(w, 'maxhp');
    assert.eq(p.hpMax, hpMax0 + it.addHpMax * it.maxPurchases, '상한까지의 총 증가');
  });

  test('defense / movespeed / magnet 누적', () => {
    const w = mkWorld(); const p = w.player; const s = SHOP();
    const d0 = p.defense;
    buy(w, 'defense');
    assert.eq(p.defense, d0 + s.defense.addDefense, 'defense += addDefense');
    buy(w, 'movespeed');
    assert.near(w.shopMoveSpeedPct, s.movespeed.addMoveSpeedPct, 1e-9, 'shopMoveSpeedPct 누적');
    buy(w, 'magnet');
    assert.near(w.shopMagnetPct, s.magnet.addMagnetPct, 1e-9, 'shopMagnetPct 누적');
  });

  test('resist — statusDurationPct(음수)가 저항(양수)으로 누적, 최대 구매 시 0.60', () => {
    const w = mkWorld(); const p = w.player;
    const it = SHOP().resist;
    buy(w, 'resist');
    assert.near(p.statusResist, -it.statusDurationPct, 1e-9, '1회 = +0.20 저항');
    for (let i = 1; i < it.maxPurchases; i += 1) buy(w, 'resist');
    assert.near(p.statusResist, -it.statusDurationPct * it.maxPurchases, 1e-9, '최대 = 0.60');
  });

  test('재고형 — reroll/bomb/shield/timeToken 은 addStock 만큼 늘어난다', () => {
    const w = mkWorld(); const p = w.player; const s = SHOP();
    const r0 = p.rerolls; buy(w, 'reroll');
    assert.eq(p.rerolls, r0 + s.reroll.addStock, 'reroll +addStock');
    const b0 = p.bombs; buy(w, 'bomb');
    assert.eq(p.bombs, b0 + s.bomb.addStock, 'bomb +addStock');
    const sh0 = p.shields; buy(w, 'shield');
    assert.eq(p.shields, sh0 + s.shield.addStock, 'shield +addStock');
    const t0 = p.tokens; buy(w, 'timeToken');
    assert.eq(p.tokens, t0 + s.timeToken.addStock, 'timeToken +addStock');
  });

  test('10항목 전부 최소 1회 구매 가능하고 효과가 적용된다 (미구현 항목 없음)', () => {
    const w = mkWorld();
    let bought = 0;
    for (const id of Object.keys(SHOP())) {
      if (buy(w, id)) bought += 1;
    }
    assert.eq(bought, Object.keys(SHOP()).length, '10항목 전부 구매 성공(throw 없음)');
  });
});
