/**
 * src/core/weapons/barrage.js — 바라지 (§9.5)
 *
 * 폐쇄된 파라미터 계약 (§9.5 12행 표):
 *   base            : dmg cooldownSec targetMode strikeIntervalSec strikesPerVolley
 *                     blastRadius telegraphSec
 *   evolution.params: evoRadiusMul
 *
 * §9.6.1 훅은 state.recomputeEff 가 이미 적용했다:
 *   rateKey "cooldownSec" (H1) · countKey "strikesPerVolley" (+projCountAdd) · pierceApplies false
 *   areaKeys ["blastRadius"]
 *
 * ★ 착탄은 **예고 뒤에 온다** — telegraphs 풀에 kind 'strike' 로 예고를 놓고(§7.4 의 예고 어휘와
 *   같은 자리), telegraphSec 이 지나면 그 자리에서 폭발시킨다. step.hazards 는 'laser' 만 소유하므로
 *   이 예고의 수명·반납은 **이 파일이 소유**한다(§12.1 — 새 풀을 만들지 않는다).
 * ★ 조준: 기본은 randomInArena(rng.pattern — 정본이 플레이어 무기에 허용한 유일한 스트림, §9.5).
 *   진화(오비탈 스트라이크)는 **가장 밀집한 곳**을 노리고 반경이 evoRadiusMul 배가 된다.
 *
 * 슬롯 스크래치: a0 = 다음 볼리까지 / a1 = 이번 볼리의 남은 착탄 수 / a2 = 다음 착탄까지
 */

import { spawnTelegraph } from '../state.js';
import { playerToEnemy, noteDamage } from '../damage.js';
import { stampFor } from '../stance.js';
import { killEnemy } from '../step.js';

const RANDOM_IN_ARENA = 'randomInArena';
const DENSEST = 'densest';
const STRIKE = 'strike';

/** §3.1 의 컨텍스트. ★ 모듈 스코프 1회 (§10.3) */
const ctx = { matrix: null, dmgMulSum: 0, elementBonusMul: 1, coreGateMul: 0 };
const _at = { x: 0, y: 0 };

/**
 * 착탄 지점. targetMode 'densest' 면 «그 반경 안 이웃이 가장 많은 적», 아니면 아레나 무작위.
 *   ★ densest 는 **레벨 8 이 targetMode 로 바꾼다**(진화 플래그가 아니라 계약 값이 정한다).
 */
function aim(world, slot, eff, r, out) {
  const a = world.data.rules.view.arena;
  if (eff.targetMode === DENSEST) {
    const en = world.enemies.items;
    let best = null;
    let bestN = 0;
    for (let i = 0; i < en.length; i += 1) {
      const e = en[i];
      if (!e.alive) continue;
      let n = 0;
      for (let j = 0; j < en.length; j += 1) {
        const o = en[j];
        if (!o.alive) continue;
        const dx = o.x - e.x;
        const dy = o.y - e.y;
        if (dx * dx + dy * dy <= r * r) n += 1;
      }
      if (best === null || n > bestN) { bestN = n; best = e; }
    }
    if (best !== null) { out.x = best.x; out.y = best.y; return out; }
  }
  out.x = a.x + world.rng.pattern.f() * a.w;
  out.y = a.y + world.rng.pattern.f() * a.h;
  return out;
}

/** 예고가 익으면 그 자리에서 폭발. */
function detonate(world, slot, eff, x, y, r) {
  ctx.matrix = world.data.elements.matrix;
  ctx.dmgMulSum = world.stats.dmgMul;
  ctx.elementBonusMul = world.stats.elementBonusMul;
  ctx.coreGateMul = world.data.rules.boss.coreGateMul;
  const stamp = stampFor(world, slot.index, 'spawn', slot.stampElement);
  const en = world.enemies.items;
  for (let i = 0; i < en.length; i += 1) {
    const e = en[i];
    if (!e.alive) continue;
    const dx = e.x - x;
    const dy = e.y - y;
    if (dx * dx + dy * dy > r * r) continue;
    const dealt = playerToEnemy(ctx, eff.dmg, 1, stamp, e);
    e.hp -= dealt;
    noteDamage(world, slot.family, dealt);
    if (e.hp <= 0) killEnemy(world, e);
  }
}

export function update(world, slot, eff, dt) {
  if (eff.targetMode !== RANDOM_IN_ARENA && eff.targetMode !== DENSEST) {
    throw new Error(`barrage: 미구현 targetMode "${eff.targetMode}" — 계약 = randomInArena | densest (§9.5)`);
  }
  // ★ slot.evolved 분기 정확히 1개 (§9.5) — 오비탈 스트라이크: 반경 확대
  const radius = slot.evolved ? eff.blastRadius * eff.evoRadiusMul : eff.blastRadius;

  // ── 익은 예고를 터뜨린다(이 무기가 소유) ─────────────────────────────────
  const ts = world.telegraphs.items;
  for (let i = 0; i < ts.length; i += 1) {
    const t = ts[i];
    if (!t.alive || t.kind !== STRIKE) continue;
    if (t.age < t.durSec) continue;
    detonate(world, slot, eff, t.x, t.y, t.r);
    world.telegraphs.release(t);
  }

  // ── 볼리 스케줄 — cooldownSec 마다 strikesPerVolley 발, 간격 strikeIntervalSec ──
  if (slot.a1 <= 0) {
    slot.a0 -= dt;
    if (slot.a0 <= 0) { slot.a0 += eff.cooldownSec; slot.a1 = eff.strikesPerVolley; slot.a2 = 0; }
  }
  if (slot.a1 <= 0) return;

  slot.a2 -= dt;
  if (slot.a2 > 0) return;
  slot.a2 += eff.strikeIntervalSec;
  slot.a1 -= 1;

  aim(world, slot, eff, radius, _at);
  // 예고를 놓는다 — r 에 폭발 반경, durSec 에 예고 시간. 익으면 위에서 터진다
  spawnTelegraph(world, STRIKE, _at.x, _at.y, radius, eff.telegraphSec, slot.index);
}

export default { update };
