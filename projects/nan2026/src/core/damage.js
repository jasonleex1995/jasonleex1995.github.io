/**
 * src/core/damage.js
 *
 * 정본 v1.4 구현 절:
 *   §3.1  플레이어 → 적 (구조 동결 — 항 추가/제거 금지. 5항 + 지역 배율 폐쇄 목록 3종)
 *   §3.2  적 → 플레이어 (정액 감산 방어력 모델 + 25% 하한)
 *   §2.4  i-frame 은 데미지 계산 **전에** 게이트한다 (여기 밖 — step.js)
 *   §9.1  core 순수성
 *
 * ★ 이 파일에 숫자 리터럴로 된 밸런스 값이 없다. 전부 주입된 데이터에서 온다.
 * ★ 데미지 경로에서 RNG를 호출하지 않는다 (크리티컬 없음 · 데미지 난수 없음 — §3.1).
 * ★ 적의 방어력은 존재하지 않는다 (§3.1) — 평가할 항 자체가 없다.
 */

/**
 * §3.1 — 플레이어 → 적. float 를 돌려준다 (적용은 float 누산, 표시만 반올림 — 6항).
 *
 * @param ctx   { matrix, dmgMulSum, elementBonusMul, coreGateMul }
 *                matrix          = elements.matrix                     (§9.4.4)
 *                dmgMulSum       = Σ(패시브 dmgMul)  — 가산 풀        (§9.6)
 *                elementBonusMul = resonance 의 k. 미보유면 1.0        (§3.1)
 *                coreGateMul     = rules.boss.coreGateMul              (§8.13)
 * @param dmg      w.dmg — 무기 레벨 행에서 읽은 값                     (§3.1-1항)
 * @param localMul Π(패밀리 지역 배율) — falloff | rearBias | evoSecondaryDmgMul 의 곱.
 *                 ★ 폐쇄 목록 3종이며 전부 **1항 안**에서 곱해진다 (§3.1).
 *                 해당 없으면 1.
 * @param stamp    피해 개체에 각인된 속성 (§4.4)
 * @param target   { element, isCore, aliveArmorPartCount }
 */
export function playerToEnemy(ctx, dmg, localMul, stamp, target) {
  // 1항 — base = w.dmg × Π(패밀리 지역 배율)
  const base = dmg * localMul;

  // 2항 — 가산 풀 → 1회 적용 (곱연산 폭주 방지)
  const dmgMul = 1 + ctx.dmgMulSum;

  // 3항 — 상성. elem > 1 일 때만 resonance 가 증폭한다 (×1 · ×0.5 불변)
  const row = ctx.matrix[stamp];
  if (row === undefined) throw new Error(`damage: 미지의 각인 속성 "${stamp}" (§4.1)`);
  const raw = row[target.element];
  if (raw === undefined) throw new Error(`damage: 미지의 대상 속성 "${target.element}" (§4.1)`);
  const elem = raw > 1 ? 1 + (raw - 1) * ctx.elementBonusMul : raw;

  // 4항 — 코어 소프트 게이트. 살아있는 **armor 타입 부위 수**만 지수에 들어간다 (§3.1 · §8.13)
  const gate = target.isCore ? Math.pow(ctx.coreGateMul, target.aliveArmorPartCount) : 1;

  // 5항
  return base * dmgMul * elem * gate;
}

/**
 * §3.1-6항 — 표시용 반올림. 적용에는 절대 쓰지 않는다.
 */
export function displayDamage(v) {
  return Math.round(v);
}

/**
 * §3.2 — 적 → 플레이어. 정액 감산 + 원본의 damageFloorRatio 하한.
 * ★ i-frame 게이트를 통과한 뒤에만 호출한다 (§2.4 — 게임초당 최대 1회).
 * ★ 실드는 호출자가 처리한다 (§3.2: taken = 0 + 실드 −1 + i-frame 발동).
 *
 * @param player { defense }
 * @param rules  rules.player  (damageFloorRatio)
 * @param raw    bullets[].dmg | contactDmg | zone dmg
 */
export function enemyToPlayer(rules, player, raw) {
  return Math.ceil(Math.max(raw - player.defense, raw * rules.damageFloorRatio));
}
