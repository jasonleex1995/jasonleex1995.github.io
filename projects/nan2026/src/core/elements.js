/**
 * src/core/elements.js
 *
 * 정본 v1.4 구현 절:
 *   §4.1    상성 (LOCKED) — 물 > 불 > 풀 > 물, 노말은 항상 ×1
 *   §3.1-3항 데미지 공식의 배율 항 (elem)
 *   §9.4.4  elements.json 스키마 — order · investable · matrix
 *   §9.1    core 순수성
 *
 * ★ 이 파일에 상성 숫자가 없다. matrix 는 elements.json 이 유일하게 소유한다 (C-2 · C-8).
 * ★ 적의 공격에는 속성이 없다 (§4.1) — 그러므로 이 표는 플레이어→적 경로에서만 조회된다.
 */

/**
 * §4.1 · §3.1-3항 — 공격 속성 → 방어 속성 배율. ∈ {0.5, 1.0, 2.0}
 * 미지 속성은 에러다 (§9.3 정신: 폴백 금지 — 조용히 ×1 을 돌려주면 상성이 죽는다).
 */
export function elementMul(matrix, attacker, defender) {
  const row = matrix[attacker];
  if (row === undefined) {
    throw new Error(`elements.matrix: 미지의 공격 속성 "${attacker}" (§4.1)`);
  }
  const m = row[defender];
  if (m === undefined) {
    throw new Error(`elements.matrix["${attacker}"]: 미지의 방어 속성 "${defender}" (§4.1)`);
  }
  return m;
}

/**
 * §3.1-3항 전체 — 상성 배율 + resonance 증폭.
 *   elem = matrix[stamp][target]
 *   if (elem > 1) elem = 1 + (elem - 1) × elementBonusMul
 * ★ elem ≤ 1 에는 절대 적용되지 않는다 (×0.5 · ×1 불변, §9.6).
 * ★ elementBonusMul 의 기본값은 1.0 이다 (§3.1 — resonance 미보유).
 */
export function elementTerm(matrix, attacker, defender, elementBonusMul) {
  const elem = elementMul(matrix, attacker, defender);
  if (elem > 1) return 1 + (elem - 1) * elementBonusMul;
  return elem;
}

/** §4.2 — 투자 가능 속성인가. 노말은 투자축이 아니다 */
export function isInvestable(elements, element) {
  return elements.investable.indexOf(element) >= 0;
}

/** §4.1 — 스탠스로 선택 가능한 4속성인가 (order = normal fire water grass) */
export function isElement(elements, element) {
  return elements.order.indexOf(element) >= 0;
}
