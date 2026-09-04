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
 * ★ §3.1-3항 상성항은 elements.js 의 elementTerm 이 **단일 소스**다 (공식 중복 = drift hazard 차단).
 */

import { elementTerm, hitTier } from './elements.js';

/**
 * §8.20(v1.8) ★ 가시 피해 — 화면 밖은 때릴 수 없다.
 *   판정 사각형은 render/draw.drawWorld 의 ctx.clip() 사각형(= rules.view.arena)과 «같은 것»이다.
 *   한 픽셀이라도 그려지면 맞고, 안 그려지면 안 맞는다(WYSIWYG).
 *   ★ 중심이 아니라 «몸»(중심 ± radius) 기준인 이유: 중심 기준은 반지름만큼 「보이는데 안 맞는」
 *     띠를 만들어 정반대 불평을 낳고, 몸 기준만이 렌더 클립과 같은 사각형이다.
 *   ★ 파생 필드가 아니라 순수 함수인 이유: 피해가 들어오는 시점이 셋(탄 충돌=틱 후반 ·
 *     직접피해=틱 전반 · 조준 스캔)이고 서로 다른 훅에 산다. 캐시하면 셋 중 하나는 반드시
 *     한 틱 낡은 값을 읽는다.
 *   ★ 밸런스 손잡이가 아니다 — 새 데이터 키 0개, 순수 기하 (§9.3 무관).
 */
export function onScreen(a, e) {
  return e.x + e.radius > a.x && e.x - e.radius < a.x + a.w
      && e.y + e.radius > a.y && e.y - e.radius < a.y + a.h;
}

/**
 * §3.1 — 플레이어 → 적. float 를 돌려준다 (적용은 float 누산, 표시만 반올림 — 6항).
 *
 * @param ctx   { matrix, dmgMulSum, elementBonusMul, coreGateMul }
 *                matrix          = elements.matrix                     (§9.4.4)
 *                dmgMulSum       = Σ(패시브 dmgMul)  — 가산 풀        (§9.6)
 *                elementBonusMul = resonance 의 k. 미보유면 1.0        (§3.1)
 *                coreGateMul     = rules.boss.coreGateMul              (§8.13)
 * @param dmg      w.dmg — 무기 레벨 행에서 읽은 값                     (§3.1-1항)
 * @param localMul Π(패밀리 지역 배율) — v1.5 기준 실재하는 것은 `evoSecondaryDmgMul` 뿐이다
 *                 (falloff · rearBias 는 omni 와 함께 소멸했다).
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

  // 3항 — 상성. elem > 1 일 때만 resonance 가 증폭한다 (×1 · ×0.5 불변).
  //   공식은 elements.elementTerm 이 소유한다 (미지 속성은 그 안에서 에러 — §4.1 폴백 금지).
  const elem = elementTerm(ctx.matrix, stamp, target.element, ctx.elementBonusMul);

  // 4항 — 코어 소프트 게이트. 살아있는 **armor 타입 부위 수**만 지수에 들어간다 (§3.1 · §8.13)
  const gate = target.isCore ? Math.pow(ctx.coreGateMul, target.aliveArmorPartCount) : 1;

  // 5항
  return base * dmgMul * elem * gate;
}

/**
 * §3.1 · §6.3 · §11.3 — 직접피해(광역·빔) 무기의 **단일 피해 적용 지점**.
 *   탄 충돌(step.collide)이 하는 네 가지를 한 곳으로 모은다 — 개별 무기가 하나씩 빠뜨리던 것을 막는다:
 *   ① §6.3 페이즈 전환 무적: 보스는 그 구간에 피해 0(탄 경로와 동일 게이트) — 안 하면 무적을 우회한다.
 *   ①' §8.11 레이어 봉인 무적: 앞 레이어가 살아있는 부위는 피해 0(step.collide 와 동일 게이트).
 *       ★ v1.5 가 봉인을 신설하며 탄 경로(step.js)에만 넣어, 직접피해 4무기
 *       (nova·lance·barrage·fan 진화)가 봉인을 그대로 뚫고 있었다 — 이 파일의 존재 이유가
 *       「게이트를 한 곳에 모은다」인데 정작 새 게이트를 안 받은 것이다.
 *   ② §11.3 초효과 처치 보너스의 근거: e.dmgTotal·e.dmgSuper 적립 — 안 하면 직접피해 처치가 보너스를 못 받는다.
 *   ③ §13.1.1 텔레메트리 noteDamage.
 *   반환 = 실제로 적용한 피해(무적이면 0). 호출자는 e.hp<=0 이면 killEnemy 를 부른다.
 */
export function hitEnemy(world, ctx, family, dmg, localMul, stamp, e, slotIndex) {
  // §9.5 D3 — slotIndex 는 §8.12 장갑 게이트의 «키»다. 조용히 undefined 를 색인하면
  //   floorAt[undefined] 가 NaN 비교로 항상 통과해 게이트가 죽는다. 죽은 게이트는 통과가 아니므로 던진다.
  if (slotIndex === undefined) throw new Error('damage.hitEnemy: slotIndex 누락 — §8.12 장갑 게이트의 키다 (§9.5 D3)');
  // ①''' §8.20 가시 피해 — 화면 밖은 때릴 수 없다. 직접피해 4무기(nova·lance·barrage·fan 진화)는
  //   탄을 만들지 않아 step.collide 를 지나가지 않는다. v1.5 봉인·v1.7 장갑이 정확히 이 자리에서
  //   «한쪽만 막는» 사고를 두 번 냈다 — 세 번째를 만들지 않는다.
  if (!onScreen(world.data.rules.view.arena, e)) return 0;                                // ①'''
  if (e.isBoss && world.run !== undefined && world.run.bossTransitionT > 0) return 0;   // ①
  if (e.isBoss && !e.isCore && e.sealedNow) return 0;                                    // ①' §8.11
  // ①'' §8.17(v1.7) 장갑 — 이 파일의 존재 이유가 「모든 직접피해가 공유하는 단 하나의 입구」다.
  //   여기 게이트를 안 두면 노바·랜스·바라지·팬진화 4무기가 장갑을 그대로 뚫는다
  //   (v1.5 가 봉인 sealedNow 를 탄 경로에만 넣어 같은 사고를 낸 자리 = 바로 위 ①' 줄이다).
  if (e.hitFloorSec > 0) {
    const at = e.floorAt[slotIndex];
    if (at !== 0 && world.time - at < e.hitFloorSec) return 0;
    e.floorAt[slotIndex] = world.time;
  }
  const dealt = playerToEnemy(ctx, dmg, localMul, stamp, e);
  // §11.6(v1.10 ㉒·㉗) 흡혈 — «실제로 깎은 HP»의 lifestealPct 만큼 회복(오버킬은 안 센다: 잡몹 hp 6 에 피해 17 이면 6 만). 입구 하나.
  //   ★ ㉗ 페널티: 내 HP 가 hpMax × lifestealHpRatio(0.5) «이하»일 때만 듣는다 — 위험할 때의 안전망이지 상시 회복이 아니다
  //     (사용자 2026-09-05 「흡혈에 페널티를 줘서 3택이 선택지가 되게」). 한 타로 50% 를 살짝 넘길 수는 있고, 그다음부터 안 듣는다.
  {
    const fx = world.traitFx;
    const p = world.player;
    if (fx.lifestealPct > 0 && p.hp > 0 && p.hp <= p.hpMax * fx.lifestealHpRatio) {
      const removed = e.hp > 0 ? (dealt < e.hp ? dealt : e.hp) : 0;
      if (removed > 0) { p.hp += removed * fx.lifestealPct; if (p.hp > p.hpMax) p.hp = p.hpMax; }
    }
  }
  e.hp -= dealt;
  e.dmgTotal += dealt;                                                                  // ②
  if (hitTier(ctx.matrix, stamp, e.element) === 'super') e.dmgSuper += dealt;
  noteDamage(world, family, dealt);                                                     // ③
  return dealt;
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

/**
 * §13.1.1 — 시뮬 텔레메트리 싱크. `world.tele` 가 있을 때만 적립한다.
 *   ★ 게임 실행에는 `world.tele` 가 **없다** → 이 함수는 즉시 반환하고 판정·rng·좌표 어디에도
 *     되먹임이 없다(결정성 무영향, §10.2). 시뮬만 켠다.
 */
export function noteDamage(world, family, amount) {
  const t = world.tele;
  if (t === undefined) return;
  t.dmgByFamily[family] = (t.dmgByFamily[family] === undefined ? 0 : t.dmgByFamily[family]) + amount;
}

/** §13.1.1 maxArchetypeLethalityShare — 플레이어가 «어느 아키타입에게» 맞았는가. */
export function noteDamageTaken(world, srcArch, amount) {
  const t = world.tele;
  if (t === undefined) return;
  const k = srcArch === '' ? 'other' : srcArch;
  t.dmgTakenByArch[k] = (t.dmgTakenByArch[k] === undefined ? 0 : t.dmgTakenByArch[k]) + amount;
}

