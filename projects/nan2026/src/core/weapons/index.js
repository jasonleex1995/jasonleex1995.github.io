/**
 * src/core/weapons/index.js — 패밀리 레지스트리 (§9.5)
 *
 * 사용 (§9.1 의 파일 배치는 확정돼 있으나 **합성 계약이 인쇄되지 않아** state.js 가 주입으로 뒀다):
 *   import { weapons } from './src/core/weapons/index.js';
 *   const world = createWorld({ data, seed, weapons, hooks: { enemies: null, emitters: null } });
 *
 * 각 모듈은 `{ update(world, slot, eff, dt) }` 를 default export 한다.
 * step.fireWeapons 가 슬롯마다 recomputeEff(world, slot) 를 계산해 넘기므로
 * ★ 무기 모듈은 JSON 을 스스로 읽지 않는다 → weapons/** 숫자 리터럴 제약이 자연히 지켜진다.
 *
 * ★ 1주차 범위 = 3 패밀리. 나머지 9 (lance orbit aura mine boomerang barrage omni drone nova)
 *   는 아직 없다 — step.fireWeapons 는 미등록 패밀리를 만나면 **조용히 넘어가지 않고 던진다**.
 *   ★ 보고 대상: draft.js 의 newWeapon 후보는 data/weapons.json 의 12 패밀리 전부에서 나온다.
 *   1주차에 미구현 패밀리 카드를 뽑으면 그 던지기에 걸리므로, **드래프트 후보를 이 레지스트리의
 *   키로 교집합하는 일은 호출자(main.js)의 몫**이다 — core 는 무엇이 구현됐는지 모른다.
 */

import forward from './forward.js';
import fan from './fan.js';
import seeker from './seeker.js';

/** 키 = §9.5 의 family (id == family, 12종 1:1) */
export const weapons = { forward, fan, seeker };

export default weapons;
