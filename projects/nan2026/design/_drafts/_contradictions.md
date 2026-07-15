# 비평가 판정 — 초안 6종의 모순 & 미결 목록

> 정본(canon)이 **반드시 결정해야 할 항목 리스트**. 각 항목마다 하나의 답을 확정할 것.

## 판정

**아직 아니다. 개발 시작 가능한 상태가 아니며 "거의 완성"과도 거리가 멀다.**

핵심 진단: 이 섹션들은 구멍을 **메운** 것이 아니라, **각자 독립적으로 게임 전체를 다시 설계**했다. 6개 섹션이 아레나 규격, 히트박스, 이동 모델, 무적시간, 방어력 공식, 팔레트, 키맵, 적 스키마, 무기 스키마, 보스 구조, 상점, 점수, 캡, 텔레그래프 하한, localStorage, 파일 매니페스트를 **각각 독립적으로 확정**했고 그중 어느 두 개도 일치하지 않는다. 사용자가 걱정한 "개발 중 설계 변경"은 이제 확률이 아니라 **확정**이다 — 개발자는 `enemies.json`을 만드는 **첫날 오전에** D와 F 중 하나를 버리는 설계 결정을 내려야 한다.

특히 치명적인 것 5가지:

1. **파일 매니페스트가 6벌.** F는 9개 파일을 고정하고 "미지 키 = 에러, 누락 키 = 에러, 기본값 폴백 금지"를 선언한다. A의 `tuning.json`, B의 `visual.json`, C의 `draft.json`, D의 `themes.json`, E의 `flow.json`/`onboarding.json`은 F의 로더에서 **전부 로드 실패**한다. 밸런싱 이전에 부팅의 문제다.

2. **"동결" 선언이 서로를 동결한다.** D와 F 둘 다 적 스키마를 "필드 동결"로, C와 F 둘 다 무기 파라미터를 "닫힌 계약"으로 선언했다. 동결된 두 어휘가 충돌하면 어느 쪽을 풀어도 상대 섹션 전체가 무효화된다.

3. **검증 게이트가 상호 배타.** 텔레그래프 하한 0.40(B) / 0.8(D) / 0.55(F), 동시 텔레그래프 6(B) / 2(D), 탄속 420(B) / 260(F) — 셋 다 "위반 시 로드 실패"다. 더 나쁜 건 시뮬 인증: D의 보스당 격파율(분산 ≥0.95)은 런 0.735를 함의해 F의 `clearRate.max: 0.65`를 **베이스라인이 실패**시키고, D의 무투자 ≥0.55는 런 0.028로 F의 `noDeadLuck ≥ 0.20`을 실패시킨다. **어떤 숫자로도 동시 통과 불가능한 두 개의 `exit 1` 게이트**를 문서가 동시에 요구한다.

4. **자기 예시가 자기 규칙을 위반.** F의 유일한 보스 예시(강철 가오리)는 D의 R4·R5·R6을 전부 위반한다. E의 온보딩 보스는 D의 동결된 `boss_part_count: 4`를 위반한다. E의 대비 웨이브는 D의 G7(±3%p)을 위반한다. E의 온보딩 핵심 비트는 B가 폐지한 잡몹 데미지 숫자에 의존한다. **문서가 자기 자신의 검증기를 통과하지 못한다.**

5. **스케일 전제 붕괴.** C의 생존 패시브(`bulkhead` +1/+1/+2/+2/+3)와 상점 `maxhp`(+1)는 HP ~10 규모를 전제로 쓰였는데 A/F는 `hpMax: 100`이다. 숫자 튜닝이 아니라 C의 패시브 12종 구성을 다시 짜야 하는 문제다.

**그럼에도 좋은 것**: 개별 섹션의 **품질과 논증은 높다**. B의 불변식 4개와 플레이어 FX 오프스크린 합성 하드캡(= 4칸 상한의 실제 집행 장치), D의 어휘 조합(신규 코드 0)과 테마 커버리지 증명(5-of-6 비복원 → 3속성 보장), C의 진화 트리거(Lv8 픽 자체 = 새 카테고리 0·새 RNG 0)와 성장 예산 대차대조표, E의 전이 순서 근거(회복 → 테마 → 상점)와 상성 처치 = 데미지 지분(막타 기준의 체스식 최적화 회피), F의 배속 = 틱 수(dt 스케일 아님 → 터널링·시뮬 무효화 동시 회피)와 봇 반응지연의 실제ms→틱 환산 — 이것들은 각각 **정확히 옳고 기둥과 정합**한다. 재료는 훌륭하다. 조립이 안 됐을 뿐이다.

**권고 — 다음 액션은 "새 섹션"이 아니라 "중재(arbitration) 패스" 하나다:**

- **A0. 단일 정본(canonical) 확정 세션.** 위 충돌 55개를 표 하나에 놓고 각 항목에 **소유 섹션 1개**를 지정한 뒤 나머지를 **삭제**한다. 추천 소유권: 규격·캡·팔레트·스키마·파일 목록 = F(구현 스펙이 최종 권위) / 시각 언어·레이어·불변식 = B / 콘텐츠 어휘 = D / 로스터·드래프트 = C / 플로우·점수 = E / 조작·HUD 배치 = A. **단, 소유자가 아닌 섹션의 해당 수치·스키마는 전부 지워야 한다.** 남겨두면 반드시 다시 충돌한다.
- **A1. 명명·단위 규약 1장.** 틱/게임초/초 중 하나(권장: `_gs` 게임초 — D·B의 논증이 여기 걸려 있음), snake vs camel 중 하나. F의 검증기가 미지 키를 거부하므로 선택이 아니라 필수.
- **A2. 인증 게이트 재산정.** D의 보스당 지표를 런 지표로 환산하거나 그 반대로 통일하고, 산술적으로 동시 만족 가능한 창을 계산해 넣는다. 지금은 커밋이 영원히 막힌다.
- **A3. 진짜 구멍 6개만 추가 작성**(아무도 안 쓴 것): 상태이상 모델(둔화 배율·스택·스턴 중 입력), 적 방어력 유무, `wave_clear_advance` 시맨틱, 오디오/BGM, 옵션 화면, 아트·스프라이트 방침.
- **A4. `farm_xp_ratio ≥ 2.0` 산술 검산.** 웨이브 간격 9초·스폰 창 95초·상한 14웨이브로는 ~1.3배가 천장이다. 게이트를 낮추든 메커니즘을 바꾸든 **지금** 결정해야 한다. 나중에 발견하면 그게 바로 4주 예산을 태우는 설계 변경이다.

**정직한 요약**: 지금 개발을 시작하면 1주차는 밸런싱이 아니라 **설계 중재**가 된다. 다만 중재는 새 설계가 아니라 **선택**이므로 1~2일이면 끝난다. 그 후에는 "밸런싱 = 숫자만"이 실제로 성립할 가능성이 높다 — 재료의 품질은 충분하다. 지금 필요한 건 여섯 번째 섹션이 아니라 **하나를 고르고 다섯을 지우는 일**이다.

## 모순 (54건)

- ★구조★ 적 데이터 스키마가 두 벌이고 **둘 다 "필드 동결"을 선언** — D §2.9 `{band, movement_id(dive/weave/column/strafe/anchor/orbit_drift/charge/rear_in), attack_id(6종), attack_params, hitbox_r_px, theme_only}` vs F §14 `{tier, elementPolicy, hp, contactDmg, radius, spriteId, move.type(straightDown/sineDown/diveAtPlayer/enterStop/strafeLR/arcIn/orbitPoint/retreatUp), attack.emitterId, hpScalePerStage}`. 어휘 이름·개수·의미가 전부 다른데 파일명은 같은 `enemies.json`. 공격 어휘도 D 6종(`homing`·`status` 포함) vs F 8종 프리미티브(`aimed`·`ring`·`spiral`·`wall` 포함, `homing`·`status`는 타입조차 아님).
- ★구조★ 무기 스키마 두 벌, 같은 `weapons.json` — C: id `blitz`, `per_level[0..7]` **8행 전체 파라미터**, **틱 단위**(`cooldown_ticks`), snake_case / F: family `forward`, `base` + `levels` **부분 오버라이드**, **초 단위**(`cooldownSec`), camelCase. 12무기 ID 네임스페이스 자체가 불일치(`blitz/fanout/pulse/return/rearguard/option` vs `forward/fan/aura/boomerang/omni/drone`). `starting_weapon: "blitz"`(C) vs `startWeaponId: "forward"`(F).
- ★구조★ 데이터 파일 매니페스트가 6개 상호 배타 — A `data/tuning.json` / B `data/visual.json` / C `weapons·passives·draft.json` / D `enemies·themes·bosses·stages.json` / E `flow·shop·score·onboarding.json` / F **9파일 고정 목록**(`rules elements weapons passives bullets enemies bosses stages meta`) + "미지 키 = 에러, 누락 키 = 에러, 기본값 폴백 금지". F의 매니페스트엔 `tuning.json`·`visual.json`·`themes.json`·`draft.json`·`flow.json`·`onboarding.json`이 **존재하지 않음** → 다른 5개 섹션의 모든 키가 로드 시 에러.
- ★핵심★ 드래프트가 보스전에 뜨는가 — A §2.4 "보스 페이즈 중 뜬 레벨업 드래프트: 재정렬 불가(카드 선택만)"(=뜬다) / E §4.1 "보스 페이즈: **발생하지 않음, 큐에 적립**" / F §17 `pauseGame: true, pauseBossTimer: false`(=뜨고, 화면은 멈추고, 타이머는 간다) / D §6.6 보스 XP 0 + 부위 XP 0 + 소환 금지(=구조적으로 발생 불가). **4개 섹션이 3개의 답**을 내고 각자 자기 답이 기둥을 지킨다고 논증.
- ★핵심★ 방어력 모델 3개 — A: 정액 감산 `taken = ceil(max(dmg - defense, dmg × 0.25))`, 상점 상한 방어력 8 / C: 상점 `defense`가 `damage_taken_mult ×0.92` **곱연산** / F: 정률 `1 - def/(def+60)`, 상한 방어력 60(perStack 15 × 4). `damageFloorRatio`도 0.25(A) vs 0.10(F). 셋은 상호 배타적 모델이며 숫자 튜닝으로 수렴 불가.
- ★핵심★ 상점 진열 방식 — C §5-1 "**전체 목록 상시, 랜덤 진열 없음**"(랜덤 진열은 리롤이라는 운 완화 밸브와 자기모순이라 **명시 거부**) vs E §6.1 "**고정 4칸 + 로테이션 2칸**, 직전 방문 중복 금지"(전체 목록은 6회 방문 최적해를 푸는 **바둑/체스**라 명시 거부) vs F `stockMode: "full"`. 두 섹션이 서로의 근거를 정면 반박하며 반대 결론.
- ★핵심★ 팔레트 두 벌 — B(`visual.json`): fire `#FF6B2C` / water `#31C0FF` / grass `#5FD13B`, 적탄 = 텔레그래프 = **자홍 `#FF2E88`**(점선/실선 2상태 언어의 전제), 상태이상 = 호박 `#FFB000` 단일 채널 / F(`rules.json > palette`, "속성 색의 **단일 진실**"): fire `#E2452F` / water `#2E8BE6` / grass `#3FBF5B`, 적탄 `#FF3FE0`, **텔레그래프 노랑 `#FFE24A`**. F의 노란 텔레그래프는 B의 2상태 언어와 B의 호박 단일 채널을 **동시에** 파괴.
- 보스 HP 구조 — D §6.7 `hp_total: 42000` + `parts[].hp_share` 합계 1.0(코어도 부위, share 0.5) vs F §15 "**코어 HP가 곧 보스 HP**, 부위 HP는 별도(합산 아님)" `core.hp: 4000` + `parts[].hp` 절대값. 3분 타이머가 무엇을 재는지가 달라짐.
- 코어 소프트 게이트 — D `part_armor_core_mult 0.4 ^ (살아있는 **armor** 부위 수)`(2개면 ×0.16) + "mobility·armament 파괴는 코어 배율에 **영향 없음**"(= 보스전 트레이드오프의 근거) vs F `damageTakenMulWhilePartsAlive: 0.25` — **부위 종류 무관, 하나라도 살아있으면** ×0.25. F는 D가 명시적으로 배제한 것을 게이트에 넣음.
- 보스 부위 효과 어휘 — D `part_type` 4종(`mobility/armament/armor/core`) + 타입별 고정 효과 vs F `onDestroy: [{op, v}]` 6종(`moveSpeedMul dmgMul emitterOff spawnWave phaseAdvance openCore`). F의 `spawnWave`는 D의 `boss_summons_allowed: false`("스테이지 보스는 잡몹을 소환하지 않는다")와 직접 충돌.
- 강철 가오리가 3판 — DESIGN.md L103(엔진=불 / 좌우 날개=물 / 몸체=노말) vs D §7.1(스러스터 mobility 물 / 좌 지느러미 armor 불 / 우 지느러미 armor 풀 / 코어 노말) vs F §15(engine=fire / wingL=water / wingR=water / core=normal). **F의 예시는 D의 R4(테마 속성 부위 최대 1 — water 2개), R5(armor ≠ 테마 — armor 부위 자체가 없음), R6(armor 1~2개)를 전부 위반** → F의 유일한 보스 예시가 D의 하드 게이트에서 로드 실패.
- 보스 부위 속성 제약 — D R1~R6(코어 = 항상 노말 / 주변부 노말 금지 / 서로 다른 속성 ≥2 / 테마 속성 최대 1부위 / **armor ≠ 테마 속성** / armor 1~2개)이 "셀링 포인트를 규칙으로 보장"한다고 선언 vs F §21 검증 게이트 4번은 "부위 속성 ≥ 2종" **하나만** 강제. R1·R4·R5·R6이 F의 검증기에 없음 → AI 생성물이 D의 의도를 합법적으로 우회.
- 텔레그래프 최소 리드타임 3값, **셋 다 로드 실패 하드 게이트** — B `minLeadSecGame: 0.40`("모든 공격의 절대 하한, 이보다 짧으면 로드 시 거부") / D `telegraph_min_gs: 0.8`(G4) / F `fairness.minTelegraphSec: 0.55`(check.mjs 3번). 스턴 텔레그래프도 B 1.50 / D 1.0 / F `minStunTelegraphSec: 1.2`. 어떤 콘텐츠도 셋을 동시에 만족 못 함.
- 동시 텔레그래프 상한 — B `caps.telegraphsVisible: 6`(초과 시 `deferAttack`) vs D `telegraph_concurrent_max: 2`("**보스 스크립트의 가장 강한 제약**, 3개를 읽으라는 순간 그건 트위치", G5로 정적 검사). 3배 차이, 둘 다 하드 게이트.
- 탄속·틈 상한 — B `enemyBulletMaxSpeedPxGameSec: 420` vs F `fairness.maxBulletSpeed: 260`(조준탄 200). 통과 틈 B `minGapWidthPx: 56` vs F 46. D는 두 키를 "시뮬 인증 대상"으로 두고 값 없음. 또 B §4는 "배경 스크롤 90 ≤ 적 탄 **최저속**(`≥140`)의 절반"이라 쓰지만 `140`은 B 자신의 `enemyBulletMinSpawnRadiusPx`(**반지름**)이며 B는 최저 탄속 키를 정의한 적이 없음 = B 내부 오류.
- 엔티티 캡 3벌 — B(적탄 320 / 적 90 / 플탄 220 / 파티클 400) vs F(적탄 **640** / 적 **160** / 플탄 512, 그러면서 `fairness.maxSimultaneousEnemyBullets: 320`으로 **자기 캡과도 충돌**) vs D(`enemy_concurrent_max: 40`, 새떼 70 — 그러면서 §5.1에서 "**60기**가 화면을 뒤덮고"라고 서술해 40과 충돌). 게다가 B는 `caps.enemies: 90`·`caps.pickups: 200`을 선언하면서 `caps.overflow` 맵에 **그 두 개의 초과 정책을 안 씀**(제목은 "초과 시 동작(시뮬 결정성 필수)").
- 아레나·히트박스 — 아레나 **560×720@x360**(A) vs **580×720@x350**(F) vs 밀도 저작 기준 **640×960**(B `refPlayfieldPx`). 패널 폭 360(A) vs 350(F). 플레이어 히트박스 r=**4**(A) vs r=**3**(B §7.1) vs r=**6**(F). 이동 영역 인셋 상56/하56/좌우20(A, 520×608) vs "`hitboxRadius`만큼"(F, 568×708).
- 이동 모델 — A "**관성 없음**, 즉시 정지, `moveResponseTau = 0`(관성은 미세 조작 실패 = 트위치 = 기둥 위반), `moveSpeedBase: 280`, SOCD **마지막 키 우선**" vs F "`moveSpeed: 260, accel: 4200, decel: 5200`(= **관성 존재**), 반대키 동시 입력은 **0으로 상쇄**". 모델·값·SOCD 셋 다 반대.
- 무적시간 — A `iFrameDuration: 1.0` 게임초 / 깜빡임 **10Hz** vs B `iFrameBlinkHz: **8.0**` vs F `iframeSec: **1.2**`. A는 "hpMax=100 + i-frame 1초 → 죽으려면 최소 5~12초"라는 **관대함의 산술적 보증**을 1.0에 근거해 세움.
- 기체 부착 HUD의 z-order가 정반대 — A §3.6 "기체 부착 요소는 적 탄 **아래** z-order(탄 가독성 우선)" vs B §9.1 레이어 10 "**히트박스 도트** ★ | 적 탄보다 **위**(추적 대상은 항상 보인다)". 각자 상대의 근거를 부정하며, 이건 B의 "눈이 있는 곳에서 스탠스를 읽는다" 시스템 전체의 전제.
- 스탠스 링의 색과 기체 부착 규격 — A §1.3 "현재 스탠스 색 링"(무조건) vs B §7.1 "투자 0인 스탠스면 **은색**(실제로 아무것도 부여 안 됨 → I-2 준수)" — B는 A의 규칙이 자신의 불변식 I-2(No Lying Color) 위반이라고 사실상 선언. 규격도 전부 다름: 임뷰 칩 기체 아래 10px·5×4px(A) vs +14px·7px(B); 상태 배지 위 12px(A) vs -16px(B); 스탠스 전환 링 120ms·r14→30(A) vs 180ms·r0→28(B `ringExpandSec 0.18`).
- ★온보딩이 폐지된 기능에 의존★ — E §12의 결정적 비트 "0:12–0:20 같은 적을 같은 무기로 쏘는데 **숫자와** 이펙트가 방금 전과 다르다 → ×2/×0.5를 몸으로 목격"은 **잡몹 데미지 숫자**를 전제하는데, B §6.2가 "데미지 숫자 — **잡몹에는 없다(확정)**"로 폐지(밀도 붕괴 1순위 원인, I-4 위반). E의 60초 무-텍스트 교육 계획에서 가장 중요한 순간이 존재하지 않는 채널에 걸려 있음.
- 온보딩 강제 드래프트 vs 드래프트 보장 규칙 — E "첫 드래프트 = **3장 전부** 속성 카드(불+1/물+1/풀+1)" vs C `guarantee_element_card_on_first_draft: true` = **1장** 확정 + `guarantee_new_weapon_until_slots: 2` = 무기 <2면 **new_weapon 카드 1장 보장**. C의 보장 규칙이 E의 "3장 전부 속성"을 구조적으로 불가능하게 만듦.
- 온보딩 대비 웨이브 vs 혼합 비율 하드 게이트 — E의 "좌 레인 = 테마 속성 / 우 레인 = 테마를 이기는 속성" 25초 웨이브(0:20~0:45)는 D G7("스테이지 총 개체 수 기준 속성 비율이 `theme_mix`에 **±3%p**")를 크게 초과. D가 비율 제외로 명시한 것은 엘리트·중간보스·새떼·보스뿐 — **온보딩 웨이브는 제외 대상이 아님** → 스테이지 1이 검증 게이트에서 로드 실패.
- 온보딩 보스 vs 보스 부위 수 동결 — E "온보딩 보스 = **부위 2개**짜리 단순 복합 보스(좌 = A속성 / 우 = A를 이기는 속성 / 코어 = 노말)" = 3부위 vs D `boss_part_count: **4**`(**동결**, 최종만 5). 또 E의 "좌 = 테마 속성" 부위가 armor면 D의 R5(armor 속성 ≠ 테마)도 위반.
- 노말 테마의 존재 여부 — E §2.3 `stage1ExcludesNormalTheme: true`를 튜너블 키로 두고 §15에서 "테마 추첨 섹션"에 **계약으로 요구**(= 노말 테마가 풀에 존재한다는 전제) vs D §1.1 "노말 테마 **폐기**(v0.5의 평원 삭제), 모든 테마는 물·불·풀 중 하나" / F "`stages[].element ∈ {water,fire,grass}` — **스키마가 노말 테마를 금지**". E의 키가 죽은 키이며 E의 계약이 충족 불가능.
- ★최종 스테이지가 F의 스키마로 표현 불가★ — F는 `stages[].element ∈ {water,fire,grass}`를 **스키마 강제**로 선언하는데 D의 최종(`theme_id: "finale"`)은 테마 속성이 **없음**(물30/불30/풀30/노말10 `finale_mix`). F의 `themeDraw.finalStageId: "apex"`엔 finale mix·15종 총출동·로테이션 새떼·5부위 TETRARCH·`finale_exempt_rules`가 전무.
- 테마 풀 — D 6종 `sea/glacier/volcano/desert/forest/bog`(**속성당 정확히 2종** → 5-of-6 비복원 추출로 커버리지가 **자동 증명**) vs F 6종 `sea/forest/volcano/swamp/tundra/canyon` + `guaranteeElements` **제약**(= 거부 샘플링 필요, canyon의 속성 미상). 스테이지 1 후보도 D `intro_ok` **3종** vs F `firstStageFrom` **2종**.
- 혼합 비율 — D 70/10/10/10(counter/prey 규칙에서 파생, `element_assign: "per_wave"`, ±3%p 게이트) vs E §3.1 예시 "물 70% / 불 12% / 풀 12% / 노말 6%" vs F `mix: {themeShare: 0.70, otherWeights: {fire:0.34, grass:0.33, normal:0.33}}`(스테이지별 임의 데이터, counter/prey 규칙 없음). 테마 고지에 표시할 **정확한 숫자**가 3벌.
- 잡몹 페이즈 스킵 + 드랍 회수 — D `mob_phase_skip: **false**` / E "**스킵 없음**" vs F `skippable: **true**`("조기 스킵 가능(XP ↔ 보스 타이머 여유의 트레이드)"). 미수집 드랍도 D `phase_end_autocollect: true` / E `sweepPickupsOnPhaseEnd: true`("전환 손실 0") vs F "남은 적은 소멸하고 **XP는 자동 정산 없이 소멸**" — F는 무-노가다 기둥을 정면으로 어김.
- 엘리트의 정의 — D §2.7 "엘리트 = 웨이브 안의 일반 아키타입 1기에 붙는 **접두(prefix) 플래그**, 별도 개체가 **아니다**" vs F §14 "`tier: \"elite\"` — **엘리트가 여기서 정의된다**. 잡몹 로스터의 **별도 tier 엔트리(별개 개체)**" + F §16 `elites: [{enemyId: "reef_elite"}]`(테마 전용 개체). **둘 다 "여기서 정의된다"고 선언**.
- 적 HP 스테이지 스케일 — D §1.4 표 `enemy_hp_scale [1.0, 1.5, 2.2, 3.2, 4.5, 6.0]`(스테이지 6 = **6배**) vs F §14 개체별 `hpScalePerStage: 0.18` 선형(스테이지 6 = **1.9배**). 3배 이상 차이.
- 위기 세션·중간보스·보스 인트로 — 새떼 25초/경고 3.0(D, E) vs `durationSec: **18**, warnSec: **2.5**`(F). 중간보스 이탈 **30**초(D `midboss_leave_after_gs`, E `midBossLingerSeconds` = "선택적의 정의") vs **40**초(F `despawnAfterSec`); 전 테마 공용 3종 `mb_hammer/mb_lancer/mb_nest`(D) vs 테마 전용 `kraken_mini`(F); 적 어휘(D)로 정의 vs 보스 스키마 `core/parts/phases`(F). 보스 등장 연출 **2.5**초(D) vs **3.0**초(E, F).
- ★시뮬 인증 게이트가 상호 만족 불가★ — D §9.3은 **보스당** 격파율(분산 ≥0.95 / 특화 ≥0.85 / 무투자 ≥0.55)을, F §20은 **런** 클리어율 `0.35~0.65` + `noDeadLuck.minClearRateWorstDraftPolicy ≥ 0.20`을 요구. D의 분산 0.95/보스 → 런 0.95⁶ = **0.735 > F의 상한 0.65**(베이스라인이 F를 실패). D의 무투자 0.55/보스 → 런 0.55⁶ = **0.028 << F의 0.20**(F를 실패). 둘 다 `exit 1` 커밋 차단 게이트 → **어떤 밸런스 값으로도 동시 통과 불가**.
- 속성 투자 합계 상한 — C `element_level_cap_total: **8**`("4/4/0, 4/2/2, 3/3/2 같은 **진짜 다이얼**"이라는 근거 서술) vs F `elementCapTotal: **6**`("불+4·물+4·풀+4 붕괴 봉인"이라는 근거 서술). 게다가 F는 이걸 **튜너블**로, C는 설계 결정으로 분류 → 밸런서가 6→8로 바꾸면 C의 설계 근거를 밟음. 성장 예산도 69(C) vs 68(F)이고 `certify.growthBudget.maxLevelUps: 68`이 하드 게이트.
- 드래프트 가중치·피티·폴백·리롤 — 가중치 `{new_weapon:20, weapon_level:40, element_level:20, passive:20}` + 4개 동적 스케일(C) vs `{newWeapon:30, weaponLevel:40, elementLevel:15, passive:15}` 고정, 동적 스케일 **없음**(F). 피티 6(C) vs 8(F). 폴백 코인 **40**(C, E) vs **25**(F). 드래프트당 리롤 `reroll_max_per_draft: 2`(C) vs 무제한(E "스톡이 있는 한 연속", F `chainable: true`).
- 점수 — 무피격 스코프 **스테이지 단위**(E, "2스테이지에서 깨지면 나머지 28분이 무의미"라는 feel-bad 근거) vs `noHitScope: "perBoss"`(F, "런 전체는 죽은 점수면"이라는 다른 근거). 퍼펙트 30000(E) vs 20000(F); `coinToScore` **20**(E) vs **10**(F); 시간 보너스 50(E) vs 40(F); 상성 처치 인코딩 `superEffectiveKillMultiplier: 0.5`(가산, E) vs `superEffectiveKillMul: 1.5`(배율, F). **F의 score에는 `bossClearBonus`·`midBossClearBonus`·`runClearBonus`가 아예 없고**, E가 LOCKED로 선언한 `timeTokenForfeitsTimeBonus`도 F에 없음 → 토큰이 공짜가 되어 3분 허들이 코인으로 삭제됨.
- 컨티뉴 — C/F는 **상점 판매**(`continue` 항목, 250) vs E §6.1 "**컨티뉴는 상점에 없음** → 사망 시점에 제시(오락실식)", 비용 **150**. E는 카운트다운 없음·부활 상태·타이머 `max(잔여, 60)`을 전부 확정했는데 C/F의 상점 모델엔 그 플로우가 없음.
- 상점 수치 전면 불일치 — 실드 상한 **3**(C) / **2**(E, F); 리롤 상한 **5**(C) / **3**(E, F); 물약 35%·상한 8(C) / 30%·**무제한**(E) / 35%·상한 3(F); 방어 +2×4=**8**(A) / ×0.92 곱연산(C) / +15×4=**60**(F); 최대HP +10(A, F) / **+1**(C); 이속 +6%×4(A) / +5%×4(C) / +6%×3(F); 자석 +20%×4(C) / +30%×3(F); 가격 곡선 항목별 growth 1.35~1.80(C) / **전역** `priceGrowth: 1.6`(E) / `mul`+`add` **2종**(F).
- ★최대 HP의 스케일 전제가 붕괴★ — A/F `hpMax: 100`(정수 HP + 5칸 세그먼트 바)인데 C의 `bulkhead` 패시브는 **+1/+1/+2/+2/+3**(Lv5 만렙에 +9), 상점 `maxhp`는 "**최대 HP +1**(소폭)". C는 HP 총량 ~10 규모(하트/라이프제)를 전제로 쓰였음 → 100 HP 풀에서 C의 유일한 생존 스탯 패시브가 +9%로 무의미해짐. 숫자 튜닝이 아니라 C의 패시브 구성 자체를 다시 짜야 함.
- 패시브 슬롯 수 — A §3.1 ASCII HUD와 §3.4 "③ **패시브 5칸**" vs C §0 "패시브 슬롯 **6칸 확정**"(성장 예산 산술이 6에서 성립) / F `passiveSlots: 6`. HUD가 6번째 슬롯을 못 그림.
- 패시브 훅 어휘 두 벌 — C 12개 훅(`cooldown_mult`, `super_effective_mult`, `untargetable_ticks_on_hit`, `hit_bullet_clear_radius`, `pierce_add`, `count_add` …) vs F 20개 `stats`(`dmgMul`, `elementBonusMul`, `ghostSecOnHit`, `durationMul`, `bombDmgMul`, `iframeSecAdd`, `critChance`, `critMul` …). 이름 전부 다름. **특히 C의 `resonance` values `[2.1 … 2.5]`(= 결과 배율)를 F의 `elementBonusMul`(= `elem = 1 + (elem-1) × k`) 슬롯에 넣으면 ×3.1~×3.5**가 됨 = 의미가 다른 두 키를 같은 축으로 취급.
- 크리티컬 — A §1.1 "데미지 난수 없음. **크리티컬 없음**"(확정) vs F §7 데미지식 3단계에 crit 항 존재 + `critChance/critMul`이 **패시브 스탯 어휘에 포함** + "크리티컬을 나중에 켜는 것이 숫자 변경이 되도록 항을 지금 넣어둔다". A가 폐지한 것을 F가 확장 지점으로 설계.
- 적 탄 데미지의 소유자 — A 전역 티어 6종(`enemyBullet.damageSmall: 8`, `damageMedium: 14`, …) vs F `bullets.json`의 개체별 `dmg: 8` vs **D의 "동결된" `attack_params`에는 damage 필드가 아예 없음**(`straight: count, speed, interval_gs`가 전부). D의 스키마로는 적 탄 데미지·반경을 표현할 수 없는데 D가 스키마 소유권을 주장.
- 장판 데미지 모델 — A `enemyBullet.damageZoneTick: 10` + "모든 피해원이 i-frame 공유 → **초당 최대 1회**" vs F `zone` emitter의 `dps` 파라미터(= 연속 드레인). A의 i-frame 공유는 "오라/오빗 다중 히트 즉사 불가"라는 12패밀리 중 2개의 설계 근거.
- 픽업 운동 — B §10 "**자석 반경 밖: 플레이어를 절대 안 쫓음**(스크롤 따라 하강만)" + "한 문장: 천천히 곡선으로 쫓아오면 100% 탄, 순식간에 직선으로 빨려오면 100% 보상" vs F `orbAutoHomeDelaySec: 3.0, orbAutoHomeSpeed: 55`(3초 후 반경 무관 **느리게 추적**). F의 오브는 B가 "100% 탄"이라고 가르친 운동 특성과 정확히 일치 → B의 유일한 픽업/유도탄 분리 장치가 붕괴.
- 조준 철학 — C §1-4 "보스 부위는 각각 독립 타겟 엔티티. `nearest` 계열은 가장 가까운 부위를 노림 → **플레이어가 위치로 부위를 고른다**(자동 타겟이 퍼즐을 대신 풀지 않음)" vs F §11 `targetMode: "bossPartPriority"` — "**복합 보스에서 부위를 지정할 수단**을 데이터로 제공, 어느 부위를 때리나는 밸런싱 가능한 숫자". "의미 있는 이동" 기둥에 대해 정반대 결론. 어휘도 배타 — C는 `densest`(barrage 진화가 사용) 있고 `bossPartPriority` 없음, F는 그 반대.
- `element_bind`/`elementStampMode`의 드론 취급 — C는 `option`(드론)을 **투사체 = `spawn`** 목록에 넣음 vs F "`live`(매 틱 현재 스탠스 재평가: 오라·오빗·**드론**)". C는 `spawn` 고정이 "불 기뢰 대량 설치 → 물 전환" 악용을 막는 근거라고 서술.
- 후방 진입 — D `rear_in` 이동 + `rear_spawn_allowed[stage]`(스테이지 3+) 게이트 + `warn_gs: 0.8` 사전 표식 **필수**("뒤에서 소리 없이 나오는 적은 금지", 무-트위치 기둥) vs F `move.entrySide: "bottom"` — 스테이지 게이트도 경고도 없음.
- 재정렬 UI 3벌 — A(←→ 커서 + **Space** 집기/놓기; 잡몹 드래프트 + **상점 '빌드' 탭**에서 가능, `reorderAllowedIn: ["draftMobPhase","betweenStages"]`) vs E(**Tab**으로 모드 진입; **레벨업 드래프트 화면에서만**, 상점 빌드 탭 자체가 없음) vs F(`slotPick: [Digit1..4]`, `slotMove: [←,→]`).
- 키맵 3벌 — 리롤 **`F`**(A `"reroll":"KeyF"`) / **`R`**(E) / **`T`**(F `"reroll":"KeyT"`); 시간 토큰 **`ShiftLeft`**(A, E) / **`KeyF`**(F); 드래프트 선택 **`1/2/3`**(A, F `draftPick`) / **`←→`+Space**(E). E의 리롤 `R`은 A의 "드래프트 화면에서 QWER은 **죽은 키**"(오조작 원천 차단)와 직접 충돌. DESIGN.md 미결 "1234 vs QWER"은 QWER로 수렴했으나 **나머지 키는 오히려 3갈래로 벌어짐**.
- localStorage — **같은 키 이름 `"nan2026.v1"`에 배타적 스키마 2개**: E `{v, best: {normal: 0(숫자), …}, runs}` vs F 같은 키 `{v, best: {normal: {score, seed, at, stage}}, opts: {mute, volume, shake, flash, cbMode, damageNumbers}, seen: {tutorial}}`. 초기화 위치도 "타이틀 하단 기록 초기화"(E) vs "**옵션 화면** '기록 삭제'"(F — E의 상태 기계에 그 화면이 없음).
- 단위·명명 규약 3종 혼재 — 틱(C `cooldown_ticks`, `lifetime_ticks`, `telegraph_ticks`) / 게임초(D `_gs` 접미) / 초(F `Sec` 접미). snake_case(C, D) vs camelCase(A, B, E, F). `schema_version`(C, D) vs `schemaVersion`(A, F). F의 검증기는 **미지 키를 에러로 거부**하므로 이 혼재는 곧 로드 실패.
- E의 상태 기계 내부 불일치 — §1 다이어그램은 `THEME_BANNER → MOB_PHASE → CRISIS → BOSS_INTRO → BOSS`(CRISIS가 **별개 상태**)인데 §3.3 본문은 "위기 세션 = 잡몹 페이즈 **내부**의 마지막 25초"이고 D `crisis_start_gs: 95` / `mob_phase_duration_gs: 120`도 내부. 개발자가 구현할 상태 기계가 자기 문서와 불일치.

## 여전히 미결 (22건)

### blocker (3)

- **상태이상 모델 (둔화/스턴)** — 둔화의 **이동속도 배율**이 어느 섹션에도 없음. 스택 규칙(중첩/갱신/무시)도 없음. 스턴 중 자동발사가 계속되는지, 스탠스 전환(QWER)이 먹히는지도 미정 — B §8은 "입력 무시"라고만 씀. `statusResistMul`/`resist`가 지속시간을 줄이는지 강도를 줄이는지도 C(지속시간)와 F(키 이름만) 사이에서 미확정. 개발자가 4개를 전부 발명해야 함.
- **데미지 계산식 — 적의 방어력** — F §7의 8단계 식 7번 항 `final = max(pre × (1 - def/(def+defenseK)), pre × damageFloorRatio)`에서 `def`가 정의 안 됨. 식은 플레이어→적인데 `defenseK`·`damageFloorRatio`는 `rules.player` 스코프이고, D/F의 `enemies.json` 어느 스키마에도 적의 defense 필드가 없음. 적 방어가 존재하는가? 없다면 죽은 항, 있다면 스키마 누락 — 식 자체가 평가 불가.
- **파밍 리스크 — `wave_clear_advance` 의미** — "전멸 시 다음 웨이브 즉시 소환"이 **남은 스케줄 전체를 압축**하는지, **다음 1개만 당기고 나머지는 절대시각 유지**인지 미정. D의 웨이브 레코드는 `at_gs` 절대시각이라 규칙과 자료구조가 불일치하고, F의 `waves[].atSec` 절대 타임라인은 이 규칙을 표현조차 못 함. 파밍=리스크/리워드 기둥의 유일한 메커니즘인데 구현 형태가 없음.

### major (14)

- **파밍 리스크 — 산술이 자기 게이트를 못 받침** — `wave_interval_gs 9.0` + `crisis_suspends_waves: true`(스폰 창 95초) → 스폰 웨이브 ≈10.5, `mob_phase_max_waves: 14`. 즉 `wave_clear_advance`의 최대 이득은 ~3.5웨이브(≈1.3배)인데 시뮬 게이트는 `farm_xp_ratio ≥ 2.0`을 요구. 비율 대부분이 "이탈 시 XP 소멸"에서 나와야 하고 이는 `enemy_hp_scale`·`spawn_density_scale`·적 속도와 강결합. 2.0을 숫자만으로 달성 가능한지 아무도 검증 안 함 → 실패 시 설계 변경.
- **오디오 · BGM** — 섹션 자체가 없음. B §12가 "BGM·믹스·에셋 방침은 [오디오] 섹션 소관"으로 3회 미루지만 그 섹션이 존재하지 않음. BGM 유무, 스테이지별 트랙, 절차적 합성 vs 파일 에셋, 믹스/덕킹, 볼륨·뮤트 UI 위치 전부 미정. 400KB 예산과 30~60초 제출 영상에 직결.
- **옵션 · 설정 화면** — F의 `opts{mute,volume,shake,flash,cbMode,damageNumbers}`, B의 `a11y.cvdMode`/`reduceFlash`/`screenShake`, `hud.shipImbueChips` 토글이 전부 존재하는데 **어디서 켜는지 정의된 화면이 없음**. E §1의 전역 상태 기계에 OPTIONS 상태가 없고, F는 "옵션 화면 '기록 삭제'"를 참조하나 E는 "타이틀 하단 기록 초기화". 색맹 모드가 접근 불가면 B §11 전체가 죽은 스펙.
- **픽업 소멸 규칙** — B §10은 픽업이 "자석 반경 밖에서는 스크롤 따라 하강만" 한다고 확정 → 아레나 하단으로 나가면? F의 `orbLifetimeSec: null`(만료 없음)과 D `phase_end_autocollect`/E `sweepPickupsOnPhaseEnd`(전량 회수)가 **화면 밖 픽업까지 회수하는지** 미정. 무-노가다 기둥과 "전환 손실 0" 약속의 성립 여부가 여기 달림.
- **넉백 모델** — `knockback`이 C·F 양쪽 무기 파라미터 어휘(닫힌 목록)에 있는데 **모델이 없음**: 단위(px/임펄스), 스크립트 경로를 도는 적(`movement_id`/`move.type`)에 어떻게 적용되는지, `column`·`anchor`·`pincer` 편대가 깨지는지, 보스/부위에 적용되는지. 어휘에 있으므로 AI가 값을 생성할 텐데 의미가 없음.
- **아트 · 스프라이트 방침** — F의 `enemies.json`에 `spriteId: "drifter"`가 있으나 스프라이트 레지스트리·에셋 파일·아틀라스를 정의한 섹션이 없음. B는 절차적 도형(중립 차콜 본체 + 속성 외곽선 + 글리프)을 전제 → 두 모델이 배타. 무기 12 + 진화 12의 시각 정체성, 보스 6종의 형상, 저작 주체(AI/코드) 미정. 400KB 예산 직결.
- **시뮬 산출물 `stagePar[i]`** — E §7·§9.3·§15가 `bossParGhost`(라이브 페이스 고스트)와 결과 화면 파 막대 6개의 **입력**으로 `stagePar[i]`를 시뮬 섹션에 계약으로 요구했으나, F §20의 텔레메트리 산출물(summary/weapons/elements/bosses/economy/deaths)에도 `certify` 지표에도 없음. E가 "이유 모를 죽음"을 제거하려 넣은 2개 장치가 데이터 소스 없이 떠 있음.
- **`swarm_total_scale[stage]`** — D §5.2가 `crisis_total: 60 × swarm_total_scale[stage]`로 참조하는데 §1.4 진행 곡선 표에 그 열이 없고 값도 없음. E의 `stage1CrisisScale: 0.5`가 같은 것인지 다른 것인지 불명(같은 개념에 이름 2개).
- **온보딩 트리거 모델** — E §12는 "0:12 Lv2 / 0:20 Lv3 / 0:45 Lv4 / 0:20–0:45 대비 웨이브"를 **시각**으로 서술하지만 실제 트리거는 XP. 파밍이 느리면 대비 웨이브가 속성 2개 이전에 도착해 "유레카" 레슨이 통째로 무산. 레벨업을 강제하는지, 웨이브가 레벨을 기다리는지, 스테이지 1만 XP를 조작하는지 미정 = 60초 온보딩의 실행 모델이 없음.
- **온보딩 2번째 드래프트가 레슨을 보장 못 함** — E §12의 Lv3 드래프트 = "남은 속성 2장 + **무기 1장**". 플레이어가 무기를 고르면 속성 1개뿐 → 대비 웨이브의 "키를 눌러본다 → 반전 → 유레카"가 성립 불가(속성 1개로는 좌/우 레인 배율이 뒤집히지 않음). E는 "이제 속성 2개 보유"를 무조건으로 서술. C의 `guarantee_new_weapon_until_slots: 2`가 오히려 그 무기 카드를 강제.
- **진화의 데이터 표현** — C의 `evolution.flags: ["distinct_targets","retarget_on_kill"]`은 **코드 분기 플래그**로, C 자신의 "코드에 없는 훅을 요구하는 콘텐츠 생성 금지"와 F의 "진화는 100% 데이터, `family`를 바꿀 수 있다" 모두와 충돌. 진화 12종이 각각 새 코드 경로를 요구하는지, 파라미터 오버라이드뿐인지 미정 = 4주 예산 산정의 직접 입력값.
- **중간보스의 소속 파일·스키마** — D §9.1은 `bosses.json`에 "중간보스 3종", D §4.1은 중간보스를 적 어휘(`movement_id`/`attack_id`)로 정의. F는 `enemies.json`의 `tier:"midboss"`와 `bosses.json`의 `tier:"mid"`를 **둘 다** 정의. 중간보스가 적 스키마인지 보스 스키마인지, 어느 파일인지 미정 = 로더·검증기 작성 불가.
- **문자열 · 로컬라이제이션** — F가 `descKey: "w.seeker"`/`"p.ghost"`를 참조하나 문자열 테이블 파일이 F의 9파일 매니페스트에 없음. C/D는 `name_ko`를 인라인 — 두 모델 배타. 무기 12 + 진화 12 + 패시브 12 + 적 15 + 보스 6의 이름·설명 1줄(드래프트 카드 필수 표기)을 누가 어디에 저작하는지 미정.
- **`afterimage`/`ghost` × 비행 중 유도탄** — "피격 후 N초간 적 조준·유도 타겟에서 제외"가 **이미 발사된 유도탄**에 적용되는지 미정. 적용되면 유도탄이 추적 포기(강력), 아니면 재조준 대상에서만 제외(약함). D의 `homing` 파라미터엔 재조준 주기 키가 없어 "재조준 대상" 개념 자체가 표현 불가. 이 패시브의 강도가 2배 차이 → 밸런서가 값이 아니라 규칙을 정하게 됨.

### minor (5)

- **HUD — 속성 투자 합계 예산** — A §3.4는 속성별 4핍만 표시. `element_level_cap_total`(8 또는 6)이라는 **런 전체 예산**이 화면 어디에도 없음. "이제 몇 발 남았나"가 안 보이면 4/4/0 vs 3/3/2 다이얼이 정보 기반 선택이 아니라 암산이 됨 — C가 이 캡에 부여한 설계 의도가 UI로 전달되지 않음.
- **폭탄 캐스트** — F의 `bomb.castSec: 0.4` 동안 이동 가능한지, 입력이 잠기는지, 스탠스 전환이 되는지 미정. A는 캐스트 개념 없이 "발동 중 무적 1.5초"만. 드래프트/상점/일시정지/보스 인트로 중 발동 차단 규칙도 없음.
- **실드 × i-frame** — A "실드 흡수 시에도 i-frame 1.0초 동일 발동" + C "실드는 순차 소모(1피격 = 1개)" → 실드 2장이어도 1초에 1장만 소모 = 사실상 2초 연속 무적. 의도인지 미정. `shieldPreservesNoHit`(무피격 유지)과 합치면 코인으로 2초 무적 + 무피격 보너스를 사는 경로.
- **보스 페이즈 전환 타이머 정지** — D의 `boss_timer_pauses_on_phase_transition: true` + `boss_phase_transition_gs: 1.5`(전환 중 무적 = "공짜 숨돌릴 틈")가 E §7 타이머 UX 표(정지 조건 = PAUSE만)와 F 스키마 어디에도 없음. 3분 허들의 실효 길이가 3:00인지 3:03인지 미정.
- **제목** — 여전히 미정(가제 TETRA). 제출물 3번 "게임 소개 문서"의 필수 항목이자 타이틀 화면·localStorage 키 접두어·README의 입력값.

