# 전사 감사 — 정본 v1.2 → data/*.json + check.mjs

> `__AMBIGUOUS__` **130칸** · 정본 패치 **59건** (blocker 14 · major 29 · minor 16)
> 기계적 결정 가능 **37건** · 설계자 판단 필요 **22건**

## 요약

## 1. 상호참조 교차검증 — 어휘는 전부 일치, 그러나 「값의 이중 거처」가 실제로 갈라졌다

**어휘 id 정합: 결함 0.** 9파일을 python으로 좌조인했다 — `weapons[].family`(12) ≡ `rules.passiveHooks` 키(12) · `elements.order`(4) ≡ stages/bosses의 전 element 값 · `themes[].id`(6+finale) ⊇ `themeDraw.pool`(6) ≡ `enemies.themeOnly`(6) ≡ `bosses[].themeId`(6) · `formations`(6) 전량 사용·미정의 0 · `passives[].stat`(12) 1:1 전단사. **참조 무결성: dangling 0건**(bulletId·emitterId·archetypeId·bossId·formationId 전부).

**★ 그러나 전사가 실제 데이터 모순 3건을 드러냈다 — 산문으로는 볼 수 없는 것들이다:**

- **★★ `bullets[].speed` ↔ `emitters[].speed`가 이미 갈라져 있다** (신규, ROUND4 미기재). `pelletS`(bullets.speed=150)가 **4개 이미터에서 3개 속도로 발사된다**: spitStraight 150 · antStraight 150 · **dartStraight 140** · **lancerStraight 120**. 18개 중 16개는 두 값이 정확히 중복되고 2개가 갈라진다. 이것이 결정적인 이유: 같은 탄이 이미터별로 다른 속도로 날아야 한다는 것은 `emitters[].speed`가 **유일 권위**임을 데이터가 증명하는 것이고, 그러면 `bullets[].speed`는 「덮어써지는 기본값」인데 **§9.3이 「기본값 폴백 금지」로 그 독법을 정면으로 금지한다** = 두 필드를 동시에 정합하게 만드는 독법이 존재하지 않는다. 게다가 **S6은 `enemies.json > emitters`「만」 검사**(§13.4 L3284)하므로 엔진이 `bullets[].speed`를 읽으면 **공정성 게이트가 엔진이 읽지 않는 수를 인증**한다. 오늘은 260 이하라 위반이 안 나지만, 그것이 결함을 숨긴다. `laser`/`zone`은 §8.5상 `speed` 파라미터가 **없어** beamCore의 `speed: 0`은 죽은 필드다 → **`bullets[].speed` 삭제가 유일한 정합 해**다.
- **★ `player.hpSegment`(20) ↔ `hud.hpBarSegCount`(5)** — hpMax 100에서 우연히 일치(20 = 100/5)하고, §11.2가 확정한 `maxhp` ×4 구매로 hpMax 140이 되는 순간 **7칸×20 vs 5칸×28**로 갈라진다. 로더도 check.mjs도 못 잡고(둘 다 유효 키) **기본값에서 일치하므로 테스트로도 안 잡힌다.**
- **★ S20 위반 9건** — `pincer ⟺ moveId=="strafe"`를 stages.waves × enemies.archetypes로 조인하면 04의 저작 웨이브 9개가 불법이다(forest `thornWeaver` arc/scatter ×5 · `flanker` scatter/lineH ×2, bog `flanker` lineH ×2). **정본이 옳고 데이터가 틀렸다**(§9.9.2의 근거 「측면기가 한 줄로만 지나가면 strafe 거동이 죽는다」가 ⟺의 역방향을 지지).

**부수 발견:** `stunMark`(유일한 stun 탄)를 참조하는 이미터가 **0개** — 보스 부위 이미터 66개가 전부 `__AMBIGUOUS__`이기 때문이다 → **스턴이 게임에서 도달 불가능한 콘텐츠**이고 S13·§12.4의 스턴 하한이 읽을 대상이 없다. / `moveId` 8종 중 **`charge`는 잡몹 사용자가 0**이고 유일한 사용자가 `mbLancer`인데 **중간보스 `moveId`의 인쇄 자리가 없다**(blocker) → 어휘 1/8이 인쇄 자리 없는 필드에만 걸려 있다.

**`__AMBIGUOUS__` = 130칸** (bosses 75 · stages 38 · enemies 14 · rules 2 · meta 1 · bullets/elements/passives/weapons **0**). 내역: 보스 부위 emitterIds **66** · finale roster unlockStageMin 15 · 중간보스 이미터 from/repeat/restSec 12 · midBossAtSec 7 · elitesAtSec 6 · skinId 6 · themeId 4 · eliteIndex 4 · 중간보스 hp 3 · leaveAfterSec 2 · themeOnly 2 · rules 2 · **meta.shop 1(단 1칸이 ~50개 값을 감춘다)**.

**★ 전사자 신고 1건은 거짓임을 확인했다** — weapons 전사자의 「pierce ✖ 5패밀리는 `base.pierce`가 없어 `eff.pierce = base.pierce + v`가 undefined를 읽는다」는 **틀렸다**: `passiveHooks`의 `pierce` 열이 식을 게이트하므로 hook.pierce=True인 6패밀리는 **전부 base.pierce를 갖는다**(검산 완료). 반면 같은 신고의 `areaKeys` → evo* 3건(fan `evoBlastRadius` · mine `evoClusterRadius` · nova `evoRing2Radius`가 `base`가 아니라 `evolution.params`에 산다)은 **참이다.**

## 2·3. 정본 패치 목록 / 중복·신규

**canonPatches 42건**: blocker 12 · major 20 · minor 10. `needsUserCall: false` **27건**(64% — 인쇄된 쪽 채택·백틱 제거·경로 정정 등 기계적) · `true` **15건**.

**ROUND4 기지 22건과의 관계**: 중복 **9건**(finale.armorCoreRatio · bombStockMax · 중간보스 hp · render 이중블록/루트자리 · boss.armorCoreRatio 유령스코프 · shop 스키마 · audio.bgm · leaveAfterSec · elite.perWaveChance · boss.healDrop). **신규 33건.** ROUND4의 게이트 산술 항목(m 배열 · runFarmDpsRatio 허용오차 · §13.5↔§13.5.1 · §16-#14 · §15-m3 · N3/N4 · specialist 궤적)은 전사 범위 밖이라 재확인만 했고 패치 목록에는 상위 3건만 실었다.

**★ 전사자들끼리 갈라진 곳 1건 = 진짜 설계자 콜**: `bombStockMax`. rules 전사자·ROUND4는 「`rules.bomb.stockMax` 존치」를, meta 전사자는 「`shop.<id>.stockMax`로 4개 통일 + `rules.bomb.stockMax` 삭제」를 권고한다. 양쪽 다 논거가 성립한다(전자 = 상점 없이도 존재하는 플레이어 상태의 상한 / 후자 = 같은 열의 나머지 3개와의 대칭 + 상점이 파는 것). **두 독립 전사자가 반대 방향을 권고했다는 사실 자체가 `needsUserCall: true`의 증거다.**

## 4. 전사가 드러낸 것 — 산문 리뷰 3라운드가 못 찾은 것

**핵심: 산문 리뷰는 「답이 있는가」를 검증하고, 전사는 「그 답을 정확히 한 번 받아적을 수 있는가」를 검증한다.** 130칸은 전부 **정본이 산문으로는 답을 아는데 데이터에 주소가 없는 자리**다. ROUND4가 22건을 잡고도 놓친 33건은 우연이 아니라 **방법의 구멍**이다:

1. **좌조인은 「두 값이 같은가」를 못 묻는다.** `bullets[].speed`↔`emitters[].speed`, `hpSegment`↔`hpBarSegCount`, `introSec`↔`bossEntrySec` — **셋 다 양쪽이 유효 키라 로더·감사·문법검사 전부 통과**한다. 값을 나란히 써 봐야만 보인다. `hpSegment`는 **기본값에서 일치해 테스트로도 안 잡히고** 상점 구매 시점에만 갈라진다.
2. **조인해야만 보이는 위반.** S20 9건은 `stages.waves` × `enemies.archetypes.moveId` 조인의 산물이다. 두 파일을 각각 읽는 리뷰는 구조적으로 못 본다.
3. **「표는 있는데 필드가 없다」는 타이핑할 때만 아프다.** `shop` 10항목의 **숫자는 전부 확정**인데 담을 이름이 없다(`defense +2`·`maxhp +10`이 산문). 리뷰는 「§11.2에 값이 있다 ✔」로 통과시킨다. 같은 형태가 `weapons[].base` 패밀리별 키 집합(**9/12 패밀리가 공통키를 드롭**하는데 정본은 `targetMode` 하나만 면책)에서 재현된다 — §9.3의 「누락 키 = 에러」가 **평가 자체가 불가능**하다.
4. **§21의 감사 축이 잘못됐다.** §21이 「✔」로 사인한 블록에서 결함이 나왔다: `stages.phase`(`crisisPerStage` 부재) · `bosses[].*`(중간보스 팔 미인쇄 · `summon`/`armorCoreRatio`가 표에만) · `render.*`(§12.2 두 번째 블록) · `passiveHooks`(컨테이너 형태 미확정). **`difficultyMul`은 점이 없어서** 점표기 리프만 추출한 §21·ROUND4의 좌조인을 **둘 다 통과했다** — 05가 점수 정수화 공식 전체를 존재하지 않는 키 위에 세웠는데도.
5. **산술 검산이 「해석의 분기」를 증명했다.** stages를 **블록 단위 필터**로 재집계하면 04의 전 체크섬을 한 자리도 안 틀리고 재현(S8 = 0.0%p)하고, **레코드 단위 필터**로 같은 데이터를 재집계하면 **S8이 4건 깨진다**(bog s1 78.6% = +8.6%p). 두 해석 중 어느 것도 데이터에 표현되어 있지 않다 — 웨이브 레코드 6필드에 블록 티어를 담을 자리가 없다. **리뷰는 「S8 통과 ✔」를 읽었고, 전사는 「S8의 통과 여부가 미정」을 증명했다.**
6. **연쇄가 콘텐츠를 죽인다.** 보스 이미터 id 부재(blocker) → `stunMark` 참조 0 → **스턴 메커닉 전체가 도달 불가** + S13·S7·S16이 읽을 대상 없음. 단일 결함의 사정거리는 그 결함을 신고한 절에서 보이지 않는다.

**정본의 건강도:** 값 자체는 놀랍도록 견고하다 — elements(16셀) · weapons(12×8) · passives(12×5) · bullets(10) · S24 HP 배분(오차 ≤0.78%) · S8(0.0%p) · R1~R7 전부 통과, 4개 파일은 `__AMBIGUOUS__` 0. **결함은 「값이 틀렸다」가 아니라 「값의 주소가 없거나 둘이다」에 집중**되어 있고, 42건 중 27건이 표·한 줄·백틱 제거로 닫힌다(**새 값 0 · 새 키 4**). 남은 진짜 저작은 `shop` 효과 파라미터 이름 · 보스 이미터 id 규칙 · 중간보스 스키마 팔 · `audio.bgm` 40값 · 아이콘 5종 — **5건뿐이다.**

## 파일 간 상호참조 검증

- ★★ 신규 blocker — `bullets[].speed` ↔ `emitters[].speed` 이중 거처가 **실제로 갈라졌다**: `pelletS`가 bullets.speed=150인데 dartStraight 140 · lancerStraight 120으로 발사된다(18개 이미터 중 16개는 정확히 중복, 2개 갈라짐). 같은 탄이 이미터별로 다른 속도여야 한다는 것은 emitters[].speed가 유일 권위임을 데이터가 증명하는 것이고, 그러면 bullets[].speed는 「덮어써지는 기본값」인데 §9.3이 「기본값 폴백 금지」로 그 독법을 금지한다 = 정합한 독법이 없다. S6은 `enemies.json > emitters`만 검사하므로 엔진이 bullets[].speed를 읽으면 공정성 게이트가 엔진이 읽지 않는 수를 인증한다. laser/zone은 §8.5상 speed 파라미터가 없어 beamCore의 speed:0은 죽은 필드 → bullets[].speed 삭제가 유일 해.
- ★ 신규 blocker 연쇄 — `stunMark`(유일한 stun 탄, status="stun")를 참조하는 이미터가 **0개**다. 보스 부위 이미터 66개가 전부 `__AMBIGUOUS__`(id 미저작)이기 때문 → **스턴 메커닉 전체가 도달 불가능한 콘텐츠**이고, S13(「스턴 탄은 patternSet[2]에만」) · §12.4의 스턴 텔레그래프 하한(≥1.5s) · `rules.status.stunMinDifficulty="hard"`가 전부 읽을 대상이 없다. 단일 결함(보스 emitterId)의 사정거리가 그것을 신고한 절에서는 보이지 않는다.
- ★ 신규 major — `moveId` 8종 중 **`charge`의 잡몹 사용자가 0**이다(enemies.archetypes는 7종만 사용: anchor·column·dive·orbitDrift·rearIn·strafe·weave). 유일한 사용자가 `mbLancer`(중간보스)인데 **중간보스의 `moveId` 필드는 §9.8 인쇄 블록에 존재하지 않는다** → 동결 어휘의 1/8이 인쇄 자리 없는 필드에만 걸려 있다. S3(moveId 8종 검사)는 union으로 통과하므로 이 구멍을 못 잡는다.
- ★ 신규 major — **S20 위반 9건**(stages.waves × enemies.archetypes.moveId 조인으로만 발견): forest `thornWeaver` arc/scatter ×5 · forest `flanker` scatter/lineH ×2 · bog `flanker` lineH ×2가 `moveId=="strafe"`인데 formation이 `pincer`가 아니다. §9.9.2의 근거(「측면기가 한 줄로만 지나가면 strafe 거동이 죽는다」)가 ⟺의 역방향을 지지하므로 정본이 옳고 데이터가 틀렸다. formationId는 XP·S8·S22에 영향 0이므로 수정 비용 0.
- ★ 신규 major — `player.hpSegment`(20) ↔ `hud.hpBarSegCount`(5)가 hpMax=100에서 **우연히 일치**(20 = 100/5)하고 §11.2가 확정한 `maxhp` ×4 구매(hpMax=140)에서 **7칸×20 vs 5칸×28**로 갈라진다. 둘 다 유효 키라 로더·check.mjs가 못 잡고, **기본값에서 일치하므로 테스트로도 안 잡힌다.** §9.4.1의 근거 열이 이미 「hpMax가 140까지 커져도 칸 수는 5」로 답을 확정했으므로 `hpSegment`는 그 채택 이전 모델의 잔재다.
- 어휘 id 정합 — **결함 0 (전수 검증 완료)**: `weapons[].family`(12) ≡ `rules.passiveHooks` 키(12) · `elements.order`(4) ≡ stages/bosses의 전 element 값 · `themes[].id`(sea·glacier·volcano·desert·forest·bog·finale) ⊇ `themeDraw.pool`(6) ≡ `enemies.themeOnly`(6) ≡ `bosses[].themeId`(6) · `formations`(6) 전량 사용 · `passives[].stat`(12)와 스탯 1:1 전단사 · `passiveHooks`의 rateKey/countKey 전부 `base` 안.
- 참조 무결성 — **dangling 0건**: bulletId(9/10 참조) · emitterId(archetype 14 + boss 4 = 18/18) · archetypeId(waves·roster 전량) · bossId(7/7) · formationId(6/6). 미참조는 정당한 2건뿐: `swarmChaff`/`swarmLancer`(위기 세션 전용, S9가 강제) · `stunMark`(위 blocker의 결과).
- ★ 전사자 신고 1건 **반증**: weapons 전사자의 「pierce ✖ 5패밀리(orbit·aura·mine·barrage·nova)는 `base.pierce`가 없어 `eff.pierce = base.pierce + v`가 undefined를 읽는다」는 **거짓**이다. `passiveHooks`의 `pierce` 열이 식을 게이트하므로 hook.pierce=True인 6패밀리(forward·fan·seeker·lance·omni·drone)는 **전부 base.pierce를 보유**한다(검산 완료). boomerang은 base.pierce=-1(무제한)인데 hook off = 정합. **이 신고는 목록에서 제외했다.**
- ★ 같은 전사자 신고의 `areaKeys` 3건은 **참으로 확인**: fan `evoBlastRadius` · mine `evoClusterRadius` · nova `evoRing2Radius`가 `base`가 아니라 `evolution.params`에 산다 → §9.6.1의 식 `eff[areaKey] = base[areaKey] × (1+v)`가 undefined를 읽는다. 나머지 9패밀리의 areaKeys는 전부 base 안.
- ★ 전사자 간 **의견 충돌 1건 = 진짜 설계자 콜**: `bombStockMax`. rules 전사자·ROUND4는 「`rules.bomb.stockMax` 존치 + §11.2를 인용으로 강등」, meta 전사자는 「`shop.<id>.stockMax`로 4개 통일 + `rules.bomb.stockMax` 삭제」. 양쪽 논거가 모두 성립하며(상점 없이도 존재하는 상태의 상한 vs 같은 열 나머지 3개와의 대칭) **두 독립 전사자가 반대 방향을 권고했다는 사실이 needsUserCall의 증거다.**
- 산술 증명 — stages를 **블록 단위 필터**로 재집계하면 04의 전 체크섬을 한 자리도 안 틀리고 재현(S8 = 전 6테마×5스테이지 0.0%p)하고, **레코드 단위 필터**로 같은 데이터를 재집계하면 **S8이 4건 깨진다**(sea s1 73.6% · volcano s1 73.5% · bog s1 78.6% · glacier s2 66.7%). 두 해석 중 어느 것도 데이터에 표현할 자리가 없다(웨이브 레코드 6필드에 블록 티어 자리 없음) → **S8의 통과/실패가 미정이다.**
- `rules.json` 루트 = 18키(schemaVersion + 17 스코프)로 산문과 일치 확인 — 단 §9.4의 **인쇄 블록은 16개**만 인쇄하고 `render`에 자리표시자를 안 줬다(ROUND4 기지). 전사자는 산문의 17개를 채택했다.
- `weapons[].base` 패밀리별 키 집합 실측 — **12개 중 9개가 공통 9키를 드롭한다**: lance −4 · orbit −6 · aura −8 · mine −8 · barrage −6 · omni −3 · drone −3 · nova −8. 정본 §9.5는 `targetMode` 하나만 면책하므로 **§9.3의 「누락 키 = 에러」가 평가 불가능**하고, 현 data의 패밀리별 base는 03 §2의 표(정본이 아님)에서 왔다.

## 정본 패치 목록

### ★ 설계자 판단 필요

#### [blocker] `meta.json > shop (컨테이너 형태 + 필드 집합 + 효과 파라미터 이름 10항목)`

- **채택안**: §11.2에 **JSON 블록을 인쇄**하라(C-9: 채택은 인쇄가 아니다). 형태 = **id를 키로 하는 객체**(정본이 인쇄한 유일한 shop 경로가 §6.3의 `shop.timeToken.addSec`이므로 객체를 택하면 기존 인쇄와 충돌 0). 스키마: `shop.<id> = {basePrice, growth, maxPurchases, iconId, stockMax?, <효과 파라미터>}`. 효과 파라미터 이름(전부 기존 산문의 직역, 새 모델 0): `reroll.addStock:1` · `potion.healPct:0.35` · `bomb.addStock:1` · `shield.addStock:1` · `timeToken.addSec:30`+`addStock:1` · `defense.addDefense:2` · `maxhp.addHpMax:10`+`healsSameAmount:true` · `movespeed.addMoveSpeedPct:0.06` · `magnet.addMagnetPct:0.30` · `resist.statusDurationPct:-0.20`. 스택 규칙은 표의 「상한」 열이 이미 가산을 확정한다(0+2×4=8 ✔ / 100+10×4=140 ✔ / +6%×3=+18% ✔ / 90×(1+0.30×3)=171 ✔ / −20%×3=−60% ✔) → 한 줄이면 유도 가능.
- **근거**: C-10의 유일한 위반(7개 축약 중 6개는 해소). **숫자 10항목 ×3은 전부 확정되어 있는데 담을 이름이 없다** — `defense +2`·`maxhp +10`·`movespeed +6%`·`magnet +30%`·`resist −20%`가 「효과」 열의 산문이고 grep `"basePrice"` = 0회. meta 전사자는 발명을 거부해 `"shop": "__AMBIGUOUS__"` 1칸으로 두었고 **그 1칸이 약 50개 값을 감춘다.** needsUserCall: 컨테이너 형태는 §6.3이 기계적으로 결정하나 **효과 파라미터 이름 5개는 정본에 존재한 적이 없어 저작이 필요**하다. ROUND4 blocker 중복 — 다만 「§6.3(객체) ↔ §11.2의 id 열(배열)이 서로 다른 형태를 인쇄한다」는 대립 구도는 전사가 처음 확정했다.

#### [blocker] `bosses[].parts[].patternSet[i].emitterIds (보스 부위 이미터 66개의 id + 거처)`

- **채택안**: ① 거처를 한 줄로 확정하라 — 권고: **enemies.json > emitters**(§9.7의 문장 유지, 04-§9.4의 「bosses.json에 전개된 결과」 문장 삭제). ② §9.4에 **id 명명 규칙**을 인쇄하라 — 권고 `{bossId}{PartIdPascal}P{1..3}`(예: `mantaThrusterP1`), §0.3의 camelCase 준수 → 66개가 규칙의 인스턴스가 되어 새 저작 0이고 check.mjs가 정적 검사 가능. ③ §9.8의 예시 id 3개(`mantaStraight1`·`mantaStraight2`·`mantaFan3`)는 **삭제**하라 — 04 §9.4의 파생 규칙(「type은 절대 안 바뀐다」)과 모순되어(straight 시드의 3번째가 `mantaFan3`) 계속 오독을 초대한다.
- **근거**: 거처가 **3곳에서 3개의 답**을 낸다(정본 §9.7 = enemies.json / 04-§9.4 = bosses.json / 04-§3.3 = enemies.json 함의). 더 치명적으로 **04-§10.2의 시드 이미터 표에 `id` 열이 아예 없어** 전개된 66개의 id가 전 코퍼스 어디에도 저작되지 않았다 → 거처를 정해도 전사가 물리적으로 불가능하다. 66칸이 `__AMBIGUOUS__`(전체 130칸의 51%). 연쇄: §9.3 참조 무결성 66건 실패 + S7·S13·S16이 읽을 대상 없음 + **`stunMark`(유일한 stun 탄) 참조 0 → 스턴 메커닉 전체가 도달 불가**. needsUserCall: 명명 규칙은 설계 선택이다.

#### [blocker] `bosses[tier=="mid"] — hp · element · moveId · moveParams · patternSet · shapeId · radius · contactDmg 의 필드 경로`

- **채택안**: §9.8에 **`tier: "mid"` 판본의 JSON 블록을 따로 인쇄**하라 — 지금 인쇄된 것은 union 타입의 **스테이지 보스 팔 하나뿐**이고 「같은 스키마의 축약형」이라는 산문 한 줄이 나머지 팔을 정의하려 한 것이 결함의 정체다. 그 블록이 ⓐ `hp`·`element`를 루트 필드로 ⓑ `moveId`+`moveParams`(§8.4 어휘)를 루트 필드로 ⓒ `movePattern`/`movePatternParams`는 `tier ∈ {stage, final}` 전용임을 명시하고 ⓓ S3에 「`tier=="mid"` ⟺ `moveId` 보유 ⟺ `movePattern` 부재」 동치를 추가한다.
- **근거**: `hp`는 §8.9·§13.6.4(루트) ↔ §9.8 인쇄 블록(`core.hp`)이 갈리고 — **`moveId`/`moveParams`는 더 나쁘다: 어느 절도 경로를 말한 적이 없다**(§8.9는 표의 열 이름으로만 존재, §9.8 필드 집합에 없음) → 쓰면 미지 키, 안 쓰면 중간보스가 움직이지 않는다. ★ 전사가 드러낸 연쇄: **`moveId` 8종 동결 어휘 중 `charge`의 잡몹 사용자가 0이고 유일한 사용자가 `mbLancer`**다 → 어휘의 1/8이 인쇄 자리 없는 필드에만 걸려 있는데 S3는 union으로 통과시킨다. 값(720/600/880)은 확정이고 거처만 미정이라 3칸이 `__AMBIGUOUS__`. ROUND4 blocker 중복(hp만) — moveId/moveParams 축은 신규.

#### [blocker] `stages.themes[].waves[] — 블록 티어 필드 부재 (스테이지 필터링의 단위)`

- **채택안**: 웨이브 레코드에 **`unlockStageMin: int`**(= 그 레코드가 속한 블록의 티어) 1필드를 신설해 §9.9의 인쇄 블록에 넣어라. 그러면 레코드 단위·블록 단위 필터가 **같은 결과**가 되고 S8의 「저작 리스트」가 「waves[] 중 unlockStageMin ≤ s인 레코드」로 한 문장에 닫힌다. 동시에 §8.6의 「roster[].unlockStageMin」(아키타입 해금)과 웨이브의 그것이 **다른 축**임을 §9.9에 한 줄 명시하라.
- **근거**: ★ 산술로 증명된 분기. **블록 단위**로 재집계하면 04의 전 체크섬을 한 자리도 안 틀리고 재현(S8 = 전 6테마×5스테이지 **0.0%p**, Σ XP·S22 전부 일치). **레코드 단위**로 같은 데이터를 재집계하면 **S8이 4건 깨진다**(sea s1 73.6% · volcano s1 73.5% · **bog s1 78.6% = +8.6%p** · glacier s2 66.7%)이고 개체 수도 갈린다(sea s1 80 → 125). 인쇄된 6필드에는 블록 소속을 담을 자리가 없다 — 04의 블록은 표의 행 묶음일 뿐 데이터에 존재하지 않는다. 연쇄: §8.2.1의 조우 편차 · §8.7의 「sea s1 = 10웨이브, 최대 1.4회차 순환」 · §13.2-⑤의 farmXpRatio 산술이 **전부 블록 전제**이며, 레코드 단위로 읽으면 sea s1이 17웨이브가 되어 mobPhaseMaxWaves 14에 잘려 **순환이 아예 발생하지 않아 farmXpRatio 게이트의 전제가 소멸**한다. needsUserCall: 새 키 1개 추가.

#### [blocker] `rules.audio.bgm`

- **채택안**: §7.10 또는 §9.4에 `audio.bgm` JSON 블록을 인쇄하고 축약을 그 절 번호로 좁혀라(`"...§7.10 전 키..."`). 인쇄할 것: 8개 trackId의 **id 어휘**(테마 6종이 `themes[].skinId`와 1:1인지 별도 어휘인지 포함) + 5필드(`rootNote, scale, bpm, patternIdx, layers`)의 타입 + 8트랙 × 5필드 = **40개 값**. `scale`·`patternIdx`가 닫힌 enum인지 자유값인지도 확정할 것.
- **근거**: §9.4가 `"bgm": { "...트랙 8종..." }`으로 축약하는데 이 축약은 **어느 절도 가리키지 않는다**(C-10 위반). 필드 집합은 §7.10의 산문에만 있고 §9 밖이므로 C-7의 확정력이 없으며, trackId 8종의 id 어휘가 열거되지 않았고 40개 값이 전 코퍼스에 0개다. 값을 지어내지 않고는 전사가 불가능해 `"bgm": "__AMBIGUOUS__"`로 남았고 **이 키 하나 때문에 rules.json이 로드되지 않는다** → ROUND4는 major로 잡았으나 blocker가 옳다. needsUserCall: 40개 값의 저작이 필요하다.

#### [blocker] `enemies.emitters[hammerFan|hammerZone|lancerLaser|nestAimed].from / .repeat / .restSec`

- **채택안**: 04-§7.2의 중간보스 이미터 표에 `from`·`repeat`·`restSec` 3열을 추가하고 4종 전부의 값을 인쇄하라. 「보스만 repeat/restSec를 쓴다」(04-§3.2 L264)가 참이라면 repeat > 1 / restSec > 0인 실제 값이어야 하고, 거짓이라면 그 괄호 문장을 삭제하고 잡몹과 동일하게 1/0으로 확정하라.
- **근거**: 중간보스 이미터 4종에 §8.5·§9.7이 확정한 필수 공통 키 3개의 값이 없다 → §9.3의 「누락 키 = 에러, 폴백 금지」로 그대로 로드 실패. 잡몹의 `repeat:1, restSec:0`을 복사하는 것이 불가능한 이유가 결정적이다: **04-§3.2 L264가 「보스만 repeat/restSec를 쓴다」고 명시**해 보스는 1/0이 아닌 값을 갖는다는 뜻이므로 복사는 정본을 정면으로 어긴다. 12칸이 `__AMBIGUOUS__`. needsUserCall: 값의 저작이 필요하다.

#### [blocker] `C-2 / C-8 — §8.6의 명시적 저작 위임의 지위`

- **채택안**: C-2/C-8에 한 줄 확정: 「정본이 §8.6과 같이 **명시적으로 저작을 위임한 범위**에서는, 위임받은 섹션의 표가 C-8의 「정본이 인쇄한 값」과 동등한 지위를 갖는다.」 인정하지 않는다면 정본이 적·탄·이미터의 전 값을 §9.7에 직접 인쇄해야 하며 그 경우 §8.6 L1185의 위임 문장을 삭제해야 한다.
- **근거**: 「섹션 = 정본 인용판, 값의 출처가 아님」과 정본 §8.6(「로스터·시그니처의 구체 편성은 「적·스테이지」 섹션 소관이며 정본은 어휘·규칙·개수만 확정한다」)이 **정면 충돌**한다. C-7이 §9.7의 stalker/homing2 블록을 예시로 무효화하므로 **정본 자신에는 적 17종·탄 10종·이미터 18종의 확정값이 단 하나도 인쇄되어 있지 않다.** 「섹션은 값의 출처가 아님」을 엄격 적용하면 bullets.json·enemies.json이 **schemaVersion을 제외한 전량 `__AMBIGUOUS__`**가 되어 산출물이 성립하지 않는다. 전사자는 §8.6의 위임을 근거로 04의 표를 C-8과 동등하게 취급했고 **이 판단이 뒤집히면 두 파일을 재작성해야 한다.** 헌법적 판단이므로 설계자 콜.

#### [major] `rules.bomb.stockMax ↔ meta.shop 의 스톡 상한 4개 (rerollStockMax · bombStockMax · shieldStockMax · timeTokenStockMax)`

- **채택안**: ★ **전사자 2인이 반대 방향을 권고했다 — 설계자가 정해야 한다.** 안 A(rules 전사자·ROUND4): `rules.bomb.stockMax` 존치 + §11.2의 `bombStockMax`를 그것의 인용으로 강등. 근거 = 폭탄 스톡 상한은 상점이 없어도 존재하는 **플레이어 상태의 상한**이고 §2.6(런 시작 상태)이 이미 `bomb` 스코프로 인용한다. 안 B(meta 전사자): 네 항목 전부를 **`shop.<id>.stockMax`로 통일**하고 `rules.bomb.stockMax` 삭제(`rules.bomb`에는 발동 규칙 5키만 남긴다). 근거 = 같은 열의 나머지 3개는 다른 거처가 없어 `shop` 소속인데 **`bombStockMax`만 `rules.bomb`와 겹치는 비대칭 열**이고, 상한은 상점이 파는 것이다.
- **근거**: 같은 값(3)에 **두 이름·두 스코프·두 파일** → §9.3의 「미지 키 = 에러」상 **둘 중 하나는 반드시 로드 실패**. 02는 `bomb.stockMax`로 HUD 스톡 pip을, 03·05는 `bombStockMax`로 상점 회색 처리를 세워 **지금 세 섹션이 갈라져 있다**(02↔05 경계면 실패의 세 번째 사례). ★ needsUserCall의 근거: **두 독립 전사자가 각자 자기 파일을 쓰면서 반대 결론에 도달했고 양쪽 논거가 모두 성립한다.** 안 B는 `bombStockMax` 충돌과 스톡 상한 4개의 분산을 한 번에 닫지만 `rules.bomb` 스코프를 쪼갠다. ROUND4 blocker 중복.

#### [major] `stages.themes[].waves[].formationId — S20 위반 9건 (forest thornWeaver ×5 · forest flanker ×2 · bog flanker ×2)`

- **채택안**: 9개 웨이브의 `formationId`를 **`pincer`로 정정**하거나, `thornWeaver`/`flanker`를 쓰는 웨이브를 재저작하라. 정본이 옳고 데이터가 틀렸다.
- **근거**: ★ 신규 — 두 파일을 조인해야만 보인다(stages.waves × enemies.archetypes.moveId). §9.9.2/S20의 `pincer ⟺ moveId=="strafe"`를 문면대로 적용하면 04의 저작 웨이브 9개가 불법이다: forest `thornWeaver`(arc ×3 / scatter ×2) · forest `flanker`(scatter 1 / lineH 1) · bog `flanker`(lineH ×2). ⟺의 **역방향**(「모든 strafe는 pincer여야」)이 물리는 것이며 §9.9.2의 근거(「측면기가 한 줄로만 지나가면 strafe 거동이 죽는다 → `boomerang`의 라인 정렬 시너지 논거가 무의미해진다」)가 그 방향을 **지지**한다. 수정 비용: formationId는 XP·S8·S22에 영향 0이므로 체크섬 불변. needsUserCall: `thornWeaver`/`flanker`가 pincer 외 편대를 쓸 수 없다는 것은 저작 다양성에 대한 강한 제약이라 설계자가 재저작 방향을 정해야 한다.

#### [major] `rules.visual.a11y.{cbMode, reduceFlash, screenShake} ↔ localStorage opts.{cbMode, reduceFlash, shake} ↔ §7.3의 `options.cbMode``

- **채택안**: 정본이 한 문장으로 확정하라: 「`rules.visual.a11y.*` = **공장 기본값**(로더 필수 키) · `opts.*` = **사용자 오버라이드**(localStorage) · 런타임 유효값 = `opts.x ?? rules.visual.a11y.x`」 + 「이것은 폴백이 아니라 **오버라이드**이므로 §9.3의 「기본값 폴백 금지」(= JSON 로더 규칙)와 충돌하지 않는다」를 명기. 동시에 §14의 `opts.shake` → **`opts.screenShake`**로 개명하고 §7.3의 제목 `options.cbMode` → `opts.cbMode`로 정정.
- **근거**: ★ 신규(grep 0회). 같은 3개 옵션이 두 파일에 인쇄되고 **세 개의 이름**으로 참조된다(`visual.a11y.screenShake` / `opts.shake` / 유령 스코프 `options.cbMode`). 결함 셋: ⓐ **런타임 읽기 경로가 정본 어디에도 없다** — §9.3은 「기본값 폴백 금지」인데 §14는 localStorage 실패 시 「기본값으로 조용히 시작」이라 하고 **그 기본값의 거처를 말하지 않는다** ⓑ 이름이 갈라졌다(`screenShake` vs `shake` = §21-A12 클래스) ⓒ `options`는 어느 파일에도 없는 유령 스코프(§21-A1 `stance` 클래스). 인쇄된 대로 전사했으나 **이 3키가 실제로 읽히는 값인지 죽은 값인지 알 수 없다.** needsUserCall: 오버라이드 모델의 도입은 §9.3의 폴백 금지와의 관계를 설계자가 선언해야 한다.

#### [major] `enemies.emitters[].from — 어휘 미폐쇄`

- **채택안**: §8.5에 `from` 어휘를 폐쇄해 인쇄하고(잡몹은 전부 `self`, 보스는 부위에서 발사하므로 최소 `["self", "part"]` 검토) **S3의 어휘 검사 목록에 `from`을 추가**하라. 04-§3.2가 이미 14종 전부에 `self`를 인쇄했으므로 잡몹 측 값은 존재한다.
- **근거**: `from`은 §8.5의 필수 공통 파라미터인데 값 어휘가 전 코퍼스 어디에도 폐쇄되어 있지 않다 — `"self"`는 C-7이 예시로 무효화한 §9.7 블록에 단 1회 등장할 뿐이다(전 코퍼스 2회, 둘 다 그 블록). S3가 검사하지 않으므로 AI·구현자가 임의 문자열을 생성해도 로드가 통과한다. ★ 이것은 정본이 `bullets[].shape`를 02-B4로 폐쇄하고 S3에 추가한 것과 **정확히 같은 클래스**이며 그때의 근거(「§13.4 S3의 어휘 검사 목록에 shape가 없어 AI가 임의 문자열을 생성할 수 있었다」)가 글자 그대로 `from`에 적용된다. needsUserCall: 보스 이미터의 `part` 팔 존재 여부는 위 보스 이미터 blocker의 결정에 종속된다.

#### [major] `stages — 위기 세션 서브웨이브 편성의 거처`

- **채택안**: `stages.phase`에 **`crisisWaves: [{formationId, archetypeId, count, element, spawnEdge, eliteIndex}]`** 6~12행을 신설하라 — ★ **웨이브 레코드와 같은 스키마를 재사용**하면 새 스키마 0이고, `element`를 「테마 주입」으로 둘지는 `crisisElementRule`이 이미 소유하므로 **finale 로테이션도 같은 필드로 표현**된다(→ `finaleCrisisRotating` 불리언이 필요 없어져 아래 항목과 한 번에 닫힌다).
- **근거**: ★ 04가 **저작한 표가 들어갈 필드가 어느 인쇄 블록에도 없다.** `themes[].waves[]`는 잡몹 웨이브 리스트이고(S8이 그것만 읽는다) `phase`에는 스칼라 3개뿐이다 → 「1파=arc, 2파=vWedge 교대」·「9 `swarmChaff` + 1 `swarmLancer`」·「spawnEdge top 고정」이 **데이터에 존재하지 않는다 = 코드로 간다** → C-2 위반이고 「밸런싱은 오직 숫자」가 위기 세션에서만 성립하지 않는다(편대를 바꾸려면 .js를 연다). S26(「위기 서브웨이브 1파의 Σ count ≤ swarmConcurrentMax」)도 **읽을 데이터가 없어 안 써진다.** 전사자는 **자리를 찾지 못해 `__AMBIGUOUS__`조차 놓을 곳이 없었다** — 위기 편성은 전사되지 않은 채 남아 있다(실측: `swarmChaff`/`swarmLancer`가 어느 웨이브에도 없음). S8의 제외 대상이 이미 새떼를 빼므로 S8 영향 0. needsUserCall: 새 필드 추가.

#### [major] `stages.themes[].finaleCrisisRotating`

- **채택안**: **키를 없애고 `crisisElementRule`을 `"themePure" | "finaleRotating"` 2값 어휘로 넓혀라**(엔트리가 소유). S9(「crisisElementRule 준수 — 최종만 로테이션」)의 문면을 함께 고쳐라. 대안: 전 테마에 `finaleCrisisRotating: false` 인쇄.
- **근거**: 조건부 필드 집합이 **선언 없이** 존재한다. C-7이 sea 엔트리로 `themes[]`의 필드 집합을 확정한다면 이 키는 **미지 키**이고, 필드 집합의 일부라면 나머지 6테마가 **누락 키**다 — §21-A8이 「양방향 실패」라 부른 형태. §9.3은 폴백 금지이므로 「없으면 false」로 읽을 자유가 로더에 없다. 권고 방향의 근거: 로테이션은 **`crisisElementRule`의 예외**라고 §8.16이 이미 말했으므로 **같은 축의 값**이지 별개의 불리언이 아니다. needsUserCall: 두 모델(어휘 확장 vs 불리언 전 테마 인쇄)이 모두 성립하고 위 `crisisWaves` 결정과 연동된다.

#### [major] `§3.1 데미지 공식의 항 — weapons[].base.falloff(aura) · rearBias(omni) · evolution.params.evoSecondaryDmgMul(fan·mine·nova)`

- **채택안**: §3.1의 1항을 「base = `w.dmg` × Π(패밀리 지역 배율)」로 재인쇄하고 지역 배율의 **폐쇄 목록 = {falloff, rearBias, evoSecondaryDmgMul}** 3종임을 명시하라(항이 하나 늘지만 이미 실재하는 항을 이름 붙이는 것). 동시에 §9.5가 `falloff`·`rearBias`의 **의미**를 소유하게 하라(지금은 03에만 정의가 있는데 §17은 03에 「숫자」만 위임했다). 새 값 0.
- **근거**: §3.1의 플레이어→적 공식은 5항으로 **동결**됐고(§3 제목: 「구조 동결 — 항 추가/제거 금지」) 1항은 「base = `w.dmg` // 무기 레벨 행에서 읽음」이다. 그런데 무기 계약에 **데미지를 곱하는 키가 3종** 있다 — `falloff`(가장자리 데미지 배율) · `rearBias`(후방 탄 데미지 배율, Lv4에서 1.30) · **`evoSecondaryDmgMul`(★ 정본 §9.5 L1921이 직접 「진화 2차 피해의 데미지 배율」로 정의)**. 이 셋이 5항 어디에 들어가는지가 정본에 없고, 1항이 「무기 레벨 행에서 읽음」이라 못박아 update 함수가 미리 접어 넣는 독법도 명문 근거가 없다. C-4는 「데미지 공식의 항」을 **구조**로 분류한다 = 이 3키는 사실상 6번째 항인데 구조 동결 선언 밖에 있다. 산술적으로는 순수 곱이라 값이 안 변하지만 **구현자가 첫날 「falloff는 dmgMul 앞인가 뒤인가, elem에도 걸리는가」를 즉석 결정하게 된다**(C-6). needsUserCall: §3이 명시적으로 동결된 구조라 설계자 승인이 필요하다.

#### [major] `rules.passiveHooks.{fan|mine|nova}.areaKeys → evoBlastRadius · evoClusterRadius · evoRing2Radius`

- **채택안**: §9.6.1의 적용 식을 「`eff[areaKey] = src[areaKey] × (1 + v)`, `src = base ∪ (evolved ? evolution.params : {})`」로 재인쇄해 evo* 파라미터가 유효 파라미터 공간에 합류함을 명문화하라. 또는 areaKeys에서 evo* 3키를 빼고 「진화 2차 반경은 areaMul의 대상이 아니다」를 명시하라. 새 값 0.
- **근거**: **검산으로 확인**: areaKeys 12행 중 정확히 3개가 `base`에 없는 키를 가리킨다(fan `evoBlastRadius` · mine `evoClusterRadius` · nova `evoRing2Radius`가 `evolution.params`에 산다). 나머지 9행의 areaKeys와 12행의 rateKey·countKey는 **전부 base 안**임을 기계 검증했다. 그대로 구현하면 coil이 진화한 fan·mine·nova의 2차 반경에 `base["evoBlastRadius"] = undefined`를 읽는다 → 침묵 무효이거나 런타임 에러이며 **어느 쪽인지 정본이 답하지 않는다.** 이것은 H2(「areaMul은 닿는 범위만 키운다」)가 의도한 상태와도 다르다 — 진화 2차 폭발 반경이야말로 「닿는 범위」다. needsUserCall: 「패시브 훅이 진화 파라미터를 만질 수 있는가」는 지금 미정인 구조적 질문이다.

#### [major] `passives[].desc (12행 — 필드 자체가 없다)`

- **채택안**: §9.6의 인쇄 블록 필드 집합에 **`desc`를 추가**하고 12행을 저작하라(weapons[]와 동형).
- **근거**: C-7상 §9.6의 인쇄 블록이 필드 집합을 확정한다 → `passives[]`에 desc가 없다(전사 확인: 필드 = id·name·stat·values). 그런데 ⓐ §11.1은 passive 카드의 **필수 표기**를 「이름 · **훅 1줄** · Lv2→Lv3 수치」로 확정하고 ⓑ §9.6.1의 H4는 「카드가 훅 1줄을 반드시 표시하므로 정보는 화면에 있다」를 **죽은 조합 3종을 숨기지 않는 근거**로 삼는다(= 훅 1줄이 게이트 논거의 일부다) ⓒ §0.3·§9.2·§16-M13은 「문자열 테이블 폐기 → 각 엔티티에 name/desc 한국어 인라인」을 확정한다. **weapons[]는 desc를 갖는데 passives[]만 없다** → 「훅 1줄」의 한국어 12개의 거처가 9파일 어디에도 없고 남은 유일한 자리는 코드의 stat → 한국어 매핑 테이블 = §9.2가 폐기한 모델의 부활 + C-4 위반. needsUserCall: 12개 한국어 문자열의 저작이 필요하다.

#### [major] `weapons[].base.hitCooldownSec — lance·omni·drone에 부재 / forward·fan은 0.0 선언`

- **채택안**: §9.5가 `hitCooldownSec`의 **의미**를 소유하고(03 §2 L157의 3줄을 정본으로 승격), 계약 12행에서 이 키의 유무를 패밀리별로 확정하라 — 재히트 개념이 없는 패밀리는 키를 갖지 않는다고 명시하거나 12/12가 선언하도록 통일하라. 위 「base 필수 키 집합」 표에서 한 번에 닫힌다.
- **근거**: 계약의 적용이 **비대칭**이다 — forward·fan은 pierce를 갖는 투사체 패밀리이면서 `hitCooldownSec: 0.0`을 선언하고, lance(pierce 3→8)·omni·drone은 같은 성질인데 **키 자체가 없다**(실측 확인). §9.3은 「누락 키 = 에러, 폴백 금지」이므로 키 없는 3패밀리의 「한 대상 1회」 규칙은 **데이터 표현이 없다** → lance.js·omni.js·drone.js가 하드코딩한다(C-6). 역방향으로는 **정본 자신의 논거**(§9.5 L1918: 「0으로 선언해야 한다 = 죽은 필드 … AI가 그 0에 의미를 부여하려 든다」)가 forward·fan의 0.0에 그대로 적용된다 = 5패밀리 중 어느 쪽이 옳은지 정본이 답하지 않는다. 부수: 이 키의 **의미**의 거처를 03이 스스로 「이 문서」라 선언했는데 §17이 03에 위임한 것은 「실제 숫자」이지 어휘의 의미가 아니다(C-3·C-5). needsUserCall: 패밀리별 유무는 설계 판단.

#### [major] `rules.hud.icons (닫힌 9종) ↔ shop 10항목의 「표시 필수: 아이콘」`

- **채택안**: `hud.icons`에 **5종을 추가해 14종으로 어휘를 다시 닫아라** — 권고: `potionFlask` · `defenseChevron` · `hpCross` · `bootWing` · `magnetHorseshoe`(전부 §9.10의 절차적 도형 방침 그대로, 아트 0바이트). `shop.<id>.iconId`를 §11.2의 인쇄 블록에 필드로 넣고 10항목의 값을 인쇄하라. `check.mjs`에 **S 항목 1개**: 「`shop`의 전 항목의 `iconId` ∈ `hud.icons`」.
- **근거**: ★ 신규. §11.2와 05 §7.2가 상점 **10줄 전부**에 아이콘을 **필수**로 요구하는데 `hud.icons`는 **닫힌 9종**이고 상점 항목에 대응하는 것은 **5개뿐**(`rerollArrows`·`bombRound`·`shieldHex`·`tokenClock`·`resistShield`). **`potion`·`defense`·`maxhp`·`movespeed`·`magnet` 5항목의 아이콘이 어휘에 없다**(나머지 4종은 픽업·상태이상). 어휘가 닫혀 있으므로 선언하면 미지 값, 안 그리면 「표시 필수」 위반 = §21-A8의 양방향 실패 → 구현자가 5종을 발명한다 = **어휘를 닫은 바로 그 조항이 발명을 강제하는 자리를 만들었다.** §21·ROUND4 모두 미검출(§21은 점 표기 리프만 좌조인했고 `icons`는 **배열 원소**라 어느 축에도 안 걸린다. ROUND4는 `icons`를 1회도 언급하지 않는다). needsUserCall: 아이콘 5종의 저작.

#### [major] `§7.4 「보스 대형 패턴 → 텔레그래프 1.50」`

- **채택안**: 셋 중 택1: ⓐ 보스 부위 이미터 **전부에 1.50** 적용 ⓑ 「대형」을 판별하는 **데이터 필드**를 신설 ⓒ **렌더 규격으로 강등**(S6의 검사 대상에서 제외).
- **근거**: 「대형」이 **데이터 개념이 아니다** — 이미터·탄·부위 어디에도 크기 등급 필드가 없다 → S6이 어느 보스 이미터에 1.50을 걸어야 하는지 **결정 불가**하고 check.mjs가 이 하한을 구현할 수 없다. needsUserCall: 세 처방이 각각 다른 게임 결과를 낳는다.

#### [major] `rules.visual.statusBulletSpeedMul 의 적용 주체`

- **채택안**: 엔진이 곱하는 값이면 **`rules.status`로 이관**하라(게임플레이 값이므로). 저작 가이드일 뿐이면 **§12.4(공정성 하한 표)에서 제거**하라.
- **근거**: §12.4(공정성 하한 표)와 §9.4.3(visual/render 경계)이 충돌한다 — **visual 키가 게임플레이 속도를 바꾸면 「visual = 렌더 전용」 경계가 붕괴한다.** 엔진이 곱하는지 저작값에 이미 반영된 것인지 정본이 침묵한다(hexBolt 실측 95 vs 150×0.6=90 — 어느 쪽으로도 읽힌다). 연쇄: 위 `bullets[].speed` ↔ `emitters[].speed` 결정과 함께 「탄속의 최종 유효값을 누가 계산하는가」가 한 번에 닫혀야 한다. needsUserCall: 경계 재획정.

#### [minor] `bullets[frostBrick].dmg — 티어 가이드의 범주 비배타`

- **채택안**: §9.7 L2064의 가이드에 우선순위 한 줄: 「티어가 겹치면 **상태이상 티어가 크기 티어를 이긴다**」(04-§3.3이 텔레그래프 하한에 대해 이미 채택한 「두 하한이 겹치면 큰 쪽」과 같은 형태). 그 경우 `frostBrick.dmg`는 **6으로 정정**. 8을 유지하려면 가이드에 예외를 명시하라.
- **근거**: ★ 임무가 지정한 확인 항목의 결과: **적 탄 데미지 티어의 거처는 모호하지 않다** — §9.7 L2064와 §16-#44가 「소유자 = `bullets[].dmg`, 전역 티어 6종은 **저작 가이드**이며 코드 키가 아니다」를 명시 확정했고 §2.1 L168이 일치하며 전역 티어 키는 rules.json 인쇄 블록에도 없다 → 개체별 dmg로 전사했고 **신고 대상이 아니다.** 다만 가이드 자체에 파생 결함이 하나 있다: 티어 범주가 **배타적이지 않아** `frostBrick`(상태이상 slow + 소형 radius 8)이 「상태이상 6」과 「소형 8」 **두 티어에 동시에 걸리고** 04는 8을 택했다. 같은 상태이상 탄 `hexBolt`·`stunMark`는 6을 지킨다 → **10종 중 1종만 가이드에서 이탈**한다. 코드 키가 아니므로 로드는 통과하지만 04-§3.1이 「가이드 정합」을 주장하는 근거가 이 1종에서 거짓이다. needsUserCall: 6으로 내리면 밸런스가 움직인다.

#### [blocker] `[ROUND4 기지 · 게이트 산술] certify.m 배열 · runFarmDpsRatio 허용오차 · §13.5 ↔ §13.5.1`

- **채택안**: ⓐ **`certify.m`으로 인쇄 + 허용오차 + 시뮬 교정**(= `uptimeRef`·`runFarmDpsRatio`와 같은 처방의 세 번째 사례, 새 모델 0) **그리고 m[1] = 1.35를 재확인**하라 — 자연 모델은 1.216을 주고 그 경우 `bossTimeoutRate` = 0.324 > 0.25 = **확정 실패**다. ⓑ `runFarmDpsRatio`의 허용오차를 **비대칭으로 재선언**(④가 사는 창은 RFDR ≥ 0.828인데 선언된 창 [0.80, 0.86]의 **하위 47%가 확정 실패**). ⓒ §13.5의 누적 XP **27,029 → 26,450**으로 정정(§13.5.1이 옳다) + N3·N4를 §19에 심사 행으로 올려라.
- **근거**: **ROUND4 blocker/major 기지 4건을 묶었다** — 전사 범위 밖(게이트 산술)이라 이번 라운드는 재확인만 했으나, 이들이 안 닫히면 certify가 통과할 수 없으므로 패치 목록에서 뺄 수 없다. 전사가 보탠 것: check.mjs가 이 축들을 **전부 스텁으로 남겼다**(`runClearRate`·`bossTimeoutRate`·`dominance`×6 등 19개) — 즉 **정본이 m을 인쇄할 때까지 게이트 19개가 실행 불가**다. needsUserCall: m[1]의 재확인과 허용오차 재선언은 밸런스 판단이다.

### 기계적 결정 가능 (정본이 바로 반영)

- **[blocker]** `weapons[].base — 패밀리별 필수 키 집합 12행` → §9.5에 「패밀리별 base 필수 키」 열을 12행 표로 인쇄하고 「확정」 표시(= 03 §2 L136-149의 표를 정본으로 승격). 새 값 0 · 새 키 0 · 표 하나.
- **[blocker]** `bullets[].speed ↔ emitters[].speed` → **`bullets[].speed`를 §9.7 스키마에서 삭제**하고 `emitters[].speed`(§8.5의 타입별 파라미터)를 유일 소유자로 확정하라. §9.7에 한 줄: 「탄의 속도는 이미터가 소유한다 — 같은 탄이 이미터마다 다른 속도로 발사될 수 있다.」
- **[blocker]** `rules.boss.finale.armorCoreRatio` → §8.16의 「최종 전용 키」 행에서 `armorCoreRatio`를 **삭제**하고 「테트라크의 φ는 `bosses[tetrarch].armorCoreRatio` = 13.58이며 R7이 a=3으로 자동 재평가한다(상한 14.625)」로 문장을 고쳐라. `rules.boss.finale`은 4키로 닫히고 φ의 유일 소유자는 `bosses[]` 하나가 된다.
- **[blocker]** `passives[].values (12×5 = 60) · name · stats[] · maxLevel 의 C-7 구속력` → §9.6 블록 앞에 「★ 이 블록은 확정이다 — C-7의 명시적 예외(values의 유일한 거처)」 한 줄을 넣거나 values 12행을 「확정」 표로 재인쇄하라. **동시에 §0.1 L35를 정정하라** — 「v1.1은 §9의 전 예시 블록에 `// 예시` 주석을 넣었다」는 **거짓**이다(전 코퍼스에서 `// 예시` 문자열 0회).
- **[blocker]** `§13.4-S23 의 정의역` → S23의 정의역을 **`themeDraw.pool`의 6테마**로 명문화하라(「모든 테마의 roster 4종」 → 「`themeDraw.pool`에 속한 테마의 roster 4종」).
- **[major]** `rules.render (루트 인쇄 자리) + §12.2의 두 번째 render 블록` → ⓐ §9.4 루트 블록에 `"render": { "...§9.4.2 전 키..." }` **한 줄**을 넣어라(다른 3개 외부 스코프와 같은 관례). ⓑ **§12.2의 stale 블록(7키)을 삭제**하고 「규격 = `rules.json > render`, 인쇄 자리는 §9.4.2」 참조로 대체하라.
- **[major]** `rules.player.hpSegment ↔ rules.hud.hpBarSegCount` → **`player.hpSegment`를 삭제**하고 §2.1/§1.2의 「칸당 20 HP」를 「칸당 = `hpMax / hud.hpBarSegCount`(초기값 20)」로 정정하라.
- **[major]** `meta.difficulty[].scoreMul (↔ `difficultyMul`)` → §6.1 L493의 열 제목을 **`difficulty[].scoreMul`**로 정정하라(값 불변). 05 §6의 점수 정수화 공식 2줄도 `difficulty[].scoreMul`로 통보하라. ★ 그리고 §21에 **감사 축 하나를 추가**하라: 「**점이 없는 백틱 스팬 중 값의 이름으로 쓰인 것**」.
- **[major]** `rules.boss.introSec ↔ stages.phase.bossEntrySec` → **`boss.introSec` 유지 / `stages.phase.bossEntrySec` 삭제.** §6.3의 표 행은 `boss.introSec`의 인용으로 강등하고 §9.9 phase 블록에서 `bossEntrySec`를 빼라.
- **[major]** `§9.4 · §9.4.1~§9.4.4 인쇄 블록의 규범적 지위 (C-7 ↔ §21의 처분)` → §9.4 앞에 **§9.9와 같은 각주 한 줄**을 달아라: 「★ §9.4·§9.4.1~§9.4.4의 블록은 **예시가 아니라 확정값의 인쇄 자리다**(§21의 처분이 이 전제 위에 서 있다). C-7의 「예시」는 **`// 예시` 주석이 붙은 블록**에만 적용된다.」 또는 C-7의 문면을 「`// 예시` 주석이 붙은 블록만 예시다」로 개정하라(둘 중 하나면 족하고 값 변경 0). **추가로 §7.2의 팔레트 표에 `neutralGray`·`hud.*` 5색의 행을 넣어라.**
- **[major]** `rules.passiveHooks (컨테이너 형태) + `pierce` 열의 키 이름` → §9.6.1에 **JSON 블록을 인쇄**하라(표는 그대로 두고 그 아래 12행 전개). 형태 = **맵 권고**(§9.4.4의 「중첩 맵(배열 아님) — 키가 곧 속성이라 갈라질 자리가 없다」 논거가 그대로 적용된다. 배열이면 `family` 필드 ↔ 12 패밀리 어휘의 정합을 검사할 S 항목이 또 필요하다). `pierce` 열은 **`pierceApplies`**(불리언)로 개명해 무기 파라미터 `pierce`(정수)와의 이름 충돌을 없애라.
- **[major]** `enemies.archetypes[].unlockStageMin — 인쇄 ↔ 삭제` → §9.7 인쇄 블록 L2077과 §16-#1 L3666의 필드 집합에서 **`unlockStageMin` 문자열을 삭제**하라. C-11의 grep 대상 문자열 목록에 `unlockStageMin`을 추가하라.
- **[major]** `enemies.archetypes[swarmChaff|swarmLancer].themeOnly` → §9.7 표에 `themeOnly` 어휘를 「`themes[6]` 중 하나 | `null`」로 폐쇄 인쇄하고 S3 목록에 추가하라. 그 위에서 새떼 2종은 **`themeOnly: null`**로 확정하고 「새떼는 웨이브 편성 금지」는 이미 S9가 archetypeId 접두로 강제함을 한 줄 명시하라.
- **[major]** `enemies.bands.{line|turret|bruiser}.xpRef` → §9.7 표에 「`xpRef`는 `chaff` 전용 필드다 — 두 파생식(§8.10 swarmXp · §8.9 중간보스 xp)이 chaff만 참조하므로 다른 밴드에 기준값이 존재할 이유가 없다」를 한 줄 명시하라(또는 4밴드 전부에 인쇄).
- **[major]** `stages.themes (배열의 이름) ↔ `stages[]`` → **`stages`로 통일**하고 폐기된 이름 `themes`를 전 문서 grep하라(§21.4 C-11의 강화판). §9.9의 블록과 `themeDraw.pool`·`themeOnly`·`bosses[].themeId`(= 테마를 가리키는 다른 키들)의 관계를 한 줄 정리하라.
- **[blocker]** `stages.themes[].midBossAtSec` → `themes[].midBossAtSec`을 **삭제**하고 `stages.phase`에 **`midBossAtSec: [[35],[35],[30,70],[30,70],[30,70],[30,70]]`**(스테이지 인덱스 배열)로 인쇄하라. `check.mjs`에 「`len(phase.midBossAtSec[i]) == curve.midBossCount[i]`」 정적 검사를 추가하라.
- **[major]** `stages.themes[].elitesAtSec` → **`elitesAtSec`을 삭제**하고 §8.7의 「elites는 시각 기반」 문구에서 `elites`를 빼라(midBoss·crisis만 남긴다 — 그 둘은 실제로 `atSec` 이벤트다).
- **[major]** `stages.themes[].waves[].eliteIndex — element=="normal" 4건 (forest T3-s5 flanker · bog T2-s5 flanker · bog T3-s5 flanker · finale A-s4 hexer)` → 4건의 ★를 **제거**(`eliteIndex: null`)하라. `check.mjs`에 **S27** 신설: 「`eliteIndex != null` ⟹ (`archetypeId`의 `band` ∈ `elite.bandAllowed`) ∧ (`element` ∈ `elite.elementAllowed`)」.
- **[major]** `bosses[].armorCoreRatio — `boss.armorCoreRatio` 유령 스코프 (§8.11 · §8.13.1 절 제목 · §8.14-R7 키 열 · §17)` → 네 곳 전부 **`bosses[].armorCoreRatio`로 정정**하고, §9.4의 `armorCoreRatioBandPct`가 **R7 밴드의 유일 소유자**임을 §8.13.1에 한 줄 명기하라(새 키 0). 「rules = 규칙(밴드) / bosses[] = 값(φ)」의 분업이 명시되면 §13.6.4의 「밸런싱 손잡이 = `armorCoreRatio`」도 어느 파일을 여는 말인지 확정된다.
- **[major]** `bosses[mbHammer|mbNest].moveParams.leaveAfterSec` → 「**`tier: "mid"`의 `moveParams`는 `leaveAfterSec`를 저작하지 않는다**(이탈은 `stages.phase.midBossLeaveAfterSec`가 전담). `anchor`의 `leaveAfterSec`는 **잡몹 전용 파라미터**다」 한 줄 + `check.mjs`에 「`tier=="mid"` ⟹ `moveParams`에 `leaveAfterSec` 부재」. 동시에 **§15-M12의 `leaveAfterSec` 문자열을 삭제**하라.
- **[major]** `stages.phase.crisisPerStage` → **`phase`에 `crisisPerStage: 1` 인쇄**(가장 싸고 §8.10의 문면을 그대로 살린다). 또는 §21.2의 B1/B2 처분(「구조이지 값이 아니다」 — `crisisStartSec` 하나가 이미 그것을 말한다)으로 §8.10에서 백틱을 벗겨라. **어느 쪽이든 §21의 감사를 `stages.phase`에 다시 돌려야 한다.**
- **[minor]** `bosses[].themeId (mbHammer · mbLancer · mbNest · tetrarch)` → `element`와 **같은 처방**: 「`themeId: string | null`, `null`은 `tier ∈ {mid, final}`에만 허용」 + `check.mjs`에 「`tier == "stage"` ⟺ `themeId != null`」(S15와 대칭).
- **[minor]** `bosses[tier=="stage"|"final"].summon · armorCoreRatio · movePatternParams — §9.8 인쇄 블록의 불완전` → §9.8의 manta 예시 엔트리를 **필드 집합이 완전한 형태로 재인쇄**하라(`armorCoreRatio`·`movePatternParams`·`summon: null` 포함). ★ §21의 감사 축에 「**rules 표가 「필수」라 말하는 필드가 같은 절의 인쇄 블록에 실제로 있는가**」를 추가하라.
- **[minor]** `bosses[].core.xp · bosses[].parts[].xp` → §21.2의 처분 적용: **키가 아니다.** §6.4·§8.11의 백틱을 벗기고 「`bosses[]`의 스테이지·최종 보스 스키마에는 `xp` 필드가 **존재하지 않는다**(중간보스만 보상 필드로 갖는다)」로 고쳐라.
- **[minor]** `stages.themes[].skinId (sea 외 6개)` → §9.9에 한 줄: 「**`skinId`는 `themeId`와 같다**」 → 그러면 **키를 삭제**하는 것이 옳다. 스킨과 테마를 분리할 실익이 있다면 6개 값을 §9.9에 전부 인쇄하고 어휘를 S3에 동결하라.
- **[minor]** `stages.themes[finale].roster[].unlockStageMin (15 엔트리)` → §8.16에 한 줄: 「`finale`의 `roster[].unlockStageMin`은 **전부 1**(단일 티어 — 최종은 티어 필터링이 없다)」. **위 waves[] 블록 티어 blocker와 함께 결정해야 한다.**
- **[minor]** `stages.themes[finale]의 `"...":"..."` 맨 축약` → 축약을 **`"...§9.9의 themes[] 필드 집합과 동일..."`**로 바꾸거나 finale 엔트리를 전부 인쇄하라. ★ §21의 「축약이 가리키는 곳에 필드 집합이 실제로 있는가」 축의 **좌조인 대상에 이 자리도 넣어라.**
- **[minor]** `meta.certify — 경로 참조 3건 (`certify.capHits` · `certify.coinScarcity` · `certify.growthBudget`)` → 전체 경로로 정정하라(값·구조 불변, 새 키 0): §12.1 · §13.4-S26 · §16-#16 → **`certify.static.capHits`** / §9.7 → **`certify.runMode.coinScarcity`** / §11.1 → **`certify.static.growthBudget.maxLevelUps(60) < certify.static.growthBudget.minTotalSink(67)`**(★ 덤: §11.1의 `totalSink`는 인쇄된 필드명이 아니다 — 인쇄값은 `minTotalSink`다). ★ §21의 좌조인에 「**참조 경로가 인쇄 블록의 전체 경로와 문자열 일치하는가**」를 추가하라.
- **[minor]** `rules.visual.text.outlineColor` → **삭제**하고 「캔버스 텍스트 아웃라인 색 = `palette.threat.outline`」을 §7.9의 규칙으로 명기하라(새 키 0, 색 1개 감소).
- **[minor]** `rules.player.lives` → §2.1의 키 열에 **`player.lives`**를 적고 「0 (확정) — 잔기제 없음. 컨티뉴(§11.4)가 유일한 부활 경로」로 문장을 고쳐라(값 0 유지, 새 키 0).
- **[minor]** `rules.boss.healDrop · rules.elite.perWaveChance` → ⓐ §8.11에서 **백틱을 벗겨** 「스테이지 보스는 회복을 드랍하지 않는다(규칙이며 키가 아니다)」로 정정하고 §21.2의 B 목록에 추가하라. ⓑ §8.6·§8.7의 `elite.perWaveChance`를 **`stages.curve.elitePerWaveChance`**로 정정하라(새 키 0).
- **[minor]** `enemies.emitters[podLaser|lancerLaser].telegraphSec / .chargeSec` → §8.5에 「`laser`는 `chargeSec == telegraphSec`를 강제한다(check.mjs 동치 검사)」를 확정하거나 `chargeSec`를 파라미터 목록에서 삭제하고 `telegraphSec` 하나로 통일하라. 전자라면 S3/S6 계열에 동치 검사 번호를 부여하라 — S14(shape↔status)·S19(zone↔bulletId)와 같은 클래스다.
- **[minor]** `§9.3 의 「누락 키 = 에러의 유일한 예외」 — `passives[].levels[i]` 유령 경로` → L1645에서 **「와 `passives[].levels[i]`」를 삭제**하라 → 예외는 `weapons[].levels[i]` 하나뿐임이 「딱 하나」라는 자기 선언과 일치한다. 값 변경 0 · 문자열 삭제 1건.
- **[minor]** `§13.4 — S번호 순서 + 「S1~S24」 ↔ 「S1~S26」` → 표를 번호순으로 재배열하고 인용을 **「S1~S26」**으로 통일하라.
- **[minor]** `§13.4-S22 의 「스테이지 i」 · S10 의 좌변 · S6 의 minSpawnRadiusPx` → ⓐ S22를 「**모든 (theme, stage) 쌍**」으로 명문화(구현이 이미 그렇게 함 — 전 조합 통과, 최악 0.148). ⓑ S10의 좌변을 「XP 곡선으로 계산한 최대 레벨업」 → **선언 상수 비교 + `minTotalSink` 유도 검사**로 문면 수정. ⓒ `minSpawnRadiusPx`를 S6에서 빼고 **`fairnessViolations` 런타임 어서션**으로 옮겨라.
- **[minor]** `04-§3.2 「13종 확정」 ↔ 표의 실물 14행` → 04-§3.2의 제목과 §3.3 L284의 「13종」을 **「14종」으로 정정**하라.
- **[major]** `[ROUND4 기지 · C-11 자기적용] §16-#14 `caps.telegraphs = 8` · §15-m3 「156코인」 · §15-M12 `leaveAfterSec`` → C-11을 「본문 grep」이 아니라 **「폐기된 값의 문자열을 전 문서에서 grep」**으로 강화하고 §15·§16·§18을 §21의 감사 대상에 넣어라 — `156`·`caps.telegraphs = 8`·`leaveAfterSec`·`0.49`·`815`·`1100`·**`unlockStageMin`**·**`difficultyMul`**·**`totalSink`**가 한 번에 나온다.
