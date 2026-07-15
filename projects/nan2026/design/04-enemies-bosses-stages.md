# 섹션 — 적 · 중간보스 · 위기세션 · 보스 · 스테이지

> **지위**: 이 문서는 `design/CANON.md`(정본)의 **하위 문서**다. 정본 §17이 이 섹션에 위임한 범위 = **15 아키타입의 실제 파라미터 · 테마별 로스터 편성 · 웨이브 리스트 저작 · 이미터 인스턴스 · 6+1 보스와 3 중간보스의 부위 배치·HP·`patternSet`(R1~R6 안에서)**.
>
> **정본 재정의 금지 (C-1)**. 이 문서에 등장하는 정본 값은 전부 **인용**이며 출처 절 번호를 병기한다. 이 문서가 정본과 충돌하면 **이 문서가 틀린 것**이다.
>
> **여기서 저작하는 값의 거처 (C-2)** = `data/enemies.json` · `data/bullets.json` · `data/bosses.json` · `data/stages.json`. 아래 표는 그 JSON의 **저작 원본**이며, 확정 후 `.js`를 열지 않고 이 값들만 시뮬이 조정한다 (C-4).
>
> **정본에 없어서 새로 정한 것은 전부 §14 「정본 추가 요청」에 열거**했다. 그중 **R-1 · R-2 · R-3은 blocker**다 — 이 셋이 정본에 반영되지 않으면 이 섹션의 콘텐츠는 `check.mjs`를 통과하지 못하거나 스테이지 스케일이 성립하지 않는다.

---

## 0. 이 섹션이 지켜야 하는 정본 게이트 (작업 체크리스트)

| 게이트 | 출처 | 이 문서에서 통과를 증명한 곳 |
|---|---|---|
| **S3** 어휘 폐쇄 (`moveId` 8 · `emitterType` 8 · `formationId` 6 · `partType` 4 · `shapeId` 12) | 정본 §13.4 | §2.1 · §2.2 · §3 · §9 |
| **S4** 아키타입 겹침 금지 — `(moveId, emitterType)` 중복 실패, `band` 다르면 허용 | 정본 §13.4 | **§2.4 전수 증명표** |
| **S5** 보스 R1~R6 + `partCount` | 정본 §8.14 · §13.4 | **§10.1 전수 검증표** |
| **S6** 공정성 — 텔레그래프 하한·탄속·틈·`minSpawnRadiusPx`·스턴 | 정본 §7.4 · §12.4 | **§3.3 전수 검증표** |
| **S7** 개체당 동시 텔레그래프 ≤ 2 (3페이즈 전개) | 정본 §12.4 · §13.4 | **§9.3 라운드로빈 정적 증명** |
| **S8** 혼합 비율 ±3%p | 정본 §8.2 · §13.4 | **§5.2 블록 불변식 증명** |
| **S9** `element` null은 finale만 · `rearIn`은 `rearSpawnAllowed`만 · 새떼 아키타입 · `crisisElementRule` | 정본 §13.4 | §1.1 · §2.3 · §8 · §11 |
| `capHits == 0` | 정본 §13.1 | **§12.2** (← R-4 요청의 근거) |
| `dominance` · `crisisClearWithoutCapstone` · `farmXpRatio` | 정본 §13.1 | §13 |

---

## 1. 테마

### 1.1 테마 풀 (정본 §8.1 인용 — 재정의 아님)

`sea`(물, `introOk`) · `glacier`(물) · `volcano`(불, `introOk`) · `desert`(불) · `forest`(풀, `introOk`) · `bog`(풀) — **속성당 정확히 2종**, 노말 테마 없음. 스테이지 6 = `finale`(`element: null`, 추첨 대상 아님).

**커버리지 보장 방식 = 추첨 구조 그 자체다 (정본 §8.1의 증명을 인용).** `themeDraw.count = 5`, `allowRepeat = false` → 6종 중 **정확히 1종만 탈락** → 각 속성은 2종 중 최대 1종만 빠짐 → **모든 런에 물·불·풀 테마가 각각 최소 1회.** 그리고 `introOk` 3종 중 최대 1종만 탈락 → 스테이지 1 배치가 항상 가능.

> **이 섹션이 여기에 더할 것은 없다.** 커버리지는 로스터 편성이나 웨이브 저작으로 보정하는 것이 아니라 **추첨 구조가 이미 증명**한다. 별도의 거부 샘플링·보장 키를 **만들지 않는다**(정본이 `guaranteeElements`·`firstStageFrom`을 폐기한 이유와 동일).

### 1.2 테마별 구성 — 정체성이 무엇으로 구현되는가

각 테마의 정체성(정본 §8.1의 1줄)은 **새 규칙이 아니라 로스터 4종의 `moveId`/`emitterType` 조합**으로만 구현된다.

| `themeId` | 정체성 (정본 §8.1) | **이 섹션의 구현** (로스터의 이동/이미터 조합) |
|---|---|---|
| `sea` | 개활 수면. 느린 대형 개체, 유도 위주. 가장 읽기 쉬움 | `dive`+무장 0 · `weave`+`straight` · `orbitDrift`+유도탄 2종(`straight`/`ring`). **화면에 느린 곡선만 있다** → 정본 §7.8 "천천히 곡선으로 쫓아오면 100% 탄"을 스테이지 1에서 통째로 가르친다 |
| `glacier` | 체류형 포대 + 둔화. 라인 회피 시험 | `anchor` 2종(`laser` / `wall`+둔화). **정지한 발사대가 라인을 긋는다** → 이동이 곧 정답 |
| `volcano` | 장판·분출. 공간 제한 | `zone` 2종(`anchor`+`zone` / `dive`+`zone`) + `rearIn`. **바닥이 계속 줄어든다** |
| `desert` | 초고속 소형. **파밍 리스크의 화신** | `dive`(초고속, 무장 0) + `column` + `rearIn`. **전부 스쳐 지나간다** → `enemyExitForfeitsReward`(정본 §8.8)가 가장 아프게 물리는 테마 |
| `forest` | 확산탄 벽. 편대 밀집 | `strafe` 2종(`fan` / `wall`) + `weave`+`aimed`. **수평선이 레인을 강제** |
| `bog` | 상태이상(둔화) 다발 + 시야 압박 | `weave`+`spiral`(둔화) · `weave`+`aimed`(둔화) · `anchor`+`laser`. **화면이 느린 육각탄으로 찬다** |

- **아레나 스킨은 값이지 규칙이 아니다**: `themes[].skinId`(정본 §9.9)는 절차적 배경 파라미터의 id일 뿐이며, 정본 §7.9의 상한(채도 ≤0.25 / 명도 ≤0.28, 자홍 금지, 소프트 엣지)을 그대로 받는다. **테마가 게임플레이에 주는 것은 로스터·`mix`·보스뿐이고, 배경은 아무 규칙도 만들지 않는다.**

### 1.3 `mix` (정본 §8.2 인용) — 이 섹션은 값을 전개만 한다

`counter`/`prey`는 **빌드타임 생성 규칙**이며, `stages[].mix`에는 **4속성 실제 키로 전개된 맵 하나**만 들어간다(정본 §8.2).

| `themeId` | `element` | `stages[].mix` (전개된 값) | 정답 스탠스 | 세금(×0.5) |
|---|---|---|---|---|
| `sea` · `glacier` | `water` | `{"water":0.70,"fire":0.10,"grass":0.10,"normal":0.10}` | **R (풀)** | 불 10% |
| `volcano` · `desert` | `fire` | `{"fire":0.70,"water":0.10,"grass":0.10,"normal":0.10}` | **E (물)** | 풀 10% |
| `forest` · `bog` | `grass` | `{"grass":0.70,"water":0.10,"fire":0.10,"normal":0.10}` | **W (불)** | 물 10% |
| `finale` | `null` | `{"water":0.30,"fire":0.30,"grass":0.30,"normal":0.10}` | 없음(회전) | — |

> 검산 (정본 §4.1 상성표): 물 테마 → 물을 이기는 것 = **풀** = counter ✔ / 물이 이기는 것 = **불** = prey ✔ → 풀 스탠스로 불(10%)을 때리면 ×0.5 ✔. 불 테마 → counter = 물, prey = 풀 ✔. 풀 테마 → counter = 불, prey = 물 ✔. **키 순서 `Q W E R` = 노말·불·물·풀** 표기 준수(정본 §4.1).

### 1.4 진행 곡선 (정본 §8.3 인용, 재정의 아님)

`enemyHpScale` · `xpScale` · `spawnDensityScale` · `midBossCount` · `elitePerWaveChance` · `swarmTotalScale` · `rearSpawnAllowed` — **6×7 테이블의 값은 정본 §8.3/§9.9가 유일 소유자**다. 이 섹션은 그 값을 읽어 쓸 뿐 적지 않는다.

- **후반 스펀지 방지의 실물 = `unlockStageMin`**(정본 §8.3). 그 실제 편성이 §4다.
- ★ **보스 HP는 이 테이블의 소유가 아니다** — 정본 §8.3은 *적* HP 스케일이고, §8.9는 *중간보스*에 `enemyHpScale`을 명시했으나 **스테이지 보스에는 어떤 스테이지 스케일도 지정되어 있지 않다.** 테마 순서가 랜덤이므로 같은 보스가 스테이지 1에도 5에도 온다 → **스케일이 없으면 보스 밸런싱이 성립하지 않는다.** → §14 **R-1 (blocker)**.

---

## 2. 적 — 어휘 매핑과 아키타입 확정

### 2.1 이동 어휘 사용 현황 (정본 §8.4의 8종, 동결 — 인용)

`dive` · `weave` · `column` · `strafe` · `anchor` · `orbitDrift` · `charge` · `rearIn`

- ★ **`charge`를 가진 잡몹 아키타입은 존재하지 않는다.** 정본 §8.4에서 다른 7종은 전부 특정 무기 패밀리와 짝지어져 있어(`column`↔랜스, `strafe`↔부메랑, `orbitDrift`↔오빗/오라, `rearIn`↔리어가드) **그 짝이 아키타입 배정을 강제**하지만, `charge`의 존재 이유는 "위치 판단 시험"으로 **짝이 없다.** 그리고 `charge`는 `windUpSec ≥ minTelegraphSec`(정본 §8.4)를 요구하므로, `enemyConcurrentMax = 40` 규모에서 chaff가 들면 **동시 텔레그래프 예산을 혼자 먹는다**(§12.2). → **`charge`는 중간보스 `mbLancer`가 독점한다**(정본 §8.9가 이미 그렇게 배정했다). 어휘는 100% 사용되며, 스코프만 개체 크기로 갈린다.
- `rearIn`은 `rearSpawnAllowed[stage]`(스테이지 3+)에서만 + `warnSec = 0.8` 진입 표식 선행 (정본 §8.4 · S9).

### 2.2 이미터 어휘 사용 현황 (정본 §8.5의 8종, 동결 — 인용)

`straight` · `fan` · `aimed` · `ring` · `spiral` · `laser` · `zone` · `wall` — **8종 전부 사용**(§2.3 표).
**유도 = `bullets[].turnRateDegSec > 0`, 상태이상 = `bullets[].status`** — 이미터 타입이 아니다(정본 §8.5). 사격 안 함 = `attack: null`.

### 2.3 ★ 아키타입 17종 확정 (`data/enemies.json > archetypes`)

> 공용 9 + 시그니처 6 + 새떼 2 = **17**(정본 §8.6). `element`·`tier`·`hpScalePerStage`·`spriteId` 필드는 **없다**(정본 §9.7). `unlockStageMin`은 §4의 로스터 엔트리가 소유한다(→ §14 R-6).
> `hp`는 **밴드 배수 적용 전 기본값**이다: `실효 HP = hp × band.hpMult × enemyHpScale[stage]` (정본 §8.6).

| `id` | `name` | `band` | `shapeId` | `radius` | `moveId` | `moveParams` | `attack` | `contactDmg` | `hp` | `xp` | `score` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `drifter` | 표류체 | `chaff` | `orb` | 7 | `dive` | `{"speed":70}` | **`null`** | 6 | 6 | 2 | 20 |
| `spitter` | 산탄충 | `chaff` | `delta` | 7 | `weave` | `{"speed":80,"ampPx":60,"freqHz":0.5}` | `{"emitterId":"spitStraight","firstDelaySec":1.2}` | 6 | 8 | 3 | 30 |
| `columnAnt` | 행렬병 | `line` | `wedge` | 10 | `column` | `{"speed":95,"gapSec":0.35}` | `{"emitterId":"antStraight","firstDelaySec":1.0}` | 7 | 10 | 4 | 45 |
| `flanker` | 측면기 | `line` | `dart` | 10 | `strafe` | `{"speed":110,"yPx":180}` | `{"emitterId":"flankFan","firstDelaySec":0.9}` | 7 | 10 | 4 | 50 |
| `hexer` | 저주충 | `line` | `hexPod` | 11 | `weave` | `{"speed":65,"ampPx":90,"freqHz":0.35}` | `{"emitterId":"hexAimed","firstDelaySec":1.4}` | 7 | 12 | 5 | 60 |
| `turretPod` | 정착포 | `turret` | `slab` | 17 | `anchor` | `{"enterSpeed":120,"yHoldPx":150,"swayAmpPx":40,"leaveAfterSec":22}` | `{"emitterId":"podLaser","firstDelaySec":1.6}` | 15 | 16 | 9 | 90 |
| `stalker` | 추격체 | `bruiser` | `claw` | 18 | `orbitDrift` | `{"speed":60,"turnRateDegSec":90,"keepDistPx":140}` | `{"emitterId":"homing2","firstDelaySec":0.8}` | 14 | 30 | 6 | 120 |
| `mortarHulk` | 박격거구 | `bruiser` | `bulb` | 20 | `anchor` | `{"enterSpeed":90,"yHoldPx":120,"swayAmpPx":70,"leaveAfterSec":26}` | `{"emitterId":"hulkZone","firstDelaySec":1.8}` | 16 | 28 | 10 | 150 |
| `rearDart` | 후방침 | `chaff` | `spike` | 6 | `rearIn` | `{"speed":150,"warnSec":0.8}` | `{"emitterId":"dartStraight","firstDelaySec":1.0}` | 6 | 7 | 3 | 40 |
| `sirenRay` | 해류 가오리 | `bruiser` | `fin` | 22 | `orbitDrift` | `{"speed":45,"turnRateDegSec":50,"keepDistPx":200}` | `{"emitterId":"sirenRing","firstDelaySec":1.5}` | 15 | 26 | 9 | 140 |
| `frostLance` | 빙주 | `turret` | `cross` | 16 | `anchor` | `{"enterSpeed":110,"yHoldPx":170,"swayAmpPx":25,"leaveAfterSec":24}` | `{"emitterId":"frostWall","firstDelaySec":1.5}` | 15 | 15 | 8 | 95 |
| `magmaBomb` | 용암포 | `bruiser` | `ring` | 21 | `dive` | `{"speed":40}` | `{"emitterId":"magmaZone","firstDelaySec":1.2}` | 16 | 24 | 9 | 135 |
| `dustRunner` | 질주충 | `line` | `dart` | 8 | `dive` | `{"speed":230}` | **`null`** | 7 | 4 | 15 | 60 |
| `thornWeaver` | 가시직조 | `line` | `spike` | 12 | `strafe` | `{"speed":70,"yPx":140}` | `{"emitterId":"thornWall","firstDelaySec":1.1}` | 7 | 14 | 5 | 70 |
| `bogHexer` | 늪령 | `bruiser` | `hexPod` | 19 | `weave` | `{"speed":50,"ampPx":110,"freqHz":0.3}` | `{"emitterId":"bogSpiral","firstDelaySec":1.6}` | 15 | 25 | 9 | 145 |
| `swarmChaff` | 새떼 | `chaff` | `delta` | 6 | `weave` | `{"speed":130,"ampPx":40,"freqHz":1.2}` | **`null`** | 6 | 3 | 2 | 10 |
| `swarmLancer` | 새떼창병 | `chaff` | `wedge` | 7 | `dive` | `{"speed":110}` | `{"emitterId":"lancerStraight","firstDelaySec":1.5}` | 6 | 4 | 2 | 15 |

**`themeOnly`**: `sirenRay`=`"sea"` · `frostLance`=`"glacier"` · `magmaBomb`=`"volcano"` · `dustRunner`=`"desert"` · `thornWeaver`=`"forest"` · `bogHexer`=`"bog"` · `swarmChaff`/`swarmLancer`=`"*swarm"`(웨이브 편성 금지, 위기 세션 전용 — S9). 나머지 9종 = `null`(공용).

**`desc`** (한국어 인라인, 정본 §0.3): `drifter` "떠내려오며 아무것도 하지 않는다" · `spitter` "흔들리며 앞으로 뱉는다" · `columnAnt` "같은 줄로 줄줄이 내려온다" · `flanker` "옆에서 가로지르며 부채꼴로 쏜다" · `hexer` "느린 저주탄을 조준해 쏜다" · `turretPod` "자리를 잡고 레이저를 긋는다" · `stalker` **"플레이어를 선회하며 유도탄을 쏜다"**(정본 §9.7) · `mortarHulk` "멈춰서 장판을 깐다" · `rearDart` "뒤에서 올라온다" · `sirenRay` "느린 유도 고리를 퍼뜨린다" · `frostLance` "틈이 있는 얼음 벽을 세운다" · `magmaBomb` "내려오며 용암을 흘린다" · `dustRunner` "스쳐 지나간다. 놓치면 끝" · `thornWeaver` "가로지르며 가시 벽을 남긴다" · `bogHexer` "느린 나선 저주를 뿌린다" · `swarmChaff` "숫자로 민다" · `swarmLancer` "새떼 속에서 한 발 쏜다".

**정본 인용 확인 — `stalker`는 정본 §9.7이 직접 저작했다**: `band:"bruiser"`, `shapeId:"claw"`, `radius:18`, `moveId:"orbitDrift"`, `moveParams:{speed:60,turnRateDegSec:90,keepDistPx:140}`, `attack:{emitterId:"homing2",firstDelaySec:0.8}`, `contactDmg:14`, `hp:30`, `xp:6`, `score:120`, `unlockStageMin:3`. **이 값은 그대로 옮겼고 한 글자도 바꾸지 않았다.**

**저작 가이드 정합 확인** (정본 §2.5): 잡몹 `contactDmg` 6~8 ✔ (`drifter`~`thornWeaver`) / 중형·대형은 14~16 ✔ (`turretPod` 15 · `stalker` 14 · `mortarHulk` 16 · 시그니처 bruiser 15~16). `elite.contactDmgMul = 1.5`(정본 §8.6)가 곱해져도 최대 24 → **최대 단발 22 + i-frame 1.0 → 5초 연속 피격 논증**(정본 §2.1)은 접촉이 게임초당 최대 1회이므로 유지된다.

### 2.4 ★ S4 전수 증명 — `(moveId, emitterType)` 중복 없음

| `(moveId, emitterType)` | 아키타입 | `band` | 판정 |
|---|---|---|---|
| `(dive, —)` | `drifter` | `chaff` | ★ 중복 쌍이나 **`band` 다름 → 허용** |
| `(dive, —)` | `dustRunner` | `line` | ★ 위와 쌍 동일, **`band` 다름 → 허용** |
| `(dive, straight)` | `swarmLancer` | `chaff` | 유일 |
| `(dive, zone)` | `magmaBomb` | `bruiser` | 유일 |
| `(weave, —)` | `swarmChaff` | `chaff` | 유일 |
| `(weave, straight)` | `spitter` | `chaff` | 유일 |
| `(weave, aimed)` | `hexer` | `line` | 유일 |
| `(weave, spiral)` | `bogHexer` | `bruiser` | 유일 |
| `(column, straight)` | `columnAnt` | `line` | 유일 |
| `(strafe, fan)` | `flanker` | `line` | 유일 |
| `(strafe, wall)` | `thornWeaver` | `line` | 유일 |
| `(anchor, laser)` | `turretPod` | `turret` | 유일 |
| `(anchor, wall)` | `frostLance` | `turret` | 유일 |
| `(anchor, zone)` | `mortarHulk` | `bruiser` | 유일 |
| `(orbitDrift, straight)` | `stalker` | `bruiser` | 유일 |
| `(orbitDrift, ring)` | `sirenRay` | `bruiser` | 유일 |
| `(rearIn, straight)` | `rearDart` | `chaff` | 유일 |

**결론: 17종 중 중복 쌍은 `(dive, —)` 하나뿐이고 `band`가 `chaff`/`line`으로 갈린다 → S4 통과 ✔**

- `drifter`(무해한 XP 사료, 모든 `introOk` 로스터의 상성 교보재)와 `dustRunner`(XP ×5 도망자)는 **둘 다 "내려가기만 하고 쏘지 않는다"가 정체성**이라 쌍이 겹치는 것이 필연이다. 정본 S4의 `band` 예외가 정확히 이 경우를 위해 있다: `chaff`(HP 6, 속도 70, 무시해도 됨) vs `line`(HP ×2.5, 속도 230, XP 15 — **요격하려면 위로 올라가야 한다**). **HP 2.5배와 속도 3.3배가 두 개체를 완전히 다른 결정으로 만든다.**
- `charge`·`fan`(보스 전용 추가 사용 있음)을 포함해 **`moveId` 8종 중 7종, `emitterType` 8종 중 8종이 잡몹에서 사용**된다. 나머지 `charge`는 §2.1의 근거로 `mbLancer`가 독점.

### 2.5 `shapeId` 배정 (정본 §9.10의 12종, 동결 — 인용)

`orb`(drifter) · `delta`(spitter, swarmChaff) · `wedge`(columnAnt, swarmLancer) · `dart`(flanker, dustRunner) · `hexPod`(hexer, bogHexer) · `slab`(turretPod) · `claw`(stalker) · `bulb`(mortarHulk) · `spike`(rearDart, thornWeaver) · `fin`(sirenRay) · `cross`(frostLance) · `ring`(magmaBomb)

- **도형 재사용은 규칙 위반이 아니다.** 정본 §7.6의 3층 분리(배경 명도 / 중립 차콜 실루엣 / 속성색 외곽선·글리프)에서 개체 식별은 **`radius` + `band` 크기 + 속성색 외곽선 + 코어 글리프**가 하며, `shapeId`는 실루엣 계열을 주는 채널이다. 같은 `hexPod`를 쓰는 `hexer`(r 11, line)와 `bogHexer`(r 19, bruiser)는 **크기가 1.7배 다르고 밴드 색 명도가 다르다** → 혼동 불가. 오히려 **"육각 포드 = 저주 계열"**이라는 학습이 이월된다(정본 §7.4의 "학습 1회로 끝난다" 원칙과 동일한 이득).
- 12 도형 전부 사용 ✔.

---

## 3. 탄 · 이미터 인스턴스

### 3.1 `data/bullets.json > bullets` (10종 확정)

> `element` 키는 **존재하지 않는다**(정본 §9.7 — 스키마가 "적 공격에는 속성이 없다"를 강제). 색 = `palette.threat.enemyBullet` 자홍 단일.
> `dmg` 저작 가이드(정본 §9.7): 소형 8 / 중형 14 / 대형·레이저 22 / 장판 10 / 상태이상 6.

| `id` | `radius` | `hitboxScale` | `speed` | `dmg` | `shape` | `status` | `statusDurationSec` | `accel` | `turnRateDegSec` | `retargetSec` | 쓰임 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `pelletS` | 5 | 0.8 | 150 | 8 | `circle` | `null` | 0 | 0 | 0 | 0 | 기본 직선 소형탄 |
| `fanShard` | 5 | 0.8 | 130 | 8 | `circle` | `null` | 0 | 0 | 0 | 0 | 확산 소형탄 |
| `homingM` | 7 | 0.8 | 90 | 14 | `circle` | `null` | 0 | 0 | **60** | 0.5 | `stalker` 유도탄 (정본 §9.7 인용) |
| `driftHoming` | 8 | 0.8 | 95 | 14 | `circle` | `null` | 0 | **30** | 0.6 | `sirenRay` 느린 유도 고리 |
| `hexBolt` | 9 | 0.8 | 95 | 6 | `hex` | **`"slow"`** | 2.5 | 0 | 0 | 0 | 둔화 조준탄·나선탄 |
| `frostBrick` | 8 | 0.8 | 105 | 8 | `hex` | **`"slow"`** | 2.0 | 0 | 0 | 0 | 얼음 벽 (둔화) |
| `thornBrick` | 7 | 0.8 | 120 | 8 | `circle` | `null` | 0 | 0 | 0 | 0 | 가시 벽 |
| `heavyRound` | 10 | 0.8 | 115 | 14 | `circle` | `null` | 0 | 0 | 0 | 0 | 보스 중형탄 |
| `beamCore` | 4 | 0.8 | 0 | **22** | `circle` | `null` | 0 | 0 | 0 | 0 | **레이저 빔의 데미지 소유자** |
| `stunMark` | 11 | 0.8 | 80 | 6 | `hex` | **`"stun"`** | **1.0** | 0 | 0 | 0 | 스턴 (Hard+ 전용, §10.4) |

- `driftHoming`의 `turnRateDegSec` 30 + `speed` 95 = **정본 §7.8의 "느리게 곡선으로 추적 = 100% 탄"의 교과서적 실물.** 바다 테마가 `introOk`이자 "가장 읽기 쉬움"인 이유가 여기서 데이터로 성립한다 — 픽업(직선 스트릭·글로우 없음·≤6px)과 운동이 겹치는 구간이 없다.
- `beamCore`의 `speed = 0`: `laser`는 빔이지 이동하는 탄이 아니다. 기하는 `widthPx`가, **데미지는 `bullets[].dmg`가**(정본 §3.2의 유일한 소유자 목록) 담당한다. `radius` 4 = `minBulletRadiusPx` 하한(정본 §12.4).
- `stunMark`: `statusDurationSec = 1.0` = `fairness.maxStunSec` 정확히 하한 준수(정본 §2.7).
- **`zone`은 탄을 쓰지 않는다** — `zone` 이미터가 `dmg`를 직접 갖는다(정본 §8.5). → §14 **R-5**(`bulletId: null` 허용 명시 요청).

### 3.2 `data/enemies.json > emitters` (13종 확정)

공통 파라미터(정본 §8.5): `type, bulletId, from, telegraphSec, everySec, offsetSec, repeat, restSec`

| `id` | `type` | `bulletId` | `from` | `telegraphSec` | `everySec` | 고유 파라미터 | 하한(정본 §7.4) |
|---|---|---|---|---|---|---|---|
| `spitStraight` | `straight` | `pelletS` | `self` | **0.55** | 4.5 | `count:2, spreadDeg:14, speed:150` | 0.55 ✔ |
| `antStraight` | `straight` | `pelletS` | `self` | **0.55** | 5.0 | `count:1, spreadDeg:0, speed:150` | 0.55 ✔ |
| `dartStraight` | `straight` | `pelletS` | `self` | **0.55** | 5.5 | `count:2, spreadDeg:24, speed:140` | 0.55 ✔ |
| `lancerStraight` | `straight` | `pelletS` | `self` | **0.60** | 6.0 | `count:1, spreadDeg:0, speed:120` | 0.55 ✔ |
| `homing2` | `straight` | `homingM` | `self` | **0.80** | **2.2** | `count:2, spreadDeg:20, speed:90` | 0.55 ✔ |
| `flankFan` | `fan` | `fanShard` | `self` | **0.60** | 5.0 | `count:5, arcDeg:70, speed:130` | 0.60 ✔ |
| `hexAimed` | `aimed` | `hexBolt` | `self` | **0.80** | 5.5 | `count:1, spreadDeg:0, speed:95, leadSec:0.35` | 0.80 (상태이상) ✔ |
| `sirenRing` | `ring` | `driftHoming` | `self` | **0.60** | 5.0 | `count:8, speed:95, rotOffsetDeg:22.5` | 0.60 ✔ |
| `bogSpiral` | `spiral` | `hexBolt` | `self` | **0.80** | 7.0 | `count:10, speed:95, rotStepDeg:26, durationSec:2.0, rateSec:0.2` | 0.80 (상태이상) ✔ |
| `podLaser` | `laser` | `beamCore` | `self` | **1.20** | 6.0 | `chargeSec:1.2, widthPx:16, activeSec:0.5, angleDeg:90, trackDuringCharge:false` | 1.20 ✔ |
| `hulkZone` | `zone` | `null` | `self` | **0.90** | 5.0 | `radius:70, activeSec:3.0, dmg:10` | 0.90 ✔ |
| `magmaZone` | `zone` | `null` | `self` | **0.90** | 4.0 | `radius:58, activeSec:3.5, dmg:10` | 0.90 ✔ |
| `frostWall` | `wall` | `frostBrick` | `self` | **0.80** | 7.0 | `count:9, gapCount:1, gapWidthPx:84, speed:105` | 0.80 (wall·상태이상 동률) ✔ |
| `thornWall` | `wall` | `thornBrick` | `self` | **0.80** | 6.5 | `count:8, gapCount:1, gapWidthPx:92, speed:120` | 0.80 ✔ |

`offsetSec: 0`, `repeat: 1`, `restSec: 0` — 잡몹은 전부 단발 반복(보스만 `repeat`/`restSec`를 쓴다, §9.3).

> ★ **`homing2`의 `telegraphSec: 0.8` / `everySec: 2.2` / `count: 2` / `spreadDeg: 20` / `speed: 90`은 정본 §9.7이 직접 저작한 값**이며 그대로 인용했다.

### 3.3 ★ S6 전수 검증 — 공정성 하한 (정본 §12.4)

| 검사 | 하한/상한 | 최악값 | 판정 |
|---|---|---|---|
| `telegraphSec ≥ minTelegraphSec` | **0.55** | `spitStraight`/`antStraight`/`dartStraight` = **0.55** | ✔ (등호 허용 — 정본 §13.2-⑦이 명시) |
| 거동별 표 (정본 §7.4) | `straight` 0.55 / `fan` 0.60 / `aimed` 0.60 / `ring` 0.60 / `spiral` 0.60 / `wall` 0.80 / `zone` 0.90 / `laser` 1.20 / 상태이상(slow) 0.80 / 상태이상(stun) **1.50** | 13종 전부 §3.2 표에서 하한 이상 | ✔ |
| **상태이상 탄의 하한 우선** | `hexAimed`(aimed 0.60 vs slow 0.80) → **0.80 채택** / `bogSpiral`(spiral 0.60 vs slow 0.80) → **0.80** / `frostWall`(wall 0.80 vs slow 0.80) → **0.80** | — | ✔ **두 하한이 겹치면 큰 쪽** |
| `speed ≤ maxBulletSpeed` | **260** | `pelletS` **150** | ✔ (여유 42%) |
| `speed ≤ maxAimedBulletSpeed` | **200** | `hexAimed` = `hexBolt` **95** | ✔ (aimed 이미터는 `hexAimed` 하나뿐) |
| `radius ≥ minBulletRadiusPx` | **4** | `beamCore` **4** | ✔ (등호) |
| `gapWidthPx ≥ minGapWidthPx` | **46** | `frostWall` **84** | ✔ (히트박스 r=4 → 좌우 각 38px 여유) |
| 점블랭크 금지 | `minSpawnRadiusPx` **140** | 엔진 강제 — `orbitDrift.keepDistPx`가 `stalker` 140 / `sirenRay` 200으로 **이미 하한 이상에 정착**하므로 `rejectSpawn`이 발생하지 않는다 | ✔ |
| 스턴 텔레그래프 | ≥ **1.50** | `stunMark` 사용 이미터 = **1.50** (§10.4) | ✔ |
| 스턴 지속 | ≤ **1.00** | `stunMark.statusDurationSec` **1.00** | ✔ |
| 스턴 개체 수 | `statusStunMaxPerStage` **2** | 스테이지당 **1** (§10.4) | ✔ |
| 스턴 난이도 | `stunMinDifficulty` `"hard"` | Normal에서 생성 안 됨 | ✔ (→ R-3) |

★ **`orbitDrift.keepDistPx`가 `minSpawnRadiusPx`(140)의 실제 집행 장치다.** `stalker`가 `keepDistPx: 140`(정본 §9.7 저작값)으로 정확히 하한에 서는 것은 우연이 아니다 — 이 값보다 가까이 붙는 적은 **탄을 쏠 수 없어 무해해진다.** 그래서 `sirenRay`는 200으로 더 멀리 세워 "느린 대형 개체"의 사거리 압박을 준다. **`keepDistPx < 140`인 `orbitDrift` 적은 저작 금지**(§12.1 G-11).

---

## 4. 테마별 로스터 (`data/stages.json > themes[].roster`)

> **테마당 정확히 4종 = 공용 3 + 시그니처 1** (정본 §8.6). `unlockStageMin`은 **로스터 엔트리가 소유**한다 → 같은 공용 아키타입이 테마마다 다른 시점에 해금된다 (§14 **R-6**).

| 테마 | `element` | 로스터 4종 (`unlockStageMin`) | 티어 구성 |
|---|---|---|---|
| `sea` | 물 | `drifter`(1) · `spitter`(1) · **`sirenRay`**(2) · `stalker`(**3**) | T1{drifter,spitter} T2{+sirenRay} T3{+stalker} |
| `glacier` | 물 | `spitter`(1) · `turretPod`(2) · **`frostLance`**(2) · `columnAnt`(3) | T1{spitter} T2{+turretPod,frostLance} T3{+columnAnt} |
| `volcano` | 불 | `spitter`(1) · **`magmaBomb`**(2) · `mortarHulk`(3) · `rearDart`(**3**) | T1{spitter} T2{+magmaBomb} T3{+mortarHulk,rearDart} |
| `desert` | 불 | `drifter`(1) · **`dustRunner`**(1) · `columnAnt`(2) · `rearDart`(**3**) | T1{drifter,dustRunner} T2{+columnAnt} T3{+rearDart} |
| `forest` | 풀 | `spitter`(1) · **`thornWeaver`**(1) · `flanker`(2) · `hexer`(3) | T1{spitter,thornWeaver} T2{+flanker} T3{+hexer} |
| `bog` | 풀 | `hexer`(1) · `flanker`(2) · **`bogHexer`**(2) · `turretPod`(3) | T1{hexer} T2{+flanker,bogHexer} T3{+turretPod} |
| `finale` | `null` | **공용 9 + 시그니처 6 = 15종 전부**(`unlockStageMin` 전부 충족, 정본 §8.16) | 단일 티어 |

- `stalker`의 `unlockStageMin: 3`은 **정본 §9.7이 직접 적은 값**이며 `sea`에서 그대로 지킨다.
- ★ **`rearDart`의 `unlockStageMin: 3`은 `rearSpawnAllowed[3+]`(정본 §8.3)와 정확히 일치하도록 배정했다.** → `rearIn`/`spawnEdge:"bottom"` 웨이브가 **스테이지 1·2의 편성에 물리적으로 존재할 수 없다** → **S9가 자동으로 통과**한다(검사가 잡을 일이 없다). 규칙을 검사로 지키는 게 아니라 **구조로 지킨다.**
- **시그니처 해금이 늦은 테마 = 정체성이 늦게 온다.** `sea`(sirenRay@2)·`desert`(dustRunner@**1**)의 대비가 의도다: 사막은 **첫 순간부터** 파밍 리스크를 들이민다(`introOk: false`이므로 최소 스테이지 2), 바다는 천천히 보여준다(`introOk: true`, 스테이지 1 후보).

---

## 5. ★ 웨이브 리스트 저작 — 블록 모델

### 5.1 자료구조 (정본 §8.7 인용)

```
waves: [ { formationId, archetypeId, count, element, spawnEdge, eliteIndex } ]   // 시각 없음, 순서만
```
`다음 스폰 = max(직전 스폰 + waveIntervalSec, 직전 웨이브 전멸 시각)` · `waveClearAdvance` = **남은 스케줄 전체 압축** · `mobPhaseMaxWaves = 14` · `crisisSuspendsWaves` · 초과 시 `defer`.

### 5.2 ★ 블록 불변식 — S8을 오차 0으로 통과시키는 장치

**문제**: `unlockStageMin` 필터링은 스테이지마다 **다른 웨이브 리스트**를 만든다. S8은 "**스테이지** 총 개체 수 기준 비율이 `mix`에 ±3%p"를 요구한다 → **5개의 서로 다른 리스트가 전부 70/10/10/10이어야 한다.** 웨이브를 임의로 나열하면 이 5중 제약을 동시에 만족시키기가 사실상 불가능하다.

**해법 — 블록 불변식**:

> **웨이브 리스트는 3개의 "블록"으로만 구성된다. 블록의 티어 = 그 블록이 쓰는 아키타입의 `unlockStageMin` 최댓값. 그리고 ★ 모든 블록은 각각 독립적으로 정확히 70/10/10/10이다.**

**증명**: 스테이지 `s`의 필터링 결과 = `{블록 T : T ≤ s}`의 합집합. 각 블록의 비율이 전부 동일한 상수 벡터 `(0.7, 0.1, 0.1, 0.1)`이면, **그 합집합의 비율 = 크기로 가중한 평균 = 같은 상수 벡터.** 블록을 몇 개 넣든 빼든 비율은 변하지 않는다. ∎

**결과: 전 테마 · 전 스테이지에서 S8 오차 = 0.0%p** (±3%p 예산을 통째로 남긴다). 검산은 §5.5.

### 5.3 블록 슬롯 규격 (동결 — AI가 발명하지 못하게)

블록 단위 `u ∈ {1, 2}`. `T`=테마 속성, `C`=counter, `P`=prey, `N`=노말.

| 슬롯 | `element` | `count` | 역할 |
|---|---|---|---|
| **s1** | `T` | `4u` | 본대 |
| **s2** | `C` | `1u` | ×1 소대 |
| **s3** | `T` | `3u` | 본대 |
| **s4** | `P` | `1u` | ★ **×0.5 소대 = 특화의 세금이 눈앞에 서는 자리** |
| **s5** | `N` | `1u` | ×1 소대 |

블록 합계 = `10u` 개체, `T:C:P:N = 7u:1u:1u:1u` = **70/10/10/10 정확** ✔

- **`u = 2`(20 개체) = chaff 위주 블록 / `u = 1`(10 개체) = 대형 개체가 섞인 블록.** `u`가 달라도 블록 비율은 동일하므로 §5.2의 증명은 그대로 성립한다.
- ★ **`s4`(prey)가 이 게임의 셀링 포인트가 매 웨이브 실물로 나타나는 자리다.** 테마 정답 스탠스를 켜고 있으면 s1·s3(70%)은 ×2로 녹지만 **s4는 ×0.5로 튕긴다**(정본 §7.7: 회색 방패 호 + 파티클 0 + 금속 "틴"). 플레이어는 무기를 탓하지 않고 **스탠스를 탓한다** → 정본 §9.9의 온보딩이 "스크립트가 아니라 스테이지 1의 *구성*"이라고 한 것의 실물이 **바로 이 슬롯**이다. 대비 웨이브가 폐기되고도 레슨이 성립하는 이유가 여기 있다.
- `s4`에 **가능하면 큰 밴드를 배치**한다(`line`+). ×0.5로 오래 버티는 개체라야 세금이 체감된다. chaff에 ×0.5는 어차피 1~2히트라 안 보인다.

### 5.4 `eliteIndex` 저작 규칙

`eliteIndex ≠ null`인 웨이브만 **엘리트 후보**이며, 스폰 시 `rng.elite`로 `elitePerWaveChance[stage]`(정본 §8.3)를 굴려 성공하면 그 인덱스의 개체에 접두가 붙는다.

> **저작 규칙 (G-12)**: `eliteIndex = 0`은 **`band ∈ {line, turret, bruiser}` 이고 `element ≠ normal`인 웨이브에만** 적는다. 나머지는 전부 `null`.
> 근거: 정본 §8.6의 `elite.bandAllowed`(chaff 엘리트 = 허수아비 금지) + `elite.elementAllowed`(노말 엘리트 = 상성 재미 0)를 **런타임 거부가 아니라 저작 단계에서** 만족시킨다 → 엘리트 출현율이 굴림 실패로 조용히 낮아지는 사고가 없다.

### 5.5 ★ `sea` 웨이브 리스트 전량 (15 웨이브 — 다른 5 테마의 판형)

테마 `T`=물 · `C`=풀 · `P`=불 · `N`=노말.

| # | 블록 | `formationId` | `archetypeId` | `count` | `element` | `spawnEdge` | `eliteIndex` |
|---|---|---|---|---|---|---|---|
| 1 | T1 (`u=2`) s1 | `vWedge` | `drifter` | 8 | `water` | `top` | `null` (chaff) |
| 2 | T1 s2 | `scatter` | `spitter` | 2 | `grass` | `top` | `null` (chaff) |
| 3 | T1 s3 | `lineH` | `spitter` | 6 | `water` | `top` | `null` (chaff) |
| 4 | T1 s4 | `arc` | `spitter` | 2 | **`fire`** | `top` | `null` (chaff) |
| 5 | T1 s5 | `scatter` | `drifter` | 2 | `normal` | `top` | `null` |
| 6 | T2 (`u=1`) s1 | `vWedge` | `spitter` | 4 | `water` | `top` | `null` (chaff) |
| 7 | T2 s2 | `scatter` | `sirenRay` | 1 | `grass` | `top` | **`0`** |
| 8 | T2 s3 | `scatter` | `sirenRay` | 3 | `water` | `top` | **`0`** |
| 9 | T2 s4 | `scatter` | `spitter` | 1 | **`fire`** | `top` | `null` (chaff) |
| 10 | T2 s5 | `lineH` | `drifter` | 1 | `normal` | `top` | `null` |
| 11 | T3 (`u=1`) s1 | `lineH` | `spitter` | 4 | `water` | `top` | `null` (chaff) |
| 12 | T3 s2 | `scatter` | `stalker` | 1 | `grass` | `top` | **`0`** |
| 13 | T3 s3 | `scatter` | `stalker` | 3 | `water` | `top` | **`0`** |
| 14 | T3 s4 | `arc` | `sirenRay` | 1 | **`fire`** | `top` | **`0`** |
| 15 | T3 s5 | `vWedge` | `drifter` | 1 | `normal` | `top` | `null` |

**S8 검산 (스테이지별 필터링 후, 개체 수 기준)**

| 스테이지 | 포함 블록 | 총 개체 | 물 | 풀 | 불 | 노말 | 비율 | 오차 |
|---|---|---|---|---|---|---|---|---|
| 1 | T1 | 20 | 14 | 2 | 2 | 2 | **70/10/10/10** | **0.0%p** ✔ |
| 2 | T1+T2 | 30 | 21 | 3 | 3 | 3 | **70/10/10/10** | **0.0%p** ✔ |
| 3~5 | T1+T2+T3 | 40 | 28 | 4 | 4 | 4 | **70/10/10/10** | **0.0%p** ✔ |

★ **14번 웨이브가 이 테마의 급소다.** 바다를 **물+4로 밀고 온 특화 빌드**가 만나는 `sirenRay`(불, `bruiser`, HP 26×12×스케일)는 **물 스탠스로 ×0.5**다. 노말(Q)로 바꿔 ×1로 때리는 것이 정답이고(정본 §8.16이 증명한 Q의 존재 이유), 그 판단을 **잡몹 페이즈에서 미리 시킨다.** 그리고 이 웨이브는 `eliteIndex: 0` → 엘리트가 붙으면 HP ×4 = **×0.5로는 절대 못 녹는 벽**. 스테이지 보스(§10)가 요구할 커버리지를 여기서 예고한다.

### 5.6 나머지 5 테마의 블록 편성 (슬롯 규격은 §5.3 고정 → `formationId:archetypeId` 만 지정)

| 테마 | 블록 | s1 (`T`, 4u) | s2 (`C`, 1u) | s3 (`T`, 3u) | s4 (`P`, 1u) | s5 (`N`, 1u) |
|---|---|---|---|---|---|---|
| **`glacier`** (T=물, C=풀, P=불) | T1 `u=2` | `vWedge:spitter` | `scatter:spitter` | `lineH:spitter` | `arc:spitter` | `scatter:spitter` |
| | T2 `u=1` | `arc:frostLance` ★(2) · `arc:spitter`(2) 분할 | `scatter:turretPod` ★ | `lineH:frostLance` ★ | `scatter:turretPod` ★ | `vWedge:spitter` |
| | T3 `u=1` | `columnV:columnAnt` | `arc:frostLance` ★ | `scatter:turretPod` ★ | `scatter:frostLance` ★ | `lineH:spitter` |
| **`volcano`** (T=불, C=물, P=풀) | T1 `u=2` | `vWedge:spitter` | `scatter:spitter` | `lineH:spitter` | `arc:spitter` | `scatter:spitter` |
| | T2 `u=1` | `arc:spitter` | `scatter:magmaBomb` ★ | `lineH:magmaBomb` ★ | `scatter:magmaBomb` ★ | `vWedge:spitter` |
| | T3 `u=1` | `scatter:rearDart` **`bottom`** | `lineH:mortarHulk` ★ | `arc:magmaBomb` ★ | `scatter:mortarHulk` ★ | `scatter:rearDart` **`bottom`** |
| **`desert`** (T=불, C=물, P=풀) | T1 `u=2` | `scatter:dustRunner` ★ | `vWedge:drifter` | `lineH:drifter` | `scatter:dustRunner` ★ | `scatter:drifter` |
| | T2 `u=1` | `columnV:columnAnt` ★ | `scatter:dustRunner` ★ | `scatter:dustRunner` ★ | `scatter:columnAnt` ★ | `lineH:drifter` |
| | T3 `u=1` | `scatter:rearDart` **`bottom`** | `scatter:dustRunner` ★ | `columnV:columnAnt` ★ | `scatter:rearDart` **`bottom`** | `scatter:drifter` |
| **`forest`** (T=풀, C=불, P=물) | T1 `u=2` | `pincer:thornWeaver` ★ (4) · `vWedge:spitter`(4) 분할 | `scatter:spitter` | `vWedge:spitter` | `arc:thornWeaver` ★ | `lineH:spitter` |
| | T2 `u=1` | `pincer:flanker` ★ | `scatter:thornWeaver` ★ | `lineH:spitter` | `scatter:flanker` ★ | `vWedge:spitter` |
| | T3 `u=1` | `arc:hexer` ★ | `scatter:thornWeaver` ★ | `vWedge:spitter` | `scatter:hexer` ★ | `lineH:flanker` ★ |
| **`bog`** (T=풀, C=불, P=물) | T1 `u=2` | `arc:hexer` ★ | `scatter:hexer` ★ | `vWedge:hexer` ★ | `scatter:hexer` ★ | `lineH:hexer` ★ |
| | T2 `u=1` | `pincer:flanker` ★ | `scatter:flanker` ★ | `arc:bogHexer` ★ | `scatter:bogHexer` ★ | `lineH:flanker` ★ |
| | T3 `u=1` | `pincer:flanker` ★ | `arc:hexer` ★ | `lineH:turretPod` ★ | `scatter:bogHexer` ★ | `scatter:hexer` ★ |

★ = `eliteIndex: 0` (band ≥ `line` 이고 `element ≠ normal` → §5.4 규칙 충족). 무표기 = `null`.
`spawnEdge` 무표기 = `top`. **`bottom`은 `rearDart`(T3 블록 = `unlockStageMin 3`) 뿐** → `rearSpawnAllowed[3+]` 자동 준수 ✔

- **`forest` T1 s1**은 `4u = 8`을 `pincer:thornWeaver`(4) + `vWedge:spitter`(4)로 **두 레코드로 분할**한다. 분할해도 `element`가 같으므로 블록 비율은 불변이고, **8기의 `thornWeaver`가 동시에 `wall`을 치는 텔레그래프 폭발**(§12.2)을 피한다. → **분할은 허용되며 유일한 제약은 "같은 슬롯의 레코드는 `element`가 같고 `count` 합이 슬롯 규격과 같을 것"**이다.
- **`glacier` T2 s1**도 같은 이유로 `arc:frostLance`(2) + `arc:spitter`(2)로 분할한다.
- `pincer`는 `strafe` 전용, `columnV`는 `column` 전용 (정본 §8.7) → `flanker`/`thornWeaver` / `columnAnt` 에서만 사용 ✔. **`count = 1`에 `pincer`(좌·우 동시 진입)·`columnV`(종대)는 저작 금지** (G-13) — 1기로는 편대가 성립하지 않는다.
- **`bog` T1이 전부 `hexer`인 것은 의도다.** `bog`는 `introOk: false`이므로 **스테이지 1에 배치될 수 없고**, 따라서 T1 블록이 단독으로 등장하는 스테이지가 존재하지 않는다 — 항상 T2와 함께 온다. 그래도 S8은 T1 단독으로도 70/10/10/10이므로 **어떤 필터링에서도 안전**하다(§5.2).

### 5.7 `formations` 파라미터 (`data/stages.json > formations`)

정본 §9.9는 `formations`를 `{"lineH":"...", ...}`로 자리만 잡아 두었다 → 필드 확정은 §14 **R-14**.

| `formationId` | 파라미터 | 배치 |
|---|---|---|
| `lineH` | `{"spreadPx":380,"jitterPx":0,"stepSec":0}` | 아레나 폭 380px에 균등 수평, 동시 진입 |
| `columnV` | `{"spreadPx":0,"jitterPx":0,"stepSec":0.35}` | 같은 x, `stepSec` 간격 순차 (`column.gapSec`와 짝) |
| `vWedge` | `{"spreadPx":300,"jitterPx":0,"stepSec":0,"depthPx":34}` | V자, 뒤로 갈수록 `depthPx`씩 하강 |
| `arc` | `{"spreadPx":440,"jitterPx":0,"stepSec":0,"depthPx":60}` | 상단 호 |
| `pincer` | `{"spreadPx":0,"jitterPx":0,"stepSec":0,"yStartPx":140,"yStepPx":90}` | 좌·우 벽 동시. **`i`번째 개체의 `strafe.yPx`를 `yStartPx + floor(i/2) × yStepPx`로 덮어쓴다** |
| `scatter` | `{"spreadPx":500,"jitterPx":40,"stepSec":0.5}` | 랜덤 x(`rng.spawn`), `stepSec` 순차 |

- ★ **`pincer`만 `moveParams`를 덮어쓴다.** 그렇지 않으면 `flanker`의 `yPx: 180`이 고정이라 **모든 측면기가 같은 높이로만 지나가** `strafe`가 한 줄짜리 거동이 된다. 편대가 배치를 소유한다는 원칙의 유일한 예외이며, `strafe` 전용 편대이므로 스코프가 닫혀 있다.
- `spreadPx ≤ 540`(플레이어 이동 영역 폭, 정본 §1.1) — `arc`의 440이 최대. **아레나 밖 스폰은 없다.**

---

## 6. 엘리트 (정본 §8.6 전면 인용 — 이 섹션은 아무것도 정하지 않는다)

**엘리트 = 접두(prefix) 플래그이며 별도 개체가 아니다.** `enemies[].tier` 필드는 존재하지 않는다.

| 키 | 값 | 출처 |
|---|---|---|
| `eliteIndex` | 웨이브 레코드의 `int \| null` | 정본 §8.6 |
| `elite.perWaveMax` / `hpMult` / `sizeMult` / `contactDmgMul` / `xpMult` / `coin` / `healDropChance` | 1 / 4.0 / 1.4 / 1.5 / 6.0 / 3 / 0.12 | 정본 §9.4 `rules.elite` |
| `elite.bandAllowed` / `elementAllowed` | `["line","turret","bruiser"]` / `["fire","water","grass"]` | 정본 §9.4 |
| 표식 | 회전 이중 외곽선(속성색) + 개체 위 속성색 HP 바 + 상시 글리프 | 정본 §7.6 |

**이 섹션이 하는 일 = §5.4의 저작 규칙(G-12) 하나뿐**: `bandAllowed`/`elementAllowed`를 **런타임 거부가 아니라 웨이브 저작 시점에** 만족시킨다.

- **엘리트 후보 웨이브 수 검산 (`dominance`·`coinScarcity` 게이트 대비)**: `sea` = 15 웨이브 중 후보 5개(#7,8,12,13,14). 스테이지 5(`elitePerWaveChance = 0.30`) 기준 기댓값 = **1.5기/스테이지** → 코인 4.5 + 회복 드랍 기대 0.18. `bog` = 후보 15개 전부 → 스테이지 5 기준 **4.5기** → 코인 13.5. **테마 간 코인 수급 격차가 3배**다.
  → ★ **이것은 버그가 아니라 `salvage`(코인 획득 패시브)·상점 가격 곡선이 흡수해야 할 분산이며, `certify.coinScarcity.medianEndCoins ∈ [0, 80]`이 판정한다.** 다만 `dominance.maxThemeClearStddev ≤ 0.06`을 위협할 수 있으므로 **시뮬 1차 조정 손잡이 = 후보 웨이브 수**임을 명시한다(값이지 규칙이 아니다). `bog`의 후보를 5개로 줄이는 것이 첫 번째 조정이다.

---

## 7. 중간보스 (`data/bosses.json`, `tier: "mid"`)

### 7.1 정본이 확정한 것 (인용)

**단일 몸체·부위 없음·3종 전 테마 공용.** `midBossCount`(1~2) · `midBossAtSec` `[35]`/`[30,70]` · `hp = midBoss.hp × enemyHpScale[stage]` · **`midBossLeaveAfterSec = 30`**(= "선택적"의 정의) · `midBossElementRule = "notThemeAndNotNormal"` · `midBossForcedLeaveOnCrisis` · `midBossSummonsAllowed = "mbNest"만` · 보상 `xp` = chaff XP × 25 / `coin` **5** / `healDropChance` **0.35**. (정본 §8.9)

**속성 규칙 = 보스전의 예고편**: 테마 정답 스탠스를 켠 채로는 안 죽는다 → **스탠스를 바꾸게 만든다.**

### 7.2 3종 확정

| 필드 | `mbHammer` 파쇄추 | `mbLancer` 창병 | `mbNest` 산란모함 |
|---|---|---|---|
| `moveId` | `anchor` | **`charge`** | `anchor` |
| `moveParams` | `{"enterSpeed":110,"yHoldPx":150,"swayAmpPx":190,"leaveAfterSec":30}` | `{"windUpSec":1.20,"dashSpeed":300}` | `{"enterSpeed":40,"yHoldPx":210,"swayAmpPx":60,"leaveAfterSec":30}` |
| `patternSet[0].emitterIds` | `["hammerFan","hammerZone"]` (교대) | `["lancerLaser"]` | `["nestAimed"]` |
| `summon` | — | — | `{"archetypeId":"drifter","count":3,"everySec":6.0,"formationId":"scatter"}` |
| `shapeId` / `radius` | `slab` / 34 | `spike` / 28 | `bulb` / 38 |
| `hp` (base) | 900 | 750 | 1100 |
| `contactDmg` | 16 | 15 | 14 |
| `xp` / `coin` / `healDropChance` / `score` | 75 / 5 / 0.35 / 800 | 75 / 5 / 0.35 / 700 | 75 / 5 / 0.35 / 900 |
| `element` | **`null`** (런타임 주입, →R-11) | `null` | `null` |
| `leaveAfterSec` | 30 | 30 | 30 |
| 시험하는 것 (정본 §8.9) | 공간 압박 — 위치 판단 | 라인 회피 — 텔레그래프 읽기 | **DPS 체크 — 보스전 예고편** |

**이미터** (`enemies.json > emitters`에 추가)

| `id` | `type` | `bulletId` | `telegraphSec` | `everySec` | `offsetSec` | 고유 |
|---|---|---|---|---|---|---|
| `hammerFan` | `fan` | `heavyRound` | **1.20** | 6.0 | **0.0** | `count:9, arcDeg:120, speed:115` |
| `hammerZone` | `zone` | `null` | **1.20** | 6.0 | **3.0** | `radius:82, activeSec:3.0, dmg:10` |
| `lancerLaser` | `laser` | `beamCore` | **1.20** | 5.0 | 0.0 | `chargeSec:1.2, widthPx:22, activeSec:0.6, angleDeg:90, trackDuringCharge:true` |
| `nestAimed` | `aimed` | `homingM` | **1.20** | 3.2 | 0.0 | `count:3, spreadDeg:26, speed:90, leadSec:0.4` |

- **`telegraphSec = 1.20` = 정본 §7.4의 "중간보스 패턴 = 전신 자홍 림 라이트, 하한 1.20"** ✔ 전 이미터 준수. `zone`(0.90)·`fan`(0.60)·`aimed`(0.60) 각자의 하한보다 **중간보스 하한이 크므로 1.20이 이긴다** — §3.3의 "두 하한이 겹치면 큰 쪽" 규칙 재적용.
- ★ **`hammerFan`/`hammerZone`의 `offsetSec` 0.0 / 3.0 (주기 6.0)** → 두 텔레그래프가 **정확히 3.0초 간격으로 교대**하며 각각 1.20초만 켜진다 → **동시 텔레그래프 = 항상 1개** ≤ 2 ✔ **S7 정적 통과.** "fan + zone 교대"(정본 §8.9)가 새 규칙 없이 `offsetSec` 하나로 표현된다.
- `lancerLaser.trackDuringCharge: true` + `charge.windUpSec 1.20` — **둘 다 ≥ `minTelegraphSec`** ✔. 돌진과 레이저가 각각 1.2초 예고를 갖는다.
- `mbLancer`의 `dashSpeed: 300`은 **적 기체 속도**이며 `fairness.maxBulletSpeed`(260, **탄** 상한)의 적용 대상이 아니다. 회피 예산은 `windUpSec 1.20` + 아레나 h=720이 준다.
- `nestAimed`의 `speed: 90` ≤ `maxAimedBulletSpeed` 200 ✔

### 7.3 스키마가 표현하지 못하는 것 2건 (→ §14)

| 정본이 확정한 기능 | 스키마에 있는가 | 요청 |
|---|---|---|
| `mbHammer` = "`fan` + `zone` **교대**" (§8.9) | ✖ — §9.8은 중간보스 `patternSet`을 **1개**로 못박았고 엔트리는 `{emitterId}` **단수**다 | **R-8** `patternSet[i] = {emitterIds: [...]}` (배열, 길이 1~2) |
| `mbNest` = "**잡몹 소환**" (§8.9) + `midBossSummonsAllowed` (§8.9) | ✖ — 소환을 표현할 필드가 `bosses.json` 어디에도 없다. `onDestroy.spawnWave`는 폐기됨(§8.12) | **R-9** `bosses[].summon` 필드 신설 |

> 두 건 모두 **정본 §8.9(기능 확정)와 §9.8(스키마)의 내부 불일치**이며, 이 섹션이 값을 정해 우회할 수 있는 종류가 아니다. 위 표의 `emitterIds`/`summon`은 **요청이 승인된다는 전제의 저작**이다.

### 7.4 `mbNest`의 소환이 정본을 깨지 않는 이유 (확인)

`boss.summonsAllowed = false`(정본 §8.11)는 **스테이지 보스** 규칙이고, `mbNest`는 **잡몹 페이즈**에 있다(정본 §8.9: "중간보스는 잡몹 페이즈에 있으므로 XP 획득이 정상"). → §6.4의 "보스 페이즈에 XP 획득원이 존재하지 않는다"는 전제는 **영향받지 않는다** ✔ 소환된 `drifter`는 `xp: 2`를 정상 지급하며 이것이 `mbNest`를 "DPS 체크 + 파밍 기회"의 이중 선택으로 만든다.

---

## 8. 위기 세션 (새떼)

### 8.1 정본이 확정한 것 (인용)

`crisisPerStage 1` · `crisisStartSec 95` / `crisisDurationSec 25` / `crisisWarnSec 3.0` · `crisisSuspendsWaves` · **`crisisTotal = 60 × swarmTotalScale[stage]`** · `crisisSubWaves 6` · **`crisisFailCondition` 없음** · **`crisisElementRule = "themePure"`(테마 속성 100%)** · `swarmXp` = chaff XP × 0.5. (정본 §8.10)

**핵심**: 60기 전부 테마 속성 → **정답 스탠스 투자자는 25초 동안 ×2로 전부 갈아버린다.** 실패해도 죽지 않는다(잔존 전원 소멸). 정답 3개 = `nova` / `aura` / **폭탄**(상점 상시).

### 8.2 서브웨이브 편성 (이 섹션의 위임분)

`crisisTotal 60` ÷ `crisisSubWaves 6` = **10기 × 6파**. 편대 = `arc` / `vWedge` (정본 §8.10).

| 서브웨이브 | `formationId` | `swarmChaff` | `swarmLancer` | `element` |
|---|---|---|---|---|
| 1 | `arc` | 9 | 1 | 테마 |
| 2 | `vWedge` | 9 | 1 | 테마 |
| 3 | `arc` | 9 | 1 | 테마 |
| 4 | `vWedge` | 9 | 1 | 테마 |
| 5 | `arc` | 9 | 1 | 테마 |
| 6 | `vWedge` | 9 | 1 | 테마 |

- **비율 90/10 = `swarmChaff` 54기 : `swarmLancer` 6기** (정본 §8.6: `swarmChaff`(90%) / `swarmLancer`(10%)) ✔ 서브웨이브마다 9:1로 **균등 분산** → 어느 25초 구간을 잘라도 비율이 같다.
- 서브웨이브 간격 = `crisisDurationSec / crisisSubWaves` = **25 / 6 ≈ 4.17 게임초**. (→ R-15: 정본에 간격 키가 없다. 파생값으로 두는 것을 제안.)
- `spawnEdge` = `top` 고정. **새떼는 `bottom`에서 오지 않는다** — 25초 최고밀도 구간에 후방 진입을 겹치면 관대함이 붕괴한다. `rearSpawnAllowed`가 참인 스테이지에서도 금지 (G-14).
- `eliteIndex` = **전부 `null`.** 새떼는 혼합 비율 제외 대상이자 엘리트 제외 대상이다(정본 §8.2: "혼합 비율 제외 대상 = 엘리트 · 중간보스 · 새떼 · 보스"). `swarm*`은 `chaff` 밴드라 `elite.bandAllowed`에서도 이미 배제된다 → **이중으로 막혀 있다** ✔

### 8.3 밀도·캡 검산

| 항목 | 계산 | 판정 |
|---|---|---|
| 동시 개체 | 10기 × 6파, 간격 4.17초. `swarmChaff.speed 130` → 아레나 760px 종단 **5.8초** → 동시 ≈ 10 × (5.8/4.17) ≈ **14기** (플레이어가 전혀 못 잡을 때). 스테이지 6(`swarmTotalScale 1.0`) 최악에도 **≪ `swarmConcurrentMax` 70** | ✔ |
| B층 캡 | `caps.enemies` 96 > 새떼 최악 14 + 잔존 웨이브(`crisisSuspendsWaves` = 신규 정지) | ✔ `capHits = 0` |
| 적 탄 | `swarmLancer` 6기 × `lancerStraight`(count 1, everySec 6.0) → **동시 탄 ≪ 320** | ✔ |
| **텔레그래프** | `swarmLancer` 6기 × duty(0.60/6.0 = 10%) = **0.6 동시** | ✔ |

★ **`swarmChaff`의 `attack: null`이 위기 세션을 캡 안에 묶는 유일한 장치다.** 60기 중 54기가 텔레그래프를 갖지 않으므로, 밀도 최고 구간에서 **`caps.telegraphs`가 문제가 되지 않는다**(§12.2의 예산 문제는 잡몹 페이즈 쪽이지 새떼 쪽이 아니다). 정본 §8.6이 `swarmChaff`를 "사격 안 함"으로 못박은 것이 여기서 캡 산술로 회수된다.

### 8.4 `crisisClearWithoutCapstone ≥ 0.80` 대비 (정본 §13.1)

| 해답 | 이 섹션의 편성에서 실제로 작동하는가 |
|---|---|
| `nova` | 60기 × HP(3×1.0×`enemyHpScale`) — 스테이지 5 기준 개체 HP **13.5** → 노바 1회 광역이 서브웨이브 1파를 통째로 지운다 ✔ |
| `aura` | `arc`/`vWedge` = **밀집 편대** → 오라 반경에 다수가 동시에 들어온다 ✔ |
| **폭탄** | `bomb.mobDmg 9999` + 탄 전량 소거 → **1개로 화면의 새떼 전멸.** 상점 `bomb` `basePrice 50`, 스톡 상한 3 | ✔ **보편 해답** |

- ★ **개체 HP를 `swarmChaff.hp = 3`(chaff, `hpMult 1.0`)로 낮게 잡은 것이 이 게이트의 실물이다.** 스테이지 6(`enemyHpScale 6.0`)에서도 개체 HP는 **18** — 즉 **어떤 무기 조합이든 1~2히트**다. 새떼가 시험하는 것은 단발 화력이 아니라 **광역·화면 커버리지**이며, 그것을 못 갖춘 빌드도 **폭탄 1개**로 통과한다.
- **XP 덩어리 검산**: 60기 × `xp 2` = **120**. `sea` 스테이지 3 웨이브 총 XP ≈ 40개체 × 평균 3.4 = 136 → 새떼 = **120/(136+120) ≈ 47%**. 정본 §8.10의 "스테이지 총 XP의 ~25%"보다 **크다** → `swarmChaff.xp`를 **2 → 1**로 내리면 ≈ 31%, `xp: 1` + `swarmLancer.xp: 1`이면 60 XP → **31%**. → **시뮬 1차 조정 손잡이 = `swarmChaff.xp` (값이지 규칙이 아니다).** 정본의 25%를 목표로 두고 초기값 `2`로 시작해 `--certify`의 `farmXpRatio`와 함께 내린다.

---

## 9. 복합 보스 — 저작 규칙

### 9.1 정본이 확정한 것 (인용, 재정의 아님)

`boss.partCount` **4** (최종만 5) · `partRegen false` · **`summonsAllowed false`** · `partHitPriority "outermostFirst"` · **코어 HP = 보스 HP, 절대값**(`hpShare` 없음) · `phaseThresholds [0.6, 0.3]`(**코어 HP** 기준) · `introSec 3.0` · `timerStartsAfterIntro` · 부위/보스 XP **0/0** · 부위는 개체이며 자체 `score`·상성 처치 보너스 대상 · `coin 12` / `partCoin 2` / `healDrop` 없음 · `bomb.bossDmgRatio`는 **코어 최대 HP 기준, 부위 미적용**. (정본 §8.11)
`partType` 4종 `mobility`(이동 ×0.5·회피 기동 중단) / `armament`(그 부위 이미터 영구 삭제) / `armor`(코어 게이트 1단계) / `core`(사망). `onDestroy` **없음**. (정본 §8.12)
**코어 게이트** = `coreGateMul(0.4) ^ 살아있는 armor 수` → armor 2 = ×0.16 / 1 = ×0.40 / 0 = ×1.00. **`mobility`·`armament`는 게이트에 영향 없음** = 3분 타이머 하의 진짜 트레이드오프. (정본 §8.13)
**R1~R6** — 전부 `check.mjs`(S5)가 강제. (정본 §8.14)

★ **`partCount`는 코어를 포함한다.** 정본 §8.15가 강철 가오리(스러스터 + 좌 + 우 + 코어)를 "partCount = 4 ✔"로 검증했다 → **`parts[]` 배열 길이 = 3, 코어 1 = 합 4.** 최종은 `parts[]` 4 + 코어 1 = 5. **코어는 `parts[]`에 들어가지 않는다**(§9.8 스키마가 `core`를 별도 필드로 둔다).

★ **코어는 사격하지 않는다.** 정본 §9.8의 `core` 스키마에 `patternSet`이 없다 → **탄을 쏘는 것은 주변부 3개(최종 4개)뿐.** 이것이 §9.3의 텔레그래프 산술의 전제다. 그리고 "부위를 다 깨면 보스가 무장해제된다"가 규칙 없이 성립한다 — 마지막 국면이 항상 순수 DPS 레이스가 되는 이유.

### 9.2 ★ 부위 파괴 시 그 부위의 이미터는 정지한다 (해석 명시)

정본 §8.12는 `armament`에 대해서만 "그 부위가 담당하던 이미터가 영구 삭제"라고 적었다. **`armor`·`mobility` 부위가 파괴된 뒤에도 사격하는가?** → 이 섹션은 **정지**로 전제한다. 근거: 정본 §7.6이 파괴된 부위를 "외곽선 소멸 + 본체 `#3A3A3A` + 연기 + 글리프 제거"로 규정 → **화면에서 죽은 것이 탄을 쏘면 I-2(색은 거짓말하지 않는다) 위반.** `armament`에만 그 문구가 있는 이유는 **`armament`의 유일한 효과가 이미터 상실**이기 때문이지(다른 둘은 추가 효과가 있다), 다른 부위가 계속 쏜다는 뜻이 아니다. → §14 **R-13**(명시 요청).

### 9.3 ★ 라운드로빈 `offsetSec` — S7을 정적으로 증명하는 장치

**문제**: 주변부 3개가 동시에 텔레그래프를 켜면 `telegraphConcurrentMaxPerEntity = 2`(정본 §12.4)를 넘고, S7("보스 `patternSet`을 3페이즈 전부 전개해 개체당 ≤ 2 정적 검사")이 실패한다.

**해법 — 저작 규칙 3줄 (전 보스 공통, 동결)**:

```
① 페이즈별 everySec 고정:   phase1 = 6.0 · phase2 = 4.5 · phase3 = 3.6      (게임초)
② 부위 i(0-based)의 offsetSec = everySec × i / P                             (P = 사격 부위 수)
③ 전 부위 telegraphSec = 1.50                                                (= 정본 §7.4 "보스 대형 패턴" 하한)
```

**정적 증명**: 부위들의 발사 위상이 `everySec / P` 균등 간격이므로, 길이 `T`인 텔레그래프가 동시에 겹치는 최대 개수 = `ceil(T ÷ (everySec/P))` = `ceil(P × T / everySec)`.

| | `P` | `everySec` | `P × T / everySec` (T = 1.50) | 동시 텔레그래프 | 판정 |
|---|---|---|---|---|---|
| 일반 보스 페이즈 1 | 3 | 6.0 | 0.75 | **1** | ✔ |
| 일반 보스 페이즈 2 | 3 | 4.5 | 1.00 | **1** | ✔ |
| 일반 보스 페이즈 3 | 3 | 3.6 | **1.25** | **2** | ✔ |
| 최종 페이즈 1 | 4 | 6.0 | 1.00 | **1** | ✔ |
| 최종 페이즈 2 | 4 | 4.5 | 1.33 | **2** | ✔ |
| 최종 페이즈 3 | 4 | 3.6 | **1.67** | **2** | ✔ |

**정적 부등식 (`check.mjs`가 검사할 형태)**: `P × telegraphSec ≤ 2 × everySec`. 최악(최종 페이즈 3) = `4 × 1.50 = 6.0 ≤ 2 × 3.6 = 7.2` ✔ **여유 20%.**

> ★ **이 규칙이 "AI가 생성한 보스 스크립트가 트위치가 되는" 실패 모드를 산술로 봉인한다.** AI는 `everySec`·`offsetSec`을 자유 저작할 수 없고 위 3줄에서 파생한다 → **동시 텔레그래프 초과가 발생할 자유도가 존재하지 않는다.** 그리고 `caps.telegraphs`(8) 안전망에 닿을 일도 없다(최대 2) → 보스전은 `capHits = 0`이 자명하다.
> ★ **부수효과: 보스전이 "1.5초 예고 → 한 부위가 쏜다"의 리듬으로 고정된다.** 페이즈가 올라가면 그 리듬이 6.0 → 4.5 → 3.6초로 조여든다. **읽는 부담은 그대로(항상 1~2개), 여유만 줄어든다** = 정본 §6.2의 "어려워지는 것은 읽고 반응할 시간이지 무엇인지 알아보는 것이 아니다"의 보스전 판본.

### 9.4 ★ 페이즈 파생 규칙 — 이미터 66종을 시드 22종에서 만든다

정본 §8.11: "페이즈마다 각 부위의 `patternSet` 인덱스가 교체된다(`everySec` 단축, `count`/`arcDeg` 증가). **새 이미터 타입이 생기지는 않는다.**" → 이 문장을 **빌드타임 생성 규칙**으로 못박는다.

```
부위마다 시드 이미터 1개(type · bulletId · 고유 파라미터)를 저작한다.
patternSet[p] 의 이미터 (p = 0,1,2):
  type, bulletId          = 시드 그대로 (★ 타입은 절대 안 바뀐다)
  telegraphSec            = 1.50 (고정)
  everySec                = [6.0, 4.5, 3.6][p]
  offsetSec               = everySec × partIndex / P
  count                   = round(seed.count × [1.0, 1.4, 1.8][p])
  arcDeg / rotStepDeg     = round(seed.value × [1.0, 1.15, 1.30][p])
  그 외 파라미터           = 시드 그대로
```

- 7 보스 × 주변부 3(최종 4) = **시드 22개** → `patternSet` 전개 시 **이미터 66개**. 손으로 쓰는 것은 22개뿐이고 나머지는 규칙의 인스턴스다.
- **`speed`는 페이즈 스케일 대상이 아니다** — `fairness.maxBulletSpeed`(260)를 페이즈 스케일이 몰래 넘기는 사고를 원천 차단한다. 페이즈가 올리는 것은 **밀도(`count`)와 빈도(`everySec`)**뿐.
- 생성 규칙이므로 `data/bosses.json`에는 **전개된 결과**가 들어간다(로더는 규칙을 모른다 = 런타임 코드 0).

### 9.5 ★ 보스 HP 모델 — `bossHpScale`가 없으면 성립하지 않는다

**문제 (구조적)**: 테마 순서는 매 런 셔플된다(정본 §8.1) → **`manta`는 스테이지 1에도 5에도 온다.** 그런데 정본은 스테이지 보스에 **어떤 스테이지 스케일도 지정하지 않았다**(§8.3은 *적* HP, §8.9는 *중간보스*에만 `enemyHpScale`을 걸었고, §8.11은 "절대값"이라고만 했다). 절대값 하나로는 5배 차이 나는 플레이어 화력을 **같은 숫자로 상대**하게 된다.

> 정본 §9.9가 `stage1BossHpScale: 0.6`을 폐기하며 남긴 근거 — "스테이지 1 보스는 `bosses.json`에서 **HP가 낮게 저작**되면 그만" — 은 **"스테이지 1 보스"가 고정이라는 전제**에 서 있다. 테마 셔플 하에서 그 전제는 거짓이다. (모순 대장 #25도 같은 전제를 공유한다.) → §14 **R-1 (blocker)**.

**이 섹션의 전제**: `실효 HP = bosses[].core.hp(또는 parts[].hp) × stages.curve.bossHpScale[stage]`, **저작값은 전부 "스테이지 1 기준 base"**.

`bossHpScale = [1.0, 1.9, 3.4, 5.6, 8.4, 11.5]` — **`enemyHpScale`(6배)과 다른 곡선인 이유**: 잡몹 HP는 "chaff는 끝까지 1~2히트"를 유지해야 하므로 완만해야 하고(정본 §8.6), 보스 HP는 **3분 타이머가 재는 것 = 코어 DPS**(정본 §8.11)이므로 **플레이어 화력 성장(≈11.5배)을 그대로 따라가야 한다.** 두 곡선이 같으면 후반 보스가 무너진다.

**참조 화력 곡선** (`report/bosses.csv`가 교정할 시드값): 스테이지 1~6 유효 DPS ≈ **45 / 85 / 150 / 240 / 360 / 520**. (평균 상성 배율 ≈1.4 포함, 정본 §11.1의 성장 예산 54픽 기준.)

### 9.6 HP 배분 규칙 (전 보스 공통)

| 규칙 | 값 | 근거 |
|---|---|---|
| 목표 격파 시간 | **110~135 게임초** (par 잔여 45~70) | 180초 타이머에 회피·전환·오조준 여유 |
| `armor` 부위 HP 총합 | **코어 HP의 20~50%** | armor 2개 보스 = 20~25% / armor 1개 보스 = **45~50%** (게이트가 약하므로 부위 자체가 무거워야 균형) |
| `mobility`/`armament` 부위 HP | **코어 HP의 15~25%** | 선택이므로 **부담 없이 포기 가능해야** 한다 |
| 코어 직행 성립 조건 | `코어 HP ÷ (0.4^armor수) ÷ DPS` vs 180 | armor 2 → **항상 불가**(≥222초) / armor 1 → **아슬하게 성립** |
| `contactDmg` | 코어 16 / 부위 14~15 | 정본 §2.5 저작 가이드(중간보스·보스·부위 14~16) |

---

## 10. 보스 7종 (`data/bosses.json`)

> **테마 6 + 최종 1 = 7종.** 런당 등장은 5 + 1 = **6종**. → 정본 §9.10의 "보스 6종"은 런당 등장 수이며 저작 수는 7이다(§14 **R-16**, 표기 정정 요청).
> HP는 전부 **스테이지 1 기준 base** (§9.5).

### 10.1 ★ R1~R6 + `partCount` 전수 검증표 (S5)

| 보스 | 테마속성 | 주변부 (partType/속성) | **R1** core=노말 | **R2** 노말 주변부 금지 | **R3** distinct ≥2 | **R4** 테마속성 ≤1 | **R5** armor≠테마 | **R6** armor ∈{1,2} | `partCount` |
|---|---|---|---|---|---|---|---|---|---|
| **`manta`** 강철 가오리 | 물 | `mobility`/물 · `armor`/불 · `armor`/풀 | ✔ | ✔ | **3** ✔ | 물 **1** ✔ | 불,풀 ≠ 물 ✔ | **2** ✔ | 3+1 = **4** ✔ |
| **`frostCrown`** 빙관 | 물 | `armament`/불 · `armor`/풀 · `armor`/불 | ✔ | ✔ | **2** ✔ | 물 **0** ✔ | 풀,불 ≠ 물 ✔ | **2** ✔ | **4** ✔ |
| **`kiln`** 용광로 거신 | 불 | `armament`/불 · `armament`/풀 · `armor`/물 | ✔ | ✔ | **3** ✔ | 불 **1** ✔ | 물 ≠ 불 ✔ | **1** ✔ | **4** ✔ |
| **`scarab`** 태양갑충 | 불 | `armor`/물 · `mobility`/불 · `armament`/풀 | ✔ | ✔ | **3** ✔ | 불 **1** ✔ | 물 ≠ 불 ✔ | **1** ✔ | **4** ✔ |
| **`thornKing`** 가시왕 | 풀 | `armor`/물 · `armor`/불 · `armament`/풀 | ✔ | ✔ | **3** ✔ | 풀 **1** ✔ | 물,불 ≠ 풀 ✔ | **2** ✔ | **4** ✔ |
| **`mire`** 늪주 | 풀 | `armor`/불 · `mobility`/물 · `armament`/풀 | ✔ | ✔ | **3** ✔ | 풀 **1** ✔ | 불 ≠ 풀 ✔ | **1** ✔ | **4** ✔ |
| **`tetrarch`** 테트라크 | **없음**(`null`) | `armor`/물 · `armor`/불 · `armor`/풀 · `armament`/**노말** | ✔ | **면제** (`allowNormalPeripheral`) | **4** ✔ | **면제** (`exemptRules:["R4"]`) | 테마 없음 → 공허참 ✔ | **3** ← ★ **R6 위반** | 4+1 = **5** ✔ |

★ **`tetrarch`가 R6를 위반한다 (blocker).** 정본 §9.4는 `armorPartCountRange: [1,2]`와 `finale.armorPartCount: 3`을 **둘 다** 선언했고, S5는 "R1~R6 전부 + `partCount == 4`(finale는 5, `exemptRules` 적용) + **armor 수 ∈ [1,2]**"라고 적었다. `finale.exemptRules`에는 `["R4"]`만 있다 → **문면 그대로 실행하면 `check.mjs`가 최종 보스를 로드 실패시킨다.** 이것은 정본 §8.15가 초안 F의 강철 가오리 판본에서 잡아낸 것과 **정확히 같은 종류의 결함**이며, 이번엔 정본 자신에게 있다. → §14 **R-2 (blocker)**.

**구조 다양성 확인 (`dominance.maxThemeClearStddev ≤ 0.06` 대비)**: armor 2 보스 3종(`manta`·`frostCrown`·`thornKing`) / armor 1 보스 3종(`kiln`·`scarab`·`mire`) / armor 3 = 최종. **속성당 2 테마가 정확히 armor 2 + armor 1로 갈린다** → 같은 속성 테마 두 개를 뽑아도 보스전 구조가 다르다.

### 10.2 「강철 가오리 (Manta)」 — `sea` / 물 — **정본 §8.15 판본**

| 부위 | `partType` | `element` | **정답 스탠스** | `hp` | `radius` | `anchor` | `shapeId` | `contactDmg` | `score` | 파괴 효과 |
|---|---|---|---|---|---|---|---|---|---|---|
| `thruster` 스러스터 | `mobility` | **물** | 풀(R) | **900** | 22 | `[0,44]` | `fin` | 14 | 1200 | 이동 ×0.5, 좌우 회피 기동 중단 |
| `finL` 좌 지느러미 | `armor` | 불 | **물(E)** | 900 | 26 | `[-52,10]` | `fin` | 14 | 1500 | 코어 게이트 1/2 |
| `finR` 우 지느러미 | `armor` | 풀 | **불(W)** | 900 | 26 | `[52,10]` | `fin` | 14 | 1500 | 코어 게이트 2/2 |
| `core` 몸체 | `core` | 노말 | 아무거나 | **4000** | 42 | `[0,0]` | `bulb` | 16 | 8000 | **사망** |

`movePattern: "sway"`, `movePatternParams: {"speedPxSec":34,"ampPx":150,"yHoldPx":170}`

> ★ **`core.hp 4000` · `core.radius 42` · `core.contactDmg 16` · `core.shapeId "bulb"` · `core.score 8000` · `thruster.hp 900` · `thruster.radius 22` · `thruster.anchor [0,44]` · `thruster.contactDmg 14` · `thruster.shapeId "fin"` · `thruster.score 1200` 은 정본 §9.8이 직접 저작한 값이며 그대로 인용했다.** 이 섹션이 더한 것은 `finL`/`finR`의 수치와 `movePattern`뿐이다.

**시드 이미터** (§9.4가 3페이즈로 전개)

| 부위 | 시드 `type` | `bulletId` | 시드 `count` | 고유 |
|---|---|---|---|---|
| `thruster` | `straight` | `pelletS` | 3 | `spreadDeg:26, speed:150` |
| `finL` | `fan` | `fanShard` | 5 | `arcDeg:70, speed:130` |
| `finR` | `aimed` | `heavyRound` | 2 | `spreadDeg:18, speed:115, leadSec:0.4` |

**산술 검증** (스테이지 1, DPS 45 / `bossHpScale[0] = 1.0`)

| 경로 | 계산 | 결과 |
|---|---|---|
| **균형 빌드(2/2/2)** | `finL` 900÷(45×2)=10s + `finR` 900÷(45×2)=10s + 코어 4000÷45=89s | **109s** ✔ par 71s |
| **물+4 특화** | `finL`(불) ×2 = 10s + `finR`(풀) → 불 투자 0 → **Q로 ×1** = 20s + 코어 89s | **119s** ✔ 아슬 |
| **코어 직행** (armor 2, ×0.16) | 4000 ÷ 0.16 ÷ 45 | **555s** ≫ 180 ✗ **불가** |
| **armor 1개만 깨고 직행** (×0.40) | 10s + 4000 ÷ 0.4 ÷ 45 = 222s | **232s** ✗ **불가** → **둘 다 깨야 한다** |
| **스테이지 5 검산** (`bossHpScale[4]=8.4`, DPS 360) | armor 7560÷(360×2)×2 = 21s + 코어 33600÷360 = 93s | **114s** ✔ 곡선 일관 |

- ★ **커버리지 시험이 산술로 작동한다**: 바다를 **물+4**로 밀고 온 특화 빌드는 `finL`(불)만 10초에 녹이고 `finR`(풀)에서 **2배로 오래 걸린다**(정본 §8.15의 "우 지느러미에서 막힌다"). 그리고 **불 스탠스로 `finR`을 때리면 ×0.5**가 아니라 — `finR`은 풀이고 불 → 풀은 ×2다. 즉 **불에 1이라도 투자했으면 즉시 보상**받는다. `elementCapTotal = 6`(정본 §4.2)이 만드는 `4/2/0` 다이얼이 여기서 정확히 값을 갖는다.
- `thruster`(물)는 **순수 선택**: 풀 스탠스로 10초를 써서 깨면 보스가 느려져 남은 89초가 편해진다. **깨지 않아도 코어 게이트에 영향 없음**(정본 §8.13) → 3분 타이머 하의 트레이드오프가 실물이 된다.

### 10.3 「용광로 거신 (Kiln)」 — `volcano` / 불 — **코어 직행이 성립하는 유일한 보스**

| 부위 | `partType` | `element` | **정답 스탠스** | `hp` | `radius` | `anchor` | `shapeId` | `contactDmg` | `score` | 파괴 효과 |
|---|---|---|---|---|---|---|---|---|---|---|
| `turretL` 좌 포탑 | `armament` | **불** | 물(E) | 700 | 24 | `[-58,-6]` | `slab` | 14 | 1300 | **용암 `zone` 이미터 영구 삭제** |
| `turretR` 우 포탑 | `armament` | 풀 | 불(W) | 700 | 24 | `[58,-6]` | `slab` | 14 | 1300 | **`laser` 이미터 영구 삭제** |
| `plate` 흉갑 | `armor` | 물 | **풀(R)** | **3000** | 30 | `[0,34]` | `ring` | 15 | 2400 | 코어 게이트 1/1 (×0.4 → ×1.0) |
| `core` 노심 | `core` | 노말 | 아무거나 | **3000** | 40 | `[0,0]` | `bulb` | 16 | 8000 | **사망** |

`movePattern: "holdCenter"`, `movePatternParams: {"speedPxSec":0,"ampPx":40,"yHoldPx":150}`

**시드 이미터**: `turretL` = `zone`/`null`/`radius:76, activeSec:3.0, dmg:10` · `turretR` = `laser`/`beamCore`/`chargeSec:1.5, widthPx:20, activeSec:0.6, angleDeg:90, trackDuringCharge:true` · `plate` = `ring`/`heavyRound`/`count:10, speed:115, rotOffsetDeg:18`

**산술 검증** (스테이지 1, DPS 45)

| 경로 | 계산 | 결과 |
|---|---|---|
| **풀 투자 있음** | `plate` 3000÷(45×2)=33s + 코어 3000÷45=67s | **100s** ✔ 최적 |
| **풀 투자 0 (Q로 ×1)** | `plate` 3000÷45=67s + 코어 67s | **133s** ✔ |
| ★ **코어 직행** (armor 1, ×0.40) | 3000 ÷ 0.4 ÷ 45 = 167s | **167s** ✔ **성립 (여유 13s)** |
| **불 스탠스로 `plate`(물)** = ×0.5 | 6000÷45=133s + 67s | **200s** ✗ → **Q가 낫다** |

- ★ **`armorPartCountRange`의 하한 1이 왜 있는지가 이 보스 하나로 증명된다.** armor 1 = 게이트 ×0.4 = **"무시하고 직행"이 산술적으로 가능해지는 유일한 지점.** 6개 보스 중 3개(armor 1)가 이 자유를 갖고, 3개(armor 2)는 갖지 않는다 → **정본 §8.13의 "소프트 게이트"가 말뿐이 아니게 된다.**
- ★ **마지막 줄이 Q(노말 스탠스)의 존재 이유를 세 번째로 증명한다** (정본 §4.3 · §8.16에 이은). 불+4 특화가 `plate`(물)를 불로 때리면 200초 = 사망. **Q로 바꾸면 133초 = 클리어.** "모르겠으면 Q"가 안전 바닥이라는 것이 여기서 **13초짜리 실물 마진**이 된다.
- `turretL`(불)은 **테마 속성 부위** = 화산을 물+4로 밀고 온 빌드가 유일하게 ×2로 녹이는 부위. 깨면 **용암 장판이 사라진다** → 화산 테마의 정체성("공간 제한")이 플레이어의 손으로 꺼진다 = `armament`의 페이오프가 가장 시각적인 순간.

### 10.4 나머지 4종

| 보스 | 부위 | `partType`/`element` | 정답 | `hp` | `radius` | `anchor` | `shapeId` | 시드 이미터 | 파괴 효과 |
|---|---|---|---|---|---|---|---|---|---|
| **`frostCrown`** 빙관 (`glacier`/물) `movePattern:"sway"` | `pylon` 서리탑 | `armament`/불 | 물(E) | 800 | 22 | `[0,-38]` | `cross` | `wall`/`frostBrick`/`count:9, gapCount:1, gapWidthPx:84, speed:105` ★ **페이즈 3 = `stunMark`** | 얼음 벽 영구 삭제 |
| | `crownL` 좌관 | `armor`/풀 | **불(W)** | 950 | 26 | `[-56,16]` | `spike` | `straight`/`pelletS`/`count:4, spreadDeg:34, speed:150` | 게이트 1/2 |
| | `crownR` 우관 | `armor`/불 | **물(E)** | 950 | 26 | `[56,16]` | `spike` | `spiral`/`pelletS`/`count:8, speed:140, rotStepDeg:24, durationSec:2.0, rateSec:0.25` | 게이트 2/2 |
| | `core` | `core`/노말 | 아무거나 | **3800** | 40 | `[0,0]` | `orb` | — | **사망** |
| **`scarab`** 태양갑충 (`desert`/불) `movePattern:"orbitArc"` | `carapace` 갑각 | `armor`/물 | **풀(R)** | **2800** | 30 | `[0,26]` | `slab` | `fan`/`fanShard`/`count:7, arcDeg:96, speed:130` | 게이트 1/1 |
| | `legs` 다리마디 | `mobility`/**불** | 물(E) | 750 | 20 | `[-48,40]` | `claw` | `straight`/`pelletS`/`count:3, spreadDeg:20, speed:150` | 이동 ×0.5, 호 기동 중단 |
| | `stinger` 침포 | `armament`/풀 | 불(W) | 750 | 20 | `[48,40]` | `spike` | `aimed`/`homingM`/`count:2, spreadDeg:20, speed:90, leadSec:0.35` | 유도탄 영구 삭제 |
| | `core` | `core`/노말 | 아무거나 | **3100** | 38 | `[0,0]` | `hexPod` | — | **사망** |
| **`thornKing`** 가시왕 (`forest`/풀) `movePattern:"sway"` | `podL` 좌 꼬투리 | `armor`/물 | **풀(R)** | 900 | 25 | `[-54,12]` | `orb` | `ring`/`fanShard`/`count:9, speed:130, rotOffsetDeg:20` | 게이트 1/2 |
| | `podR` 우 꼬투리 | `armor`/불 | **물(E)** | 900 | 25 | `[54,12]` | `orb` | `straight`/`pelletS`/`count:5, spreadDeg:40, speed:150` | 게이트 2/2 |
| | `bloom` 화포 | `armament`/**풀** | 불(W) | 800 | 26 | `[0,-34]` | `ring` | `wall`/`thornBrick`/`count:8, gapCount:1, gapWidthPx:92, speed:120` | **가시 벽 영구 삭제** |
| | `core` | `core`/노말 | 아무거나 | **3900** | 40 | `[0,0]` | `bulb` | — | **사망** |
| **`mire`** 늪주 (`bog`/풀) `movePattern:"holdCenter"` | `shell` 이끼갑 | `armor`/불 | **물(E)** | **2900** | 30 | `[0,28]` | `hexPod` | `fan`/`fanShard`/`count:7, arcDeg:110, speed:130` | 게이트 1/1 |
| | `tendril` 촉수각 | `mobility`/물 | 풀(R) | 750 | 20 | `[-50,38]` | `claw` | `straight`/`pelletS`/`count:3, spreadDeg:24, speed:150` | 이동 ×0.5, 흔들림 중단 |
| | `sac` 포자낭 | `armament`/**풀** | 불(W) | 750 | 22 | `[50,38]` | `orb` | `spiral`/`hexBolt`/`count:10, speed:95, rotStepDeg:26, durationSec:2.0, rateSec:0.2` ★ **페이즈 3 = `stunMark`** | ★ **둔화·스턴 이미터 영구 삭제** |
| | `core` | `core`/노말 | 아무거나 | **3100** | 38 | `[0,0]` | `bulb` | — | **사망** |

**★ 스턴의 유일한 거처 (정본 §2.7 · §12.4 준수)**

| 규칙 | 이 섹션의 준수 |
|---|---|
| `difficulty.stunMinDifficulty = "hard"` | `frostCrown.pylon` · `mire.sac`의 **페이즈 3 이미터만** `bulletId: "stunMark"`. Normal에서는 생성 안 됨 |
| `stage.statusStunMaxPerStage = 2` | **스테이지당 1기** (보스 부위 1개). 잡몹·엘리트·중간보스·새떼는 **스턴을 절대 갖지 않는다** ✔ 여유 100% |
| `fairness.minStunTelegraphSec = 1.5` | 보스 부위 `telegraphSec = 1.50` 고정(§9.3) → **자동 충족** ✔ |
| `fairness.maxStunSec = 1.0` | `stunMark.statusDurationSec = 1.0` ✔ |
| 텔레그래프 = 화면 가장자리 호박 1회 사전 펄스 (정본 §7.4·§7.11) | 1.50초 리드 + 전신 광휘 + 패턴 점선 프리뷰와 **동시** |

- ★ **스턴을 `glacier`·`bog`(= 둔화 정체성 테마 2종)의 보스에만 둔 이유**: 상태이상이 **테마 → 잡몹 → 보스**로 한 줄로 이어져야 학습이 이월된다. 늪에서 25분간 `hexBolt`(둔화)를 맞아 온 플레이어가 보스 페이즈 3에서 **같은 육각탄인데 흰 이중 링 + 가장자리 펄스**를 보는 것 — 그게 스턴이다. 새 채널이 아니라 **아는 채널의 강조**(정본 §7.11의 3층 구조 그대로).
- ★ **`sac`(포자낭) 파괴 = 둔화와 스턴이 동시에 사라진다.** `armament`의 정의("그 부위가 담당하던 이미터가 영구 삭제")가 **런에서 가장 큰 페이오프**가 되는 지점 — 늪주는 "상태이상을 끄고 싸울 것인가, 시간을 아낄 것인가"가 곧 보스전의 결정이다. 그리고 `sac`는 **풀(테마 속성)**이라 늪을 불+4로 밀고 온 빌드가 ×2로 가장 빨리 끌 수 있다 = **테마 특화가 보상받는 유일한 부위**(R4가 1개만 허용하므로).
- **Normal에서 페이즈 3의 그 부위 하나가 침묵한다.** 이것은 버그가 아니라 **Normal의 관대함이 최종 국면에 나타나는 형태**다(정본 §6.2: 무-트위치 기둥은 Normal에서 보장된다). → 다만 구현 규칙 명시가 필요하다: §14 **R-3**.

---

## 11. 스테이지 6 — 최종 (`stageId: "finale"`)

### 11.1 정본이 확정한 것 (인용)

`element: null`(**`null`은 `finale`에만**, S9) · `mix {water .30, fire .30, grass .30, normal .10}`(**같은 스키마**) · 고지 "테마 없음 — 4속성 혼재" · 로스터 **15종 총출동** · 중간보스 2회·3종 중 2종·**서로 다른 속성** · `finaleCrisisRotating: true`(물×2 → 불×2 → 풀×2) · 보스 「테트라크」 · `finale.partCount 5` / `armorPartCount 3` / `exemptRules ["R4"]` / `allowNormalPeripheral true` · **격파 = 런 클리어.** (정본 §8.16)

### 11.2 웨이브 리스트 — 최종 블록 슬롯 규격 (30/30/30/10)

`mix`가 다르므로 **슬롯 규격도 다르다.** 블록 = 4 슬롯, 단위 `u`:

| 슬롯 | `element` | `count` |
|---|---|---|
| s1 | `water` | `3u` |
| s2 | `fire` | `3u` |
| s3 | `grass` | `3u` |
| s4 | `normal` | `1u` |

블록 합 = `10u`, `3u:3u:3u:1u` = **30/30/30/10 정확** ✔ §5.2의 블록 불변식이 그대로 성립한다(같은 증명, 다른 상수 벡터).

**5 블록 × 4 웨이브 = 20 웨이브 / 70 개체** — `unlockStageMin`은 전부 충족되므로 티어 필터링이 없다(단일 티어).

| 블록 | `u` | s1 (`water`) | s2 (`fire`) | s3 (`grass`) | s4 (`normal`) |
|---|---|---|---|---|---|
| A | 2 | `vWedge:drifter`(6) | `columnV:columnAnt`(6) ★ | `arc:spitter`(6) | `scatter:hexer`(2) ★ |
| B | 2 | `pincer:flanker`(6) ★ | `scatter:dustRunner`(6) ★ | `pincer:thornWeaver`(6) ★ | `lineH:drifter`(2) |
| C | 1 | `scatter:sirenRay`(3) ★ | `lineH:magmaBomb`(3) ★ | `arc:bogHexer`(3) ★ | `scatter:frostLance`(1) |
| D | 1 | `lineH:frostLance`(3) ★ | `scatter:rearDart`(3) **`bottom`** | `scatter:stalker`(3) ★ | `lineH:turretPod`(1) |
| E | 1 | `scatter:turretPod`(3) ★ | `arc:magmaBomb`(3) ★ | `scatter:hexer`(3) ★ | `lineH:mortarHulk`(1) |

★ = `eliteIndex: 0`. `spawnEdge` 무표기 = `top`.

**S8 검산**: 70 개체 = water **21** / fire **21** / grass **21** / normal **7** → **30.0 / 30.0 / 30.0 / 10.0** — **오차 0.0%p** ✔
**로스터 검산**: `drifter` `columnAnt` `spitter` `hexer` `flanker` `dustRunner` `thornWeaver` `sirenRay` `magmaBomb` `bogHexer` `frostLance` `rearDart` `stalker` `turretPod` `mortarHulk` = **15종 전부 등장** ✔ (정본 §8.16 "지나온 5개 테마의 회고")
`mobPhaseMaxWaves 14 < 20` → **순환 없음**, 앞 14개만 소환 → S8은 **저작 리스트 기준**이므로 영향 없음(§5.2 주석).

> ★ **최종 스테이지의 잡몹 페이즈가 곧 5분짜리 졸업 문제다.** 어떤 스탠스를 켜도 화면의 70%가 그 스탠스에 ×1 또는 ×0.5다 — **처음으로 "정답 스탠스"가 존재하지 않는다.** 지금까지 5개 스테이지가 "70%를 ×2로 녹이는 법"을 가르쳤다면 최종은 **"내가 지금 무엇을 때리고 있는가"**를 묻는다. `2/2/2` 분산 빌드가 처음으로 `4/2/0` 특화를 이기는 구간이며, `elementCapTotal = 6`(정본 §4.2)이 만든 다이얼의 결산이다.

### 11.3 위기 세션 — 로테이션 (`finaleCrisisRotating: true`)

`crisisElementRule`의 **두 번째이자 마지막 예외** (정본 §8.16).

| 서브웨이브 | `formationId` | `swarmChaff` | `swarmLancer` | `element` | 요구 스탠스 |
|---|---|---|---|---|---|
| 1 | `arc` | 9 | 1 | **`water`** | **R (풀)** |
| 2 | `vWedge` | 9 | 1 | **`water`** | **R (풀)** |
| 3 | `arc` | 9 | 1 | **`fire`** | **E (물)** |
| 4 | `vWedge` | 9 | 1 | **`fire`** | **E (물)** |
| 5 | `arc` | 9 | 1 | **`grass`** | **W (불)** |
| 6 | `vWedge` | 9 | 1 | **`grass`** | **W (불)** |

- 서브웨이브 간격 4.17 게임초 → **전환 시점은 8.3초 / 16.7초.** 25초 동안 **`R → E → W` 2번의 전환**(정본 §8.16: "스탠스를 3번 갈아타는 졸업 시험").
- ★ **`stance.switchCooldown = 0`(정본 §4.3)의 존재 이유가 이 25초에 전부 결산된다.** 쿨다운이 1초라도 있었으면 이 장면은 성립하지 않는다. 그리고 **`2/2/2` 빌드는 3파 전부 ×2**, **`4/2/0` 빌드는 한 색 앞에서 ×1로 밀린다** → 정본 §4.2가 `elementCapTotal`을 8이 아니라 6으로 정한 근거가 **눈에 보이는 25초**가 된다.
- **`crisisTotal = 60 × swarmTotalScale[5](1.0) = 60`** → 20/20/20 = 정확히 3등분 ✔
- 로테이션이 `crisisClearWithoutCapstone ≥ 0.80`을 깨지 않는가: **폭탄은 속성이 없다**(`bomb.mobDmg 9999`) → 어느 파에서 터뜨려도 유효 ✔ 보편 해답은 최종에서도 그대로다.

### 11.4 「테트라크 (TETRARCH)」

| 부위 | `partType` | `element` | **정답 스탠스** | `hp` | `radius` | `anchor` | `shapeId` | `contactDmg` | `score` | 파괴 효과 |
|---|---|---|---|---|---|---|---|---|---|---|
| `coffinWater` 물의 관 | `armor` | 물 | **풀(R)** | 900 | 24 | `[-64,20]` | `slab` | 15 | 2000 | 게이트 1/3 |
| `coffinFire` 불의 관 | `armor` | 불 | **물(E)** | 900 | 24 | `[64,20]` | `slab` | 15 | 2000 | 게이트 2/3 |
| `coffinGrass` 풀의 관 | `armor` | 풀 | **불(W)** | 900 | 24 | `[0,52]` | `slab` | 15 | 2000 | 게이트 3/3 |
| `throne` 왕좌 | `armament` | **노말** | 없음 (항상 ×1) | 700 | 26 | `[0,-44]` | `ring` | 15 | 1800 | **유도탄 이미터 영구 삭제** |
| `core` 핵 | `core` | 노말 | 아무거나 | **4600** | 44 | `[0,0]` | `orb` | 16 | **15000** | **사망 = 런 클리어** |

`movePattern: "holdCenter"`, `movePatternParams: {"speedPxSec":0,"ampPx":60,"yHoldPx":160}`

**시드 이미터** (`P = 4` → `offsetSec = everySec × i / 4`, §9.3)

| 부위 | `type` | `bulletId` | 시드 `count` | 고유 |
|---|---|---|---|---|
| `coffinWater` | `ring` | `heavyRound` | 10 | `speed:115, rotOffsetDeg:18` |
| `coffinFire` | `zone` | `null` | — | `radius:72, activeSec:3.0, dmg:10` |
| `coffinGrass` | `wall` | `thornBrick` | 8 | `gapCount:1, gapWidthPx:92, speed:120` |
| `throne` | `aimed` | `homingM` | 3 | `spreadDeg:24, speed:90, leadSec:0.4` |

**산술 검증** (스테이지 6, `bossHpScale[5] = 11.5`, DPS 520 — `dpsProbe` 게이트 3종과 직접 대응)

| 빌드 | base 소요 피해 | ×11.5 실효 | 시간 | 게이트 (정본 §13.1) | 판정 |
|---|---|---|---|---|---|
| **균형 `2/2/2`** | 관 3×450 + 핵 4600 = **5950** | 68,425 | **132s** | `balancedPass ≥ 0.95` | ✔ 여유 48s |
| **특화 `4/2/0`(불+4)** | 풀관 450 + 물관 900(Q) + 불관 900(Q) + 핵 4600 = **6850** | 78,775 | **152s** | `specialistPass ≥ 0.85` | ✔ **여유 28s = 아슬** |
| **무투자** | 관 3×900 + 핵 4600 = **7300** | 83,950 | **161s** | `noElementPass ≥ 0.55` | ✔ **여유 19s = 브릭 아님** |
| **코어 직행** (armor 3, ×0.4³ = **×0.064**) | 4600 ÷ 0.064 = **71,875** | 826,563 | **1590s** | — | ✗ **완전 불가** |

- ★ **정본 §13.3이 요구한 세 지표가 하나의 HP 배분에서 동시에 성립한다**: 132 / 152 / 161 / **180**. 세 빌드가 **28초 · 19초**의 마진으로 차례로 늘어서고 직행만 두 자릿수 배로 튕겨 나간다 → **"특화 = 아슬아슬하게 성공, 분산 = 여유롭게 성공, 무투자 = 벽에 안 부딪힘, 직행 = 없음"**(정본 §8.16)이 **숫자의 결과**이지 선언이 아니다.
- ★ **무투자 161s가 `noElementPass ≥ 0.55`의 정확한 그림이다.** 19초 마진은 **회피 손실 하나면 날아가는 폭** → 시뮬에서 일부는 통과하고 일부는 실패한다 = 0.55~0.75 구간. 정본 §13.2-②가 "0.55는 런을 깬다가 아니라 벽에 안 부딪힌다"라고 한 것의 실물.
- **`throne`(노말)은 700 base = 15.5초**. 상성이 통하지 않으므로 **어떤 빌드도 같은 시간**이 든다 → 정본 §8.16의 "항상 ×1인 안전 바닥의 존재 증명 부위"가 **모든 빌드에게 동일한 가격표**로 실재한다. 유도탄이 괴로우면 15초를 낸다.
- **고유 연출 (정본 §8.16의 규칙 인용)**: 관이 하나 파괴될 때마다 아레나 배경이 그 속성의 색을 잃는다 → 3개 다 깨지면 **무채색 아레나에서 핵과 단둘이.** 배경 파라미터는 `skinId: "finale"`의 3단 상태이며 **게임플레이 규칙을 만들지 않는다**(정본 §7.9 상한 준수).

### 11.5 최종 중간보스

2회(`midBossAtSec [30, 70]`), `mb*` 3종 중 2종, **서로 다른 속성**(정본 §8.16).
- **테마가 없으므로 `midBossElementRule = "notThemeAndNotNormal"`의 "테마가 아님" 조건이 공허참** → 후보 = `{fire, water, grass}` 전부. `rng.spawn`이 **비복원 2개**를 뽑는다.
- 종류도 `rng.spawn`이 3종 중 비복원 2개 → **최종은 매번 다른 2개의 예고편**을 준다. `mbNest`(DPS 체크)가 뽑히면 테트라크의 4600 코어를 미리 재보는 셈이 된다.

---

## 12. AI 빌드타임 생성 제약 (이 섹션의 콘텐츠에 한함)

### 12.1 하드 제약 — 위반 = 생성물 폐기·재생성

> **G1~G10은 정본 §13.4(S3~S12)의 재확인이며 새 규칙이 아니다. G11~G17이 이 섹션이 추가하는 제약**이고, 전부 **`check.mjs`에 정적 검사로 넣을 수 있는 형태**로 적었다.

| # | 제약 | 근거 |
|---|---|---|
| **G1** | `moveId`(8)·`emitterType`(8)·`formationId`(6)·`partType`(4)·`shapeId`(12) **어휘 밖 = 실패.** AI는 새 거동·새 도형을 발명할 수 없다 | 정본 §0.1 C-3, S3 |
| **G2** | `(moveId, emitterType)` 중복 = 실패 (`band` 다르면 허용) | S4 · §2.4 |
| **G3** | 보스 **R1~R6 전부** + `partCount`(코어 포함 4 / finale 5) + `armor` 수 | S5 · §10.1 |
| **G4** | `telegraphSec ≥ max(이미터 타입 하한, 상태이상 하한, 개체 등급 하한)` — ★ **하한이 겹치면 큰 쪽이 이긴다** | 정본 §7.4 · §3.3 |
| **G5** | 보스 부위: `everySec ∈ [6.0, 4.5, 3.6][phase]` · `offsetSec = everySec × i / P` · `telegraphSec = 1.50` · 부등식 **`P × telegraphSec ≤ 2 × everySec`** | §9.3 (S7의 정적 형태) |
| **G6** | 탄 `speed ≤ 260`(조준탄 ≤ 200) · `radius ≥ 4` · `gapWidthPx ≥ 46` | 정본 §12.4 · §3.3 |
| **G7** | ★ **블록 불변식** — 웨이브 리스트는 블록으로만 구성하고, **모든 블록이 각각 정확히 `mix`**여야 한다. 슬롯 규격(§5.3 / §11.2)에서 벗어난 `count` = 실패 | S8을 오차 0으로 만드는 장치 |
| **G8** | `crisisElementRule` = 테마 100% (최종만 로테이션). 새떼에 `swarm*` 외 아키타입 금지 | S9 · §8 |
| **G9** | `rearIn` / `spawnEdge: "bottom"` 은 `rearSpawnAllowed[stage]`일 때만 | S9 · §4 |
| **G10** | 난수는 명명된 **7 스트림**(`theme` `draft` `spawn` `elite` `drop` `pattern` `boss`)만. 스트림 간 공유 금지 | 정본 §10.2 · S11 |
| **G11** ★ | `orbitDrift`의 `keepDistPx ≥ fairness.minSpawnRadiusPx`(**140**) | §3.3 — 미만이면 그 적은 **탄을 못 쏴 무해해진다**(점블랭크 금지에 걸림) = 조용한 콘텐츠 버그 |
| **G12** ★ | `eliteIndex ≠ null`은 **`band ∈ {line,turret,bruiser}` 이고 `element ≠ normal`인 웨이브에만** | §5.4 — 런타임 거부가 아니라 저작 시점에 `elite.bandAllowed`/`elementAllowed`(정본 §8.6)를 만족 |
| **G13** ★ | `count == 1`에 `pincer`·`columnV` 저작 금지 | 1기로는 "좌·우 동시 진입"·"종대"가 성립하지 않는다 |
| **G14** ★ | 새떼 `spawnEdge` = **`top` 고정** (`rearSpawnAllowed`가 참인 스테이지에서도) | 25초 최고밀도에 후방 진입을 겹치면 관대함이 붕괴 |
| **G15** ★ | 페이즈 파생은 **`count`·`arcDeg`·`rotStepDeg`·`everySec`만** 스케일한다. **`speed`·`type`·`bulletId` 스케일 금지** | §9.4 — 페이즈 스케일이 `maxBulletSpeed`(260)를 몰래 넘는 사고 차단 |
| **G16** ★ | `shapeId` 재사용은 허용, **신규 발명은 금지**. 같은 `shapeId`를 쓰는 두 아키타입은 `radius`가 **1.5배 이상** 차이나거나 `band`가 달라야 한다 | §2.5 — 실루엣 계열 학습을 이월시키되 혼동은 막는다 |
| **G17** ★ | `bullets[].status == "stun"`을 참조하는 이미터는 **`tier:"stage"` 보스 부위의 `patternSet[2]`(페이즈 3)에만**, 그리고 **스테이지당 최대 1 부위** | 정본 §2.7의 `statusStunMaxPerStage`(2)를 **절반의 여유로** 만족 + `minStunTelegraphSec`(1.5)를 §9.3의 `telegraphSec = 1.50`이 자동 충족 |

### 12.2 ★ 동시 텔레그래프 예산 — `caps.telegraphs = 8`은 이 콘텐츠를 통과시키지 못한다

**계산**: 이 섹션의 이미터 13종 duty(`telegraphSec ÷ everySec`) = 10.0% ~ 36.4%, **가중 평균 ≈ 15%**.

```
잡몹 페이즈 최악 (스테이지 5~6, 동시 개체 ≈ enemyConcurrentMax 40, 사격 개체 ≈ 30):
  기대 동시 텔레그래프 = 30 × 0.15 = 4.5
  포아송 근사 P(동시 > 8) ≈ 3.5%
  A층 이론 상한 = enemyConcurrentMax(40) × telegraphConcurrentMaxPerEntity(2) = 80
```

- `certify.static.capHits.max = 0`(정본 §13.1)은 **한 번이라도 닿으면 실패**다. 8000런 × 5스테이지 × 95게임초 × 60틱 ≈ **2×10⁸ 틱 표본**에서 3.5%는 **결정적으로 발생**한다 → `deferAttack`이 상시 작동 → **`--certify` 영구 실패.**
- 8을 지키려면 개체당 duty ≤ 0.5% → `everySec ≥ 110 게임초` → **적이 스테이지당 1발 쏜다.** 콘텐츠가 성립하지 않는다.
- ★ **원인 진단**: 정본 §12.1의 2층 검증표에서 `enemies`(A 40 < B 96)와 `enemyBullets`(A 320 < B 384)는 **전역 대 전역**을 비교하는데, `telegraphs` 행만 **"A 2/개체 < B 8"** 로 **개체당 예산과 전역 풀을 비교**한다. 그리고 그 행의 괄호 주석 —**"잡몹 다수가 각자 1~2개 → 8이 안전망"** — 은 스스로 결론을 반박한다(다수가 각자 1~2개면 합은 8을 훨씬 넘는다). **단위가 다른 두 수를 부등호로 이은 것**이며, 정본이 초안 F의 `caps.enemyBullets 640` vs `fairness.maxSimultaneousEnemyBullets 320` 자기 충돌을 잡아낸 것과 **같은 종류의 결함**이다.
- **→ §14 R-4 (blocker)**: A층 `fairness.telegraphConcurrentMaxGlobal = 48` 신설 + B층 `caps.telegraphs = 8 → 96`. 2층 규율(A < B) 유지 ✔
- **이 섹션의 실제 최악값은 ≈ 13**(3σ) → A층 48 대비 **여유 73%**. 보스전 ≤ 2(§9.3), 위기 세션 ≈ 0.6(§8.3). **어디서도 예산에 닿지 않는다** → `capHits = 0` ✔

---

## 13. 검증 게이트 대조 — 이 섹션의 콘텐츠가 실제로 통과하는가

| 게이트 (정본 §13.1) | 이 섹션의 대응 | 판정 |
|---|---|---|
| `dpsProbe.balancedPass ≥ 0.95` | 테트라크 132s / 강철 가오리 109s (제한 180) | ✔ §10.2 · §11.4 |
| `dpsProbe.specialistPass ≥ 0.85` | 테트라크 **152s** (여유 28s) / 가오리 119s | ✔ **아슬** = 의도 |
| `dpsProbe.noElementPass ≥ 0.55` | 테트라크 **161s** (여유 19s) — 회피 손실 하나면 날아가는 폭 | ✔ 0.55~0.75 예상 |
| `bossTimeoutRate ≤ 0.25` | 7 보스 전부 목표 격파 110~135s (§9.6) → 타임아웃은 언더레벨 런에 집중 | ✔ |
| `dominance.maxArchetypeLethalityShare ≤ 0.25` | 17 아키타입, 최대 위협 = `turretPod`/`mortarHulk`(dmg 22/10) 이나 **테마당 최대 2 로스터**에만 등장 | ✔ |
| `dominance.maxThemeClearStddev ≤ 0.06` | armor 2 보스 3 / armor 1 보스 3이 **속성당 정확히 1:1**로 갈림 (§10.1) | ✔ 구조적 |
| `crisisClearWithoutCapstone ≥ 0.80` | `swarmChaff.hp = 3`(chaff) → 스테이지 6에서도 개체 HP **18** = 1~2히트 + **폭탄 보편 해답** | ✔ §8.4 |
| `farmXpRatio ≥ 2.0` | `dustRunner`(xp 15, speed 230, `line` 밴드) = **이탈 = 영구 0** / `mbNest` 무시 = 0 / 새떼 120 | ✔ §13.1 아래 |
| `coinScarcity` | 코인원 = `turret`·`bruiser`(15%) + 엘리트(3) + 중간보스(5) + 보스(12 + 부위당 2) | ⚠ **테마 간 3배 분산** → §6 조정 손잡이 |
| `capHits == 0` | 적 ≤ 40 / 탄 ≪ 320 / **텔레그래프 ≈ 13** | ✔ **단 R-4 승인 전제** |
| `fairnessViolations == 0` | §3.3 전수표 · §9.3 부등식 | ✔ |
| S8 (±3%p) | **전 테마 · 전 스테이지 오차 0.0%p** | ✔ §5.5 · §11.2 |

**`farmXpRatio ≥ 2.0`의 이 섹션 기여** (정본 §13.2-⑤: "비율을 만드는 것은 `enemyExitForfeitsReward`이지 `waveClearAdvance`가 아니다")

| 장치 | 이 섹션의 실물 | 소극 파밍 | 최대 파밍 |
|---|---|---|---|
| 이탈 = XP 영구 소멸 | `desert` T1 s1 = `scatter:dustRunner`(8기, **speed 230, 아레나 3.3초 종단**) | ~0 (요격 불가 위치) | ~0.95 × 8 × **xp 15** |
| 중간보스 무시 | `midBossLeaveAfterSec 30` | 0 | `xp 75` × 1~2 |
| 새떼 | 60기 × xp 2 | ~0.20 | ~0.90 |
| 하단 고정 시 사거리 밖 이탈 | `dive` 계열(`drifter` speed 70)은 하단까지 오지만 `anchor`(`turretPod` `yHoldPx 150`)·`strafe`(`flanker` `yPx 180`)는 **상단에서만 산다** | 0 | 전량 |

★ **`anchor`·`strafe` 아키타입의 `yHoldPx`/`yPx`(120~180)가 `farmXpRatio`의 숨은 주역이다.** 아레나 h=720에서 이들은 **위쪽 1/4에서만 존재**하고 하단으로 내려오지 않는다 → **하단에 붙어 있는 소극적 플레이어는 이들의 XP를 구조적으로 0으로 만든다.** 잡으려면 위로 올라가야 하고, 위 = 반응 여유 최소 = **위치 리스크**(정본 §8.8). 트위치가 아니라 "얼마나 앞에 설 것인가"가 XP 배율이 된다.

---

## 14. 정본 추가 요청

> **정본에 없어서 이 섹션이 새로 정한 것 전부.** 각 항목은 정본과 **모순되지 않으며**, 대부분 **정본 내부의 두 절이 어긋난 지점**을 닫는다. R-1 · R-2 · R-3 · R-4는 **blocker** — 반영되지 않으면 이 섹션의 콘텐츠가 로드 실패하거나 `--certify`가 영구 실패한다.

| # | 등급 | 항목 | 요청 | 근거 (정본 내부 어긋남) |
|---|---|---|---|---|
| **R-1** | ★**blocker** | **보스 HP의 스테이지 스케일이 없다** | `stages.curve.bossHpScale = [1.0, 1.9, 3.4, 5.6, 8.4, 11.5]` 신설. `실효 HP = bosses[].{core,parts[]}.hp × bossHpScale[stage]`, 저작값 = **스테이지 1 기준 base** | §8.1이 테마를 셔플하므로 **`manta`는 스테이지 1에도 5에도 온다.** §8.3은 *적* HP만, §8.9는 *중간보스*에만 `enemyHpScale`을 걸었고 §8.11은 "절대값"이라고만 했다 → **같은 절대값으로 5배 차이 나는 화력을 상대**하게 된다. §9.9가 `stage1BossHpScale`을 폐기하며 남긴 근거("스테이지 1 보스는 HP를 낮게 저작하면 그만")와 **모순 대장 #25**는 **"스테이지 1 보스가 고정"이라는 거짓 전제**에 서 있다. `enemyHpScale`(6배)과 별도 곡선인 이유 = 잡몹은 "chaff 1~2히트"를 지켜야 하고 보스는 **화력 성장(≈11.5배)을 따라가야** 한다 |
| **R-2** | ★**blocker** | **`tetrarch`가 R6를 위반한다** | `boss.finale.exemptRules`를 `["R4"]` → **`["R4","R6"]`**. (또는 S5에 "`tier == "final"`이면 `finale.armorPartCount`가 `armorPartCountRange`를 오버라이드한다"를 명시) | §9.4가 `armorPartCountRange: [1,2]`와 `finale.armorPartCount: 3`을 **둘 다** 선언했고, S5는 "R1~R6 전부 + ... + **armor 수 ∈ [1,2]**"라고 적었다. `exemptRules`에는 `R4`만 있다 → **문면대로면 `check.mjs`가 최종 보스를 로드 실패시킨다.** §8.15가 초안 F의 가오리에서 잡아낸 것과 같은 종류의 결함 |
| **R-3** | ★**blocker** | **`stunMinDifficulty` 미달 시의 동작이 없다** | "**해당 이미터는 발사를 스킵한다** (탄 치환 없음)"를 §2.7에 명시 | §2.7은 "Normal에서 스턴 탄 **자체가 생성되지 않는다**"까지만 적었다. 그 이미터가 ① 침묵하는지 ② 다른 탄으로 치환되는지에 따라 **콘텐츠 저작이 완전히 달라진다.** 이 섹션은 ①을 전제로 스턴을 **보스 부위의 페이즈 3에만** 두어(§12.1 G-17), Normal의 침묵이 **"최종 페이즈가 조금 조용하다" = 의도된 관대함**이 되게 설계했다 |
| **R-4** | ★**blocker** | **`caps.telegraphs = 8`이 구조적으로 도달 불가** | A층 **`fairness.telegraphConcurrentMaxGlobal = 48`** 신설 + B층 **`caps.telegraphs` 8 → 96**. §12.1 검증표의 `telegraphs` 행을 `A 48 < B 96`으로 정정 | §12.1의 다른 두 행은 **전역 대 전역**인데 `telegraphs`만 **"A 2/개체 < B 8"** 로 단위가 다른 두 수를 부등호로 이었다. 괄호 주석("잡몹 다수가 각자 1~2개")이 스스로 결론을 반박한다. A층 이론 상한 = 40 × 2 = **80 > 8**. `capHits.max = 0`이 **영구 실패**한다 (계산 = §12.2) |
| **R-5** | major | **`zone` 이미터의 `bulletId`** | `zone`은 `bulletId: null`을 **허용**한다고 §8.5·§9.7에 명시 | §8.5의 공통 파라미터에 `bulletId`가 있으나 `zone`은 `dmg`를 **직접** 갖는다(§3.2의 피해원 목록 = `bullets[].dmg | contactDmg | zone dmg`). "누락 키 = 에러"(§9.3)이므로 `null` 허용이 명시돼야 한다 |
| **R-6** | major | **`unlockStageMin`의 거처가 두 곳** | **`stages.themes[].roster[]`가 소유**로 확정하고 `enemies[].unlockStageMin`은 삭제 | §9.7의 아키타입 예시(`stalker`)에도, §9.9의 `roster[{archetypeId, unlockStageMin}]`에도 있다. **테마마다 다른 해금이 필요**하므로(같은 `columnAnt`가 `glacier`에선 3, `desert`에선 2) 로스터 엔트리가 옳다. "미지 키 = 에러"이므로 한 곳이어야 한다 |
| **R-7** | major | **웨이브 리스트 소진 시 동작** | **순환 재시작**(리스트 처음으로) 을 §8.7에 명시 | `unlockStageMin` 필터링 후 스테이지 1의 리스트는 5 웨이브인데 `mobPhaseMaxWaves`는 14다 → **소진 시 동작이 없으면 잡몹 페이즈 절반이 빈다.** 순환해도 S8은 **저작 리스트 기준 정적 검사**이므로 영향 없다(순환 시 실제 조우 편차 ≤ 4%p, §5.5) |
| **R-8** | major | **중간보스의 `patternSet`이 2 이미터를 표현 못 한다** | `patternSet[i]`를 **`{emitterIds: [...]}`**(배열, 길이 1~2)로 일반화. 보스 부위 = 길이 1, 중간보스 = 1~2 | §8.9는 `mbHammer`를 "**`fan` + `zone` 교대**"로 확정했으나, §9.8은 중간보스 `patternSet`을 **1개**로 못박고 엔트리를 `{emitterId}` **단수**로 정의했다 → **정본이 확정한 기능을 정본의 스키마가 표현하지 못한다.** `telegraphConcurrentMaxPerEntity`가 그대로 상한이므로 안전(§7.2에서 `offsetSec` 3.0 교대로 동시 1개 증명) |
| **R-9** | major | **`mbNest`의 소환을 표현할 필드가 없다** | **`bosses[].summon: {archetypeId, count, everySec, formationId}`** 신설. `tier:"mid"` + `midBossSummonsAllowed` 통과 시에만 허용, 그 외 = 에러 | §8.9가 `mbNest` = "`aimed` + **잡몹 소환**" + `midBossSummonsAllowed: "mbNest"만`으로 **기능을 확정**했으나 §9.8 스키마에 필드가 없고, `onDestroy.spawnWave`는 §8.12에서 폐기됐다. `boss.summonsAllowed: false`(§8.11)는 **스테이지 보스** 규칙이므로 충돌하지 않는다(§7.4) |
| **R-10** | minor | **중간보스 보상 필드의 거처** | `xp`·`coin`·`healDropChance`·`score`를 **`bosses[]` 개체 필드**로 확정 | §8.9는 값을 확정했으나(`coin 5` / `healDropChance 0.35`) `rules.json`에 `midBoss` 블록이 없고 `stages.json > phase`에도 없다 |
| **R-11** | major | **`midBossElementRule`의 적용 시점** | **런타임 주입** 확정: 스테이지 테마를 보고 `rng.spawn`이 `{counter, prey}` 중 택1 → `bosses[].element`는 중간보스에서 **`null` 저작**(주입 대상 표식) | §9.8은 "중간보스는 `element` 필드 보유"라고 적었으나, **고정 저작값은 `notThemeAndNotNormal`을 모든 테마에서 만족시킬 수 없다**(같은 `mbHammer`가 화산에선 물/풀, 숲에선 물/불이어야 한다). 잡몹의 `element`가 웨이브 편성이 주입하는 것(§9.7)과 같은 모델 |
| **R-12** | major | **보스 이동 어휘 `movePattern`이 정의되지 않았다** | 어휘 **3종 폐쇄**: `sway`(좌우 왕복) · `orbitArc`(호 기동) · `holdCenter`(중앙 고정 + 소폭). `movePatternParams: {speedPxSec, ampPx, yHoldPx}`. **`mobility` 파괴 = `speedPxSec × mobilityPenalty(0.5)` + `ampPx → 0`**(= §8.12의 "회피 기동 패턴 중단"의 정의) | §9.8이 `"movePattern": "mantaSway"`를 예시로 썼으나 **어휘도 파라미터도 어디에도 없다**(§8.4의 `moveId` 8종은 잡몹·중간보스용). 그리고 §8.12의 `mobility` 효과 "회피 기동 패턴 중단"이 **무엇을 중단하는지** 정의되지 않아 파괴 효과가 평가 불가다. S3의 동결 어휘 목록에 `movePattern`(3)을 추가 |
| **R-13** | minor | **파괴된 부위가 사격을 멈추는가** | "**부위 파괴 = 그 부위의 이미터 정지**"를 §8.12에 명시. `armament`의 "영구 삭제" 문구는 **그 부위의 유일한 효과가 이미터 상실**이라는 뜻 | §8.12는 `armament`에만 이미터 삭제를 적었다 → `armor`·`mobility` 파괴 후 사격 여부가 미정. 정지가 옳다: §7.6이 파괴된 부위를 "외곽선 소멸 + `#3A3A3A` + 글리프 제거"로 규정했으므로 **죽은 것이 쏘면 I-2 위반** |
| **R-14** | minor | **`stages.formations`의 필드가 `"..."`** | §5.7의 6종 파라미터 확정. **`pincer`만 `strafe.yPx`를 덮어쓴다**(`yStartPx + floor(i/2) × yStepPx`) | §9.9가 자리만 잡아 두었다. `pincer` 예외가 없으면 `flanker.yPx`가 고정이라 **모든 측면기가 한 줄로만 지나간다** = `strafe` 거동이 죽는다 |
| **R-15** | minor | **위기 서브웨이브 간격 키가 없다** | **파생값으로 확정**: `crisisDurationSec / crisisSubWaves` = 25 / 6 ≈ 4.17 게임초. 새 키 없음 | §8.10이 `crisisDurationSec`·`crisisSubWaves`만 정하고 간격을 정하지 않았다. 파생이면 새 키 0 |
| **R-16** | minor | **"보스 6종" 표기** | §9.10·§8.14의 "6개 보스"를 **"테마 6 + 최종 1 = 7종 저작 / 런당 등장 6종"**으로 정정 | 테마 6종이 각각 `bossId`를 갖고(§8.1·§9.9) `finale`도 갖는다 → 저작 수는 7. "6"은 런당 등장 수(5 추첨 + 최종) |
| **R-17** | minor | **`bands[].sizePx`와 `archetypes[].radius`의 관계** | **`radius` = 히트박스이자 실제 크기(단일 진실)**, `bands[].sizePx` = **저작 가이드(밴드 표준 지름)이며 코드 키가 아님**을 명시. `elite.sizeMult`는 `radius`에 곱한다 | §9.7의 `bands`에 `sizePx`가, §9.10에 "`sizePx`는 `band`가 결정"이 있는데 §2.3은 "적 기체 히트박스 = `enemies[].radius`"다 → **크기 소스가 둘**이면 그려진 것과 맞는 것이 어긋나 I-2의 정신(§2.3의 "크기도 거짓말하지 않는다")에 반한다 |

### 14.1 이 섹션이 정본을 바꾸지 **않은** 것 (확인)

- **테마 커버리지 보장** — 추첨 구조가 이미 증명한다(§1.1). 보정 장치를 만들지 않았다.
- **엘리트 정의** — 정본 §8.6을 전면 인용했고 값을 하나도 정하지 않았다. 이 섹션이 더한 것은 **저작 규칙 G-12 하나**(런타임 거부를 저작 시점으로 이동).
- **중간보스 3종·이탈 30초·속성 규칙·보상** — 정본 §8.9 그대로. 이 섹션은 `moveParams`·이미터·HP만 채웠다.
- **위기 세션의 모든 규칙** — 정본 §8.10 그대로. 이 섹션은 6 서브웨이브의 9:1 분산만 정했다.
- **코어 게이트·R1~R6·`partType` 4종·`phaseThresholds`·부위 XP 0** — 정본 §8.11~§8.14 그대로.
- **`manta`의 코어·스러스터 수치, `stalker`의 전 필드, `homing2`의 전 파라미터** — 정본 §9.7·§9.8이 직접 저작한 값을 **한 글자도 바꾸지 않았다.**
- **`charge`를 잡몹에 주지 않은 것** — 정본 §8.9가 이미 `mbLancer`에 배정했다. 어휘 커버리지를 위해 억지로 잡몹에 넣으면 §8.4의 무기 짝 논거를 밟게 된다(§2.1).

