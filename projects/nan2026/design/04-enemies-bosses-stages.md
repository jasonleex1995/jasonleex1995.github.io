# 섹션 — 적 · 중간보스 · 위기세션 · 보스 · 스테이지

> **버전: v1.2 정합판.** 정본 `CANON.md` **v1.2**를 인용해 개정했다. **v1.2가 통보한 10건(§20.3-04)은 전부 이행**되었고, 그 귀결로 **이 섹션의 산술을 전수 재계산**했다 — 보스 HP ×1.20 · 중간보스 base ×0.80 · `bossHpScale` 교체 · `dpsRef` 재정의(**maxFarm**) · 새떼 지분 분모 정정.
>
> **지위**: 이 문서는 `design/CANON.md`(정본)의 **하위 문서**다. 정본 §17이 이 섹션에 위임한 범위 = **17 아키타입의 실제 파라미터 · 테마별 로스터 편성 · 웨이브 리스트 저작 · 이미터 인스턴스 · 새떼 서브웨이브 편성** 그리고 **7 보스 + 3 중간보스의 부위 배치·`radius`·`anchor`·`shapeId`·`score`·`patternSet`·시드 이미터 (R1~R7 안에서)**.
>
> ★ **정본이 소유하며 이 섹션이 정하지 않는 것**: **`dpsRef` 곡선과 그 farm 정책**(§13.5 · §13.5.1) · **`runFarmDpsRatio`**(§13.5.1) · **보스 HP 전량**(§13.6) · **`bossHpScale`**(§13.6.1) · **중간보스 base HP**(§13.6.4) · **`armorCoreRatio` φ**(§8.13.1) · ★ **`boss.optionalPartArmorRatio`**(§13.6.4 — v1.2 개명) · **`uptimeRef`**(§10.4.3) · ★ **`bands.chaff.xpRef`**(§8.10 — v1.2 신설, 이 섹션의 `chaffXpRef`가 흡수됐다) · **`stages.formations` 파라미터**(§9.9.2) · **`killTimeMedianBalanced` 밴드**(§13.6.5) · **전 `certify` 지표의 측정 정의**(§13.1.1).
>
> **정본 재정의 금지 (C-1)**. 이 문서에 등장하는 정본 값은 전부 **인용**이며 출처 절 번호를 병기한다. 이 문서가 정본과 충돌하면 **이 문서가 틀린 것**이다.
> **인쇄된 스키마 예시는 전부 예시다 (C-7)**. 이전 판본이 정본 §9.8의 `core.hp: 4000`을 "정본이 직접 저작한 값"이라 인용한 것은 **무효**이며, 그 자리는 §13.6.2의 표가 소유한다.
>
> **여기서 저작하는 값의 거처 (C-2)** = `data/enemies.json` · `data/bullets.json` · `data/bosses.json` · `data/stages.json`. 아래 표는 그 JSON의 **저작 원본**이며, 확정 후 `.js`를 열지 않고 이 값들만 시뮬이 조정한다 (C-4).
>
> **남은 「정본 추가 요청」은 §14.3에 3건**이며 **blocker 1 · minor 2**다. v1.1의 요청 3건(N-1·N-2·N-3)은 **정본 v1.2가 전건 채택**해 소멸했다(§14.2). 새 blocker 1건은 **정본 v1.2가 신설한 `crisisWaveResidualMax = 10`이 이 섹션의 최대 웨이브 레코드(16)보다 작다**는 것이며, 이 섹션이 정본의 지시대로 **실측을 실제로 계산해서** 발견했다(§8.3).

---

## 0. 이 섹션이 지켜야 하는 정본 게이트 (작업 체크리스트)

| 게이트 | 출처 | 이 문서에서 통과를 증명한 곳 |
|---|---|---|
| **S3** 어휘 폐쇄 (`moveId` 8 · `emitterType` 8 · `formationId` 6 · `partType` 4 · `shapeId` 12 · ★ `movePattern` 3 · ★ `bullets[].shape` 2) | 정본 §13.4 | §2.1 · §2.2 · §2.5 · §3.1 · §9 |
| **S4** 아키타입 겹침 금지 — `(moveId, emitterType)` 중복 실패, `band` 다르면 허용 | 정본 §13.4 | **§2.4 전수 증명표** |
| **S5** 보스 **R1~R7** 전부 + `partCount` + ★ **`armor` 수 == 2**(finale 3) + `tier=="final"` ⟺ `bossHpScale` 미적용 | 정본 §8.14 · §13.4 | **§10.1 전수 검증표** |
| **S6** 공정성 — 텔레그래프 하한·탄속·틈·`minSpawnRadiusPx`·스턴. ★ **`enemies.json > emitters`만** 검사 | 정본 §7.4 · §12.4 | **§3.3 전수 검증표** |
| **S7** 개체당 동시 텔레그래프 ≤ 2 (3페이즈 전개) | 정본 §12.4 · §13.4 | **§9.3 라운드로빈 정적 증명** |
| **S8** 혼합 비율 ±3%p — ★ **저작 리스트 기준**(실제 스폰 아님) | 정본 §8.2.1 · §13.4 | **§5.2 블록 불변식 증명** |
| **S9** `element` null은 finale만 · `rearIn`은 `rearSpawnAllowed`만 · 새떼 아키타입 · `crisisElementRule` | 정본 §13.4 | §1.1 · §2.3 · §8 · §11 |
| ★ **S13** 스턴의 거처 = 보스 부위 `patternSet[2]`만, 스테이지당 ≤ 2 개체 | 정본 §13.4 | **§10.5** |
| ★ **S14** `shape ↔ status` 동치 | 정본 §13.4 | **§3.1** |
| ★ **S15** `tier=="mid"` ⟺ `element == null` | 정본 §13.4 | **§7.2** |
| ★ **S16** `patternSet[i].emitterIds` 길이 — 부위 1 / 중간보스 1~2 | 정본 §13.4 | **§7.2 · §9.4** |
| ★ **S17** `summon != null` ⟺ (`tier=="mid"` ∧ `midBossSummonsAllowed`) | 정본 §13.4 | **§7.2** |
| ★ **S18** `movePattern == "holdCenter"` 보스는 `mobility` 부위 금지 | 정본 §8.12.1 | **§10.1** |
| ★ **S19** `emitterType == "zone"` ⟺ `bulletId == null` | 정본 §13.4 | **§3.2** |
| ★ **S20** `pincer` ⟺ `strafe` / `columnV` ⟺ `column` | 정본 §13.4 | **§5.6** |
| ★ **S22** 새떼 XP ≤ 0.30 | 정본 §13.4 | **§8.4 전 테마 검산** |
| ★ **S23** 테마 로스터의 `turret`+`bruiser` = 1~2종 | 정본 §13.4 | **§4 전수표** |
| ★ **S24** `Σ(armor hp) == core.hp × armorCoreRatio` (±1%) · 선택 부위 == armor 1개 × ★ **`boss.optionalPartArmorRatio`** (±5%) | 정본 §13.4 | **§10.1** |
| ★ **S26** 동시 개체 예산의 정적 하한 — 웨이브 **1개**의 `Σ count` ≤ `enemyConcurrentMax`(40) · 위기 서브웨이브 1파의 `Σ count` ≤ `swarmConcurrentMax`(70) | 정본 §13.4 (v1.2 신설) | **§8.3 · §12.2** |
| `capHits == 0` (★ v1.2: **A층 `defer` 발화까지** 집계) | 정본 §13.1 · §13.1.1 | **§12.2** · ★ **§8.3 (A층 `crisisWaveResidualMax` — 이 섹션이 위반을 발견했다, §14.3-N-4)** |
| `dominance`(6종) · `crisisKillShareWithoutCapstone` · `coinScarcity`(3종) · `farmXpRatio` | 정본 §13.1 · §13.1.1 | §13 |

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
| `desert` | 초고속 소형. **파밍 리스크의 화신** | `dive`(초고속, 무장 0) + `rearIn` + **`anchor`+`laser`**. **전부 스쳐 지나가는데 그 사이를 레이저가 긋는다** → 요격하러 위로 올라가는 것 자체가 값을 갖는다. `enemyExitForfeitsReward`(정본 §8.8)가 가장 아프게 물리는 테마 |
| `forest` | 확산탄 벽. 편대 밀집 | `strafe` 2종(`fan` / `wall`) + `orbitDrift`+유도탄. **수평선이 레인을 강제하고, 그 레인 안으로 유도탄이 따라온다** |
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

★ **고지의 표기 = 물결표 (정본 §8.2.1 인용 — 이 섹션은 표기를 정하지 않는다)**

| 정본 §8.2.1이 확정한 것 | 이 섹션에의 귀결 |
|---|---|
| **S8 검사 대상 = 저작 리스트**(`stages[].waves[]` 전체의 개체 수), ±3%p | §5.2의 블록 불변식이 이것을 **오차 0.0%p**로 통과시킨다. 실제 스폰이 검사 대상이 아니므로 `waveClearAdvance`·순환·`mobPhaseMaxWaves` 절단이 **S8에 영향을 주지 않는다** |
| **배너 표기 = `물 ~70% · 불 ~10% · 풀 ~10% · 노말 ~10%`** (물결표 필수) | 이 섹션은 `mix` 맵만 저작하고 **배너 문자열은 02·05의 소관**이다. 위 표의 `0.70`은 **`mix`의 값**이지 화면 문자열이 아니다 |
| **실제 조우와의 허용 괴리 ≤ 5%p** (검사하지 않음) | §5.5가 이 섹션의 실제 편성에 대해 **최악 조우 편차를 계산**해 밴드 안에 있음을 보인다 |

> ★ 이전 판본은 이 표의 `0.70`을 "고지에 그대로 표시되는 수치"로 읽었다. **정본 v1.1이 그 약속을 물결표로 교체**했다 — `70%`라 쓰면 75%를 만난 플레이어가 배신당하고, `~70%`는 **약속의 정밀도가 이행 가능성과 일치**한다. 이 섹션이 바꿀 것은 없고 인용만 한다.

### 1.4 진행 곡선 (정본 §8.3 · §13.5 · §13.6.1 인용, 재정의 아님)

`enemyHpScale` · `xpScale` · `spawnDensityScale` · `midBossCount` · `elitePerWaveChance` · `swarmTotalScale` · `rearSpawnAllowed` — **6×7 테이블의 값은 정본 §8.3/§9.9가 유일 소유자**다. 이 섹션은 그 값을 읽어 쓸 뿐 적지 않는다.

★ **화력·보스 HP의 두 곡선 — 둘 다 정본 소유이며 이 섹션은 인용만 한다 (★ v1.2에서 전량 재산출됐다)**

| 곡선 | s1 | s2 | s3 | s4 | s5 | s6 | 출처 |
|---|---|---|---|---|---|---|---|
| **`certify.dpsRef`** (명목 · 단일표적 · 무속성 · 보스전 진입 시점 · ★ **farm = `maxFarm`**) | **60** | **108** | **249** | **371** | **708** | **708** | **정본 §13.5 · §13.5.1** |
| 대비 배수 | 1.00 | 1.80 | 4.15 | 6.18 | 11.80 | 11.80 | 정본 §13.5 |
| `× uptimeRef 0.60` = **실효** | **36.0** | **64.8** | **149.4** | **222.6** | **424.8** | **424.8** | 정본 §10.4.3 |
| 파생: baseline(`balanced` farm) **명목** | 51 | 90 | 207 | 308 | 588 | 588 | 정본 §13.5 |
| **`bossHpScale`** (`tier ∈ {stage, mid}`) | **1.00** | **1.80** | **4.67** | **6.96** | **14.42** | (**14.42**) | **정본 §13.6.1** |
| **`bossRamp`** (= `bossHpScale ÷ 대비 배수`, **키가 아니라 파생의 기록** — 정본 §21.2-B5) | 1.00 | 1.00 | 1.125 | 1.125 | 1.222 | **1.222** | 정본 §13.6.1 |

★ **v1.2의 사슬 — 이 섹션이 반드시 알아야 하는 것 (정본 §13.5.1 · §13.6.2)**

| | |
|---|---|
| **바뀐 것** | `dpsRef`의 **farm 정책이 `maxFarm`으로 확정**됐다(v1.1은 farm 축을 정의하지 않았고 곡선을 `levelUpsPerRunTarget: 54` = **전형값**에서 유도했다). → 곡선이 `50/90/195/335/485/708` → **`60/108/249/371/708/708`**로 재산출됐다 |
| ★ **구조는 한 자리도 안 바뀌었다** | `killTime = 소요피해 × bossRamp[i] / (dpsRef[1] × 0.60 × m)`에서 **`dpsRef`가 정확히 상쇄**되므로(정본 §13.6.1), farm 정책이 바뀌어도 **φ·`bossRamp`·`noElementPass` 곡선이 전부 불변**이고 **스테이지 1 앵커만 30 → 36**으로 움직인다 → **보스 HP 전량이 ×1.20**(= 60/50) |
| ★ **`dpsRef[5] == dpsRef[6] == 708`** | 제너럴리스트의 화력 패시브가 `warhead` Lv4 × `overclock` Lv4에서 멈추므로 **명목 ST의 천장이 708**이고, maxFarm은 그 천장에 **스테이지 5에서** 도달한다. → **`bossHpScale[5] == bossHpScale[6] == 14.42`인 것은 버그가 아니라 정합**이다(화력이 같은 두 스테이지의 보스 HP가 같다) |
| ★ **`certify.runFarmDpsRatio` = 0.83** | baseline(`balanced` farm) 명목 ÷ `dpsRef`(maxFarm). **`run` 모드의 모든 지표가 이 비율을 통과한다** — 이 섹션의 격파 시간은 전부 **probe(maxFarm) 기준**이며, baseline 플레이어의 시간은 **÷ 0.83**이다(정본 §13.6.5). **이 섹션은 이 상수를 만지지 않는다** |

- ★ **이전 판본의 「참조 화력 곡선 45/85/150/240/360/520」은 전량 폐기**(정본 §18.3-1). 그것은 **가정**이었고, 정본의 곡선은 **03의 실제 무기 수치**에서 산출된다. 그리고 **대립의 진짜 원인은 단위였다** — 04의 520은 *실효*, 03의 708은 *명목*이며 `708 × 0.60 = 425`가 같은 종류의 수다. **`uptimeRef`를 정본이 소유하므로 이 대립은 재발할 수 없다**(정본 §10.4.3). ★ **v1.2가 같은 처방을 farm 축에 적용**했다 — `runFarmDpsRatio`가 「maxFarm vs balanced」를 소유한다.
- ★ **이전 판본의 `bossHpScale` 제안값 `[1.0, 1.9, 3.4, 5.6, 8.4, 11.5]`도 폐기** — **필요성의 진단은 채택**되었고(테마 셔플 → `manta`가 s1에도 s5에도 온다) **값만 정본이 재산출**했다(정본 §19.3 R-1: 수정채택). v1.1의 `[1.0, 1.8, 4.4, 7.5, 11.9, 17.3]`도 **폐기**됐다(정본 §20.3-04-3).
- ★ **`tier: "final"`에는 `bossHpScale`을 적용하지 않는다** — `bossHpScale`은 **셔플 때문에 존재**하고 `tetrarch`는 셔플되지 않는다. 예외가 아니라 **규칙의 정의역**이다(정본 §13.6.1).
- ★ **`bossHpScale[6] = 14.42`는 계산되지만 스테이지 보스에는 쓰이지 않는다** — 스테이지 6에는 `tier: "stage"` 보스가 오지 않는다. **중간보스는 사용한다**(§7.2 · §11.5).
- **후반 스펀지 방지의 실물 = `unlockStageMin`**(정본 §8.3). 그 실제 편성이 §4다.

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

> 공용 9 + 시그니처 6 + 새떼 2 = **17**(정본 §8.6). `element`·`tier`·`hpScalePerStage`·`spriteId` 필드는 **없다**(정본 §9.7). ★ **`unlockStageMin`은 §4의 로스터 엔트리가 유일 소유자**이며 `enemies.json`에서 삭제되었다(정본 §9.7, R-6 채택). ★ **`bands[].sizePx`도 삭제** — `radius`가 히트박스이자 실제 크기의 단일 진실이고 `elite.sizeMult`는 `radius`에 곱한다(정본 §9.7, R-17 채택).
> `hp`는 **밴드 배수 적용 전 기본값**이다: `실효 HP = hp × band.hpMult × enemyHpScale[stage]` (정본 §8.6).
>
> ★ **`bands.chaff.xpRef` = 2 — 정본이 소유한다 (v1.2, 정본 §8.10 인용. 이 섹션의 N-1 요청이 채택돼 정본에 흡수됐다)**: 정본의 두 파생식(§8.10 `swarmXp = chaff XP × 0.5` · §8.9 `중간보스 xp = chaff XP × 25`)이 **둘 다 "chaff XP"를 참조**하는데 `chaff` 밴드에는 `drifter`(2)·`spitter`(3)·`rearDart`(3)·`swarmChaff`가 공존한다 → **기준 개체가 없으면 두 식이 값을 갖지 못한다.** 정본의 확정: **`bands.chaff.xpRef = 2`** (= `drifter.xp`, chaff 3종의 **최솟값** → 파생식이 과대평가되지 않는다 = S22가 상한 검사이므로 보수적 방향). **거처 = `enemies.json > bands.chaff`** — 밴드가 이미 `hpMult`·`coinDropChance`·`coin`을 소유하므로 밴드의 기준값이 밴드에 산다.
> → 파생: **`swarmXp = 1`**(§8) · **중간보스 `xp = 50`**(§7.2). 사격하는 chaff(`spitter`·`rearDart`)의 3은 **저작 편차**이며 앵커가 아니다.
> ★ **이 섹션의 의무 = `drifter.xp = 2`를 지키는 것**이다. `drifter.xp`를 바꾸면 `bands.chaff.xpRef`와 갈라져 두 파생식이 조용히 어긋난다 → **`drifter.xp`는 이 표에서 자유 저작 대상이 아니다**(정본 §8.10이 그 값을 이미 인쇄했다).

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
| `swarmChaff` | 새떼 | `chaff` | `delta` | 6 | `weave` | `{"speed":130,"ampPx":40,"freqHz":1.2}` | **`null`** | 6 | 3 | **1** | 10 |
| `swarmLancer` | 새떼창병 | `chaff` | `wedge` | 7 | `dive` | `{"speed":110}` | `{"emitterId":"lancerStraight","firstDelaySec":1.5}` | 6 | 4 | **1** | 15 |

**`themeOnly`**: `sirenRay`=`"sea"` · `frostLance`=`"glacier"` · `magmaBomb`=`"volcano"` · `dustRunner`=`"desert"` · `thornWeaver`=`"forest"` · `bogHexer`=`"bog"` · `swarmChaff`/`swarmLancer`=`"*swarm"`(웨이브 편성 금지, 위기 세션 전용 — S9). 나머지 9종 = `null`(공용).

**`desc`** (한국어 인라인, 정본 §0.3): `drifter` "떠내려오며 아무것도 하지 않는다" · `spitter` "흔들리며 앞으로 뱉는다" · `columnAnt` "같은 줄로 줄줄이 내려온다" · `flanker` "옆에서 가로지르며 부채꼴로 쏜다" · `hexer` "느린 저주탄을 조준해 쏜다" · `turretPod` "자리를 잡고 레이저를 긋는다" · `stalker` **"플레이어를 선회하며 유도탄을 쏜다"**(정본 §9.7) · `mortarHulk` "멈춰서 장판을 깐다" · `rearDart` "뒤에서 올라온다" · `sirenRay` "느린 유도 고리를 퍼뜨린다" · `frostLance` "틈이 있는 얼음 벽을 세운다" · `magmaBomb` "내려오며 용암을 흘린다" · `dustRunner` "스쳐 지나간다. 놓치면 끝" · `thornWeaver` "가로지르며 가시 벽을 남긴다" · `bogHexer` "느린 나선 저주를 뿌린다" · `swarmChaff` "숫자로 민다" · `swarmLancer` "새떼 속에서 한 발 쏜다".

★ **`stalker`의 지위 정정 (C-7)**: 이전 판본은 `stalker`의 전 필드를 "**정본 §9.7이 직접 저작한 값이며 한 글자도 바꾸지 않았다**"고 적었다. **그 근거는 무효다** — 정본 v1.1 **C-7**이 "§9.7의 `stalker`/`homing2` 블록은 전부 예시이며 확정값이 아니다"라고 못박았다(`core.hp: 4000`을 정본 저작값이라 인용한 것과 **정확히 같은 오독**이었다). **결과는 유효하다**: 그 값들은 이제 **이 섹션이 저작한 값**이며(`band:"bruiser"` · `radius:18` · `moveParams:{speed:60,turnRateDegSec:90,keepDistPx:140}` · `contactDmg:14` · `hp:30` · `xp:6` · `score:120`), 스키마 형태(필드 집합·타입)만 정본에서 온다. **`unlockStageMin`은 여기서 삭제**되고 §4의 로스터가 갖는다.

★ **새떼 XP = 1 의 산출 (정본 §8.10의 파생식)**: `swarmXp = bands.chaff.xpRef(2) × 0.5 = **1**` → `swarmChaff.xp = swarmLancer.xp = 1`. 이전 판본의 **2는 폐기**한다 — 그것이 §8.4의 「새떼 = 스테이지 총 XP의 47%」(**S22 위반**)를 만든 두 원인 중 하나였다. 나머지 하나(**웨이브 총 개체 수 부족**)는 §5가 닫는다. ★ **v1.2 주의**: 정본 §8.10의 「~25%」는 **폐기**됐으므로 이제 47%가 위반이었던 근거는 **S22(≤0.30) 하나**다 — 47%는 S22를 1.6배 넘었다. **S22는 상한이지 목표가 아니다**(§8.4).

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
| `driftHoming` | 8 | 0.8 | 95 | 14 | `circle` | `null` | 0 | **0** | **30** | **0.6** | `sirenRay` 느린 유도 고리 |
| `hexBolt` | 9 | 0.8 | 95 | 6 | `hex` | **`"slow"`** | 2.5 | 0 | 0 | 0 | 둔화 조준탄·나선탄 |
| `frostBrick` | 8 | 0.8 | 105 | 8 | `hex` | **`"slow"`** | 2.0 | 0 | 0 | 0 | 얼음 벽 (둔화) |
| `thornBrick` | 7 | 0.8 | 120 | 8 | `circle` | `null` | 0 | 0 | 0 | 0 | 가시 벽 |
| `heavyRound` | 10 | 0.8 | 115 | 14 | `circle` | `null` | 0 | 0 | 0 | 0 | 보스 중형탄 |
| `beamCore` | 4 | 0.8 | 0 | **22** | `circle` | `null` | 0 | 0 | 0 | 0 | **레이저 빔의 데미지 소유자** |
| `stunMark` | 11 | 0.8 | 80 | 6 | `hex` | **`"stun"`** | **1.0** | 0 | 0 | 0 | 스턴 (Hard+ 전용, §10.4) |

★ **`driftHoming` 행의 셀 오정렬 정정 (정본 §18.3-10)**: 이전 판본은 이 행에 **10열에 9개 값**을 넣어 `accel: 30 / turnRateDegSec: 0.6`으로 읽혔다. **그대로 JSON화하면 유도가 사실상 죽는다**(0.6°/게임초 = 아레나를 종단하는 8초 동안 4.8° 선회 = 직선탄). 정본이 확정한 정정: **`accel 0 / turnRateDegSec 30 / retargetSec 0.6`** — 위 표는 그것이다.

- `driftHoming`의 `turnRateDegSec` 30 + `speed` 95 = **정본 §7.8의 "느리게 곡선으로 추적 = 100% 탄"의 교과서적 실물.** 바다 테마가 `introOk`이자 "가장 읽기 쉬움"인 이유가 여기서 데이터로 성립한다 — 픽업(직선 스트릭·글로우 없음·≤6px)과 운동이 겹치는 구간이 없다.
- ★ **S14 동치 검사 (정본 §9.7 · §13.4)**: `bullets[].shape` 어휘는 **`["circle", "hex"]`로 폐쇄**되었고 `(status === null) === (shape === "circle")`가 강제된다. 위 표 전수 확인: `status: null` 7종(`pelletS` `fanShard` `homingM` `driftHoming` `thornBrick` `heavyRound` `beamCore`) = 전부 `circle` ✔ / `status ≠ null` 3종(`hexBolt` `frostBrick` `stunMark`) = 전부 `hex` ✔ → **10종 전부 동치** ✔. 정본 §7.12.4-⑤가 여기에 **호박 테두리**를 얹어 `(status ≠ null) ⟺ (hex) ⟺ (호박 테두리)` **3중 동치**가 되며, 색맹 모드에서도 무손실이다.
- `beamCore`의 `speed = 0`: `laser`는 빔이지 이동하는 탄이 아니다. 기하는 `widthPx`가, **데미지는 `bullets[].dmg`가**(정본 §3.2의 유일한 소유자 목록) 담당한다. `radius` 4 = `minBulletRadiusPx` 하한(정본 §12.4).
- `stunMark`: `statusDurationSec = 1.0` = `fairness.maxStunSec` 정확히 하한 준수(정본 §2.7).
- **`zone`은 탄을 쓰지 않는다** — `zone` 이미터가 `dmg`를 직접 갖는다(정본 §8.5). ★ **`bulletId: null` 허용은 정본 §9.7이 명시했고 `check.mjs` S19가 `type == "zone"` ⟺ `bulletId == null`을 동치로 강제**한다(R-5 채택). → §3.2의 `hulkZone`·`magmaZone`이 `null`인 것은 이제 **누락이 아니라 규격**이다.

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
| `podLaser` | `laser` | `beamCore` | `self` | **1.20** | 6.0 | `widthPx:16, activeSec:0.5, angleDeg:90, trackDuringCharge:false` | 1.20 ✔ |
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
| 스턴 개체 수 | `statusStunMaxPerStage` **2** | 스테이지당 **1** (§10.5) | ✔ 여유 100% |
| 스턴 난이도 | `stunMinDifficulty` `"hard"` | ★ **Normal에서는 그 이미터가 발사를 스킵한다**(정본 §2.7, R-3 채택) — 탄 치환 없음·텔레그래프 없음·침묵, 그리고 **rng draw를 소비하지 않아** 난이도 간 스트림 정렬이 유지된다 | ✔ |
| ★ **S6의 검사 대상** | `enemies.json > emitters`**만** (`fairness.playerWeaponsExempt`, 정본 §13.4) | 이 표의 13종 + §7.2의 중간보스 4종 + §9.4의 보스 시드 22종 | ✔ 플레이어 무기는 대상 아님 |

★ **`orbitDrift.keepDistPx`가 `minSpawnRadiusPx`(140)의 실제 집행 장치다.** `stalker`가 `keepDistPx: 140`(정본 §9.7 저작값)으로 정확히 하한에 서는 것은 우연이 아니다 — 이 값보다 가까이 붙는 적은 **탄을 쏠 수 없어 무해해진다.** 그래서 `sirenRay`는 200으로 더 멀리 세워 "느린 대형 개체"의 사거리 압박을 준다. **`keepDistPx < 140`인 `orbitDrift` 적은 저작 금지**(§12.1 G-11).

---

## 4. 테마별 로스터 (`data/stages.json > themes[].roster`)

> **테마당 정확히 4종 = 공용 3 + 시그니처 1** (정본 §8.6). ★ **`unlockStageMin`의 유일한 거처 = 로스터 엔트리**(정본 §9.7, R-6 채택) → 같은 공용 아키타입이 테마마다 다른 시점에 해금된다.

| 테마 | `element` | 로스터 4종 (`unlockStageMin`) | 티어 구성 | **S23** `turret`+`bruiser` |
|---|---|---|---|---|
| `sea` | 물 | `drifter`(1) · `spitter`(1) · **`sirenRay`**(2) · `stalker`(**3**) | T1{drifter,spitter} T2{+sirenRay} T3{+stalker} | `sirenRay`·`stalker` = **2** ✔ |
| `glacier` | 물 | `spitter`(1) · `turretPod`(2) · **`frostLance`**(2) · `columnAnt`(3) | T1{spitter} T2{+turretPod,frostLance} T3{+columnAnt} | `turretPod`·`frostLance` = **2** ✔ |
| `volcano` | 불 | `spitter`(1) · **`magmaBomb`**(2) · `mortarHulk`(3) · `rearDart`(**3**) | T1{spitter} T2{+magmaBomb} T3{+mortarHulk,rearDart} | `magmaBomb`·`mortarHulk` = **2** ✔ |
| `desert` | 불 | `drifter`(1) · **`dustRunner`**(1) · ★ **`turretPod`**(2) · `rearDart`(**3**) | T1{drifter,dustRunner} T2{+turretPod} T3{+rearDart} | `turretPod` = **1** ✔ |
| `forest` | 풀 | `spitter`(1) · **`thornWeaver`**(1) · `flanker`(2) · ★ **`stalker`**(3) | T1{spitter,thornWeaver} T2{+flanker} T3{+stalker} | `stalker` = **1** ✔ |
| `bog` | 풀 | `hexer`(1) · `flanker`(2) · **`bogHexer`**(2) · `turretPod`(3) | T1{hexer} T2{+flanker,bogHexer} T3{+turretPod} | `bogHexer`·`turretPod` = **2** ✔ |
| `finale` | `null` | **공용 9 + 시그니처 6 = 15종 전부**(`unlockStageMin` 전부 충족, 정본 §8.16) | 단일 티어 | 4종 ✔ (finale는 `themeDraw` 대상이 아니므로 S23 대상 밖) |

★ **`desert`·`forest` 로스터 재편성의 근거 = S23 (정본 §13.4 신설)**

정본 v1.1은 `coinScarcity`를 처음으로 검산하며(§13.2-⑪) **테마 간 코인 분산의 유일한 원천 = 로스터의 `turret`+`bruiser` 비율**임을 밝히고 **S23**(테마당 1~2종)을 신설했다. 이전 판본의 로스터를 그 문에 넣으면:

| 테마 | 이전 로스터의 `turret`+`bruiser` | 판정 |
|---|---|---|
| `desert` | `drifter`(chaff) · `dustRunner`(line) · `columnAnt`(line) · `rearDart`(chaff) → **0종** | ✗ **S23 위반 — 웨이브 코인 수입이 구조적으로 0** |
| `forest` | `spitter`(chaff) · `thornWeaver`(line) · `flanker`(line) · `hexer`(line) → **0종** | ✗ **S23 위반** |

→ 로스터 편성은 **이 섹션의 위임 범위**이므로 정본 개정이 아니라 **이 섹션이 고친다**:
- **`desert`: `columnAnt` → `turretPod`.** 그리고 이것이 **정체성을 강화한다** — 사막은 "파밍 리스크의 화신"이고 리스크의 정의는 **"XP를 잡으려면 위로 올라가야 한다"**(정본 §8.8)인데, 이전 로스터에는 **위가 위험할 이유가 없었다**(전부 스쳐 지나가는 소형). `turretPod`(`anchor` `yHoldPx 150` + `laser`)가 **바로 그 위에 서서 레이저를 긋는다** → `dustRunner`를 쫓아 올라가는 것에 **처음으로 가격표가 붙는다.** 잃은 것은 `column`↔랜스 시너지 1건인데, `columnAnt`는 `glacier` T3와 `finale`에 그대로 남아 어휘가 죽지 않는다.
- **`forest`: `hexer` → `stalker`.** 숲의 정체성은 "확산탄 벽 = 수평선이 레인을 강제"인데 `hexer`(느린 조준 저주탄)는 그 문장에 아무것도 더하지 않았다. `stalker`(`orbitDrift` + 유도탄)는 **레인 안으로 따라 들어온다** → "벽이 나를 가두고 유도탄이 그 안에서 찾아온다"가 되어 벽의 의미가 커진다. `hexer`는 `bog`(자기 정체성이 저주인 테마)와 `finale`에 남는다.

- `stalker`의 `unlockStageMin: 3`은 유지한다 — `sea`·`forest` 양쪽에서 **가장 늦은 티어**다(정본 §9.7의 예시값이었으나 C-7에 의해 이제 이 섹션의 저작값이다, §2.3).
- ★ **`rearDart`의 `unlockStageMin: 3`은 `rearSpawnAllowed[3+]`(정본 §8.3)와 정확히 일치하도록 배정했다.** → `rearIn`/`spawnEdge:"bottom"` 웨이브가 **스테이지 1·2의 편성에 물리적으로 존재할 수 없다** → **S9가 자동으로 통과**한다(검사가 잡을 일이 없다). 규칙을 검사로 지키는 게 아니라 **구조로 지킨다.**
- **시그니처 해금이 늦은 테마 = 정체성이 늦게 온다.** `sea`(sirenRay@2)·`desert`(dustRunner@**1**)의 대비가 의도다: 사막은 **첫 순간부터** 파밍 리스크를 들이민다(`introOk: false`이므로 최소 스테이지 2), 바다는 천천히 보여준다(`introOk: true`, 스테이지 1 후보).

**공용 9종의 테마 분포 (`dominance.maxArchetypeLethalityShare ≤ 0.25` 대비)**: `drifter` 2 · `spitter` **4** · `columnAnt` 1 · `flanker` 2 · `hexer` 1 · `turretPod` **3** · `stalker` 2 · `mortarHulk` 1 · `rearDart` 2. → ★ **`spitter`(4테마)가 이 게이트의 유일한 관측 대상**이다. 분모 = 플레이어가 입은 총 피해(정본 §13.1.1)이고 `spitter`의 탄은 `pelletS`(dmg **8** = 저작 최소)이므로 **개체 수는 많고 발당 피해는 최저** — 상한 추정 0.15 ≤ 0.25 ✔. 넘으면 **첫 조정 = 한 테마의 `spitter`를 `drifter`로 교체**(값이지 규칙이 아니다).

---

## 5. ★ 웨이브 리스트 저작 — 블록 모델

### 5.1 자료구조 (정본 §8.7 인용)

```
waves: [ { formationId, archetypeId, count, element, spawnEdge, eliteIndex } ]   // 시각 없음, 순서만
```
`다음 스폰 = max(직전 스폰 + waveIntervalSec, 직전 웨이브 전멸 시각)` · `waveClearAdvance` = **남은 스케줄 전체 압축** · `mobPhaseMaxWaves = 14` · `crisisSuspendsWaves` · 초과 시 `defer` · ★ **`waveListExhausted = "cycle"`**(리스트 소진 시 인덱스 0으로 순환, 정본 §8.7 — R-7 채택). 순환 시 `eliteIndex`는 무시하고 `elite.perWaveChance`로 **재롤**하며 `element`는 **저작값 그대로**(혼합비의 분모와 분자가 함께 늘어난다).

### 5.2 ★ 블록 불변식 — S8을 오차 0으로 통과시키는 장치

**문제**: `unlockStageMin` 필터링은 스테이지마다 **다른 웨이브 리스트**를 만든다. S8은 "**저작 리스트**의 개체 수 기준 비율이 `mix`에 ±3%p"를 요구한다(정본 §8.2.1) → **5개의 서로 다른 리스트가 전부 70/10/10/10이어야 한다.** 웨이브를 임의로 나열하면 이 5중 제약을 동시에 만족시키기가 사실상 불가능하다.

**해법 — 블록 불변식**:

> **웨이브 리스트는 "블록"으로만 구성된다. 블록의 티어 = 그 블록이 쓰는 아키타입의 `unlockStageMin` 최댓값. 그리고 ★ 모든 블록은 각각 독립적으로 정확히 70/10/10/10이다.**

**증명**: 스테이지 `s`의 필터링 결과 = `{블록 T : T ≤ s}`의 합집합. 각 블록의 비율이 전부 동일한 상수 벡터 `(0.7, 0.1, 0.1, 0.1)`이면, **그 합집합의 비율 = 크기로 가중한 평균 = 같은 상수 벡터.** 블록을 몇 개 넣든 빼든, `u`가 얼마든 비율은 변하지 않는다. ∎

- ★ **한 티어에 블록이 여러 개 있어도 증명은 그대로다** — 합집합의 비율은 구성 블록의 개수와 무관하다. v1.1 개정에서 **T1 블록이 2개로 늘어난 것**(§5.3)이 이 자유도를 쓴 것이다.
- ★ **S8은 저작 리스트만 본다**(정본 §8.2.1) → 순환·`waveClearAdvance`·`mobPhaseMaxWaves` 절단은 **S8에 영향을 주지 않는다.** 그것들이 만드는 **실제 조우 편차는 §5.5가 따로 계산**해 정본의 ≤5%p 밴드 안에 있음을 보인다.

**결과: 전 테마 · 전 스테이지에서 S8 오차 = 0.0%p** (±3%p 예산을 통째로 남긴다). 검산은 §5.5.

### 5.3 ★ 블록 슬롯 규격 (v1.1 개정 — S22가 블록 예산을 다시 썼다)

블록 단위 `u ∈ ℤ≥1`. `T`=테마 속성, `C`=counter, `P`=prey, `N`=노말.

| 슬롯 | `element` | `count` | 역할 |
|---|---|---|---|
| **s1** | `T` | `4u` | 본대 |
| **s2** | `C` | `1u` | ×1 소대 |
| **s3** | `T` | `3u` | 본대 |
| **s4** | `P` | `1u` | ★ **×0.5 소대 = 특화의 세금이 눈앞에 서는 자리** |
| **s5** | `N` | `1u` | ×1 소대 |

블록 합계 = `10u` 개체, `T:C:P:N = 7u:1u:1u:1u` = **70/10/10/10 정확** ✔

★ **v1.1이 바꾼 것 — 테마당 블록 3개(u ∈ {1,2}) → 블록 4개 `T1a · T2 · T3 · T1b`**

| 무엇이 강제했나 | 산술 |
|---|---|
| ★ **S22** (정본 §13.4 신설): `(crisisTotal × swarmTotalScale[i] × swarmXp) ÷ (스테이지 i 저작 리스트의 Σ XP) ≤ 0.30` | `swarmXp = 1`(§2.3) → 위기 XP = `60 × swarmTotalScale[i]` = **[30, 42, 51, 60, 60]** → **필요 Σ XP ≥ [100, 140, 170, 200, 200]** |
| 이전 판본의 스테이지 1 리스트 | **T1 블록 1개(`u=2`) = 5 웨이브 · 20기 · Σ XP 50** → `30/50 = 0.60` ✗ **S22를 2배로 위반** |
| ★ **왜 `u`만 키울 수 없나** | `u`를 키우면 `s1 = 4u`가 한 레코드의 편대 수용량을 넘는다(§5.7 G-18: `lineH`/`vWedge` ≤ 9). 그리고 T1은 **chaff만** 쓸 수 있어(`unlockStageMin = 1`) XP 밀도가 구조적으로 낮다 |
| ★ **해법 = T1 블록을 2개로** | 스테이지 1 리스트 = `T1a + T1b` = **10 웨이브**. §5.2의 증명은 블록 개수에 불변이므로 **S8은 여전히 0.0%p** |
| ★ **정본 §8.2.1의 조우 편차 예시가 그대로 유효한 이유 (검산)** | 정본이 인쇄한 `sea` s1 케이스(소극 11웨이브 → **75.0%**, 최대 파밍 14웨이브 → **72.4%**)는 **슬롯 패턴의 주기성만** 쓴다. 블록이 전부 같은 5슬롯 구조·같은 `u`이면 **11웨이브 = 1순환 + 1웨이브**의 개체 구성이 **5웨이브 리스트일 때와 완전히 동일**하다 → 두 수치가 **한 자리도 바뀌지 않는다** ✔ (아래 §5.5에서 재검산) |

- **T1 블록(T1a·T1b) = 정확히 5 레코드, 슬롯 분할 금지.** ← 위 주기성이 정본 §8.2.1의 산술을 보존하는 조건이다.
- **T2·T3 블록 = 슬롯 분할 허용** (`u=2~3`). 근거: 이 블록들은 **스테이지 1에 존재하지 않으므로** 정본 §8.2.1의 s1 산술에 관여하지 않는다. 분할은 **대형 개체 완충**에 쓴다 — `s3 = 3u`를 전부 `bruiser`로 채우면 한 웨이브가 DPS 벽이 되므로 `bruiser 일부 + chaff 나머지`로 나눈다.
- **분할의 유일한 제약**: 같은 슬롯의 레코드는 **`element`가 같고 `count` 합이 슬롯 규격과 같을 것.** (블록 비율 불변 → §5.2의 증명 유지)
- ★ **블록 순서 = `T1a → T2 → T3 → T1b` (저작 순서, 동결).** 근거: `mobPhaseMaxWaves = 14`가 리스트를 앞에서 자르므로 **T3를 뒤에 두면 스테이지 3~5에서 T3 아키타입(`stalker` 등)이 한 번도 안 나온다.** 이 순서면 s3~5의 앞 14 웨이브가 **T1a(5) + T2(6) + T3의 앞부분**을 덮어 **전 티어가 조우된다.** 그리고 스테이지 1의 필터 결과는 `T1a → T1b`(10웨이브)로 연속이라 순환이 정상 작동한다.

★ **저작 계약 — 테마별 Σ XP 정규화 (이 섹션이 스스로에게 거는 제약)**

| 스테이지 | 포함 블록 | **목표 Σ XP** | S22 필요 하한 | 근거 |
|---|---|---|---|---|
| 1 | T1a+T1b | **220 ± 10%** | 100 | 정본 §13.5의 `dpsRef` 곡선은 **스테이지별 XP 배분 ∝ `xpScale × spawnDensityScale`**를 전제한다 — 그 전제는 **테마가 XP를 균질하게 준다**는 뜻이다 |
| 2 | +T2 | **340 ± 10%** | 140 | 테마별 Σ XP가 갈리면 **레벨 진행이 테마의 함수**가 되어 `dpsRef`가 예측력을 잃고 `dominance.maxThemeClearStddev ≤ 0.06`이 위협받는다 |
| 3~5 | +T3 | **440 ± 10%** | 170 / 200 / 200 | |

→ **`u`는 테마마다 다르다.** chaff의 XP가 낮은 테마(`sea` drifter 2 / spitter 3)는 `u`를 키우고, 높은 테마(`bog` hexer 5)는 줄인다. **`u`는 값이고 슬롯 규격은 규칙이다.**
- ★ **`s4`(prey)가 이 게임의 셀링 포인트가 매 웨이브 실물로 나타나는 자리다.** 테마 정답 스탠스를 켜고 있으면 s1·s3(70%)은 ×2로 녹지만 **s4는 ×0.5로 튕긴다**(정본 §7.7: 회색 방패 호 + 파티클 0 + 금속 "틴"). 플레이어는 무기를 탓하지 않고 **스탠스를 탓한다** → 정본 §9.9의 온보딩이 "스크립트가 아니라 스테이지 1의 *구성*"이라고 한 것의 실물이 **바로 이 슬롯**이다. 대비 웨이브가 폐기되고도 레슨이 성립하는 이유가 여기 있다.
- `s4`에 **가능하면 큰 밴드를 배치**한다(`line`+). ×0.5로 오래 버티는 개체라야 세금이 체감된다. chaff에 ×0.5는 어차피 1~2히트라 안 보인다.

### 5.4 `eliteIndex` 저작 규칙

`eliteIndex ≠ null`인 웨이브만 **엘리트 후보**이며, 스폰 시 `rng.elite`로 `elitePerWaveChance[stage]`(정본 §8.3)를 굴려 성공하면 그 인덱스의 개체에 접두가 붙는다.

> **저작 규칙 (G-12)**: `eliteIndex = 0`은 **`band ∈ {line, turret, bruiser}` 이고 `element ≠ normal`인 웨이브에만** 적는다. 나머지는 전부 `null`.
> 근거: 정본 §8.6의 `elite.bandAllowed`(chaff 엘리트 = 허수아비 금지) + `elite.elementAllowed`(노말 엘리트 = 상성 재미 0)를 **런타임 거부가 아니라 저작 단계에서** 만족시킨다 → 엘리트 출현율이 굴림 실패로 조용히 낮아지는 사고가 없다.

### 5.5 ★ `sea` 웨이브 리스트 전량 (22 웨이브 — 다른 5 테마의 판형)

테마 `T`=물 · `C`=풀 · `P`=불 · `N`=노말. 블록 순서 = **T1a → T2 → T3 → T1b** (§5.3).

| # | 블록·슬롯 | `formationId` | `archetypeId` | `count` | `element` | `spawnEdge` | `eliteIndex` | XP |
|---|---|---|---|---|---|---|---|---|
| 1 | **T1a** (`u=4`) s1 | `scatter` | `drifter` | 16 | `water` | `top` | `null` (chaff) | 32 |
| 2 | T1a s2 | `scatter` | `spitter` | 4 | `grass` | `top` | `null` (chaff) | 12 |
| 3 | T1a s3 | `arc` | `spitter` | 12 | `water` | `top` | `null` (chaff) | 36 |
| 4 | T1a s4 | `arc` | `spitter` | 4 | **`fire`** | `top` | `null` (chaff) | 12 |
| 5 | T1a s5 | `scatter` | `drifter` | 4 | `normal` | `top` | `null` | 8 |
| 6 | **T2** (`u=3`) s1 | `arc` | `spitter` | 12 | `water` | `top` | `null` (chaff) | 36 |
| 7 | T2 s2 | `scatter` | `sirenRay` | 3 | `grass` | `top` | **`0`** | 27 |
| 8 | T2 s3a | `scatter` | `sirenRay` | 3 | `water` | `top` | **`0`** | 27 |
| 9 | T2 s3b | `arc` | `spitter` | 6 | `water` | `top` | `null` (chaff) | 18 |
| 10 | T2 s4 | `scatter` | `spitter` | 3 | **`fire`** | `top` | `null` (chaff) | 9 |
| 11 | T2 s5 | `lineH` | `drifter` | 3 | `normal` | `top` | `null` | 6 |
| 12 | **T3** (`u=3`) s1 | `arc` | `spitter` | 12 | `water` | `top` | `null` (chaff) | 36 |
| 13 | T3 s2 | `scatter` | `stalker` | 3 | `grass` | `top` | **`0`** | 18 |
| 14 | T3 s3a | `scatter` | `stalker` | 3 | `water` | `top` | **`0`** | 18 |
| 15 | T3 s3b | `vWedge` | `spitter` | 6 | `water` | `top` | `null` (chaff) | 18 |
| 16 | T3 s4 | `arc` | `sirenRay` | 3 | **`fire`** | `top` | **`0`** | 27 |
| 17 | T3 s5 | `scatter` | `drifter` | 3 | `normal` | `top` | `null` | 6 |
| 18 | **T1b** (`u=4`) s1 | `scatter` | `spitter` | 16 | `water` | `top` | `null` (chaff) | 48 |
| 19 | T1b s2 | `arc` | `spitter` | 4 | `grass` | `top` | `null` (chaff) | 12 |
| 20 | T1b s3 | `arc` | `spitter` | 12 | `water` | `top` | `null` (chaff) | 36 |
| 21 | T1b s4 | `scatter` | `spitter` | 4 | **`fire`** | `top` | `null` (chaff) | 12 |
| 22 | T1b s5 | `lineH` | `drifter` | 4 | `normal` | `top` | `null` | 8 |

**블록 합계**: T1a = 40기 / **XP 100** · T2 = 30기 / **XP 123** · T3 = 30기 / **XP 123** · T1b = 40기 / **XP 116**

**S8 검산 (저작 리스트, 스테이지별 필터링 후, 개체 수 기준 — 정본 §8.2.1)**

| 스테이지 | 포함 블록 | 총 개체 | 물 | 풀 | 불 | 노말 | 비율 | 오차 |
|---|---|---|---|---|---|---|---|---|
| 1 | T1a+T1b | 80 | 56 | 8 | 8 | 8 | **70/10/10/10** | **0.0%p** ✔ |
| 2 | T1a+T2+T1b | 110 | 77 | 11 | 11 | 11 | **70/10/10/10** | **0.0%p** ✔ |
| 3~5 | 전 4블록 | 140 | 98 | 14 | 14 | 14 | **70/10/10/10** | **0.0%p** ✔ |

**★ S22 검산 (위기 XP ÷ 저작 Σ XP ≤ 0.30)**

| 스테이지 | Σ XP | 위기 XP = `60 × swarmTotalScale × 1` | 비율 | 판정 |
|---|---|---|---|---|
| 1 | 100 + 116 = **216** | 60 × 0.5 = 30 | **0.139** | ✔ 여유 2.2배 |
| 2 | + 123 = **339** | 60 × 0.7 = 42 | **0.124** | ✔ |
| 3 | + 123 = **462** | 60 × 0.85 = 51 | **0.110** | ✔ |
| 4·5 | **462** | 60 × 1.0 = 60 | **0.130** | ✔ |

**★ 실제 조우 검산 — 정본 §8.2.1의 예시 2행이 그대로 재현된다 (≤5%p 밴드)**

스테이지 1의 필터 결과 = `T1a → T1b` = **10 웨이브**, `waveListExhausted: "cycle"`(정본 §8.7)로 순환한다.

| 케이스 | 조우 구성 | 개체 | 물 | 불 | 풀 | 노말 | 편차 |
|---|---|---|---|---|---|---|---|
| **저작 리스트** (S8 대상) | 전 80기 | 80 | 70.0 | 10.0 | 10.0 | 10.0 | **0.0%p** |
| `sea` s1 · **소극 파밍 11웨이브** | 10웨이브(80기) + 순환 1웨이브(T1a s1 = 16 물) | 96 | **75.0** | 8.3 | 8.3 | 8.3 | **+5.0%p** |
| `sea` s1 · **최대 파밍 14웨이브** | 10웨이브 + T1a s1~s4 (16+4+12+4 = 36기) | 116 | **72.4** | 10.3 | 10.3 | 6.9 | +2.4%p |

> ★ **정본 §8.2.1이 인쇄한 75.0 / 72.4와 한 자리도 다르지 않다.** 리스트가 5웨이브에서 10웨이브로, `u`가 2에서 4로 바뀌었는데도 그렇다 — **비율은 슬롯 패턴의 주기성에만 의존하고 `u`와 블록 개수에 불변**이기 때문이다(§5.3). **정본의 ≤5%p 선언이 이 섹션의 개정에 의해 무효화되지 않는다.**

**스테이지 3~5의 조우 편차** (앞 14웨이브 = T1a(5) + T2(6) + T3의 s1·s2·s3a(3) — `T1b`는 조우하지 않는다)

| | 개체 | 물 (`T`) | 불 (`P`) | 풀 (`C`) | 노말 |
|---|---|---|---|---|---|
| 조우 **88기** | 40 + 30 + 18 | **72.7** | 8.0 | 11.4 | 8.0 |
| 편차 | | **+2.7%p** | −2.0%p | +1.4%p | −2.0%p |

→ **최악 +2.7%p ≤ 5%p** ✔ 그리고 **조우 개체 88기**는 정본 §13.2-⑪의 코인 저작 계약 ★ **「66~88기」(v1.2에서 밴드로 개정 — 이 섹션의 N-3 채택)**의 **상단**이다 ✔ (`sea` s3~5가 최다, `bog` s2가 66기로 최소 — §6)

★ **16번 웨이브가 이 테마의 급소다.** 바다를 **물+4로 밀고 온 특화 빌드**가 만나는 `sirenRay`(불, `bruiser`)는 **물 스탠스로 ×0.5**다. 노말(Q)로 바꿔 ×1로 때리는 것이 정답이고(정본 §8.16이 증명한 Q의 존재 이유), 그 판단을 **잡몹 페이즈에서 미리 시킨다.** 그리고 이 웨이브는 `eliteIndex: 0` → 엘리트가 붙으면 HP ×4 = **×0.5로는 절대 못 녹는 벽**. 스테이지 보스(§10)가 요구할 커버리지를 여기서 예고한다.

### 5.6 나머지 5 테마의 블록 편성 (슬롯 규격은 §5.3 고정 → `formationId:archetypeId(count)` 지정)

| 테마 | 블록 | s1 (`T`, 4u) | s2 (`C`, 1u) | s3 (`T`, 3u) | s4 (`P`, 1u) | s5 (`N`, 1u) | 블록 XP |
|---|---|---|---|---|---|---|---|
| **`glacier`** (T=물, C=풀, P=불) | T1a `u=4` | `scatter:spitter`(16) | `scatter:spitter`(4) | `arc:spitter`(12) | `arc:spitter`(4) | `lineH:spitter`(4) | **120** |
| | T2 `u=2` | `arc:frostLance`(4) ★ · `arc:spitter`(4) | `scatter:turretPod`(2) ★ | `lineH:frostLance`(3) ★ · `vWedge:spitter`(3) | `scatter:turretPod`(2) ★ | `lineH:spitter`(2) | **119** |
| | T3 `u=2` | `columnV:columnAnt`(8) ★ | `arc:frostLance`(2) ★ | `scatter:turretPod`(2) ★ · `columnV:columnAnt`(4) ★ | `scatter:frostLance`(2) ★ | `lineH:spitter`(2) | **104** |
| | T1b `u=4` | `arc:spitter`(16) | `arc:spitter`(4) | `vWedge:spitter`(9)·`lineH:spitter`(3) | `scatter:spitter`(4) | `scatter:spitter`(4) | **120** |
| **`volcano`** (T=불, C=물, P=풀) | T1a `u=4` | `scatter:spitter`(16) | `scatter:spitter`(4) | `arc:spitter`(12) | `arc:spitter`(4) | `lineH:spitter`(4) | **120** |
| | T2 `u=2` | `arc:spitter`(8) | `scatter:magmaBomb`(2) ★ | `lineH:magmaBomb`(2) ★ · `vWedge:spitter`(4) | `scatter:magmaBomb`(2) ★ | `lineH:spitter`(2) | **96** |
| | T3 `u=2` | `scatter:rearDart`(8) **`bottom`** | `lineH:mortarHulk`(2) ★ | `arc:magmaBomb`(2) ★ · `arc:spitter`(4) | `scatter:mortarHulk`(2) ★ | `scatter:rearDart`(2) **`bottom`** | **100** |
| | T1b `u=4` | `arc:spitter`(16) | `arc:spitter`(4) | `vWedge:spitter`(9)·`lineH:spitter`(3) | `scatter:spitter`(4) | `scatter:spitter`(4) | **120** |
| **`desert`** (T=불, C=물, P=풀) | T1a `u=3` | `arc:drifter`(12) | `scatter:drifter`(3) | `lineH:drifter`(9) | `scatter:dustRunner`(3) ★ | `scatter:drifter`(3) | **99** |
| | T2 `u=2` | `scatter:dustRunner`(4) ★ · `arc:drifter`(4) | `scatter:turretPod`(2) ★ | `lineH:turretPod`(2) ★ · `vWedge:drifter`(4) | `scatter:turretPod`(2) ★ | `lineH:drifter`(2) | **134** |
| | T3 `u=2` | `scatter:rearDart`(8) **`bottom`** | `scatter:dustRunner`(2) ★ | `scatter:turretPod`(2) ★ · `arc:drifter`(4) | `scatter:rearDart`(2) **`bottom`** | `scatter:drifter`(2) | **90** |
| | T1b `u=3` | `scatter:drifter`(12) | `scatter:dustRunner`(3) ★ | `vWedge:drifter`(9) | `scatter:dustRunner`(3) ★ | `lineH:drifter`(3) | **138** |
| **`forest`** (T=풀, C=불, P=물) | T1a `u=3` | `pincer:thornWeaver`(12) ★ | `scatter:spitter`(3) | `vWedge:spitter`(9) | `arc:thornWeaver`(3) ★ | `lineH:spitter`(3) | **120** |
| | T2 `u=2` | `pincer:flanker`(8) ★ | `scatter:thornWeaver`(2) ★ | `arc:thornWeaver`(2) ★ · `vWedge:spitter`(4) | `scatter:flanker`(2) ★ | `lineH:spitter`(2) | **78** |
| | T3 `u=2` | `pincer:flanker`(8) ★ | `scatter:stalker`(2) ★ | `scatter:stalker`(2) ★ · `arc:spitter`(4) | `scatter:stalker`(2) ★ | `lineH:flanker`(2) ★ | **88** |
| | T1b `u=3` | `scatter:spitter`(12) | `arc:thornWeaver`(3) ★ | `pincer:thornWeaver`(9) ★ | `scatter:thornWeaver`(3) ★ | `lineH:spitter`(3) | **120** |
| **`bog`** (T=풀, C=불, P=물) | T1a `u=2` | `arc:hexer`(8) ★ | `scatter:hexer`(2) ★ | `lineH:hexer`(6) ★ | `scatter:hexer`(2) ★ | `vWedge:hexer`(2) | **100** |
| | T2 `u=2` | `pincer:flanker`(8) ★ | `scatter:bogHexer`(2) ★ | `arc:bogHexer`(2) ★ · `vWedge:hexer`(4) ★ | `scatter:bogHexer`(2) ★ | `lineH:flanker`(2) ★ | **114** |
| | T3 `u=2` | `arc:hexer`(8) ★ | `pincer:flanker`(2) ★ | `scatter:turretPod`(2) ★ · `arc:hexer`(4) ★ | `scatter:bogHexer`(2) ★ | `lineH:flanker`(2) ★ | **112** |
| | T1b `u=2` | `scatter:hexer`(8) ★ | `arc:hexer`(2) ★ | `vWedge:hexer`(6) ★ | `scatter:hexer`(2) ★ | `lineH:hexer`(2) | **100** |

★ = `eliteIndex: 0` (band ≥ `line` 이고 `element ≠ normal` → §5.4 규칙 충족). 무표기 = `null`.
`spawnEdge` 무표기 = `top`. **`bottom`은 `rearDart`(T3 블록 = `unlockStageMin 3`) 뿐** → `rearSpawnAllowed[3+]` 자동 준수 ✔

**★ S8 · S22 전 테마 검산 (저작 리스트 기준)**

| 테마 | `introOk` | Σ XP s1 | Σ XP s2 | Σ XP s3~5 | S22 s1 | S22 s2 | S22 s3 | S22 s4·5 | S8 |
|---|---|---|---|---|---|---|---|---|---|
| `sea` | ✔ | 216 | 339 | 462 | **0.139** | 0.124 | 0.110 | 0.130 | 0.0%p ✔ |
| `glacier` | ✖ | (240) | 359 | 463 | — | 0.117 | 0.110 | 0.130 | 0.0%p ✔ |
| `volcano` | ✔ | 240 | 336 | 436 | **0.125** | 0.125 | 0.117 | 0.138 | 0.0%p ✔ |
| `desert` | ✖ | (237) | 371 | 461 | — | 0.113 | 0.111 | 0.130 | 0.0%p ✔ |
| `forest` | ✔ | 240 | 318 | 406 | **0.125** | 0.132 | 0.126 | **0.148** | 0.0%p ✔ |
| `bog` | ✖ | (200) | 314 | 426 | — | 0.134 | 0.120 | 0.141 | 0.0%p ✔ |

→ **최악 = `forest` s4·s5의 0.148 ≤ 0.30** ✔ **여유 2.0배.** 그리고 Σ XP 스프레드 = s1 **200~240**(±10%) · s2 **314~371**(±8%) · s3~5 **406~463**(±7%) → §5.3의 정규화 계약 충족 ✔ → **레벨 진행이 테마의 함수가 아니다** = 정본 §13.5의 `dpsRef` 곡선이 전제하는 균질성이 성립한다.

- ★ **`desert` T1b s1·s2·s4에 `dustRunner`가 몰린 것은 의도다.** `dustRunner`(xp 15, speed 230)를 **`s2`(counter)·`s4`(prey)에 두면** 불+4 특화 빌드는 **가장 값비싼 개체를 ×0.5로 튕긴다** → §5.3의 s4 설계("×0.5로 오래 버티는 개체라야 세금이 체감된다")가 **XP 15짜리 도망자**와 결합해 **이 게임에서 가장 비싼 한 순간**을 만든다. 그리고 소극적 플레이어는 그 XP를 **구조적으로 0**으로 만든다(정본 §13.2-⑤의 `farmXpRatio` 논거 그 자체).
- **`glacier`/`volcano` T2 s3, `desert` T2 s3** 등의 분할은 **대형 개체 완충**이다: `s3 = 3u`를 전부 `turret`/`bruiser`로 채우면 한 웨이브가 DPS 벽이 되고 텔레그래프가 몰린다 → `대형 일부 + chaff 나머지`로 나눈다. 분할해도 `element`가 같으므로 **블록 비율은 불변**이다(§5.3).
- `pincer`는 `strafe` 전용, `columnV`는 `column` 전용 → `flanker`/`thornWeaver` / `columnAnt` 에서만 사용 ✔ **`check.mjs` S20**(정본 §13.4)이 `pincer` ⟺ `moveId == "strafe"` · `columnV` ⟺ `moveId == "column"`을 **동치로** 강제한다. **`count = 1`에 `pincer`·`columnV` 저작 금지**(G-13) — 위 표의 최소 `pincer` count = 2(`bog` T3 s2) ✔
- **`bog` T1이 전부 `hexer`인 것은 의도다.** `bog`는 `introOk: false`이므로 **스테이지 1에 배치될 수 없고**, T1 블록이 단독으로 등장하는 스테이지가 존재하지 않는다 — 항상 T2와 함께 온다. 그래도 S8은 T1 단독으로도 70/10/10/10이므로 **어떤 필터링에서도 안전**하다(§5.2). (위 표의 괄호 친 s1 Σ XP도 같은 이유로 **평가되지 않는 값**이지만, S22를 만족시켜 두어 `introOk`가 개정돼도 안전하다.)

### 5.7 `formations` 파라미터 — ★ **정본 §9.9.2가 소유한다 (인용)**

이전 판본은 이 자리에서 6종의 파라미터를 **스스로 저작**했다(`spreadPx`/`jitterPx`/`stepSec`/`depthPx`). **요청(R-14)은 채택되었으나 정본이 자기 규격으로 확정**했다(정본 §9.9.2) → **이 섹션의 파라미터는 폐기하고 정본을 인용한다.**

| `formationId` | **정본 §9.9.2의 파라미터** | 배치 |
|---|---|---|
| `lineH` | `gapPx: 64` | 수평 1열, `spawnEdge` 중앙 기준 좌우 대칭 |
| `columnV` | `gapSec: 0.5` | **`column` 전용.** 같은 x, `gapSec` 간격 순차 스폰 |
| `vWedge` | `gapPx: 56, angleDeg: 35` | V자, 선두 1기 + 좌우 대칭 |
| `arc` | `radiusPx: 180, spanDeg: 120` | 호, 중심 = 아레나 중앙 상단 |
| `pincer` | `yStartPx: 120, yStepPx: 60` | **`strafe` 전용. 좌우 교대 진입** |
| `scatter` | `jitterPx: 90, minSepPx: 40` | `rng.spawn` 산포, 최소 간격 강제 |

- ★ **`pincer`만 `strafe.yPx`를 덮어쓴다**: `yPx = yStartPx + floor(i/2) × yStepPx`. 이 예외가 없으면 `flanker.yPx`가 아키타입 고정값이라 **모든 측면기가 한 줄로만 지나간다 = `strafe` 거동이 죽는다**(정본 §9.9.2). 최대 `count 12`(`forest` T1a s1)에서 `yPx = 120 + 5×60 = 420` ≤ 아레나 h 720 ✔
- ★ **G-18 (이 섹션의 파생 저작 규칙 — 편대 수용량)**: 정본의 파라미터와 **이동 영역 폭 540**(정본 §1.1)에서 레코드당 `count` 상한이 파생된다. **새 값이 아니라 정본 값의 귀결이다.**

| 편대 | 파생 상한 | 산술 |
|---|---|---|
| `lineH` · `vWedge` | **9** | `(n−1) × gapPx(64) ≤ 540` → n ≤ 9.4 |
| `arc` · `pincer` | **12** | 호 길이 = `radiusPx 180 × 120°(2.094 rad)` = **377px** → n=12에서 간격 34px ≥ 최대 `radius` 22 × 1.3 ✔ / `pincer`는 `yStepPx`가 세로로 펼치므로 아레나 h가 상한 |
| `scatter` · `columnV` | **16** | `scatter`: `minSepPx 40`, 스폰 밴드 540×180 → 이론 상한 ≫ 16 / `columnV`: 시간 순차라 공간 상한 없음 |

→ §5.5·§5.6의 전 레코드가 이 상한 안이다 ✔ (최대 = `scatter:*`(16) · `arc:*`(12) · `lineH:drifter`(9))

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

★ **코인 분산 ⚠는 닫혔다 — 손잡이의 소유자는 정본이다 (정본 §13.2-⑪ · §18.3-8)**

이전 판본은 "테마 간 코인 수급 격차 3배"를 발견하고 **조정 손잡이 = 후보 웨이브 수**라고 적으며 §6을 가리켰다. 정본 v1.1의 판정:

| | |
|---|---|
| **진단은 옳았다** | 코인 분산은 실재했고, `bands[].coin`이 없어 `coinScarcity`는 **계산 자체가 불가능**했다(03-§9.6 채택 → 정본 §9.7이 `chaff 0 / line 0 / turret 1 / bruiser 2` 신설) |
| ★ **가리킨 곳이 없었다** | 이전 판본이 가리킨 §6의 제목은 **「이 섹션은 아무것도 정하지 않는다」**였다 → **존재하지 않는 손잡이 = 코인 분산이 무주공산.** 정본이 손잡이를 소유한다 |
| **손잡이 = `elite.coin`(3)** | 엘리트는 `elitePerWaveChance`(0.10→0.35)로 **이미 스테이지 곡선을 갖고 있어** 후반 수입만 선택적으로 올릴 수 있는 **유일한 수도꼭지**다 |
| **분산의 원천은 닫혔다** | ★ **S23**: 모든 테마의 로스터 4종 중 `turret`+`bruiser`가 **1~2종**(§4 전수표) → **분산의 유일한 원천이 구조적으로 봉쇄**된다. 후보 웨이브 수는 손잡이가 아니다 |

**이 섹션의 실제 수급 검산 (정본 §13.2-⑪의 모델에 §5의 편성을 대입)**

★ **v1.2: 이 섹션의 실측이 정본의 계약이 되었다 (N-3 채택 — 정본 §13.2-⑪ · §20.3-04-10)**

| 항목 | v1.1 정본의 계약 | **이 섹션의 실측** | ★ **v1.2 정본의 계약** |
|---|---|---|---|
| 스테이지당 조우 개체 | ~90기 (점 추정) | **66~88기** (`sea` s3~5 = **88기** / `bog` s2 = **66기**) | ★ **66~88기 (밴드로 기입)** |
| `turret`+`bruiser` 조우 | 20% × 90 = 18기 → **코인 4** | 15~20% → **코인 3.2~4.0** | ★ **13~18기 → 코인 3~4** |
| 런 총수입 | 228 | **224~228** | ★ **222~228** (−2.6%. 정본이 코인 하한을 3.2 → **3**으로 보수적으로 잡아 하단이 224 → 222가 됐다) |
| `medianEndCoins ∈ [0, 120]` | ~78 | **~74~78** | ★ **~72~78** ✔ 밴드 중앙 |

> ★ **왜 개체 수가 테마별로 갈리나**: §5.6이 **Σ XP 정규화(±10%)를 우선**하기 때문이다 — chaff의 XP가 높은 테마(`bog` hexer 5)는 같은 XP를 **적은 개체**로 채운다. **XP 균질성과 개체 수 균질성은 동시에 만족될 수 없고**, 이 섹션은 **XP를 택했다**(레벨 진행이 테마의 함수가 되면 `dpsRef`가 예측력을 잃는다, §5.3). 정본이 그 선택을 **밴드 표기로 수용**했다.

→ ★ **웨이브 코인은 스테이지 수입의 9~14%뿐**(나머지는 보스 16~18 + 중간보스 5~10 + 엘리트 3~13)이므로 부족분은 총수입의 **2.6%**다. `medianEndCoins`는 밴드 한복판에 있고, **하한에 붙어 있는 것은 `medianPurchasesPerVisit`(1.0)**이다(정본 §13.2-⑪) → **첫 조정 손잡이는 정본이 확정한 `elite.coin`이며 이 섹션은 그것을 만지지 않는다.**

---

## 7. 중간보스 (`data/bosses.json`, `tier: "mid"`)

### 7.1 정본이 확정한 것 (인용)

**단일 몸체·부위 없음·3종 전 테마 공용.** `midBossCount`(1~2) · `midBossAtSec` `[35]`/`[30,70]` · ★ **`hp = bosses[].hp × bossHpScale[stage]`** · **`midBossLeaveAfterSec = 30`**(= "선택적"의 정의, ★ **거처 = `stages.phase` — 유일 소유자**) · `midBossElementRule = "notThemeAndNotNormal"`(런타임 주입) · `midBossForcedLeaveOnCrisis` · ★ **`boss.midBossSummonsAllowed = ["mbNest"]`**(거처 = **`rules.json > boss`**, v1.2 확정) · 보상 `xp` = `bands.chaff.xpRef` × 25 = **50** / `coin` **5** / `healDropChance` **0.35** — ★ **거처 = `bosses[]` 개체 필드**(R-10 채택). (정본 §8.9)

★ **v1.2가 이 절에 통보한 3건 (정본 §8.9 · §20.3-04)**

| # | 정본의 확정 | 이 섹션의 처분 |
|---|---|---|
| 1 | ★ **base HP를 ×0.80으로 낮춘다** — `bossHpScale[5]`가 11.9 → **14.42**로 올랐으므로 `mbNest` **1100 → 880** | §7.2 표 재인쇄 · §7.5 전수 재검산 |
| 2 | ★ **`boss.midBossSummonsAllowed`의 거처 = `rules.json > boss`** — v1.1은 이 값을 확정하고 **S17로 강제**해 놓고 **어느 인쇄 블록에도 넣지 않았다**(= S17이 읽을 값이 없었다). 이 섹션은 §7.3에서 S17을 인용하면서 **거처를 묻지 않았다** — 그것이 이 섹션의 누락이다 | §7.1 · §7.3 인용 정정 |
| 3 | ★ **`bosses[].leaveAfterSec`는 존재하지 않는다** — §9.8의 잔재였고 `stages.phase.midBossLeaveAfterSec`(30)가 **이미 유일 소유자**였다 → **이중 거처 = 둘 중 하나가 미지 키 = 로드 실패** | §7.2 표에서 **행 삭제** |

★ **HP 스케일 정정: `enemyHpScale` → `bossHpScale` (정본 §8.9 · §18.3-7) — 인용**

이전 판본은 정본 v1.0을 따라 `enemyHpScale`을 썼다. **정본 v1.1이 그것을 자기 결함으로 정정**했고, **v1.2가 base 값을 재산출**했다:

```
enemyHpScale[5] = 4.5   vs   bossHpScale[5] = 14.42   vs   플레이어 화력 성장 = 11.80배 (dpsRef 대비 배수)
mbNest(880 base)를 enemyHpScale로 재면:  880 × 4.5 = 3,960  →  3,960 / 424.8 = 9.3 게임초에 죽는다
→ 정본 §8.9가 확정한 「DPS 체크 — 보스전 예고편」이 소멸한다 (leaveAfterSec 30의 1/3)
```
- ★ **base 값은 v1.2에서 ×0.80으로 재산출됐다**: `mbHammer` 900 → **720** · `mbLancer` 750 → **600** · `mbNest` 1100 → **880**. **1100은 `bossHpScale[5] = 11.9`였을 때 30초에 아슬아슬하도록 저작된 값**이고, 14.42에서는 37.3초가 되어 **이탈 시간을 넘긴다**(정본 §8.9). ★ **이것이 「밸런싱은 오직 숫자」의 실물이다** — 규칙도 구조도 안 바뀌고 `bosses.json`의 수 셋이 바뀐다.
- **잡몹 HP 곡선이 완만한 이유**("chaff는 끝까지 1~2히트", 정본 §8.6)는 **중간보스에 적용되지 않는다.** 중간보스는 화력을 재는 개체이므로 화력 곡선을 따라가야 한다.

**속성 규칙 = 보스전의 예고편**: 테마 정답 스탠스를 켠 채로는 안 죽는다 → **스탠스를 바꾸게 만든다.**

### 7.2 3종 확정

| 필드 | `mbHammer` 파쇄추 | `mbLancer` 창병 | `mbNest` 산란모함 |
|---|---|---|---|
| `moveId` | `anchor` | **`charge`** | `anchor` |
| `moveParams` | `{"enterSpeed":110,"yHoldPx":150,"swayAmpPx":190,"leaveAfterSec":30}` | `{"windUpSec":1.20,"dashSpeed":300}` | `{"enterSpeed":40,"yHoldPx":210,"swayAmpPx":60,"leaveAfterSec":30}` |
| `patternSet[0].emitterIds` | `["hammerFan","hammerZone"]` (교대) | `["lancerLaser"]` | `["nestAimed"]` |
| `summon` | — | — | `{"archetypeId":"drifter","count":3,"everySec":6.0,"formationId":"scatter"}` |
| `shapeId` / `radius` | `slab` / 34 | `spike` / 28 | `bulb` / 38 |
| ★ **`hp` (base)** — **정본 §13.6.4 소유** | **720** | **600** | **880** |
| `contactDmg` | 16 | 15 | 14 |
| `xp` / `coin` / `healDropChance` / `score` | **50** / 5 / 0.35 / 800 | **50** / 5 / 0.35 / 700 | **50** / 5 / 0.35 / 900 |
| `element` | **`null`** (런타임 주입 — S15) | `null` | `null` |
| 시험하는 것 (정본 §8.9) | 공간 압박 — 위치 판단 | 라인 회피 — 텔레그래프 읽기 | **DPS 체크 — 보스전 예고편** |

★ **`leaveAfterSec` 행은 삭제됐다 (정본 §9.8 · §21-A12)**: 이탈 30초의 **유일 소유자 = `stages.phase.midBossLeaveAfterSec`**이며 `bosses[].leaveAfterSec`는 **존재하지 않는다**. 이전 판본이 그 행을 인쇄한 것은 **정본 §9.8의 잔재를 그대로 베낀 것**이고, 「미지 키 = 에러」이므로 **그대로 JSON화하면 로드 실패**였다.
- ★ **`mbHammer`·`mbNest`의 `moveParams.leaveAfterSec: 30`은 별개 문제이며 요청으로 올린다 (§14.3-N-5)** — 그것은 `anchor` **거동의 파라미터**(정본 §8.4)이지 보상/이탈 규칙이 아니다. 그러나 `mbLancer`는 `charge`(그 파라미터가 **없다**)인데도 30초에 이탈하므로 **엔진은 `tier:"mid"`에서 `midBossLeaveAfterSec`를 읽을 수밖에 없다** → `anchor`의 30은 **읽히지 않는 두 번째 거처**다. 정본이 「`tier:"mid"`의 `anchor`는 `leaveAfterSec`를 저작하지 않는다」를 한 줄 확정할 때까지 **값을 30으로 유지**한다(생략하면 「누락 키 = 에러」에 걸릴 수 있으므로 보수적 선택).

**이미터** (`enemies.json > emitters`에 추가)

| `id` | `type` | `bulletId` | `telegraphSec` | `everySec` | `offsetSec` | 고유 |
|---|---|---|---|---|---|---|
| `hammerFan` | `fan` | `heavyRound` | **1.20** | 6.0 | **0.0** | `count:9, arcDeg:120, speed:115` |
| `hammerZone` | `zone` | `null` | **1.20** | 6.0 | **3.0** | `radius:82, activeSec:3.0, dmg:10` |
| `lancerLaser` | `laser` | `beamCore` | **1.20** | 5.0 | 0.0 | `widthPx:22, activeSec:0.6, angleDeg:90, trackDuringCharge:true` |
| `nestAimed` | `aimed` | `homingM` | **1.20** | 3.2 | 0.0 | `count:3, spreadDeg:26, speed:90, leadSec:0.4` |

- **`telegraphSec = 1.20` = 정본 §7.4의 "중간보스 패턴 = 전신 자홍 림 라이트, 하한 1.20"** ✔ 전 이미터 준수. `zone`(0.90)·`fan`(0.60)·`aimed`(0.60) 각자의 하한보다 **중간보스 하한이 크므로 1.20이 이긴다** — §3.3의 "두 하한이 겹치면 큰 쪽" 규칙 재적용.
- ★ **`hammerFan`/`hammerZone`의 `offsetSec` 0.0 / 3.0 (주기 6.0)** → 두 텔레그래프가 **정확히 3.0초 간격으로 교대**하며 각각 1.20초만 켜진다 → **동시 텔레그래프 = 항상 1개** ≤ 2 ✔ **S7 정적 통과.** "fan + zone 교대"(정본 §8.9)가 새 규칙 없이 `offsetSec` 하나로 표현된다.
- `lancerLaser.trackDuringCharge: true` + `charge.windUpSec 1.20` — **둘 다 ≥ `minTelegraphSec`** ✔. 돌진과 레이저가 각각 1.2초 예고를 갖는다.
- `mbLancer`의 `dashSpeed: 300`은 **적 기체 속도**이며 `fairness.maxBulletSpeed`(260, **탄** 상한)의 적용 대상이 아니다. 회피 예산은 `windUpSec 1.20` + 아레나 h=720이 준다.
- `nestAimed`의 `speed: 90` ≤ `maxAimedBulletSpeed` 200 ✔

### 7.3 스키마 2건은 정본 v1.1에 반영되었다 (인용)

이전 판본이 "**요청이 승인된다는 전제의 저작**"이라 단서를 단 두 필드는 **정본이 채택**했다(정본 §19.3 R-8·R-9) → 이제 **인용**이다.

| 필드 | 정본의 확정 | `check.mjs` |
|---|---|---|
| ★ `patternSet[i]` = **`{emitterIds: [...]}`** (배열) | **보스 부위 = 길이 1 강제 / 중간보스 = 1~2 허용** (정본 §8.9 · §9.8) | **S16** |
| ★ `bosses[].summon` = **`{archetypeId, count, everySec, formationId}` \| `null`** | **`tier == "mid"` 이고 ★ `boss.midBossSummonsAllowed`(= `rules.json > boss`, v1.2 확정) 통과 시에만 non-null**, 그 외 = 로드 실패 (정본 §8.9 · §9.8) | **S17** |

- **`mbHammer`의 동시 텔레그래프 정적 증명 (S7)**: `telegraphConcurrentMaxPerEntity`(2)가 그대로 상한이고, `everySec 6.0` / `offsetSec 0.0`·**3.0** 교대로 **동시 텔레그래프 = 항상 1개**가 정적으로 증명된다(§7.2 표) → 정본 §8.9가 확정한 「`fan` + `zone` 교대」가 **새 규칙 0으로 표현된다** ✔
- **`mbNest`의 `summon`은 `tier:"mid"` + `boss.midBossSummonsAllowed`(`["mbNest"]`) 이중 통과** ✔ S17이 동치로 강제하므로 **`mbHammer`·`mbLancer`의 `summon`은 반드시 `null`**이다(위 §7.2 표의 "—" = `null`).
- ★ **v1.2 승복 — 이 섹션은 S17을 인용하면서 「그 키가 어디 사는가」를 묻지 않았다**(정본 §21-A5): `midBossSummonsAllowed`는 v1.1에서 **확정되고 S17로 강제되면서도 어느 인쇄 블록에도 없었다** → **S17이 읽을 값이 존재하지 않아 검증기가 안 써지는 상태**였다. 정본 v1.2가 `rules.json > boss`로 확정했다. **교훈 = 「검사 항목을 인용할 때 그 검사가 읽을 값의 거처를 함께 확인한다」** — 이 섹션의 R-10이 중간보스 **보상** 필드의 거처를 옮길 때 **이 키만 빠졌고**, 그것을 알아챌 자리가 정확히 여기였다.

### 7.4 `mbNest`의 소환이 정본을 깨지 않는 이유 (확인)

`boss.summonsAllowed = false`(정본 §8.11)는 **스테이지 보스** 규칙이고, `mbNest`는 **잡몹 페이즈**에 있다(정본 §8.9: "중간보스는 잡몹 페이즈에 있으므로 XP 획득이 정상"). → §6.4의 "보스 페이즈에 XP 획득원이 존재하지 않는다"는 전제는 **영향받지 않는다** ✔ 소환된 `drifter`는 `xp: 2`를 정상 지급하며 이것이 `mbNest`를 "DPS 체크 + 파밍 기회"의 이중 선택으로 만든다. **정본이 R-9를 채택한 근거가 정확히 이것이다**(정본 §19.3).

### 7.5 ★ DPS 체크 검산 — v1.2 전수 재산출 (base ×0.80 · `bossHpScale` 교체 · ★ **모델 정정**)

`실효 HP = hp × bossHpScale[i]` · `killTime = 실효 HP ÷ (dpsRef[i] × uptimeRef(0.60) × m)`.
★ **상쇄식**(정본 §13.6.1): `bossHpScale[i] ÷ (dpsRef[i] × 0.60) = bossRamp[i] / 36` → **`killTime = base × bossRamp[i] / (36 × m)`.**

★ **모델 정정 — 이 표의 `m` = **1.0**이다 (v1.1은 `m` = 균형 빌드를 넣었고 그것이 틀렸다)**

| | |
|---|---|
| **v1.1의 표** | `m` = 균형 빌드(`[1.35 … 1.62]`)를 넣고 `mbNest` 27~29초를 「아슬아슬」이라 불렀다 |
| ★ **왜 틀렸나** | `midBossElementRule = "notThemeAndNotNormal"`은 **테마 정답 스탠스를 켠 채 온 플레이어**에게 `m` **≤ 1.0**을 준다 — 주입 속성이 `counter`면 **동속성 = ×1**, `prey`면 **×0.5**다(정본 §4.1). **균형 빌드의 `m` 1.62는 「스탠스를 바꾼 뒤」의 수**이고, 그 수로 「30초에 아슬아슬」을 논증하면 **이 개체가 시험하려는 바로 그 결정을 이미 통과한 플레이어를 재는 것**이 된다 |
| ★ **정본이 base 880을 유도한 모델** | `880 × 14.42 = 12,690` → `÷ 424.8` = **29.9초** (정본 §8.9의 인쇄값 **29.8초** — 절사 차, 결론 동일). **분모에 `m`이 없다 = ×1** → **정본도 `m` = 1.0으로 이 개체를 잰다.** 이 섹션이 정본에 맞춘다(C-1) |
| ★ **그리고 그 모델이 옳다** | `m` = 1.0은 **"스탠스를 안 바꾼 플레이어"**이고, **그 플레이어에게 30초가 아슬아슬한 것**이 정본 §8.9의 「스탠스를 바꾸게 만든다」의 정확한 산술이다 |

**전수 재산출** (`m` = 1.0 = **테마 스탠스를 켠 채 온 플레이어**)

| 중간보스 | base (v1.1 → **v1.2**) | s1 (`bossRamp` 1.000) | s3 (1.125) | s5·s6 (1.222) | vs `midBossLeaveAfterSec` **30** |
|---|---|---|---|---|---|
| `mbHammer` | 900 → **720** | 720 × 1.000/36 = **20.0s** | **22.5s** | **24.4s** | ✔ 전 스테이지 20~24s |
| `mbLancer` | 750 → **600** | **16.7s** | **18.8s** | **20.4s** | ✔ 전 스테이지 17~20s |
| **`mbNest`** | 1,100 → **880** | **24.4s** | **27.5s** | ★ **29.9s** | ★ **전 스테이지 24~30s = 아슬아슬** |

- ★ **`mbNest`가 30초 **직전**에 죽는 것이 s5·s6에서 정확히 성립한다** (29.9 vs 30 = **여유 0.1초**) = 정본 §8.9의 「DPS 체크 — 보스전 예고편」이 **선언이 아니라 산술**이 된다. `enemyHpScale`이었으면 s5에서 **9.3초**였고(§7.1), base 1100을 유지했으면 **37.3초 = 못 잡는다**(정본 §8.9).
- **왜 스테이지에 거의 무관한가**: `bossHpScale[i] ÷ (dpsRef[i] × 0.60) = bossRamp[i]/36`(0.0278~0.0339)만 남기 때문이다 — 정본 §13.6.1의 상쇄가 중간보스에도 그대로 작동한다. **테마 셔플 하에서 "몇 번째 스테이지에 왔나"가 중간보스의 난이도를 바꾸지 않는다** = 정본 §13.2-⑨의 `noDeadLuck.worstThemeOrder`가 성립하는 이유와 같은 구조. ★ **s1이 s5보다 22% 쉬운 것**(`bossRamp` 1.00 vs 1.222)이 유일한 스테이지 의존이며 그것이 의도다.
- ★ **스탠스를 바꾸면 무슨 일이 일어나는가 (이 개체의 존재 이유)**: 올바른 스탠스로 바꾼 균형 빌드(`m` = 1.62)는 s5의 `mbNest`를 **29.9 ÷ 1.62 = 18.4초**에 지운다 — **11.5초를 벌고 그 시간으로 웨이브를 판다.** 안 바꾸면 `counter` 주입에서 **0.1초 차로 겨우** 잡고, `prey` 주입(×0.5)이면 **59.8초 = 절대 못 잡는다.** → **"스탠스 하나가 11.5초 또는 전부"** = 정본 §8.9의 「보스전의 예고편」이 잡몹 페이즈에서 실물이 된다.
- ★ **v1.2에서 s5·s6이 같은 값인 이유**: `bossHpScale[5] == bossHpScale[6] == 14.42`이고 `dpsRef[5] == dpsRef[6] == 708`이므로(정본 §13.5.1의 화력 천장) **두 스테이지의 중간보스는 정확히 같은 개체다.** `bossHpScale[6]`을 배열에 남기는 유일한 이유가 이것이다(정본 §13.6.1: "중간보스는 사용한다"). → §11.5.

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
- ★ **서브웨이브 간격 = `crisisSubWaveIntervalSec` = `crisisDurationSec / crisisSubWaves` = 25 / 6 ≈ 4.17 게임초 — 파생값, 새 키 0** (정본 §9.9.3, R-15 채택). 그리고 이 4.17초가 **정본 §7.12.3의 근거**가 되었다: 최종 로테이션에 시각 규격을 신설하지 않는 이유가 "**적의 외곽선이 이미 답이고, 4.17초는 무-트위치 예산(0.55)의 7배**"이기 때문이다. → **이 섹션이 요청한 파생값이 정본의 시각 결정 하나를 지탱한다.**
- `spawnEdge` = `top` 고정. **새떼는 `bottom`에서 오지 않는다** — 25초 최고밀도 구간에 후방 진입을 겹치면 관대함이 붕괴한다. `rearSpawnAllowed`가 참인 스테이지에서도 금지 (G-14).
- `eliteIndex` = **전부 `null`.** 새떼는 혼합 비율 제외 대상이자 엘리트 제외 대상이다(정본 §8.2: "혼합 비율 제외 대상 = 엘리트 · 중간보스 · 새떼 · 보스"). `swarm*`은 `chaff` 밴드라 `elite.bandAllowed`에서도 이미 배제된다 → **이중으로 막혀 있다** ✔

### 8.3 밀도·캡 검산

| 항목 | 계산 | 판정 |
|---|---|---|
| 동시 개체 | 10기 × 6파, 간격 4.17초. `swarmChaff.speed 130` → 아레나 760px 종단 **5.8초** → 동시 ≈ 10 × (5.8/4.17) ≈ **14기** (플레이어가 전혀 못 잡을 때). 스테이지 6(`swarmTotalScale 1.0`) 최악에도 **≪ `swarmConcurrentMax` 70** | ✔ 여유 80% |
| ★ **S26** (v1.2 신설) | 위기 서브웨이브 **1파**의 `Σ count` = **10** ≤ `swarmConcurrentMax` **70** | ✔ 여유 85% |
| B층 캡 | `caps.enemies` **96** > 새떼 최악 14 + 웨이브 잔존 최악 **16**(아래) = **30** | ✔ 여유 69% |
| 적 탄 | `swarmLancer` 6기 × `lancerStraight`(count 1, everySec 6.0) → **동시 탄 ≪ 320** | ✔ |

★ **위기 진입 시 웨이브 잔존 — 실측 (v1.2 신설. 정본 §12.1이 이 수를 「04 §8.3의 실측」이라 인용하는데 이 섹션에는 그 계산이 없었다)**

정본 v1.2가 **`fairness.crisisWaveResidualMax = 10`을 신설**하며 근거로 *"04의 실측이 이 값을 지지한다(§8.3): 위기 진입 시점의 실제 잔존은 최악 ≈ 8"*이라 적었다. **이 섹션은 그 계산을 한 적이 없다.** 그래서 지금 한다.

```
① 한 번에 살아 있는 웨이브는 정확히 1개다  ← 정본 §8.7의 스포너 규칙에서 곧바로 나온다
   다음 스폰 = max(직전 스폰 + 9.0, 직전 웨이브 전멸)
   → 웨이브 N+1은 웨이브 N이 전멸하기 전에는 절대 스폰되지 않는다
   → ★ 웨이브 잔존 ≤ max(레코드 count) = 16   (`scatter:drifter`(16) · `scatter:spitter`(16), §5.5·§5.6)

② baseline(처치율 ~0.98, 클리어 2~3초 < 간격 9.0):
   스포너가 9.0초 정주기 → t=95의 살아있는 웨이브 나이 = 5.0초 → 이미 전멸
   → 잔존 ≈ 0~2                                              ← 정본의 「≈ 8」보다도 낮다

③ ★ passive(처치율 ~0) — 이것이 최악이고 정본이 재지 않은 정책이다:
   웨이브는 자체 이탈로만 소멸한다. drifter(`dive`, speed 70) → 아레나 760px 종단 10.9초
   → 스포너 주기 = 10.9초 → 웨이브 스폰 시각 = 0, 10.9, 21.7, …, 86.9
   → t=95에 86.9의 웨이브는 나이 8.1초 → 이동 567px < 760 → ★ 16기 전원 생존
   → ★ 잔존 = 16  >  crisisWaveResidualMax 10

④ 중간보스 소환물 (정의의 공백):
   `mbNest`(midBossAtSec 70) → drifter 3기 × everySec 6.0 → t=70~95에 12기 소환
   drifter 이탈 10.9초 → t=95 생존 ≈ 5~6기
   `midBossForcedLeaveOnCrisis`는 모함을 지우지만 이미 소환된 drifter는 남는다
   → 이 6기는 「웨이브 잔존」인가? 정본이 말하지 않는다
```

| 층 | 값 | 판정 |
|---|---|---|
| **B층** `caps.enemies` **96** vs 실측 최악 (새떼 14 + 잔존 16 + 소환 6 = **36**) | 여유 **62%** | ✔ **`capHits`의 B층 축은 안전하다** |
| ★ **A층** `crisisWaveResidualMax` **10** vs **산술적 최악 16** (passive, 스테이지 1) | ★ **초과** | ✗ ★ **`defer` 발화 → `capHits > 0` → certify 실패** |

★ **판정: 이 섹션의 편성은 A층 `crisisWaveResidualMax = 10`을 만족시킬 수 없다 — 그리고 그것은 편성의 결함이 아니다.**

- **한 웨이브 = 한 레코드**이고 `T1a`/`T1b`의 `s1 = 4u`는 `u = 3~4`에서 **12~16기**다. `u`는 §5.3의 Σ XP 정규화 계약이 정하고, **T1 블록의 슬롯 분할은 G-19-①이 금지**한다(정본 §8.2.1의 조우 편차 산술 11/14웨이브 주기성을 보존하는 조건). → **10 이하로 내릴 자유도가 이 섹션에 없다.**
- **그리고 내릴 필요가 없다**: 잔존 16은 **B층 96에 62%의 여유를 남긴다.** 문제는 안전망이 아니라 **A층 예산의 값**이다.
- → **§14.3-N-4 (blocker)**. ★ **권고 = `crisisWaveResidualMax` 10 → 16**: 16은 자유 숫자가 아니라 **`max(stages[].waves[].count)`**이고 **①(한 번에 한 웨이브)에 의해 잔존이 그 수를 넘는 것이 산술적으로 불가능**하므로 → **`defer`가 영원히 발화하지 않는다 = `capHits == 0`이 자명해진다.** 그러면 A층 `enemies` = 70 + 16 = **86 < 96**(여유 11.6%)이며, 정본의 「여유 20%」 관례를 지키려면 `caps.enemies` **96 → 104**(= 86 × 1.2)가 따라온다.

★ **부수 발견 — 「동시 개체 수는 처치율의 함수라 정적 검사가 불가능하다」(정본 §12.1)는 **웨이브 축에서는** 참이 아니다**: ①에 의해 **웨이브 개체의 동시 최대 = max(레코드 count)**이고 이것은 **완전히 정적**이다(S26이 이미 그 수를 읽는다). 처치율의 함수인 것은 **웨이브 + 중간보스 + 그 소환물 + 엘리트의 합**이며, 이 섹션의 실측 최악은 **웨이브 16 + 중간보스 2 + 소환 6 = 24 ≪ `enemyConcurrentMax` 40**(여유 40%)다. → 정본의 런타임 `defer` 확정(판정 9)은 **여전히 옳고**(합은 정적으로 못 잰다), 다만 **A층 40이 이 섹션의 콘텐츠에서 발화할 일은 없다.**
| **텔레그래프** | `swarmLancer` 6기 × duty(0.60/6.0 = 10%) = **0.6 동시** | ✔ |

★ **`swarmChaff`의 `attack: null`이 위기 세션을 캡 안에 묶는 유일한 장치다.** 60기 중 54기가 텔레그래프를 갖지 않으므로, 밀도 최고 구간의 동시 텔레그래프가 **≈ 0.6**에 그친다 — A층 `telegraphConcurrentMaxGlobal`(**80**, 정본 §12.1) 대비 **여유 99%**. 정본 §8.6이 `swarmChaff`를 "사격 안 함"으로 못박은 것이 여기서 캡 산술로 회수된다. (§12.2의 예산 문제는 **잡몹 페이즈 쪽이지 새떼 쪽이 아니다.**)

### 8.4 ★ S22 — 새떼 XP 상한 (이전 판본이 위반했던 게이트)

**정본 §18.3-9의 통보(v1.1)**: 「새떼 XP = 스테이지 총 XP의 **47%**」는 **위반이다.** **S22가 ≤ 0.30을 강제**한다. → **웨이브 편성의 총 개체 수를 늘려야 한다.** → **이행 완료**(아래).

★ **v1.2가 이 절에 통보한 것 — 「~25%」는 폐기됐고 S22는 상한이지 목표가 아니다 (정본 §8.10, 판정 7)**

| | 확정 |
|---|---|
| **v1.1의 「~25%」** | ★ **폐기.** 정본의 산술: **S22의 상한(`crisisXP = 0.30 × waveXP`)까지 올려도 총 XP 대비 18.2%가 천장**이므로 **~25%는 S22 하에서 산술적으로 존재할 수 없는 수**였다 |
| ★ **S22의 지위** | ★ **상한(cap)이지 목표가 아니다.** 「≤ 0.30」은 「새떼가 잡몹 페이즈를 대체하지 못하게」의 **안전망**이며, 실측이 그 절반 아래인 것은 **위반이 아니라 여유**다 |
| **정본이 소유하지 않는 것** | ★ **지분(%)** — 그것은 **웨이브 편성의 함수**이고 편성은 이 섹션 소관이다(§17). v1.1이 **소유하지 않는 양을 확정값으로 인쇄한 것**이 결함의 정체였다 |
| **이 섹션의 처분** | ★ **편성 변경 0.** §18.3-9의 명령은 **이미 이행됐고**(블록 3→4개, `swarmXp` 2→1), v1.2는 그 귀결을 **정본에 기입**했다(정본 §20.3-04-5: "04의 편성은 그대로 유효하며 추가 조치 없음") |
| ★ **이 섹션의 승복** | ★ **v1.1의 이 절은 2배 이탈(25% → 12~13%)을 「물결표가 붙은 목표」라며 요청으로 올리지 않았다.** **물결표는 5%p용이지 2배용이 아니다** — 정본이 그렇게 판정했고 옳다. **정본이 소유한 수가 이 섹션의 개정으로 깨졌으면 그것은 요청 사유다**, 물결표의 유무와 무관하게 |

**v1.1 위반의 두 원인과 처분 (기록 — 처분은 완료됐다)**

| # | 원인 | 처분 |
|---|---|---|
| 1 | `swarmChaff.xp = 2` | ★ **1로 정정.** 정본 §8.10의 파생식 `swarmXp = bands.chaff.xpRef × 0.5`에 **2**(정본 소유, §2.3)를 대입한 값이다. **2는 애초에 파생식을 만족하지 않았다** |
| 2 | ★ **웨이브 저작 총 개체 수 부족** | **§5.3이 블록 예산을 다시 썼다** — T1 블록 1개(20기) → **T1a+T1b(40~80기)**, 테마당 3블록 → **4블록**. 이것이 개정의 **가장 큰 구조 변경**이다 |
| — | ★ **이전 판본이 제안한 손잡이는 틀렸다** | "`swarmChaff.xp`를 2→1로 내리면 31%"라고 적었으나 **31%도 S22 위반**이다(≤0.30). 그리고 `swarmXp`는 **손잡이가 아니라 정본의 파생값**이다 — 만질 수 있는 것은 **분모(웨이브 편성)**뿐이다. 정본이 정확히 그렇게 판정했다 |

**S22 전수 검산** — `(crisisTotal 60 × swarmTotalScale[i] × swarmXp 1) ÷ (스테이지 i 저작 리스트 Σ XP) ≤ 0.30`

| 스테이지 | 위기 XP | Σ XP 최악 테마 | **비율** | 판정 |
|---|---|---|---|---|
| 1 | 30 | `sea` 216 | **0.139** | ✔ |
| 2 | 42 | `bog` 314 | **0.134** | ✔ |
| 3 | 51 | `forest` 406 | **0.126** | ✔ |
| 4 · 5 | 60 | `forest` 406 | **0.148** | ✔ **여유 2.0배** |
| 6 (`finale`) | 60 | 448 (§11.2) | **0.134** | ✔ |

→ **전 테마·전 스테이지 최악 0.148 ≤ 0.30** ✔ **여유 2.0배.**

★ **총 XP 대비 새떼 지분의 재계산 — v1.1의 이 절은 분모를 틀렸다 (이 섹션이 스스로 찾은 산술 오류)**

```
v1.1이 쓴 것:  지분 = S22 / (1 + S22) = 0.148/1.148 = 12.9%  ~  0.139/1.139 = 12.2%
★ 그 분모 = 웨이브 + 위기 뿐이다.  중간보스 XP와 엘리트 XP가 빠져 있다.
   → S22의 분모(= 저작 리스트 Σ XP)를 「총 XP」로 착각한 것이며, 지분이 구조적으로 과대평가된다.

옳은 분모 = 웨이브 + 위기 + 중간보스 + 엘리트
   중간보스 XP = midBossCount[i] × 50            (정본 §8.3 · §8.9)
   엘리트 XP 증분 = (eliteIndex ≠ null 웨이브 수) × elitePerWaveChance[i] × (그 개체 xp × (xpMult 6 − 1))
```

| 케이스 | 웨이브 | 위기 | 중간보스 | 엘리트 | **총 XP** | ★ **새떼 지분** |
|---|---|---|---|---|---|---|
| `sea` s1 (엘리트 후보 0 — T1a·T1b가 전부 chaff) | 216 | 30 | 1 × 50 | 0 | **296** | **10.1%** |
| `sea` s4 (★ **정본 §8.10의 재검산 케이스**) | 462 | 60 | 2 × 50 | 5후보 × 0.25 × 7.8 × 5 ≈ **49** | **671** | ★ **8.9%** |
| `forest` s4·s5 (**S22 최악 0.148**) | 406 | 60 | 2 × 50 | 14후보 × 0.25 × 4.8 × 5 ≈ **84** | **650** | **9.2%** |
| `bog` s2 | 314 | 42 | 1 × 50 | ≈ 30 | **436** | **9.6%** |
| `finale` s6 | 448 | 60 | 2 × 50 | ≈ 60 | **668** | **9.0%** |

→ ★ **실측 지분 = ~9~10% (전 테마·전 스테이지).** 그리고 **정본 §8.10이 자기 손으로 계산한 `sea` s4 = 8.8%를 이 섹션이 8.9%로 재현한다**(차 0.1%p = 엘리트 모델의 차이).

> ★ **그러므로 정본 §8.10의 헤드라인 「실측 **12~13%** (04 §8.4)」는 이 섹션의 틀린 분모를 인용한 것이며, **정본 자신의 재검산(8.8%)과 갈라진다.** 같은 절 안에서 같은 양에 두 값이 인쇄돼 있다 → **C-8 위반이고 그 원인은 이 섹션이다.** → §14.3-**N-6**(minor). **게이트 무영향** — S22의 분모는 「저작 리스트 Σ XP」이고 지분(%)은 어느 게이트도 읽지 않는다.

> ★ **왜 v1.1의 47%가 중대했나 (정본 §13.2-⑤의 논거 — 여전히 유효하다)**: 새떼가 총 XP의 47%면 **위기 세션 하나가 잡몹 페이즈 95초 전체보다 큰 XP원**이 되어 정본 §8.8의 파밍 장치 4개(이탈 소멸 · 전멸 압축 · 고XP 도망자 · 중간보스 병행)가 **전부 무의미해진다** — 하단에 앉아 있다가 마지막 25초만 잘하면 되기 때문이다. **`farmXpRatio ≥ 2.0`(파밍 = 진짜 선택의 증명)이 이 한 값에 물려 있었다.**
> ★ **그리고 9~10%가 설계를 깨지 않는 이유 (정본 §8.10 v1.2의 답)**: 위기 세션의 페이오프는 **지분이 아니라 밀도가 나른다** — 「25초 안에 60기가 **전부 테마 속성으로**」 = **초당 2.4기 × 100% × ×2**는 잡몹 페이즈 어디에도 없는 밀도이고, **그 순간의 화면**이 페이오프다. **9%든 25%든 그 장면은 같다.** v1.1이 25%에 매달린 것은 **페이오프를 XP로 착각한 것**이다.

### 8.5 `crisisKillShareWithoutCapstone ≥ 0.80` 대비 (정본 §13.1.1 — 지표가 교체되었다)

★ **정본 v1.1이 `crisisClearWithoutCapstone`을 폐기하고 `crisisKillShareWithoutCapstone`으로 대체했다.** 근거: **`crisisFailCondition = 없음`(정본 §8.10)이므로 "클리어"가 정의 불가** = **재는 대상 자체가 존재하지 않았다.** 새 정의 —

| 항목 | 정본 §13.1.1의 확정 |
|---|---|
| `capstone` | 보유 무기에 **`nova` 또는 `aura`**가 있는 것 |
| 대상 세션 | capstone 미보유 **그리고** 그 위기 세션 동안 **폭탄을 쓰지 않은** 세션 |
| `killShare` | 그 세션에 처치한 새떼 수 ÷ `crisisTotal × swarmTotalScale[stage]` |
| 게이트 | 그런 세션들의 **중앙 `killShare` ≥ 0.80** |

| 해답 | 이 섹션의 편성에서 실제로 작동하는가 |
|---|---|
| `nova` / `aura` (capstone) | `arc`/`vWedge` = **밀집 편대** → 광역 1회가 서브웨이브 1파를 지운다 ✔ **단 이 게이트의 분모 밖이다** |
| **폭탄** | `bomb.mobDmg 9999` + 탄 전량 소거 → 1개로 화면의 새떼 전멸 ✔ **역시 분모 밖이다** |
| ★ **분모 안의 빌드** | capstone도 폭탄도 없는 순수 ST 빌드 — **이것이 게이트가 재는 유일한 대상** |

**밀도·화력 검산 (정본 §13.2-⑫의 모델에 이 섹션의 편성을 대입)**

```
새떼 총 HP (스테이지 6 최악):  swarmChaff.hp 3 × enemyHpScale[6] 6.0 = 18 HP/기 × 60기 = 1,080
capstone 없는 최악 빌드 = 순수 ST 4종의 군중 DPS ≈ 1,266
25 게임초 × 1,266 × uptimeRef 0.60 = 18,990 피해   vs   필요 1,080   →   17.6배
```
→ ★ **게이트는 화력이 아니라 「도달」의 시험이다.** 피해량은 17배 여유이므로 `killShare`를 결정하는 것은 **60기가 6파로 나뉘어 오는 것을 이동으로 쫓아갈 수 있는가**이다. 최악 빌드(`forward` = 정면 전용)의 `killShare` 추정 = **0.85~0.95** ✔

- ★ **개체 HP를 `swarmChaff.hp = 3`(chaff, `hpMult 1.0`)로 낮게 잡은 것이 이 게이트의 실물이다.** 스테이지 6(`enemyHpScale 6.0`)에서도 개체 HP는 **18** — **어떤 무기 조합이든 1~2히트**다.
- ★ **폭탄과 게이트는 겹치지 않는다** (정본 §13.2-⑫): 게이트는 **폭탄 없이도 되는가**를 묻고, 폭탄은 **그래도 안 되면**을 담당한다. 정본 §8.10이 폭탄을 「보편 해답」으로 든 것은 **게이트의 근거가 아니라 게이트를 우회하는 안전망**이다. 이전 판본이 폭탄을 이 게이트의 통과 근거로 든 것은 **분모를 잘못 읽은 것**이다.

---

## 9. 복합 보스 — 저작 규칙

### 9.1 정본이 확정한 것 (인용, 재정의 아님)

`boss.partCount` **4** (최종만 5) · `partRegen false` · **`summonsAllowed false`** · `partHitPriority "outermostFirst"` · **`core.hp`·`parts[].hp` 전부 절대값**(`hpShare` 없음) · `phaseThresholds [0.6, 0.3]`(**코어 HP** 기준) · `introSec 3.0` · `timerStartsAfterIntro` · 부위/보스 XP **0/0** · 부위는 개체이며 자체 `score`·상성 처치 보너스 대상 · `coin 12` / `partCoin 2` / `healDrop` 없음 · `bomb.bossDmgRatio`는 **코어 최대 HP 기준, 부위 미적용**. (정본 §8.11)
`partType` 4종 `mobility`(`speedPxSec × 0.5` **그리고 `ampPx → 0`**) / `armament`(그 부위 이미터 영구 삭제) / `armor`(코어 게이트 1단계) / `core`(사망). `onDestroy` **없음**. (정본 §8.12 · §8.12.1)
**코어 게이트** = `coreGateMul(0.4) ^ 살아있는 armor 수` → armor 3 = ×0.064 / 2 = ×0.16 / 1 = ×0.40 / 0 = ×1.00. **`mobility`·`armament`는 게이트에 영향 없음** = 3분 타이머 하의 진짜 트레이드오프. (정본 §8.13)
**R1~R7** — 전부 `check.mjs`(S5)가 강제. (정본 §8.14)

### 9.1.1 ★ 승복 — 「코어 HP = 보스 HP」를 이 섹션이 오독했고, HP는 armor에 있다

이전 판본은 정본 v1.0 §8.11의 「**코어 HP = 보스 HP**」·「**3분 타이머가 재는 것 = 코어 DPS**」를 **문면 그대로** 읽어 §9.6에 「`armor` 총합 = 코어 HP의 **20~50%**」를 저작했다. **정본 v1.1이 그 배분을 뒤집었다** (정본 §8.11 · §13.6 · §18.3-2 · §19.5-7). **산술이 이 섹션에 반대한다:**

```
코어는 R1에 의해 항상 노말 = 어떤 스탠스로도 ×1 = 속성 투자와 완전히 무관하다.
→ 속성 투자가 줄일 수 있는 시간은 오직 armor 부위의 시간뿐이다.
→ 코어에 HP를 몰수록 속성 투자의 레버리지가 죽는다.
→ 코어가 HP의 63%(이 섹션의 이전 배분)이면 무투자 빌드는 균형 빌드보다 겨우 24% 느리다
   → noElementPass가 어떤 bossHpScale로도 0.5 아래로 내려가지 않는다
   → 「후반으로 갈수록 속성 투자가 필수」가 산술적으로 성립 불가능해진다.
```

| | **확정 (정본 §8.13.1)** |
|---|---|
| **정정된 문장** | **3분 타이머가 재는 것 = `armor` 부위 파괴 + 코어 격파의 총 소요시간이며, 그 대부분은 `armor`다.** 코어는 **마무리**이지 시험이 아니다 — 시험은 관(棺)에 있다 |
| **`armorCoreRatio` φ** | **4.90** (a=2) — `armor` 총합 = 코어의 **490%**. 이전 판본의 20~50%와 **정확히 반대 방향** |
| **R7** | `φ ∈ [0.85·B, B)`, `B = 0.4^-a − 1` — **G-1의 뜻**: φ가 상한을 넘으면 armor가 코어 직행보다 비싸져서 **모든 빌드가 armor를 무시**한다 → **게이트가 게이트이기 위한 조건** |
| **산술적 천장** | `ρ = T_noEl/T_bal = (φ+1)/(φ/m+1) < m` — **속성 투자의 레버리지는 `m`(≈1.62)을 절대 넘을 수 없다.** 코어가 항상 노말이기 때문이다 |

> ★ **정본은 이것을 「정본 자신의 결함」으로 기록했다**(§18.4-1): *"04는 정본을 **정확히 읽고** armor를 코어의 20~50%로 저작했고 그 결과 `noElementPass`가 0.55~0.75에 고착됐다. **정본의 문장이 04를 그 배분으로 이끈 것이다.**"* — 그러나 **결과가 틀린 것은 이 문서**이고, 이 개정이 그것을 고친다. §9.5·§9.6·§10의 HP는 **전량 폐기**되었다.

★ **`partCount`는 코어를 포함한다.** 정본 §8.15가 강철 가오리(스러스터 + 좌 + 우 + 코어)를 "partCount = 4 ✔"로 검증했다 → **`parts[]` 배열 길이 = 3, 코어 1 = 합 4.** 최종은 `parts[]` 4 + 코어 1 = 5. **코어는 `parts[]`에 들어가지 않는다**(§9.8 스키마가 `core`를 별도 필드로 둔다).

★ **코어는 사격하지 않는다.** 정본 §9.8의 `core` 스키마에 `patternSet`이 없다 → **탄을 쏘는 것은 주변부 3개(최종 4개)뿐.** 이것이 §9.3의 텔레그래프 산술의 전제다. 그리고 "부위를 다 깨면 보스가 무장해제된다"가 규칙 없이 성립한다 — 마지막 국면이 항상 순수 DPS 레이스가 되는 이유.

### 9.2 부위 파괴 = 그 부위의 이미터 정지 (정본 §8.12 인용 — R-13 채택)

이전 판본이 "이 섹션은 정지로 **전제한다**"고 단서를 단 해석은 **정본이 확정**했다:

| 규칙 | 정본 §8.12의 확정 |
|---|---|
| 전 `partType` 공통 | **부위가 파괴되면 그 부위의 `patternSet` 이미터는 즉시 정지**하고 진행 중인 텔레그래프도 소멸한다. **이미 발사된 탄은 남는다** |
| `armament`의 "영구 삭제"가 뜻하는 것 | **`armament`의 *유일한* 효과가 이미터 상실이라는 뜻**이지, 다른 타입은 죽어도 쏜다는 뜻이 **아니었다** |
| 근거 | 정본 §7.6이 파괴 부위를 "외곽선 소멸 + `#3A3A3A` + 글리프 제거"로 규정 → **죽은 것이 쏘면 I-2 위반** |

★ **저작에의 귀결 (정본이 명시한 것)**: 부위 파괴 순서가 **보스의 탄막 밀도를 단조 감소**시킨다 → 후반부가 자동으로 관대해진다 = 3분 타이머 하의 "무장을 깰까" 트레이드오프에 **실물 보상**이 붙는다. → §10의 전 보스에서 **선택 부위(`armament`/`mobility`)를 깨는 것의 값**이 여기서 나온다.

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

> ★ **이 규칙이 "AI가 생성한 보스 스크립트가 트위치가 되는" 실패 모드를 산술로 봉인한다.** AI는 `everySec`·`offsetSec`을 자유 저작할 수 없고 위 3줄에서 파생한다 → **동시 텔레그래프 초과가 발생할 자유도가 존재하지 않는다.** 그리고 A층 `telegraphConcurrentMaxGlobal`(**80**, 정본 §12.1)·B층 `caps.telegraphs`(**96**) 안전망에 닿을 일도 없다(보스전 최대 = 2/부위 × 5부위 = **10**) → 보스전은 `capHits = 0`이 자명하다.
> ★ **부수효과: 보스전이 "1.5초 예고 → 한 부위가 쏜다"의 리듬으로 고정된다.** 페이즈가 올라가면 그 리듬이 6.0 → 4.5 → 3.6초로 조여든다. **읽는 부담은 그대로(항상 1~2개), 여유만 줄어든다** = 정본 §6.2의 "어려워지는 것은 읽고 반응할 시간이지 무엇인지 알아보는 것이 아니다"의 보스전 판본.

### 9.4 ★ 페이즈 파생 규칙 — 이미터 66종을 시드 22종에서 만든다

정본 §8.11: "페이즈마다 각 부위의 `patternSet` 인덱스가 교체된다(`everySec` 단축, `count`/`arcDeg` 증가). **새 이미터 타입이 생기지는 않는다.**" → 이 문장을 **빌드타임 생성 규칙**으로 못박는다.

```
부위마다 시드 이미터 1개(type · bulletId · 고유 파라미터)를 저작한다.
patternSet[p] 의 이미터 (p = 0,1,2):
  id                      = {bossId}{PartIdPascal}P{p+1}          (정본 §9.8.1 — S36)
  from                    = "part"                                (정본 §8.5 — S28)
  type                    = 시드 그대로 (★ 타입은 절대 안 바뀐다)
  bulletId                = 시드 그대로.  ★ 예외 1개: 스턴 부위의 p == 2 → "stunMark" (§10.5 — S13)
  telegraphSec            = 1.50 (고정)
  everySec                = [6.0, 4.5, 3.6][p]
  offsetSec               = everySec × partIndex / P
  repeat                  = 2                                     ★ v1.3 신설 (아래)
  restSec                 = everySec                              ★ v1.3 신설 (아래)
  count                   = round(seed.count × [1.0, 1.4, 1.8][p])
  arcDeg / rotStepDeg     = round(seed.value × [1.0, 1.15, 1.30][p])
  durationSec (spiral만)   = count × rateSec                       ★ v1.3 신설 (아래)
  그 외 파라미터           = 시드 그대로
```

- 7 보스 × 주변부 3(최종 4) = **시드 22개** → `patternSet` 전개 시 **이미터 66개**. 손으로 쓰는 것은 22개뿐이고 나머지는 규칙의 인스턴스다.
- **`speed`는 페이즈 스케일 대상이 아니다** — `fairness.maxBulletSpeed`(260)를 페이즈 스케일이 몰래 넘기는 사고를 원천 차단한다. 페이즈가 올리는 것은 **밀도(`count`)와 빈도(`everySec`)**뿐.
- 생성 규칙이므로 `data/bosses.json`에는 **전개된 결과**가 들어간다(로더는 규칙을 모른다 = 런타임 코드 0).

**★ `repeat` = 2 · `restSec` = `everySec` — 악절의 값 (v1.3 저작, 정본 §23-D4의 위임분)**

정본 v1.3 §8.5가 **「보스만 `repeat`/`restSec`를 쓴다」를 S30의 동치로 승격**시켰다(`repeat ≥ 2` ∧ `restSec > 0`). v1.2의 이 규칙 블록은 **세 필수 공통 키를 하나도 인쇄하지 않았다** → 「누락 키 = 에러」로 66개가 전부 로드 실패였다.

| | `everySec` | `repeat` | `restSec` | **유효 주기 `EP`** | 보스 전체의 사격 창 | **화력 창(정적)** |
|---|---|---|---|---|---|---|
| 페이즈 1 | 6.0 | 2 | 6.0 | **18.0** | 0 ~ 10.0 | **6.5초** |
| 페이즈 2 | 4.5 | 2 | 4.5 | **13.5** | 0 ~ 7.5 | **4.5초** |
| 페이즈 3 | 3.6 | 2 | 3.6 | **10.8** | 0 ~ 6.0 | **3.3초** |

- ★ **`EP = 3 × everySec`이 되어 §9.3의 S7 정적 증명이 한 자도 안 바뀌고 성립한다.** `restSec = everySec`이므로 발사 시각이 **`everySec` 격자의 부분집합**(격자 상 `0,1,3,4,6,7…`번 박자)이고, **부분집합은 동시 텔레그래프를 늘릴 수 없다.** 그리고 한 보스의 전 부위가 `repeat`·`restSec`를 공유하므로 **`EP`가 동일** → 정본 §8.5의 「유효 주기가 같으면 위상차가 영구히 유지된다」는 정합 조건을 만족한다. **전수 검산: 7보스 × 3페이즈 = 21셀 전부 최대 1~2 ≤ 2** ✔ (§9.3의 표를 재현)
- ★ **왜 `repeat`를 페이즈마다 올리지 않는가 (2로 고정)**: 페이즈는 **코어 HP**로만 갈린다(정본 §8.11 `phaseThresholds [0.6, 0.3]`)는데 **코어는 armor의 1/4.9**다 → **페이즈 2·3의 실제 길이는 각 ~10 게임초뿐**이다(정본 §13.6.2: 코어 976 × `bossRamp` ÷ 36 = 33.1초 → 그 30%씩). `repeat`를 2/3/4로 키우면 `EP`가 18.0으로 **고정**되어 ★ **페이즈 2·3에서 `restSec`가 영원히 도달 불가능**해진다 = **플레이어가 코어를 마무리할 창이 정확히 필요한 국면에서 사라진다.** `repeat` 2 고정은 `EP`를 **18.0 → 13.5 → 10.8로 함께 조여** 세 페이즈 **전부에서 악절이 완주**하게 만든다.
- **압박은 그래도 오른다 (탄 발생률 기준)**: 악절당 보스 사격 수는 6발로 같지만 `EP`가 줄고 `count`가 ×1.4/×1.8이므로 **탄 발생률 = ×1.87 / ×3.00**. 화력 창은 **6.5 → 4.5 → 3.3초**로 조여든다 → 정본 §8.5의 「몰아치고 → 쉰다」가 **읽는 부담은 그대로(항상 1~2개) 여유만 줄어든다**(§9.3)와 **같은 축**이다.
- **캡 검산**: 보스 페이즈 최대 동시 적 탄 = **78**(`thornKing` P3) ≪ A층 `maxSimultaneousEnemyBullets` **320** ≪ B층 `caps.enemyBullets` **384** → 보스전 `capHits = 0`이 여전히 자명하다.

**★ `durationSec` = `count × rateSec` — `spiral`의 삼중 필드 정합 (v1.3 저작)**

★ **v1.2의 이 규칙 블록은 `spiral`에서 자기모순이었다**: `count`를 ×1.4/×1.8로 키우면서 `durationSec`·`rateSec`를 **「그 외 = 시드 그대로」**로 묶었는데, ★ **저작된 `spiral` 시드 2개가 `count × rateSec == durationSec`를 정확히 만족한다** (`bogSpiral` 10 × 0.2 = 2.0 · `crownR` 8 × 0.25 = 2.0). 규칙대로 전개하면 `crownR` P2가 **11 × 0.25 = 2.75 ≠ 2.0**이 되어 **세 필드가 갈라진다** → 엔진이 「탄을 자르는가 시간을 늘리는가」를 **런타임에 결정**해야 한다 = **C-6(구현자의 즉석 결정) 위반.**
- **처분**: `rateSec`(= 나선의 결, 고유 파라미터)를 **시드에 고정**하고 `durationSec`를 **`count`에서 파생**시킨다. → **P1은 시드값을 정확히 재현**(2.0 = 8 × 0.25 = 10 × 0.2)하므로 **새 값 0**이고, 페이즈가 올리는 것은 여전히 **밀도뿐**이다.
- **귀결**: `crownR` **2.0 / 2.75 / 3.5** · `sac` **2.0 / 2.8 / 3.6**. ★ `sac` P3의 `durationSec`(3.6) == `everySec`(3.6) = **악절의 2박 동안 나선이 끊기지 않는다** — 늪주 페이즈 3이 이 게임에서 가장 압박이 센 국면이 되는 것은 **의도다**: `sac`는 `armament`이고 **8.1초면 끌 수 있으며**(§10.5), 그것이 정본 §8.12의 「`armament`의 페이오프」가 **런에서 가장 큰 값**이 되는 지점이다.

### 9.5 ★ 보스 HP 모델 — **정본 §13.6이 소유한다 (인용)**

> ★ **이전 판본의 §9.5(참조 화력 곡선)와 §9.6(HP 배분 규칙)은 전량 폐기되었다** (정본 §18.3-1·2·3 · §19.5-7). 이 섹션은 **HP를 정하지 않는다.** 아래는 정본의 인용이며, §10의 전 보스 표가 이 표에서 파생된다.

**모델 (정본 §8.11 · §13.6.1)**
```
실효 HP = hp × bossHpScale[stage]          // tier ∈ {stage, mid}
저작값  = 스테이지 1 기준 base
tier == "final" → bossHpScale 미적용, hp가 곧 실효 HP     ← 셔플되지 않으므로 규칙의 정의역 밖
```

**★ 테마 보스 6종의 HP = 전 6종 동일 (정본 §13.6.2 — ★ v1.2에서 ×1.20 재산출)**

| 부위 | **base hp** (v1.1 → **v1.2**) | 근거 (정본 소유) |
|---|---|---|
| `core` | 815 → **976** | `noElementPass` 목표 곡선에서 **역산**된 값 |
| `armor` ×2 | 1,997 → **2,391** 각 (총 **4,782**) | `armorCoreRatio` φ = **4.90** → `976 × 4.90 = 4,782` |
| 선택 부위 ×1 | 399 → **478** | ★ **`boss.optionalPartArmorRatio`** = **0.20** × `armor` **부위 1개** (= 부담 없이 포기 가능) |
| **소요 피해** (armor 2 + core) | 4,809 → **5,758** = `5.90 × core` | 선택 부위는 **소요 피해에 없다** — 순수 선택이므로 |
| ×1환산 총합 (선택 부위 포함) | 5,208 → **6,236** | s5 실효 = `6,236 × 14.42` = **89,923** |

★ **왜 ×1.20인가 — 바뀐 것은 앵커 하나다 (정본 §13.6.2)**: `killTime(i,m) = 소요피해 × bossRamp[i] / (dpsRef[1] × 0.60 × m)`에서 **`dpsRef`가 정확히 상쇄되므로**, farm 정책이 `maxFarm`으로 확정되며 바뀐 것은 **스테이지 1 앵커 `dpsRef[1] × 0.60`이 30 → 36**이 된 것뿐이다 → **HP 전량 × (36/30) = ×1.20.** **φ·`bossRamp`·`noElementPass` 곡선·구조는 한 자리도 안 바뀌었다.**

★ **`optionalPartCoreRatio` → `optionalPartArmorRatio` 개명 (정본 §13.6.4 · §21-A11, 판정 1) — 이 섹션의 값은 안 바뀐다**

| | |
|---|---|
| 무엇이 틀렸나 | v1.1의 **키 이름**(`×core`)과 §9.4의 **인쇄값 0.49**(`×core`)가, **표·S24·§13.6.2·이 섹션**이 뜻하는 **0.20(`×armor` 부위 1개)**과 달랐다. 두 해석은 **a=2에서만 우연히 일치**한다(`0.20 × 4.90/2 = 0.49`) |
| 어디서 갈라지나 | **테트라크에서 86%**: `0.20 × 25,800 = 5,160 ≈ 저작 5,200` ✔ vs `0.49 × 5,700 = 2,793` ✗ |
| 왜 blocker였나 | ★ **S24는 armor 기준으로 읽는다** → 0.49를 `rules.json`에 넣으면 `2,391 × 0.49 = 1,171.6` vs 저작 **478** → **`check.mjs`가 보스 7종 전부를 로드 실패**시킨다 |
| 이 섹션의 처분 | ★ **인용 정정만.** 이 섹션은 v1.1에서도 **0.20만 인용했고 §9.4의 0.49를 보지 못했다** — 정본이 **값이 아니라 이름을 고쳤으므로** 이 섹션의 HP 배분은 불변이다(비율 0.20 유지, 값은 ×1.20 사슬로 399 → 478) |

★ **전 6종이 같은 HP인 것은 게으름이 아니라 분업이다**: **스테이지 축은 `bossHpScale`이, 구조 축은 저작(부위 속성·이미터·`movePattern`)이** 담당한다. 보스마다 HP를 다르게 하면 **테마 셔플 하에서 "어느 보스가 몇 번째로 오나"가 운이 된다** — 정본 §13.2-⑨(`noDeadLuck.worstThemeOrder`)가 성립하는 이유가 이 균일성이다.

**★ 역산의 재현 (정본 §13.6.2의 산술을 이 섹션이 검산한다 — ★ v1.2: 앵커 30 → 36)**
```
통과 모델: killTime T에 대해 pass(T) = P(u ≥ T/300),  u = 실효가동률 ~ N(0.60, 0.05)
목표 (사용자 확정): noElementPass = [0.90, 0.90, 0.50, 0.50, 0.15, 0.15]
  → T_noEl 목표 = [160.0, 160.0, 180.0, 180.0, 195.5, 195.5]

★ 핵심 상쇄 (정본 §13.6.1):
  killTime(i) = 소요피해 × bossHpScale[i] / (dpsRef[i] × 0.60 × m)
              = 소요피해 × bossRamp[i] / (dpsRef[1] × 0.60 × m)      ← dpsRef가 정확히 상쇄된다
              = 소요피해 × bossRamp[i] / (36 × m)                     ← ★ v1.2: dpsRef[1] 60 × 0.60 = 36
T_noEl(i) = 5.90 × core × bossRamp[i] / 36
  i=1: 5.90 × core × 1.000 / 36 = 160.0  →  core = 976.3
  i=3: 5.90 × core × 1.125 / 36 = 180.0  →  core = 976.3
  i=5: 5.90 × core × 1.222 / 36 = 195.5  →  core = 976.2
  → core = 976  (세 방정식이 ★ ±0.02%로 일치 — bossRamp가 곡선을 정확히 실현한다)
```
- ★ **v1.2의 역산이 v1.1보다 정확하다** (±0.3% → ±0.02%): v1.1의 s3/s4 이탈(`noElementPass` 0.478 vs 0.515)은 **`bossHpScale`을 소수 1자리로 반올림한 데서 온 인공물**이었고(4.4 vs 정확값 4.3875 = +0.28%), v1.2는 **2자리로 인쇄**해 그것을 없앴다 → 곡선이 **0.909/0.909/0.500/0.499/0.151**로 사용자 확정값을 더 정확히 재현한다(정본 §13.6.2).

**★ 이전 판본의 「520 vs 708」이 실은 무엇이었나 (정본 §10.4.3 · §13.5)**

| | |
|---|---|
| 이 섹션의 520 | **실효** DPS였다 (`uptimeRef`를 아예 적용하지 않았다: `83,950 ÷ 520 = 161s`) |
| 03의 708 | **명목** DPS였다 (`180 × 708 × 0.6`) |
| ★ 진실 | **`708 × 0.60 = 425`와 `520`은 같은 종류의 수이며 1.22배 차이다.** 두 문서는 **다른 단위로 같은 것을 말하며 그것을 부등호로 이었다** |
| 처분 | **`uptimeRef`(0.60)를 정본이 소유**한다(§10.4.3) → **재발 불가.** 그리고 사용자의 「대략 ×1.36」은 두 수가 같은 단위라는 전제의 비율이었으므로, 정본은 **비율이 아니라 목표 격파시간에서 역산**했다 |
| 결과 | 테트라크 ×1환산 총 HP = **83,100 vs 이 섹션의 83,950 = 1% 차이.** ★ **총량은 그대로였고 바뀐 것은 배분(핵 63% → 7%)과 스테이지 곡선이다** |

### 9.6 저작 계약 (정본 §13.6.4 — 「보스」 섹션이 따라야 하는 것)

| 규칙 | 값 | 소유 |
|---|---|---|
| `core.hp` | ★ **976** (테마 보스 전 6종) / **5,700** (`tetrarch`, 절대 — **v1.2에서 불변**) | **정본 §13.6.4** |
| `armorCoreRatio` φ | **4.90** (a=2) / **13.58** (a=3). R7 밴드 `[0.85·B, B)`, `B = 0.4^-a − 1` | **정본 §8.13.1** |
| 각 `armor` 부위 hp | `core.hp × φ ÷ armorCount` = ★ **2,391** (테마 보스) — ★ **armor 부위끼리 균등**(비대칭이면 특화 짝 6종의 대칭이 깨진다) | **정본 §13.6.4** |
| ★ **`boss.optionalPartArmorRatio`** (v1.2 개명) | **0.20** — 선택 부위 hp = **`armor` 부위 1개 hp × 0.20** = ★ **478** (테마 보스) / 왕좌 **5,200** ≈ `25,800 × 0.20` | **정본 §13.6.4** |
| 목표 격파 시간 | ★ **균형(probe) = `killTimeMedianBalanced` ∈ [120, 150]** (v1.2 개정) → `flow.stagePar[i]` = `180 − 그것` = **30~60** | **정본 §13.6.4 · §13.6.5** |
| `contactDmg` | 코어 **16** / 부위 **14~15** | 정본 §2.5 저작 가이드 |
| `radius` · `anchor` · `shapeId` · `score` · 시드 이미터 | **이 섹션이 저작** (정본은 HP와 구조만 소유) | **04** |

★ **밸런싱 손잡이 — v1.2가 §13.6.4의 「유일한 손잡이」를 스스로 뒤집었다 (정본 §13.6.5, 이 섹션은 어느 것도 만지지 않는다)**

| 축 | 손잡이 | 주의 |
|---|---|---|
| `noElementPass` | **`bossRamp` → `bossHpScale`** | 이 축에서는 유일한 손잡이가 맞다 |
| ★ **`killTime`** | ★ **`m` 궤적 = `categoryWeights.elementLevel`(20)** | ★ **`bossHpScale`은 쓸 수 없다** — `noElementPass`와 `killTime`을 **같은 HP로 동시에** 움직이므로 **둘을 독립으로 못 맞춘다** |
| 최후 | `armorCoreRatio`(R7 밴드 내) → `elementCapTotal`(= 정본 개정) | **`.js`를 여는 경우는 없다** |

> ★ **그 결합의 귀결을 이 섹션이 알아야 하는 이유** (정본 §13.6.5): `noElementPass[3] = 0.50`이 `T_noEl(3) = 180.0`을 고정하므로 **balanced-farm 플레이어의 스테이지 3 격파가 2.71분(중앙값 2.55분)으로 강제된다** — 사용자 요구 「~2~2.5분」의 상단을 2% 넘는다. 정본은 **사용자 확정 곡선(설계의 척추)을 보존하고 물결표를 초과하는 쪽**을 택했다. **이 섹션의 보스 HP는 그 선택의 산물이며, 되돌리려면 손댈 곳은 이 섹션이 아니라 드래프트 가중치다.**

★ **산술적 천장 (정본 §8.13.1 G-3 · §13.6.5 — 이 섹션이 아무리 저작해도 넘을 수 없다)**: `ρ = T_noEl/T_bal = (φ+1)/(φ/m+1) < m`. **속성 투자의 레버리지는 `m`(≈1.62)을 절대 넘을 수 없다** — 코어가 R1에 의해 항상 노말이기 때문이다.
★ **`check.mjs` S24**가 `Σ(armor hp) == core.hp × armorCoreRatio`(±1%)와 `선택 부위 hp == armor 1개 × ★ boss.optionalPartArmorRatio`(±5%)를 **정적으로 강제**한다 → 이 섹션이 HP를 임의로 적으면 **로드 실패**한다.

---

## 10. 보스 7종 (`data/bosses.json`)

> **테마 6 + 최종 1 = 7종 저작 / 런당 등장 6종**(5 추첨 + 최종). ★ 정본 §13.6·§8.12의 표기가 이렇게 정정되었다(R-16 채택).
> **HP는 정본 §13.6.2·§13.6.3이 소유**하며 아래 표는 그것을 **인용**한다. 테마 보스는 **스테이지 1 기준 base**, `tetrarch`는 **절대값**(§9.5).

### 10.0 ★ armor 1 보스 3종의 재구성 (`armorPartCountRange` `[1,2]` → **`[2,2]`**)

정본 §8.13.2가 **`tier: "stage"` 보스의 `armor` 수를 정확히 2로 확정**했다(최종만 3). → 이전 판본의 armor 1 보스 3종(`kiln`·`scarab`·`mire`)은 **선택 부위 하나를 `armor`로 승격**해 재구성한다(정본 §18.3-4).

| 보스 | 이전 (armor 1) | **v1.1 (armor 2)** |
|---|---|---|
| `kiln` | `turretL` armament/불 · **`turretR` armament/풀** · `plate` armor/물 | `turret` armament/불 · ★ **`vent` armor/풀** · `plate` armor/물 |
| `scarab` | `carapace` armor/물 · `legs` mobility/불 · **`stinger` armament/풀** | `carapace` armor/물 · `legs` mobility/불 · ★ **`stinger` armor/풀** |
| `mire` | `shell` armor/불 · **`tendril` mobility/물** · `sac` armament/풀 | `shell` armor/불 · ★ **`tendril` armor/물** · `sac` armament/풀 |

**★ 승복 — armor 1은 두 게이트를 충돌시킨다 (정본 §8.13.2의 산술)**

이전 판본은 `kiln`을 "**코어 직행이 성립하는 유일한 보스**"로 세워 하한 1의 존재 이유를 증명했다. **정본의 판정: 논거는 옳았고 목표는 v1.1이 더 잘 달성한다.** 반대 산술:

1. ★ **`specialistPass`와 `noElementPass`가 같은 측정이 된다.** `4/2/0` 특화는 3속성 중 2개만 투자한다. `armor`가 **1개**뿐이면 그 부위의 카운터가 미투자 속성일 확률 = **1/3**이고, 그 경우 특화 빌드는 그 보스에 대해 **문자 그대로 무투자 빌드**다(유일한 게이트 경로에 ×1). → `specialistPass ≤ 2/3 + 1/3 × noElementPass`. s5에서 `noElementPass ≈ 0.15`를 목표하면 **`specialistPass ≤ 0.72` < 게이트 0.85** → 직접 충돌. 실측: armor-1 보스만으로 채우면 s5 특화 통과율 **0.70**.
2. **R3의 정신을 게이트 경로에서 위반한다.** R3(`partElementDistinctMin = 2`)의 목적은 "스탠스 1개 샌드백 금지"인데, armor 1이면 **코어로 가는 유일한 문이 단일 속성**이다.
3. **ρ 천장이 1.287로 낮다** → armor-1 보스가 로스터의 절반이면 s5의 `noElementPass` 평균이 **0.49**가 되어 「후반엔 속성 투자가 필수」가 **절반의 보스에서 거짓**이 된다.

★ **armor 2 + R3 + R5의 결합이 만드는 보장 (정본 §8.13.2의 발견)**
```
R5: armor 부위 속성 ≠ 테마 속성 · 투자 가능 속성은 3종
→ armor 2개의 속성은 「테마가 아닌 2종」으로 사실상 결정된다
→ armor 2개의 카운터 2종은 서로 다르다 (R3 distinct ≥ 2 + 상성이 순환)
→ 4/2/0 특화는 3속성 중 2종에 투자하므로 armor 2개 중 최소 1개는 반드시 커버한다
→ ★ 특화의 전멸(total whiff)이 구조적으로 불가능하고, 특화의 대가는 항상 「부분」이다
```

### 10.1 ★ R1~R7 + `partCount` + S18 + S24 전수 검증표 (S5)

**구조 (전 테마 보스 6종 공통, 정본 §13.6.2)**: `armor` 2 + 선택 부위 1(`mobility` 또는 `armament`) + `core` 1 = **`partCount` 4.**

| 보스 | 테마 | `armor` #1 | `armor` #2 | 선택 부위 | **R1** | **R2** | **R3** | **R4** | **R5** | **R6** | **R7** φ | `partCount` | **S18** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`manta`** 강철 가오리 | `sea`/물 | `finL` 불 | `finR` 풀 | `thruster` **mobility**/물 | ✔ | ✔ | **3** ✔ | 물 **1** ✔ | 불,풀 ≠ 물 ✔ | **2** ✔ | 4.90 ∈ [4.46, 5.25) ✔ | 3+1 = **4** ✔ | `sway` → mobility 허용 ✔ |
| **`frostCrown`** 빙관 | `glacier`/물 | `crownL` 풀 | `crownR` 불 | `pylon` **armament**/불 | ✔ | ✔ | **2** ✔ | 물 **0** ✔ | 풀,불 ≠ 물 ✔ | **2** ✔ | 4.90 ✔ | **4** ✔ | `sway` · mobility 없음 ✔ |
| **`kiln`** 용광로 거신 | `volcano`/불 | `plate` 물 | ★ `vent` 풀 | `turret` **armament**/불 | ✔ | ✔ | **3** ✔ | 불 **1** ✔ | 물,풀 ≠ 불 ✔ | **2** ✔ | 4.90 ✔ | **4** ✔ | ★ **`holdCenter` → mobility 부위 없음** ✔ |
| **`scarab`** 태양갑충 | `desert`/불 | `carapace` 물 | ★ `stinger` 풀 | `legs` **mobility**/불 | ✔ | ✔ | **3** ✔ | 불 **1** ✔ | 물,풀 ≠ 불 ✔ | **2** ✔ | 4.90 ✔ | **4** ✔ | `orbitArc` → mobility 허용 ✔ |
| **`thornKing`** 가시왕 | `forest`/풀 | `podL` 물 | `podR` 불 | `bloom` **armament**/풀 | ✔ | ✔ | **3** ✔ | 풀 **1** ✔ | 물,불 ≠ 풀 ✔ | **2** ✔ | 4.90 ✔ | **4** ✔ | `sway` · mobility 없음 ✔ |
| **`mire`** 늪주 | `bog`/풀 | `shell` 불 | ★ `tendril` 물 | `sac` **armament**/풀 | ✔ | ✔ | **3** ✔ | 풀 **1** ✔ | 불,물 ≠ 풀 ✔ | **2** ✔ | 4.90 ✔ | **4** ✔ | ★ **`holdCenter` → mobility 부위 없음** ✔ |
| **`tetrarch`** 테트라크 | **없음**(`null`) | 관 3개 (물·불·풀) | — | `throne` **armament**/**노말** | ✔ | **면제**(`allowNormalPeripheral`) | **4** ✔ | **면제**(`exemptRules`) | 테마 없음 → **공허참** ✔ | **면제**(`exemptRules`, `armorPartCount 3`) | **13.58** ∈ [12.43, 14.625) ✔ | 4+1 = **5** ✔ | `holdCenter` · mobility 없음 ✔ |

★ **`tetrarch`의 R6 위반은 정본이 닫았다 (R-2 채택)**: `finale.exemptRules = ["R4"]` → **`["R4", "R6"]`**(정본 §8.16). 이전 판본의 진단("문면 그대로면 `check.mjs`가 최종 보스를 로드 실패시킨다")은 **정확했고 정본이 자기 결함으로 수용**했다.
★ **R7은 면제하지 않는다** — R7은 `a`의 함수이므로 **a=3에서 자동 재평가**된다(상한 `0.4^-3 − 1` = 14.625) → **최종 보스도 "게이트가 게이트인가"의 검사를 받는다.** R6은 **개수**의 규칙(예외 가능), R7은 **비율**의 규칙(예외 불필요).

**★ S24 검산 (HP 배분의 정적 무결성)**

| 보스 | `Σ(armor hp)` | `core.hp × φ` | 오차 | 선택 부위 | `armor 1개 × 0.20` | 오차 |
|---|---|---|---|---|---|---|
| 테마 보스 6종 | **2,391** × 2 = **4,782** | **976** × 4.90 = **4,782.4** | **−0.01%** ✔ | **478** | 2,391 × 0.20 = **478.2** | **−0.04%** ✔ |
| `tetrarch` | 25,800 × 3 = **77,400** | 5,700 × 13.58 = **77,406** | **−0.01%** ✔ | **5,200** | 25,800 × 0.20 = **5,160** | **+0.78%** ✔ |

→ 전 7종이 S24의 ±1% / ±5% 안 ✔ (★ v1.2에서 테마 보스의 오차가 **−0.10% → −0.04%**로 줄었다 — `976 × 4.90 / 2 = 2,391.2`가 반올림에 더 가깝게 떨어진다)
★ **S24가 읽는 비율은 `boss.optionalPartArmorRatio`(0.20, `×armor`)다** — v1.1의 인쇄값 0.49(`×core`)를 넣으면 `2,391 × 0.49 = 1,171.6` vs 저작 478 → **이 표의 7행 전부가 로드 실패**한다(정본 §21-A11).

**★ 구조 다양성 (`dominance.maxThemeClearStddev ≤ 0.06` 대비) — 근거가 바뀌었다**

이전 판본은 "armor 2 보스 3종 / armor 1 보스 3종이 속성당 1:1로 갈린다"를 근거로 삼았다. **그 축은 사라졌다**(전부 armor 2). **새 근거 = 선택 부위의 타입과 armor 속성쌍의 대칭:**

| 축 | 분포 |
|---|---|
| 선택 부위 타입 | `mobility` **2종**(`manta`·`scarab`) / `armament` **4종**(`frostCrown`·`kiln`·`thornKing`·`mire`) |
| `movePattern` | `sway` 3 / `orbitArc` 1 / `holdCenter` 2 |
| ★ **armor 속성쌍** | `{불,풀}` 2회(`manta`·`kiln`) · `{풀,불}` — `frostCrown`, `{물,풀}` 2회(`kiln`·`scarab`), `{물,불}` 2회(`thornKing`) … → ★ **3속성이 armor로 등장하는 횟수가 정확히 4회씩 균등**(물 4 · 불 4 · 풀 4 = 12 = 6보스 × 2) |

→ ★ **`maxElementWinShare`(균등 0.333, 상한 0.42)가 구조적으로 보장된다** — 정본 §13.2-⑩이 "**armor 속성쌍이 3종 전부 정확히 2보스씩**"을 근거로 든 것의 실물이 이 표다. **비대칭의 원천이 존재하지 않는다.**

### 10.2 「강철 가오리 (Manta)」 — `sea` / 물 — **정본 §8.15 판본**

| 부위 | `partType` | `element` | **정답 스탠스** | `hp` (base) | `radius` | `anchor` | `shapeId` | `contactDmg` | `score` | 파괴 효과 |
|---|---|---|---|---|---|---|---|---|---|---|
| `thruster` 스러스터 | `mobility` | **물** | 풀(R) | **478** | 22 | `[0,44]` | `fin` | 14 | 1200 | `speedPxSec × 0.5` **그리고 `ampPx → 0`** = 좌우 왕복 정지 |
| `finL` 좌 지느러미 | `armor` | 불 | **물(E)** | **2,391** | 26 | `[-52,10]` | `fin` | 15 | 1500 | 코어 게이트 1/2 (×0.16 → ×0.40) |
| `finR` 우 지느러미 | `armor` | 풀 | **불(W)** | **2,391** | 26 | `[52,10]` | `fin` | 15 | 1500 | 코어 게이트 2/2 (×0.40 → ×1.00) |
| `core` 몸체 | `core` | 노말 | 아무거나 | **976** | 42 | `[0,0]` | `bulb` | 16 | 8000 | **사망** |

`movePattern: "sway"`, `movePatternParams: {"speedPxSec":34,"ampPx":150,"yHoldPx":170}`, `armorCoreRatio: 4.90`

> ★ **HP는 정본 §13.6.2의 표가 유일한 소유자다** (★ v1.2: `core` **976** / `finL` **2,391** / `finR` **2,391** / `thruster` **478**). v1.1의 815/1,997/399도, 그 이전의 `core.hp 4000` · `finL/finR/thruster 900`도 **전량 폐기**다. 그리고 이전 판본이 "정본 §9.8이 직접 저작한 값이며 그대로 인용했다"고 적은 것은 **C-7에 의해 무효** — §9.8의 블록은 **전부 예시**다. `radius`·`anchor`·`shapeId`·`score`·`contactDmg`는 **이 섹션의 저작값**으로 남는다(정본 §13.6.4의 위임).
> **HP 배분이 뒤집힌 것에 주목하라**: 코어 4000 + 부위 2700(코어의 68%) → **코어 976 + armor 4,782(코어의 490%)**. ×1환산 총합은 6,700 → **6,236**이지만 **`bossHpScale`이 스테이지 축을 따로 담당**하므로 실효 총량은 오히려 커진다(s5: `6,236 × 14.42` = **89,923**).

**시드 이미터** (§9.4가 3페이즈로 전개)

| 부위 | 시드 `type` | `bulletId` | 시드 `count` | 고유 |
|---|---|---|---|---|
| `thruster` | `straight` | `pelletS` | 3 | `spreadDeg:26, speed:150` |
| `finL` | `fan` | `fanShard` | 5 | `arcDeg:70, speed:130` |
| `finR` | `aimed` | `heavyRound` | 2 | `spreadDeg:18, speed:115, leadSec:0.4` |

**★ 산술 검증 — 정본 §13.6.2의 상쇄식을 이 보스에 적용** (★ v1.2: `killTime = 소요피해 × bossRamp[i] / (**36** × m)`, **probe = maxFarm 기준**)

| 경로 | 스테이지 1 (`bossRamp` 1.00) | 스테이지 5 (`bossRamp` 1.222) | 판정 |
|---|---|---|---|
| **균형 (2/2/2)**, `m` = 1.35 / 1.62 | `4782/1.35 + 976` = 4518 → **125.5s** | `4782/1.62 + 976` = 3928 → **133.3s** | ✔ `killTimeMedianBalanced ∈ [120,150]` |
| **무투자** (`m` = 1.0) | `4782 + 976` = 5758 → **160.0s** | 5758 × 1.222/36 → **195.5s** | `noElementPass` **0.909** → **0.151** ★ |
| **코어 직행** (armor 2, ×0.16) | `976 × 6.25` = 6,100 → **169.4s** | 169.4 × 1.222 → **207.1s** | ★ **s1 성립(≤180) / s5 불가** |
| `thruster` 추가 비용 (풀 스탠스 ×2) | `478/2.0` = 239 → **6.6s** | 6.6 × 1.222 → **8.1s** | 순수 선택 |

> ★ **v1.2에서 정본과의 ±0.5% 차이가 ±0.1초로 줄었다 (v1.1의 자진 신고가 닫혔다 — 은폐하지 않고 정확히 보고한다)**: v1.1은 정본의 s5 인쇄값(196.6 / 208.3 / 134.1)과 이 섹션의 재산출(195.9 / 207.5 / 133.6)이 **±0.5% 갈렸다**고 신고했고, 원인을 **`core`의 반올림**(역산 세 방정식이 817.6/813.6/813.4로 흩어짐)으로 진단했다. **v1.2의 역산은 976.3/976.3/976.2로 ±0.02%에 모인다** → 이 섹션의 독립 재산출이 정본 §13.6.2의 인쇄값을 **전 항목 ±0.1초 안에서 재현**한다:
> 
> | | s1 | s3 | s4 | s5 | 최악 특화 짝 | 6짝 평균 |
> |---|---|---|---|---|---|---|
> | **정본 §13.6.2** | 125.5 | 135.1 | 128.3 | 133.3 | 0.851 | 0.935 |
> | **이 섹션의 재산출** | 125.5 | **135.0** | **128.2** | 133.3 | **0.851** | **0.935** |
> 
> **잔차 ±0.1초의 원인 = `core`를 976으로 확정한 것**(정확값 976.2~976.3)이며 **`pass`로는 0.000~0.003**이다 — v1.1의 잔차(±0.7초 / pass 0.011)의 **1/7**이다. ★ **원인은 정본이 `bossHpScale`을 소수 2자리로 인쇄한 것**이다(4.4 → 4.67) — v1.1의 반올림 인공물이 실제로 있었고 v1.2가 그것을 지웠다. **정본의 표가 소유자이므로 인쇄값을 따른다**(C-8).

**★ 특화 `4/2/0` 6짝 전수 (스테이지 5, `bossRamp` 1.222)** — `finL` 카운터 = 물, `finR` 카운터 = 불. 투자 4 → `m` 2.0 / 투자 2 → 1.62 / 투자 0 → 1.0

| (primary 4, secondary 2) | `finL` `m` | `finR` `m` | 계산 | killTime | pass |
|---|---|---|---|---|---|
| (물4, 불2) · (불4, 물2) | 2.0 / 1.62 | 1.62 / 2.0 | `2391/2.0 + 2391/1.62 + 976` = 3,647 | **123.8s** | ~1.000 |
| (물4, 풀2) · (불4, 풀2) | 2.0 / 1.0 | 1.0 / 2.0 | `2391/2.0 + 2391/1.0 + 976` = 4,563 | **154.9s** | 0.953 |
| (풀4, 물2) · (풀4, 불2) | 1.62 / 1.0 | 1.0 / 1.62 | `2391/1.62 + 2391/1.0 + 976` = 4,843 | **164.4s** | **0.851** |
| **평균** | | | | **147.7s** | ★ **0.935** ≥ 0.85 ✔ |

> ★ **정본 §13.6.2의 `specialistPass` s5 = 「6짝 평균 0.935 · 최악 짝 0.851」을 이 섹션이 소수점까지 재현한다** ✔ — `m`의 세 값(2.0 / 1.62 / 1.0)이 §4.3의 부여 규칙(투자 N → 슬롯 1..N)과 §13.6.2의 균형 `m` 궤적에서 **동시에** 나오므로 두 문서가 같은 모델을 쓴다.
> ★ **최악 짝 0.851은 razor다** (게이트 0.85, 여유 **0.1%p**). 정본 §13.2.1이 이것을 razor 3건 중 하나로 명시했고 **손잡이 = `armorCoreRatio`(R7 밴드 4.46~5.25 안에서 φ↓)**를 지정했다. ★ **이 섹션은 φ를 만지지 않는다** — 그러나 **최악 짝이 `(풀4, 물2)`/`(풀4, 불2)`, 즉 「테마 속성(물)의 카운터인 풀에 4를 몰고 온 빌드」**라는 것은 이 섹션의 저작이 만든 사실이다: 바다에서 **풀+4는 웨이브 70%를 ×2로 녹이는 최적해**인데 **보스에서는 최악 짝**이 된다. → **「잡몹의 정답이 보스의 오답」이 곧 R4+R5가 만들려던 장면**이며, 그것이 razor에 서 있는 것은 **의도의 실현**이다(정본 §8.14).

- ★ **특화의 전멸이 구조적으로 불가능하다** — `finL`(불)·`finR`(풀)의 카운터는 **물·불로 서로 다르고**(R3 + 상성 순환), `4/2/0`은 3속성 중 2종을 갖는다 → **최악의 짝(풀4, 물2/불2)조차 한 부위는 반드시 덮는다.** 이것이 정본 §8.13.2가 `armorPartCountRange`를 `[2,2]`로 좁힌 이유의 실물이다. armor 1이었으면 1/3 확률로 **전멸**이었다.
- ★ **커버리지 시험**: 바다를 **물+4**로 밀고 온 특화 빌드는 `finL`(불)을 ×2로 녹이고 `finR`(풀)에서 막힌다(정본 §8.15). **불에 2만 투자했어도 즉시 ×1.62로 보상**받는다 — `elementCapTotal = 6`(정본 §4.2)의 `4/2/0` 다이얼이 여기서 값을 갖는다.
- `thruster`(물)는 **순수 선택**: 풀 스탠스로 8.1초를 써서 깨면 보스가 **왕복을 멈추고 제자리에 선다**(`ampPx → 0`, 정본 §8.12.1) → 남은 시간이 편해진다. **깨지 않아도 코어 게이트에 영향 없음**(정본 §8.13).
- ★ **코어 직행이 스테이지 1~2에서만 성립하는 것은 버그가 아니라 「소프트 게이트」 그 자체다**(정본 §13.6.2): 169.4s ≤ 180 → 가능하되 **armor를 깨는 것(160.0s)보다 6% 손해.** **무투자 빌드조차 armor를 깨는 것이 이득**이고(R7의 G-1), 스테이지 3부터는 직행이 **물리적으로 불가능**해진다(190.7s > 180).

### 10.3 「용광로 거신 (Kiln)」 — `volcano` / 불 — **armor 2로 재구성**

| 부위 | `partType` | `element` | **정답 스탠스** | `hp` (base) | `radius` | `anchor` | `shapeId` | `contactDmg` | `score` | 파괴 효과 |
|---|---|---|---|---|---|---|---|---|---|---|
| `turret` 포탑 | `armament` | **불** | 물(E) | **478** | 24 | `[0,-40]` | `slab` | 14 | 1300 | **`laser` 이미터 영구 삭제** |
| `plate` 흉갑 | `armor` | 물 | **풀(R)** | **2,391** | 30 | `[-52,20]` | `ring` | 15 | 2400 | 코어 게이트 1/2 |
| ★ `vent` 배기구 | `armor` | 풀 | **불(W)** | **2,391** | 30 | `[52,20]` | `ring` | 15 | 2400 | 코어 게이트 2/2 · **용암 `zone` 이미터 소멸**(§9.2) |
| `core` 노심 | `core` | 노말 | 아무거나 | **976** | 40 | `[0,0]` | `bulb` | 16 | 8000 | **사망** |

`movePattern: "holdCenter"`, `movePatternParams: {"speedPxSec":0,"ampPx":40,"yHoldPx":150}`, `armorCoreRatio: 4.90`
★ **S18 준수**: `holdCenter`이므로 **`mobility` 부위를 가질 수 없다** — 이 보스는 갖지 않는다 ✔ (정본 §8.12.1: `ampPx`가 애초에 작아 파괴 효과가 무의미한 부위 = **트레이드오프가 거짓인 부위**)

**시드 이미터**: `turret` = `laser`/`beamCore`/`widthPx:20, angleDeg:90, activeSec:0.6, trackDuringCharge:true` · `plate` = `ring`/`heavyRound`/`count:10, speed:115, rotOffsetDeg:18` · `vent` = `zone`/`null`/`radius:76, activeSec:3.0, dmg:10`

**산술 검증** (전 6종이 같은 HP이므로 §10.2의 표와 **완전히 동일**하다 — 다른 것은 속성 배치뿐)

| 경로 | 스테이지 1 | 스테이지 5 | 판정 |
|---|---|---|---|
| **균형 (2/2/2)** | **125.5s** | **133.3s** | ✔ |
| **무투자** | **160.0s** | **195.5s** | `noElementPass` 0.909 → 0.151 |
| **코어 직행** (×0.16) | **169.4s** ✔ 성립(손해) | **207.1s** ✗ 불가 | ★ 아래 |
| `turret` 추가 비용 (물 스탠스 ×2) | **6.6s** | **8.1s** | 순수 선택 |

★ **승복 — 「코어 직행이 성립하는 유일한 보스」 서사는 폐기한다** (정본 §18.3-5)

이전 판본은 `kiln`을 armor 1로 세워 "6개 보스 중 3개만 갖는 자유"로 소프트 게이트를 증명했다. **정본의 판정: 목표는 옳고 v1.1이 더 잘 달성한다.**

| | 이전 (armor 1의 특권) | **v1.1 (진행의 곡선)** |
|---|---|---|
| 코어 직행 | `kiln`·`scarab`·`mire`에서만 성립 | ★ **전 보스에서 스테이지 1~2에 성립(169.4s ≤ 180), 스테이지 3부터 불가(190.7s)** |
| 무엇의 함수인가 | **보스의 개성** (6종 중 3종의 특권) | ★ **진행의 곡선** (누구나 초반엔 되고 후반엔 안 된다) |
| 왜 후자가 나은가 | — | ① **학습이 이월된다** ② 테마 셔플 하에서 "그 특권 보스가 몇 번째로 오는가"라는 **운이 사라진다** ③ 「후반으로 갈수록 속성 투자가 필수」와 **같은 문장**이 된다 |

> ★ **"때릴 수는 있다. 하지만 손해다"가 초반이고, "산술적으로 못 죽인다"가 후반이다. 두 문장이 한 곡선 위에 있다.** — 그리고 스테이지 1~2의 직행(169.4s)조차 **armor를 깨는 것(160.0s)보다 6% 손해**다: **무투자 빌드에게조차 armor가 이득**이라는 것이 R7의 G-1이며, **그것이 게이트가 게이트이기 위한 조건**이다.

- ★ **Q(노말 스탠스)의 존재 이유는 여기서도 증명된다** — 화산을 **불+4**로 밀고 온 특화가 `plate`(물)를 **불 스탠스로 때리면 ×0.5**다: `2391/0.5 = 4,782` → 그 부위 하나에 s5에서 `4782 × 1.222/36` = **162.3초** = 타이머 초과. **Q로 바꾸면 ×1 = 81.2초.** "모르겠으면 Q"가 **81초짜리 실물 마진**이 된다(v1.0의 13초보다 크다 — HP가 armor로 옮겨갔기 때문이다).
- `turret`(불)은 **테마 속성 부위**(R4가 1개만 허용) = 화산을 물+4로 밀고 온 빌드가 **유일하게 ×2로 녹이는 부위**. 깨면 레이저가 사라진다.
- ★ **`vent`(풀, armor)를 깨면 용암 장판도 함께 꺼진다** — 정본 §9.2("부위 파괴 = 그 부위의 이미터 정지, 전 `partType` 공통")의 귀결이다. `armor`가 **게이트 해제 + 탄막 감소**를 동시에 주므로 화산 테마의 정체성("공간 제한")이 **플레이어의 손으로 꺼진다.** 이전 판본에서 `armament`가 하던 그 장면을 **armor가 상속**한다 = armor 2 재구성으로 **잃은 것이 없다.**

### 10.4 나머지 4종

> **HP는 전 6종 동일**(정본 §13.6.2, ★ v1.2 ×1.20): `armor` **2,391** 각 · 선택 부위 **478** · `core` **976** · `armorCoreRatio` **4.90**. 아래 표는 **속성 배치·이미터·형상**만 다르다 — 그것이 이 섹션의 위임분이다.
> ★ **산술 검증은 §10.2의 표와 전 6종이 완전히 동일하다** — 균형 **125.5s**(s1) / **133.3s**(s5) · 무투자 **160.0s** / **195.5s** · 코어 직행 **169.4s** / **207.1s** · 선택 부위 **6.6s** / **8.1s**. **HP가 같으므로 시간도 같고, 다른 것은 「어느 스탠스가 어느 부위에 붙는가」뿐**이다.

| 보스 | 부위 | `partType`/`element` | 정답 | `hp` | `radius` | `anchor` | `shapeId` | `contactDmg` | `score` | 시드 이미터 | 파괴 효과 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`frostCrown`** 빙관 (`glacier`/물) `movePattern:"sway"` `{speedPxSec:30, ampPx:140, yHoldPx:165}` | `pylon` 서리탑 | `armament`/불 | 물(E) | **478** | 22 | `[0,-38]` | `cross` | 14 | 1300 | `wall`/`frostBrick`/`count:9, gapCount:1, gapWidthPx:84, speed:105` ★ **페이즈 3 = `stunMark`** | 얼음 벽 영구 삭제 |
| | `crownL` 좌관 | `armor`/풀 | **불(W)** | **2,391** | 26 | `[-56,16]` | `spike` | 15 | 2400 | `straight`/`pelletS`/`count:4, spreadDeg:34, speed:150` | 게이트 1/2 |
| | `crownR` 우관 | `armor`/불 | **물(E)** | **2,391** | 26 | `[56,16]` | `spike` | 15 | 2400 | `spiral`/`pelletS`/`count:8, speed:140, rotStepDeg:24, durationSec:2.0, rateSec:0.25` | 게이트 2/2 |
| | `core` | `core`/노말 | 아무거나 | **976** | 40 | `[0,0]` | `orb` | 16 | 8000 | — (코어는 사격하지 않는다) | **사망** |
| **`scarab`** 태양갑충 (`desert`/불) `movePattern:"orbitArc"` `{speedPxSec:38, ampPx:130, yHoldPx:150}` | `carapace` 갑각 | `armor`/물 | **풀(R)** | **2,391** | 30 | `[-50,26]` | `slab` | 15 | 2400 | `fan`/`fanShard`/`count:7, arcDeg:96, speed:130` | 게이트 1/2 |
| | ★ `stinger` 침포 | `armor`/풀 | **불(W)** | **2,391** | 30 | `[50,26]` | `spike` | 15 | 2400 | `aimed`/`homingM`/`count:2, spreadDeg:20, speed:90, leadSec:0.35` | 게이트 2/2 · **유도탄 소멸** |
| | `legs` 다리마디 | `mobility`/**불** | 물(E) | **478** | 20 | `[0,44]` | `claw` | 14 | 1200 | `straight`/`pelletS`/`count:3, spreadDeg:20, speed:150` | `speedPxSec × 0.5` + `ampPx → 0` = **호 기동 정지** |
| | `core` | `core`/노말 | 아무거나 | **976** | 38 | `[0,0]` | `hexPod` | 16 | 8000 | — | **사망** |
| **`thornKing`** 가시왕 (`forest`/풀) `movePattern:"sway"` `{speedPxSec:32, ampPx:150, yHoldPx:170}` | `podL` 좌 꼬투리 | `armor`/물 | **풀(R)** | **2,391** | 25 | `[-54,12]` | `orb` | 15 | 2400 | `ring`/`fanShard`/`count:9, speed:130, rotOffsetDeg:20` | 게이트 1/2 |
| | `podR` 우 꼬투리 | `armor`/불 | **물(E)** | **2,391** | 25 | `[54,12]` | `orb` | 15 | 2400 | `straight`/`pelletS`/`count:5, spreadDeg:40, speed:150` | 게이트 2/2 |
| | `bloom` 화포 | `armament`/**풀** | 불(W) | **478** | 26 | `[0,-34]` | `ring` | 14 | 1300 | `wall`/`thornBrick`/`count:8, gapCount:1, gapWidthPx:92, speed:120` | **가시 벽 영구 삭제** |
| | `core` | `core`/노말 | 아무거나 | **976** | 40 | `[0,0]` | `bulb` | 16 | 8000 | — | **사망** |
| **`mire`** 늪주 (`bog`/풀) `movePattern:"holdCenter"` `{speedPxSec:0, ampPx:40, yHoldPx:155}` | `shell` 이끼갑 | `armor`/불 | **물(E)** | **2,391** | 30 | `[-50,28]` | `hexPod` | 15 | 2400 | `fan`/`fanShard`/`count:7, arcDeg:110, speed:130` | 게이트 1/2 |
| | ★ `tendril` 촉수각 | `armor`/물 | **풀(R)** | **2,391** | 30 | `[50,28]` | `claw` | 15 | 2400 | `straight`/`pelletS`/`count:3, spreadDeg:24, speed:150` | 게이트 2/2 |
| | `sac` 포자낭 | `armament`/**풀** | 불(W) | **478** | 22 | `[0,-40]` | `orb` | 14 | 1300 | `spiral`/`hexBolt`/`count:10, speed:95, rotStepDeg:26, durationSec:2.0, rateSec:0.2` ★ **페이즈 3 = `stunMark`** | ★ **둔화·스턴 이미터 영구 삭제** |
| | `core` | `core`/노말 | 아무거나 | **976** | 38 | `[0,0]` | `bulb` | 16 | 8000 | — | **사망** |

★ **`mire`의 `tendril`이 `mobility` → `armor`가 된 것은 S18이 강제한 것이기도 하다.** `mire`는 `holdCenter`(`ampPx` 40)이므로 **`mobility` 부위를 가질 수 없다**(정본 §8.12.1 · S18) — `ampPx`가 애초에 작아 파괴 효과가 무의미하고, **트레이드오프가 거짓인 부위는 §8.13의 소프트 게이트를 흐린다.** 이전 판본은 `holdCenter` 보스에 `mobility`를 달아 **"이동 ×0.5, 흔들림 중단"이라는 사실상 아무 일도 일어나지 않는 파괴 효과**를 저작하고 있었다. → armor 승격이 **R6와 S18을 동시에 해소**한다.
★ **`scarab`의 `legs`(mobility)는 유효하다** — `orbitArc`(`ampPx` 130)이므로 파괴 시 **호 기동이 실제로 멎는다** ✔ S18 통과.

### 10.5 ★ 스턴의 유일한 거처 (정본 §2.7 · §12.4 · S13 준수)

| 규칙 | 이 섹션의 준수 |
|---|---|
| ★ **S13** — 스턴 탄을 쓰는 이미터는 **보스 부위의 `patternSet[2]`(페이즈 3)에만** | `frostCrown.pylon` · `mire.sac`의 **페이즈 3 이미터만** `bulletId: "stunMark"` ✔ 잡몹·엘리트·중간보스·새떼는 스턴을 **절대 갖지 않는다** |
| `stage.statusStunMaxPerStage = 2` | **스테이지당 1기** (보스 부위 1개) ✔ **여유 100%** — 정본 §2.7의 저작 계약("스턴 이미터는 보스 부위의 페이즈 3에만") 그대로 |
| ★ `difficulty.stunMinDifficulty = "hard"` 미달 시 | ★ **그 이미터는 발사를 스킵한다** — 탄 치환 없음·텔레그래프 없음·침묵(정본 §2.7, R-3 채택). 판정 시점 = **이미터 발사 시점**이므로 **콘텐츠는 난이도를 모른다**. 그리고 **rng draw를 소비하지 않아** 난이도 간 스트림 정렬이 유지된다 |
| `fairness.minStunTelegraphSec = 1.5` | 보스 부위 `telegraphSec = 1.50` 고정(§9.3) → **자동 충족** ✔ |
| `fairness.maxStunSec = 1.0` | `stunMark.statusDurationSec = 1.0` ✔ |
| 텔레그래프 = 화면 가장자리 호박 1회 사전 펄스 (정본 §7.4·§7.11·§7.12.4-④⑤) | 1.50초 리드 + 전신 광휘 + 패턴 점선 프리뷰와 **동시** |

- ★ **왜 치환이 아니라 침묵인가 (정본 §2.7의 근거)**: ① 치환하면 "Normal의 그 부위는 무슨 탄을 쏘는가"가 **새 저작 항목**이 된다(보스 부위마다 2벌) = **4주 예산 위반** ② 치환탄은 `caps`·`fairness`를 다시 검증해야 한다 ③ **침묵은 그 자체로 관대함의 표현이다** — 정본 §6.2가 "무-트위치 기둥은 Normal에서 보장된다"고 했고, **최종 페이즈가 조금 조용해지는 것이 정확히 그 보장의 형태**다.
- ★ **스턴을 `glacier`·`bog`(= 둔화 정체성 테마 2종)의 보스에만 둔 이유**: 상태이상이 **테마 → 잡몹 → 보스**로 한 줄로 이어져야 학습이 이월된다. 늪에서 `hexBolt`(둔화)를 맞아 온 플레이어가 보스 페이즈 3에서 **같은 육각탄인데 흰 이중 링 + 가장자리 펄스**를 보는 것 — 그게 스턴이다. 새 채널이 아니라 **아는 채널의 강조**(정본 §7.11의 3층 구조 그대로).
- ★ **`sac`(포자낭) 파괴 = 둔화와 스턴이 동시에 사라진다.** `armament`의 정의가 **런에서 가장 큰 페이오프**가 되는 지점 — 늪주는 "상태이상을 끄고 싸울 것인가, 시간을 아낄 것인가"가 곧 보스전의 결정이다. 그리고 `sac`는 **풀(테마 속성)**이라 늪을 불+4로 밀고 온 빌드가 ×2로 가장 빨리 끌 수 있다 = **테마 특화가 보상받는 유일한 부위**(R4가 1개만 허용하므로). **선택 부위 hp = 478**이므로 그 값은 s5에서 `478 × 14.42 ÷ (424.8 × 2.0)` = **8.1초** = 싸다 → "끄는 것"이 **현실적인 선택지**가 된다.

---

## 11. 스테이지 6 — 최종 (`stageId: "finale"`)

### 11.1 정본이 확정한 것 (인용)

`element: null`(**`null`은 `finale`에만**, S9) · `mix {water .30, fire .30, grass .30, normal .10}`(**같은 스키마**) · 고지 "테마 없음 — 4속성 혼재" · 로스터 **15종 총출동** · 중간보스 2회·3종 중 2종·**서로 다른 속성** · `finaleCrisisRotating: true`(물×2 → 불×2 → 풀×2) · 보스 「테트라크」 · `finale.partCount 5` / `finale.armorPartCount 3` / ★ **`exemptRules ["R4", "R6"]`**(R-2 채택) / `allowNormalPeripheral true` / ★ **`finale.armorCoreRatio 13.58`**(R7은 면제가 아니라 **a=3으로 재평가**: 상한 `0.4^-3 − 1` = 14.625) · ★ **`tier: "final"` → `bossHpScale` 미적용, `hp`가 곧 실효 HP** · **격파 = 런 클리어.** (정본 §8.16 · §13.6.1)

★ **최종 로테이션의 시각 규격은 「없다」가 확정이다 (정본 §7.12.3)**: 02가 "서브웨이브마다 띠가 속성색으로 펄스"를 제안했으나 **기각**되었다 — ① 정본 §7.2의 **가장 강한 예약**("적 탄·텔레그래프에 4속성 색 절대 금지") 위반 ② **불필요**하다: **적의 외곽선이 이미 답**이고(§7.6의 <200ms 판독 계약), 서브웨이브 간격 **4.17초**(§8.2의 파생값)는 무-트위치 예산의 7배이며, ★ **`player.stanceSwitchCooldown` = 0**이다(정본 §4.3 — ★ v1.2: `stance` 스코프는 **존재하지 않는다**, `player`에 편입됐다). → **"미리 알려주면 시험이 아니라 대본이다."** 이 섹션은 **새떼의 `element`만 저작**하고 시각 채널을 만들지 않는다.

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

| 블록 | `u` | s1 (`water`) | s2 (`fire`) | s3 (`grass`) | s4 (`normal`) | 블록 XP |
|---|---|---|---|---|---|---|
| A | 2 | `vWedge:drifter`(6) | `columnV:columnAnt`(6) ★ | `arc:spitter`(6) | `scatter:hexer`(2) ★ | **64** |
| B | 2 | `pincer:flanker`(6) ★ | `scatter:dustRunner`(6) ★ | `pincer:thornWeaver`(6) ★ | `lineH:mortarHulk`(2) | **164** |
| C | 1 | `scatter:sirenRay`(3) ★ | `lineH:magmaBomb`(3) ★ | `arc:bogHexer`(3) ★ | `scatter:frostLance`(1) | **89** |
| D | 1 | `scatter:turretPod`(3) ★ | `scatter:rearDart`(3) **`bottom`** | `scatter:stalker`(3) ★ | `lineH:drifter`(1) | **56** |
| E | 1 | `lineH:frostLance`(3) ★ | `arc:magmaBomb`(3) ★ | `scatter:hexer`(3) ★ | `scatter:turretPod`(1) | **75** |

★ = `eliteIndex: 0` (band ≥ `line` 이고 `element ≠ normal`). `spawnEdge` 무표기 = `top`.

**S8 검산**: 70 개체 = water **21** / fire **21** / grass **21** / normal **7** → **30.0 / 30.0 / 30.0 / 10.0** — **오차 0.0%p** ✔
**★ S22 검산**: Σ XP = 64+164+89+56+75 = **448**. 위기 XP = `60 × swarmTotalScale[6](1.0) × swarmXp(1)` = **60** → `60/448` = **0.134 ≤ 0.30** ✔
**로스터 검산**: `drifter` `columnAnt` `spitter` `hexer` `mortarHulk` `flanker` `dustRunner` `thornWeaver` `sirenRay` `magmaBomb` `bogHexer` `frostLance` `turretPod` `rearDart` `stalker` = **15종 전부 저작** ✔ (정본 §8.16 "지나온 5개 테마의 회고")

★ **「15종 총출동」의 정확한 의미 — 저작이지 조우가 아니다**

`mobPhaseMaxWaves = 14`이고 **한 웨이브 레코드 = 한 아키타입**이므로 **한 런이 조우할 수 있는 아키타입은 산술적으로 최대 14종**이다. 15종을 한 런에 전부 만나는 것은 **불가능**하다. 정본 §8.16의 문장은 **「로스터」**를 말하며(= 저작 리스트), 이 섹션은 그것을 만족한다.
→ **블록 순서를 A→B→C→D→E로 두어 앞 14 웨이브가 14종을 덮게 했다**(= 산술적 최대):

| 앞 14 웨이브 | 조우 아키타입 |
|---|---|
| A(4) | `drifter` `columnAnt` `spitter` `hexer` |
| B(4) | `flanker` `dustRunner` `thornWeaver` **`mortarHulk`** |
| C(4) | `sirenRay` `magmaBomb` `bogHexer` `frostLance` |
| D s1·s2 (2) | **`turretPod`** **`rearDart`** |
| **합** | **14종** ✔ (미조우 = `stalker` 1종, D s3 = 15번째 웨이브) |

- **B s4를 `mortarHulk`로, D s1을 `turretPod`로 앞당긴 것이 이 커버리지의 전부다** — 이전 판본의 순서(D s4 `turretPod` / E s4 `mortarHulk`)로는 **12종**만 조우했고 `stalker`·`turretPod`·`mortarHulk` 3종이 통째로 빠졌다.
- **S8은 저작 리스트 기준**이므로 14 웨이브 절단은 **S8에 영향이 없다**(정본 §8.2.1). 실제 조우 편차 = 정본이 계산한 `finale` 첫 14웨이브 케이스 **32.1/32.1/26.8/8.9 = 최대 −3.2%p** ≤ 5%p ✔

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
- ★ **`player.stanceSwitchCooldown = 0`(정본 §4.3 — ★ v1.2 개명, §21-A1)의 존재 이유가 이 25초에 전부 결산된다.** 쿨다운이 1초라도 있었으면 이 장면은 성립하지 않는다. 그리고 **`2/2/2` 빌드는 3파 전부 ×2**, **`4/2/0` 빌드는 한 색 앞에서 ×1로 밀린다** → 정본 §4.2가 `elementCapTotal`을 8이 아니라 6으로 정한 근거가 **눈에 보이는 25초**가 된다.
- **`crisisTotal = 60 × swarmTotalScale[스테이지 6](1.0) = 60`** → 20/20/20 = 정확히 3등분 ✔
- ★ **로테이션이 `crisisKillShareWithoutCapstone ≥ 0.80`을 깨지 않는가**: 게이트의 분모는 **capstone 미보유 그리고 폭탄 미사용 세션**이다(정본 §13.1.1) → **폭탄은 이 게이트의 답이 아니다.** 실제 답: `killShare`를 결정하는 것은 **도달**이고(§8.5의 17.6배 화력 여유), 로테이션은 **화력이 아니라 스탠스**를 요구한다. 무투자·무-capstone 빌드조차 **×1로 1,080 HP를 25초에 지운다**(필요 화력의 5.9%) → **로테이션은 `killShare`를 낮추지 않는다.** 로테이션이 시험하는 것은 **"×2를 받을 것인가"**이지 "지울 수 있는가"가 아니다.

### 11.4 「테트라크 (TETRARCH)」

| 부위 | `partType` | `element` | **정답 스탠스** | `hp` (**절대**) | `radius` | `anchor` | `shapeId` | `contactDmg` | `score` | 파괴 효과 |
|---|---|---|---|---|---|---|---|---|---|---|
| `coffinWater` 물의 관 | `armor` | 물 | **풀(R)** | **25,800** | 24 | `[-64,20]` | `slab` | 15 | 2000 | 게이트 1/3 |
| `coffinFire` 불의 관 | `armor` | 불 | **물(E)** | **25,800** | 24 | `[64,20]` | `slab` | 15 | 2000 | 게이트 2/3 |
| `coffinGrass` 풀의 관 | `armor` | 풀 | **불(W)** | **25,800** | 24 | `[0,52]` | `slab` | 15 | 2000 | 게이트 3/3 |
| `throne` 왕좌 | `armament` | **노말** | 없음 (항상 ×1) | **5,200** | 26 | `[0,-44]` | `ring` | 15 | 1800 | **유도탄 이미터 영구 삭제** |
| `core` 핵 | `core` | 노말 | 아무거나 | **5,700** | 44 | `[0,0]` | `orb` | 16 | **15000** | **사망 = 런 클리어** |

`movePattern: "holdCenter"`, `movePatternParams: {"speedPxSec":0,"ampPx":60,"yHoldPx":160}`, `armorCoreRatio: 13.58`, `tier: "final"` (★ **`bossHpScale` 미적용 — `hp`가 곧 실효 HP**)

> ★ **HP는 정본 §13.6.3이 소유한다.** 이전 판본의 `관 3×900 + 왕좌 700 + 핵 4600`(base) × `bossHpScale[5] 11.5`는 **전량 폐기**다(정본 §18.3-3).
> ★ **v1.2에서 테트라크의 HP는 한 자리도 안 바뀌었다** (정본 §20.3-04-1: "테트라크는 불변"). **테마 보스가 ×1.20된 것과 대비된다** — 이유는 **`tier:"final"`이 `bossHpScale`의 정의역 밖**이라 스테이지 1 앵커(30 → 36)의 사슬을 타지 않고, **`dpsRef[6] = 708`이 사용자 확정으로 보존**되기 때문이다(정본 §13.5.1: 제너럴리스트의 화력 패시브가 `warhead` Lv4 × `overclock` Lv4에서 멈추므로 708은 **두 farm 정책의 공통 천장**이다). → ★ **테트라크만 farm 정책 변경에 면역인 것이 우연이 아니라 구조다.**
> ★ **왜 최종만 절대값인가**: `bossHpScale`은 **셔플 때문에 존재**하고 `tetrarch`는 셔플되지 않는다 → **적용할 이유 자체가 없다.** 그리고 적용하면 `tetrarch`의 base가 `5700/14.42 = 395`가 되어 **`manta`의 코어(976)보다 작아 보인다** = **AI·리뷰어가 읽는 JSON이 거짓말을 한다**(I-2의 정신).

**시드 이미터** (`P = 4` → `offsetSec = everySec × i / 4`, §9.3)

| 부위 | `type` | `bulletId` | 시드 `count` | 고유 |
|---|---|---|---|---|
| `coffinWater` | `ring` | `heavyRound` | 10 | `speed:115, rotOffsetDeg:18` |
| `coffinFire` | `zone` | `null` | — | `radius:72, activeSec:3.0, dmg:10` |
| `coffinGrass` | `wall` | `thornBrick` | 8 | `gapCount:1, gapWidthPx:92, speed:120` |
| `throne` | `aimed` | `homingM` | 3 | `spreadDeg:24, speed:90, leadSec:0.4` |

**★ 산술 검증** (스테이지 6, `dpsRef[6] = 708` **명목** → `× uptimeRef 0.60` = **실효 424.8**, `m` = 1.62 — 정본 §13.6.3의 재현)

| 빌드 | 계산 | 시간 | 게이트 (정본 §13.1) | 판정 |
|---|---|---|---|---|
| **균형 `2/2/2`** | `77,400/(424.8×1.62) + 5,700/424.8` = 112.5 + 13.4 | **125.9s** | `balancedPass ≥ 0.95` | ✔ **1.000** (여유 54s) |
| **특화 `4/2/0`** | `25,800/849.6 + 25,800/688.2 + 25,800/424.8 + 13.4` = 30.4 + 37.5 + 60.7 + 13.4 | **142.0s** | `specialistPass ≥ 0.85` | ✔ **0.994** (여유 38s) |
| **무투자** | `83,100/424.8` | **195.6s** | `noElementPass ∈ [0.05, 0.28]` | ✔ **0.149** (**15.6초 초과 = 실패**) |
| **코어 직행** (armor 3, ×0.064) | `5,700/0.064/424.8 = 89,062/424.8` | **209.7s** | — | ✗ **완전 불가** |
| `throne` 추가 비용 | `5,200/424.8` | **+12.2s** | — | 모든 빌드에 **동일한 가격표** |

> ★ **이 표는 v1.2에서 한 줄도 안 바뀌었다** — `dpsRef[6] = 708` · 실효 424.8 · `m` 1.62 · HP 5종이 전부 보존됐기 때문이다. **정본 §13.6.3의 인쇄값과 5행 전부 일치** ✔

★ **승복 — 이전 판본의 「무투자 161s = `noElementPass ≥ 0.55` = 브릭 아님」은 뒤집혔다**

| | 이전 판본 | **정본 v1.1** |
|---|---|---|
| 게이트 | `noElementPass ≥ **0.55**` (단일 값, 하한만) | ★ **스테이지별 곡선 + 상한**: `min [0.78, 0.78, 0.35, 0.35, 0.05, 0.05]` / `max [0.98, 0.98, 0.65, 0.65, 0.30, 0.28]` |
| 무투자 결과 | 161s = **여유 19s = 통과** | **195.6s = 15.6초 초과 = 실패(0.149)** |
| 무엇이 바뀌었나 | 핵이 총 HP의 **63%** | ★ **핵이 7%.** 총량은 83,950 → 83,100(**1% 차이**)이고 **바뀐 것은 배분뿐이다** |
| 왜 중대한가 | — | ★ **핵 63% 배분에서는 `noElementPass`가 어떤 `bossHpScale`로도 0.5 아래로 안 내려간다** — 핵은 R1에 의해 속성 무관이므로. 「3속성 투자의 졸업 시험」이 **산술적으로 성립 불가능**했다 |

- ★ **`noElementPass`에 상한이 생긴 것이 이 게이트의 성격을 바꿨다.** 단일 값 0.55는 **"브릭 없음"만** 증명했고 「필수」를 증명하지 못했다. 곡선 + 상한은 **「초반엔 무투자로도 되고(0.909), 후반엔 속성 투자가 필수(0.149)」를 6개의 숫자로 증명**한다(정본 §13.3). **무투자가 최종 보스를 깨면 그것은 통과가 아니라 실패다.**
- ★ **6개의 특화 짝이 전부 같은 시간을 낸다** — 관 3개가 3속성 전부이고 `4/2/0`은 항상 2개를 덮으므로 `2.0 / 1.62 / 1.0`의 **순열만 바뀐다** → **최종 보스에는 "운 나쁜 특화"가 존재하지 않는다.** 졸업 시험이 공정하다. (테마 보스의 6짝은 §10.2처럼 **갈린다** — 관이 2개뿐이라 커버 여부가 짝에 의존하기 때문이다. 최종만 대칭이다.)
- ★ **특화의 세 번째 관은 Q(×1)로 때린다** — 잘못된 스탠스로 ×0.5를 맞느니 Q가 낫다: ×0.5면 `25,800/212.4` = **121.5초**(vs Q의 60.7초) → 총 202.8초 = 실패. **Q의 존재 이유가 여기서 22.8초짜리 차이로 증명된다**(정본 §8.16).
- **`throne`(노말) = 12.2초.** 상성이 통하지 않으므로 **어떤 빌드도 같은 시간**이 든다 → 정본 §8.16의 "항상 ×1인 안전 바닥의 존재 증명 부위"가 **모든 빌드에게 동일한 가격표**로 실재한다. 유도탄이 괴로우면 12초를 낸다. **`armament`이므로 게이트에 영향이 없다** → 깰지 말지가 순수 선택이다.
- **03의 계약(「테트라크 총 HP 45,000~65,000」)도 함께 기각되었다** (정본 §13.6.3) — 03의 밴드는 "가용 피해의 40~55%"라는 **가정**에서 나왔고, 정본의 **88,300**(= 83,100 + 왕좌 5,200)은 **목표 격파시간에서 역산**한 것이다. 방법이 다르고 후자가 검증 가능하다.
- **고유 연출 (정본 §8.16의 규칙 인용)**: 관이 하나 파괴될 때마다 아레나 배경이 그 속성의 색을 잃는다 → 3개 다 깨지면 **무채색 아레나에서 핵과 단둘이.** 배경 파라미터는 `skinId: "finale"`의 3단 상태이며 **게임플레이 규칙을 만들지 않는다**(정본 §7.9 상한 준수).

### 11.5 최종 중간보스

2회(`midBossAtSec [30, 70]`), `mb*` 3종 중 2종, **서로 다른 속성**(정본 §8.16).
- **테마가 없으므로 `midBossElementRule = "notThemeAndNotNormal"`의 "테마가 아님" 조건이 공허참** → 후보 = `{fire, water, grass}` 전부. `rng.spawn`이 **비복원 2개**를 뽑는다.
- 종류도 `rng.spawn`이 3종 중 비복원 2개 → **최종은 매번 다른 2개의 예고편**을 준다.
- ★ **`mbNest`가 뽑히면 테트라크의 관을 미리 재보는 셈이 된다** — 스테이지 6의 `mbNest` = `880 × bossHpScale[6](**14.42**)` = **12,690** → `12,690 ÷ 424.8` = ★ **29.9초** vs `midBossLeaveAfterSec` **30** ✔ **여전히 아슬아슬**(여유 0.1초, `m` = 1.0 = 스탠스를 안 바꾼 플레이어 — §7.5의 모델).
  - **예고편의 크기**: 관 하나 = 25,800 → ×1로 **60.7초** / 균형(×1.62) **37.5초** / 특화(×2.0) **30.4초**. → ★ **`mbNest`를 ×1로 잡는 29.9초 ≈ 관 하나를 ×2로 잡는 30.4초.** 관 3개를 상대할 몸이 되었는지를 **정확히 관 하나 값어치로** 미리 재는 셈이다.
  - `bossHpScale[6] = 14.42`가 "계산되지만 스테이지 보스에는 쓰이지 않는" 값인데도 **배열에 남아 있는 이유가 이것이다**(정본 §13.6.1: "중간보스는 사용한다"). 그리고 **`bossHpScale[5] == bossHpScale[6]`**이므로 **스테이지 5와 6의 중간보스는 같은 개체**다 — `dpsRef[5] == dpsRef[6] == 708`(화력 천장)의 정합적 귀결이다(§7.5).

---

## 12. AI 빌드타임 생성 제약 (이 섹션의 콘텐츠에 한함)

### 12.1 하드 제약 — 위반 = 생성물 폐기·재생성

> **G1~G10은 정본 §13.4(S3~S12)의 재확인이며 새 규칙이 아니다. G11~G17이 이 섹션이 추가하는 제약**이고, 전부 **`check.mjs`에 정적 검사로 넣을 수 있는 형태**로 적었다.

| # | 제약 | 근거 |
|---|---|---|
| **G1** | `moveId`(8)·`emitterType`(8)·`formationId`(6)·`partType`(4)·`shapeId`(12) **어휘 밖 = 실패.** AI는 새 거동·새 도형을 발명할 수 없다 | 정본 §0.1 C-3, S3 |
| **G2** | `(moveId, emitterType)` 중복 = 실패 (`band` 다르면 허용) | S4 · §2.4 |
| **G3** | 보스 **R1~R7 전부** + `partCount`(코어 포함 4 / finale 5) + ★ **`armor` 수 == 2**(finale 3) + ★ **S24 HP 배분**(선택 부위 = `armor` 1개 × ★ `boss.optionalPartArmorRatio` **0.20**) + ★ **S18 `holdCenter` ⇒ mobility 금지** | S5 · S18 · S24 · §10.1 |
| **G4** | `telegraphSec ≥ max(이미터 타입 하한, 상태이상 하한, 개체 등급 하한)` — ★ **하한이 겹치면 큰 쪽이 이긴다** | 정본 §7.4 · §3.3 |
| **G5** | 보스 부위: `everySec ∈ [6.0, 4.5, 3.6][phase]` · `offsetSec = everySec × i / P` · `telegraphSec = 1.50` · 부등식 **`P × telegraphSec ≤ 2 × everySec`** | §9.3 (S7의 정적 형태) |
| **G6** | 탄 `speed ≤ 260`(조준탄 ≤ 200) · `radius ≥ 4` · `gapWidthPx ≥ 46` | 정본 §12.4 · §3.3 |
| **G7** | ★ **블록 불변식** — 웨이브 리스트는 블록으로만 구성하고, **모든 블록이 각각 정확히 `mix`**여야 한다. 슬롯 규격(§5.3 / §11.2)에서 벗어난 `count` = 실패 | S8을 오차 0으로 만드는 장치 |
| **G8** | `crisisElementRule` = 테마 100% (최종만 로테이션). 새떼에 `swarm*` 외 아키타입 금지 | S9 · §8 |
| **G9** | `rearIn` / `spawnEdge: "bottom"` 은 `rearSpawnAllowed[stage]`일 때만 | S9 · §4 |
| **G10** | 난수는 명명된 ★ **8 스트림**(`theme` `draft` `spawn` `elite` `drop` `pattern` `boss` · ★ **`bot`**)만. 스트림 간 공유 금지 (`rng.pattern`만 적·플레이어 양쪽 접근 허용) | 정본 §10.2 · S11 |
| **G11** ★ | `orbitDrift`의 `keepDistPx ≥ fairness.minSpawnRadiusPx`(**140**) | §3.3 — 미만이면 그 적은 **탄을 못 쏴 무해해진다**(점블랭크 금지에 걸림) = 조용한 콘텐츠 버그 |
| **G12** ★ | `eliteIndex ≠ null`은 **`band ∈ {line,turret,bruiser}` 이고 `element ≠ normal`인 웨이브에만** | §5.4 — 런타임 거부가 아니라 저작 시점에 `elite.bandAllowed`/`elementAllowed`(정본 §8.6)를 만족 |
| **G13** ★ | `count == 1`에 `pincer`·`columnV` 저작 금지 | 1기로는 "좌·우 동시 진입"·"종대"가 성립하지 않는다 |
| **G14** ★ | 새떼 `spawnEdge` = **`top` 고정** (`rearSpawnAllowed`가 참인 스테이지에서도) | 25초 최고밀도에 후방 진입을 겹치면 관대함이 붕괴 |
| **G15** ★ | 페이즈 파생은 **`count`·`arcDeg`·`rotStepDeg`·`everySec`만** 스케일한다. **`speed`·`type`·`bulletId` 스케일 금지** | §9.4 — 페이즈 스케일이 `maxBulletSpeed`(260)를 몰래 넘는 사고 차단 |
| **G16** ★ | `shapeId` 재사용은 허용, **신규 발명은 금지**. 같은 `shapeId`를 쓰는 두 아키타입은 `radius`가 **1.5배 이상** 차이나거나 `band`가 달라야 한다 | §2.5 — 실루엣 계열 학습을 이월시키되 혼동은 막는다 |
| **G17** ★ | `bullets[].status == "stun"`을 참조하는 이미터는 **`tier:"stage"` 보스 부위의 `patternSet[2]`(페이즈 3)에만**, 그리고 **스테이지당 최대 1 부위** | 정본 **S13**의 저작 시점 판본. `statusStunMaxPerStage`(2)를 **절반의 여유로** 만족 + `minStunTelegraphSec`(1.5)를 §9.3의 `telegraphSec = 1.50`이 자동 충족 |
| **G18** ★ | **편대 수용량**: `lineH`·`vWedge` ≤ **9** / `arc`·`pincer` ≤ **12** / `scatter`·`columnV` ≤ **16** | §5.7 — 정본 §9.9.2의 `gapPx`/`radiusPx`/`spanDeg`와 이동 영역 폭 540에서 **파생**된다. **새 값이 아니다** |
| **G19** ★ | **블록 불변식의 2조항**: ① **T1 블록은 정확히 5 레코드, 슬롯 분할 금지** ② T2·T3 블록의 분할은 **같은 `element` + `count` 합 = 슬롯 규격** | §5.3 — ①이 정본 §8.2.1의 조우 편차 산술(11/14웨이브 주기성)을 **보존**한다. 위반하면 **정본의 ≤5%p 선언이 무효가 된다** |
| **G20** ★ | **웨이브 1개의 `Σ count` ≤ `enemyConcurrentMax`(40)** · **위기 서브웨이브 1파의 `Σ count` ≤ `swarmConcurrentMax`(70)** | 정본 **S26**(v1.2 신설)의 저작 시점 판본. 이 섹션의 최대 = 웨이브 **16**(여유 60%) / 서브웨이브 **10**(여유 85%). ★ **한 웨이브가 혼자 40을 넘으면 그 웨이브는 어떤 처치율에서도 예산을 깬다 = 편성 버그**이며, 그것만이 정적으로 검사 가능하다(정본 §12.1) |

### 12.2 ★ 동시 텔레그래프 예산 — 진단은 채택되었고 값은 정본이 파생시켰다

**이 섹션이 발견한 것 (정본 §19.3 R-4: 수정채택 — "진단은 완벽하다")**

> 정본 v1.0 §12.1의 2층 검증표에서 `enemies`(A 40 < B 96)와 `enemyBullets`(A 320 < B 384)는 **전역 대 전역**을 비교하는데, `telegraphs` 행만 **"A 2/개체 < B 8"** 로 **개체당 예산과 전역 풀을 비교**했다. 그리고 그 행의 괄호 주석 —"잡몹 다수가 각자 1~2개 → 8이 안전망" — 은 **스스로 결론을 반박**한다. **단위가 다른 두 수를 부등호로 이은 것**이며 → **`capHits == 0`이 영구 실패**한다.

**정본의 처분 (v1.1 §12.1 · §12.4)**

| 층 | **정본의 확정** | 이 섹션의 제안 | 판정 |
|---|---|---|---|
| A층 | ★ **`fairness.telegraphConcurrentMaxGlobal = 80` (파생)** = `enemyConcurrentMax(40) × telegraphConcurrentMaxPerEntity(2)` | `48` | ★ **기각** — **48은 새 자유 숫자**다. 80은 정본이 이미 확정한 **두 값의 곱**이므로 **새 숫자 0**이고, `check.mjs` **S12**가 곱의 무결성을 검사한다 |
| B층 | **`caps.telegraphs` 8 → 96** (= 80 × 1.2, 다른 두 행과 **같은 20% 여유**) | `96` | **채택** |

> ★ **승복**: 48은 이 섹션이 "실측 최악 13에 여유를 붙인 수"로 고른 것이었다 — **근거가 관측이지 구조가 아니었다.** 정본의 80은 **A층의 정의(오써링 예산)에서 곧바로 나온다**: 40기가 각자 2개까지 쓸 수 있다면 전역 상한은 80이며, 그것이 **A층이 뜻하는 바 그 자체**다. **v1.0에 없던 것은 값이 아니라 「A층 전역 상한이라는 개념」이었다.**

**이 섹션의 콘텐츠 재검산 (정본의 80/96 기준)**

**이미터 duty** (`telegraphSec ÷ everySec`) = 잡몹 13종 **10.0% ~ 36.4%**, **가중 평균 ≈ 15%**

```
잡몹 페이즈 최악 (스테이지 5~6, 동시 개체 ≈ enemyConcurrentMax 40, 사격 개체 ≈ 30):
  기대 동시 텔레그래프 λ = 30 × 0.15 = 4.5
  포아송 3σ ≈ 13                    ≪  A층 80   →  여유 84%
보스전:   ≤ 2/부위 × 5부위 = 10      ≪  A층 80   →  §9.3의 정적 부등식이 보증
위기 세션: swarmLancer 6기 × duty 10% ≈ 0.6      ←  새떼 90%(swarmChaff)가 사격하지 않는다
```
→ **어디서도 A층 80에 닿지 않고, B층 96의 `deferAttack`은 발동할 일이 없다** → **텔레그래프 축의 `capHits = 0`** ✔ (정본 §13.2-⑦의 결론과 일치)

- ★ **위 「동시 개체 ≈ 40」은 보수적 상한이며 이 섹션의 실제 최악은 24다** (§8.3의 ①): 정본 §8.7의 스포너 규칙(`다음 스폰 = max(직전 + 9.0, 직전 전멸)`)에 의해 **한 번에 살아 있는 웨이브는 1개**이므로 웨이브 개체 ≤ **16**(최대 레코드) + 중간보스 **2** + `mbNest` 소환물 **6** = **24 ≪ 40**(여유 40%). 그중 사격 개체 ≈ 18 → **λ = 2.7 → 3σ ≈ 9**. **더 안전한 쪽이므로 인쇄값은 보수적인 λ=4.5를 유지한다.**
- ★ **`swarmChaff`의 `attack: null`이 위기 세션을 예산 안에 묶는 유일한 장치다** — 정본 §8.6이 그것을 "사격 안 함"으로 못박은 것이 여기서 **캡 산술로 회수**된다.

★ **v1.2 — `capHits`의 정의가 A층까지 확장됐고, 그것이 이 섹션에서 결함 1건을 드러낸다 (정본 §13.1.1)**

v1.1의 `capHits`는 「`caps.*`의 overflow 정책」만 셌다 → **A층 40/70/80은 어느 게이트에도 안 잡히는 「검사기 없는 죽은 제약」**이었다. v1.2는 **A층 `defer` 발화 틱까지** 센다(`enemyConcurrentMax` 40 · `swarmConcurrentMax` 70 · ★ **`crisisWaveResidualMax` 10** · `telegraphConcurrentMaxGlobal` 80).

| A층 축 | 이 섹션의 실측 최악 | 판정 |
|---|---|---|
| `enemyConcurrentMax` **40** | **24** (웨이브 16 + 중간보스 2 + 소환 6) | ✔ 여유 40% |
| `swarmConcurrentMax` **70** | **14** | ✔ 여유 80% |
| `telegraphConcurrentMaxGlobal` **80** | **13** (λ=4.5의 3σ) | ✔ 여유 84% |
| ★ **`crisisWaveResidualMax` 10** | ★ **16** (passive 정책, 스테이지 1 — §8.3) | ✗ ★ **초과 → §14.3-N-4 (blocker)** |

→ ★ **4축 중 3축은 큰 여유로 통과하고, 신설된 1축만 실패한다.** 그리고 그 실패는 **편성이 아니라 값의 문제**다(§8.3의 논증).

---

## 13. 검증 게이트 대조 — 이 섹션의 콘텐츠가 실제로 통과하는가

> ★ **전 지표가 probe(= `maxFarm`) 기준이다** (정본 §13.5.1). baseline 플레이어의 시간은 `÷ runFarmDpsRatio(0.83)`이며 그것은 **정본 §13.6.5가 소유하는 진단**이지 이 섹션의 저작이 아니다.

| 게이트 (정본 §13.1 · §13.1.1) | 이 섹션의 대응 | 판정 |
|---|---|---|
| `dpsProbe.balancedPass ≥ 0.95` (전 스테이지) | 테마 보스 **125.5~135.1s** / 테트라크 **125.9s** (제한 180) | ✔ **0.999~1.000** §10.2 · §11.4 |
| `dpsProbe.specialistPass ≥ 0.85` | 테마 보스 6짝 평균 **147.7s → 0.935** (최악 짝 164.4s → **0.851**) / 테트라크 **142.0s → 0.994** | ✔ ★ **razor**(최악 짝 여유 0.1%p) = 의도. 손잡이 = `armorCoreRatio`(정본 소유) |
| ★ `dpsProbe.noElementPass` **곡선** `min [0.78,0.78,0.35,0.35,0.05,0.05]` / `max [0.98,0.98,0.65,0.65,0.30,0.28]` | ★ **0.909 / 0.909 / 0.500 / 0.499 / 0.151 / 0.149** (§10.2 · §11.4) | ✔ **전 6항 밴드 안** — 사용자 확정 곡선(0.90/0.90/0.50/0.50/0.15/0.15)을 **v1.1보다 정확히** 재현(반올림 인공물 제거) |
| ★ `killTimeMedianBalanced ∈ [120, 150]` (**v1.2 개정** — 사용자 요구가 처음으로 게이트가 됐다) | 6보스 `T_bal`(probe) = `[125.5, 125.5, 135.1, 128.3, 133.3, 125.9]`(테마 5 + 테트라크) → ★ **중앙값 127.1s = 2.12분** | ✔ **한복판** (정본 §13.6.5) |
| `bossTimeoutRate ∈ [0.02, 0.25]` | ★ **0.2389** (정본 §13.2-④의 자기 콘텐츠 검산). 이 섹션의 기여: 무투자가 s5·s6에서 **구조적으로 초과**하므로(195.5s / 195.6s vs 180) **타임아웃 사망이 0이 될 수 없다** = 3분 타이머가 **살아 있는 규칙** | ✔ ★ **razor**(상한 여유 4.4%) · 하한 여유 12배 |
| `dominance.maxArchetypeLethalityShare ≤ 0.25` (분모 = 플레이어가 입은 총 피해, 17종) | 최다 등장 = `spitter`(**4테마**)이나 탄이 `pelletS`(dmg **8** = 저작 최소). 최대 위협 = `turretPod`(레이저 22)·`mortarHulk`(장판 10)이나 **각 3·1테마**에만 등장 → 상한 추정 **0.15** | ✔ §4 |
| `dominance.maxElementWinShare ≤ 0.42` (균등 0.333) | ★ **armor 속성이 물 4 · 불 4 · 풀 4로 정확히 균등**(§10.1) + 테마가 속성당 2종 + 상성 순환 → **비대칭의 원천이 구조적으로 없다** | ✔ 구조적 |
| `dominance.maxThemeClearStddev ≤ 0.06` | ★ **근거 교체**: armor 1/2 축은 사라졌다(전부 armor 2). 새 근거 = **선택 부위 타입 2:4 · `movePattern` 3:1:2 · Σ XP ±10% 정규화**(§5.6) + `s_t ∈ [0.89, 0.97]` 저작 계약(정본 §13.2-⑨) | ✔ 정본 §13.2-⑩의 자기 콘텐츠 검산 = **0.015** (여유 4.0배) |
| ★ `crisisKillShareWithoutCapstone ≥ 0.80` (분모 = capstone **그리고** 폭탄 없는 세션) | `swarmChaff.hp = 3` → s6에서도 개체 HP **18** = 1~2히트. 순수 ST 빌드의 화력 여유 **17.6배** → 결정 요인은 **도달**, 추정 **0.85~0.95** | ✔ §8.5 |
| `farmXpRatio ≥ 2.0` | `dustRunner`(xp 15, speed 230) = **이탈 = 영구 0** / `mbNest` 무시 = 0 / `anchor`·`strafe`가 상단에만 산다 / ★ **`waveListExhausted: "cycle"`이 정본에 반영됨** | ✔ §13.1 아래 |
| `coinScarcity` (3종) | ★ **⚠ 해소.** 손잡이 = **`elite.coin`**(정본 소유) · 분산의 원천은 **S23**이 봉쇄 · `bands[].coin`(turret 1 / bruiser 2)이 계산을 가능하게 함 | ✔ §6 |
| ★ **S22** 새떼 XP ≤ 0.30 (★ **상한이지 목표가 아니다**) | **최악 0.148**(`forest` s4·s5) — `swarmXp 1` + 블록 4개 편성. 총 XP 대비 실측 지분 = ★ **~9~10%**(§8.4 재계산) | ✔ **여유 2.0배** §8.4 |
| ★ **S23** 코인원 균질성 | 전 6테마 `turret`+`bruiser` = **1~2종** | ✔ §4 |
| ★ **S24** HP 배분 (★ 키 = `boss.optionalPartArmorRatio`) | 테마 보스 ±0.04% / `tetrarch` ±0.78% | ✔ §10.1 |
| ★ **S26** 동시 개체 예산의 정적 하한 (v1.2 신설) | 웨이브 1개 최대 **16** ≤ 40 (여유 60%) / 위기 서브웨이브 1파 **10** ≤ 70 (여유 85%) | ✔ §8.3 · §12.2 |
| `capHits == 0` (★ v1.2: **A층 + B층**) | B층: 적 ≤ 36 ≪ 96 / 탄 ≪ 320 / 텔레그래프 3σ ≈ 13 ≪ 96. A층: 40 → **24** ✔ · 70 → **14** ✔ · 80 → **13** ✔ · ★ **`crisisWaveResidualMax` 10 → 16** ✗ | ✗ ★ **1축 실패 — §14.3-N-4 (blocker).** 나머지 7축 전부 통과 |
| `fairnessViolations == 0` | §3.3 전수표 · §9.3 부등식 | ✔ |
| S8 (±3%p, **저작 리스트**) | **전 테마 · 전 스테이지 오차 0.0%p** | ✔ §5.5 · §5.6 · §11.2 |
| 실제 조우 편차 ≤ 5%p (**검사 안 함**, 정본 §8.2.1) | `sea` s1 **+5.0%p**(정본 인쇄값과 일치) · s3~5 **+2.7%p** · `finale` **−3.2%p** | ✔ §5.5 · §11.2 |

**`farmXpRatio ≥ 2.0`의 이 섹션 기여** (정본 §13.2-⑤: "비율을 만드는 것은 `enemyExitForfeitsReward`이지 `waveClearAdvance`가 아니다")

| 장치 | 이 섹션의 실물 | 소극 파밍 | 최대 파밍 |
|---|---|---|---|
| 이탈 = XP 영구 소멸 | `desert` T1b s1 = `scatter:dustRunner`(12기, **speed 230, 아레나 3.3초 종단**) = **XP 180 한 웨이브** | ~0 (요격 불가 위치) | ~0.95 × 12 × **xp 15** |
| 중간보스 무시 | `midBossLeaveAfterSec 30` · `mbNest`는 **27~29초**에 죽는다(§7.5) = 아슬아슬한 선택 | 0 | `xp 50` × 1~2 |
| 새떼 | 60기 × **xp 1** = 60 | ~0.20 | ~0.90 |
| 하단 고정 시 사거리 밖 이탈 | `dive` 계열(`drifter` speed 70)은 하단까지 오지만 `anchor`(`turretPod` `yHoldPx 150`)·`strafe`(`flanker` `yPx 180`)는 **상단에서만 산다** | 0 | 전량 |
| ★ **웨이브 수 순환** | 스테이지 1 리스트 = **10 웨이브**(T1a+T1b) < `mobPhaseMaxWaves` 14 → **순환이 실제로 발동**한다. ★ **정본 §8.7이 이 실측을 기입했다**(v1.1의 「5웨이브 → 2.8회차」 폐기 → **「10웨이브 → 최대 1.4회차」**) | 11 웨이브 | 14 웨이브 |

★ **`anchor`·`strafe` 아키타입의 `yHoldPx`/`yPx`(120~180)가 `farmXpRatio`의 숨은 주역이다.** 아레나 h=720에서 이들은 **위쪽 1/4에서만 존재**하고 하단으로 내려오지 않는다 → **하단에 붙어 있는 소극적 플레이어는 이들의 XP를 구조적으로 0으로 만든다.** 잡으려면 위로 올라가야 하고, 위 = 반응 여유 최소 = **위치 리스크**(정본 §8.8). 트위치가 아니라 "얼마나 앞에 설 것인가"가 XP 배율이 된다.

---

## 14. 정본 대조 · 남은 요청

### 14.0 ★ v1.0판의 요청 17건 — 정본 v1.1이 **전수 채택**했다

정본 §19.3의 심사 결과: **채택 15 · 수정채택 2 · 기각 0.** blocker 4건 전부 해소되었다.

| # | 요청 | 판정 | **정본이 반영한 곳** | 이 문서의 인용 위치 |
|---|---|---|---|---|
| **R-1** | `bossHpScale` 신설 (blocker) | **수정채택** — 필요성은 정확, **값은 폐기**하고 정본이 재산출. ★ **v1.1의 `[1.0, 1.8, 4.4, 7.5, 11.9, 17.3]`도 v1.2에서 폐기** → **`[1.00, 1.80, 4.67, 6.96, 14.42, 14.42]`**. `tier:"final"`에는 미적용 | §13.6.1 | §1.4 · §9.5 |
| **R-2** | `finale.exemptRules`에 `R6` 추가 (blocker) | **채택** — "정본이 자기 자신과 충돌했다" | §8.16 | §10.1 · §11.1 |
| **R-3** | `stunMinDifficulty` 미달 = 발사 스킵 (blocker) | **채택** — 치환은 저작 2벌(4주 예산 위반). **침묵이 Normal의 관대함의 형태** | §2.7 | §3.3 · §10.5 |
| **R-4** | 텔레그래프 예산 (blocker) | **수정채택** — 진단은 완벽. **48은 새 자유 숫자라 기각**, **파생값 80**(= 40 × 2) 채택. B층 96 채택 | §12.1 · §12.4 | §12.2 |
| **R-5** | `zone`의 `bulletId: null` 허용 명시 | **채택** + **S19** 동치 검사 | §9.7 | §3.1 |
| **R-6** | `unlockStageMin`의 거처 = 로스터 | **채택** — `enemies.json`에서 삭제 | §9.7 | §2.3 · §4 |
| **R-7** | 웨이브 리스트 소진 = 순환 | **채택** — **`farmXpRatio` 게이트와 파밍 기둥이 이 하나에 물려 있었다** | §8.7 | §5.1 · §13 |
| **R-8** | `patternSet[i]` → `{emitterIds: [...]}` | **채택** + **S16** | §9.8 | §7.3 |
| **R-9** | `bosses[].summon` 신설 | **채택** + **S17** | §8.9 · §9.8 | §7.3 · §7.4 |
| **R-10** | 중간보스 보상 필드의 거처 = `bosses[]` | **채택** | §8.9 | §7.1 |
| **R-11** | `midBossElementRule` = 런타임 주입 | **채택** + **S15** | §8.9 | §7.2 |
| **R-12** | `movePattern` 어휘 3종 + `mobility` 파괴의 정의 | **채택** — 원안 그대로. + **S18**(`holdCenter` ⇒ mobility 금지) 추가 | §8.12.1 | §10.1 · §10.4 |
| **R-13** | 부위 파괴 = 그 부위의 이미터 정지 | **채택** | §8.12 | §9.2 |
| **R-14** | `stages.formations` 6종 파라미터 | **채택** — ★ **단 정본이 자기 규격으로 확정**했다(`gapPx`/`radiusPx`/`spanDeg`). **이 섹션의 파라미터는 폐기** | §9.9.2 | §5.7 |
| **R-15** | 위기 서브웨이브 간격 = 파생값 | **채택** — 그리고 이 4.17초가 **정본 §7.12.3의 근거**가 되었다 | §9.9.3 | §8.2 |
| **R-16** | 「보스 6종」 → 「저작 7 / 런당 6」 | **채택** | §13.6 · §8.12 | §10 |
| **R-17** | `bands[].sizePx` 삭제, `radius`가 단일 진실 | **채택** | §9.7 | §2.3 |

> ★ **기각이 0건이었다.** 정본 §19의 진단: *"섹션들은 거의 전부 옳았다. 문제는 요청의 품질이 아니라 **요청을 심사할 사람이 없었다**는 것이다."* — 그러나 **채택이 곧 무사통과는 아니었다**: 정본은 이 섹션의 **본문에 이미 쓰여 있던 값**(요청으로 올리지 않은 것)을 §14.1에서 뒤집었다.

### 14.1 ★ 정본 v1.1이 뒤집은 것 — 이 섹션이 승복한 11건 (정본 §18.3)

| # | 뒤집힌 것 | 정본의 근거 | 이 문서의 처분 |
|---|---|---|---|
| 1 | **참조 화력 곡선 45/85/150/240/360/520** | **가정이었다.** 정본은 03의 **실제 무기 수치**에서 산출 → `50/90/195/335/485/708`. 그리고 **520은 실효, 708은 명목** — 두 문서가 **다른 단위로 같은 것을 말하며 부등호로 이었다** | §9.5 전면 폐기 → §1.4가 인용 |
| 2 | **HP 배분 「armor 총합 = 코어의 20~50%」** | ★ **정확히 반대 방향이다.** 코어는 R1에 의해 속성 무관 → 코어에 HP를 몰면 **속성 투자의 레버리지가 죽는다**. `armorCoreRatio` = **4.90**(490%) | §9.6 전면 폐기 → §9.1.1이 승복 |
| 3 | **보스 7종 HP 전량** | §13.6.2·§13.6.3이 소유 | §10 · §11.4 전면 재산출 |
| 4 | **armor 1 보스 3종** (`kiln`·`scarab`·`mire`) | `armorPartCountRange` `[1,2]` → **`[2,2]`**. armor 1은 `specialistPass`(≤0.72)와 `noElementPass`(0.15)를 **충돌**시킨다 | §10.0 재구성 + R1~R7 재검증 |
| 5 | **「코어 직행이 성립하는 유일한 보스」 서사** | **「스테이지 1~2는 전 보스에서 성립, 3부터 불가」**가 낫다 — 학습 이월 + **테마 셔플 하에서 운이 사라진다** | §10.3 서사 교체 |
| 6 | **`bossHpScale` 제안값 `[1.0,1.9,3.4,5.6,8.4,11.5]`** | 이 섹션의 DPS 곡선에서 파생됐으므로 함께 폐기 | §1.4가 정본값 인용 |
| 7 | **중간보스 HP 스케일 = `enemyHpScale`** | `bossHpScale`이 옳다 — `enemyHpScale`이면 `mbNest`가 s5에서 **9.3초**에 죽어 「DPS 체크」가 소멸. ★ **v1.1의 「base 900/750/1100은 그대로 유효」는 v1.2에서 뒤집혔다** → **720/600/880**(§14.2-2) | §7.1 정정 · §7.5 검산 신설 |
| 8 | **`coinScarcity` ⚠ 「조정 손잡이 = 후보 웨이브 수」** | ★ **이 섹션이 가리킨 §6의 제목은 「아무것도 정하지 않는다」였다 = 존재하지 않는 손잡이.** 손잡이 = **`elite.coin`**(정본 소유), 분산은 **S23**이 봉쇄 | §6 전면 재작성 |
| 9 | **「새떼 XP = 스테이지 총 XP의 47%」** | **S22**가 ≤0.30 강제 → **웨이브 편성의 총 개체 수를 늘려야 한다**. ★ **v1.1이 함께 든 근거 「정본 §8.10의 ~25%가 확정」은 v1.2에서 폐기**됐다(§14.2-5) — 47%가 위반이었던 근거는 **S22 하나**이며 그것으로 충분했다 | §2.3(`swarmXp 1`) · §5.3(블록 4개) · §8.4 |
| 10 | **`driftHoming` 행의 셀 오정렬** | 10열에 9값 → `accel 30 / turnRateDegSec 0.6`으로 읽혀 **유도가 사실상 죽는다** | §3.1 정정 |
| 11 | **`holdCenter` 보스의 `mobility` 부위** | **S18** — `ampPx`가 작아 파괴 효과가 무의미 = **트레이드오프가 거짓인 부위** | `mire.tendril` → `armor`(§10.4) |

**★ 이 섹션이 스스로 고친 것 (정본이 지적하지 않았으나 v1.1의 규칙이 강제한 것)**

| # | 결함 | 강제한 규칙 | 처분 |
|---|---|---|---|
| a | ★ **`desert`·`forest` 로스터에 `turret`+`bruiser`가 0종** → **웨이브 코인 수입이 구조적으로 0** | **S23** (신설) | `columnAnt`→`turretPod` / `hexer`→`stalker` (§4). **둘 다 테마 정체성을 강화한다** |
| b | ★ **`stalker`를 "정본 §9.7이 직접 저작한 값"이라 인용** | **C-7** (신설) — §9.7의 블록은 **전부 예시** | §2.3에서 지위 정정 (`core.hp: 4000` 오독과 **같은 종류**) |
| c | ★ **`finale` 앞 14웨이브가 12종만 조우** (`stalker`·`turretPod`·`mortarHulk` 누락) | `mobPhaseMaxWaves` 14 | 블록 순서 재배치 → **14종**(= 산술적 최대, §11.2) |
| d | **`formations` 파라미터를 스스로 저작** | **C-1** — R-14는 채택됐으나 **정본이 자기 규격을 인쇄**했다 | §5.7 폐기 → 정본 §9.9.2 인용 |

### 14.2 ★ v1.2가 통보한 10건 — 전건 이행 (정본 §20.3-04)

| # | 정본 v1.2의 통보 | 이 섹션의 처분 | 값이 바뀌었나 |
|---|---|---|---|
| 1 | **보스 HP 전량 ×1.20** — `core` 815 → **976** / `armor` 1,997 → **2,391** / 선택 478. **테트라크 불변** | §9.5 · §10.1 · §10.2 · §10.3 · §10.4 재인쇄 + **전 산술 재계산** | ✔ 전량 |
| 2 | **중간보스 base ×0.80** — `mbNest` 1100 → **880** (`bossHpScale[5]` 11.9 → 14.42가 37.3초를 만들었다) | §7.1 · §7.2 재인쇄 · §7.5 **전수 재검산 + 모델 정정**(`m` 1.62 → **1.0**) | ✔ 720 / 600 / 880 |
| 3 | **`bossHpScale` = `[1.00, 1.80, 4.67, 6.96, 14.42, 14.42]`** | §1.4 인용 교체 | ✔ |
| 4 | **`boss.optionalPartArmorRatio` = 0.20** (개명) | §0 · §9.5 · §9.6 · §10.1 · §12.1-G3 인용 정정 | ✖ **이 섹션은 이미 0.20만 인용했다** — 이름만 |
| 5 | **「새떼 = 총 XP의 ~25%」 폐기** · 실측이 정본에 기입 · **S22는 상한이지 목표가 아니다** · **04의 편성은 그대로 유효** | §8.4 재작성. ★ **그리고 재계산하다 이 섹션의 분모 오류를 찾았다** → **N-6** | ✖ 편성 불변 |
| 6 | **`bands.chaff.xpRef` = 2 신설** — 이 섹션의 `chaffXpRef`가 정본에 흡수 | §2.3 = 「이 섹션이 확정」 → **인용**으로 강등. **N-1 소멸** | ✖ |
| 7 | **`sea` s1 = 10웨이브가 정본에 반영** · 최대 순환 **1.4회차** | §13 인용 갱신. **N-2 소멸** | ✖ |
| 8 | **`fairness.crisisWaveResidualMax` = 10 신설** — *"04 §8.3의 실측(최악 ≈ 8)이 이 값을 지지한다"* | ★ **§8.3에 그 실측을 실제로 신설했다. 지지하지 않는다** → **N-4 (blocker)** | ✖ (요청) |
| 9 | **`bosses[].leaveAfterSec`는 존재하지 않는다** | §7.2 표에서 **행 삭제**. `moveParams`의 잔재는 **N-5** | ✔ 행 삭제 |
| 10 | **코인 계약이 밴드가 됐다** (66~88기) · **04의 편성은 그대로 유효** | §6 표 갱신. **N-3 소멸** | ✖ |

> ★ **10건 중 값이 바뀐 것은 4건이고 전부 「정본이 소유하는 수」다.** 이 섹션이 **저작한 값**(로스터·웨이브 편성·`radius`·`anchor`·`shapeId`·`score`·시드 이미터·서브웨이브)은 **한 자리도 안 바뀌었다** — 그것이 정본 §13.5.1이 「구조 불변, 숫자만」이라 부른 것의 이 섹션에서의 실물이다.

### 14.3 ★ 남은 정본 추가 요청 — 3건 (**blocker 1 · minor 2**)

> v1.1의 3건(N-1·N-2·N-3)은 **정본 v1.2가 전건 채택**해 소멸했다(§14.2의 6·7·10). 아래 3건 중 **N-4·N-6은 v1.2의 개정 자체가 만든 것**이고 **N-5는 v1.2가 닫은 결함(§21-A12)의 한 단계 아래**다. **어느 것도 이 섹션의 편성·저작값을 바꾸지 않는다.**

| # | 등급 | 항목 | 요청 | 근거 |
|---|---|---|---|---|
| **N-4** | ★ **blocker** | **`fairness.crisisWaveResidualMax = 10`이 이 섹션의 최대 웨이브 레코드(16)보다 작다 → `capHits == 0`이 깨진다** | ★ **10 → 16으로 개정.** 16은 자유 숫자가 아니라 **`max(stages[].waves[].count)`**이며 **S26이 이미 그 수를 읽는다.** 따라오는 것: A층 `enemies` = **70 + 16 = 86 < 96**(여유 11.6%). 정본의 「여유 20%」 관례를 지키려면 `caps.enemies` **96 → 104**(= 86 × 1.2). ★ **부수 요청: 「중간보스 소환물이 「웨이브 잔존」에 포함되는가」를 한 줄 확정할 것** (`mbNest`의 drifter가 위기 진입 시 5~6기 남는다) | ★ **정본 §12.1이 이 값의 근거로 *"04 §8.3의 실측(최악 ≈ 8)"*을 들었으나 **이 섹션에는 그 계산이 없었다.** 지금 했다(§8.3): 정본 §8.7의 스포너 규칙(`다음 스폰 = max(직전 + 9.0, 직전 전멸)`)에 의해 **한 번에 살아 있는 웨이브는 1개**이므로 잔존 = 그 웨이브의 생존자이고 **산술적 최악 = 최대 레코드 = 16**이다. **passive 정책의 스테이지 1에서 실제로 도달한다**(웨이브가 자체 이탈로만 소멸 → 주기 10.9초 → t=95에 나이 8.1초 → 16기 전원 생존). → **`defer` 발화 → `capHits > 0` → certify 실패.** ★ **이 섹션이 고칠 수 없다**: `T1a`/`T1b`의 `s1 = 4u`는 `u = 3~4`에서 12~16기이고, `u`는 §5.3의 Σ XP 정규화가 정하며 **T1 블록의 슬롯 분할은 G-19-①이 금지**한다(정본 §8.2.1의 조우 편차 산술을 보존하는 조건). **그리고 고칠 필요가 없다** — 잔존 16은 **B층 96에 62%의 여유**를 남긴다. ★ **16으로 올리면 잔존이 그 수를 넘는 것이 산술적으로 불가능해져 `defer`가 영원히 발화하지 않는다 = `capHits == 0`이 자명해진다** |
| **N-5** | **minor** | **`tier:"mid"`의 `anchor` `moveParams.leaveAfterSec`가 두 번째 거처다** | **「`tier == "mid"`인 보스는 `moveParams.leaveAfterSec`를 저작하지 않는다(엔진이 `stages.phase.midBossLeaveAfterSec`를 읽는다)」를 §8.4 또는 §9.8에 한 줄** 확정할 것 | v1.2가 `bosses[].leaveAfterSec`를 **이중 거처**로 삭제했다(§21-A12). 그러나 `anchor`의 파라미터 목록(정본 §8.4)에 `leaveAfterSec`가 있으므로 **`mbHammer`·`mbNest`의 `moveParams`에 30이 그대로 남는다.** ★ **`mbLancer`는 `charge`(그 파라미터가 **없다**)인데도 30초에 이탈한다 → 엔진은 `tier:"mid"`에서 `midBossLeaveAfterSec`를 읽을 수밖에 없다 → `anchor`의 30은 「읽히지 않는 두 번째 거처」**이며 §21-A12가 닫은 것과 **정확히 같은 클래스**다(한 단계 아래에서). **지금은 로드 실패가 아니다**(값이 같아 부팅된다) → **minor.** 위험은 **드리프트**: 30이 개정되면 `moveParams`가 조용히 갈라진다. 이 섹션은 **생략 시 「누락 키 = 에러」에 걸릴 수 있으므로 30을 유지**했다(§7.2) |
| **N-6** | **minor** | **정본 §8.10의 헤드라인 「실측 12~13% (04 §8.4)」가 정본 자신의 재검산(8.8%)과 갈라진다** | **「~9~10%」로 정정** (C-8) | ★ **원인은 이 섹션이다.** v1.1의 §8.4가 지분을 `S22/(1+S22)`로 계산했는데 **그 분모는 웨이브 + 위기뿐**이고 **중간보스 XP와 엘리트 XP가 빠져 있다**(= S22의 분모를 「총 XP」로 착각했다). 정본이 그 틀린 수를 **헤드라인으로 인용**하면서 **자기 재검산(`sea` s4 = 8.8%)은 같은 절에 인쇄**했다 → **같은 양에 두 값.** 이 섹션의 재계산(§8.4): **전 테마·전 스테이지 ~9~10%**, `sea` s4 = **8.9%**(정본의 8.8%를 0.1%p 차로 재현). ★ **게이트 무영향** — S22의 분모는 「저작 리스트 Σ XP」이고 지분(%)은 어느 게이트도 읽지 않는다. **그러나 「~25%를 폐기하고 실측을 기입한다」가 v1.2의 판정 7이었으므로, 기입된 수가 틀리면 판정이 절반만 이행된 것이다** |

### 14.4 이 섹션이 정본을 바꾸지 **않은** 것 (확인)

- **테마 커버리지 보장** — 추첨 구조가 이미 증명한다(§1.1). 보정 장치를 만들지 않았다.
- **엘리트 정의** — 정본 §8.6을 전면 인용했고 값을 하나도 정하지 않았다. 이 섹션이 더한 것은 **저작 규칙 G-12 하나**(런타임 거부를 저작 시점으로 이동).
- **코인 손잡이** — ★ **`elite.coin`은 정본이 소유한다**(§13.2-⑪). 이 섹션은 실측만 보고한다(§6).
- **중간보스 3종·이탈 30초·속성 규칙·보상** — 정본 §8.9 그대로. 이 섹션은 `moveParams`·이미터·`radius`·`shapeId`·`score`만 채웠다. **HP base는 유효하나 스케일은 정본이 정정**했다.
- **위기 세션의 모든 규칙** — 정본 §8.10 그대로. 이 섹션은 6 서브웨이브의 9:1 분산만 정했다. **`swarmXp`는 정본의 파생값이며 손잡이가 아니다.** ★ **`bands.chaff.xpRef`(2)도 이제 정본 소유**이며, 이 섹션의 의무는 **`drifter.xp = 2`를 지키는 것**뿐이다(§2.3).
- ★ **`certify.dpsRef`의 farm 정책(`maxFarm`) · `certify.runFarmDpsRatio`(0.83) · `killTimeMedianBalanced` 밴드([120,150])** — **전부 정본 §13.5.1 · §13.6.5 소유.** 이 섹션은 **probe 기준으로만 격파 시간을 인쇄**하고 baseline 환산(÷0.83)은 하지 않는다.
- ★ **중간보스 base HP(720/600/880)** — v1.1까지 이 섹션의 저작값이었으나 **v1.2에서 정본 §13.6.4가 소유**한다(「`mbNest` = `midBossLeaveAfterSec`에 아슬아슬 → base 880이 그 값이다」). 이 섹션은 **`moveParams`·이미터·`radius`·`shapeId`·`score`·`contactDmg`만** 저작한다.
- ★ **A층 예산 4종(`enemyConcurrentMax` · `swarmConcurrentMax` · `crisisWaveResidualMax` · `telegraphConcurrentMaxGlobal`)** — 정본 §12.1 소유. 이 섹션은 **실측만 보고**하며, `crisisWaveResidualMax`가 편성과 충돌하는 것도 **값을 고치지 않고 요청으로 올린다**(§14.3-N-4).
- **코어 게이트·R1~R7·`partType` 4종·`movePattern` 3종·`phaseThresholds`·부위 XP 0** — 정본 §8.11~§8.14 그대로.
- ★ **보스 HP·`armorCoreRatio`·★ `boss.optionalPartArmorRatio`·`bossHpScale`·`dpsRef`·`uptimeRef`** — **전부 정본 소유**(§13.5·§13.6). 이 섹션은 **인용과 검산만** 한다.
- ★ **`stages.formations` 파라미터** — 정본 §9.9.2 소유. G-18의 수용량 상한은 그 값에서 **파생**한 것이지 새 값이 아니다.
- ★ **최종 로테이션·위기 예고의 시각 규격** — **정본 §7.12.2·§7.12.3이 「만들지 않는다」로 확정**했다. 이 섹션은 새떼의 `element`만 저작한다.
- **`charge`를 잡몹에 주지 않은 것** — 정본 §8.9가 이미 `mbLancer`에 배정했다. 어휘 커버리지를 위해 억지로 잡몹에 넣으면 §8.4의 무기 짝 논거를 밟게 된다(§2.1).
