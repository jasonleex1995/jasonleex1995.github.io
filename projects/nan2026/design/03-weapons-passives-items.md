# 무기 · 패시브 · 아이템 로스터 (섹션 문서) — **정본 v1.1 정합판**

> **정본 종속.** 이 문서는 `design/CANON.md` **v1.1**을 **참조만** 한다. 정본의 값·공식·어휘·스키마를 재정의하지 않는다(C-1, C-2).
> **위임 범위 (정본 §17)**: 12 패밀리의 `base`/`levels[8]` 실제 숫자 · 진화 12종의 `evo*` 파라미터 값 · 패시브 `values` 미세 조정.
> ★ **`dpsRef` 곡선과 보스 HP는 정본 소유(§13.5 / §13.6)** — 이 문서는 **입력을 제공했을 뿐 소유하지 않는다.**
> 이 문서가 **새로 정한 것**은 전부 §9 「정본 추가 요청」에 열거된다. 그 밖의 신규 결정은 없다.

---

## 0. 이 문서가 서 있는 땅 (정본 v1.1 인용 — 재정의 아님)

| 항목 | 정본 위치 | 인용 |
|---|---|---|
| **★ 인쇄된 스키마 예시의 지위** | **C-7** | **§9의 JSON 블록은 전부 예시이며 숫자에 구속력이 없다.** 확정값은 **표로 인쇄되고 "확정"이라 표시된 것**뿐이다 |
| **★ 확정값의 거처** | **C-8** | 정본의 표에 인쇄된 숫자 = `data/*.json`이 반드시 가져야 하는 값. **섹션은 인용만 한다** |
| **★ DPS 기준선 `certify.dpsRef`** | **§13.5** | **50 / 90 / 195 / 335 / 485 / 708** (스테이지 1~6, **명목 · 단일표적 · 무속성 · 보스전 진입 시점**). **정본 소유** |
| **★ 실효 가동률 `uptimeRef`** | **§10.4.3** | **0.60.** `effectiveDps = dpsRef × uptimeRef × m`. **명목 ≠ 실효** — 이 단위가 「520 vs 708」의 진짜 원인이었다 |
| **★ 보스 HP** | **§13.6** | 테마 보스 `core` 815 / `armor` ×2 각 1,997 / 선택 부위 399 · `bossHpScale [1.0,1.8,4.4,7.5,11.9,17.3]` · `tetrarch` 총 **88,300**. **정본 소유** |
| **★ `rules.passiveHooks`** | **§9.6.1** | 훅 4종 → 12 패밀리 파라미터 매핑 12행 **동결** + 규칙 H1~H4. **정본 소유** (이 문서의 요청 9.3이 채택된 결과) |
| id 네임스페이스 | §9.5 | `id == family`, 12종: `forward fan seeker lance orbit aura mine boomerang barrage omni drone nova` |
| 레벨 구조 | §9.5 | `base` + `levels` **정확히 8행**, 부분 오버라이드 (§9.3의 유일한 예외) |
| **계약 = `base`의 필수 키 집합** | §9.5 | **채택됨.** 「누락 키 = 에러」의 대상은 **그 패밀리의 계약(공통 9 + 고유 n)**이지 공통 목록 전체가 아니다 |
| 진화 트리거 | §9.5 | **Lv7 → Lv8 레벨업 카드 그 자체가 진화 카드다** |
| 진화 코드 표현 | §9.5 | `w.evolved` 불리언 1개 + `if (w.evolved)` 분기 **정확히 1개** + 계약에 선언된 `evo*` 파라미터 |
| `family` 변경 | §9.5 | **금지** |
| **`evoSecondaryDmgMul`** | §9.5 | **`fan`·`mine`·`nova` 계약에 존재한다** (채택됨). 코드 리터럴 금지 |
| **`evoPullForce`의 정의역** | §9.5 | **`chaff` 밴드 + 새떼(`swarm*`)에만.** `line`·`turret`·`bruiser`·엘리트·중간보스·보스·**보스 부위 전부 제외** (채택됨) |
| **`pierce: -1` = 무제한 관통** | §9.5 · §9.6.1 | **채택됨.** `-1`은 `weapons/**`의 허용 리터럴. `pierceAdd`는 `-1`에 적용되지 않는다 |
| **`rng.pattern`을 플레이어 무기가 사용** | §9.5 · §10.2 · S11 | **허용 확정.** `forward.jitterDeg`가 사용. 데미지 경로가 아니다 |
| **플레이어 무기는 `fairness` 대상이 아님** | §9.5 | **`fairness.playerWeaponsExempt: true`.** S6이 검사하는 것은 `enemies.json > emitters`뿐 |
| **플레이어 텔레그래프의 시각 규약** | **§7.12.6** | **현재 스탠스 색 · 검은 외곽선 금지 · 자홍 절대 금지 · 레이어 3**(지면형 `barrage`도 3) · additive + `render.playerFxCompositeAlpha` 하드캡 |
| 각인 | §4.4 | `live` = `orbit` `aura` 둘뿐. 나머지 10 패밀리 = `spawn` |
| 데미지 공식 | §3.1 | `base × dmgMul × elem × gate`. **크리티컬 없음 · 데미지 난수 없음 · 적 방어력 없음** |
| 지속형 `dmg` | §3.1 | **1회 적용당 값**. DPS 정규화 없음 |
| 패시브 스탯 12종 | §9.6 | 폐쇄 어휘. `values` = **각 레벨의 절대 총량** |
| 상점 10항목 | §11.2 | `price(item, n) = ceil(basePrice × growth^n)` |
| **`bands[].coin`** | §9.7 | **신설 채택.** `chaff 0` / `line 0` / `turret 1` / `bruiser 2` |
| 성장 예산 | §11.1 | 총 싱크 **67** > `maxLevelUps` **60** (= 3 + 28 + 6 + 30) |
| 시작 상태 | §2.6 | `forward` Lv1 → 슬롯 1. 나머지 3칸 빈칸 |
| 플레이어 탄 캡 | §12.1 / §12.2 / §9.4.2 | `caps.playerBullets 256` · `drones 8` · `zones 64` · `render.playerBulletMaxRadiusPx 10` |
| 무기 아트 | §9.10 | 속성 글리프(●▲◆✚) × `projRadius` × 궤적. **무기별 별도 아트 0** |
| `bossPartPriority` · `knockback` · `zone.dps` · 진화 `flags` · 그레이즈 | §9.5 / §3.1 / §2.3 | **존재하지 않는다** |

**단위 (§0.2)**: JSON의 모든 시간 = `Sec` = **게임초**. 이 문서의 모든 DPS = **게임초당 데미지**. 배속과 무관.
★ **그리고 명목/실효를 구분한다** — 이 문서의 「ST DPS」는 전부 **명목**(§13.5의 정의)이다. 실효로 바꾸려면 `× uptimeRef 0.60`을 곱한다. **두 수를 부등호로 잇지 않는다.**

### 0.1 ★ 정본 v1.1이 이 문서에서 뒤집은 것 (승복 — 재논의 없음)

| # | v1.0판 03의 서술 | 정본 v1.1의 처분 | 이 문서의 조치 |
|---|---|---|---|
| 1 | **§8.1 DPS 엔벨로프 스테이지 1~5** (20 / 55 / 130 / 300 / 550) | ★ **폐기** — 「1무기 Lv2~3」이 §11.1의 드래프트 규칙과 모순. **11픽이면 4칸이 찬다** | §8.1을 **정본 §13.5의 인용**으로 교체. 곡선을 만들지 않는다 |
| 2 | **§8.1 「테트라크 총 HP 45,000~65,000」 계약** | ★ **기각** — 「가용 피해의 40~55%」라는 **가정**에서 나왔다. 정본은 **목표 격파시간에서 역산** → **88,300** (§13.6.3) | 사이징 계약을 **삭제**. 보스 HP는 인용만 한다 |
| 3 | **§3 「`seeker` 예시를 한 글자도 바꾸지 않고 채택」의 근거** | **근거 무효 (C-7)** — 그 블록은 예시였다. **결과는 유효** — 정본이 §13.5에서 `shape[]`의 입력으로 **사후 승인** | §3의 근거를 **C-7 기준으로 재작성.** 곡선은 **03의 저작물**임을 명기 |
| 4 | **§5.2 훅 매핑표** (이 문서의 「진짜 산출물」) | **채택 → 정본 §9.6.1이 소유** (12행 동결 + H1~H4) | §5.2를 **인용**으로 교체. 매핑을 이 문서가 정하지 않는다 |
| 5 | §8.5 **「ST 스프레드 1.75배」** | (정본이 §13.2-⑩에 **그대로 인용**했다 — 아래 참조) | ★ **이 문서의 산술 오류.** 150 ÷ 62.9 = **2.38배**. §8.5에서 정정하고 **§9.2-N2로 정본에 통보** |
| 6 | §4.4 `lance` **군중(column 5기) Lv1 34.5 / Lv8 692.5** | (정본 무관 — 이 문서 내부 모순) | ★ **자체 정정.** Lv1은 `pierce 3` = **4기**, Lv8은 §4.4 자신의 문장(「잡몹은 1열만」)과 충돌 → **27.5 / 231** |
| 7 | §3 **R-W1 상단 8.9 · R-W2 「×3.2」 · R-W6 「×1.8 이상」** | (정본 무관 — 이 문서 내부 모순) | ★ **자체 정정.** Lv1 최댓값은 `orbit` **8.3**(8.9는 seeker Lv2) / **×3.15** / 계단 임계 **×1.6** |

> ★ **5·6·7은 정본이 지적하지 않았다.** 정본 §19의 심사가 이 문서의 **요청 10건**을 봤을 뿐 **본문의 산술**을 재검산하지는 않았기 때문이다(정본 §19.5가 같은 성격의 공백을 스스로 인정한다). **요청이 통과했다고 본문이 옳은 것은 아니다** — 이 개정에서 전 수치를 재검산했고 그 결과가 위 3건이다.

---

## 1. 무기 특색 원칙 — "거동이 다르면 자세가 달라야 한다"

로스터의 합격 조건은 "12종이 다르다"가 아니라 **"12종이 나를 다른 곳에 세운다"**이다. 아래 표의 **「서 있어야 할 곳」 열에 중복이 없다**는 것이 이 로스터의 유일한 검증 기준이며, §8.2에서 기계적으로 재확인한다.

| `id` | 이름 | **서 있어야 할 곳** | **해야 할 일** | **하면 안 되는 것** | 짝지어진 적 `moveId` (§8.4) |
|---|---|---|---|---|---|
| `forward` | 벌컨 | 적의 **바로 아래, 정면 정렬** | 물고 늘어져 램프를 올린다 | **피격**(램프 리셋) | `dive` `charge` |
| `fan` | 팬아웃 | 전방 **근거리(정렬 불요)** | 밀착해 다발을 한 몸에 박는다 | 원거리에서 쏘기(1발만 맞음) | `weave` · 밀집 편대 |
| `seeker` | 시커 | **어디든** | 회피에만 전념한다 | — (안전 바닥) | 전 거동 |
| `lance` | 랜스 | **종대의 축 위** | 세로로 줄을 세워 꿴다 | 흩어진 적에게 쏘기 | `column` |
| `orbit` | 오빗 | **적 속(파고들기)** | 몸으로 문지른다 | 도망치기(공전체가 안 닿음) | `orbitDrift` `anchor` |
| `aura` | 펄스필드 | **밀집의 한가운데** | 중심에 서서 버틴다 | 산개한 적을 상대하기 | 새떼 · `arc`/`vWedge` |
| `mine` | 마인필드 | 적이 **올 자리** | 미리 깔고 그 자리를 뜬다 | 적을 쫓아다니기 | `dive` `column` |
| `boomerang` | 리턴 | **횡단 라인에 수직** | 던지고 되받는다 | 세로로 정렬하기 | `strafe` · `pincer` |
| `barrage` | 바라지 | **아무 데나** | 회피에 100% 전념한다 | 조준하려 애쓰기 | 전 거동 |
| `omni` | 리어가드 | **어디든(사각 없음)** | 뒤를 신경 끈다 | — | `rearIn` `strafe` |
| `drone` | 옵션 | **드론 사거리 안에 적을 두고 나는 자유** | 화력을 내 몸 밖에 배치한다 | 드론 사거리 밖으로 도망 | `anchor` |
| `nova` | 노바 | **밀집 중심, 폭발 주기에 맞춰** | 터질 때 그 자리에 있는다 | 주기를 무시하고 이동 | 새떼(위기 세션) |

**"어디든" 3종의 분리 (특색 붕괴 방지)**
- `seeker` = **무조준 + 자동추적**. ST가 안정적이고 낮다. → *회피에 전념하는 값*.
- `barrage` = **무조준 + 광역 + 착탄 지점 불확정**. ST가 불안정하다. 진화(`densest`)에서만 확정적이 된다. → *방치의 값*.
- `omni` = **무조준 + 전방위**. ST가 로스터 최저권, 군중 커버리지 최고. → *사각을 없애는 값*.

세 무기 모두 "서 있어야 할 곳"이 없다는 점이 같지만, **대신 포기하는 것이 각각 다르다**(안정성 / 확정성 / 단일 화력). 이것이 특색이다.

**로스터 전체를 관통하는 대가 축 (설계 의도)**

```
조준 부담이 클수록 ST DPS가 높다.   forward(정렬) > boomerang·drone(반정렬) > seeker(무조준)
군중 화력이 클수록 ST DPS가 낮다.   aura·omni·fan(원거리) << seeker << forward
리스크를 질수록 보상이 크다.        orbit(파고들기 = 몸통 데미지 감수) → ST 중상위
```

---

## 2. 파라미터 계약 (패밀리별 폐쇄 목록 — 정본 §9.5 표의 전개)

> 정본 §9.5는 **공통 파라미터 9 + 고유 파라미터 + 허용 `targetMode`**를 선언하고, ★ **「계약 = `base`의 필수 키 집합」을 확정했다**(이 문서의 요청 9.1 채택). 아래는 그것을 **패밀리별 실제 키 목록으로 전개**한 것이다. **어휘를 추가하지 않았다.**
>
> 정본의 근거 문장이 전개의 규칙을 이미 준다: *"`orbit`이 `cooldownSec`·`projSpeed`·`lifetimeSec`·`pierce`·`count`를 전부 0으로 선언해야 한다 = 죽은 필드 5개"* → **그 패밀리가 쓰지 않는 공통 키는 계약에 없다.** `targetMode`가 없는 5 패밀리는 정본이 직접 열거했다.

| `family` | base 필수 키 | `evo*` (진화 전용, §9.5 계약) |
|---|---|---|
| `forward` | `dmg cooldownSec count projSpeed projRadius lifetimeSec pierce hitCooldownSec targetMode spreadDeg jitterDeg burstCount burstIntervalSec` | `evoRampSec evoRampFireRateMul` |
| `fan` | `dmg cooldownSec count projSpeed projRadius lifetimeSec pierce hitCooldownSec targetMode arcDeg` | `evoBlastRadius evoSecondaryDmgMul` |
| `seeker` | `dmg cooldownSec count projSpeed projRadius lifetimeSec pierce hitCooldownSec targetMode turnRateDegSec acquireRadius retargetSec` | `evoDistinctTargets evoRetargetOnKill` |
| `lance` | `dmg cooldownSec count pierce targetMode beamWidthPx chargeSec rangePx` | `evoFullHeight` |
| `orbit` | `dmg hitCooldownSec projRadius orbitRadius angularSpeedDegSec bodyCount` | `evoBulletClearCooldownSec` |
| `aura` | `dmg radius tickIntervalSec falloff` | `evoPullForce` |
| `mine` | `dmg placeIntervalSec armSec triggerRadius blastRadius maxAlive` | `evoClusterCount evoClusterRadius evoSecondaryDmgMul` |
| `boomerang` | `dmg cooldownSec count projSpeed projRadius lifetimeSec pierce hitCooldownSec targetMode outRangePx returnSpeed canRehit` | `evoChainCount` |
| `barrage` | `dmg cooldownSec targetMode strikeIntervalSec strikesPerVolley blastRadius telegraphSec` | `evoRadiusMul` |
| `omni` | `dmg cooldownSec projSpeed projRadius lifetimeSec pierce dirCount dirOffsetDeg rearBias` | `evoRingRotDeg` |
| `drone` | `dmg projSpeed projRadius lifetimeSec pierce targetMode droneCount anchorOffsets droneFireSec droneRangePx` | `evoTrailDelaySec` |
| `nova` | `dmg intervalSec radius expandSec telegraphSec` | `evoRing2Radius evoClearBullets evoSecondaryDmgMul` |

> ★ **`evoSecondaryDmgMul`은 더 이상 요청이 아니다** — 정본 §9.5가 `fan`·`mine`·`nova` 계약에 편입했다. 이 표는 그것을 인용한다.

**계약 해석 규칙 3개 (기존 어휘의 의미 — 전부 정본에 편입됨)**

| 키 | 확정 의미 | 거처 |
|---|---|---|
| `hitCooldownSec` | **`0` = 한 피해 개체는 한 대상을 정확히 1회만 때린다**(`pierce`는 *서로 다른* 대상의 관통 수). **`> 0` = 같은 대상 재히트 허용, 그 간격.** → `orbit`(문지르기)·`boomerang`(왕복)·`seeker`(관통 후 재접촉)만 `> 0` | 이 문서 (§17 위임: `base` 숫자의 의미) |
| `pierce: -1` | **무제한 관통.** `boomerang` base와 `lance` 진화가 사용. `pierceAdd`는 여기에 적용되지 않는다 | ★ **정본 §9.5 · §9.6.1** (요청 9.2 채택) |
| `targetMode` 부재 | `orbit` `aura` `mine` `omni` `nova`는 이 키를 **갖지 않는다.** `targetMode: null`이 아니라 **키 자체가 없다** | ★ **정본 §9.5** (요청 9.1 채택) |

> **`pierce` N = 최초 1대상 + 관통 N대상 = 최대 N+1 대상.** `seeker` Lv7(`pierce 1`)이 ST를 전혀 올리지 못하는 것이 이 정의의 직접 귀결이다(§3).

---

## 3. 레벨 곡선의 형태 (12종 공통 규칙)

**기준 곡선 = 이 문서가 저작한 `seeker` 곡선이다.**

> ★ **v1.0판의 근거는 무효다 (C-7).** 이 문서는 §9.5의 `seeker` JSON 블록을 *"정본이 인쇄한 유일한 완성 로스터 항목"*이라며 채택했으나, **정본 v1.1의 C-7이 그 블록을 예시로 확정**했다 — 인쇄된 스키마 예시에는 구속력이 없다.
> **결과는 유효하다.** 정본 §13.5가 이 곡선(과 `forward` 곡선)을 **`shape[]` 정규화 곡선의 입력으로 채택**함으로써 사후 승인했기 때문이다. 즉 **이 곡선은 정본의 것이 아니라 03의 저작물이고, 정본이 그것을 골랐다.**

```
seeker ST DPS:  Lv1  6.7 → Lv2 8.9 → Lv3 20.0 → Lv4 26.7 → Lv5 40.0 → Lv6 64.0 → Lv7 64.0 → Lv8 117.3
                (= ×17.5, 레벨당 평균 ×1.51, 계단 3회 — Lv3 / Lv6 / Lv8)
```
> Lv7(`{"pierce":1}`)은 **ST가 오르지 않는 유일한 레벨**이다 — 관통은 *뒤에 줄 선 다른 개체*에게만 값이 있다. **"레벨업은 항상 DPS가 아니다"가 기준 곡선 안에 이미 심겨 있다.**

**★ 정본 §13.5가 이 문서에서 가져간 것 (인용 — 검산은 §8.1)**
```
shape[L] = (seeker[L]/117.3 + forward[L]/150) / 2
         = [0.055, 0.071, 0.152, 0.207, 0.357, 0.545, 0.576, 1.000]     ← 정본 §13.5가 인쇄한 값
정본은 이 shape[]로 스테이지 1~5의 dpsRef를 재산출했다 = 50 / 90 / 195 / 335 / 485.
→ 이 문서의 스테이지 1~5 곡선(20/55/130/300/550)은 폐기됐다. 정의가 달랐다(§0.1-1).
```

| 규칙 | 내용 | 이유 |
|---|---|---|
| **R-W1** | **Lv1 ST DPS ∈ [4.3, 8.3]** — 하단 4.3은 `fan`(원거리)·`aura` 두 군중 무기의 몫, 상단 8.3은 `orbit` | 상단(`orbit` 8.3, `forward` 8.0)이 `chaff`를 1~2히트로 잡는다(§8.6 밴드표). 하단 두 무기는 **Lv1부터 군중 화력으로 값을 낸다**(`fan` 밀착 12.9 / `aura` 반경 내 15기 = 45) — ST가 낮은 것이 약한 것이 아니다 |
| **R-W2** | **Lv8 ST DPS ∈ [47.6, 150]** (**×3.15** 폭) — 폭은 §1의 대가 축이 만든다 | 군중 화력이 높은 무기(`aura` 47.6, `omni` 62.9)가 ST 하단, 조준 부담이 큰 무기(`forward` 150, `barrage` 진화 147.5)가 상단. §8.5가 이 폭이 승률로 번지지 않는 이유를 검산한다 |
| **R-W3** | **`dmg`는 절대 하락하지 않는다** | 카드가 `변하는 파라미터의 before → after`를 강제 표시(§11.1)한다. 숫자가 내려가는 줄이 보이면 그 카드는 정보가 아니라 함정이다 |
| **R-W4** | **`cooldownSec`이 나빠질 수 있는 행은 `levels[7]`(진화 행) 하나뿐** | 진화 카드는 **진화 폼 프리뷰를 표시**(§11.1)하므로 대가가 화면에 있다. 그 외 레벨업은 순수 상승 |
| **R-W5** | **레벨업은 계약 안의 숫자만 바꾼다. 새 거동은 오직 진화에서만 생긴다** | 정본 §9.5. `pierce 0→1`·`count 1→3`은 어휘 안이므로 숫자 |
| **R-W6** | **계단(**×1.6 이상** 도약)은 무기당 2~3회** — 그 무기의 **정체성 축**(ST형은 ST 곡선, 군중형은 군중 곡선)에서 센다 | 기준 곡선(seeker)이 Lv3(×2.25)·Lv6(×1.60)·Lv8(×1.83)에서 도약한다. 레벨업 54회 중 **"이번 건 크다"가 무기당 3번** = 페이오프 리듬 |

> ★ **R-W1·R-W2·R-W6의 수치는 v1.0판에서 틀려 있었다** — 상단 8.9(= seeker **Lv2**), ×3.2(실제 3.15), ×1.8(그 임계로는 seeker의 Lv6이 계단이 아니게 되어 §3 본문과 모순). **정본이 지적하지 않은 이 문서 자신의 결함**이며 이 개정에서 닫았다.

> **왜 곡선을 공식이 아니라 8행 배열로 쓰는가**: 시뮬이 `levels[3]`에 키 하나를 추가해 **그 레벨만 핀포인트 튜닝**할 수 있다(§9.3의 유일한 예외). 공식이면 한 레벨을 고치는 순간 8레벨이 전부 움직인다.

---

## 4. 무기 로스터 12종 (`data/weapons.json`)

> 각 항목의 DPS 수치는 **`data`가 아니라 검산**이다. JSON에 들어가는 것은 `base`/`levels`/`evolution`뿐이다.
> **ST** = 단일 대상 **명목** DPS(§13.5의 정의). **군중** = 밀집 대상 전체 합산 명목 DPS.

### 4.1 `forward` 벌컨 — spawn · targetMode `forward`

**자세**: 적의 바로 아래로 파고들어 정면을 고정하고 **물고 늘어진다.** 진화하면 안 맞고 버틴 시간이 곧 화력이 된다.

```json
{ "id":"forward", "family":"forward", "name":"벌컨", "desc":"정면으로 연사한다. 물고 늘어질수록 빨라진다",
  "elementStampMode":"spawn",
  "base": { "dmg":4, "cooldownSec":0.50, "count":1, "projSpeed":420, "projRadius":4,
            "lifetimeSec":1.6, "pierce":0, "hitCooldownSec":0.0, "targetMode":"forward",
            "spreadDeg":0, "jitterDeg":1.5, "burstCount":1, "burstIntervalSec":0.07 },
  "levels": [ {}, {"dmg":5}, {"burstCount":2}, {"dmg":7},
              {"count":2,"spreadDeg":6}, {"dmg":9,"cooldownSec":0.44},
              {"dmg":10,"pierce":1}, {"burstCount":3,"cooldownSec":0.40} ],
  "evolution": { "name":"오버드라이브", "desc":"연사를 유지할수록 발사 속도가 오른다. 피격 시 리셋",
                 "params": { "evoRampSec":3.0, "evoRampFireRateMul":1.55 } } }
```

| Lv | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ST | 8.0 | 10.0 | 20.0 | 28.0 | **56.0** | 81.8 | 90.9 | **150.0** |

- **계단**: Lv3(버스트 2발, ×2.0) · Lv5(2열 = `count 2`, ×2.0) · Lv8(버스트 3발, ×1.65).
- **진화 「오버드라이브」**: `evoRampSec 3.0` 동안 무피격·연사 유지 시 발사 속도가 **×1.55까지 선형 램프** → 최대 **232 ST**(로스터 최고). **피격 즉시 램프 0.** → 로스터에서 유일하게 *"안 맞는 것"이 곧 DPS인* 무기.
- **`projSpeed 420`은 `fairness.maxBulletSpeed`(260)의 대상이 아니다** — 정본 §9.5가 `fairness.playerWeaponsExempt: true`를 명문화했다(요청 9.10 채택). 260의 논거는 **내가 피해야 하는 것**에 대한 것이며, 벌컨의 420은 **회피의 보상**이다.
- `jitterDeg 1.5`는 **`rng.pattern` 스트림을 사용한다** — 정본 §9.5·§10.2·S11이 허용을 확정했다(요청 9.9 채택). **데미지 경로가 아니므로 §3.1의 "데미지 난수 없음"과 충돌하지 않는다.**
- ★ **`autoload`의 1순위 레버가 여기 있다**: `count 2 → 4`가 ST를 **×2**로 만든다(§5.3 · §8.5).

### 4.2 `fan` 팬아웃 — spawn · targetMode `forward`

**자세**: **밀착한다.** 부채꼴이므로 멀면 한 발만 맞고, 붙으면 여러 발이 한 몸에 박힌다. **정렬은 필요 없다** — `forward`와 갈라지는 지점.

```json
{ "id":"fan", "family":"fan", "name":"팬아웃", "desc":"전방 부채꼴로 흩뿌린다. 붙을수록 많이 박힌다",
  "elementStampMode":"spawn",
  "base": { "dmg":3, "cooldownSec":0.70, "count":3, "projSpeed":300, "projRadius":4,
            "lifetimeSec":1.0, "pierce":0, "hitCooldownSec":0.0, "targetMode":"forward",
            "arcDeg":40 },
  "levels": [ {}, {"dmg":4}, {"count":5,"arcDeg":50}, {"dmg":6},
              {"count":7,"arcDeg":62,"cooldownSec":0.62}, {"dmg":9},
              {"dmg":11,"pierce":1}, {"count":11,"arcDeg":90,"dmg":13} ],
  "evolution": { "name":"플레어 팬", "desc":"탄이 소멸할 때 작게 터진다",
                 "params": { "evoBlastRadius":40, "evoSecondaryDmgMul":0.5 } } }
```

| Lv | 1 (`count 3`) | 4 (`count 5`) | 8 (`count 11`) | 8+진화 |
|---|---|---|---|---|
| ST (원거리, 1발) | 4.3 | 8.6 | 21.0 | 31.5 |
| **ST (밀착, `min(count, 4.5)`발)** | 12.9 | 38.6 | **94.4** | **141.5** |
| 군중 (편대 전체) | 12.9 | 42.9 | 230.6 | 346.0 |

> 밀착 명중 수 모델 = `min(count, 4.5)`. **Lv1은 `count`가 3이라 3발 전부가 박힌다** — 시작 직후 팬아웃을 뽑으면 밀착 ST 12.9로 로스터 최고다. 대신 `count`가 늘어도 밀착 명중은 4.5에서 포화하므로 **후반의 성장은 전부 군중 쪽으로 간다**(Lv8 군중 230.6 = 밀착 ST의 2.4배). "붙으면 강하다"가 "붙으면 언제나 최강"이 되지 않는 지점이 여기다.

- **`lifetimeSec 1.0 × projSpeed 300` = 사거리 300px.** 아레나 높이 720에서 **의도적으로 짧다** — "밀착하지 않으면 닿지도 않는다"가 데이터로 강제된다.
- **`arcDeg`는 Lv8에서 90에서 멈춘다.** 150°로 열면 전방위가 되어 **`omni`의 정체성을 침범**한다. 부채꼴은 부채꼴로 끝난다.
- **`arcDeg`는 `areaKeys`에 없다** — 정본 §9.6.1의 **H2**가 산포를 `areaMul`에서 배제했다(이 문서가 제기한 근거 그대로: `coil`로 90→122가 되면 밀착 ST가 26% 깎여 **패시브가 무기를 나쁘게 만든다**).
- **진화 「플레어 팬」**: 탄이 **소멸할 때**(적중 소멸 / 수명 종료) 반경 40 폭발, 데미지 `dmg × evoSecondaryDmgMul`. 밀착 시 폭발이 겹쳐 ST가 ×1.5, 편대 상대로는 폭발이 옆 개체를 물어 실효 관통이 된다.

### 4.3 `seeker` 시커 — spawn · targetMode `nearest`

**자세**: **없다.** 이것이 이 무기의 값이다 — 화면을 보지 않고 **회피에만 100% 전념**할 수 있다. 대가는 ST 기준선(=1.0배).

> ★ **이 항목은 이 문서의 저작물이다.** v1.0판은 "정본 §9.5가 인쇄한 값이므로 그대로 채택"이라고 썼으나 **C-7이 그 블록을 예시로 확정**했다(§0.1-3). 정본 §13.5가 이 곡선을 `shape[]`의 입력으로 채택함으로써 **결과를 사후 승인**했다.

```json
{ "id":"seeker", "family":"seeker", "name":"시커", "desc":"최근접 적을 자동 추적",
  "elementStampMode":"spawn",
  "base": { "dmg":6, "cooldownSec":0.9, "count":1, "projSpeed":190, "projRadius":5,
            "lifetimeSec":2.4, "pierce":0, "hitCooldownSec":0.2,
            "turnRateDegSec":220, "acquireRadius":320, "retargetSec":0.3,
            "targetMode":"nearest" },
  "levels": [ {}, {"dmg":8}, {"count":2,"dmg":9}, {"dmg":12,"turnRateDegSec":260},
              {"count":3}, {"dmg":16,"cooldownSec":0.75}, {"pierce":1},
              {"count":4,"dmg":22} ],
  "evolution": { "name":"스웜", "desc":"각 탄이 서로 다른 타겟을 노린다",
                 "params": { "evoDistinctTargets":true, "evoRetargetOnKill":true } } }
```

| Lv | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ST | 6.7 | 8.9 | 20.0 | 26.7 | 40.0 | 64.0 | **64.0** | **117.3** |

- **이 곡선이 로스터 전체의 ×1.0 기준선**이다(§3). Lv7은 `pierce 1`뿐이라 **ST가 정확히 그대로**이고, 편대에서만 값이 난다.
- 보스전에서 `nearest`는 **가장 가까운 부위**를 노린다 → **플레이어가 위치로 부위를 고른다**(§9.5, `bossPartPriority` 부재). 진화 `evoDistinctTargets`는 4발이 서로 다른 부위로 흩어지므로 **보스전에서는 오히려 집중이 흐려진다** = 진화가 만능이 아닌 유일한 사례. 잡몹 페이즈의 무기다.

### 4.4 `lance` 랜스 — spawn · targetMode `forward`

**자세**: **종대(`column`)의 축 위에 선다.** 줄을 세운 만큼이 데미지다. 그리고 **표적이 클수록 여러 열이 동시에 박힌다** → 대보스 저격.

```json
{ "id":"lance", "family":"lance", "name":"랜스", "desc":"차지 후 관통선을 쏜다. 줄 세운 만큼 꿴다",
  "elementStampMode":"spawn",
  "base": { "dmg":11, "cooldownSec":1.60, "count":1, "pierce":3, "targetMode":"forward",
            "beamWidthPx":10, "chargeSec":0.35, "rangePx":420 },
  "levels": [ {}, {"dmg":15}, {"pierce":5,"rangePx":520}, {"dmg":22,"cooldownSec":1.45},
              {"dmg":32}, {"count":2,"beamWidthPx":12}, {"dmg":46,"pierce":8},
              {"count":3,"dmg":60,"cooldownSec":1.30} ],
  "evolution": { "name":"레일건", "desc":"아레나 세로 전체를 무제한 관통한다",
                 "params": { "evoFullHeight":true } } }
```

| Lv | 1 | 4 | 6 | 8 |
|---|---|---|---|---|
| ST (소형 표적, 1열) | 6.9 | 15.2 | 22.1 | 46.2 |
| **ST (대형 표적, 전 열)** | 6.9 | 15.2 | 44.1 | **138.5** |
| 군중 (`column` 종대) | **27.5** (4기, `pierce 3`) | 76.0 (5기) | 220.5 (5기 × 2열) | **231** (5기 × **1열**) |

> ★ **군중 행은 v1.0판에서 두 곳이 틀려 있었다** (§0.1-6). ① Lv1은 `pierce 3` = **최대 4기**인데 5기로 곱했다. ② Lv8의 692.5는 **3열이 전부 잡몹에 박힌다**고 가정했는데, 그것은 **바로 아래 자기 문장과 모순**이다(폭 48px에서 `radius 14` 잡몹은 1열만 맞는다). **정정 후의 그림이 오히려 정확하다**: 랜스의 Lv6→Lv8 성장은 군중이 아니라 **전부 대형 표적 쪽으로 간다**(대형 44.1 → 138.5 = ×3.1, 잡몹 220.5 → 231 = ×1.05). **"대보스 저격"이라는 정체성이 수치로 강제된다** — `fan`의 「밀착 포화」와 정확히 같은 형태의 장치다.

- **`count`는 평행 관통선의 열 수**다. 열 간격(중심 간) = `beamWidthPx × 2`. Lv8 = 3열 × 간격 24px = **폭 48px** → 잡몹(`radius 14`)은 **1열만**, 보스 부위(`radius 22`)·코어(`radius 42`)는 **3열이 전부 박힌다.** 이 한 줄이 랜스를 "대보스 저격"으로 만든다.
- `chargeSec 0.35`는 **플레이어 텔레그래프**다 — **정본 §7.12.6이 규약을 확정했다**(요청 9.5 채택): **현재 스탠스 색 점선 · 레이어 3 · 검은 외곽선 금지 · 자홍 절대 금지.**
- **진화 「레일건」**: `rangePx` 무시(아레나 상단까지) + `pierce` **무제한**(`-1`). 한 번의 `if (w.evolved)` 분기가 두 효과를 낸다(§9.5 허용).
  > 귀결: **진화 후 `coating`은 랜스에 무효가 된다** — §9.6.1의 「`base.pierce == -1`이면 `pierceAdd` 무효」의 산술적 귀결이다. 숨기지 않는다: 카드가 훅 1줄을 표시하므로 정보는 화면에 있다(H4).

### 4.5 `orbit` 오빗 — **live** · `targetMode` 없음

**자세**: **적 속으로 파고들어 몸으로 문지른다.** 정본 §2.4의 **"모든 피해원이 i-frame을 공유한다"**가 이 무기를 성립시키는 단 하나의 장치다.

```json
{ "id":"orbit", "family":"orbit", "name":"오빗", "desc":"공전체가 몸 주위를 돈다. 닿는 것을 갈아버린다",
  "elementStampMode":"live",
  "base": { "dmg":5, "hitCooldownSec":1.20, "projRadius":7,
            "orbitRadius":60, "angularSpeedDegSec":90, "bodyCount":2 },
  "levels": [ {}, {"dmg":7}, {"bodyCount":3}, {"dmg":10,"angularSpeedDegSec":120},
              {"bodyCount":4,"orbitRadius":72}, {"dmg":14,"hitCooldownSec":1.00},
              {"dmg":18}, {"bodyCount":5,"dmg":20,"hitCooldownSec":0.90} ],
  "evolution": { "name":"이지스", "desc":"공전체가 닿은 적 탄을 소거한다",
                 "params": { "evoBulletClearCooldownSec":1.0 } } }
```

| Lv | 1 | 3 | 5 | 8 |
|---|---|---|---|---|
| ST (파고든 상태) | 8.3 | 17.5 | 33.3 | **111.1** |

- **DPS 모델**: 대상이 궤도 반경 안에 있을 때 각 공전체가 `hitCooldownSec`마다 1회. `bodyCount × dmg / hitCooldownSec`. **파고들지 않으면 0에 가깝다.**
- **파고들기의 비용은 정본이 이미 정했다**: 몸통 충돌 = `contactDmg` 잡몹 6~8 / 보스·부위 14~16, **게임초당 최대 1회**(i-frame 공유), 방어력 최대 8. → **감수할 만한 비용**이지 즉사가 아니다(§2.5). ST 111.1은 그 비용의 값이다.
- `projRadius 7 ≤ render.playerBulletMaxRadiusPx(10)` ✔ — **클램프는 §9.6.1의 H3이 소유**(판정·렌더 동시, 요청 9.8 채택).
- **진화 「이지스」**: 공전체가 닿은 **적 탄을 소거**(공전체당 `evoBulletClearCooldownSec 1.0` 쿨). 파고들기의 캡스톤 — **파고드는 자세 자체가 방어가 된다.**
- `live` 각인(§4.4): 스탠스를 바꾸면 **다음 적용부터 즉시 새 배율.** 화면의 공전체 색이 그 순간 바뀐다 = I-2 준수.

### 4.6 `aura` 펄스필드 — **live** · `targetMode` 없음

**자세**: **밀집의 한가운데에 선다.** 로스터 최저 ST, 최고 군중 화력. **위기 세션(새떼)의 캡스톤 정답 2종 중 하나**(§8.10 · §13.1.1의 `capstone` 정의).

```json
{ "id":"aura", "family":"aura", "name":"펄스필드", "desc":"몸 주위에 지속 피해장을 편다",
  "elementStampMode":"live",
  "base": { "dmg":3, "radius":70, "tickIntervalSec":0.70, "falloff":0.35 },
  "levels": [ {}, {"dmg":4}, {"radius":88,"tickIntervalSec":0.60}, {"dmg":6},
              {"radius":104,"falloff":0.50}, {"dmg":9,"tickIntervalSec":0.50},
              {"dmg":12}, {"dmg":20,"radius":120,"tickIntervalSec":0.42,"falloff":0.60} ],
  "evolution": { "name":"싱귤래리티", "desc":"반경 안의 잡몹을 중심으로 끌어당긴다",
                 "params": { "evoPullForce":90 } } }
```

| Lv | 1 | 3 | 6 | 8 |
|---|---|---|---|---|
| ST (중심) | 4.3 | 6.7 | 18.0 | **47.6** |
| 군중 (반경 내 15기, `falloff` 평균) | ~45 | ~71 | ~191 | **~530** |

- `falloff` = **가장자리 데미지 배율**(중심 ×1.0 → 가장자리 `falloff`, 선형). Lv5·Lv8의 `falloff` 상승이 "장이 단단해진다"의 표현.
- **`dmg`는 1회 적용당 값이고 `tickIntervalSec`가 주기다** — 정본 §3.1. `dps` 개념은 존재하지 않는다.
- **`autoload`가 무효인 2 패밀리 중 하나**(§9.6.1: `countKey: null`) — 자기중심 폭발에 개체 수 개념이 없다. **드래프트가 거르지 않는다**(H4).
- **진화 「싱귤래리티」**: `evoPullForce 90` (PxSec) = **정본 §9.5가 정의역을 확정했다**(요청 9.7 채택): 반경 내 **`chaff` 밴드 + 새떼(`swarm*`)에만** 적용되는 중심 방향 추가 속도. **`line`·`turret`·`bruiser`·엘리트·중간보스·보스·보스 부위 전부 제외.**
  > 근거(정본이 인용): `knockback`을 삭제한 논거(`column`·`anchor`·`pincer` 편대 파괴)가 **`evoPullForce`에 그대로 적용된다.** 제한이 없으면 「싱귤래리티」는 **정본이 삭제한 기능의 뒷문**이 된다.

### 4.7 `mine` 마인필드 — spawn · `targetMode` 없음

**자세**: **적이 올 자리에 깔고 그 자리를 뜬다.** 유일하게 **미래에 데미지를 두는** 무기.

```json
{ "id":"mine", "family":"mine", "name":"마인필드", "desc":"기뢰를 깔아둔다. 지나가면 터진다",
  "elementStampMode":"spawn",
  "base": { "dmg":12, "placeIntervalSec":1.40, "armSec":0.50,
            "triggerRadius":26, "blastRadius":50, "maxAlive":3 },
  "levels": [ {}, {"dmg":16}, {"maxAlive":5,"placeIntervalSec":1.20},
              {"dmg":24,"blastRadius":60}, {"maxAlive":7,"dmg":30},
              {"dmg":42,"placeIntervalSec":1.00,"triggerRadius":32},
              {"dmg":58,"blastRadius":72}, {"dmg":80,"maxAlive":10,"placeIntervalSec":0.90} ],
  "evolution": { "name":"클러스터 마인", "desc":"폭발이 자탄으로 흩어져 2차 폭발한다",
                 "params": { "evoClusterCount":6, "evoClusterRadius":32, "evoSecondaryDmgMul":0.5 } } }
```

| Lv | 1 | 4 | 8 | 8+진화 |
|---|---|---|---|---|
| ST (명중률 ~0.8) | 6.9 | 16.0 | 71.1 | **124.4** |
| 군중 (밀집 5기) | 34 | 80 | 355 | ~890 |

- **`spawn` 각인의 존재 이유가 여기서 가장 선명하다**(§4.4): 불 스탠스로 10개 깔고 물로 전환해도 **기뢰는 불이다. 색도 안 바뀐다.** 각인 세탁이 데이터 구조로 불가능하다.
- `armSec 0.5` = 설치 후 무장까지. `triggerRadius` = 기폭 감지, `blastRadius` = 실제 피해 반경. **감지 < 피해**이므로 적이 밟는 순간 이미 피해 안에 있다.
- **진화 「클러스터 마인」**: 폭발 시 자탄 6개 산개 → 각 반경 32의 2차 폭발, 데미지 `dmg × evoSecondaryDmgMul`(계약 파라미터, §9.5). 예측이 맞으면 밀집 전체가 지워진다 = **흐름 예측의 페이오프.**
- `maxAlive 10` + `autoload` Lv5(+2, `countKey = maxAlive`) = **12 ≤ `caps.zones` 64** ✔

### 4.8 `boomerang` 리턴 — spawn · targetMode `forward`

**자세**: **횡단(`strafe`)하는 라인에 수직으로 선다.** 던지고 되받는 왕복 2히트.

```json
{ "id":"boomerang", "family":"boomerang", "name":"리턴", "desc":"던져서 되받는다. 오갈 때 두 번 벤다",
  "elementStampMode":"spawn",
  "base": { "dmg":4, "cooldownSec":1.10, "count":1, "projSpeed":260, "projRadius":6,
            "lifetimeSec":3.0, "pierce":-1, "hitCooldownSec":0.50, "targetMode":"forward",
            "outRangePx":240, "returnSpeed":320, "canRehit":true },
  "levels": [ {}, {"dmg":6}, {"count":2,"outRangePx":300}, {"dmg":9,"returnSpeed":380},
              {"dmg":12,"cooldownSec":1.00}, {"count":3,"dmg":13},
              {"dmg":16,"hitCooldownSec":0.40}, {"dmg":20,"cooldownSec":0.95} ],
  "evolution": { "name":"체인 리턴", "desc":"귀환 경로가 최근접 적들을 경유한다",
                 "params": { "evoChainCount":3 } } }
```

| Lv | 1 | 3 | 6 | 8 |
|---|---|---|---|---|
| ST (왕복 2히트) | 7.3 | 21.8 | 78.0 | **126.3** |

- **`pierce: -1` = 무제한 관통** (정본 §9.5 채택 규약). 부메랑은 라인을 통째로 지나간다. → `coating`(pierceAdd) **무효**(§9.6.1의 12 중 6 중 하나).
- `hitCooldownSec 0.5 > 0` + `canRehit true` → **같은 적을 나갈 때 한 번, 돌아올 때 한 번.** 이것이 왕복 2히트의 데이터 표현이다.
- **진화 「체인 리턴」**: 귀환 경로가 최근접 적 3기를 **경유**(지그재그), 경유마다 히트 → 밀집 시 실효 ×2. 던지는 방향만 고르면 나머지는 알아서 훑는다.

### 4.9 `barrage` 바라지 — spawn · targetMode `randomInArena` → `densest`

**자세**: **아무 데나. 조준하지 않는다.** 회피에 100% 전념하는 대가로 ST가 불확정하다. **진화가 그 불확정성을 사는 카드다.**

```json
{ "id":"barrage", "family":"barrage", "name":"바라지", "desc":"아레나 어딘가에 포격이 떨어진다",
  "elementStampMode":"spawn",
  "base": { "dmg":40, "cooldownSec":2.20, "targetMode":"randomInArena",
            "strikeIntervalSec":0.25, "strikesPerVolley":1, "blastRadius":45,
            "telegraphSec":0.50 },
  "levels": [ {}, {"dmg":52}, {"strikesPerVolley":2,"cooldownSec":2.05},
              {"dmg":68,"blastRadius":52}, {"strikesPerVolley":3,"dmg":74},
              {"dmg":92,"cooldownSec":1.90}, {"dmg":110,"blastRadius":58,"telegraphSec":0.45},
              {"dmg":118,"cooldownSec":2.40,"targetMode":"densest"} ],
  "evolution": { "name":"오비탈 스트라이크", "desc":"가장 밀집한 곳을 노리고, 착탄 반경이 두 배가 된다",
                 "params": { "evoRadiusMul":2.0 } } }
```

| Lv | 1 | 4 | 7 | **8 (진화)** |
|---|---|---|---|---|
| ST | 6.4 | 23.2 | 60.8 | **147.5** |
| 착탄 반경 | 45 | 52 | 58 | **116** |

- **Lv1~7의 ST는 명중 기대 0.35를 곱한 값**이다 — `randomInArena`는 진짜로 아무 데나 떨어진다. 그래서 **낮게 보이는 숫자가 정직하다.**
- **Lv8 = `targetMode` 전환 + 반경 ×2 = 로스터 최대의 진화 도약(×2.43).** `cooldownSec` 1.90 → 2.40으로 **나빠진다**(R-W4의 유일한 허용 사례) — 카드가 세 줄을 전부 보여준다: `데미지 110 → 118` / `쿨다운 1.90 → 2.40` / `조준 랜덤 → 최다 밀집`.
- `telegraphSec`은 **플레이어 텔레그래프** — **§7.12.6**: 스탠스 색 점선, **지면형이지만 레이어 3**(정본이 명시), 자홍 금지. 이것은 나를 위한 정보이지 위협 표시가 아니다.
- 보스전에서 `densest` = 부위가 밀집한 지점 → 사실상 확정 명중. 그것이 진화의 값이며, 그래서 Lv8 ST가 147.5로 뛴다.

### 4.10 `omni` 리어가드 — spawn · `targetMode` 없음

**자세**: **없앤다.** `rearIn`(후방 진입, 스테이지 3+)의 존재 이유이자 답. ST 하단, 커버리지 최고.

```json
{ "id":"omni", "family":"omni", "name":"리어가드", "desc":"여러 방향으로 동시에 쏜다. 사각이 없다",
  "elementStampMode":"spawn",
  "base": { "dmg":5, "cooldownSec":0.85, "projSpeed":300, "projRadius":4,
            "lifetimeSec":1.40, "pierce":0, "dirCount":3, "dirOffsetDeg":0, "rearBias":1.0 },
  "levels": [ {}, {"dmg":7}, {"dirCount":4,"dmg":8},
              {"dmg":11,"cooldownSec":0.78,"rearBias":1.30}, {"dirCount":5,"dmg":14},
              {"dmg":20}, {"dmg":28,"pierce":1}, {"dirCount":8,"dmg":44,"cooldownSec":0.70} ],
  "evolution": { "name":"링 버스트", "desc":"발사마다 링이 회전해 사각이 완전히 사라진다",
                 "params": { "evoRingRotDeg":22.5 } } }
```

| Lv | 1 | 4 | 6 | 8 |
|---|---|---|---|---|
| ST (1방향만 명중) | 5.9 | 14.1 | 25.6 | **62.9** |
| 군중 (전 방향) | 17.6 | 56.4 | 128.0 | **503.2** |

- `dirCount 3` = **후방 + 좌 + 우**(정면은 시작 무기 `forward`가 이미 본다). `rearBias` = **후방 방향 탄의 데미지 배율** → Lv4에서 1.30 = `rearIn` 요격 특화.
- **정본 §8.4가 `rearIn`에 `warnSec 0.8` 진입 표식을 강제**하므로, 리어가드는 "뒤를 안 봐도 되는" 무기이지 "뒤를 못 보는 것을 구제하는" 무기가 아니다. 관대함은 이미 보장돼 있고, 이 무기는 **손을 덜어줄 뿐**이다.
- **진화 「링 버스트」**: 발사마다 링 전체가 `evoRingRotDeg 22.5°` 회전 → 8방향 × 회전 = **사각 0**. `dirOffsetDeg`가 이미 계약에 있으므로 진화는 그 값을 매 발사 증분시키는 분기 하나다.

### 4.11 `drone` 옵션 — spawn · targetMode `nearest`

**자세**: **화력을 내 몸이 아닌 곳에 둔다.** 드론이 `anchor`(체류형 포대)를 맡는 동안 나는 다른 곳에 있을 수 있다 → **다른 무기의 자세를 자유롭게 해주는 유일한 무기.**

```json
{ "id":"drone", "family":"drone", "name":"옵션", "desc":"위성이 따라다니며 대신 쏜다",
  "elementStampMode":"spawn",
  "base": { "dmg":4, "projSpeed":340, "projRadius":4, "lifetimeSec":1.20, "pierce":0,
            "targetMode":"nearest", "droneCount":1, "anchorOffsets":[[0,-38]],
            "droneFireSec":0.55, "droneRangePx":260 },
  "levels": [ {}, {"dmg":5},
              {"droneCount":2,"anchorOffsets":[[-34,-22],[34,-22]]},
              {"dmg":7,"droneFireSec":0.50}, {"dmg":10},
              {"droneCount":3,"anchorOffsets":[[-40,-18],[40,-18],[0,-52]]},
              {"dmg":14,"pierce":1},
              {"droneCount":4,"dmg":16,"droneFireSec":0.46,
               "anchorOffsets":[[-44,-14],[44,-14],[-22,-52],[22,-52]]} ],
  "evolution": { "name":"잔상 편대", "desc":"드론이 내가 지나온 자리를 따라간다",
                 "params": { "evoTrailDelaySec":2.5 } } }
```

| Lv | 1 | 3 | 6 | 8 |
|---|---|---|---|---|
| ST | 7.3 | 18.2 | 60.0 | **139.1** |

- `anchorOffsets` = 플레이어 기준 드론 앵커 좌표 배열. **레벨마다 편성이 손으로 저작된다** — 1기는 정면, 2기는 좌우, 3기는 좌우+정면, 4기는 사각. 드론은 **앞쪽(−y)에 배치**된다 — 세로 슈팅에서 화력은 위로 나간다.
- ★ **`projCountAdd`(= `autoload`)와의 충돌은 미해소다 — §9.2-N1(blocker).** §9.6.1이 `drone`의 `countKey`를 `droneCount`로 확정했으나 `anchorOffsets`는 **정확히 4개**뿐이다 → **5·6번째 드론의 앵커가 어디에도 없다.** v1.0판의 "`droneCount`와 길이가 일치해야 한다"는 서술은 **그 자체로는 옳지만 훅이 그것을 깨뜨린다.** 결정은 정본의 것이다.
- **진화 「잔상 편대」**: `evoTrailDelaySec 2.5` → 드론이 `anchorOffsets`를 버리고 **2.5 게임초 전의 내 위치**를 따라간다. **양날이다** — 잘 움직이면 화면 곳곳에 화력이 깔리고, 가만히 있으면 드론이 겹쳐 아무 이득이 없다. **진화가 자세를 요구하는 두 번째 사례**(첫째는 `forward`의 램프).

### 4.12 `nova` 노바 — spawn · `targetMode` 없음

**자세**: **주기에 맞춰 그 자리에 있는다.** 정본 §8.10·§13.1.1이 지정한 **위기 세션의 캡스톤 정답.**

```json
{ "id":"nova", "family":"nova", "name":"노바", "desc":"주기적으로 자기 중심에서 대폭발한다",
  "elementStampMode":"spawn",
  "base": { "dmg":30, "intervalSec":5.00, "radius":120, "expandSec":0.35, "telegraphSec":0.60 },
  "levels": [ {}, {"dmg":40}, {"radius":150,"intervalSec":4.50}, {"dmg":60},
              {"radius":180,"dmg":75}, {"dmg":110,"intervalSec":4.00},
              {"dmg":150,"radius":210}, {"dmg":230,"intervalSec":3.20,"radius":240} ],
  "evolution": { "name":"슈퍼노바", "desc":"2단 폭발 + 확산 링이 적 탄을 지운다",
                 "params": { "evoRing2Radius":320, "evoClearBullets":true, "evoSecondaryDmgMul":0.5 } } }
```

| Lv | 1 | 3 | 6 | 8 | **8+진화** |
|---|---|---|---|---|---|
| ST | 6.0 | 8.9 | 27.5 | 71.9 | **107.8** |
| 군중 (반경 내 20기) | 120 | 178 | 550 | 1438 | **~2100** |

- **`telegraphSec 0.60`은 나를 위한 것**이다 — 폭발 0.6초 전에 **스탠스 색 점선 원**이 뜬다(§7.12.6). 그래야 "그 자리에 있는다"는 자세가 성립한다. **자홍 금지.**
- **`autoload`가 무효인 2 패밀리 중 하나**(§9.6.1). 대신 `coil`이 가장 크게 작동한다.
- **진화 「슈퍼노바」**: 중심 폭발(반경 240) → **확산 링**(반경 `evoRing2Radius 320`, 데미지 `dmg × evoSecondaryDmgMul`) 2단. 링 반경 안의 **적 탄 소거**(`evoClearBullets`).
- **폭탄과 정체성이 겹치지 않는다**(정본 §11.4 대조): 폭탄은 **전화면 · 즉발 · 패닉 버튼 · 스톡 소모**. 슈퍼노바는 **반경 320 한정 · 3.2초 주기 · 예측 가능 · 무한**. → 폭탄은 *못 피한 순간*의 카드, 슈퍼노바는 *피할 필요가 없게 만드는* 카드.
  > 정본 §13.2-⑫가 이 분리를 게이트로 승격했다: `crisisKillShareWithoutCapstone`의 대상 = **폭탄을 쓰지 않은 세션.** 게이트는 *폭탄 없이도 되는가*를 묻고 폭탄은 *그래도 안 되면*을 담당한다.
- 아레나 폭 580 기준 반경 320은 좌우를 덮지만 **세로 720의 절반도 안 된다** → 화면 청소가 아니라 **내 주변 청소**다.

---

## 5. 패시브 12종 (`data/passives.json`)

### 5.1 값 — **정본 §9.6을 그대로 채택한다 (미세 조정 0)**

> 위임 범위는 "`values` 미세 조정"이지만, **조정하지 않는 것이 이 문서의 결정**이다. 근거:
> - `resonance`는 **k 의미로 이미 재작성**되어 유효 배율 ×2.1~2.5 = 초안 C의 원래 의도와 정확히 일치한다(§9.6 · §3.1).
> - `bulkhead [6,12,18,24,30]`은 **`hpMax 100`에 맞춰 이미 재작성**되었다(Lv5 = +30%).
> - 나머지 10종은 `hpMax`·`moveSpeed`·`magnetRadius` 스케일과 이미 정합하며(§5.3 검산), **바꿀 근거가 없는 값을 섹션이 흔드는 것 자체가 C-1 위반의 온상**이다.
>
> ★ **v1.0판이 「패시브에서 실제로 비어 있던 것은 `values`가 아니라 훅 매핑이다」라고 쓴 것은 옳았고, 정본 v1.1이 그것을 §9.6.1로 채택해 소유했다.** 이 문서에는 이제 그 매핑이 **없다** — 인용만 있다(§5.2).

### 5.2 훅 적용 — **정본 §9.6.1이 소유한다 (인용)**

> **이것이 v1.0판 03의 「진짜 산출물」이었고, 정본 §9.6.1이 blocker 해소로 **채택·편입**했다**(요청 9.3). C-1에 따라 **이 문서는 더 이상 매핑을 정하지 않는다.**

| 정본이 확정한 것 | 위치 |
|---|---|
| 적용 식 4종 (`fireRateMul` 나눗셈 / `areaMul` 곱 / `pierceAdd` 가산 / `projCountAdd` 가산) | §9.6.1 |
| **`rules.passiveHooks` 12행** (`rateKey` · `countKey` · `pierce` 적용 · `areaKeys`) — **동결** | §9.6.1 |
| **H1** `fireRateMul`은 12 패밀리 전부에 적용 (`overclock`이 죽는 빌드가 구조적으로 없다) | §9.6.1 |
| **H2** `areaMul`은 "닿는 범위"만 키우고 **산포**(`spreadDeg` `jitterDeg` `arcDeg`)는 절대 안 건드린다 | §9.6.1 |
| **H3** `projRadius`는 `render.playerBulletMaxRadiusPx`(10)로 **판정·렌더 동시 클램프** | §9.6.1 |
| **H4** `countKey: null` = 그 패시브는 그 무기에 무효. **드래프트는 거르지 않는다** | §9.6.1 · §11.1 |
| **죽은 조합의 전수** — `autoload` → `aura` `nova` (12 중 2) / `coating` → `orbit` `aura` `mine` `barrage` `nova` + `boomerang`(`pierce: -1`) (12 중 6) | §9.6.1 |
| **`autoload` × `forward` = 시뮬의 1순위 튜닝 레버** (`count 2→4` = ×2 DPS) | §9.6.1 |
| **`afterimage`(고스트)와 이미 발사된 유도탄** — 적용됨. 고스트 중 유도탄은 재조준을 멈추고 직진 | §9.6.1 |

**이 문서가 이 절에서 남기는 것 = 훅이 로스터 수치에 미치는 산술뿐** (§5.3 · §8.3 · §8.5).

> ★ **미해소 1건**: `passiveHooks.drone.countKey = "droneCount"`가 `drone.anchorOffsets`(길이 4)와 충돌한다 → **§9.2-N1 (blocker).**

### 5.3 스케일 검산 (정본 값이 정본 스케일과 정합하는가)

| 패시브 | Lv5 효과 | 대상 스케일 | 비율 | 판정 |
|---|---|---|---|---|
| `bulkhead` | `maxHpAdd 30` | `hpMax 100` | **+30%** | ✔ 상점 `maxhp`(4회 = +40)와 합산 시 **170** |
| `frame` | `moveSpeedMul 0.20` | `moveSpeed 280` | +56 PxSec | ✔ 상점 3스택(+18%) 합산 = **396.5** (정본 §2.2의 파생값과 **정확히 일치**) |
| `resonance` | `elementBonusMul 1.50` | `elem = 1 + (2−1)×1.5` | **×2.5** | ✔ `×0.5`·`×1`에는 적용되지 않음(§3.1) |
| `warhead` | `dmgMul 0.30` | §3.1 2항 | ×1.30 | ✔ 가산 풀, 1회 적용 |
| `overclock` | `fireRateMul 0.28` | 주기 ÷1.28 | ×1.28 DPS | ✔ H1에 의해 12/12 유효 |
| `coil` | `areaMul 0.36` | `aura.radius 120` → **163.2** | +36% | ✔ 아레나 폭 580의 28% — 화면을 덮지 않는다 |
| `coating` | `pierceAdd 3` | `lance.pierce 8` → 11 / `forward.pierce 1` → 4 | — | ✔ 진화 랜스(`-1`)에는 무효(§4.4) |
| `autoload` | `projCountAdd 2` | `forward.count 2` → **4** (×2 DPS) / `fan.count 11` → 13 (×1.18) | — | ⚠ **§8.3·§8.5의 튜닝 레버 1번** |
| `reactive` | `hitBulletClearRadius 180` | 아레나 580×720 | 반경 180 | ✔ 폭탄(전화면)과 분리 유지 |
| `afterimage` | `ghostSecOnHit 2.6` | `iframeSec 1.0` | 2.6배 | ✔ 고스트 중 유도탄 재조준 정지(§9.6.1) |
| `study` / `salvage` | `xpGainMul 0.36` / `coinGainMul 0.43` | §8.3 `xpScale` / §13.2-⑪ | — | ✔ `salvage` Lv5가 `p90EndCoins` 검산의 입력(228 × 1.43) |

> ⚠ **`autoload` × `forward`**: `count 2 → 4`는 벌컨 ST를 **×2**로 만든다(150 → 300, 진화 램프 시 465). 같은 카드가 `fan`에서는 ×1.18이다. **이것은 버그가 아니라 `countKey` 매핑의 산술적 귀결**이며 — **정본 §9.6.1이 이 문서의 경고를 그대로 승인했다.** 값을 바꿀 필요가 생기면 **`passives.json`의 배열 하나만** 바뀐다 = C-4 준수.
> ★ **v1.1에서 이 경고는 더 무거워졌다**: 이것을 잡는 게이트가 `dominance.maxWeaponWinShare`만이 아니라 **`startWeaponDamageShare ≤ 0.40`**(신설)이며, 그 게이트의 여유가 **0.10밖에 없다**(§8.5).

---

## 6. 진화 12종 요약 (트리거·계약 = 정본 §9.5, 값 = 이 문서)

**트리거 (인용, 재정의 아님)**: **Lv7 → Lv8 레벨업 카드 그 자체가 진화 카드다.** 자동 아님 · 별도 카테고리 아님 · 패시브 페어링 아님 · 새 RNG 의존 0. 카드는 **진화 폼 프리뷰**를 표시한다(§11.1). 불가역. **같은 슬롯 인덱스 유지**(임뷰 계산에 특별 취급 없음). `family` 불변. 진화 `flags` 없음 — **선언된 `evo*` 키로만**.

| `id` | 진화 이름 | `evo*` 파라미터 | **무엇이 바뀌는가** | 코드 |
|---|---|---|---|---|
| `forward` | **오버드라이브** | `evoRampSec 3.0` `evoRampFireRateMul 1.55` | 무피격 연사 3초 → 발사 속도 ×1.55 (**피격 시 리셋**) | `if (w.evolved)` ×1 |
| `fan` | **플레어 팬** | `evoBlastRadius 40` `evoSecondaryDmgMul 0.5` | 탄 소멸 시 반경 40 폭발 | ×1 |
| `seeker` | **스웜** | `evoDistinctTargets true` `evoRetargetOnKill true` | 각 탄이 다른 타겟 / 처치 시 재유도 | ×1 |
| `lance` | **레일건** | `evoFullHeight true` | 사거리 = 아레나 전체 + `pierce` **`-1`**(무제한) | ×1 |
| `orbit` | **이지스** | `evoBulletClearCooldownSec 1.0` | 공전체가 닿은 **적 탄 소거** | ×1 |
| `aura` | **싱귤래리티** | `evoPullForce 90` | **`chaff`·새떼만** 중심으로 흡인 (§9.5의 정의역) | ×1 |
| `mine` | **클러스터 마인** | `evoClusterCount 6` `evoClusterRadius 32` `evoSecondaryDmgMul 0.5` | 폭발 → 자탄 6개 2차 폭발 | ×1 |
| `boomerang` | **체인 리턴** | `evoChainCount 3` | 귀환 경로가 적 3기 경유 | ×1 |
| `barrage` | **오비탈 스트라이크** | `evoRadiusMul 2.0` | 반경 ×2 (+ `levels[7]`의 `targetMode: densest`) | ×1 |
| `omni` | **링 버스트** | `evoRingRotDeg 22.5` | 발사마다 링 회전 → 사각 0 | ×1 |
| `drone` | **잔상 편대** | `evoTrailDelaySec 2.5` | 드론이 2.5초 전 내 위치를 추종 | ×1 |
| `nova` | **슈퍼노바** | `evoRing2Radius 320` `evoClearBullets true` `evoSecondaryDmgMul 0.5` | 2단 폭발 + 링 반경 내 탄 소거 | ×1 |

**진화 12개의 성격 분포 (설계 검증)**

| 성격 | 해당 | 확인 |
|---|---|---|
| **숫자만 커짐** | 없음 | ✔ 12/12가 규칙을 바꾼다 = 캡스톤의 정의 |
| **자세를 요구함** | `forward`(무피격) · `drone`(이동) · `nova`(위치) | 3종 |
| **약점을 지움** | `lance`(사거리) · `omni`(사각) · `barrage`(조준) | 3종 |
| **방어가 됨** | `orbit`(탄 소거) · `nova`(탄 소거) | 2종 |
| **군중을 지움** | `fan` · `mine` · `aura` · `boomerang` | 4종 |
| **보스에서 오히려 흐려짐** | `seeker`(타겟 분산) | 1종 — **진화가 항상 정답은 아니다는 유일한 증명** |

- **코드 예산 확인 (정본 §9.5의 확정 입력값)**: 패밀리당 `if (w.evolved)` 분기 1개 × ~15줄 × 12 = **약 180줄.** 위 12종 전부가 이 예산 안이다 — `targetMode` 전환(`barrage`)은 `levels[7]`이 처리하므로 분기가 아니고, 2차 폭발 3종은 **같은 형태의 분기**다.
- **진화가 `.js`를 열게 만들지 않는다**: 2차 피해 배율 3종이 전부 `evoSecondaryDmgMul` 파라미터이므로(§9.5 채택), 밸런서는 `weapons.json` 한 줄만 만진다 = C-4 준수.

---

## 7. 상점 (정본 §11.2 인용 + 이 문서의 전개)

### 7.1 항목 — **정본이 확정한 10종. 추가·삭제·값 변경 없음.**

| `id` | 효과 | `basePrice` | `growth` | `maxPurchases` | 스톡 상한 |
|---|---|---|---|---|---|
| `reroll` | 드래프트 1회 재추첨 스톡 | 30 | 1.60 | 6 | `rerollStockMax 3` |
| `potion` | HP `healPct 0.35` 즉시 회복 | 40 | 1.50 | 8 | — |
| `bomb` | 폭탄 스톡 +1 | 50 | 1.35 | 5 | `bombStockMax 3` |
| `shield` | 피격 1회 무효 | 60 | 1.50 | 4 | `shieldStockMax 2` |
| `timeToken` | 보스 타이머 +30 게임초 | 80 | 1.80 | 2 | `timeTokenStockMax 2` |
| `defense` | 방어력 +2 (정액 감산) | 45 | 1.45 | 4 → 상한 8 | — |
| `maxhp` | 최대 HP +10 (즉시 동량 회복) | 55 | 1.50 | 4 → 140 | — |
| `movespeed` | 이동 속도 +6% | 55 | 1.50 | 3 → +18% | — |
| `magnet` | 수집 반경 +30% | 40 | 1.40 | 3 → 171px | — |
| `resist` | 상태이상 지속시간 −20% | 40 | 1.40 | 3 → −60% | — |

**컨티뉴는 상점에 없다** — 사망 시점 제시, 150 고정, 런당 1회(§11.4).

### 7.2 가격 수열 전개 (`ceil(basePrice × growth^n)`) + 점수 환산

> 정본이 공식과 계수를 확정했으므로 **이 표는 계산 결과이지 결정이 아니다.** `= −N 점`은 `coinToScore 20`(§11.3)으로 환산한 것이며, **상점 UI가 실시간으로 표시할 바로 그 숫자**다(§11.2의 필수 표기 · `hud.coinShowsScoreValue`).

| `id` | 1회 | 2회 | 3회 | 4회 | 5회 | 6회 | 7회 | 8회 | **전량 비용** | **전량 = 점수** |
|---|---|---|---|---|---|---|---|---|---|---|
| `reroll` | 30 | 48 | 77 | 123 | 197 | 315 | — | — | **790** | −15,800 |
| `potion` | 40 | 60 | 90 | 135 | 203 | 304 | 456 | 684 | **1,972** | −39,440 |
| `bomb` | 50 | 68 | 92 | 124 | 167 | — | — | — | **501** | −10,020 |
| `shield` | 60 | 90 | 135 | 203 | — | — | — | — | **488** | −9,760 |
| `timeToken` | 80 | 144 | — | — | — | — | — | — | **224** | −4,480 |
| `defense` | 45 | 66 | 95 | 138 | — | — | — | — | **344** | −6,880 |
| `maxhp` | 55 | 83 | 124 | 186 | — | — | — | — | **448** | −8,960 |
| `movespeed` | 55 | 83 | 124 | — | — | — | — | — | **262** | −5,240 |
| `magnet` | 40 | 56 | 79 | — | — | — | — | — | **175** | −3,500 |
| `resist` | 40 | 56 | 79 | — | — | — | — | — | **175** | −3,500 |
| | | | | | | | | | **합 5,379** | **−107,580** |

> ★ **`shield`의 60 / 90 = 150코인 = 정확히 3000점 = `stageNoHitBonus`와 동률**은 정본 §11.3이 **자기 자신의 오산(156코인)을 정정하며 확정한 산술**이다. 이 표가 그 정정과 일치한다(마진 0). `shield.basePrice`·`growth`를 바꾸면 **정본 §11.3의 3중 상쇄 논거를 다시 계산해야 한다.**

### 7.3 희소성 — **정본 §13.2-⑪이 소유한다 (인용)**

> ★ **v1.0판의 이 절은 「평균 1.5개 드랍」이라는 존재하지 않는 값을 가정하고 있었다.** 이 문서가 그 구멍을 요청(9.6)했고 **정본이 `bands[].coin`을 신설**했다(§9.7) → 이제 계산이 **가능해졌고, 정본이 직접 했다.** 이 문서는 인용한다.

| 정본이 확정한 것 (§13.2-⑪ · §9.7 · §13.1.1) | 값 |
|---|---|
| `bands[].coin` | `chaff 0` / `line 0` / `turret 1` / `bruiser 2` |
| 스테이지별 코인 수입 | **s1 ≈ 28 · s2 ≈ 32 · s3 ≈ 36 · s4 ≈ 40 · s5 ≈ 44 · s6 ≈ 48** (누적 **228**) |
| 상점 방문 시점의 누적 수입 (5회) | **28 / 60 / 96 / 136 / 180** |
| `medianPurchasesPerVisit ∈ [1.0, 2.0]` | **중앙값 1.0 — 하한에 정확히 붙는다** |
| `medianEndCoins ∈ [0, 120]` | 228 − 지출 ~150 = **~78** ✔ |
| `p90EndCoins ≤ 260` | 228 × 1.43(`salvage` Lv5) − 150 = **~176** ✔ |
| ★ **조정 손잡이** | **`elite.coin`(3)** — 정본이 소유한다 |
| ★ 테마 간 분산 차단 | **S23** (모든 테마의 로스터 4종 중 `turret`+`bruiser`가 1~2종) |
| **`medianAffordablePerVisit`는 폐기됐다** | [1,3] 밴드가 **산술적으로 도달 불가**였다(§13.1.1) |

**이 문서가 §7.2에서 더하는 것 — 가격표와 수급의 대조 (계산, 결정 아님)**

| 대조 | 계산 | 뜻 |
|---|---|---|
| **전량 구매 5,379 vs 런 수급 228** | **23.6배 부족** | ★ **"항상 살짝 부족"의 산술.** 런당 실제 구매는 **4~6개** |
| **`continueCost` 150 vs 런 수급 228** | **66%** | 컨티뉴 = **상점 5회를 거의 전부 포기한 사람만의 카드** ✔ 의도대로 |
| **첫 방문(28코인) vs 최저가(`reroll` 30)** | **아무것도 못 산다** | 이것이 `medianPurchasesPerVisit`가 1.0에 붙는 이유이자, **1스테이지 상점이 "다음엔 사자"를 가르치는 화면**인 이유 |

> ★ **"어느 항목을 살까"에 최적해가 존재하지 않는다는 것의 확인**: 위 10항목은 **서로의 가격·효과에 아무 영향을 주지 않는다**(독립 곡선 10개). 방문 5회에 걸친 구매 순서 최적화 = **각 항목의 그리디 판단 10개**이며 조합 폭발이 없다. 정본 §11.2의 "전체 목록 상시"가 체스식 최적화를 부르지 않는 이유가 이 표에서 눈으로 보인다.

---

## 8. 검산 — 이 로스터가 정본의 게이트를 통과하는가

### 8.1 DPS 기준선 — **정본 §13.5가 소유한다 (인용 + 이 로스터가 그 입력임의 검산)**

> ★ **v1.0판의 「DPS 엔벨로프」와 「보스 HP 사이징 계약」은 폐기·기각됐다**(§0.1-1, §0.1-2). **이 절은 더 이상 계약을 넘기지 않는다** — 정본이 곡선과 보스 HP를 **둘 다 소유**한다. 이 절이 하는 일은 **정본의 곡선이 이 로스터에서 실제로 나오는지 재검산**하는 것뿐이다.

**정본 §13.5의 확정 곡선 (인용)**

| 스테이지 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **`dpsRef`** (명목 · ST · 무속성 · 보스전 진입 시점) | **50** | **90** | **195** | **335** | **485** | **708** |
| `× uptimeRef 0.60` = **실효** | 30.0 | 54.0 | 117.0 | 201.0 | 291.0 | **424.8** |

**★ 이 로스터가 708을 만드는 산술 (정본이 채택한 계산의 재검산)**

```
Lv8 ST 12종 (정렬):
  47.6(aura) · 62.9(omni) · 71.1(mine) · 71.9(nova) · 94.4(fan 밀착) · 111.1(orbit)
  117.3(seeker) · 126.3(boomerang) · 138.5(lance 대형) · 139.1(drone) · 147.5(barrage) · 150.0(forward)
  → 중앙값 = (111.1 + 117.3) / 2 = 114.2                                  ✔ 정본 §13.5와 일치
  → 4무기 Lv8 = 456.8
  → × (1 + warhead Lv4 0.26) × (1 + overclock Lv4 0.23) = × 1.5498
  → 707.96  ≈  708                                                         ✔
필요 픽 = 3(무기) + 28(레벨) + 8(패시브: 획득 2 + 레벨 6) + 6(속성) = 45  <  54  ✔
```

**★ `shape[]`의 재검산 (정본 §13.5가 이 문서의 두 곡선에서 뽑은 값)**

| Lv | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `seeker` ÷ 117.3 | 0.057 | 0.076 | 0.171 | 0.228 | 0.341 | 0.546 | 0.546 | 1.000 |
| `forward` ÷ 150.0 | 0.053 | 0.067 | 0.133 | 0.187 | 0.373 | 0.545 | 0.606 | 1.000 |
| **평균 = `shape[L]`** | **0.055** | **0.071** | **0.152** | **0.207** | **0.357** | **0.545** | **0.576** | **1.000** |

→ **정본 §13.5가 인쇄한 `shape[]`와 8행 전부 일치.** 이 로스터는 정본 곡선의 **입력이며, 곡선의 소유자가 아니다.**

**★ 스탠스 배율의 케이스 (스테이지 6, `dpsRef[6] = 708` 기준 — 파생 계산)**

| 빌드 | `m` (§13.6의 스탠스 배율) | **명목** | **실효** (`× 0.60`) |
|---|---|---|---|
| 무속성 / `static` 봇 (항상 노말) | 1.00 | 708 | **424.8** |
| **균형 `2/2/2`** — 어느 관을 상대해도 슬롯 1~2가 ×2 | **1.62** | 1,147 | **688.2** |
| **특화 `4/2/0`** — primary 속성의 관 (전 슬롯 ×2) | **2.00** | 1,416 | **849.6** |
| **특화 `4/2/0`** — secondary 속성의 관 | 1.62 | 1,147 | 688.2 |
| **특화 `4/2/0`** — 투자 0인 속성의 관 | 1.00 | 708 | 424.8 |
| 역상성 (잘못된 스탠스, 4슬롯 전부 ×0.5) | 0.50 | 354 | 212.4 |

- **`m`은 정본 §13.6.2의 값이다** (제너럴리스트, 슬롯 DPS 내림차순 정렬 기준: `[1.35, 1.35, 1.43, 1.53, 1.62, 1.62]`). **이 문서가 정하지 않는다.**
- **역상성(354)이 무속성(708)의 정확히 절반이라는 것이 `Q` 키의 존재 이유의 수치**다(§4.3). 잘못된 스탠스로 싸우는 것은 **노말로 싸우는 것보다 정확히 2배 나쁘다.** 모르겠으면 Q.
- **`resonance` Lv5**를 얹으면 ×2가 **×2.5**가 된다(§3.1: `elem = 1 + (2−1)×1.5`) — 단 **`×1`·`×0.5`에는 절대 적용되지 않으므로 무속성 708은 그대로다.** 이 비대칭이 `resonance`를 "속성 투자 빌드 전용 카드"로 만든다.

**★ 보스 HP는 이 문서의 것이 아니다 (인용 — §13.6.3)**

| `tetrarch` | 값 | 이 로스터로 재검산한 격파 시간 |
|---|---|---|
| 관 3종(`armor`) | 25,800 각 = **77,400** | 균형: `77,400/688.2 + 5,700/424.8` = 112.5 + 13.4 = **125.9s** ✔ |
| `core` | **5,700** | 특화(전 6짝 동일): 30.4 + 37.5 + 60.7 + 13.4 = **142.0s** ✔ |
| `throne`(`armament`) | **5,200** | 무투자: `83,100/424.8` = **195.6s** → 180 초과 = `noElementPass` **0.149** ✔ |
| **총 저작 HP** | **88,300** | 코어 직행(gate `0.4³` = 0.064): `89,062/424.8` = **209.7s** ✗ 완전 불가 ✔ |

- ★ **v1.0판의 「45,000~65,000」은 기각됐다** — 그 밴드는 「가용 피해의 40~55%」라는 **가정**이었고, 정본은 **목표 격파시간에서 역산**했다. 방법이 다르고 **후자가 검증 가능하다.** 승복한다.
- ★ **v1.0판이 `uptimeRef` 0.6을 쓴 것 자체는 옳았다** (`180 × 708 × 0.6 ≈ 76,000`). 틀린 것은 **그 다음 한 줄**이었다. 그리고 04는 0.6을 **아예 적용하지 않았다** — 정본 §10.4.3이 `uptimeRef`를 소유함으로써 **이 대립은 재발할 수 없다.**
- ★ **「520 vs 708」의 정답은 「둘 다」였다**: `708 × 0.60 = 425`와 04의 `520`은 **같은 종류의 수**(실효)다. 두 문서는 **다른 단위로 같은 것을 말하며 그것을 부등호로 이었다.** → 이 문서는 이제 모든 DPS에 **명목/실효를 명시**한다(§0).
- **`stanceValue` 게이트의 검산은 정본 §13.2-③이 소유한다** — v1.0판이 여기서 계산한 「1,416 vs 708 = 2배 → `d_static ≈ 0.55` 재현」은 **정본이 스테이지별 곡선으로 완전히 다시 계산했다**(`Πd_static = 0.00404` → `Δ = 0.346~0.646`). **이 문서는 그 계산에 `dpsRef`와 `m`만 제공한다.**

### 8.2 특색 원칙의 기계적 재확인 (`check.mjs` S4의 무기판)

> 정본 S4는 **적**에게 `(moveId, emitterType)` 쌍 중복을 금지한다. **무기에는 S4가 걸리지 않는다**(S4의 정의역은 `enemies.json`이다). 아래는 같은 원칙을 **이 문서가 자기 검증에 적용한 것**이며, 정본에 검사를 요청하지 않는다 — 로스터는 12행으로 닫혀 있어 눈으로 전수 확인이 가능하고, **저작이 12개로 끝나는 것에 검증기를 붙이는 것은 예산 낭비**다.

| `id` | 조준 자유도 | 거리 | 시간축 | 각인 | **쌍 중복** |
|---|---|---|---|---|---|
| `forward` | 정렬 필요 | 밀착 | 즉발·연속 | spawn | — |
| `fan` | 방향만 | 밀착 | 즉발·연속 | spawn | — |
| `seeker` | 무조준·추적 | 전 | 즉발·연속 | spawn | — |
| `lance` | 정렬 필요 | 원거리 | 차지 | spawn | ≠ forward (거리·시간축) |
| `orbit` | 무조준·부착 | 접촉 | 지속 | **live** | — |
| `aura` | 무조준·부착 | 접촉 | 지속 | **live** | ≠ orbit (반경 vs 궤도, ST vs 군중) |
| `mine` | 예측 배치 | 후방·과거 | **지연** | spawn | **유일한 지연형** |
| `boomerang` | 방향만 | 중 | 왕복 | spawn | **유일한 왕복형** |
| `barrage` | 무조준·랜덤 | 전역 | 볼리 | spawn | ≠ seeker (확정성) |
| `omni` | 무조준·전방위 | 근~중 | 즉발·연속 | spawn | ≠ seeker (커버리지 vs ST) |
| `drone` | 무조준·위임 | 앵커 기준 | 즉발·연속 | spawn | **유일한 화력 분리형** |
| `nova` | 무조준·자기중심 | 대반경 | **주기** | spawn | **유일한 주기형** |

**중복 0.** 가장 가까운 두 쌍(`orbit`/`aura`, `seeker`/`barrage`/`omni`)은 §1에서 각각 분리 근거를 갖는다.

### 8.3 캡 검산 (`certify.static.capHits.max = 0` — 플레이어 측)

**최악 케이스**: `forward` + `fan` + `omni` + `boomerang` 전부 Lv8 + `autoload` Lv5(+2) + `overclock` Lv5(÷1.28)

| 무기 | 발사당 탄 | 주기(가속 후) | 수명 | **동시 생존 탄** = 탄 × (수명 / 주기) |
|---|---|---|---|---|
| `forward` | `count 4` × `burstCount 3` = 12 | 0.31 | 1.6 | **~61** |
| `fan` | `count 13` | 0.48 | 1.0 | **~27** |
| `omni` | `dirCount 10` | 0.55 | 1.4 | **~26** |
| `boomerang` | `count 5` | 0.74 | 3.0 | **~20** |
| | | | **합계** | **~134** |

> 이것은 **상한**이다 — 실제로는 탄이 적에 맞아 수명 전에 사라지고(`pierce 0`), 네 무기 전부를 이 조합으로 뽑을 확률 자체가 낮다. **가장 나쁜 경우를 세도 여유가 있다**는 것이 요점이다.

| 캡 (§12.1 B층) | 정본 값 | 최악 | 여유 |
|---|---|---|---|
| `caps.playerBullets` | **256** | ~134 | **×1.9** ✔ |
| `caps.drones` | **8** | §9.2-N1의 결정에 무관하게 **4**(countKey null) 또는 **6**(현행 매핑) | ✔ **어느 쪽이든 닿지 않는다** |
| `caps.zones` | **64** | `mine.maxAlive` 12 + `barrage` 5 + `nova` 2 + `aura` 1 = **~20** | **×3.2** ✔ |

> **캡에 닿는 콘텐츠는 콘텐츠 버그다**(§12.1). 이 로스터는 어떤 조합으로도 닿지 않는다. `overflow.playerBullet: "rejectSpawn"`이 발동하는 런은 존재하지 않아야 하며, 위 산술이 그것을 보증한다.
> **텔레그래프 캡은 플레이어 측 관심사가 아니다** — §12.1의 `caps.telegraphs`(8 → **96**으로 정정)와 A층 `telegraphConcurrentMaxGlobal`(**80**, 파생)은 **적 이미터 전용**이다. `lance.chargeSec`·`barrage.telegraphSec`·`nova.telegraphSec`는 **플레이어 텔레그래프**(§7.12.6)이며 그 예산의 대상이 아니다.

### 8.4 성장 예산 (`certify.static.growthBudget`)

```
새 무기 3  +  무기 레벨 28 (4무기 × Lv1→8)  +  속성 6  +  패시브 30  =  67
maxLevelUps 60  <  67   ✔   (목표 54 → 충족률 81%)
```
- **무기 레벨 28 중 4픽이 진화 픽**(각 무기의 Lv8)이다 → **런당 진화는 최대 4회, 현실적으로 1~2회.** 진화가 흔해지지 않는 것이 이 부등식으로 보장된다.
- 12 패밀리 중 **런당 보는 것은 4종**(1/3) → 로스터 12는 다양성 대비 저작비의 균형점.
- ★ **`newWeaponSlotScale [3.0, 1.5, 1.0]`이 「4칸이 빨리 찬다」를 보장하고, 그것이 v1.0판 §8.1의 스테이지 1~5 곡선을 무너뜨린 바로 그 규칙이다**(§13.5: 보스전 진입 픽 수 = **11 / 18 / 25 / 34 / 43 / 53**). **스테이지 1에서 이미 4무기다.**

### 8.5 `dominance` 게이트에 대한 이 로스터의 방어 (§13.1.1의 분모 정의로 재계산)

> ★ **v1.0판의 이 절은 분모 없이 계산했다.** 정본 §13.1.1이 **분모·임계·집계를 확정**했고(피해 지분 정규화 + `forward` 분리), 그 정의에서 다시 센다.

| 게이트 | 분모 · 균등값 | 상한 | 이 로스터의 답 |
|---|---|---|---|
| `maxWeaponPickShare` | `runs × 3` = 24,000 픽 / **11종**(`forward` 제외) / 균등 **0.0909** | **0.16** (1.76×) | 드래프트 가중치는 **아이템 단위 균등**(§11.1)이고 **12종에 편향 필드가 없다** → 편차의 원천은 `newWeaponSlotScale`의 **슬롯 의존성뿐이며 무기 종류에 무관** → 관측 ≈ 0.091 ✔ **여유 큼** |
| `maxWeaponWinShare` | **피해 지분**의 평균 / 클리어 런 / `forward` 제외 후 11종 재정규화 / 균등 **0.0909** | **0.20** (2.2×) | ★ **ST 스프레드 `omni` 62.9 ~ `forward` 150 = ×2.38**이 **군중 화력에서 역전된다**(`omni` 503.2 vs `forward` ~150). 군중 화력이 높은 무기는 잡몹 페이즈를 빨리 지워 **레벨이 높아지고**, 그것이 보스전 ST를 되사온다. **두 축이 서로를 상쇄하도록 설계했다.** `forward` 제외 후 최상위 = `barrage` 147.5 → 최악 추정 **0.15** ✔ |
| ★ **`startWeaponDamageShare`** | `forward` 전용 / 분모 = 클리어 런의 **4무기 총 피해** | **[0.10, 0.40]** | `forward` Lv8 = 150 (로스터 최고) → `150 / (150 + 114.2×3)` = **0.305** ∈ 밴드 ✔ **단 상한 여유가 0.095뿐이다** — 아래 ⚠ |
| `maxElementWinShare` | 속성 투자 픽 / 3종 / 균등 0.3333 | 0.42 | **이 문서의 관할 밖**(속성 픽은 §4.2·§11.1). 정본 §13.2-⑩이 구조적 대칭으로 닫았다 |
| `crisisKillShareWithoutCapstone` | capstone(= `nova` 또는 `aura` 보유) **미보유** 그리고 폭탄 미사용 세션의 중앙 `killShare` | **≥ 0.80** | 정본 §13.2-⑫: 피해량은 **17.6배 여유**이므로 게이트는 **커버리지의 시험**이다. **12 중 7종**이 새떼를 지운다 — `nova`·`aura`(캡스톤) + `fan`(군중 346) · `omni`(503.2) · `mine`(~890) · `barrage`(반경 116) · `seeker`(진화 다중 타겟) → **캡스톤 2종을 빼도 5종이 답** ✔ |

> ⚠ ★ **`startWeaponDamageShare`가 이 로스터의 유일한 얇은 곳이다.** `autoload`가 `forward.count`를 2→4로 만들면 지분이 `300/(300+342.6)` = **0.467 > 0.40**이 된다. 게이트는 **클리어 런 평균**이고 `autoload` Lv5(+2)는 30픽 싱크에서 드문 사건이므로(§13.5의 제너럴리스트 궤적 = `warhead` Lv4 + `overclock` Lv4) 평균은 **0.31~0.34** 대역에 머문다 ✔ — **그러나 여유가 0.06~0.09뿐이다.**
> → **정본 §9.6.1이 `autoload.values`를 「시뮬의 1순위 튜닝 레버」로 확정한 것이 이 게이트에서 재확인된다.** 밴드를 벗어나면 손대는 것은 **`passives.json`의 배열 하나**(`autoload.values`)이며, 다음 후보는 `forward`의 `levels[7]`(`burstCount 3` → 2)이다. **둘 다 `.js`가 아니다** = C-4 준수.

---

## 9. 정본 추가 요청

### 9.1 v1.0 라운드의 요청 10건 — **전부 채택됨 (이제 요청이 아니라 인용)**

> 정본 §19.2의 심사 결과: **채택 10 / 수정채택 0 / 기각 0.** 각 항목은 이제 **정본에 있으므로** 이 문서는 인용만 한다.

| # | 요청 | 판정 | 정본의 거처 | 이 문서의 인용 위치 |
|---|---|---|---|---|
| 9.1 | 「계약 = `base` 필수 키 집합」 명문화 | **채택** | §9.5 | §0 · §2 |
| 9.2 | `pierce: -1` = 무제한 관통 규약 | **채택** | §9.5 · §9.6.1 | §2 · §4.8 · §4.4 |
| 9.3 | ★ `rules.passiveHooks` 신설 | **채택** | **§9.6.1** (+ `rules.json` 루트 키 17개에 편입) | **§5.2** |
| 9.4 | `evoSecondaryDmgMul`을 `fan`·`mine`·`nova` 계약에 | **채택** | §9.5 | §2 · §6 |
| 9.5 | 플레이어 텔레그래프의 시각 규약 | **채택** | **§7.12.6** | §4.4 · §4.9 · §4.12 |
| 9.6 | `bands[].coin` 신설 | **채택** | §9.7 (`turret 1` / `bruiser 2` = 원안 그대로) | §7.3 |
| 9.7 | `evoPullForce`를 `chaff`+새떼로 제한 | **채택** | §9.5 (+ **보스 부위**까지 명시적으로 제외) | §4.6 |
| 9.8 | `areaMul × projRadius` 클램프 (판정·렌더 동시) | **채택** | §9.6.1 **H3** | §4.5 · §5.2 |
| 9.9 | `rng.pattern`을 플레이어 무기가 써도 되는가 | **채택 (허용)** | §9.5 · §10.2 · **S11** | §4.1 |
| 9.10 | 플레이어 무기는 `fairness` 대상이 아님 | **채택** | §9.5 (`fairness.playerWeaponsExempt: true`) · **S6** | §4.1 · §8.3 |

### 9.2 ★ 남은 요청 — **2건**

| # | severity | 요청 |
|---|---|---|
| **N1** | ★ **blocker** | **`passiveHooks.drone.countKey`와 `drone.anchorOffsets`의 충돌 해소** |
| **N2** | **minor (정정)** | **정본 §13.2-⑩의 「ST 스프레드 1.75배」는 이 문서의 산술 오류를 인용한 것이다 → 2.38배** |

---

**N1 (blocker) — `drone`의 초과 앵커가 정의되지 않는다**

| | |
|---|---|
| **사실** | §9.6.1이 `drone`의 `countKey`를 **`droneCount`**로 확정했다(동결 12행). 그런데 `anchorOffsets`는 **레벨마다 손으로 저작된 편성**이고 `levels[7]`에서 **정확히 4개**다. `autoload` Lv5(`projCountAdd 2`)는 `eff.droneCount = 6`을 만든다. |
| **결과** | ★ **5·6번째 드론의 앵커가 어느 데이터에도, 어느 규칙에도 없다.** → `src/core/weapons/drone.js`를 **쓸 수 없다**(정본 §9.6.1이 훅 매핑을 만든 목적 그 자체가 여기서 미달성). → 구현자가 배치를 **발명한다** = 개발 중 설계 결정 = **C-6 위반.** 그리고 §8.3의 「`droneCount 4` + autoload 2 = 6 ≤ `caps.drones` 8」도 **그 발명에 기대고 있었다**(이 개정에서 양방향으로 다시 썼다). |
| **왜 정본이 정해야 하나** | 이 문서의 위임 범위는 **`base`/`levels`의 숫자**다. "`eff.droneCount > anchorOffsets.length`일 때 어떻게 되는가"는 **숫자가 아니라 규칙**이며, `passiveHooks`(§9.6.1)와 `weapons.json` 스키마(§9.5)의 **경계면**에 있다. **한 항목에 답은 하나**(C-1). |
| **03의 권고 = (A)** | **`passiveHooks.drone.countKey = null`.** 새 값 0 · 새 규칙 0 — **H4가 이미 정의한 상태**(`null` = 무효, 드래프트는 거르지 않는다)로 떨어진다. 귀결: 죽은 조합이 `autoload` → `aura` · `nova` · **`drone`** = **12 중 3**. |
| **(A)의 대가 (정직하게)** | `autoload`가 무효인 패밀리가 2 → 3이 된다. 그러나 `drone`은 `overclock`(`droneFireSec`) · `coil`(`droneRangePx` `projRadius`) · `coating`(✔) · `warhead`가 **전부 유효**하므로 죽은 슬롯이 아니고, "드론이 늘어난다"는 기대는 **`levels`가 이미 3번 준다**(Lv3 · Lv6 · Lv8). |
| **(B)를 권고하지 않는 이유** | 초과분을 **절차적으로 배치하는 규칙**(링 균등 분할 등)을 신설하면 ① `anchorOffsets`가 저작하는 **레벨별 편성 자체가 무의미**해지고(4기까지는 손으로, 5기부터는 링으로 도는 무기 = **I-2의 정신 위반: 화면이 두 개의 규칙을 동시에 보여준다**) ② **새 모델 = 새 구조**이므로 §9.10·§8의 「어휘 조합, 신규 코드 0」과 어긋나며 ③ 새 자유 숫자(링 반경)를 부른다. 정본 v1.1이 `telegraphConcurrentMaxGlobal`에서 **파생값을 골라 새 숫자를 0으로 만든 것**과 같은 이유다. |
| **정본이 (B)를 택하는 경우** | 이 문서는 **인용만 한다.** §4.11과 §8.3의 캡 검산은 **이미 양쪽 결과에 대해 통과**하도록 써 두었다(4 또는 6, 둘 다 ≤ 8). |

**N2 (minor) — 정본 §13.2-⑩의 인용값 정정**

정본 §13.2-⑩의 `maxWeaponWinShare` 행은 *"ST 스프레드 **1.75배**(`omni` 62.9 ~ `forward` 150)"*라고 쓴다. **이것은 v1.0판 03 §8.5의 문장을 그대로 가져간 것이고, 그 문장의 산술이 틀렸다**: `150 ÷ 62.9 = **2.38**`. (참고로 R-W2의 Lv8 전체 폭은 `150 ÷ 47.6 = **3.15**`이며, v1.0판의 「×3.2」도 반올림이 거칠었다.)
- **결론은 바뀌지 않는다** — 스프레드가 클수록 「군중 화력에서 역전된다」는 논거가 **더 필요해질 뿐**이고, 최악 추정 0.15 < 0.20은 `forward` 제외 후 `barrage` 147.5로 계산한 것이라 스프레드 수치에 의존하지 않는다.
- **그러나 C-8에 의해 정본의 표에 인쇄된 숫자는 `data`·문서와 갈라질 수 없다.** 이 문서는 §8.5를 2.38로 정정했으므로 **정본 §13.2-⑩도 함께 정정되어야 한다.** severity가 minor인 이유는 **게이트 판정이 바뀌지 않기 때문**이며, 정정하는 이유는 **틀린 숫자가 정본에 있으면 다음 라운드가 그것을 근거로 쓰기 때문**이다(이번 라운드가 정확히 그렇게 됐다).

### 9.3 요청하지 않은 것 (의도적)

- **보스·적 HP** → 정본 §13.6 소유. **이 문서는 이제 사이징 계약조차 넘기지 않는다** — `dpsRef`(§13.5)의 **입력**만 제공한다.
- **`dpsRef` 곡선** → 정본 §13.5 소유(§17이 명시). 03이 값을 바꾸면 **정본을 개정해야 한다.**
- **패시브 `values`** → **정본 값 그대로.** 미세 조정 권한이 있으나 **바꿀 근거가 없다**(§5.1). 단 `autoload.values`는 **시뮬이 만질 1순위**임을 §5.3·§8.5가 명시한다.
- **훅 매핑** → 정본 §9.6.1 소유(이 문서의 요청이 채택된 결과). **N1 외에는 인용만 한다.**
- **상점 항목·가격·상한·희소성** → 정본 §11.2·§13.2-⑪ 소유. §7.2는 **계산**이지 결정이 아니다.
- **드래프트 가중치·필터·보장** → 정본 §11.1 소관. 이 문서는 참조만 한다.
- **무기용 `check.mjs` 검사** → **요청하지 않는다.** 로스터는 12행으로 닫혀 있어 전수 확인이 눈으로 되고, 저작이 12개로 끝나는 것에 검증기를 붙이는 것은 예산 낭비다(§8.2).

---

## 10. 이 문서가 닫은 것

**무기**: 12 패밀리의 `base` 전수 · `levels[8]` 전수 · 레벨 곡선 규칙 R-W1~R-W6 · 12 진화의 `evo*` 값 전수와 그 성격 분포 · 패밀리별 파라미터 계약 12표(정본 §9.5의 전개) · `hitCooldownSec`/`pierce` 의미 · 특색 원칙의 자체 검증(중복 0).
**패시브**: 정본 §9.6 값의 스케일 검산 12종 · 훅이 로스터 수치에 미치는 산술(`autoload` × `forward` = ×2) · `coating` × 진화 랜스의 귀결.
**아이템**: 상점 10종의 가격 수열 전개 + 점수 환산 · 가격표 대 수급의 대조(23.6배 / 66% / 첫 방문 0개).
**검산**: 정본 `dpsRef` 곡선의 **재현**(중앙값 114.2 → 708, `shape[]` 8행 일치) · 스탠스 배율 케이스 6종 · 정본 보스 HP의 **재검산**(125.9 / 142.0 / 195.6 / 209.7) · 캡 검산(playerBullets 134 ≤ 256 / drones / zones) · 성장 예산 67 > 60 · `dominance` 3종 + `crisisKillShareWithoutCapstone`을 §13.1.1의 분모로 재계산.

**이 개정이 되돌린 것**: DPS 엔벨로프 s1~s5(폐기) · 보스 HP 사이징 계약(기각) · `seeker` 채택 근거(C-7로 재작성) · 훅 매핑 소유권(정본 §9.6.1로 이관) · **자체 발견 산술 오류 3건**(R-W1 상단 8.9→8.3 / R-W2 ×3.2→×3.15 · R-W6 ×1.8→×1.6 / `lance` 군중 34.5→27.5 · 692.5→231) · §8.5의 스프레드 1.75→2.38.

**남긴 의존**: **§9.2-N1의 정본 판정**(blocker — 이것 없이는 `drone.js`가 안 써진다) · §9.2-N2의 정본 표 정정 · 드래프트 카드 UI 레이아웃(「HUD」) · `data/*.json`의 최종 값(「밸런싱」, **숫자만**).
