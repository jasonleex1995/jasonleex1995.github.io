## 데이터 스키마 · 구현 스펙 (밸런싱=수치만)

> **이 섹션의 계약**: 이후 모든 밸런싱은 `data/*.json`의 **값 하나를 고치는 일**이어야 한다. 값을 고치기 위해 `.js`를 열어야 하는 순간이 오면 그것은 밸런싱이 아니라 **설계 변경**이며, 이 섹션이 실패한 것이다. 아래 스키마·경계·불변조건은 그 실패를 구조적으로 막기 위한 것이다.

---

### 1. 파일 레이아웃 · 모듈 경계

```
projects/nan2026/
  index.html            # 유일한 엔트리. <script type="module" src="main.js">
  package.json          # {"type":"module","private":true} — 3줄. 의존성 없음, 빌드 스텝 없음
  main.js               # 부트스트랩: 캔버스·rAF·입력·오디오·화면 전환. 게임 로직 없음
  src/
    core/               # ★순수★ — DOM/시간/난수 접근 절대 금지. 게임 + 시뮬이 공유하는 유일 진실
      rng.js            # sfc32 + 스트림 파생
      world.js          # 상태 컨테이너 + 풀
      step.js           # step(world, input) — 1틱. 이 함수가 게임 그 자체
      damage.js         # 데미지 계산식 (§8)
      collide.js        # 유니폼 그리드 브로드페이즈
      weapons/          # 12(+진화) 패밀리 update 함수. 숫자 리터럴 금지
      emitters.js       # 탄막 프리미티브 8종
      enemies.js        # 이동 어휘 8종
      boss.js           # 부위·페이즈·파괴효과 어휘
      draft.js  shop.js  score.js  stage.js
      schema.mjs        # JSON 검증기 (손으로 쓴 ~200줄, 의존성 0)
    render/             # 브라우저 전용. core를 읽기만 함. core는 render를 모름
    ui/                 # HUD·드래프트·상점·결과 화면
  data/                 # AI 생성 정적 JSON (§9~§20)
  tools/                # 개발 전용. 배포물에 포함되나 실행되지 않음 (수 KB)
    sim.mjs             # 헤드리스 봇 러너 (node, 의존성 0)
    check.mjs           # 순수성·스키마·공정성 게이트
    report/             # 시뮬 산출물 (공모전 기술문서 근거)
```

**모듈 경계 규칙 (위반 = 빌드 실패)**

| 규칙 | 강제 방법 |
|---|---|
| `src/core/**` 는 `src/core/**` 외 import 금지 | `tools/check.mjs` 정적 검사 |
| `src/core/**` 에 `window document canvas requestAnimationFrame Date performance Math.random fetch localStorage console` 식별자 등장 금지 | 동일 |
| `src/core/weapons/**` 에 `0, 1, -1, 0.5, 2` 외 숫자 리터럴 금지 | 동일 |
| `render/**` 는 world를 **읽기만** 함 (쓰기 금지) | 리뷰 + core 객체 freeze(dev 빌드만) |

> ES 모듈은 `file://`에서 동작하지 않는다. 로컬 개발은 `python3 -m http.server 8000` 고정. 배포(GitHub Pages)는 https라 그대로 동작. **번들러 없음, node_modules 없음** — node는 `tools/`를 돌리는 개발 도구일 뿐 배포물에 관여하지 않는다.

---

### 2. 코드/데이터 경계 — 절대 하드코딩 금지 목록

**코드에 있어도 되는 것 (구조)**: 12 무기 패밀리의 *거동 함수*, 8개 탄막 프리미티브, 8개 이동 어휘, 부위 파괴 효과 어휘, 데미지 계산식의 *항 순서*, 상태 머신.

**반드시 JSON이어야 하는 것 (수치·구성)**:

| 카테고리 | 예 |
|---|---|
| 모든 데미지·HP·속도·반경·쿨다운·지속시간 | 전부 |
| 모든 확률·가중치·비율 | 드래프트 가중치, 코인 드랍률, 70/30 |
| 모든 개수 | 탄 수, 웨이브 적 수, 슬롯 수, 캡 |
| 모든 곡선 | XP 곡선, 가격 상승, 스테이지 스케일 |
| 모든 구성 | 적 로스터, 웨이브 타임라인, 보스 부위·페이즈, 상점 목록, 테마 풀 |
| 모든 키 바인딩 | `rules.json > input.bindings` |
| 모든 색 | `rules.json > palette` (속성 색 = 단일 진실) |
| 모든 상한·임계값 | 엔티티 캡, 공정성 불변조건, 시뮬 인증 임계 |

**핵심 규칙 (무기)**: 각 패밀리 함수는 **선언된 파라미터 계약(§12)의 키만** 읽는다. 레벨업 효과는 **전부 파라미터 델타로 표현 가능해야 한다.** "Lv3에서 관통이 생긴다"는 `pierce: 0 → 1`이지 코드 분기가 아니다. 함수는 모든 레벨에서 관통을 지원하고, Lv1은 그냥 `pierce: 0`일 뿐이다. **계약으로 표현 불가능한 요구가 나오면 그것은 설계 변경이며, 밸런싱 세션에서 처리하지 않는다.**

---

### 3. 고정 타임스텝 · 배속 · 결정론

```
TICK_HZ   = 60                 // 잠금. 영원히 변하지 않음
TICK_DT   = 1/60 게임초        // 잠금. 배속과 무관하게 상수
```

**배속의 구현 = 초당 소비하는 틱 수** (dt를 곱하는 것이 **아님**):

| 난이도 | speed | 실제 1초당 틱 | dt |
|---|---|---|---|
| Normal | 1.0 | 60 | 1/60 |
| Hard | 1.5 | 90 | 1/60 |
| Hell | 2.0 | 120 | 1/60 |
| Disaster | 3.0 | 180 | 1/60 |

> **왜 dt 스케일이 아닌가**: dt를 ×3 하면 탄이 1틱에 13px 이동 → 반경 6px 플레이어를 **관통(터널링)** 하고, 같은 시드가 난이도마다 다른 결과를 내 시뮬 인증이 무효가 된다. 틱 수 스케일은 dt가 상수라 판정이 난이도 불변이고, **게임시간 기준 시뮬 1회가 4난이도 전부의 물리를 동시에 인증**한다. 난이도 차이는 오직 §24의 봇 반응지연(실제ms→틱 환산)에서만 발생한다 — 이것이 "난이도=인간 반응시간뿐"의 정확한 기계적 표현이다.

**메인 루프 (main.js)**
```js
const tickDur = 1000 / (TICK_HZ * speed);   // 실제 ms
acc += min(now - last, rules.loop.maxFrameGapMs);
let steps = 0;
while (acc >= tickDur && steps < rules.loop.maxStepsPerFrame) { step(world, pollInput()); acc -= tickDur; steps++; }
if (acc >= tickDur) acc = 0;                // 나선형 죽음 방지: 남은 시간 폐기(빨리감기 금지)
render(world, acc / tickDur);               // alpha 보간은 위치만. 로직 금지
```
- `maxStepsPerFrame: 6`, `maxFrameGapMs: 250`.
- **blur(탭 이탈) → 자동 일시정지 + acc=0.** 30분 무세이브 런에서 alt-tab 사망 방지. (일시정지 정책은 조작 섹션 소관, 여기선 acc 처리만 규정.)
- 입력은 **이벤트가 아니라 틱마다 키상태 폴링**. 대각선 정규화 `×0.7071`, 반대키 동시 입력은 0으로 상쇄.
- 렌더 보간은 위치 lerp만. 파티클·데미지 숫자는 **core 바깥**(render 전용, 결정론 무관).

**결정론 범위 (명시적 선언)**

| 수준 | 보장 | 근거 |
|---|---|---|
| L1 콘텐츠 | 같은 시드 → 같은 테마 순서·웨이브·드래프트 후보·상점 진열. **엔진 무관 보장** | RNG가 정수 연산뿐 |
| L2 런 | 같은 시드 + 같은 입력열 → 같은 결과. **동일 엔진 내 보장** | 아래 규칙 |
| L3 리플레이 | 엔진 간 비트 동일 — **비보장·비목표** | `Math.sin/cos` ULP 차이. 리플레이 기능 없음, 시뮬 인증은 통계적이라 무관 |

**L2 유지 규칙**: 사전할당 풀 + `alive` 플래그, **인덱스 오름차순 순회만**, 게임플레이 로직에서 `Map/Set` 순회 금지, `for...in` 금지, `sort` 사용 시 반드시 `id` 총순서 타이브레이크, core 내 객체 생성 금지(핫패스 GC 0).

---

### 4. 시드 RNG

```js
// rng.js — sfc32. 정수 연산만 → 모든 엔진에서 비트 동일
export function makeRng(seed)            // → {u32(), f(), range(a,b), int(a,b), pick(arr), weighted(map), shuffle(arr)}
export function stream(masterSeed, name) // → makeRng(hash32(masterSeed, name))
```

**독립 스트림 (필수)** — 한 스트림에 draw를 추가해도 다른 스트림이 밀리지 않는다. 이것이 없으면 콘텐츠를 하나 고칠 때마다 이전 시뮬 인증이 전부 무효가 된다.

| 스트림 | 용도 |
|---|---|
| `theme` | 5개 테마 순서. **런 시작 시 한 번에 전부 추첨** |
| `draft` | 드래프트 후보 추첨 · 리롤 |
| `spawn` | 웨이브 스폰 · 적 속성 70/30 결정 |
| `drop` | 코인 · 회복 드랍 |
| `shop` | 상점 진열 |
| `pattern` | 탄막 jitter |
| `boss` | 보스 페이즈 선택 |

- 마스터 시드 = uint32. 브라우저: `(Date.now() ^ (performance.now()*1000)) >>> 0` — **비결정성이 들어오는 유일한 지점이며 core 바깥(main.js)에서 생성해 core로 주입**한다. 시뮬: 명시 시드.
- 시드는 결과 화면에 8자리 hex로 표시(오락실 감성 + 버그 리포트 재현, 비용 0).
- `Math.random` 은 core에서 **금지**(§2 정적 검사).

---

### 5. 캔버스 · 좌표계 · 스케일

```json
"view": {
  "logicalW": 1280, "logicalH": 720,
  "arena": { "x": 350, "y": 0, "w": 580, "h": 720 },
  "panelLeftW": 350, "panelRightW": 350,
  "minViewportW": 1024, "minViewportH": 576,
  "maxDpr": 2
}
```
- **논리 해상도 1280×720 고정 (16:9)** — 심사자 노트북에 레터박스 없이 꽉 차고, 제출 영상(YouTube 720p)과 픽셀 1:1.
- **아레나는 세로 580×720**(고전 종슈팅 비율). 좌우 350px는 **HUD 전용 패널** — HUD가 아레나를 덮지 않는다(오버레이 금지). 세로 슈팅 × 가로 화면의 남는 공간을 HUD가 정확히 채우므로, "HUD 자리가 없다"는 문제가 구조적으로 발생하지 않는다.
- 스케일: `s = min(vw/1280, vh/720)`, CSS로 중앙 정렬 레터박스. 백킹스토어 = `1280·min(dpr,2) × 720·min(dpr,2)`, 컨텍스트에 `setTransform(s,0,0,s,0,0)`. **모든 게임 좌표는 논리 픽셀 단위**이고 스케일은 렌더에만 존재한다.
- 뷰포트가 최소 미만 → 플레이 차단 + "창을 키워주세요" 안내(입력 무시). 데스크톱 키보드 전용, 터치 미지원(명시).
- 아레나는 **고정 화면**(카메라 없음). 배경만 스크롤. 플레이어 이동 가능 영역 = 아레나 rect를 `player.hitboxRadius` 만큼 인셋한 사각형(상하 포함 전체).

---

### 6. 엔티티 상한 · 충돌 · 성능 목표

```json
"caps": {
  "playerBullets": 512, "enemyBullets": 640, "enemies": 160,
  "pickups": 256, "zones": 64, "drones": 8,
  "onCap": { "playerBullets": "skip", "enemyBullets": "skip",
             "enemies": "defer", "pickups": "merge" }
}
```
| 오버플로 정책 | 동작 |
|---|---|
| `skip` | 신규 스폰 무시(살아있는 탄을 재활용하지 않음 — 플레이어 발밑에 탄이 순간이동하는 불공정 방지) |
| `defer` | 스포너가 다음 틱에 재시도(웨이브가 공짜로 사라지지 않음) |
| `merge` | 신규 픽업 값을 최근접 기존 픽업에 합산(**손실 0 = 무-노가다 기둥 보존**) |

> **캡은 안전망이지 밸런스 노브가 아니다.** 오써링된 밀도 예산은 캡보다 낮아야 하며, 시뮬 인증은 `capHits == 0`을 요구한다(§24). 캡에 닿는 콘텐츠는 콘텐츠 버그다.

**충돌**
- 브로드페이즈: **유니폼 그리드, 셀 64px**, 매 틱 재구축(적 + 적탄). 플레이어↔적탄은 플레이어 주변 9셀만 검사 → 실질 O(n).
- 플레이어 히트박스: **원, `player.hitboxRadius: 6`** (스프라이트 폭 ~28px 대비 대폭 관대). 적탄도 원.
- **무적시간 존재**: HP가 감소한 모든 경우 `player.iframeSec: 1.2` 게임초. 이 게이트가 오라/오빗 패밀리의 다중히트 즉사와 몸통 충돌 연타를 동시에 막는다.
- **몸통 충돌 = 데미지 있음**, `enemy.contactDmg` 1회 적용, i-frame이 게이트. 따라서 "파고들어 몸빵"은 성립하되 공짜는 아니다(수치로 조절).
- 아군 탄은 적탄을 **소거하지 않음**(폭탄만 소거). 관통은 `pierce` + `hitCooldownSec`(동일 대상 재히트 간격)로 표현.

**성능 목표 (기준기: 2020 MacBook Air M1 / Chrome, 하한: 2019 Intel i5 노트북)**

| 항목 | 목표 |
|---|---|
| `step()` 1틱 | 평시 ≤ 0.8ms, 최악(스테이지6 위기세션 @ 캡) ≤ 2.0ms |
| `render()` 1프레임 | ≤ 4.0ms |
| Disaster(×3) 최악 프레임 | 3틱×2.0 + 4.0 = **10ms** (16.6ms 예산 내) |
| core 핫패스 할당 | **0 bytes/tick** (풀 사전할당) |
| 총 다운로드 | ≤ 400KB (JSON ≤ 120KB) |

---

### 7. 데미지 계산식 (구조 고정 · 항 순서 잠금)

```
1. base   = w.dmg                                     (레벨 행에서 읽음)
2. addMul = 1 + Σ(패시브 dmgMulBonus)                 (가산 풀 → 1회 적용. 곱연산 폭주 방지)
3. crit   = rng<critChance ? critMul : 1              (기본 critChance=0 → 꺼져 있으나 항은 존재)
4. elem   = elementMul(stanceElem, targetElem)        (2 / 1 / 0.5)
   if (elem > 1) elem = 1 + (elem-1) × elementBonusMul   (패시브로 ×2를 ×2.2로)
5. gate   = target.isCore && bossPartsAlive ? core.damageTakenMulWhilePartsAlive : 1
6. pre    = base × addMul × crit × elem × gate
7. final  = max( pre × (1 - def/(def + defenseK)) , pre × damageFloorRatio )
8. 적용은 float. 표시만 Math.round.
```
- 방어력 = **정률**, `defenseK` 소프트캡. `damageFloorRatio: 0.10` — 방어력이 데미지를 10% 밑으로 내릴 수 없다.
- 지속형(오라·장판)은 `dmg`가 **1회 적용당** 값이고 `tickIntervalSec`로 주기를 정한다. DPS 정규화 없음 — 모든 데미지는 "적용 1회"로 통일.
- `critChance: 0`으로 출고. 크리티컬을 나중에 켜는 것이 **숫자 변경**이 되도록 항을 지금 넣어둔다.
- **데미지 난수 없음**(가독성 + 시뮬 분산 축소).

---

### 8. JSON 공통 규약

| 규약 | 내용 |
|---|---|
| 위치 | `data/*.json`, 카테고리별 분할(캐시·diff·AI 재생성 단위) |
| 로드 | `main.js`가 `Promise.all(fetch)` 9개 병렬. 정적 호스트에서 문제 없음 |
| 버전 | 모든 파일 루트에 `"schemaVersion": 1`. 불일치 → 로드 실패 + 에러 화면 |
| 검증 | `src/core/schema.mjs`가 **게임과 시뮬 양쪽에서 동일하게** 실행. 개발 빌드는 항상, 배포 빌드도 항상(수 ms) |
| 미지 키 | **에러**. AI가 엔진이 무시하는 필드를 생성하는 것을 차단 |
| 누락 키 | **에러**. 기본값 폴백 금지(조용한 밸런스 드리프트의 근원) |
| 참조 무결성 | 모든 `*Id`는 로드 시 대상 존재 확인 |
| 공정성 | 모든 emitter가 §10 `fairness` 불변조건 통과 필수. 위반 → 로드 실패 |

파일 목록: `rules.json` `elements.json` `weapons.json` `passives.json` `bullets.json` `enemies.json` `bosses.json` `stages.json` `meta.json`(draft·shop·score·difficulty·xp·bot·certify)

---

### 9. `rules.json` — 엔진 상수 · 불변조건 · 팔레트

```json
{
  "schemaVersion": 1,
  "loop":  { "tickHz": 60, "maxStepsPerFrame": 6, "maxFrameGapMs": 250, "interpolate": true },
  "view":  { "...§5..." },
  "caps":  { "...§6..." },
  "collide": { "gridCellPx": 64 },
  "player": {
    "maxHp": 100, "hitboxRadius": 6, "spriteRadius": 14,
    "moveSpeed": 260, "accel": 4200, "decel": 5200,
    "iframeSec": 1.2, "defense": 0, "defenseK": 60, "damageFloorRatio": 0.10,
    "magnetRadius": 90, "orbAutoHomeDelaySec": 3.0, "orbAutoHomeSpeed": 55,
    "bombStockStart": 1, "bombStockMax": 3,
    "startWeaponId": "forward", "startStance": "normal",
    "elementCapPerElement": 4, "elementCapTotal": 6,
    "weaponSlots": 4, "passiveSlots": 6
  },
  "fairness": {
    "minTelegraphSec": 0.55, "minReactionWindowSec": 0.50,
    "maxBulletSpeed": 260, "maxAimedBulletSpeed": 200,
    "minGapWidthPx": 46, "maxSimultaneousEnemyBullets": 320,
    "maxStatusStunSec": 1.0, "minStunTelegraphSec": 1.2
  },
  "input": { "bindings": { "up":"ArrowUp","down":"ArrowDown","left":"ArrowLeft","right":"ArrowRight",
    "stanceNormal":"KeyQ","stanceFire":"KeyW","stanceWater":"KeyE","stanceGrass":"KeyR",
    "bomb":"Space","timeToken":"KeyF","pause":"Escape",
    "draft1":"Digit1","draft2":"Digit2","draft3":"Digit3","reroll":"KeyT",
    "slotPick":["Digit1","Digit2","Digit3","Digit4"], "slotMove":["ArrowLeft","ArrowRight"], "confirm":"Enter" } },
  "palette": { "water":"#2E8BE6","fire":"#E2452F","grass":"#3FBF5B","normal":"#B9BEC7",
               "enemyBullet":"#FF3FE0","telegraph":"#FFE24A","pickupXp":"#7FF0C0","pickupCoin":"#FFC53D" }
}
```
- `fairness.*`는 **로더가 강제하는 기계 검사 조건**이다. "느리고 큰 텔레그래프"라는 정성적 기둥이 여기서 처음으로 검증 가능한 수치가 된다. AI가 트위치 탄막을 생성하면 빌드가 깨진다.
- `palette`는 속성 색의 **단일 진실**이다. 적 틴트·아군 탄·HUD·드래프트 카드·데미지 FX가 전부 이 키를 읽는다(가독성 섹션 소관, 키 소유는 여기).
- `player.*` 의 값은 튜너블, **모델은 잠금**: HP는 정수·바 표시, i-frame 존재, 히트박스는 원.

---

### 10. `elements.json`

```json
{ "schemaVersion": 1,
  "list": ["normal","fire","water","grass"],
  "matrix": { "water":{"fire":2.0,"grass":0.5,"water":1.0,"normal":1.0},
              "fire": {"grass":2.0,"water":0.5,"fire":1.0,"normal":1.0},
              "grass":{"water":2.0,"fire":0.5,"grass":1.0,"normal":1.0},
              "normal":{"water":1.0,"fire":1.0,"grass":1.0,"normal":1.0} },
  "investable": ["fire","water","grass"],
  "normalStance": { "investable": false, "alwaysMul": 1.0 }
}
```
- `investable`에 `normal`이 없다 = **무속성은 투자 축이 아니다.** Q키의 역할은 "부여 해제(전 무기 ×1로 통일)"이며, 이는 ×0.5를 확실히 피하는 안전 스탠스로서 기계적으로 유의미하다(다른 스탠스는 투자 0이 아닌 이상 ×0.5 위험을 동반). 드래프트 풀에 `normal +1` 카드는 존재하지 않는다.
- `matrix`가 데이터이므로 상성 자체도 시뮬로 검증 가능한 숫자다.

---

### 11. `weapons.json` — ★ 코드/데이터 힌지 ★

**구조**: 12(+진화) 패밀리 = **코드의 update 함수** (`src/core/weapons/<family>.js`, 각 30~60줄). **DSL 없음**(DSL은 4주 예산에서 몇 주짜리 항목). 대신 각 패밀리는 **폐쇄된 파라미터 계약**을 선언하고, 모든 수치를 JSON에서 읽는다.

```json
{ "schemaVersion": 1,
  "weapons": [{
    "id": "seeker", "family": "seeker", "name": "시커", "descKey": "w.seeker",
    "elementStampMode": "spawn",
    "base": { "dmg": 6, "cooldownSec": 0.9, "count": 1, "projSpeed": 190, "projRadius": 5,
              "lifetimeSec": 2.4, "pierce": 0, "hitCooldownSec": 0.2, "knockback": 0,
              "turnRateDegSec": 220, "acquireRadius": 320, "retargetSec": 0.3,
              "targetMode": "nearest" },
    "levels": [ {}, {"dmg":8}, {"count":2,"dmg":9}, {"dmg":12,"turnRateDegSec":260},
                {"count":3}, {"dmg":16,"cooldownSec":0.75}, {"pierce":1},
                {"count":4,"dmg":22,"targetMode":"lowestHp"} ],
    "evolveOnMax": true,
    "evolution": { "id":"seeker_swarm", "family":"drone", "name":"시커 스웜",
                   "params": { "droneCount": 3, "droneFireSec": 0.5, "...": "..." } }
  }]
}
```

| 규칙 | 내용 |
|---|---|
| `levels` | **정확히 8행**. 각 행은 `base`에 대한 **부분 오버라이드**(수식 아님 → 곡선이 아니라 표. AI/시뮬이 행 단위로 튠 가능) |
| 레벨업 = 파라미터 델타 | 거동 변화도 파라미터로만 표현(관통 = `pierce:0→1`, 조준 변경 = `targetMode`) |
| 진화 | **`family`를 바꿀 수 있다** → 진화가 진짜 거동 변화가 되면서도 100% 데이터. 슬롯은 그대로 1칸, 불가역, Lv8 도달 시 **자동**(드래프트 픽 소모 없음, `evolveOnMax`) |
| `elementStampMode` | `"spawn"`(생성 시 각인: 탄·기뢰·부메랑) / `"live"`(매 틱 현재 스탠스 재평가: 오라·오빗·드론). **패밀리별 구조 결정 = 잠금 키**(밸런싱 대상 아님) |

**패밀리 파라미터 계약 (폐쇄 목록)** — 공통: `dmg cooldownSec count projSpeed projRadius lifetimeSec pierce hitCooldownSec knockback`

| family | 고유 파라미터 | targetMode 허용 |
|---|---|---|
| `forward` | `spreadDeg jitterDeg burstCount burstIntervalSec` | `forward` |
| `fan` | `arcDeg` | `forward` |
| `seeker` | `turnRateDegSec acquireRadius retargetSec` | `nearest lowestHp randomOnScreen bossPartPriority` |
| `lance` | `beam widthPx chargeSec rangePx` | `forward nearest bossPartPriority` |
| `orbit` | `orbitRadius angularSpeedDegSec bodyCount` | — |
| `aura` | `radius tickIntervalSec falloff` | — |
| `mine` | `placeIntervalSec armSec triggerRadius blastRadius maxAlive zoneDurationSec` | — |
| `boomerang` | `outRangePx returnSpeed canRehit` | `forward nearest` |
| `barrage` | `strikeIntervalSec strikesPerVolley blastRadius telegraphSec` | `randomOnScreen highestHp bossPartPriority` |
| `omni` | `dirCount dirOffsetDeg rearBias` | — |
| `drone` | `droneCount anchorOffsets droneFireSec droneRangePx` | `nearest lowestHp forward bossPartPriority` |
| `nova` | `intervalSec radius expandSec telegraphSec` | — |

> `targetMode` 어휘(폐쇄): `forward nearest lowestHp highestHp randomOnScreen bossPartPriority`. **`bossPartPriority`의 존재가 "복합 보스에서 부위를 지정할 수단"을 데이터로 제공**한다(우선순위 = `parts` 배열 순서). 조준 규칙이 패밀리별 데이터이므로 "어느 부위를 때리나"는 밸런싱 가능한 숫자가 된다.

---

### 12. `passives.json` — 폐쇄 스탯 어휘

```json
{ "schemaVersion": 1,
  "maxLevel": 5,
  "stats": ["dmgMul","fireRateMul","projSpeedMul","projCountAdd","areaMul","durationMul",
            "pierceAdd","moveSpeedMul","maxHpAdd","defenseAdd","magnetRadiusMul",
            "iframeSecAdd","xpGainMul","coinGainMul","bombDmgMul","elementBonusMul",
            "critChance","critMul","statusResistMul","ghostSecOnHit"],
  "passives": [
    { "id":"ghost", "name":"페이즈 클로크", "descKey":"p.ghost",
      "levels":[ {"ghostSecOnHit":0.8},{"ghostSecOnHit":1.2},{"ghostSecOnHit":1.6},
                 {"ghostSecOnHit":2.0},{"ghostSecOnHit":2.6}] }
  ]
}
```
- **패시브는 오직 `stats`의 키만 건드린다.** 엔진에 훅이 없는 스탯은 존재할 수 없다 → AI가 코드 변경을 요구하는 패시브를 생성하는 사고를 원천 차단.
- `ghostSecOnHit`(피격 후 적 조준·유도 타겟에서 제외)처럼 **기존의 특색 있는 후보를 스탯 어휘에 미리 편입**해 둔다. 어휘에 없는 아이디어 = 설계 변경.
- `*Mul` 스탯은 **가산 풀**(Σ 후 1회 적용, §7-2), `*Add`는 단순 가산. 곱연산 스택 없음.
- **패시브는 Lv1~5** — 이것이 성장 예산 산술(§21)을 성립시키는 축이다.

---

### 13. `bullets.json` · emitter 프리미티브

```json
{ "schemaVersion": 1,
  "bullets": [ {"id":"round_m","radius":7,"speed":140,"dmg":8,"shape":"circle",
                "status":null,"accel":0,"turnRateDegSec":0} ] }
```

**emitter 프리미티브 8종 (코드) / 파라미터 (데이터)** — 공통: `type bulletId from telegraphSec everySec offsetSec repeat restSec status`

| type | 파라미터 |
|---|---|
| `straight` | `count spreadDeg speed` |
| `fan` | `count arcDeg speed` |
| `aimed` | `count spreadDeg speed leadSec` |
| `ring` | `count speed rotOffsetDeg` |
| `spiral` | `count speed rotStepDeg durationSec rateSec` |
| `laser` | `chargeSec widthPx activeSec angleDeg trackDuringCharge` |
| `zone` | `radius telegraphSec activeSec dps` |
| `wall` | `count gapCount gapWidthPx speed` |

- **적 탄에는 속성이 없다** — `bullets.json`에 `element` 키가 **존재하지 않는다**(스키마가 규칙을 강제). 색은 `palette.enemyBullet` 단일 색.
- `status ∈ {null,"slow","stun"}`. `stun`은 `fairness.minStunTelegraphSec` 강제.
- 이 emitter 엔진을 **플레이어 무기와 적이 공유**한다(문서의 "탄 엔진 재사용" = 이 표).

---

### 14. `enemies.json`

```json
{ "schemaVersion": 1,
  "enemies": [{
    "id":"drifter_s", "tier":"mob", "name":"드리프터",
    "elementPolicy":"theme", "element":null,
    "hp":18, "contactDmg":6, "radius":9, "spriteId":"drifter",
    "move":{"type":"sineDown","speed":70,"ampPx":40,"freqHz":0.6,"entrySide":"top"},
    "attack":{"emitterId":"straight_1","firstDelaySec":0.6},
    "xp":3, "score":10,
    "coin":{"chance":0.06,"amount":1},
    "healDrop":{"chance":0.0,"amount":0},
    "hpScalePerStage":0.18
  }]
}
```
| 필드 | 규칙 |
|---|---|
| `tier` | `"mob" \| "elite" \| "midboss"` — **엘리트가 여기서 정의된다.** 엘리트 = 잡몹 로스터의 별도 tier 엔트리(별개 개체, 중간보스 아님) |
| `elementPolicy` | `"theme"`(스테이지 70/30 규칙으로 스폰 시 결정) / `"fixed"`(`element` 명시) |
| `hpScalePerStage` | `hp × (1 + hpScalePerStage × (stageIndex-1))` — **스테이지 진행 스케일은 로스터 교체가 아니라 계수**(콘텐츠 물량 1배 유지). 난이도(배속)와 직교 |
| `move.type` | 폐쇄 어휘: `straightDown sineDown diveAtPlayer enterStop strafeLR arcIn orbitPoint retreatUp` |
| `move.entrySide` | `top bottom left right` — **후방 진입이 데이터로 존재**하므로 `omni`(후방/전방위) 패밀리의 존재 이유가 성립 |

**적 데이터 스키마가 이 표로 동결된다.** 필드 추가는 밸런싱이 아니라 스키마 변경(= 시뮬·JSON·로더 동시 변경)이며 별도 결정을 요한다.

---

### 15. `bosses.json`

```json
{ "schemaVersion": 1,
  "bosses": [{
    "id":"manta", "name":"강철 가오리", "tier":"stage",
    "core": { "element":"normal", "hp":4000, "radius":42,
              "damageTakenMulWhilePartsAlive": 0.25 },
    "parts": [
      {"id":"engine","element":"fire","hp":1800,"radius":18,"anchor":[0,44],
       "regenSec":null,"onDestroy":[{"op":"dmgMul","v":0.6},{"op":"emitterOff","v":"e_laser"}]},
      {"id":"wingL","element":"water","hp":1400,"radius":22,"anchor":[-58,0],
       "regenSec":null,"onDestroy":[{"op":"moveSpeedMul","v":0.7}]},
      {"id":"wingR","element":"water","hp":1400,"radius":22,"anchor":[58,0],
       "regenSec":null,"onDestroy":[{"op":"moveSpeedMul","v":0.7}]}
    ],
    "phases": [
      {"id":"p1","until":{"coreHpBelow":0.66},"emitters":[
        {"type":"fan","from":"core","bulletId":"round_m","telegraphSec":0.8,
         "count":9,"arcDeg":120,"speed":140,"everySec":2.4,"offsetSec":0,"repeat":3,"restSec":1.2,"status":null}]},
      {"id":"p2","until":{"coreHpBelow":0.33},"emitters":[ "..." ]},
      {"id":"p3","until":null,"emitters":[ "..." ]}
    ]
  }]
}
```
| 규칙 | 내용 |
|---|---|
| 보스 HP | **코어 HP가 곧 보스 HP**. 부위 HP는 별도(합산 아님) → 3분 타이머가 재는 것 = 코어 DPS |
| **코어 게이팅** | 하드 게이트 없음. 부위가 하나라도 살아있으면 코어 피해가 `damageTakenMulWhilePartsAlive`(=0.25) 배. **부위를 무시한 코어 직행이 가능하지만 4배 느리다** → 스탠스 퍼즐이 강하게 인센티브되면서 브릭은 없다. 그리고 이 긴장 전체가 **숫자 하나(0.25)** 다 |
| 부위 파괴 효과 | 폐쇄 어휘: `moveSpeedMul dmgMul emitterOff spawnWave phaseAdvance openCore`. **보스별 특수 코드 금지** → 보스 6종 + 중간보스 N종을 코드 추가 0으로 저작 가능(4주 예산의 핵심) |
| `regenSec` | `null` = 재생 없음. 키는 존재(튜너블) |
| `tier` | `"stage" \| "mid" \| "final"` — 중간보스도 동일 스키마(부위 0~1개짜리 축약형) |
| 폭탄 | `bomb.bossDmgRatio`는 **코어 최대 HP 기준**, 부위에는 적용 안 함. 캡 = `bomb.bossDmgCap` |
| 부위 속성 구성 제약 | 검증: 스테이지 보스는 `parts`의 속성이 **2종 이상**이어야 함(단일 스탠스 샌드백 방지). 로더가 강제 |

---

### 16. `stages.json`

```json
{ "schemaVersion": 1,
  "themeDraw": { "pool":["sea","forest","volcano","swamp","tundra","canyon"],
                 "count":5, "allowRepeat":false,
                 "guaranteeElements":["water","fire","grass"],
                 "firstStageFrom":["sea","forest"],
                 "finalStageId":"apex" },
  "stages": [{
    "id":"sea", "element":"water", "skinId":"sea", "bossId":"manta",
    "mobPhase": {
      "durationSec": 120, "endMode": "timer", "skippable": true,
      "mix": { "themeShare":0.70, "otherShare":0.30, "otherWeights":{"fire":0.34,"grass":0.33,"normal":0.33} },
      "mixGranularity": "perWave",
      "waves":[ {"atSec":0,"enemyId":"drifter_s","formation":"vLine","count":7,
                 "entrySide":"top","spacingPx":48,"intervalSec":0.15,"xPct":0.5} ],
      "elites":[ {"atSec":40,"enemyId":"reef_elite","count":1} ],
      "midBoss":[ {"atSec":55,"bossId":"kraken_mini","despawnAfterSec":40} ],
      "crisis": {"atSec":95,"waveId":"birds_swarm","durationSec":18,"warnSec":2.5}
    },
    "boss": { "timerSec":180, "entrySec":3.0, "redAlertSec":30 }
  }],
  "formations": { "vLine":"...", "line":"...", "stream":"...", "arc":"...", "flank":"...", "swarm":"..." }
}
```
| 규칙 | 내용 |
|---|---|
| `stages[].element` | `∈ {water,fire,grass}` — **스키마가 노말 테마를 금지**한다. 속성 투자가 통째로 죽는 스테이지는 존재할 수 없다 |
| `themeDraw` | 풀 6 중 5개 무중복 추첨 + `guaranteeElements` → **모든 런에 물·불·풀 스테이지가 최소 1개씩 보장** = "막다른 운" 구조적 제거. `firstStageFrom`으로 1스테이지 극단 테마 방지 |
| `mixGranularity: "perWave"` | 70/30 롤은 **웨이브 단위 1회**(개체 단위 아님). 웨이브가 속성적으로 일관되어 읽을 수 있고, "테마 = 예측 가능"이라는 정보 약속이 성립 |
| `endMode: "timer"` + `skippable` | 잡몹 페이즈는 게임시간 고정 타이머. 남은 적은 보스 진입 시 **소멸하고 XP는 자동 정산 없이 소멸**(= 파밍 압력의 실제 메커니즘). 조기 스킵 가능(XP ↔ 보스 타이머 여유의 트레이드) |
| `midBoss[].despawnAfterSec` | **"선택적"의 기계적 정의**: 무시하면 시간 후 화면 밖으로 퇴장. 추격하지 않음 |
| `crisis.warnSec` | 위기 세션은 예고된다(무-트위치 기둥). 값이 데이터 |

---

### 17. `meta.json` — 드래프트 · XP · 상점 · 점수 · 난이도 · 봇 · 인증

```json
{ "schemaVersion": 1,

  "xp": { "curve":"poly", "base":6, "exp":1.32, "stageBonusMul":1.0,
          "orbLifetimeSec":null, "levelUpQueueMode":"serial" },

  "draft": {
    "optionCount": 3,
    "categoryWeights": { "newWeapon":30, "weaponLevel":40, "elementLevel":15, "passive":15 },
    "distinctCategories": false,
    "noDuplicateWithinDraft": true,
    "filterInvalid": true,
    "renormalizeOnEmptyCategory": true,
    "newWeaponWhenSlotsFull": "excluded",
    "pityElementLevelBy": 8,
    "fallback": { "type":"coins", "amount":25 },
    "pauseGame": true, "pauseBossTimer": false,
    "reroll": { "granularity":"all3", "canRepeatPrevious":false, "chainable":true }
  },

  "shop": {
    "openBetweenStages": true, "stockMode":"full", "skippable": true,
    "items": [
      {"id":"reroll",    "kind":"consumable","maxStack":3, "basePrice":30, "priceCurve":{"type":"mul","factor":1.6},"scope":"run"},
      {"id":"potion",    "kind":"consumable","maxStack":3, "basePrice":40, "priceCurve":{"type":"mul","factor":1.7},"scope":"run","healPct":0.35},
      {"id":"bomb",      "kind":"consumable","maxStack":3, "basePrice":35, "priceCurve":{"type":"mul","factor":1.4},"scope":"run"},
      {"id":"shield",    "kind":"consumable","maxStack":2, "basePrice":60, "priceCurve":{"type":"mul","factor":1.8},"scope":"run"},
      {"id":"timeToken", "kind":"consumable","maxStack":2, "basePrice":70, "priceCurve":{"type":"mul","factor":1.8},"scope":"run","addGameSec":30,"requiresRedAlert":true},
      {"id":"continue",  "kind":"oneShot",   "maxStack":1, "basePrice":250,"priceCurve":{"type":"mul","factor":1.0},"scope":"run"},
      {"id":"defense",   "kind":"stat","stat":"defenseAdd",     "maxStack":4,"basePrice":45,"priceCurve":{"type":"add","step":20},"scope":"run","perStack":15},
      {"id":"maxhp",     "kind":"stat","stat":"maxHpAdd",       "maxStack":4,"basePrice":50,"priceCurve":{"type":"add","step":25},"scope":"run","perStack":10},
      {"id":"movespeed", "kind":"stat","stat":"moveSpeedMul",   "maxStack":3,"basePrice":55,"priceCurve":{"type":"add","step":30},"scope":"run","perStack":0.06},
      {"id":"magnet",    "kind":"stat","stat":"magnetRadiusMul","maxStack":3,"basePrice":40,"priceCurve":{"type":"add","step":20},"scope":"run","perStack":0.30},
      {"id":"resist",    "kind":"stat","stat":"statusResistMul","maxStack":2,"basePrice":45,"priceCurve":{"type":"add","step":25},"scope":"run","perStack":0.30}
    ]
  },

  "bomb": { "mobDmg":9999, "clearsEnemyBullets":true, "clearsDuringBoss":true,
            "bossDmgRatio":0.04, "bossDmgCap":220, "iframeSec":1.5, "castSec":0.4 },

  "score": {
    "superEffectiveKillMul": 1.5,
    "attribution": "majorityDamage",
    "bossTimeRemainPerGameSec": 40,
    "noHitScope": "perBoss", "noHitBonus": 3000,
    "perfectScope": "perRun", "perfectBonus": 20000, "shieldBreaksNoHit": false,
    "coinToScore": 10,
    "difficultyMul": { "normal":1.0, "hard":1.6, "hell":2.5, "disaster":4.0 }
  },

  "difficulty": { "normal":{"speed":1.0}, "hard":{"speed":1.5}, "hell":{"speed":2.0}, "disaster":{"speed":3.0},
                  "stunFromDifficulty":"hard" }
}
```
**규칙 고정 (숫자가 아니라 판정)**
- `draft.filterInvalid: true` — 맥스 무기/캡 도달 속성/만석 시 새 무기는 **추첨 전에 풀에서 제거**. 무효 선택지가 3택에 섞이는 일이 없다(정보 기반 선택 기둥).
- `newWeaponWhenSlotsFull: "excluded"` — 교체 없음. 슬롯 순서(=부여 순서)가 무관한 선택 중에 조용히 바뀌는 사고를 차단.
- `pityElementLevelBy: 8` — 8회 연속 속성 카드가 없으면 강제 1장. 시그니처 시스템이 운으로 사라지지 않는다.
- `fallback` — 유효 후보 < 3이면 코인 카드로 채움(빈 드래프트 없음).
- `draft.pauseGame: true / pauseBossTimer: false` — **화면은 멈추지만 보스 타이머는 계속 간다.** 사고 시간은 주되(무-트위치) 3분 허들은 무력화되지 않는다. 타이머는 드래프트 화면에도 표시.
- `score.attribution: "majorityDamage"` — 상성 처치 보너스는 **누적 데미지 과반 속성** 기준. 막타 기준이면 "막타만 상성으로 맞추기"라는 체스식 최적화가 생기므로 채택하지 않는다.
- `score.noHitScope: "perBoss"` — 런 전체 무피격은 사실상 도달 불가 = 죽은 점수면. 보스별이면 매 스테이지 살아있는 목표.
- `perfect` = 런 전체 무피격 + 컨티뉴 미사용. 정의가 여기 존재한다(유령 개념 제거).
- `bomb.clearsDuringBoss: true` — 보스전에서도 탄 소거. 그것이 폭탄의 유일한 정체성이며, 데미지는 `bossDmgRatio`(코어 최대 HP 기준)로 봉인.
- `timeToken.requiresRedAlert: true` + `stages[].boss.redAlertSec` — 빨간불은 **발동 조건**이고 그 임계는 데이터. 연장 단위는 **게임초**(`addGameSec`, 배속과 일관).
- `maxStack` — 모든 스탯 항목에 상한. 무한 스택으로 코인이 빌드를 사는 우회로를 차단.

---

### 18. 튜너블 인벤토리 (전체 노브)

> **잠금(구조)** = 밸런싱에서 손대지 않음. 바꾸면 설계 변경 + 재인증. **튜너블** = AI 시뮬이 자유롭게 조정.

| 그룹 | 잠금(구조) | 튜너블 |
|---|---|---|
| 엔진 | `tickHz` `view.*` `caps.*` `onCap.*` `gridCellPx` `maxStepsPerFrame` | — |
| 상성 | `elements.list` `investable` `normalStance.investable` | `matrix.*` (2.0/0.5) |
| 플레이어 | HP=정수·바, i-frame 존재, 히트박스=원 | `maxHp hitboxRadius moveSpeed accel decel iframeSec defense defenseK damageFloorRatio magnetRadius orbAutoHomeDelaySec orbAutoHomeSpeed` |
| 슬롯·투자 | `weaponSlots=4` `passiveSlots=6` `elementCapPerElement=4` | `elementCapTotal`(기본 6) |
| 무기 | `family` `elementStampMode` `targetMode` 어휘, `levels` 8행 구조 | `base.*` 전 키, `levels[].*` 전 키, `evolution.params.*` |
| 패시브 | `stats` 어휘, `maxLevel=5` | `passives[].levels[].*` |
| 탄·탄막 | emitter 8종 어휘, `bullets`에 `element` 없음 | `bullets[].*`, emitter 전 파라미터 |
| 적 | 필드 집합, `move.type` 어휘, `tier` 어휘 | `hp contactDmg radius xp score coin.* healDrop.* hpScalePerStage move.* attack.*` |
| 보스 | 부위 파괴 효과 어휘, 코어=보스HP, 소프트게이트 방식 | `core.hp` `damageTakenMulWhilePartsAlive` `parts[].hp` `regenSec` `phases[].until.*` 전 emitter |
| 스테이지 | `element ∈ {water,fire,grass}`, `mixGranularity`, `endMode` | `durationSec` `themeShare` `otherWeights` `waves[].*` `elites[].*` `midBoss[].*` `crisis.*` `boss.timerSec`(=180) `entrySec` `redAlertSec` |
| 테마 추첨 | `allowRepeat=false` `guaranteeElements` | `pool` `count` `firstStageFrom` |
| XP | `curve` 형태, `levelUpQueueMode` | `base`(6) `exp`(1.32) `stageBonusMul` |
| 드래프트 | `optionCount=3` `filterInvalid` `newWeaponWhenSlotsFull` `pauseGame/pauseBossTimer` `reroll.granularity` | `categoryWeights.*` `pityElementLevelBy` `fallback.amount` |
| 상점 | 항목 집합, `kind`, `scope`, 금지 규칙 | `basePrice priceCurve.* maxStack perStack healPct addGameSec` |
| 폭탄 | `clearsEnemyBullets` `clearsDuringBoss` | `mobDmg bossDmgRatio bossDmgCap iframeSec castSec` |
| 점수 | `attribution` `noHitScope` `perfectScope` | `superEffectiveKillMul bossTimeRemainPerGameSec noHitBonus perfectBonus coinToScore difficultyMul.*` |
| 난이도 | `speed` = 유일 축, 보스 타이머 = 게임시간 3분 | `speed` 값, `difficultyMul` 값 |
| 공정성 | 불변조건이 **존재**한다는 사실 | `minTelegraphSec maxBulletSpeed minGapWidthPx ...` (완화는 곧 기둥 약화 → 리뷰 필요) |
| 봇·인증 | 봇 모델 구조 | `bot.*` `certify.*` 전 임계값 |

**성장 예산 산술 (스키마가 보장하는 불변조건)**

| 투자처 | 픽 수 |
|---|---|
| 새 무기 | 4 |
| 무기 레벨 (4 × Lv1→8) | 28 |
| 진화 | 0 (자동) |
| 속성 레벨 (`elementCapTotal`) | 6 |
| 패시브 획득 | 6 |
| 패시브 레벨 (6 × Lv1→5) | 24 |
| **합계** | **68** |

레벨업 ~50-60회 < 68 → **런 끝에도 전부 못 찍는다** = "선택이 의미를 가진다"가 산술적으로 성립하고, `elementCapTotal: 6`(< 4×3=12)이 "불+4·물+4·풀+4 = 항상 최적 매칭" 붕괴를 봉인한다. 이 부등식은 시뮬이 매 인증마다 검사한다(`certify.growthBudget`). 곡선을 바꿔 레벨업 횟수가 68을 넘기면 **인증 실패**.

---

### 19. localStorage 스키마

> ⚠️ 이 게임은 블로그 전체(`jasonleex1995.github.io`)와 **단일 origin의 localStorage를 공유**한다. 접두사 없는 키는 블로그 상태와 충돌한다.

```
key: "nan2026.v1"          // 단일 키. 접두사 필수
```
```json
{ "v": 1,
  "best": { "normal":{"score":0,"seed":null,"at":0,"stage":0},
            "hard":{...}, "hell":{...}, "disaster":{...} },
  "opts": { "mute":false, "volume":0.7, "shake":1.0, "flash":1.0, "cbMode":"off", "damageNumbers":true },
  "seen": { "tutorial":false } }
```
| 규칙 | 내용 |
|---|---|
| 읽기 | `try/catch` 필수. throw(Safari 프라이빗)·파싱 실패·`v` 불일치 → **기본값으로 조용히 시작** |
| 쓰기 | `try/catch` 필수. 실패는 무시(런 종료 화면이 죽지 않게) |
| 마이그레이션 | `v` 불일치 = **초기화**. 오락실 감성상 점수는 저렴하다. 상위 `v` 데이터는 덮어쓰지 않음(멀티탭 안전) |
| 무결성 | **없음. 조작 가능함을 명시적으로 수용**한다(로컬 전용·계정 없음). 이 결정은 재론하지 않는다 |
| 초기화 수단 | 옵션 화면 "기록 삭제"(확인 1회) |

---

### 20. 헤드리스 시뮬 (`tools/sim.mjs`)

```bash
node tools/check.mjs                                   # 순수성·스키마·공정성 게이트
node tools/sim.mjs --runs 8000 --seed 0 --out tools/report/
node tools/sim.mjs --certify                           # 임계값 판정, 실패 시 exit 1
```
**게임 로직 재사용 방식**: `sim.mjs`가 `../src/core/step.js`를 **그대로 import** 한다. 렌더·오디오·DOM은 애초에 core에 없으므로 스텁조차 필요 없다. 루프는 `while(!world.over) step(world, bot(world))` — rAF도 시계도 없이 게임시간만 흐른다. **시뮬이 인증하는 코드 = 배포되는 코드**(문자 그대로 동일 파일). 이것이 §2 모듈 경계 규칙의 존재 이유다.

**봇 모델 (`meta.json > bot`)**
```json
"bot": {
  "reactionMs": 250, "reactionJitterMs": 80,
  "dodgeLookaheadSec": 0.8, "aimErrorPx": 12,
  "stanceSwitchMs": 180, "grazeTolerancePx": 4,
  "policies": {
    "draft": ["specialist","generalist","weaponRush","elementRush","greedyDps","random"],
    "farm":  ["maxFarm","balanced","ignoreOptional"],
    "stance":["greedyNearest","majorityOnScreen","bossPartTarget","static"]
  },
  "baseline": { "draft":"generalist", "farm":"balanced", "stance":"greedyNearest" }
}
```
★ **핵심**: 반응 지연은 **실제 ms로 선언하고 난이도 배속으로 게임틱에 환산**한다.
```
latencyTicks = round(reactionMs / 1000 × TICK_HZ × speed)
// Normal ×1 → 15틱 / Disaster ×3 → 45틱 동안 눈이 감긴 채 진행
```
이것이 없으면 게임시간 시뮬은 4난이도에서 **비트 동일**해 난이도를 아무것도 증명하지 못한다. 이 한 줄이 "난이도 = 인간 반응시간뿐"을 검증 가능한 모델로 만든다.

**인증 임계값 (`meta.json > certify`)** — 정성 서술을 기계 판정으로
```json
"certify": {
  "runs": 8000,
  "clearRate":       { "baseline":"normal", "min":0.35, "max":0.65 },
  "bossTimeoutRate": { "max":0.25 },
  "noDeadLuck":      { "minClearRateWorstThemeOrder":0.15, "minClearRateWorstDraftPolicy":0.20 },
  "dominance":       { "maxWeaponPickShare":0.16, "maxWeaponWinShare":0.20, "maxElementWinShare":0.35 },
  "coinScarcity":    { "medianEndCoins":{"min":0,"max":80}, "medianAffordablePerVisit":{"min":1,"max":3} },
  "growthBudget":    { "maxLevelUps":68 },
  "capHits":         { "max":0 },
  "fairnessViolations": { "max":0 },
  "stanceValue":     { "minClearRateDeltaVsStatic":0.20 },
  "difficultySpread":{ "disasterClearRate":{"min":0.02,"max":0.12} }
}
```
| 지표 | 무엇을 증명하나 |
|---|---|
| `clearRate 0.35~0.65` | "하드벗페어 창"의 정량 정의 |
| `noDeadLuck` | 어떤 테마 순서·드래프트 정책도 막다른 길이 아님 |
| `dominance` | 지배 무기·속성 없음 |
| `coinScarcity` | "항상 살짝 부족" |
| **`stanceValue`** | `static`(전환 안 함) 봇 대비 `greedyNearest` 봇이 클리어율 **+20pp 이상** → **속성 스탠스가 장식이 아니라 실제 축임을 수치로 증명**. 공모전 기술문서의 최강 근거 |
| `difficultySpread` | 배속이 실제로 난이도를 만드는지 (봇 지연 모델이 살아있는지) |

**텔레메트리 산출**(`tools/report/`, 공모전 기술문서 근거물): `summary.json`(위 지표 전부) · `weapons.csv`(픽률·승률·DPS 기여) · `elements.csv` · `bosses.csv`(격파 시간 분포·타임아웃률) · `economy.csv`(스테이지별 코인 수급/지출) · `deaths.csv`(사인 분류: HP/타이머).

---

### 21. 검증 게이트 (`tools/check.mjs`) — AI 산출물이 통과해야 하는 문

1. **core 순수성**: 금지 식별자 정적 검사 + `src/core/weapons/**` 숫자 리터럴 검사
2. **스키마**: 전 JSON을 `schema.mjs`로 검증 — 타입·필수 키·**미지 키 거부**·참조 무결성
3. **공정성 불변조건**: 모든 emitter가 `rules.fairness` 통과 (텔레그래프 ≥ 0.55s, 탄속 ≤ 260, 틈 ≥ 46px …)
4. **구조 제약**: 스테이지 보스 부위 속성 ≥ 2종, `stages[].element ≠ normal`, 무기 `levels` 정확히 8행, `elementCapTotal < 3 × elementCapPerElement`
5. **성장 예산**: XP 곡선으로 계산한 최대 레벨업 횟수 < 68

**개발 흐름**: AI가 JSON 생성 → `node tools/check.mjs` → `node tools/sim.mjs --certify` → 통과 시에만 커밋. **어느 하나라도 실패하면 커밋하지 않는다.** 이 파이프라인 자체가 "AI가 설계·밸런싱·공정성 인증"이라는 제출 스토리의 실물이다.
