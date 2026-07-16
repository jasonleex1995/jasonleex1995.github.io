# NAN 2026 — 게임 프로젝트 입구

> ⚠️ **이 문서는 스펙이 아니라 안내판이다.** 값·규칙의 유일한 진실은 **[`design/CANON.md`](design/CANON.md)**이고,
> 값의 유일한 거처는 **[`data/*.json`](data/)**이다. 이 문서에는 **숫자를 적지 않는다.**
> *(v0.5까지 이 파일이 스펙이었으나, CANON 도입 후 폐기하고 입구로 전환했다 — 두 문서가 다른 답을 말하는 것을 막기 위해.)*

## 한 줄 소개

**뱀서식 무기 아스널 × 물·불·풀·노말 속성 스탠스 × 194X식 비행 슈팅.**
랜덤 테마 스테이지를 돌며 속성을 맞춰 갖춰 양학하는 오락실 로그라이트. 완전 오프라인·무API·바닐라 JS.

- **목표**: [NAN 2026](https://nan2026.nhn.com/) (NHN Game × AI 해커톤) 사전과제 — *AI를 활용한 게임 제작*
- **마감**: 2026-08-10
- **플레이 주소(예정)**: `https://jasonleex1995.github.io/projects/nan2026/`

## 지금 어디까지 왔나

| 단계 | 상태 |
|---|---|
| 기획 (CANON v1.4) | ✅ 완료 |
| 데이터화 (`data/*.json` 9종) | ✅ 완료 — 빈칸 0 |
| 정적 검증 (`tools/check.mjs`) | ✅ **통과** (`node tools/check.mjs` → exit 0) |
| 게임 코드 (`src/`) | ⬜ 미착수 ← **다음** |
| AI 콘텐츠 생성 + 시뮬 밸런싱 (`tools/sim.mjs`) | ⬜ 미착수 |
| 제출물 (영상·PDF 2종) | ⬜ 미착수 |

## 문서 지도 — 무엇을 어디서 보나

| 궁금한 것 | 볼 곳 |
|---|---|
| **값·규칙 (모든 것의 진실)** | **[`design/CANON.md`](design/CANON.md)** — 유일한 정본 |
| 실제 값 | [`data/*.json`](data/) — rules · elements · weapons · passives · bullets · enemies · bosses · stages · meta |
| 이 설계가 옳은지 검사 | [`tools/check.mjs`](tools/check.mjs) — `node tools/check.mjs` |
| 색·텔레그래프·HUD 배치 | [`design/02-readability-hud.md`](design/02-readability-hud.md) |
| 무기 12종·패시브·상점 | [`design/03-weapons-passives-items.md`](design/03-weapons-passives-items.md) |
| 적·보스·스테이지 | [`design/04-enemies-bosses-stages.md`](design/04-enemies-bosses-stages.md) |
| 화면 흐름·점수·온보딩 | [`design/05-flow-shop-score-onboarding.md`](design/05-flow-shop-score-onboarding.md) |
| **왜 이렇게 정했나 (대화 기록)** | **[`log.md`](log.md)** — 공모전 'AI 활용 기술 문서'의 원천 |
| 검증 리포트 (라운드 2~4) | [`design/_VERIFY-ROUND*.md`](design/) · [`design/_AUDIT-TRANSCRIPTION.md`](design/_AUDIT-TRANSCRIPTION.md) |
| 폐기된 초안 (참고만) | [`design/_drafts/`](design/_drafts/) — ⚠️ 서로 모순됨. 스펙 아님 |

> 섹션 문서(02~05)는 **정본을 인용**할 뿐 값을 소유하지 않는다. 값이 다르면 **정본이 옳다**.

## 절대 원칙

1. **개발 중 설계 결정 0** — 막히면 그것은 정본의 결함이다. 즉석에서 정하지 말고 정본을 고쳐라.
2. **밸런싱은 오직 숫자만** — `data/*.json`의 값만 바꾼다. 값을 바꾸려고 `.js`를 여는 순간 정본 실패.
   - 정본의 표에 인쇄된 값을 바꾸면 **정본의 표도 함께 고친다** (둘이 갈라지면 실패).
3. **바꾼 뒤엔 반드시** `node tools/check.mjs` → **exit 0** 확인.

## 커밋 규칙 (공모전 심사가 커밋 히스토리를 본다)

- 게임 커밋은 접두어 **`nan2026:`**, 가능한 한 `projects/nan2026/` 파일만 건드린다.
- 심사자는 `git log --oneline -- projects/nan2026/`로 게임 개발 과정만 본다.

## 다음 할 일

1. `src/` 착수 — 이때부터 `check.mjs`의 S1(core 순수성)·S11(RNG 스트림)이 살아난다.
2. `tools/sim.mjs` — 헤드리스 시뮬. `check.mjs`의 **STUB 20건**(동적 게이트: 클리어율·보스 타임아웃·지배도·코인 희소성 등)이 여기서 채워진다.
3. 제출물: 플레이 빌드(GitHub Pages) · YouTube 30~60초 · 게임 소개 PDF · **AI 활용 기술 문서 PDF**(`log.md` 기반).
4. `data/projects.json`에 카드 등록 (플레이 가능해진 뒤).
