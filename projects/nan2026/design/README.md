# design/ — 설계 문서 지도

> 입구는 [`../DESIGN.md`](../DESIGN.md). 이 폴더는 설계의 본문이다.

## 읽는 순서

1. **[`CANON.md`](CANON.md) — 정본 (v1.4, ~5,200행)** ★ **유일한 진실**
   값·규칙·공식·스키마·검증 게이트의 **단일 소유자**. 다른 모든 문서는 여기를 **인용**만 한다.
   충돌하면 **정본이 옳다.**

2. **섹션 4종** — 정본을 인용해 각 영역을 상세화한 것. **값을 소유하지 않는다.**
   | 파일 | 내용 |
   |---|---|
   | [`02-readability-hud.md`](02-readability-hud.md) | 색 언어 · 텔레그래프 · 상태이상 표시 · 히트 피드백 · HUD 전 좌표 · 색맹 |
   | [`03-weapons-passives-items.md`](03-weapons-passives-items.md) | 무기 12종(거동×진화) · 패시브 12종 · 상점 10항목 |
   | [`04-enemies-bosses-stages.md`](04-enemies-bosses-stages.md) | 테마 6+최종 · 적 17종 · 중간보스 3 · 새떼 · 보스 7종 |
   | [`05-flow-shop-score-onboarding.md`](05-flow-shop-score-onboarding.md) | 타이틀→런→상점→결과 · 점수 · 컨티뉴 · 60초 온보딩 |

3. **값은 문서가 아니라 [`../data/*.json`](../data/)에 있다.** 정본의 표 = 그 JSON이 가져야 하는 값(C-8).

## 검증

```bash
cd projects/nan2026
node tools/check.mjs      # exit 0 이어야 한다
```

현재: **VIOLATION 0 · CANON 0 · AMBIGUOUS 0 · STUB 20 · SKIP 2 → exit 0**
- **STUB 20** = 시뮬(`tools/sim.mjs`)이 있어야 재는 동적 게이트. 미구현이라 TODO로 출력된다.
- **SKIP 2** = `src/`가 아직 없어 평가 대상이 없는 검사(S1 core 순수성 · S11 RNG 스트림). 개발 시작 시 살아난다.

## 기록 · 이력 (참고용, 스펙 아님)

| 파일 | 내용 |
|---|---|
| [`_VERIFY-ROUND2.md`](_VERIFY-ROUND2.md) | 정본 v1.0 검증 — 모순 18 · 미결 17 |
| [`_VERIFY-ROUND3.md`](_VERIFY-ROUND3.md) | 정본 v1.1 검증 — "DPS 520↔708 = 단위 충돌" · "보스 HP는 armor에" 발견 |
| [`_VERIFY-ROUND4.md`](_VERIFY-ROUND4.md) | 정본 v1.2 검증 — 게이트 12계열 전부 통과 |
| [`_AUDIT-TRANSCRIPTION.md`](_AUDIT-TRANSCRIPTION.md) | ★ 전사 감사 — 산문 리뷰 3라운드가 못 잡은 33건을 찾은 기록 |
| [`_drafts/`](_drafts/) | ⚠️ **폐기된 초안 6종.** 서로 모순됨. **절대 스펙으로 쓰지 말 것** |

## 이 폴더의 규칙

- **정본에 없는 값을 섹션이 만들지 않는다.** 필요하면 섹션 말미 "정본 추가 요청"에 올리고, 정본이 심사해 본문에 반영한다.
- **"채택했다"는 "인쇄했다"가 아니다** (C-9). 요청 목록에 적어두는 것은 해결이 아니다.
- **개정 이력에 "고쳤다"를 적기 전에 본문을 grep 한다** (C-11).

> 이 세 줄은 v1.0→v1.3의 실패가 **전부 같은 형태**(정본이 자기 말을 지키는지 확인하지 않음)였기 때문에 생겼다.
