# jasonleex1995.github.io

개인 사이트 — 순수 HTML/CSS. 빌드 단계·프레임워크 없음. GitHub Pages로 배포.

> **콘텐츠를 추가/수정하거나 발행하려면 → [MAINTAINING.md](MAINTAINING.md) (유지보수·발행 가이드)를 보세요.**
> 이 README는 사이트가 무엇이고 어떻게 구성됐는지에 대한 개요입니다.

---

## 사이트 구조

```
index.html        Home (자기소개)
resume.html       Education / Publications / Experience / Awards
projects.html     프로젝트 인덱스 (data/projects.json → 카드 목록)
books.html        독서 기록 (카테고리별)
gallery.html      사진 + 그림 갤러리 (data/gallery.json, 필터 + 라이트박스)
styles.css        모든 페이지 공용 스타일시트 (디자인 토큰은 :root)

assets/
  favicon.ico     브라우저 탭 아이콘
  profile.webp    프로필 사진 (홈)
  photos/         갤러리 사진 (웹용 압축본) — README.md 포함
  drawings/       갤러리 그림 — README.md 포함

projects/         개별 프로젝트 페이지 (각 프로젝트 = 독립 HTML)
  _template/      새 프로젝트 페이지 시작 템플릿
  README.md       프로젝트 추가 절차

data/
  gallery.json    사진/그림 메타데이터 (type 필드로 구분)
  projects.json   프로젝트 인덱스 목록

.nojekyll                      Pages가 Jekyll 처리하지 않게 함 (브랜치 배포라 필수)
```

콘텐츠는 두 가지 방식으로 관리됩니다:
- **HTML 직접 편집** — 책, 논문, 경력, 자기소개
- **`data/*.json`에 항목 추가** — 갤러리(사진/그림), 프로젝트 목록 (페이지가 JS로 자동 렌더링)

구체적인 작업 방법·예시·주의사항은 모두 **[MAINTAINING.md](MAINTAINING.md)**에 있습니다.

---

## 배포

`main` 브랜치에 push하면 **GitHub Pages가 그 브랜치를 그대로 서빙**합니다. 빌드도, Actions 워크플로도 없습니다 — 저장소에 있는 파일이 곧 사이트입니다.

```bash
git add -A && git commit -m "..." && git push
```

보통 ~1분 안에 반영됩니다. 반영 주소: [jasonleex1995.github.io](https://jasonleex1995.github.io)

> **새 저장소에 처음 올릴 때 (2가지만):**
> 1. 저장소 이름은 **`<username>.github.io`** (유저 사이트)여야 해요. 사이트가 도메인 루트(`/`) 기준 경로를 쓰기 때문 — 다른 이름의 프로젝트 저장소면 로고의 홈 링크가 깨집니다.
> 2. 저장소 **Settings → Pages → Source = "Deploy from a branch" → `main` / `(root)`** 로 한 번 설정. (그 뒤론 push만 하면 반영)
>
> push는 됐는데 화면이 그대로면 **브라우저 캐시** → 하드 리프레시(`⌘+Shift+R`).

---

## 디자인 철학 — 왜 순수 HTML/CSS인가

- **의존성 0** — Ruby/Node/Jekyll/Gem 등 외부 도구 없음. 몇 년 뒤에도 그대로 작동.
- **빌드 단계 0** — push하면 그대로 배포. 빌드 캐시·빌드 에러 같은 개념 자체가 없음.
- **`.nojekyll`** — GitHub Pages에 "그대로 서빙해줘"라고 알려주는 빈 파일.
- **시스템 폰트** — 외부 폰트 CDN 의존 없음. 어떤 OS에서도 깨끗하게.
- **단일 CSS 파일** — 모든 스타일 한 곳. 색/폰트/간격은 `:root` 변수로 중앙 관리.

**CI 의존성도 0** — Actions 워크플로가 없으니 업데이트할 액션도, 깨질 빌드도 없습니다. Pages가 `main`을 그대로 서빙할 뿐입니다.
