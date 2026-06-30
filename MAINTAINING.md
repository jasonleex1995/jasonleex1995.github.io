# 유지보수 · 발행 가이드 (Maintainer's Handbook)

이 블로그의 콘텐츠를 추가/수정하고 발행(배포)하는 방법을 정리한 문서입니다.
**코드를 깊이 몰라도** 이 문서만 따라 하면 책·사진·그림·프로젝트·논문 등을 올릴 수 있습니다.

> 사이트 전체 개요·기술 설명은 [README.md](README.md)를 보세요. 이 문서는 "실제로 어떻게 올리나"에 집중합니다.

---

## 0. 30초 요약 — 이 사이트가 동작하는 방식

- **순수 HTML/CSS 정적 사이트.** 빌드 과정이 없습니다.
- 콘텐츠 추가는 둘 중 하나예요:
  - **HTML 파일을 직접 편집** (책, 논문, 경력, 자기소개 등)
  - **`data/*.json`에 한 줄 추가** (갤러리 사진/그림, 프로젝트 목록)
- 수정한 파일을 **GitHub `main` 브랜치에 push하면 1분 안에 자동 배포**됩니다.

---

## 1. 시작 전 준비물

- **저장소 접근 권한** (GitHub collaborator로 초대받아야 함)
- **텍스트 에디터** — [VS Code](https://code.visualstudio.com/) 권장
- **두 가지 작업 방식 중 택1:**
  - **(쉬움) GitHub 웹에서 직접 편집** — 텍스트 몇 줄 고치는 정도면 브라우저에서 파일 열고 연필 아이콘으로 수정 → "Commit". git 설치 불필요.
  - **(권장) 로컬에서 작업** — `git clone` → 수정 → `git push`. 이미지 추가나 미리보기가 필요하면 이 방식.
- (선택) **로컬 미리보기용** `python3` — 갤러리/프로젝트 페이지 확인에 필요 (아래 §4)

---

## 2. 모든 작업의 공통 흐름

```bash
# 1) 최신 상태 받기
git pull

# 2) 파일 수정 (아래 §3에서 작업별 안내)

# 3) (권장) 로컬 미리보기로 확인 — §4

# 4) 발행: 커밋 + 푸시
git add -A
git commit -m "books: add OOO"      # 무엇을 했는지 짧게
git push

# 5) 배포 확인: GitHub 저장소 → Actions 탭 → 초록 체크(약 1분)
#    반영 주소: https://jasonleex1995.github.io
```

> GitHub 웹에서 편집한 경우 2~5단계가 "Commit changes" 버튼 한 번으로 끝납니다.

---

## 3. 작업별 가이드

### 빠른 참조

| 하고 싶은 것 | 건드리는 곳 |
|---|---|
| 책 추가 | `books.html` |
| 사진 / 그림 추가 | `assets/photos/` 또는 `assets/drawings/` + `data/gallery.json` |
| 프로젝트 추가 | `projects/<슬러그>/` + `data/projects.json` |
| 논문 추가 | `resume.html` (Publications) |
| 경력 추가 | `resume.html` (Experience) |
| 자기소개 수정 | `index.html` |
| 프로필 사진 교체 | `assets/profile.webp` + `index.html` |
| 소셜 링크 | 모든 HTML의 footer |

---

### 3.1 책 추가 — `books.html`

해당 **카테고리**(`<h2>소설</h2>` 등) 안의 `<ul class="book-cards">`에서, **발행일 내림차순** 위치에 카드 한 장을 넣습니다 (최신 발행일이 위).

```html
<li>
  <article class="book-card">
    <div class="book-card-title">책 제목</div>
    <div class="book-card-author">저자명</div>
    <div class="book-card-meta">출판사 &middot; 2026.01.15</div>
  </article>
</li>
```

- 날짜 형식: **`YYYY.MM.DD`**
- `&middot;`는 가운뎃점(·) 기호입니다. 그대로 두세요.
- 위치: 같은 카테고리 안에서 발행일이 더 최신인 책들 **아래**, 더 오래된 책들 **위**.

---

### 3.2 사진 / 그림 추가 — 갤러리

**2단계**입니다: ① 이미지 파일을 폴더에 넣고 ② `data/gallery.json`에 한 줄 등록.

**① 이미지 준비 & 저장**
- **사진**: 원본은 수 MB라 **반드시 압축**해서 webp로 올립니다 (**q95, 긴 변 2400px**).
  - 보통 [EXIF Frame](projects/exif-frame/)으로 프레임 입힌 PNG를 받아 → webp로 변환. (원본 PNG는 레포에 보관 안 함 — webp만 커밋. 필요하면 EXIF Frame으로 재생성)
  - 명령어: `cwebp -q 95 -m 6 -resize 2400 0 ~/Downloads/202606.png -o assets/photos/202606.webp` (세로 사진은 `-resize 0 2400`)
  - 프레임 없이 카메라 원본을 바로 쓸 땐 [Squoosh](https://squoosh.app/)가 가장 쉬움(회전 자동 처리)
  - → webp는 `assets/photos/`에 저장. 자세한 절차는 `assets/photos/README.md`
- **그림**: 보통 작아서 그대로 둬도 됩니다 → `assets/drawings/`에 저장
- 파일명은 **영문/숫자**로 (한글 파일명 X)

**② `data/gallery.json`의 `items` 배열 맨 위에 추가** (최신이 위로):

```json
{ "type": "photo", "file": "202606.webp", "date": "2026-06", "caption": "", "alt": "" }
```

- `type`: `"photo"` 또는 `"drawing"` — 이 값으로 어느 폴더에서 파일을 찾을지 정해집니다
- `file`: 저장한 파일명 (확장자 포함, 대소문자 정확히)
- `date`: **`"YYYY-MM"`** (갤러리는 이 형식! 책의 `YYYY.MM.DD`와 다름)
- `caption` / `alt`: 비워둬도 됨

> ⚠️ 세로로 찍은 사진이 가로로 누워 보이면 회전 문제입니다 → `assets/photos/README.md`의 "회전 고치기" 참고. Squoosh를 쓰면 대개 안 생겨요.

---

### 3.3 프로젝트 추가

각 프로젝트는 **자기만의 페이지**(`projects/<슬러그>/index.html`)를 갖고, `projects.html`이 목록을 보여줍니다.

```bash
cp -r projects/_template projects/my-project   # 템플릿 복사
# projects/my-project/index.html 을 채우기 (제목/저자/학회/링크/teaser/abstract/BibTeX)
```

그다음 `data/projects.json`의 `projects` 배열 맨 위에:

```json
{ "title": "제목", "venue": "ICCV", "year": 2026,
  "url": "projects/my-project/", "description": "한 줄 요약" }
```

> 인덱스 카드는 **썸네일 없이** 제목·부제(venue·year)·설명만 보여줍니다. 이미지는 각 프로젝트 페이지 안에서 자유롭게 쓰세요.

자세한 절차는 `projects/README.md` 참고.

---

### 3.4 논문 추가 — `resume.html`

`<h2>Publications</h2>` 섹션에서 기존 항목 하나를 복사해 **맨 위**에 붙이고 내용만 교체:

```html
<div class="entry">
  <h3>논문 제목</h3>
  <p class="muted"><em>Venue 2026</em> &middot; <a href="https://arxiv.org/abs/...">arXiv</a></p>
</div>
```

---

### 3.5 경력 추가 — `resume.html`

`<h2>Experience</h2>` 섹션, 가장 최근 경력이 맨 위:

```html
<div class="entry">
  <p class="meta">Mar 2026 &mdash; present</p>
  <h3>직책 &middot; 회사명 <span class="subtle small">(full-time)</span></h3>
  <p class="muted">주요 역할/프로젝트 설명.</p>
</div>
```

---

### 3.6 자기소개 / 프로필 사진 — `index.html`

- **자기소개 글**: `<div class="hero-text">` 안의 `<p class="lead">` 단락을 직접 수정.
- **프로필 사진 교체**:
  1. 새 사진을 폭 500px WebP로: `cwebp -q 85 -resize 500 0 새사진.jpg -o assets/profile.webp` (또는 Squoosh)
  2. `index.html`의 `<img class="profile-photo" ... width="…" height="…">`에서 width/height를 새 사진 픽셀 크기로 맞추기

---

### 3.7 소셜 링크 — 모든 HTML의 footer

`<ul class="footer-links">` 안에 추가. **모든 페이지**(index, resume, projects, books, gallery, 그리고 projects/*/index.html)에 똑같이 넣어야 합니다 (공용 footer로 묶는 빌드 시스템이 없는 게 trade-off).

```html
<li><a href="https://scholar.google.com/citations?user=...">Scholar</a></li>
```

---

## 4. 로컬에서 미리 보기 (발행 전 확인)

- **대부분의 페이지**(index, resume, books): HTML 파일을 더블클릭하면 브라우저에서 바로 열립니다.
- **gallery.html / projects.html은 예외**: `data/*.json`을 읽는 구조라 `file://`에선 안 떠요. 로컬 서버를 띄우세요:

```bash
cd <저장소 폴더>
python3 -m http.server 8000
# 브라우저에서 http://localhost:8000 접속
```

---

## 5. 주의사항 (이것만 지키면 안 깨집니다)

- 🟢 **건드려도 되는 것**: 위 §3에 나온 콘텐츠 파일들 (`*.html`의 본문, `data/*.json`, `assets/`의 이미지).
- 🔴 **건드리지 말 것**:
  - `styles.css` — 디자인을 바꿀 때만. (색/폰트는 맨 위 `:root` 변수만 수정)
  - `gallery.html` / `projects.html` 안의 `<script>` 블록 — 렌더링 로직. 수정 불필요.
- **이미지는 반드시 압축**해서 올리기 (원본 수 MB 그대로 X).
- **날짜 형식 주의**: 책 = `YYYY.MM.DD`, 갤러리 = `YYYY-MM`.
- **JSON 문법 주의**: 따옴표 `"`, 항목 사이 쉼표 `,`, **마지막 항목 뒤에는 쉼표 금지**. 헷갈리면 [jsonlint.com](https://jsonlint.com/)에 붙여넣어 검사.
- **한글 파일명 금지** (이미지 등) — 영문/숫자로.

---

## 6. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| 이미지가 안 보임 (깨진 아이콘) | 파일명·경로·**대소문자** 불일치. `data/gallery.json`의 `file`과 실제 파일명을 정확히 맞추기 |
| 갤러리/프로젝트가 안 바뀜 | `data/*.json` **문법 오류**(쉼표 누락/초과). jsonlint로 검사 |
| 사진이 옆으로 누움 | 회전(EXIF) 문제 → `assets/photos/README.md`의 "회전 고치기" |
| push했는데 사이트 반영 안 됨 | ① GitHub **Actions 탭**에서 배포 성공(초록)인지 확인(빨강이면 로그). ② 배포는 성공인데 화면이 그대로면 거의 **브라우저 캐시** → **하드 리프레시**(Mac `⌘+Shift+R`, Win `Ctrl+F5`). 특히 `styles.css`·이미지가 캐시에 잘 남아요 |
| 첫 배포 실패 | 저장소 **Settings → Pages → Source = GitHub Actions** 인지 확인 |

---

## 7. 더 자세한 절차 (참고 문서)

- 이미지 압축·회전: [`assets/photos/README.md`](assets/photos/README.md)
- 그림 추가: [`assets/drawings/README.md`](assets/drawings/README.md)
- 프로젝트 페이지: [`projects/README.md`](projects/README.md)
