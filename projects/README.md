# Projects

각 프로젝트는 **자기만의 독립된 페이지**를 가집니다 (예: `projects/timeblindness/index.html`).
`projects.html`(인덱스)는 `data/projects.json`을 읽어 카드 목록을 만들고, 각 카드는 해당 프로젝트 페이지로 연결됩니다.

## 새 프로젝트 추가하기

### 1. 프로젝트 페이지 만들기
`_template/`을 복사해서 슬러그 이름의 폴더로:

```bash
cp -r projects/_template projects/my-project
```

그 안의 `index.html`을 채우세요 (제목, 저자, 학회, 링크, teaser 이미지, abstract, BibTeX).
- teaser 이미지 등 자료는 그 폴더 안에 두고 상대경로로 참조 (`teaser.webp`)
- 사진은 웹용으로 압축 권장 (폭 1600px WebP) — `assets/photos/README.md`의 변환법 참고
- 완전히 다른 디자인(예: Nerfies 템플릿)을 쓰고 싶으면, 그 폴더의 `index.html`을 통째로 교체해도 됩니다. 인덱스는 URL만 알면 되니까요.

> 페이지 주소는 `https://jasonleex1995.github.io/projects/my-project/`가 됩니다.

### 2. 인덱스 목록에 등록
`data/projects.json`의 `projects` 배열 맨 위에 (최신이 위로):

```json
{
  "title": "My Project Title",
  "venue": "ICCV",
  "year": 2026,
  "url": "projects/my-project/",
  "description": "한 줄 요약."
}
```

- `title`: 프로젝트 제목
- `venue` / `year`: 학회·연도 (둘 다 생략 가능)
- `url`: 프로젝트 페이지 경로 (블로그 내부면 `projects/슬러그/`, 외부면 전체 URL도 가능)
- `description`: 카드에 보일 한 줄 설명 (생략 가능)

> 인덱스 카드는 **텍스트만** (제목·venue·설명)으로 정사각형 형태입니다. 썸네일은 쓰지 않아요 — 대표 이미지는 각 프로젝트 페이지 안에서 보여주세요.

저장 후 push하면 Projects 탭에 카드가 나타납니다. (목록이 비어 있으면 "coming soon" 문구가 보여요.)

## 미리보기
`projects.html`은 `data/projects.json`을 fetch로 읽으므로, 로컬에선 `file://`이 아니라 서버로 봐야 해요:
```bash
python3 -m http.server 8000   # http://localhost:8000/projects.html
```
개별 프로젝트 페이지는 **상대경로만 쓰면** 더블클릭(`file://`)으로도 열려요 — `_template`·`exif-frame`이 그렇습니다.
그래서 `_template`은 `../../styles.css` 형태를 씁니다. **이 규칙을 깨지 마세요** (루트 절대경로 `/styles.css`로 바꾸면
배포된 사이트에선 멀쩡한데 로컬 더블클릭에서만 무스타일로 깨집니다 — 알아채기 어려운 종류의 고장이에요).

> 예외: `prism-wing`은 게임을 iframe으로 물고 있고 그 게임(`projects/nan2026/`)은 ES 모듈이라
> `file://`에서 동작하지 않습니다. 이 둘은 위의 `python3 -m http.server 8000`으로 여세요.
