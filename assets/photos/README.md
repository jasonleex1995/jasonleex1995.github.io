# Photos

갤러리 사진을 보관합니다. 메타데이터는 `data/gallery.json`에서 관리해요
(그림과 공용 매니페스트, `type: "photo"`로 구분).

> 현재 사진들은 [EXIF Frame](../../projects/exif-frame/) 도구로 프레임을 입힌 뒤 webp로 변환한 것입니다.
> 파일명은 `YYYYMM.webp`(예: `202606.webp`) 규칙을 씁니다.

## 사진 추가하기

1. **(선택) EXIF Frame으로 프레임 입히기** — `projects/exif-frame/`에 원본 사진을 넣고,
   프레임이 적용된 **PNG(무손실·풀해상도)** 를 내려받습니다 (보통 `~/Downloads`로).
2. **webp로 변환** — 갤러리에 실제로 쓰이는 가벼운 파일. **q95, 긴 변 2400px** (소스 = 1에서 받은 PNG):
   ```bash
   cd assets/photos
   # 가로 사진
   cwebp -q 95 -m 6 -resize 2400 0 ~/Downloads/202606.png -o 202606.webp
   # 세로 사진
   cwebp -q 95 -m 6 -resize 0 2400 ~/Downloads/202606.png -o 202606.webp
   ```
   - 긴 변이 2400px가 되게: 가로면 `2400 0`, 세로면 `0 2400`
   - q95면 화면에선 원본과 구분 불가. 한 장 ~200-700KB.
   - `cwebp` 설치: `brew install webp`
3. **매니페스트 등록**: `data/gallery.json`의 `items` 배열에 (배열 위쪽 = 갤러리 앞쪽)
   ```json
   { "type": "photo", "file": "202606.webp", "date": "2026-06", "caption": "", "alt": "" }
   ```
   `date`는 `"YYYY-MM"` 형식이며 캡션으로 표시됩니다. `caption`을 쓰면 date 대신 그 문구가 보입니다.

> 💡 **원본 PNG는 레포에 보관하지 않습니다 — webp만 커밋하세요.** 원본이 다시 필요하면
> EXIF Frame으로 언제든 재생성할 수 있어요(소스 사진만 있으면). 꼭 남기고 싶으면 개인 백업으로.

## 카메라 원본을 프레임 없이 바로 올리는 경우

- [Squoosh](https://squoosh.app/)에 끌어다 WebP로 내보내기 — EXIF 회전을 자동 반영
- 또는 `cwebp -q 95 -m 6 -resize 2400 0 IMG_1234.JPG -o IMG_1234.webp` (세로면 `0 2400`)

> ⚠️ **세로 사진 주의**: `cwebp`는 카메라 JPEG의 EXIF 회전 정보를 webp에 반영하지 **않아서**
> 세로 사진이 갤러리에서 90° 누울 수 있습니다. Squoosh를 쓰면 자동 해결돼요.
> (EXIF Frame이 출력한 PNG는 이미 방향이 올바라서 이 문제가 없습니다.)

## 회전 고치기 (사진이 누워 보일 때)

이미 올린 webp가 90° 누워 보이면, 원본 없이도 이 자리에서 회전시킬 수 있습니다
(`-r` 값: 시계방향 각도. 누운 방향에 따라 `90` 또는 `270`):

```bash
cd assets/photos
dwebp -quiet 202606.webp -o /tmp/t.png        # webp → png
sips -r 90 /tmp/t.png --out /tmp/t.png         # 시계방향 90° 회전
cwebp -quiet -q 95 /tmp/t.png -o 202606.webp   # 다시 webp로 덮어쓰기
```

방향이 반대면 `90` 대신 `270`을, 뒤집혔으면 `180`을 쓰세요.
