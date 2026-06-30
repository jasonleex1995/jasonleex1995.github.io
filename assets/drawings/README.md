# Drawings

내가 그린 그림을 여기에 보관합니다. 메타데이터는 `data/gallery.json`에서 관리해요
(사진과 공용 매니페스트, `type: "drawing"`으로 구분).

## 그림 추가하기

1. **이미지 준비 — WebP 무손실 권장**
   - 그림은 평면 색·또렷한 선·투명 배경이 많아 **WebP 무손실**이 최적이에요 (PNG보다 작고, 손실 0, 투명도 유지):
     ```bash
     cwebp -lossless -m 6 내그림.png -o assets/drawings/202606.webp
     ```
   - 한 변이 2000px를 넘으면 `-resize 1600 0`(세로면 `0 1600`)도 함께
   - PNG/JPG 그대로 올려도 동작은 하지만, **사진과 달리 그림은 JPG에서 윤곽선이 뭉개지고 투명도가 깨질 수 있어요** → 그림엔 JPG 비추천
2. **이 폴더에 저장**: `assets/drawings/` (파일명은 사진과 같은 `YYYYMM` 규칙 권장 — 예: `202606.webp`)
3. **매니페스트 등록**: `data/gallery.json`의 `items` 배열 맨 위에
   ```json
   { "type": "drawing", "file": "202606.webp", "date": "2026-06", "caption": "", "alt": "" }
   ```

`type`을 `"drawing"`으로 두면 갤러리의 **Drawings** 필터에 묶이고, 파일은 이 폴더에서 찾습니다.
`date`는 `"YYYY-MM"` 형식이며 캡션으로 표시됩니다.
