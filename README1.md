# BP20-AI

## 서비스 목록

여러 독립적인 FastAPI 서비스를 이 레포 하나에서 관리합니다. 서비스마다 폴더가 분리되어 있고,
각자 자기만의 `requirements.txt`/`Dockerfile`/`tests`를 가집니다 (의존성이 서로 섞이지 않도록).

| 서비스 | 폴더 | 설명 |
|---|---|---|
| 영수증 OCR · AI 가계부 · 원가분석 | [`services/receipt-ocr-analytics/`](services/receipt-ocr-analytics/) | PaddleOCR 기반 영수증 인식 + 지출/원가 통계 분석 |

> 새 AI 기능을 추가하실 때도 `services/<기능이름>/` 밑에 독립적으로 넣어주세요.
> 레포 루트에 바로 코드를 두면 다른 서비스의 `requirements.txt` 등과 충돌합니다.
