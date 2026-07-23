# 고객 대응방안 추천 실행 예시

과거 종단간 실행 두 건의 핵심 결과를 보존했다. 기존 예시는 deprecated 직접 실행 API를
사용했으므로 재현 절차는 현재 분석 저장 후 선택적 추천 흐름으로 갱신했다.

## 상도역 호프·간이주점

| 단계 | 결과 |
|---|---|
| 입력 | `trdar_cd=3110835`, `svc_induty_cd=CS100009`, 2026년 1분기 |
| 진단 | 수요이탈, 고객 회복 필요, 위험도 0.959 |
| 추천 | 쿠폰발행, 정책 `backfill-v1` |
| SCM | donor 30개, 실측 효과 탐색적 `+18.61%` |
| RAG | 쿠폰 학술 방향성 및 플랫폼 참고 수치 검색 |
| OPE | 사용 가능, DR 0.0642, 기준정책 대비 +0.0077 |
| 승인 | 리포트 생성, 수치 위반 없음 |

## 강남역 카페

| 단계 | 결과 |
|---|---|
| 입력 | `trdar_cd=3120189`, `svc_induty_cd=CS100010`, 2026년 1분기 |
| 진단 | 정상·강점 확대, 20대 구성비 강점 |
| 추천 | 지역 제휴 마케팅, 정책 `backfill-v1` |
| SCM | donor 30개, 실측 사례 4건, 탐색적 `+5.02%` |
| RAG | 연결 가능한 방향성·허용 수치 없음 |
| OPE | 사용 가능, DR 0.1373, 기준정책 대비 +0.0878 |
| 승인 | 효과 수치를 만들지 않은 방향성 리포트 생성 |

## 현재 재현 절차

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

1. `POST /api/v1/analyses`에 CSV와 분석 기준을 전송한다.
2. `POST /api/v1/analyses/{analysis_id}/recommendations`를 호출한다.
3. 반환된 `thread_id`의 승인 대기 내용을 확인한다.
4. `POST /api/v1/agent-runs/{thread_id}/resume`에 `approve`, `edit`, `reject`를 전송한다.

상세 필드와 소유권 헤더는 `analysis_recommendation_api.md`를 따른다.
