# 매출 분석과 고객 대응방안 추천 API

## 목적

CSV 매출 분석과 고객 대응방안 추천·검증을 분리한다. 여러 자료를 분석하더라도 사용자가
선택한 분석에 대해서만 에이전트를 실행하여 불필요한 추천 연산과 LLM 호출을 방지한다.

## 사용자 흐름

```text
CSV 업로드
  → FastAPI 매출 분석 및 analysis_id 발급
  → Spring Boot가 응답을 MySQL에 저장
  → 분석 결과 조회(Spring Boot/MySQL)
  → 사용자가 대응방안 추천 여부 선택
      ├─ 원하지 않음: 종료
      └─ 원함: Spring Boot가 MySQL 분석 결과를 FastAPI에 재전달
                → 추천·효과 추정·근거 검색·검증 실행
                  → 사용자 승인/수정/거절
```

## API 계약

### 1. 매출 분석

`POST /api/v1/analyses`

`multipart/form-data` 필드:

- `file`: CSV 파일
- `trdar_cd`: 상권 코드
- `svc_induty_cd`: 서비스 업종 코드
- `yyqu_cd`: 기준 분기 코드(선택)
- `user_id`: Spring Boot 사용자·기업 식별자(신규 연동에서 필수)
- `store_id`: 매장 식별자(선택)

이 요청은 매출 분석까지만 수행한다. 분석 결과와 에이전트 입력용 진단 결과를
`model/analyses.sqlite3`에 저장하고 `analysis_id`를 반환한다.

### 2. 분석 결과 조회

`GET /api/v1/analyses/{analysis_id}`

저장된 `report`, `diagnosis`, `warnings` 및 분석 기준 코드를 반환한다. 존재하지 않는 ID는
`404`를 반환한다.

사용자 식별 기록은 `X-User-Id` 헤더가 저장된 `user_id`와 일치해야 조회할 수 있다.
`GET /api/v1/analyses`는 해당 사용자의 분석 이력을 최신순으로 반환하며, `store_id` 쿼리로
매장별 이력을 필터링한다. 기존 무식별 기록은 마이그레이션 호환을 위해 소유권 검사에서 제외한다.

### 3. 선택적 대응방안 추천·검증

신규 연동은 `POST /api/v1/recommendations`를 사용한다. Spring Boot가 MySQL에서 조회한 분석
결과를 다음 형태로 전달한다(`camelCase`와 `snake_case` 모두 허용).

```json
{
  "analysisId": "analysis-1",
  "userId": "user-a",
  "storeId": "store-1",
  "trdarCd": "3120189",
  "svcIndutyCd": "CS100010",
  "yyquCd": 20261,
  "diagnosis": {},
  "warnings": []
}
```

`X-User-Id` 헤더는 필수이며 본문의 `userId`와 일치해야 한다. 이 API는 FastAPI 내부
`analyses.sqlite3`를 조회하지 않고 전달받은 진단 결과로 에이전트를 시작한다.

기존 `POST /api/v1/analyses/{analysis_id}/recommendations`는 마이그레이션 호환용으로
유지한다.

`POST /api/v1/analyses/{analysis_id}/recommendations`

저장된 진단 결과를 LangGraph 초기 상태에 전달한다. 매출 진단은 다시 수행하지 않으며,
추천 → 효과 추정/근거 검색 → 정책 검증 후 승인 대기 상태에서 `thread_id`를 반환한다.
분석의 `user_id`와 `store_id`는 에이전트 상태와 캠페인 로그까지 자동 전달된다.

### 4. 에이전트 상태 조회

`GET /api/v1/agent-runs/{thread_id}`

추천·검증 진행 결과와 현재 승인 대기 정보를 반환한다.
사용자 식별 실행은 `X-User-Id` 헤더로 소유권을 확인한다.

### 5. 승인 워크플로우 재개

`POST /api/v1/agent-runs/{thread_id}/resume`

요청 예시:

```json
{
  "decision": "approve"
}
```

수정 시에는 다음처럼 후보 방안을 전달한다.

```json
{
  "decision": "edit",
  "modificationPlan": "쿠폰발행"
}
```

`decision`은 `approve`, `edit`, `reject` 중 하나다. 이전 한글 필드인 `결정`, `수정_방안`도
마이그레이션 기간 동안 입력 호환을 유지한다.

## 레거시 API

- `POST /api/v1/reports`
- `POST /api/v1/agent-runs`

두 API는 기존 테스트와 호출자 호환을 위해 deprecated 상태로 유지한다. 신규 Spring Boot
연동에서는 사용하지 않으며, 새 흐름이 안정화된 뒤 제거한다.

캠페인 로그 API는 추천 실행 후 성과 학습을 위한 내부 운영 API이므로 사용자용 Spring API에
직접 노출하지 않는다.

## 관련 개발 계획

인증 계약, Spring Boot 종단간 연동, deprecated API 정리는 `development_plan.md`의 1~3단계를
따른다.
