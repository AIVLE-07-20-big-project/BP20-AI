# BP20 ECS Fargate 배포 개발계획

## 1. 문서 목적

현재 프로젝트는 FE·BE·AI가 저장소별로 분리되어 있지만, 개발 환경에서는 BE의 Docker Compose가 Spring Boot, FastAPI, Celery, Redis, MySQL을 함께 실행한다.

운영 환경에서는 Amazon ECS Fargate를 사용해 FE·BE·AI를 독립적으로 배포하고, 각 서비스의 배포·확장·장애 격리를 분리한다.

이 문서는 다음 작업의 기준을 정의한다.

- 개발용 Docker Compose와 운영용 ECS 구성을 분리
- Fargate의 비영속 파일시스템에 맞게 SQLite·CSV·모델 저장 구조 개선
- FE·BE·AI별 독립 CI/CD 구성
- 운영 데이터와 비밀값의 안전한 저장

## 진행 상태 요약 (2026-08-03)

| 영역 | 상태 | 비고 |
|---|---|---|
| 개발용 SQLite fallback | ✅ 완료 | 개발 Docker Compose 호환을 위해 유지 |
| AI 작업·분석 결과 MySQL 전환 코드 | ✅ 완료 | `AI_DATABASE_URL` 설정 시 AI 전용 MySQL 테이블 사용 |
| LangGraph MySQL Checkpointer | ✅ 완료 | `ai_agent_checkpoints`, `ai_agent_writes` |
| S3 모델·RAG·Dataset 동기화 | ✅ 완료 | S3 설정 시 시작 시 필요한 자산 다운로드 |
| S3 임시 CSV 처리 | ✅ 완료 | `uploads/v1/{job_id}.csv`, 완료 후 삭제 |
| Bandit·피드백·캠페인 로그 보존 | ✅ 1차 완료 | Bandit/피드백 이벤트/S3 로그 snapshot 연결 |
| BE MySQL·S3 계약 | ⏳ 진행 중 | BE repository 및 compose 반영 필요 |
| Redis 운영 endpoint | ⏳ 대기 | ElastiCache 생성 후 연결 |
| FE/BE/AI 운영 이미지 | ⏳ 대기 | ECS 배포용 Dockerfile/이미지 필요 |
| ECS Fargate·CI/CD | ⏳ 대기 | 팀 인프라 구성 후 진행 |

---

## 2. 현재 구조

### 2.1 저장소

```text
20BGFE/BP20-FE   React + Vite
20BGBE/BP20-BE   Spring Boot + MySQL 연동
20BGAI/BP20-AI   FastAPI + Celery + AI/RAG
```

### 2.2 개발용 Compose

현재 BE `compose.yaml`은 다음 서비스를 함께 실행한다.

```text
Spring Boot
FastAPI
Celery Worker
Celery Beat
Redis
MySQL
```

이 구성은 로컬 개발과 통합 테스트를 위한 것이므로 유지한다. 운영 배포를 위해 기존 Compose를 즉시 분리하거나 삭제하지 않는다.

### 2.3 현재 데이터 저장 구조

| 영역 | 현재 저장소 | 주요 용도 |
|---|---|---|
| BE 업무 데이터 | MySQL | 사용자, 매장, 매출, 분석 결과, 추천 이력, 효과 검증 |
| AI 작업 상태 | 운영 MySQL / 개발 SQLite fallback | `ai_analysis_jobs` |
| AI 분석 결과 | 운영 MySQL / 개발 SQLite fallback | `ai_internal_analyses` |
| LangGraph 상태 | 운영 MySQL / 개발 SQLite fallback | `ai_agent_checkpoints`, `ai_agent_writes` |
| Celery 큐 | Redis | 분석 작업 큐와 결과 백엔드 |
| AI 피드백 | CSV·모델 파일 | `ai_sales_feedback.csv`, Bandit 모델 |
| 모델·RAG | 로컬 파일/Hugging Face | pickle, RAG 인덱스, 기준 데이터 |
| 업로드 원본 | 임시 CSV | 분석 완료 후 삭제 |

---

## 3. 목표 운영 아키텍처

```text
사용자
  ↓
Application Load Balancer
  ├─ /      → FE Fargate Service
  └─ /api   → BE Fargate Service
                  ↓
             AI FastAPI Fargate Service
                  ├─ AI Worker Fargate Service
                  └─ Redis/ElastiCache

BE → RDS MySQL
AI → AI 상태 DB
FE·BE·AI 이미지 → Amazon ECR
비밀값 → Secrets Manager 또는 SSM Parameter Store
업로드·모델·결과 파일 → 하나의 S3 버킷 내 prefix 분리
```

### 3.1 ECS 서비스 구성

```text
bp20-fe
bp20-be
bp20-ai-api
bp20-ai-worker
bp20-ai-beat 또는 EventBridge Scheduler
```

FE·BE·AI API는 ECS Service로 운영한다. Celery Worker도 별도 ECS Service로 운영해 API와 독립적으로 확장한다.

Celery Beat는 항상 실행되는 Service로 둘 수 있지만, 정기 작업 중심으로 전환할 경우 EventBridge Scheduler가 Fargate Task를 실행하는 방식도 검토한다.

### 3.2 네트워크

- ALB만 외부에 공개
- FE는 ALB의 `/` 경로 사용
- BE는 ALB의 `/api` 경로 사용
- AI API는 외부에 공개하지 않고 BE에서만 접근
- MySQL과 Redis는 외부에 공개하지 않음
- ECS 서비스는 private subnet 배치를 우선 검토

운영 환경의 서비스 간 주소는 Docker Compose 서비스명 대신 ECS Service Connect, Cloud Map 또는 내부 ALB 주소를 사용한다.

---

## 4. 데이터 저장소 전환 계획

## 4.1 SQLite → MySQL 전환

현재 AI는 다음 SQLite 파일에 의존한다.

```text
analysis_jobs.sqlite3
analyses.sqlite3
agent_runs.sqlite3
```

Fargate의 로컬 저장공간은 태스크 교체 시 보존을 보장하지 않으므로 운영용 AI에서 SQLite를 그대로 사용하지 않는다. 운영 RDS MySQL에 AI 전용 테이블을 추가한다.

### 목표

```text
analysis_jobs  → 운영 DB
analyses       → 운영 DB
agent_runs     → 운영 DB 기반 LangGraph checkpointer
```

### 현재 결정 및 권장안

BE가 사용하는 MySQL과 같은 RDS를 사용할 수 있지만, AI 내부 상태는 별도 테이블과 별도 DB 계정으로 관리한다.

우선순위는 다음과 같다.

1. `analysis_jobs`를 MySQL `ai_analysis_jobs`로 이전
2. `analyses`를 MySQL `ai_internal_analyses`로 이전하거나 기존 `ai_analyses`와 API 계약으로 통합
3. `agent_runs`를 MySQL `ai_agent_checkpoints`와 `ai_agent_writes`로 이전
4. LangGraph `SqliteSaver`를 MySQL 기반 checkpointer adapter로 교체
5. 운영에서는 SQLite를 사용하지 않되, 개발 Compose 호환을 위해 SQLite fallback은 유지

AI가 BE MySQL에 직접 접근하도록 만들기보다는 AI 상태 저장소를 별도 DB로 두는 것을 기본 방향으로 한다.

### 현재 구현 상태

LangGraph의 현재 `SqliteSaver`는 MySQL로 환경변수만 바꿔서 전환할 수 없으므로
`MySQLCheckpointer`를 구현했다. `AI_DATABASE_URL`이 설정된 운영 환경에서는
`ai_agent_checkpoints`와 `ai_agent_writes`를 사용하고, 개발 환경에서는 기존
`SqliteSaver` fallback을 사용한다. AI용 MySQL 계정에는 AI 테이블만 접근하도록 권한을 제한한다.

## 4.2 CSV 업로드 전환

현재 AI 분석 요청 CSV는 최대 25MiB까지 임시 파일로 저장한 후 분석 완료 시 삭제한다.

운영에서는 다음 방식으로 전환한다.

```text
FE
→ BE
→ AI API
→ S3 `uploads/v1/{job_id}.csv`
→ AI Worker가 S3에서 읽음
→ 분석 완료 후 삭제
```

### 작업 항목

- ✅ 하나의 S3 버킷 사용 및 prefix 분리
- ✅ `uploads/v1/{job_id}.csv` object key 규칙 정의
- ✅ AI 코드의 S3 업로드·다운로드·삭제 구현
- ✅ 개발 환경의 로컬 업로드 fallback 유지
- ⏳ BE가 S3 presigned URL 또는 S3 upload 계약을 사용하는 방식으로 전환
- ⏳ `uploads/v1/` Lifecycle 7일 정책 확인

CSV 전용 버킷은 별도로 만들지 않는다. 다음 prefix를 하나의 버킷에서 사용한다.

```text
models/v1/
rag/v1/
data/
uploads/v1/
feedback/v1/
bandit/v1/
logs/v2/
```

## 4.3 모델·RAG 자산

현재 자산은 Hugging Face에서 내려받는다.

```text
ai_sales_model.pkl
cox_risk.pkl
rag_index/export/*
data/processed/*
data/agent/*
```

운영에서는 태스크가 재시작될 때마다 대규모 모델을 다시 받지 않도록 다음 중 하나를 선택한다.

1. 배포 이미지에 모델을 포함
2. 배포 전 초기화 Task로 S3에 다운로드
3. EFS에 모델과 RAG 자산 저장
4. 모델 전용 S3 버전 관리 후 태스크 시작 시 필요한 파일만 다운로드

현재 선택은 S3이다. 모델·RAG는 `models/v1/`, `rag/v1/`에 저장하고,
재학습 모델은 새 prefix로 교체할 수 있다. 비공개 Hugging Face 토큰은 이미지나
Git 저장소에 포함하지 않는다.

## 4.4 Redis 운영 전환

Redis 기술은 그대로 유지한다. 개발 환경에서는 현재 Docker Redis를 사용하고, Fargate 운영 환경에서는 ElastiCache for Redis 또는 Valkey endpoint를 사용한다.

```env
CELERY_BROKER_URL=redis://운영-redis-endpoint:6379/0
CELERY_RESULT_BACKEND=redis://운영-redis-endpoint:6379/1
```

---

## 5. 서비스별 개발 작업

## 5.1 FE

### 추가 파일

```text
Dockerfile
nginx.conf
.github/workflows/deploy.yml
```

### 작업 내용

- ⏳ `npm run typecheck`와 `npm run build`를 CI에서 실행
- ⏳ Vite build 시 `VITE_API_BASE_URL` 주입
- ⏳ Nginx에서 SPA fallback 설정
- ⏳ `/api` 요청을 BE ALB 또는 BE 서비스로 라우팅
- ⏳ FE 서비스용 ECS Task Definition 작성

## 5.2 BE

### 운영용 구성

개발용 `compose.yaml`은 유지하고 운영 배포 설정은 별도로 관리한다.

```text
deploy/compose.prod.yaml 또는 ECS Task Definition
```

### 작업 내용

- ⏳ Spring Boot 이미지를 ECR에 push
- ⏳ RDS MySQL 연결
- ⏳ `FASTAPI_BASE_URL`을 AI 내부 주소로 변경
- ⏳ S3 업로드 연동
- ✅ AI 상태 DB와 BE 업무 DB의 테이블·권한 책임 분리 설계
- ⏳ Gradle 테스트 및 이미지 보안 검사 추가

## 5.3 AI

### 운영 대상

```text
FastAPI API
Celery Worker
Celery Beat
```

### 작업 내용

- ✅ SQLite 저장소를 MySQL로 이전하는 코드 및 schema 작성
- ✅ S3 기반 업로드 처리
- ⏳ Redis endpoint 환경변수화 및 운영 ElastiCache 연결
- ✅ 모델·RAG 자산 저장소 외부화
- ⏳ FastAPI와 Worker의 공통 설정 분리
- ⏳ API와 Worker의 CPU·메모리 요구량 별도 측정
- ⏳ AI 이미지와 Worker 이미지를 분리하거나 동일 이미지에 다른 command 적용

---

## 6. CI/CD 계획

저장소별로 독립적인 GitHub Actions를 구성한다.

```text
FE push
→ typecheck
→ build
→ Docker image build
→ ECR push
→ ECS FE service deployment

BE push
→ Gradle test
→ Docker image build
→ ECR push
→ ECS BE service deployment

AI push
→ Python test
→ Docker image build
→ ECR push
→ ECS AI API/Worker deployment
```

### 배포 원칙

- 이미지 태그는 `git sha`를 기본으로 사용
- `latest`만으로 운영 배포하지 않음
- ECS Task Definition revision을 생성한 뒤 Service 업데이트
- 배포 후 ALB health check 확인
- 실패 시 직전 Task Definition으로 rollback
- FE·BE·AI를 독립적으로 배포

### 비밀값

다음 값은 GitHub 저장소나 Docker 이미지에 넣지 않는다.

- `OPENAI_API_KEY`
- `HF_TOKEN`
- `MYSQL_PASSWORD`
- `JWT_SECRET`
- 외부 API 키

ECS Task Definition에서 Secrets Manager 또는 SSM Parameter Store 값을 주입한다.

---

## 7. 단계별 일정과 산출물

### 1단계: 운영 요구사항 확정 ✅

- 목표 서비스 수 결정
- AI 동시 요청량 측정
- 모델·RAG 전체 용량 확인
- 데이터 보존 기간 결정
- RDS·ElastiCache·S3 사용 여부 결정

산출물:

- 운영 아키텍처 문서
- 서비스별 CPU·메모리 기준
- 데이터 보존 정책

### 2단계: 저장소 외부화 🔶 부분 완료

- ✅ SQLite 저장소를 MySQL로 이전하는 AI 코드·migration 작성
- ✅ 임시 CSV를 S3로 처리하는 AI 코드 작성
- ✅ 모델·RAG 저장소를 S3로 결정하고 동기화 코드 작성
- ⏳ Redis endpoint와 ElastiCache 연결
- ⏳ BE MySQL migration·S3 upload 계약 반영

산출물:

- DB migration
- S3 upload/download 모듈
- 운영 환경변수 목록

### 3단계: 운영 이미지 작성 ⏳ 대기

- FE Dockerfile 작성
- BE 운영 이미지 확인
- AI API·Worker 이미지 작성
- health check 추가
- 로컬 이미지 실행 검증

산출물:

- 서비스별 Docker image
- 로컬 운영 유사 Compose

### 4단계: AWS 인프라 구성 🔶 부분 완료

- VPC와 subnet 구성
- ECS Cluster 생성
- ECR repository 생성
- RDS 생성
- ElastiCache 생성
- ✅ S3 bucket 생성
- ✅ S3 모델·RAG·Dataset 업로드
- ✅ ECS IAM Role 생성
- ⏳ Secrets Manager 또는 SSM 파라미터 생성
- ALB와 target group 생성

산출물:

- ECS Task Definition
- ECS Service
- 네트워크 보안 규칙

### 5단계: CI/CD 구성 ⏳ 대기

- FE workflow
- BE workflow
- AI workflow
- 이미지 push
- ECS rolling deployment
- 실패 rollback

### 6단계: 통합 검증 ⏳ 대기

- FE 로그인
- 매출 CSV 업로드
- 분석 작업 queued/running/completed 확인
- 분석 결과 조회
- 추천 생성
- 추천 승인·수정·거절
- 효과 검증 실행
- 검증 결과 조회
- AI 피드백 반영
- 태스크 재시작 후 데이터 보존 확인

---

## 8. 완료 조건 및 현재 상태

- ⏳ FE·BE·AI가 각각 독립적으로 ECS 배포된다.
- ⏳ BE가 AI 내부 주소로 FastAPI를 호출한다.
- ⏳ FastAPI와 Celery Worker가 ElastiCache Redis를 사용한다.
- ✅ 운영 코드가 MySQL 기반으로 분석·추천 상태를 저장하도록 준비됨
- ✅ 업로드 CSV의 S3 저장·처리·삭제 코드 작성
- ⏳ Fargate 태스크 교체 후 전체 이력 보존 통합 검증
- ✅ 모델과 RAG 자산의 S3 경로 및 자동 다운로드 정의
- ✅ 비밀값을 코드·이미지에 넣지 않는 환경변수 구조
- ⏳ GitHub Actions로 FE·BE·AI 독립 배포
- ⏳ 실패한 배포 rollback

---

## 9. 주요 위험과 대응

| 위험 | 대응 |
|---|---|
| Fargate 태스크 교체로 SQLite 유실 | MySQL 기반 AI 상태 저장소로 이전 |
| 모델 초기화 시간이 너무 김 | S3/EFS 캐시 또는 이미지 포함 |
| AI 메모리 부족 | API·Worker task size 분리 및 부하 테스트 |
| CSV 동시 업로드로 디스크 부족 | S3 직접 업로드 |
| Redis 장애로 작업 유실 | ElastiCache와 재시도 정책 사용 |
| BE와 AI API 계약 불일치 | OpenAPI/DTO 계약 테스트 추가 |
| 비밀값 노출 | Secrets Manager/SSM 사용 |
| 배포 후 이전 버전 rollback 불가 | immutable image tag와 Task Definition revision 사용 |

---

## 10. 현재 바로 할 일

1. BE가 `deploy/mysql/ai_schema.sql`을 RDS MySQL에 실행
2. BE `compose.yaml`과 ECS 환경변수에 `AI_DATABASE_URL`·S3 설정 반영
3. BE의 CSV 업로드를 S3 `uploads/v1/`와 연결
4. Redis를 ElastiCache endpoint로 연결
5. FE Dockerfile·Nginx 작성
6. FE·BE·AI ECR repository와 운영 이미지 작성
7. ECS Task Definition 및 Service 작성
8. GitHub Actions workflow 작성
9. ECS에서 매출 분석→추천→효과 검증→피드백 통합 테스트

개발용 Docker Compose는 위 작업이 끝날 때까지 그대로 유지한다.

## 11. 사용자와 개발 작업의 분담

### 11.1 사용자가 AWS에서 먼저 해야 하는 작업

AWS 계정의 리소스와 결제에 연결되는 작업은 사용자가 직접 생성한다.

- RDS MySQL 인스턴스와 AI 전용 데이터베이스 또는 스키마 생성
- ElastiCache Redis/Valkey 클러스터 생성
- S3 버킷 생성 및 Block Public Access 유지
- ECS Task가 사용할 IAM Role과 Secrets Manager/SSM 파라미터 생성
- ECR 저장소 생성
- VPC, private subnet, security group, ALB 구성

S3에 모델을 올리는 것도 사용자가 수행한다. 단순 파일 업로드만 하면 되는 것이 아니라,
버킷 이름·리전·object key·버전 식별자를 정해야 애플리케이션이 같은 자산을 재현할 수 있다.

권장 object key 구조:

```text
models/{model_version}/ai_sales_model.pkl
models/{model_version}/cox_risk.pkl
rag/{rag_version}/export/embeddings.npy
rag/{rag_version}/export/chunks.jsonl
rag/{rag_version}/export/manifest.json
```

### 11.2 사용자가 준비해 코드에 전달할 값

값 자체를 Git에 커밋하지 말고, 이름과 위치만 개발자에게 전달한다.

```text
AWS_REGION
S3_BUCKET_NAME
MODEL_S3_PREFIX
RAG_S3_PREFIX
AI_DATABASE_URL
CELERY_BROKER_URL
CELERY_RESULT_BACKEND
```

`OPENAI_API_KEY`, `HF_TOKEN`, MySQL 비밀번호, Redis 인증정보는 값이 아니라
Secrets Manager 또는 SSM Parameter Store의 ARN/parameter name으로 ECS에 주입한다.

### 11.3 제가 코드로 진행할 작업

1. ✅ AI 분석 작업과 분석 결과의 MySQL 전환 코드
2. ✅ LangGraph `SqliteSaver`의 MySQL Checkpointer 전환
3. ✅ S3 업로드·다운로드 모듈과 로컬 개발 fallback
4. ✅ 분석 완료 후 임시 CSV S3 삭제 구조
5. 🔶 AI API/Worker 운영 환경변수와 health check 정리
6. ⏳ BE의 AI 상태 테이블 migration 및 S3 upload 계약 반영
7. ⏳ FE·BE·AI Dockerfile, ECR, ECS Task Definition, CI/CD 작성

사용자가 버킷을 만들기 전에도 1·2·5번은 로컬 MySQL/Redis로 개발할 수 있다.
S3 관련 코드는 버킷 정보가 전달되면 통합 테스트를 진행한다.
