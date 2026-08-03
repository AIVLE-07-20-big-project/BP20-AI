-- BP20 AI 운영 저장소
-- BE가 사용하는 MySQL에 실행하되, AI 전용 테이블만 생성한다.

CREATE TABLE IF NOT EXISTS ai_analysis_jobs (
    job_id VARCHAR(64) PRIMARY KEY,
    celery_task_id VARCHAR(255),
    user_id VARCHAR(255),
    status VARCHAR(32) NOT NULL,
    analysis_id VARCHAR(64),
    error_code VARCHAR(64),
    error_message TEXT,
    created_at VARCHAR(40) NOT NULL,
    started_at VARCHAR(40),
    completed_at VARCHAR(40)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ai_internal_analyses (
    analysis_id VARCHAR(64) PRIMARY KEY,
    trdar_cd VARCHAR(64) NOT NULL,
    svc_induty_cd VARCHAR(64) NOT NULL,
    yyqu_cd INT,
    report_json LONGTEXT NOT NULL,
    diagnosis_json LONGTEXT NOT NULL,
    warnings_json LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    user_id VARCHAR(255),
    store_id VARCHAR(255),
    detailed_analysis_json LONGTEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ai_agent_checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_ns VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    parent_checkpoint_id VARCHAR(255),
    type VARCHAR(255) NOT NULL,
    checkpoint LONGBLOB NOT NULL,
    metadata LONGTEXT,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ai_agent_writes (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_ns VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    idx INT NOT NULL,
    channel VARCHAR(255) NOT NULL,
    type VARCHAR(255) NOT NULL,
    value LONGBLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
