-- BP20 AI 운영 저장소
-- BE and AI share the analysis table in MySQL.

CREATE TABLE IF NOT EXISTS ai_analyses (
    analysis_id VARCHAR(36) PRIMARY KEY,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    result_json LONGTEXT NOT NULL,
    store_id BIGINT NULL,
    svc_induty_cd VARCHAR(30) NOT NULL,
    trdar_cd VARCHAR(30) NOT NULL,
    user_id BIGINT NOT NULL,
    yyqu_cd INT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ai_analysis_jobs (
    job_id VARCHAR(64) PRIMARY KEY,
    celery_task_id VARCHAR(255),
    user_id VARCHAR(255),
    status VARCHAR(32) NOT NULL,
    analysis_id VARCHAR(36),
    error_code VARCHAR(64),
    error_message TEXT,
    created_at VARCHAR(40) NOT NULL,
    started_at VARCHAR(40),
    completed_at VARCHAR(40)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

ALTER TABLE ai_analysis_jobs
    ADD CONSTRAINT fk_ai_jobs_analysis
    FOREIGN KEY (analysis_id) REFERENCES ai_analyses (analysis_id);

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
