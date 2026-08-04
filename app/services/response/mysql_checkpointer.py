"""MySQL-backed LangGraph checkpointer.

LangGraph의 기본 SqliteSaver와 동일한 체크포인트/중간 write 계약을
운영 MySQL에 저장한다. 연결은 매 호출마다 열어 Fargate worker/API가
서로 다른 프로세스에서 안전하게 사용할 수 있도록 한다.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from app.core.database import connect
from app.core.config import AGENT_RUNS_DB


class MySQLCheckpointer(BaseCheckpointSaver[str]):
    """LangGraph checkpoint를 MySQL에 저장하는 동기 saver."""

    def setup(self) -> None:
        with connect(str(AGENT_RUNS_DB)) as connection:
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_agent_checkpoints (
                    thread_id VARCHAR(191) NOT NULL,
                    checkpoint_ns VARCHAR(191) NOT NULL,
                    checkpoint_id VARCHAR(191) NOT NULL,
                    parent_checkpoint_id VARCHAR(191),
                    type VARCHAR(255) NOT NULL,
                    checkpoint LONGBLOB NOT NULL,
                    metadata LONGTEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_agent_writes (
                    thread_id VARCHAR(191) NOT NULL,
                    checkpoint_ns VARCHAR(191) NOT NULL,
                    checkpoint_id VARCHAR(191) NOT NULL,
                    task_id VARCHAR(191) NOT NULL,
                    idx INT NOT NULL,
                    channel VARCHAR(191) NOT NULL,
                    type VARCHAR(191) NOT NULL,
                    value LONGBLOB NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            connection.commit()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        with connect(str(AGENT_RUNS_DB)) as connection:
            cursor = connection.cursor()
            if checkpoint_id:
                cursor.execute(
                    """SELECT thread_id, checkpoint_id, parent_checkpoint_id, type,
                       checkpoint, metadata FROM ai_agent_checkpoints
                       WHERE thread_id=%s AND checkpoint_ns=%s AND checkpoint_id=%s""",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
            else:
                cursor.execute(
                    """SELECT thread_id, checkpoint_id, parent_checkpoint_id, type,
                       checkpoint, metadata FROM ai_agent_checkpoints
                       WHERE thread_id=%s AND checkpoint_ns=%s
                       ORDER BY checkpoint_id DESC LIMIT 1""",
                    (thread_id, checkpoint_ns),
                )
            row = cursor.fetchone()
            if row is None:
                return None

            current_id = row["checkpoint_id"]
            result_config = config
            if not checkpoint_id:
                result_config = {
                    "configurable": {
                        "thread_id": row["thread_id"],
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": current_id,
                    }
                }
            cursor.execute(
                """SELECT task_id, channel, type, value FROM ai_agent_writes
                   WHERE thread_id=%s AND checkpoint_ns=%s AND checkpoint_id=%s
                   ORDER BY task_id, idx""",
                (thread_id, checkpoint_ns, current_id),
            )
            writes = cursor.fetchall()
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            return CheckpointTuple(
                result_config,
                self.serde.loads_typed((row["type"], row["checkpoint"])),
                cast(CheckpointMetadata, metadata),
                ({
                    "configurable": {
                        "thread_id": row["thread_id"],
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row["parent_checkpoint_id"],
                    }
                } if row["parent_checkpoint_id"] else None),
                [
                    (item["task_id"], item["channel"], self.serde.loads_typed((item["type"], item["value"])))
                    for item in writes
                ],
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config["configurable"]
        type_, serialized = self.serde.dumps_typed(checkpoint)
        metadata_json = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        )
        with connect(str(AGENT_RUNS_DB)) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO ai_agent_checkpoints
                   (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                    type, checkpoint, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE parent_checkpoint_id=VALUES(parent_checkpoint_id),
                   type=VALUES(type), checkpoint=VALUES(checkpoint), metadata=VALUES(metadata)""",
                (
                    str(configurable["thread_id"]),
                    configurable.get("checkpoint_ns", ""),
                    checkpoint["id"],
                    configurable.get("checkpoint_id"),
                    type_, serialized,
                    metadata_json,
                ),
            )
            connection.commit()
        return {
            "configurable": {
                "thread_id": configurable["thread_id"],
                "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return
        configurable = config["configurable"]
        with connect(str(AGENT_RUNS_DB)) as connection:
            cursor = connection.cursor()
            for idx, (channel, value) in enumerate(writes):
                type_, serialized = self.serde.dumps_typed(value)
                write_idx = WRITES_IDX_MAP.get(channel, idx)
                cursor.execute(
                    """INSERT IGNORE INTO ai_agent_writes
                       (thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                        channel, type, value)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        str(configurable["thread_id"]),
                        configurable.get("checkpoint_ns", ""),
                        str(configurable["checkpoint_id"]),
                        task_id,
                        write_idx,
                        channel,
                        type_,
                        serialized,
                    ),
                )
            connection.commit()
