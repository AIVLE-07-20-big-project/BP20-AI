# 대상 경로(AI 레포): app/sales_target/cleanup.py
#
# AI_Agent_전환_가이드라인.md 4단계("운영 정리 정책") 중 "체크포인터에 쌓이는 상태 데이터 크기
# 관리" 담당. 완료(승인/finalize)되거나 반려(discard)된 지 오래된 thread의 체크포인트 데이터를
# 지운다. "N일 경과 시 자동 반려"는 BE 쪽(SalesTargetBatchService.autoRejectStaleBatches())이
# 담당한다 — BE가 이미 각 배치의 시작 시각(createdAt)을 SalesTargetBatchRun으로 추적하고 있고,
# 기존 /jobs/{thread_id}/reject 엔드포인트를 그대로 호출하면 되므로 AI 쪽에 새 엔드포인트가
# 필요 없다. 이 모듈은 그와 별개로, "이미 끝난 thread를 얼마나 오래 보관할 것인가"만 다룬다.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

# graph.py의 _finalize/_discard/_prepare_candidates(빈 후보) 노드가 남기는 status 값은 전부
# "완료" 또는 "종료"로 시작한다(graph.py 참고). 그 외(예: "스코어링 완료 — 관리자 승인 대기",
# "승인됨 — 피칭 문구 생성 중")는 아직 진행 중이거나 승인 대기 중이므로 절대 지우면 안 된다.
_TERMINAL_STATUS_PREFIXES = ("완료", "종료")


def _is_terminal_status(status: str | None) -> bool:
    return bool(status) and status.startswith(_TERMINAL_STATUS_PREFIXES)


def _latest_checkpoint_per_thread(checkpointer: BaseCheckpointSaver) -> dict[str, CheckpointTuple]:
    """checkpointer.list(None)은 저장된 모든 체크포인트(스텝별로 여러 개)를 thread 구분 없이
    전부 순회한다. thread_id별로 가장 최근 것(체크포인트의 ts 문자열 기준, ISO8601이라 문자열
    비교로도 최신순 정렬이 성립한다) 하나만 남긴다 — 그게 그 thread의 "현재 상태"다."""
    latest: dict[str, CheckpointTuple] = {}
    for tup in checkpointer.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        ts = tup.checkpoint.get("ts", "")
        current = latest.get(thread_id)
        if current is None or ts > current.checkpoint.get("ts", ""):
            latest[thread_id] = tup
    return latest


def find_deletable_threads(
    checkpointer: BaseCheckpointSaver,
    older_than_days: int,
    now: datetime | None = None,
) -> list[str]:
    """삭제 대상 thread_id 목록을 반환한다(실제 삭제는 안 함 — delete_old_terminal_threads가 함).

    조건: 최신 체크포인트의 status가 완료/종료 계열(진행 중·승인 대기 아님)이고, 그 체크포인트가
    기록된 시각이 now - older_than_days보다 오래됐을 것.
    """
    if older_than_days < 0:
        raise ValueError("older_than_days는 0 이상이어야 합니다.")

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=older_than_days)

    deletable = []
    for thread_id, tup in _latest_checkpoint_per_thread(checkpointer).items():
        values = tup.checkpoint.get("channel_values") or {}
        if not _is_terminal_status(values.get("status")):
            continue

        ts_str = tup.checkpoint.get("ts")
        if not ts_str:
            continue
        ts = datetime.fromisoformat(ts_str)
        if ts < cutoff:
            deletable.append(thread_id)

    return deletable


def delete_old_terminal_threads(
    checkpointer: BaseCheckpointSaver,
    older_than_days: int,
    now: datetime | None = None,
) -> list[str]:
    """완료/반려된 지 older_than_days일이 지난 thread의 체크포인트 데이터를 전부 지운다.

    반환값은 실제로 지운 thread_id 목록(운영 로그/Celery 태스크 반환값으로 그대로 쓸 수 있게).
    """
    thread_ids = find_deletable_threads(checkpointer, older_than_days, now)
    for thread_id in thread_ids:
        checkpointer.delete_thread(thread_id)
    return thread_ids
