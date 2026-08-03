from __future__ import annotations

import csv
from datetime import datetime, timezone

from app.core.config import AGENT_DATA
from app.services.response.campaign_logs import _update_bandit_online
from app.services.response.graph import get_graph

SALES_FEEDBACK_LOG = AGENT_DATA / "ai_sales_feedback.csv"


def record_sales_feedback(
    thread_id: str,
    before_sales: float,
    after_sales: float,
    measurement_days: int,
    user_id: str | None,
) -> dict:
    snapshot = get_graph().get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise LookupError(f"agent-run을 찾을 수 없음: {thread_id}")
    state = snapshot.values
    owner = state.get("user_id")
    if owner is not None and str(owner) != str(user_id):
        raise PermissionError("해당 추천 실행 결과를 기록할 권한이 없습니다")
    if state.get("approval_status") not in ("approved", "edited"):
        raise RuntimeError("승인된 추천만 학습할 수 있습니다")

    action = state.get("selected_action") or {}
    action_id = action.get("방안")
    context = state.get("context_vector")
    baseline = max(abs(before_sales), 1.0)
    reward = (after_sales - before_sales) / baseline if before_sales else 0.0
    if action_id and context:
        _update_bandit_online(state, action_id, context, reward)

    row = {
        "thread_id": thread_id,
        "user_id": user_id,
        "selected_action": action_id,
        "before_sales": before_sales,
        "after_sales": after_sales,
        "change_rate": reward * 100.0 if before_sales else 0.0,
        "measurement_days": measurement_days,
        "reward": reward,
        "reward_definition_version": "revenue_lift_v1",
        "learned_at": datetime.now(timezone.utc).isoformat(),
    }
    SALES_FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    header = not SALES_FEEDBACK_LOG.exists()
    with SALES_FEEDBACK_LOG.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if header:
            writer.writeheader()
        writer.writerow(row)
    return row
