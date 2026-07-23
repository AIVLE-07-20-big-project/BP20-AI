# 캠페인 로그 수집 + 데이터 계약 검증(계획 §1단계)
from __future__ import annotations

import contextlib
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import CAMPAIGN_LOGS
from app.services.response import action_rules, bandit_store
from app.services.response.context import CONTEXT_DIM
from app.services.response.graph import get_graph

from scripts.modeling.sales_analysis import AMT, PANEL

CONTEXT_COLS = [f"context_{i}" for i in range(1, 7)]
SCHEMA_COLUMNS = [
    "decision_id", "user_id", "store_id", "trdar_cd", "svc_induty_cd", "yyqu_cd",
    "treatment_yyqu_cd", "action_id",
    *CONTEXT_COLS, "propensity", "policy_version", "executed",
    "revenue_before", "revenue_after", "reward", "데이터_출처",
]

_logger = logging.getLogger(__name__)


# 해당 thread_id의 agent-run을 찾을 수 없음
class DecisionNotFound(Exception):
    pass


# thread_id가 승인 완료 상태가 아니라 campaign-logs를 기록할 수 없음
class DecisionNotApproved(Exception):
    pass


# 요청 사용자와 결정 시점의 소유자가 다름
class DecisionOwnershipMismatch(Exception):
    pass


# 새 의존성 없이(stdlib만으로) read-modify-write를 직렬화한다
@contextlib.contextmanager
def _file_lock(path: Path, timeout: float = 10.0, poll: float = 0.05):





    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"campaign_logs 잠금 획득 실패: {lock_path}")
            time.sleep(poll)
    try:
        yield
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)


def _lookup_revenue(trdar_cd, svc_induty_cd, yyqu_cd, panel=PANEL) -> float | None:
    if yyqu_cd is None:
        return None
    p = pd.read_csv(panel, usecols=["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD", AMT])
    row = p[
        (p["TRDAR_CD"].astype(str) == str(trdar_cd))
        & (p["SVC_INDUTY_CD"] == svc_induty_cd)
        & (p["STDR_YYQU_CD"] == int(yyqu_cd))
    ]
    return float(row.iloc[0][AMT]) if not row.empty else None


def _append_row_atomic(row: dict, campaign_logs: Path) -> None:
    campaign_logs.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row], columns=SCHEMA_COLUMNS)
    with _file_lock(campaign_logs):
        if campaign_logs.exists():
            existing = pd.read_csv(campaign_logs)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row
        tmp_path = campaign_logs.with_suffix(".tmp")
        combined.to_csv(tmp_path, index=False)
        tmp_path.replace(campaign_logs)


# thread_id(승인된 agent-run)의 체크포인트에서 결정 시점 값을 읽어 한 행을 기록한다
def append_log(thread_id: str, executed: bool, treatment_yyqu_cd: int,
                revenue_after: float | None, campaign_logs: Path | None = None,
                user_id: str | None = None) -> dict:








    campaign_logs = Path(campaign_logs) if campaign_logs is not None else CAMPAIGN_LOGS
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = get_graph().get_state(config)
    if not snapshot.values:
        raise DecisionNotFound(f"agent-run을 찾을 수 없음: {thread_id}")

    state = snapshot.values
    owner = state.get("user_id")
    if owner is not None and owner != user_id:
        raise DecisionOwnershipMismatch("해당 추천 실행 결과를 기록할 권한이 없습니다")
    if state.get("approval_status") != "approved":
        raise DecisionNotApproved(
            f"승인되지 않은 결정입니다(approval_status={state.get('approval_status')})"
        )

    trdar_cd = state["trdar_cd"]
    svc_induty_cd = state["svc_induty_cd"]


    yyqu_cd = state.get("yyqu_cd") or (state.get("diagnosis") or {}).get("대상", {}).get("기준분기")
    action_id = state["selected_action"]["방안"]
    context_vector = state.get("context_vector") or [None] * len(CONTEXT_COLS)
    bandit_result = state.get("bandit_result") or {}
    propensity = (bandit_result.get("arm별_propensity") or {}).get(action_id)
    policy_version = state.get("policy_version")

    revenue_before = _lookup_revenue(trdar_cd, svc_induty_cd, yyqu_cd)
    reward = None
    if executed and revenue_after is not None and revenue_before not in (None, 0):
        reward = (revenue_after - revenue_before) / revenue_before

    row = {
        "decision_id": str(uuid.uuid4()),
        "user_id": state.get("user_id"), "store_id": state.get("store_id"),
        "trdar_cd": trdar_cd, "svc_induty_cd": svc_induty_cd, "yyqu_cd": yyqu_cd,
        "treatment_yyqu_cd": treatment_yyqu_cd, "action_id": action_id,
        **{col: context_vector[i] for i, col in enumerate(CONTEXT_COLS)},
        "propensity": propensity, "policy_version": policy_version, "executed": executed,
        "revenue_before": revenue_before, "revenue_after": revenue_after, "reward": reward,
        "데이터_출처": "real",
    }
    _append_row_atomic(row, Path(campaign_logs))
    _update_bandit_online(state, action_id, context_vector, reward)
    return row


# 실측 reward가 나온 건은 즉시 해당 등급 active 모델의 A/b를 갱신한다(저비용
def _update_bandit_online(state: dict, action_id: str, context_vector: list,
                           reward: float | None) -> None:




    if reward is None or any(v is None for v in context_vector):
        return
    등급 = state.get("문제유형")
    arms = [c["방안"] for c in state.get("candidate_actions") or []]
    if not 등급 or action_id not in arms:
        return
    try:
        bandit, _ = bandit_store.load_or_coldstart(등급, context_dim=CONTEXT_DIM, arms=arms)
        bandit.update(np.asarray(context_vector, dtype=float), arms.index(action_id), reward)
        bandit_store.save(등급, bandit)
    except Exception:
        _logger.warning("Bandit 온라인 update 실패(등급=%s, action_id=%s)", 등급, action_id, exc_info=True)


# 스키마·타입·중복·propensity 범위·reward 재계산 일치 여부를 검사해 유효/제외 행을
def validate_logs(campaign_logs: Path | None = None) -> dict:




    path = Path(campaign_logs) if campaign_logs is not None else CAMPAIGN_LOGS
    if not path.exists():
        return {"총행수": 0, "유효행수": 0, "제외행수": 0, "제외사유": {}}

    logs = pd.read_csv(path)
    missing_cols = [c for c in SCHEMA_COLUMNS if c not in logs.columns]
    if missing_cols:
        return {"총행수": int(len(logs)), "유효행수": 0, "제외행수": int(len(logs)),
                "제외사유": {"스키마 컬럼 누락": missing_cols}}

    recompute = (logs["revenue_after"] - logs["revenue_before"]) / logs["revenue_before"].replace(0, np.nan)
    reward_mismatch = (
        logs["executed"].astype(bool) & logs["reward"].notna()
        & ((recompute - logs["reward"]).abs() > 1e-6)
    )
    checks = [
        ("decision_id 중복", logs["decision_id"].duplicated(keep="first")),
        ("알 수 없는 action_id", ~logs["action_id"].isin(action_rules.ACTIONS.keys())),
        ("propensity 범위(0,1] 벗어남", ~logs["propensity"].between(0, 1, inclusive="right")),
        ("treatment_yyqu_cd가 yyqu_cd 이후가 아님", logs["treatment_yyqu_cd"] <= logs["yyqu_cd"]),
        ("executed=True인데 reward 없음", logs["executed"].astype(bool) & logs["reward"].isna()),
        ("reward 재계산 불일치", reward_mismatch),
    ]

    valid_mask = pd.Series(True, index=logs.index)
    excluded_counts: dict[str, int] = {}
    for reason, bad_mask in checks:
        newly_excluded = bad_mask & valid_mask
        count = int(newly_excluded.sum())
        if count:
            excluded_counts[reason] = count
        valid_mask &= ~bad_mask

    valid_count = int(valid_mask.sum())
    result = {
        "총행수": int(len(logs)), "유효행수": valid_count,
        "제외행수": int(len(logs) - valid_count), "제외사유": excluded_counts,
    }
    if "데이터_출처" in logs.columns:
        result["합성_행수"] = int((logs["데이터_출처"] == "synthetic").sum())
        result["실제_행수"] = int((logs["데이터_출처"] == "real").sum())
    return result
