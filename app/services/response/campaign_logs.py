# 캠페인 로그 v2 수집 + 데이터 계약 검증(계획 §2·4단계, 설계 §9)
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import CAMPAIGN_LOGS_S3_PREFIX, CAMPAIGN_LOGS_V2
from app.core.object_storage import (
    download_file,
    enabled as s3_enabled,
    upload_file,
)
from app.services.response import action_rules, bandit_store
from app.services.response.context import CONTEXT_DIM
from app.services.response.graph import get_graph
from app.services.response.reward import (
    DEFAULT_REWARD_DENOMINATOR_FLOOR,
    REWARD_DEFINITION_VERSION,
    calculate_reward_v2,
)

from scripts.modeling.sales_analysis import AMT, PANEL

CONTEXT_COLS = [f"context_{i}" for i in range(1, CONTEXT_DIM + 1)]
SCHEMA_COLUMNS = [
    "decision_id", "user_id", "store_id", "trdar_cd", "svc_induty_cd", "yyqu_cd",
    "treatment_yyqu_cd", "problem_type", "context_schema_version",
    *CONTEXT_COLS,
    "candidate_actions", "eligible_actions", "unknown_actions", "blocked_actions",
    "selectable_actions", "exploration_excluded_actions",
    "policy_mode", "action_probabilities",
    "recommended_action", "policy_selected_action", "policy_selected_propensity",
    "approved_action", "executed_action", "behavior_propensity", "decision_source",
    "selection_source", "expected_rewards", "uncertainties",
    "net_sales_before", "net_sales_after", "variable_cost_before", "variable_cost_after",
    "campaign_cost", "measurement_days",
    "execution_started_at", "execution_ended_at",
    "measurement_started_at", "measurement_ended_at",
    "baseline_period_start", "baseline_period_end", "control_store_ids",
    "reward_definition_version", "reward_denominator_floor", "reward_status", "reward",
    "policy_version", "model_version", "model_sha256",
    "ope_eligible", "training_eligible", "데이터_출처", "logged_at",
]

_logger = logging.getLogger(__name__)

PERIOD_COLUMNS = {
    "execution_started_at", "execution_ended_at", "measurement_started_at",
    "measurement_ended_at", "baseline_period_start", "baseline_period_end",
    "control_store_ids",
}
REQUIRED_SCHEMA_COLUMNS = [column for column in SCHEMA_COLUMNS if column not in PERIOD_COLUMNS]


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


def ensure_campaign_logs_snapshot(campaign_logs: Path | None = None) -> Path:
    """S3 snapshot이 있으면 로컬 분석 경로로 복구한다."""
    path = Path(campaign_logs) if campaign_logs is not None else CAMPAIGN_LOGS_V2
    if s3_enabled() and not path.exists():
        try:
            download_file(
                prefix=CAMPAIGN_LOGS_S3_PREFIX,
                name="campaign_logs_v2.csv",
                local_path=path,
            )
        except Exception:
            # 아직 로그가 없는 첫 배포에서는 빈 상태로 시작할 수 있다.
            pass
    return path


def _append_row_atomic(row: dict, campaign_logs: Path) -> None:
    ensure_campaign_logs_snapshot(campaign_logs)
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
    if s3_enabled() and campaign_logs == CAMPAIGN_LOGS_V2:
        upload_file(
            campaign_logs,
            prefix=CAMPAIGN_LOGS_S3_PREFIX,
            name="campaign_logs_v2.csv",
        )


def _actions_by_status(candidate_status: dict[str, str], status: str) -> list[str]:
    return [action for action, s in candidate_status.items() if s == status]


def _none_if_nan(value):
    return None if pd.isna(value) else value


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _validate_periods(
    execution_started_at: datetime | None,
    execution_ended_at: datetime | None,
    measurement_started_at: date | None,
    measurement_ended_at: date | None,
    baseline_period_start: date | None,
    baseline_period_end: date | None,
) -> None:
    if execution_started_at and execution_ended_at and execution_ended_at < execution_started_at:
        raise ValueError("execution_ended_at은 execution_started_at 이후여야 합니다")
    if measurement_started_at and measurement_ended_at and measurement_ended_at < measurement_started_at:
        raise ValueError("measurement_ended_at은 measurement_started_at 이후여야 합니다")
    if baseline_period_start and baseline_period_end and baseline_period_end < baseline_period_start:
        raise ValueError("baseline_period_end는 baseline_period_start 이후여야 합니다")


# 저장된 reward를 net_sales/variable_cost/campaign_cost로 재계산해 일치하는지 검증
def _reward_recompute_mismatch(row: pd.Series) -> bool:
    if row["reward_status"] != "complete":
        return False
    try:
        result = calculate_reward_v2(
            net_sales_before=_none_if_nan(row["net_sales_before"]),
            net_sales_after=_none_if_nan(row["net_sales_after"]),
            variable_cost_before=_none_if_nan(row["variable_cost_before"]),
            variable_cost_after=_none_if_nan(row["variable_cost_after"]),
            campaign_cost=_none_if_nan(row["campaign_cost"]),
            measurement_days_before=_none_if_nan(row["measurement_days"]),
            measurement_days_after=_none_if_nan(row["measurement_days"]),
            denominator_floor=_none_if_nan(row["reward_denominator_floor"]) or DEFAULT_REWARD_DENOMINATOR_FLOOR,
        )
    except (ValueError, TypeError):
        return True
    if result.status != "complete" or pd.isna(row["reward"]):
        return True
    return abs(result.reward - row["reward"]) > 1e-6


# 승인된 agent-run의 체크포인트를 읽어 로그 v2 한 행을 기록한다.
# 비용 데이터가 없으면 reward_status=incomplete로 기록되고 온라인 학습에서 제외된다.
def append_log(
    thread_id: str, executed: bool, treatment_yyqu_cd: int,
    net_sales_after: float | None = None,
    variable_cost_before: float | None = None, variable_cost_after: float | None = None,
    campaign_cost: float | None = None, measurement_days: int | None = None,
    execution_started_at: datetime | None = None, execution_ended_at: datetime | None = None,
    measurement_started_at: date | None = None, measurement_ended_at: date | None = None,
    baseline_period_start: date | None = None, baseline_period_end: date | None = None,
    control_store_ids: list[str] | None = None,
    campaign_logs: Path | None = None, user_id: str | None = None,
) -> dict:
    _validate_periods(
        execution_started_at, execution_ended_at, measurement_started_at,
        measurement_ended_at, baseline_period_start, baseline_period_end,
    )
    campaign_logs = Path(campaign_logs) if campaign_logs is not None else CAMPAIGN_LOGS_V2
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = get_graph().get_state(config)
    if not snapshot.values:
        raise DecisionNotFound(f"agent-run을 찾을 수 없음: {thread_id}")

    state = snapshot.values
    owner = state.get("user_id")
    if owner is not None and user_id is not None and str(owner) != str(user_id):
        raise DecisionOwnershipMismatch("해당 추천 실행 결과를 기록할 권한이 없습니다")
    if state.get("approval_status") not in ("approved", "edited"):
        raise DecisionNotApproved(
            f"승인되지 않은 결정입니다(approval_status={state.get('approval_status')})"
        )

    trdar_cd = state["trdar_cd"]
    svc_induty_cd = state["svc_induty_cd"]
    yyqu_cd = state.get("yyqu_cd") or (state.get("diagnosis") or {}).get("대상", {}).get("기준분기")
    action_id = state["selected_action"]["방안"]
    context_vector = state.get("context_vector") or [None] * len(CONTEXT_COLS)

    decision_source = state.get("decision_source") or "policy"
    selection_source = state.get("selection_source") or "policy"
    policy_decision = state.get("policy_decision") or {}
    policy_selected_propensity = policy_decision.get("policy_selected_propensity")
    behavior_propensity = policy_selected_propensity if decision_source == "policy" else None

    candidate_status = state.get("candidate_status") or {}
    candidate_scores = state.get("candidate_scores") or {}

    net_sales_before = _lookup_revenue(trdar_cd, svc_induty_cd, yyqu_cd)
    reward_result = None
    if executed:
        reward_result = calculate_reward_v2(
            net_sales_before=net_sales_before, net_sales_after=net_sales_after,
            variable_cost_before=variable_cost_before, variable_cost_after=variable_cost_after,
            campaign_cost=campaign_cost,
            measurement_days_before=measurement_days, measurement_days_after=measurement_days,
        )
    reward = reward_result.reward if reward_result else None
    reward_status = reward_result.status if reward_result else "not_executed"

    row = {
        "decision_id": str(uuid.uuid4()),
        "user_id": state.get("user_id"), "store_id": state.get("store_id"),
        "trdar_cd": trdar_cd, "svc_induty_cd": svc_induty_cd, "yyqu_cd": yyqu_cd,
        "treatment_yyqu_cd": treatment_yyqu_cd,
        "problem_type": state.get("문제유형"),
        "context_schema_version": state.get("context_schema_version"),
        **{col: context_vector[i] for i, col in enumerate(CONTEXT_COLS)},
        "candidate_actions": json.dumps(
            [c["방안"] for c in state.get("candidate_actions") or []], ensure_ascii=False,
        ),
        "eligible_actions": json.dumps(_actions_by_status(candidate_status, "eligible"), ensure_ascii=False),
        "unknown_actions": json.dumps(_actions_by_status(candidate_status, "unknown"), ensure_ascii=False),
        "blocked_actions": json.dumps(_actions_by_status(candidate_status, "blocked"), ensure_ascii=False),
        "selectable_actions": json.dumps(state.get("selectable_actions") or [], ensure_ascii=False),
        "exploration_excluded_actions": json.dumps(
            state.get("exploration_excluded_actions") or [], ensure_ascii=False,
        ),
        "policy_mode": policy_decision.get("policy_mode"),
        "action_probabilities": json.dumps(policy_decision.get("action_probabilities") or {}, ensure_ascii=False),
        "recommended_action": state.get("policy_selected_action"),
        "policy_selected_action": state.get("policy_selected_action"),
        "policy_selected_propensity": policy_selected_propensity,
        "approved_action": action_id,
        "executed_action": action_id if executed else None,
        "behavior_propensity": behavior_propensity,
        "decision_source": decision_source,
        "selection_source": selection_source,
        "expected_rewards": json.dumps(
            {a: s.get("expected_reward") for a, s in candidate_scores.items()}, ensure_ascii=False,
        ),
        "uncertainties": json.dumps(
            {a: s.get("uncertainty") for a, s in candidate_scores.items()}, ensure_ascii=False,
        ),
        "net_sales_before": net_sales_before, "net_sales_after": net_sales_after,
        "variable_cost_before": variable_cost_before, "variable_cost_after": variable_cost_after,
        "campaign_cost": campaign_cost, "measurement_days": measurement_days,
        "execution_started_at": _iso(execution_started_at),
        "execution_ended_at": _iso(execution_ended_at),
        "measurement_started_at": _iso(measurement_started_at),
        "measurement_ended_at": _iso(measurement_ended_at),
        "baseline_period_start": _iso(baseline_period_start),
        "baseline_period_end": _iso(baseline_period_end),
        "control_store_ids": json.dumps(control_store_ids or [], ensure_ascii=False),
        "reward_definition_version": REWARD_DEFINITION_VERSION,
        "reward_denominator_floor": DEFAULT_REWARD_DENOMINATOR_FLOOR,
        "reward_status": reward_status,
        "reward": reward,
        "policy_version": state.get("policy_version"),
        "model_version": state.get("model_version"),
        "model_sha256": state.get("model_sha256"),
        "ope_eligible": decision_source == "policy" and selection_source == "policy",
        "training_eligible": (
            decision_source == "policy" and selection_source == "policy" and reward_status == "complete"
        ),
        "데이터_출처": "real",
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_row_atomic(row, campaign_logs)
    if row["training_eligible"]:
        _update_bandit_online(state, action_id, context_vector, reward)
    return row


# 실측 reward가 나오면 즉시 해당 등급 active 모델의 A/b를 갱신한다(재학습은 retrain_cli가 별도 수행)
def _update_bandit_online(state: dict, action_id: str, context_vector: list,
                           reward: float | None) -> None:
    if reward is None or any(v is None for v in context_vector):
        return
    등급 = state.get("문제유형")
    if not 등급:
        return
    # state의 candidate_actions는 validate_candidates가 selectable로 좁혀놓은 목록이라,
    # 저장된 모델(등급 전체 arm으로 학습됨)을 불러올 때 그대로 쓰면 arm 수가 달라
    # BanditLoadMismatch로 콜드스타트되어 학습이 조용히 스킵된다. 항상 전체 arm으로 불러온다.
    arms = [c["방안"] for c in action_rules.candidate_actions(등급)]
    if action_id not in arms:
        return
    try:
        model_file = bandit_store.model_path(등급)
        # 동시 승인 시 load-update-save 경합으로 reward가 유실되지 않도록 직렬화한다
        with _file_lock(model_file):
            bandit, loaded = bandit_store.load_or_coldstart(등급, context_dim=CONTEXT_DIM, arms=arms)
            if not loaded and model_file.exists():
                # 저장된 모델과 arm 집합이 달라 콜드스타트된 경우 — 학습된 모델을 덮어쓰지 않는다
                _logger.warning(
                    "Bandit 온라인 update 건너뜀 — 저장된 모델과 arm 집합 불일치(등급=%s)", 등급,
                )
                return
            bandit.update(np.asarray(context_vector, dtype=float), arms.index(action_id), reward)
            bandit_store.save(등급, bandit)
    except Exception:
        _logger.warning("Bandit 온라인 update 실패(등급=%s, action_id=%s)", 등급, action_id, exc_info=True)


# 스키마·타입·중복·propensity 범위·reward 재계산 일치 여부를 검사해 유효/제외 행을 센다
def validate_logs(campaign_logs: Path | None = None) -> dict:
    path = ensure_campaign_logs_snapshot(campaign_logs)
    if not path.exists():
        return {"총행수": 0, "유효행수": 0, "제외행수": 0, "제외사유": {}}

    logs = pd.read_csv(path)
    # 기간 추적 필드는 기존 로그와의 하위 호환을 위해 누락되어도 기존 행을 무효화하지 않는다.
    missing_cols = [c for c in REQUIRED_SCHEMA_COLUMNS if c not in logs.columns]
    if missing_cols:
        return {"총행수": int(len(logs)), "유효행수": 0, "제외행수": int(len(logs)),
                "제외사유": {"스키마 컬럼 누락": missing_cols}}

    reward_mismatch = logs.apply(_reward_recompute_mismatch, axis=1)
    ope_eligible = logs["ope_eligible"].astype(bool)
    checks = [
        ("decision_id 중복", logs["decision_id"].duplicated(keep="first")),
        (
            "알 수 없는 action_id",
            logs["executed_action"].notna() & ~logs["executed_action"].isin(action_rules.ACTIONS.keys()),
        ),
        (
            "behavior_propensity 범위(0,1] 벗어남 — ope_eligible 행에 한함",
            ope_eligible & ~logs["behavior_propensity"].between(0, 1, inclusive="right"),
        ),
        (
            "ope_eligible=False인데 behavior_propensity가 있음",
            ~ope_eligible & logs["behavior_propensity"].notna(),
        ),
        ("yyqu_cd 또는 treatment_yyqu_cd 결측", logs["yyqu_cd"].isna() | logs["treatment_yyqu_cd"].isna()),
        ("treatment_yyqu_cd가 yyqu_cd 이후가 아님", logs["treatment_yyqu_cd"] <= logs["yyqu_cd"]),
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
        result["합성_행수"] = int((logs["데이터_출처"] == "synthetic_v2").sum())
        result["실제_행수"] = int((logs["데이터_출처"] == "real").sum())
    if "training_eligible" in logs.columns:
        result["학습가능_행수"] = int(logs["training_eligible"].astype(bool).sum())
    return result
