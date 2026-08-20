"""공통 캠페인 로그 스키마."""

from __future__ import annotations

from app.services.response.context import CONTEXT_DIM

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

PERIOD_COLUMNS = {
    "execution_started_at", "execution_ended_at", "measurement_started_at",
    "measurement_ended_at", "baseline_period_start", "baseline_period_end",
    "control_store_ids",
}

REQUIRED_SCHEMA_COLUMNS = [
    column for column in SCHEMA_COLUMNS if column not in PERIOD_COLUMNS
]
