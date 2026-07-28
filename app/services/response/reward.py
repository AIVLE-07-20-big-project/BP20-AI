from __future__ import annotations

from dataclasses import asdict, dataclass


REWARD_DEFINITION_VERSION = "contribution_margin_v2"
DEFAULT_REWARD_DENOMINATOR_FLOOR = 1.0


@dataclass(frozen=True)
class RewardResult:
    status: str
    reward: float | None
    contribution_before: float | None
    contribution_after: float | None
    reward_definition_version: str
    reward_denominator_floor: float
    missing_fields: tuple[str, ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def calculate_reward_v2(
    *,
    net_sales_before: float | None,
    net_sales_after: float | None,
    variable_cost_before: float | None,
    variable_cost_after: float | None,
    campaign_cost: float | None,
    measurement_days_before: int | None,
    measurement_days_after: int | None,
    denominator_floor: float = DEFAULT_REWARD_DENOMINATOR_FLOOR,
) -> RewardResult:
    """비용을 반영한 실행 전후 순기여이익 변화율을 계산한다."""
    if denominator_floor <= 0:
        raise ValueError("reward denominator floor는 양수여야 합니다")

    values = {
        "net_sales_before": net_sales_before,
        "net_sales_after": net_sales_after,
        "variable_cost_before": variable_cost_before,
        "variable_cost_after": variable_cost_after,
        "campaign_cost": campaign_cost,
        "measurement_days_before": measurement_days_before,
        "measurement_days_after": measurement_days_after,
    }
    missing = tuple(name for name, value in values.items() if value is None)
    if missing:
        return RewardResult(
            status="incomplete",
            reward=None,
            contribution_before=None,
            contribution_after=None,
            reward_definition_version=REWARD_DEFINITION_VERSION,
            reward_denominator_floor=float(denominator_floor),
            missing_fields=missing,
            reason="Reward v2 필수 데이터가 누락되었습니다",
        )

    if measurement_days_before <= 0 or measurement_days_after <= 0:
        raise ValueError("measurement days는 양수여야 합니다")
    if measurement_days_before != measurement_days_after:
        return RewardResult(
            status="incomplete",
            reward=None,
            contribution_before=None,
            contribution_after=None,
            reward_definition_version=REWARD_DEFINITION_VERSION,
            reward_denominator_floor=float(denominator_floor),
            reason="실행 전후 측정 기간이 다릅니다",
        )
    if variable_cost_before < 0 or variable_cost_after < 0 or campaign_cost < 0:
        raise ValueError("비용은 음수일 수 없습니다")

    contribution_before = float(net_sales_before - variable_cost_before)
    contribution_after = float(net_sales_after - variable_cost_after - campaign_cost)
    denominator = max(abs(contribution_before), float(denominator_floor))
    reward = (contribution_after - contribution_before) / denominator

    return RewardResult(
        status="complete",
        reward=float(reward),
        contribution_before=contribution_before,
        contribution_after=contribution_after,
        reward_definition_version=REWARD_DEFINITION_VERSION,
        reward_denominator_floor=float(denominator_floor),
    )
