# campaign_logs_v2.csv의 부트스트랩 로그로 등급별 Bandit을 웜스타트한다
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import CAMPAIGN_LOGS_V2
from app.services.response import action_rules, bandit_store
from app.services.response.context import CONTEXT_DIM

from scripts.response_strategy.bandit import NeuralContextualBandit

CONTEXT_COLS = ["context_1", "context_2", "context_3", "context_4", "context_5", "context_6"]


def backfill(campaign_logs: Path = CAMPAIGN_LOGS_V2, synthetic_weight: float = 0.3,
             min_samples: int = 10, epochs: int = 50, seed: int = 0) -> dict:
    df = pd.read_csv(campaign_logs)
    # v2는 executed_action/training_eligible를 사용한다. 명시적으로 legacy
    # 파일을 넘기는 호출도 기존 컬럼을 지원해 마이그레이션 기간의 호환성을 유지한다.
    if "executed_action" in df.columns:
        df = df[df["training_eligible"].astype(bool)]
        df = df.rename(columns={"executed_action": "action_id"})
    else:
        df = df[df["executed"] == True]  # noqa: E712
        df = df[df["reward"].notna()]

    report: dict[str, dict] = {}
    for 등급, actions in action_rules.ACTION_RULES.items():
        if not actions:
            report[등급] = {"상태": "스킵", "사유": "후보 방안 없음(action_rules)"}
            continue

        rows = df[df["action_id"].isin(actions)]
        if len(rows) < min_samples:
            report[등급] = {"상태": "스킵", "사유": f"표본 부족({len(rows)}건, 최소 {min_samples}건)"}
            continue

        bandit = NeuralContextualBandit(
            context_dim=CONTEXT_DIM, arms=actions, seed=seed, policy_version="backfill-v1",
        )
        arm_index = {name: i for i, name in enumerate(actions)}

        for _, row in rows.iterrows():
            ctx = row[CONTEXT_COLS].to_numpy(dtype=np.float32)
            idx = arm_index[row["action_id"]]
            reward = float(row["reward"])
            weight = synthetic_weight if row["데이터_출처"] in {"synthetic", "synthetic_v2"} else 1.0
            bandit.update(ctx, idx, reward, weight=weight)

        loss = bandit.retrain_encoder(epochs=epochs, min_samples=min_samples)
        # legacy backfill 모델은 shadow 전용이라 active를 직접 덮어쓰지 않고 candidate로만 남긴다
        saved_path = bandit_store.save_candidate(등급, bandit)

        report[등급] = {
            "상태": "완료(candidate — shadow 전용, 승격 대상 아님)", "arm목록": actions, "표본수": len(rows),
            "행_출처분포": rows["데이터_출처"].value_counts().to_dict(),
            "최종_loss": round(loss, 4),
            "저장경로": str(saved_path),
        }
    return report


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, default=0.3, help="synthetic 로그 가중치")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=10)
    args = parser.parse_args()

    result = backfill(synthetic_weight=args.weight, epochs=args.epochs, min_samples=args.min_samples)
    print(json.dumps(result, ensure_ascii=False, indent=2))
