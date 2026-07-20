"""campaign_logs.csv의 synthetic 부트스트랩 로그로 등급별 Bandit을 웜스타트한다.

`model/bandit/`가 지금까지 완전히 비어 있던 이유: Bandit은 `POST /campaign-logs`가
호출될 때만 `bandit.update()`를 거친다(app/services/response/campaign_logs.py의
`_update_bandit_online`). 그런데 데이터 부트스트랩용 6,000행은 그 API를 거치지 않고
`data/campaign_logs.csv`에 직접 채워졌다 — SCM/OPE는 파일을 직접 읽어 혜택을 봤지만
Bandit은 이 데이터를 한 번도 학습하지 못했다(docs/campaign_logs.md의 "Bandit — 합성
로그에 가중치를 두거나(예: 0.3)" 결정이 아직 구현되지 않은 부분).

이 스크립트가 그 gap을 메운다: action_id로 등급(문제유형)을 역추적해 등급별 arm
집합으로 Bandit을 만들고, 각 로그 행을 `weight`(기본 0.3, 데이터_출처=="synthetic"일 때만
적용)로 낮춰서 update() → 마지막에 retrain_encoder()로 인코더를 실제로 학습한 뒤
model/bandit/{등급}/active.pt로 저장한다.

"배달채널 확대"는 고객_회복·차별화 두 등급 모두의 후보라(action_rules.py 주석 참고)
두 버퍼 모두에 들어간다 — 억지로 하나만 고르지 않는다.

사용:
    python bandit_backfill.py                     # 기본 가중치 0.3, 전체 등급
    python bandit_backfill.py --weight 0.5 --epochs 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import CAMPAIGN_LOGS
from app.services.response import action_rules, bandit_store
from app.services.response.context import CONTEXT_DIM

from scripts.response_strategy.bandit import NeuralContextualBandit

CONTEXT_COLS = ["context_1", "context_2", "context_3", "context_4", "context_5", "context_6"]


def backfill(campaign_logs: Path = CAMPAIGN_LOGS, synthetic_weight: float = 0.3,
             min_samples: int = 10, epochs: int = 50, seed: int = 0) -> dict:
    df = pd.read_csv(campaign_logs)
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
            weight = synthetic_weight if row["데이터_출처"] == "synthetic" else 1.0
            bandit.update(ctx, idx, reward, weight=weight)

        loss = bandit.retrain_encoder(epochs=epochs, min_samples=min_samples)
        bandit_store.save(등급, bandit)

        report[등급] = {
            "상태": "완료", "arm목록": actions, "표본수": len(rows),
            "행_출처분포": rows["데이터_출처"].value_counts().to_dict(),
            "최종_loss": round(loss, 4),
            "저장경로": str(bandit_store.model_path(등급)),
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
