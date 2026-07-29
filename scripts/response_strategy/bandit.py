# Neural Contextual Bandit — 대응방안 선택 (WHAT)
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


# 저장된 모델의 아키텍처(context_dim·arm 집합)가 지금 요청과 달라 로드할 수 없음
class BanditLoadMismatch(Exception):
    pass


@dataclass(frozen=True)
class ArmScore:
    action_id: str
    expected_reward: float
    uncertainty: float
    ucb_score: float

    def as_dict(self, ndigits: int | None = None) -> dict:
        def value(number: float) -> float:
            return round(number, ndigits) if ndigits is not None else number

        return {
            "action_id": self.action_id,
            "expected_reward": value(self.expected_reward),
            "uncertainty": value(self.uncertainty),
            "ucb_score": value(self.ucb_score),
        }






class _Encoder(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int = 32, encoding_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 컨텍스트 벡터 → (선택된 대응방안 arm, 근거)
class NeuralContextualBandit:


    def __init__(self, context_dim: int, arms: list[str], encoding_dim: int = 16,
                 alpha: float = 1.0, ridge: float = 1.0,
                 prior_bias: dict[str, float] | None = None, seed: int = 0,
                 temperature: float = 1.0, policy_version: str = "coldstart",
                 model_version: str = "legacy",
                 context_schema_version: str = "legacy",
                 reward_definition_version: str = "revenue_lift_v1",
                 training_data_cutoff: str | None = None):
        torch.manual_seed(seed)
        self.arms = list(arms)
        self.context_dim = context_dim
        self.encoding_dim = encoding_dim
        self.alpha = alpha
        self.ridge = ridge
        self.temperature = temperature
        self.policy_version = policy_version
        self.model_version = model_version
        self.context_schema_version = context_schema_version
        self.reward_definition_version = reward_definition_version
        self.training_data_cutoff = training_data_cutoff
        self.encoder = _Encoder(context_dim, encoding_dim=encoding_dim)
        self.encoder.eval()

        prior_bias = prior_bias or {}
        self._prior_bias = np.array([prior_bias.get(a, 0.0) for a in self.arms])
        self._reset_linear_heads()

        self.buffer: list[tuple[np.ndarray, int, float]] = []

    def _reset_linear_heads(self) -> None:
        n = len(self.arms)
        self.A = np.stack([np.eye(self.encoding_dim) * self.ridge for _ in range(n)])
        self.b = np.zeros((n, self.encoding_dim))

    def _encode(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        if context.shape != (self.context_dim,):
            raise ValueError(
                f"context shape은 ({self.context_dim},)이어야 합니다: {context.shape}"
            )
        if not np.isfinite(context).all():
            raise ValueError("context에는 NaN 또는 Inf를 사용할 수 없습니다")
        with torch.no_grad():
            z = self.encoder(torch.as_tensor(context, dtype=torch.float32)).numpy()
        return z

    def score_arms(self, context: np.ndarray) -> list[ArmScore]:
        """MLP 표현 위에서 arm별 기대보상·불확실성·UCB를 계산한다."""
        z = self._encode(context)
        scores: list[ArmScore] = []
        for i, action_id in enumerate(self.arms):
            a_inv = np.linalg.inv(self.A[i])
            theta = a_inv @ self.b[i]
            expected_reward = float(theta @ z) + float(self._prior_bias[i])
            uncertainty = float(self.alpha * np.sqrt(max(z @ a_inv @ z, 0.0)))
            scores.append(ArmScore(
                action_id=action_id,
                expected_reward=expected_reward,
                uncertainty=uncertainty,
                ucb_score=expected_reward + uncertainty,
            ))
        return scores

    # 추천 자체(top-1)는 항상 UCB argmax로 결정론적이다 — 탐색은 여기서 하지 않는다
    def select_arm(self, context: np.ndarray) -> dict:








        arm_scores = self.score_arms(context)
        scores = [score.ucb_score for score in arm_scores]

        best = int(np.argmax(scores))
        scores_arr = np.asarray(scores)
        exp_scores = np.exp((scores_arr - scores_arr.max()) / self.temperature)
        propensities = exp_scores / exp_scores.sum()

        return {
            "선택된_arm": self.arms[best],
            "arm_index": best,
            "arm별_점수": {a: round(float(s), 4) for a, s in zip(self.arms, scores)},
            "arm별_기대보상": {
                score.action_id: round(score.expected_reward, 4) for score in arm_scores
            },
            "arm별_불확실성": {
                score.action_id: round(score.uncertainty, 4) for score in arm_scores
            },
            "arm별_UCB점수": {
                score.action_id: round(score.ucb_score, 4) for score in arm_scores
            },
            "arm별_propensity": {a: round(float(p), 6) for a, p in zip(self.arms, propensities)},
            "propensity": round(float(propensities[best]), 6),
            "불확실성_폭": round(arm_scores[best].uncertainty, 4),
            "표본수": len(self.buffer),
            "policy_version": self.policy_version,
            "model_version": self.model_version,
            "context_schema_version": self.context_schema_version,
            "reward_definition_version": self.reward_definition_version,
        }

    # LinUCB 표준 갱신 — 인코더는 그대로 두고 선택된 팔의 A/b만 갱신한다
    def update(self, context: np.ndarray, arm_index: int, reward: float, weight: float = 1.0) -> None:




        z = self._encode(context)
        self.A[arm_index] += weight * np.outer(z, z)
        self.b[arm_index] += weight * reward * z
        self.buffer.append((np.asarray(context, dtype=np.float32), arm_index, float(reward), float(weight)))

    # 버퍼에 쌓인 (컨텍스트, arm, 보상, 가중치)로 인코더를 재학습한다
    def retrain_encoder(self, epochs: int = 50, lr: float = 1e-3, min_samples: int = 10) -> float:





        if len(self.buffer) < min_samples:
            raise ValueError(f"재학습에 표본이 부족합니다(현재 {len(self.buffer)}개, 최소 {min_samples}개 필요)")

        contexts = torch.as_tensor(np.stack([c for c, _, _, _ in self.buffer]), dtype=torch.float32)
        arm_idx = np.array([a for _, a, _, _ in self.buffer])
        rewards = torch.as_tensor(np.array([r for _, _, r, _ in self.buffer]), dtype=torch.float32)
        weights = torch.as_tensor(np.array([w for _, _, _, w in self.buffer]), dtype=torch.float32)

        self.encoder.train()
        head = nn.Linear(self.encoding_dim, len(self.arms))
        optim = torch.optim.Adam(list(self.encoder.parameters()) + list(head.parameters()), lr=lr)

        last_loss = float("nan")
        for _ in range(epochs):
            optim.zero_grad()
            z = self.encoder(contexts)
            pred = head(z)[torch.arange(len(arm_idx)), arm_idx]
            loss = (weights * (pred - rewards) ** 2).mean()
            loss.backward()
            optim.step()
            last_loss = float(loss.item())
        self.encoder.eval()


        self._reset_linear_heads()
        for context, arm_index, reward, weight in self.buffer:
            z = self._encode(context)
            self.A[arm_index] += weight * np.outer(z, z)
            self.b[arm_index] += weight * reward * z
        return last_loss

    # encoder·LinUCB 통계·buffer·policy_version을 전부 저장한다
    def save(self, path: str | Path) -> None:






        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 저장 중에 select_arm의 load가 겹쳐도 부분 파일을 읽지 않도록 원자적으로 교체한다
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "context_dim": self.context_dim,
            "encoding_dim": self.encoding_dim,
            "arms": list(self.arms),
            "alpha": self.alpha,
            "ridge": self.ridge,
            "temperature": self.temperature,
            "prior_bias": self._prior_bias,
            "policy_version": self.policy_version,
            "model_version": self.model_version,
            "context_schema_version": self.context_schema_version,
            "reward_definition_version": self.reward_definition_version,
            "training_data_cutoff": self.training_data_cutoff,
            "encoder_state_dict": self.encoder.state_dict(),
            "A": self.A,
            "b": self.b,
            "buffer": self.buffer,
        }, tmp_path)
        tmp_path.replace(path)

    @classmethod
    def _from_payload(cls, payload: dict) -> "NeuralContextualBandit":
        bandit = cls(
            context_dim=payload["context_dim"], arms=payload["arms"],
            encoding_dim=payload["encoding_dim"], alpha=payload["alpha"], ridge=payload["ridge"],
            temperature=payload.get("temperature", 1.0),
            policy_version=payload.get("policy_version", "unknown"),
            model_version=payload.get("model_version", "legacy"),
            context_schema_version=payload.get("context_schema_version", "legacy"),
            reward_definition_version=payload.get("reward_definition_version", "revenue_lift_v1"),
            training_data_cutoff=payload.get("training_data_cutoff"),
        )
        bandit.encoder.load_state_dict(payload["encoder_state_dict"])
        bandit.encoder.eval()
        bandit.A = payload["A"]
        bandit.b = payload["b"]
        bandit.buffer = payload["buffer"]
        bandit._prior_bias = payload["prior_bias"]
        return bandit

    # 저장된 모델을 복원한다. context_dim·arms가 저장 당시와 다르면(예: action_rules의 arm 목록 변경)
    # BanditLoadMismatch를 던진다.
    @classmethod
    def load(cls, path: str | Path, context_dim: int, arms: list[str]) -> "NeuralContextualBandit":



        payload = torch.load(Path(path), weights_only=False)
        if payload["context_dim"] != context_dim or list(payload["arms"]) != list(arms):
            raise BanditLoadMismatch(
                f"저장된 모델(context_dim={payload['context_dim']}, arms={payload['arms']})이 "
                f"요청(context_dim={context_dim}, arms={list(arms)})과 달라 로드할 수 없음"
            )
        return cls._from_payload(payload)

    # context_dim/arms 검증 없이 저장된 그대로 복원한다
    @classmethod
    def load_any(cls, path: str | Path) -> "NeuralContextualBandit":





        payload = torch.load(Path(path), weights_only=False)
        return cls._from_payload(payload)


ROOT = Path(__file__).resolve().parents[2]
BANDIT_MODEL_DIR = ROOT / "model" / "bandit"


# 오프라인 재학습 — 온라인 update()로 이미 쌓인 active 모델의 buffer로 encoder를 재학습해 candidate로 저장한다.
def retrain_cli(등급: str, min_samples: int = 10, epochs: int = 50) -> dict:






    active_path = BANDIT_MODEL_DIR / 등급 / "active.pt"
    if not active_path.exists():
        return {"상태": "실패", "사유": f"{등급}의 active 모델이 없음(온라인 update가 아직 없었던 상태)"}

    bandit = NeuralContextualBandit.load_any(active_path)
    if len(bandit.buffer) < min_samples:
        return {"상태": "실패",
                "사유": f"buffer 표본 부족(현재 {len(bandit.buffer)}개, 최소 {min_samples}개 필요)"}

    loss = bandit.retrain_encoder(epochs=epochs, min_samples=min_samples)

    version = f"retrained-{int(time.time())}"
    bandit.policy_version = version
    # bandit_store.candidate_path()와 같은 파일명 규칙(app 계층 순환 의존 회피)
    candidate_path = BANDIT_MODEL_DIR / 등급 / f"candidate-{version}.pt"
    bandit.save(candidate_path)

    return {
        "상태": "완료(수동 검토 필요)", "등급": 등급, "버전": version,
        "표본수": len(bandit.buffer), "최종_loss": round(loss, 4),
        "후보_경로": str(candidate_path),
        "안내": "ope.evaluate_policy_value()와 activation.check_activation()으로 활성화 기준을 "
                "확인한 뒤 충족하면 bandit_store.promote_candidate(등급, 버전)으로 승격할 것 — "
                "이 스크립트는 자동 승격하지 않는다.",
    }


if __name__ == "__main__":
    import json
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    _args = sys.argv[1:]
    if _args and _args[0] == "retrain" and len(_args) >= 2:
        print(json.dumps(retrain_cli(_args[1]), ensure_ascii=False, indent=2))
    else:

        bandit = NeuralContextualBandit(context_dim=8, arms=["쿠폰_20%", "이벤트_주말", "SNS_홍보"])
        ctx = np.random.default_rng(0).normal(size=8)
        print(bandit.select_arm(ctx))
