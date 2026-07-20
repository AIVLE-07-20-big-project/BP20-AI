"""
Neural Contextual Bandit — 대응방안 선택 (WHAT)

선형 LinUCB 대신, 작은 MLP 인코더로 컨텍스트(상권 특성·연령대 분포·채널선호·문제유형 조합)를
학습된 표현으로 바꾼 뒤, 그 표현 위에서 팔(arm)별 LinUCB를 돌린다(Neural-Linear 방식 —
인코더는 배치로 재학습하고, 팔별 불확실성은 표현 공간에서 폐형식으로 빠르게 갱신한다).

콜드스타트: 실 로그가 없을 때는 인코더를 무작위 초기화한 채로 시작하되, 팔별 사전 보상
편향(prior_bias)을 문헌 기반 값으로 줄 수 있다 — 그 값 자체(어떤 문헌에서 어떤 수치를
가져올지)는 이 모듈이 정하지 않는다. app 레이어의 action_rules.py가 채워서 넘겨준다.

의존성
    필수: numpy, torch (Windows엔 CPU 전용 wheel로 설치:
          pip install torch --index-url https://download.pytorch.org/whl/cpu)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class BanditLoadMismatch(Exception):
    """저장된 모델의 아키텍처(context_dim·arm 집합)가 지금 요청과 달라 로드할 수 없음.

    에러로 죽지 않고 호출 측이 콜드스타트로 폴백할 수 있도록 별도 예외로 분리한다
    (정직한 낮은 신뢰도 > 거짓 확신, docs/response_recommendation_agent_plan.md §3).
    """


class _Encoder(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int = 32, encoding_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralContextualBandit:
    """컨텍스트 벡터 → (선택된 대응방안 arm, 근거)."""

    def __init__(self, context_dim: int, arms: list[str], encoding_dim: int = 16,
                 alpha: float = 1.0, ridge: float = 1.0,
                 prior_bias: dict[str, float] | None = None, seed: int = 0,
                 temperature: float = 1.0, policy_version: str = "coldstart"):
        torch.manual_seed(seed)
        self.arms = list(arms)
        self.context_dim = context_dim
        self.encoding_dim = encoding_dim
        self.alpha = alpha       # UCB 탐색 폭 계수
        self.ridge = ridge       # A 초기화 리지 계수(0으로 나눔 방지 + 초반 보수적 탐색)
        self.temperature = temperature  # propensity용 softmax 온도(추천 자체는 그대로 argmax)
        self.policy_version = policy_version
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
        with torch.no_grad():
            z = self.encoder(torch.as_tensor(np.asarray(context), dtype=torch.float32)).numpy()
        return z

    def select_arm(self, context: np.ndarray) -> dict:
        """추천 자체(top-1)는 항상 UCB argmax로 결정론적이다 — 탐색은 여기서 하지 않는다.

        다만 campaign_logs에 기록할 `propensity`는 필요하다(로깅 정책의 확률이 없으면
        OPE의 IPS/DR이 성립하지 않는다). 그래서 arm별 점수를 softmax해 "이 정책이 각 arm을
        골랐을 확률"을 별도로 계산한다 — 실제 선택은 여전히 argmax이지만, 승인 단계에서
        사람이 다른 후보로 edit하면 그 arm의 propensity도 이 분포에서 그대로 읽으면 된다
        (docs/response_recommendation_agent_plan.md 계획, campaign-logs 스키마 §참고).
        """
        z = self._encode(context)
        scores, widths = [], []
        for i in range(len(self.arms)):
            a_inv = np.linalg.inv(self.A[i])
            theta = a_inv @ self.b[i]
            mean = float(theta @ z) + float(self._prior_bias[i])
            width = float(self.alpha * np.sqrt(max(z @ a_inv @ z, 0.0)))
            scores.append(mean + width)
            widths.append(width)

        best = int(np.argmax(scores))
        scores_arr = np.asarray(scores)
        exp_scores = np.exp((scores_arr - scores_arr.max()) / self.temperature)
        propensities = exp_scores / exp_scores.sum()

        return {
            "선택된_arm": self.arms[best],
            "arm_index": best,
            "arm별_점수": {a: round(float(s), 4) for a, s in zip(self.arms, scores)},
            "arm별_propensity": {a: round(float(p), 6) for a, p in zip(self.arms, propensities)},
            "propensity": round(float(propensities[best]), 6),
            "불확실성_폭": round(widths[best], 4),
            "표본수": len(self.buffer),
            "policy_version": self.policy_version,
        }

    def update(self, context: np.ndarray, arm_index: int, reward: float, weight: float = 1.0) -> None:
        """LinUCB 표준 갱신 — 인코더는 그대로 두고 선택된 팔의 A/b만 갱신한다.

        `weight`는 이 표본을 얼마나 신뢰할지(1.0=실측 로그, <1.0=합성/백필 로그)를
        스케일한다 — docs/campaign_logs.md의 "합성 로그에 가중치를 두거나(예: 0.3)" 결정."""
        z = self._encode(context)
        self.A[arm_index] += weight * np.outer(z, z)
        self.b[arm_index] += weight * reward * z
        self.buffer.append((np.asarray(context, dtype=np.float32), arm_index, float(reward), float(weight)))

    def retrain_encoder(self, epochs: int = 50, lr: float = 1e-3, min_samples: int = 10) -> float:
        """버퍼에 쌓인 (컨텍스트, arm, 보상, 가중치)로 인코더를 재학습한다.

        campaign-logs가 실제로 쌓이기 전까지는 buffer가 비어 있어 호출할 이유가 없다
        (docs/response_recommendation_agent_plan.md §9 빌드 순서 6번 이후에 쓰임).
        """
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

        # 인코더가 바뀌면 표현 공간이 달라지므로 팔별 LinUCB 통계를 버퍼로 재구성한다.
        self._reset_linear_heads()
        for context, arm_index, reward, weight in self.buffer:
            z = self._encode(context)
            self.A[arm_index] += weight * np.outer(z, z)
            self.b[arm_index] += weight * reward * z
        return last_loss

    def save(self, path: str | Path) -> None:
        """encoder·LinUCB 통계·buffer·policy_version을 전부 저장한다.

        campaign-logs가 쌓이기 전까지는(§9 6번 이전) 호출할 이유가 없었다 — 매 요청마다
        새 콜드스타트 인스턴스를 만들었기 때문. 이제 등급별로 하나씩 지속시키는 자리
        (model/bandit/{등급}/active.pt)가 생겨 사용된다.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "context_dim": self.context_dim,
            "encoding_dim": self.encoding_dim,
            "arms": list(self.arms),
            "alpha": self.alpha,
            "ridge": self.ridge,
            "temperature": self.temperature,
            "prior_bias": self._prior_bias,
            "policy_version": self.policy_version,
            "encoder_state_dict": self.encoder.state_dict(),
            "A": self.A,
            "b": self.b,
            "buffer": self.buffer,
        }, path)

    @classmethod
    def _from_payload(cls, payload: dict) -> "NeuralContextualBandit":
        bandit = cls(
            context_dim=payload["context_dim"], arms=payload["arms"],
            encoding_dim=payload["encoding_dim"], alpha=payload["alpha"], ridge=payload["ridge"],
            temperature=payload.get("temperature", 1.0),
            policy_version=payload.get("policy_version", "unknown"),
        )
        bandit.encoder.load_state_dict(payload["encoder_state_dict"])
        bandit.encoder.eval()
        bandit.A = payload["A"]
        bandit.b = payload["b"]
        bandit.buffer = payload["buffer"]
        bandit._prior_bias = payload["prior_bias"]
        return bandit

    @classmethod
    def load(cls, path: str | Path, context_dim: int, arms: list[str]) -> "NeuralContextualBandit":
        """저장된 모델을 복원한다. context_dim·arms가 저장 당시와 다르면(예: action_rules의
        후보 목록이 바뀜) `BanditLoadMismatch`를 던진다 — 호출 측은 이걸 잡아 콜드스타트로
        폴백해야 한다(에러로 서비스가 죽으면 안 됨)."""
        payload = torch.load(Path(path), weights_only=False)
        if payload["context_dim"] != context_dim or list(payload["arms"]) != list(arms):
            raise BanditLoadMismatch(
                f"저장된 모델(context_dim={payload['context_dim']}, arms={payload['arms']})이 "
                f"요청(context_dim={context_dim}, arms={list(arms)})과 달라 로드할 수 없음"
            )
        return cls._from_payload(payload)

    @classmethod
    def load_any(cls, path: str | Path) -> "NeuralContextualBandit":
        """context_dim/arms 검증 없이 저장된 그대로 복원한다.

        `load()`는 "이 context_dim/arms를 기대하는 호출 측"을 위한 것이고, retrain
        스크립트처럼 "이 파일 자체가 곧 정답"인 경우엔 비교할 기준이 없다 — 그럴 때 쓴다.
        """
        payload = torch.load(Path(path), weights_only=False)
        return cls._from_payload(payload)


ROOT = Path(__file__).resolve().parents[2]
BANDIT_MODEL_DIR = ROOT / "model" / "bandit"


def retrain_cli(등급: str, min_samples: int = 10, epochs: int = 50) -> dict:
    """오프라인 재학습 — 온라인 update()로 이미 쌓인 active 모델의 buffer로 encoder를
    재학습해 새 버전 파일을 만든다(계획 §4). 자동으로 active.pt를 덮어쓰지 않는다 —
    ope.evaluate_policy()로 기존 모델과 정책가치를 비교한 뒤 사람이 수동으로 승격해야
    한다(같은 svc_induty_cd 범위로 도는 evaluate_policy와 등급 단위인 이 모델의 스코프가
    서로 다른 축이라, 자동 승격 규칙을 지금 단정하면 추측 기반 로직이 된다 — §3 원칙).
    """
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
    candidate_path = BANDIT_MODEL_DIR / 등급 / f"{version}.pt"
    bandit.save(candidate_path)

    return {
        "상태": "완료(수동 검토 필요)", "등급": 등급, "버전": version,
        "표본수": len(bandit.buffer), "최종_loss": round(loss, 4),
        "후보_경로": str(candidate_path),
        "안내": "ope.evaluate_policy()로 기존 active 대비 정책가치를 비교한 뒤 문제없으면 "
                "이 파일을 active.pt로 수동 교체할 것 — 이 스크립트는 자동 승격하지 않는다.",
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
        # 사용 예시 — 실제 arm/컨텍스트는 app 레이어(action_rules.py)가 채운다.
        bandit = NeuralContextualBandit(context_dim=8, arms=["쿠폰_20%", "이벤트_주말", "SNS_홍보"])
        ctx = np.random.default_rng(0).normal(size=8)
        print(bandit.select_arm(ctx))
