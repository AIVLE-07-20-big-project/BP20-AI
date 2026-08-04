# 대상 경로(AI 레포): app/sales_target/pitch.py
#
# 신규 가맹점 영업 타겟 — LLM 세일즈 피칭 문구 생성.
# AI_Agent_전환_가이드라인.md 2단계(§7) 구현. 신규가맹점_영업타겟추천_설계가이드.md 5.6절 스펙
# (`generate_sales_pitch(candidate, matched_hero_stores)`, 출력 예시 "~신규 영업 우선 후보로
# 추천합니다. 전환 가능성이 78%로 매우 높습니다")을 따르고, rag/generator.py의 OpenAI 클라이언트
# 설정(같은 모델 gpt-4.1, OPENAI_API_KEY)을 그대로 재사용한다.
#
# 수치 취급 원칙은 rag/generator.py와 동일하게 간다: candidate의 final_score(이미 0~100으로
# 정규화된 값) 외의 숫자는 LLM이 만들어내지 못하게 프롬프트에서 명시적으로 막는다. 문구에 나오는
# "전환 가능성 78%"의 78은 final_score를 반올림한 값이지, LLM이 새로 추정한 확률이 아니다.

from __future__ import annotations

import os

_QUALITATIVE_LABELS = [
    (85, "매우 높습니다"),
    (70, "높습니다"),
    (50, "보통입니다"),
    (0, "낮은 편입니다"),
]

_FALLBACK_TEXT = "추천 사유가 아직 생성되지 않았습니다."

_SYSTEM_PROMPT = (
    "당신은 프랜차이즈 본사의 영업 담당자를 돕는 세일즈 피칭 문구 작성 도우미다. "
    "제공된 수치 밖의 사실이나 통계를 결코 만들어내지 않는다."
)


def _conversion_label(score: int) -> str:
    for threshold, label in _QUALITATIVE_LABELS:
        if score >= threshold:
            return label
    return _QUALITATIVE_LABELS[-1][1]


def _hero_strengths_lines(matched_hero_stores: list[dict] | None, candidate_industry: str | None) -> list[str]:
    """후보 업종과 같은 업종의 우수 가맹점만 골라서 근거로 쓴다.

    이전에는 배치에서 뽑힌 우수 가맹점 top-3를 후보 업종과 무관하게 그대로 넘겨서, 노래방
    후보에게 "과학·기술 분야 우수 가맹점" 같은 안 맞는 근거가 언급되는 문제가 있었다(2026-08-04
    실제 배치에서 확인됨). hero_stores의 category는 candidate의 indsLclsNm(대분류, 예: "과학·기술")과
    같은 값으로 들어온다 — BE our_stores 응답 실측 기준.
    """
    if not matched_hero_stores:
        return ["- (없음) — 참고할 우수 가맹점 데이터가 아직 없으므로 일반적인 강점만 언급할 것"]
    matched = [
        h for h in matched_hero_stores
        if candidate_industry and h.get("category") == candidate_industry
    ]
    if not matched:
        return ["- (없음) — 후보 업종과 일치하는 우수 가맹점 사례가 없으므로 일반적인 강점만 언급할 것"]
    lines = []
    for h in matched[:3]:
        category = h.get("category") or "동종업계"
        rating = h.get("reviewAvgRating")
        rating_txt = f", 평균 평점 {rating}" if rating is not None else ""
        lines.append(f"- {category} 우수 가맹점 사례{rating_txt}")
    return lines


def _build_prompt(candidate: dict, matched_hero_stores: list[dict] | None) -> tuple[str, int]:
    raw_score = candidate.get("final_score")
    final_score = round(float(raw_score)) if raw_score is not None else 0
    final_score = max(0, min(100, final_score))

    lines: list[str] = []
    lines.append("아래 규칙을 반드시 지켜 신규 가맹점 영업 타겟 후보에 대한 세일즈 피칭 문구를 작성하라.\n")
    lines.append("[규칙]")
    lines.append(
        f"1. 숫자는 오직 {final_score} 하나만 쓸 수 있다. 매출액, 방문자 수, 비율 등 다른 숫자는 절대 새로 만들지 않는다."
    )
    lines.append(
        f"2. 반드시 \"전환 가능성이 {final_score}%로 {_conversion_label(final_score)}\"라는 표현을 문장에 그대로 포함한다."
    )
    lines.append("3. 2~3문장 이내로, 영업 담당자가 바로 읽고 쓸 수 있는 자연스러운 한국어 문장으로 작성한다.")
    lines.append("4. 업장의 지역/업종과 유사 우수 가맹점의 강점을 근거로 언급하되, 근거 없는 효과나 매출 예측은 서술하지 않는다.")
    lines.append("")
    lines.append("[후보 업장]")
    lines.append(f"- 상호: {candidate.get('bizesNm') or candidate.get('businessName') or '이름 미상'}")
    lines.append(
        f"- 업종: {candidate.get('indsMclsNm') or candidate.get('indsLclsNm') or candidate.get('industry') or '미상'}"
    )
    lines.append(f"- 주소: {candidate.get('rdnmAdr') or candidate.get('address') or '미상'}")
    lines.append(f"- 상권 성장 점수: {candidate.get('growth_score')}")
    lines.append(f"- 유동인구 점수: {candidate.get('traffic_score')}")
    lines.append(f"- 유사 우수 가맹점 점수: {candidate.get('similarity_score')}")
    lines.append("")
    lines.append("[유사한 우수 가맹점 강점]")
    candidate_industry = candidate.get("indsLclsNm") or candidate.get("industry")
    lines.extend(_hero_strengths_lines(matched_hero_stores, candidate_industry))

    return "\n".join(lines), final_score


# OpenAI 호출. 키는 환경변수 OPENAI_API_KEY 사용 (rag/generator.py의 _call_openai와 동일 패턴).
def _call_openai(prompt: str, model: str = "gpt-4.1", temperature: float = 0.4) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def generate_sales_pitch(
    candidate: dict,
    matched_hero_stores: list[dict] | None = None,
    llm=None,
) -> str:
    """후보 1건에 대한 세일즈 피칭 문구를 생성한다.

    candidate: ranked DataFrame의 한 행(dict). bizesNm/indsLclsNm/rdnmAdr/growth_score/
      traffic_score/similarity_score/final_score를 참고한다.
    matched_hero_stores: 이번 배치에서 선정된 우수 가맹점 목록(list[dict], category/
      reviewAvgRating 등). 후보별 정밀 매칭이 아니라 배치 공통 참고 자료로 쓴다 — hero_store.py의
      hero_similarity_scores()가 top-k 평균 점수만 반환하고 어느 우수 가맹점이 매칭됐는지는
      추적하지 않기 때문(현재 파이프라인의 한계).

    LLM 호출이 실패하면(키 미설정, API 오류, 레이트리밋 등) 예외를 던지지 않고 fallback 문구를
    반환한다 — 피칭 문구 생성 실패로 finalize(BE 반영) 단계 전체가 막히면 안 되기 때문이다.
    """
    llm = llm or _call_openai
    prompt, _final_score = _build_prompt(candidate, matched_hero_stores)
    try:
        text = llm(prompt)
    except Exception as exc:
        print(f"[generate_sales_pitch] LLM 호출 실패, fallback 문구 사용: {type(exc).__name__}: {exc}")
        return _FALLBACK_TEXT
    text = (text or "").strip()
    return text or _FALLBACK_TEXT
