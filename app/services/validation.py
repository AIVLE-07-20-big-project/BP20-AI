# 대응방안 추천 결과가 같은 요청의 진단 결과와 실제로 맞물리는지 결정론적으로 재대조한다
from __future__ import annotations


def validate(raw_diag: dict, recommendation: dict) -> dict:
    problems = {p["유형"]: p for p in recommendation.get("문제유형", [])}
    분석사용가능 = raw_diag.get("6_신뢰도", {}).get("분석사용가능", False)
    추천목록 = recommendation.get("대응방안_추천", [])

    findings: list[str] = []
    checks = {"진단연결": True, "사업영향도": True, "방안적합성": True, "제약조건": True}

    if not 분석사용가능 and 추천목록:
        findings.append("분석사용가능=False인데 대응방안이 생성됨 — 신뢰도 게이팅 불일치")
        checks["제약조건"] = False

    for item in 추천목록:
        문제 = problems.get(item.get("대상_문제"))
        if 문제 is None:
            msg = f"'{item.get('방안')}'이 참조하는 문제유형 '{item.get('대상_문제')}'이 진단 결과에 없음"
            findings.append(msg)
            checks["진단연결"] = False
            item["검증경고"] = msg
            continue
        if item.get("근거") != 문제.get("근거"):
            msg = f"'{item.get('방안')}'의 근거 문구가 진단 결과와 불일치"
            findings.append(msg)
            checks["진단연결"] = False
            item["검증경고"] = msg
        share = 문제.get("매출비중")
        if share is not None and share < 0.03:
            msg = f"'{item.get('방안')}' 대상 문제의 매출비중이 3% 미만"
            findings.append(msg)
            checks["사업영향도"] = False
            item["검증경고"] = msg
        if item.get("진단기반_우선순위점수") is None or not item.get("점수구성"):
            findings.append(f"'{item.get('방안')}'의 설명 가능한 우선순위 근거 누락")
            checks["방안적합성"] = False

    names = [item.get("방안") for item in 추천목록]
    if len(names) != len(set(names)):
        findings.append("동일 대응방안이 중복 추천됨")
        checks["제약조건"] = False

    return {
        "검증_통과": not findings,
        "발견사항": findings,
        "과정검증": {key: "통과" if passed else "실패" for key, passed in checks.items()},
        "효과검증": {
            "반합성": "참고가능(L1)",
            "실제파일럿": "미검증",
            "자동집행": recommendation.get("자동집행", "보류"),
        },
        "사용가능범위": recommendation.get("추천용도", "담당자 검토"),
        "안정성검증": recommendation.get("추천안정성", {"상태": "미측정"}),
        "해석주의": (
            "이 검증은 추천 결과가 같은 요청의 진단 결과를 정확히 참조하는지 확인하는 "
            "결정론적 무결성 체크이며, 추천 자체의 효과(uplift)를 검증하지 않는다. "
            "LangGraph 워크플로우에서는 이 검사 이후 GPT 수치 grounding과 "
            "human-in-the-loop 승인을 추가로 수행한다."
        ),
    }
