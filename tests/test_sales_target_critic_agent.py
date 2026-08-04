# 대상 경로(AI 레포): tests/test_sales_target_critic_agent.py
#
# critic_agent.review_batch()의 도구 호출 루프를 검증한다. 네트워크 호출이 없어(순수 pandas
# 계산) llm만 가짜 콜러블로 주입한다(pitch.py/review_matching_agent.py와 동일한 패턴).

import json
import unittest
from types import SimpleNamespace

from app.sales_target.critic_agent import review_batch


def _msg(tool_calls=None):
    return SimpleNamespace(content=None, tool_calls=tool_calls)


def _call(call_id: str, name: str, **kwargs):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(kwargs)))


def _ranked():
    return [
        {
            "bizesNm": "의심스러운카페",
            "growth_score": 20.0,
            "traffic_score": 20.0,
            "review_score": 20.0,
            "similarity_score": 95.0,
            "final_score": 71.0,
        },
        {
            "bizesNm": "평범한카페",
            "growth_score": 60.0,
            "traffic_score": 60.0,
            "review_score": 60.0,
            "similarity_score": 60.0,
            "final_score": 60.0,
        },
    ]


class ReviewBatchTests(unittest.TestCase):

    def test_flags_after_inspecting_breakdown_then_finishes(self):
        llm = iter([
            _msg(tool_calls=[_call("c1", "get_score_breakdown", bizesNm="의심스러운카페")]),
            _msg(tool_calls=[_call(
                "c2", "flag_candidate",
                bizesNm="의심스러운카페", reason="growth/traffic이 중앙값보다 훨씬 낮은데 similarity로만 상위권",
            )]),
            _msg(tool_calls=[_call("c3", "finish_review", summary="상위권 대부분 안정적, 1건만 유사도 편중")]),
        ])

        result = review_batch(_ranked(), llm=lambda m, t: next(llm))

        self.assertEqual(result["summary"], "상위권 대부분 안정적, 1건만 유사도 편중")
        self.assertEqual(len(result["flagged"]), 1)
        self.assertEqual(result["flagged"][0]["bizesNm"], "의심스러운카페")

    def test_score_breakdown_content_includes_medians(self):
        captured = {}

        def llm(messages, _tools):
            if len(messages) == 2:
                return _msg(tool_calls=[_call("c1", "get_score_breakdown", bizesNm="의심스러운카페")])
            # 직전 tool 메시지(get_score_breakdown 결과)를 검사
            tool_msg = messages[-1]
            captured["breakdown"] = json.loads(tool_msg["content"])
            return _msg(tool_calls=[_call("c2", "finish_review", summary="확인 완료")])

        review_batch(_ranked(), llm=llm)

        self.assertEqual(captured["breakdown"]["scores"]["similarity_score"], 95.0)
        self.assertEqual(captured["breakdown"]["batch_median"]["growth_score"], 40.0)  # (20+60)/2

    def test_no_suspicion_finishes_without_flagging(self):
        llm = iter([_msg(tool_calls=[_call("c1", "finish_review", summary="이상 없음")])])

        result = review_batch(_ranked(), llm=lambda m, t: next(llm))

        self.assertEqual(result["summary"], "이상 없음")
        self.assertEqual(result["flagged"], [])

    def test_flags_capped_at_max(self):
        ranked = [
            {"bizesNm": f"후보{i}", "growth_score": 50.0, "traffic_score": 50.0,
             "review_score": 50.0, "similarity_score": 50.0, "final_score": 50.0}
            for i in range(7)
        ]
        calls = [_call(f"f{i}", "flag_candidate", bizesNm=f"후보{i}", reason="테스트") for i in range(7)]
        llm = iter([_msg(tool_calls=calls), _msg(tool_calls=[_call("done", "finish_review", summary="완료")])])

        result = review_batch(ranked, llm=lambda m, t: next(llm))

        self.assertEqual(len(result["flagged"]), 5)

    def test_loop_cap_reached_without_finish_returns_fallback_summary(self):
        def always_inspect(_m, _t):
            return _msg(tool_calls=[_call("c", "get_score_breakdown", bizesNm="의심스러운카페")])

        result = review_batch(_ranked(), llm=always_inspect)

        self.assertEqual(result["summary"], "자동 요약 생성 실패")

    def test_llm_exception_returns_full_fallback(self):
        def broken_llm(_m, _t):
            raise RuntimeError("OpenAI 오류 시뮬레이션")

        result = review_batch(_ranked(), llm=broken_llm)

        self.assertEqual(result, {"summary": "자동 요약 생성 실패", "flagged": []})

    def test_empty_ranked_skips_without_calling_llm(self):
        call_count = {"n": 0}

        def llm(_m, _t):
            call_count["n"] += 1
            return _msg(tool_calls=None)

        result = review_batch([], llm=llm)

        self.assertEqual(result, {"summary": "후보가 없어 검수를 건너뜀", "flagged": []})
        self.assertEqual(call_count["n"], 0)

    def test_no_tool_call_at_all_breaks_with_fallback(self):
        result = review_batch(_ranked(), llm=lambda m, t: _msg(tool_calls=None))

        self.assertEqual(result["summary"], "자동 요약 생성 실패")
        self.assertEqual(result["flagged"], [])

    def test_unknown_bizesnm_breakdown_returns_error_without_crash(self):
        llm = iter([
            _msg(tool_calls=[_call("c1", "get_score_breakdown", bizesNm="존재하지않는후보")]),
            _msg(tool_calls=[_call("c2", "finish_review", summary="완료")]),
        ])

        result = review_batch(_ranked(), llm=lambda m, t: next(llm))

        self.assertEqual(result["summary"], "완료")


if __name__ == "__main__":
    unittest.main()
