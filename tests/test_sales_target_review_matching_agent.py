# 대상 경로(AI 레포): tests/test_sales_target_review_matching_agent.py
#
# review_matching_agent.collect_review_stats_agentic()의 도구 호출 루프를 검증한다. 실제
# OpenAI/Google Places 호출은 없다 — llm은 가짜 콜러블 주입(pitch.py의 llm= 패턴과 동일),
# HTTP는 httpx.AsyncClient.post/get 모킹(google_places.py 기존 테스트와 동일한 방식).

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pandas as pd

from app.sales_target.review_matching_agent import collect_review_stats_agentic


def _msg(tool_calls=None):
    return SimpleNamespace(content=None, tool_calls=tool_calls)


def _call(call_id: str, name: str, **kwargs):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(kwargs)))


def _fake_response(json_body):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return json_body

    return _Resp()


def _candidates() -> pd.DataFrame:
    return pd.DataFrame({"bizesNm": ["루트커피"], "rdnmAdr": ["서울 마포구 성지길 5"]})


def _details_response():
    return _fake_response({
        "rating": 4.5,
        "userRatingCount": 120,
        "reviews": [
            {"publishTime": "2026-01-01T00:00:00Z"},
            {"publishTime": "2026-01-13T00:00:00Z"},
        ],
    })


class CollectReviewStatsAgenticTests(unittest.IsolatedAsyncioTestCase):

    async def test_first_search_succeeds_then_accepts_match(self):
        llm = iter([
            _msg(tool_calls=[_call("c1", "search_google_places", query="루트커피 서울 마포구 성지길 5")]),
            _msg(tool_calls=[_call("c2", "accept_match", place_id="PID1")]),
        ])

        async def fake_post(_client, url, json=None, headers=None, **kwargs):
            return _fake_response({"places": [{"id": "PID1", "displayName": {"text": "루트커피"}}]})

        async def fake_get(_client, url, headers=None, **kwargs):
            return _details_response()

        with patch.object(httpx.AsyncClient, "post", new=fake_post), \
             patch.object(httpx.AsyncClient, "get", new=fake_get):
            result = await collect_review_stats_agentic(_candidates(), api_key="test-key", llm=lambda m, t: next(llm))

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["review_count"], 120)
        self.assertEqual(result.iloc[0]["avg_rating"], 4.5)

    async def test_retries_with_reformulated_query_after_empty_first_search(self):
        llm = iter([
            _msg(tool_calls=[_call("c1", "search_google_places", query="가게A 101호")]),
            _msg(tool_calls=[_call("c2", "search_google_places", query="가게A")]),
            _msg(tool_calls=[_call("c3", "accept_match", place_id="PID2")]),
        ])
        post_calls = {"n": 0}

        async def fake_post(_client, url, json=None, headers=None, **kwargs):
            post_calls["n"] += 1
            if post_calls["n"] == 1:
                return _fake_response({"places": []})
            return _fake_response({"places": [{"id": "PID2"}]})

        async def fake_get(_client, url, headers=None, **kwargs):
            return _details_response()

        with patch.object(httpx.AsyncClient, "post", new=fake_post), \
             patch.object(httpx.AsyncClient, "get", new=fake_get):
            result = await collect_review_stats_agentic(_candidates(), api_key="test-key", llm=lambda m, t: next(llm))

        self.assertEqual(post_calls["n"], 2)
        self.assertEqual(len(result), 1)

    async def test_explicit_give_up_skips_candidate(self):
        llm = iter([
            _msg(tool_calls=[_call("c1", "search_google_places", query="아무거나")]),
            _msg(tool_calls=[_call("c2", "give_up", reason="찾을 수 없음")]),
        ])

        async def fake_post(_client, url, json=None, headers=None, **kwargs):
            return _fake_response({"places": []})

        with patch.object(httpx.AsyncClient, "post", new=fake_post):
            result = await collect_review_stats_agentic(_candidates(), api_key="test-key", llm=lambda m, t: next(llm))

        self.assertTrue(result.empty)

    async def test_no_tool_call_at_all_skips_candidate(self):
        llm = iter([_msg(tool_calls=None)])

        result = await collect_review_stats_agentic(_candidates(), api_key="test-key", llm=lambda m, t: next(llm))

        self.assertTrue(result.empty)

    async def test_exceeding_max_attempts_forces_give_up(self):
        # max_attempts=2인데 LLM이 계속 검색만 시도하면(절대 accept/give_up 안 부름) 루프 상한에서
        # 강제로 매칭 실패 처리돼야 한다.
        def infinite_search(_m, _t):
            return _msg(tool_calls=[_call("c", "search_google_places", query="계속 검색")])

        async def fake_post(_client, url, json=None, headers=None, **kwargs):
            return _fake_response({"places": [{"id": "PID_X"}]})

        with patch.object(httpx.AsyncClient, "post", new=fake_post):
            result = await collect_review_stats_agentic(
                _candidates(), api_key="test-key", llm=infinite_search, max_attempts=2
            )

        self.assertTrue(result.empty)

    async def test_accept_match_with_unknown_place_id_is_rejected(self):
        # LLM이 검색 결과에 없던 place_id로 accept_match를 부르면(할루시네이션) 매칭 실패로 처리한다.
        llm = iter([
            _msg(tool_calls=[_call("c1", "search_google_places", query="가게")]),
            _msg(tool_calls=[_call("c2", "accept_match", place_id="NEVER_SEEN")]),
        ])

        async def fake_post(_client, url, json=None, headers=None, **kwargs):
            return _fake_response({"places": [{"id": "PID1"}]})

        with patch.object(httpx.AsyncClient, "post", new=fake_post):
            result = await collect_review_stats_agentic(_candidates(), api_key="test-key", llm=lambda m, t: next(llm))

        self.assertTrue(result.empty)

    async def test_one_candidate_failure_does_not_block_others(self):
        candidates = pd.DataFrame({
            "bizesNm": ["실패후보", "성공후보"],
            "rdnmAdr": ["서울 어딘가 1", "서울 어딘가 2"],
        })

        def flaky_llm(messages, _tools):
            business_line = messages[1]["content"]
            if "실패후보" in business_line:
                raise RuntimeError("OpenAI 오류 시뮬레이션")
            if not any(m.get("role") == "tool" for m in messages):
                return _msg(tool_calls=[_call("c1", "search_google_places", query="성공후보 서울 어딘가 2")])
            return _msg(tool_calls=[_call("c2", "accept_match", place_id="PID_OK")])

        async def fake_post(_client, url, json=None, headers=None, **kwargs):
            return _fake_response({"places": [{"id": "PID_OK"}]})

        async def fake_get(_client, url, headers=None, **kwargs):
            return _details_response()

        with patch.object(httpx.AsyncClient, "post", new=fake_post), \
             patch.object(httpx.AsyncClient, "get", new=fake_get):
            result = await collect_review_stats_agentic(candidates, api_key="test-key", llm=flaky_llm)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["rdnmAdr"], "서울 어딘가 2")

    async def test_missing_name_or_address_skips_without_llm_call(self):
        candidates = pd.DataFrame({"bizesNm": [None], "rdnmAdr": ["서울 어딘가 1"]})
        call_count = {"n": 0}

        def llm(_m, _t):
            call_count["n"] += 1
            return _msg(tool_calls=None)

        result = await collect_review_stats_agentic(candidates, api_key="test-key", llm=llm)

        self.assertTrue(result.empty)
        self.assertEqual(call_count["n"], 0)


if __name__ == "__main__":
    unittest.main()
