# 대상 경로(AI 레포): tests/test_sales_target_graph.py
#
# app/sales_target/graph.py의 StateGraph가 실제로 interrupt에서 멈췄다가 Command(resume=...)로
# 재개되는지 검증한다. 콜렉터/BE 클라이언트는 전부 모킹한다(네트워크 호출 없음) — 계산 로직
# 자체는 test_sales_target_pipeline.py에서 이미 검증됐으므로 여기서는 그래프 배선(오케스트레이션 +
# interrupt/resume)만 검증하는 게 목적이다. 체크포인터는 실제 repo의 model/ 폴더가 아니라
# 임시 디렉터리의 sqlite 파일을 쓰도록 setUp에서 리다이렉트한다(테스트가 저장소에 흔적을 안 남기게).

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from app.sales_target import graph as graph_module
from app.sales_target.router import (
    continue_sales_target_run,
    read_sales_target_run,
    start_sales_target_run,
)


def _candidate_registry_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "bizesNm": ["이미가맹점카페", "루트커피(T1)"],
        "rdnmAdr": ["서울 강남구 테스트로 1", "서울 마포구 성지길 5"],
        "lon": [127.1, 127.0],
        "lat": [37.5, 37.5],
    })


def _trdar_boundary_fixture() -> pd.DataFrame:
    # pyproj로 lon=127.0, lat=37.5(T1)로 역산되도록 만든 값. test_sales_target_pipeline.py와 동일.
    return pd.DataFrame({
        "TRDAR_CD": ["T1"],
        "XCNTS_VALUE": ["208842.54558833793"],
        "YDNTS_VALUE": ["444508.8210425943"],
        "RELM_AR": ["10000"],
    })


def _sales_raw_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "TRDAR_CD": ["T1", "T1"],
        "STDR_YYQU_CD": ["20261", "20262"],
        "THSMON_SELNG_AMT": [1000, 1500],
    })


def _traffic_raw_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "TRDAR_CD": ["T1", "T1"],
        "STDR_YYQU_CD": ["20261", "20262"],
        "TMZON_06_11_FLPOP_CO": [100.0, 100.0],
        "TMZON_11_14_FLPOP_CO": [100.0, 100.0],
        "TMZON_14_17_FLPOP_CO": [100.0, 100.0],
    })


def _store_stats_raw_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "TRDAR_CD": ["T1", "T1"],
        "STDR_YYQU_CD": ["20261", "20262"],
        "STOR_CO": [40, 45],
    })


def _our_stores_fixture() -> pd.DataFrame:
    return pd.DataFrame({"address": ["서울 강남구 테스트로 1"]})


class SalesTargetGraphTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        tmp_db_path = Path(self._tmpdir.name) / "test_sales_target_graph.sqlite3"

        db_patcher = patch.object(graph_module, "SALES_TARGET_GRAPH_DB", tmp_db_path)
        url_patcher = patch.object(graph_module, "SALES_TARGET_CHECKPOINT_DB_URL", "")
        db_patcher.start()
        url_patcher.start()
        self.addCleanup(db_patcher.stop)
        self.addCleanup(url_patcher.stop)

        graph_module._checkpointer.cache_clear()
        graph_module.get_graph.cache_clear()
        # sqlite3 커넥션을 열어둔 채로 tmpdir을 지우면 Windows에서는 PermissionError가 난다
        # (POSIX는 열린 파일도 unlink 가능해서 안 드러났음) — tmpdir.cleanup보다 먼저 실행되도록
        # addCleanup은 뒤에 등록한다(LIFO).
        self.addCleanup(self._close_checkpointer)

    def _close_checkpointer(self):
        if graph_module._checkpointer.cache_info().currsize:
            graph_module._checkpointer().conn.close()
        graph_module._checkpointer.cache_clear()

    def _patch_collectors(self, rejected_addresses: list[str] | None = None) -> tuple[AsyncMock, MagicMock]:
        """콜렉터/BE 클라이언트 4+2종 + pitch.py의 OpenAI 호출을 전부 모킹하고,
        (push_bulk_mock, openai_call_mock)을 반환한다.

        rejected_addresses: 5단계(피드백 루프) — SalesTargetIngestClient.fetch_excluded_addresses()의
        모킹 반환값. 기본값 []이면 아무 후보도 추가로 제외되지 않는다."""
        store_registry_cls = MagicMock()
        store_registry_cls.return_value.fetch_sigungus = AsyncMock(return_value=_candidate_registry_fixture())

        trdar_boundary_cls = MagicMock()
        trdar_boundary_cls.return_value.fetch_all = AsyncMock(return_value=_trdar_boundary_fixture())

        sales_cls = MagicMock()
        sales_cls.return_value.fetch_quarters = AsyncMock(return_value=_sales_raw_fixture())

        traffic_cls = MagicMock()
        traffic_cls.return_value.fetch_quarters = AsyncMock(return_value=_traffic_raw_fixture())

        store_stats_cls = MagicMock()
        store_stats_cls.return_value.fetch_quarters = AsyncMock(return_value=_store_stats_raw_fixture())

        backend_client_cls = MagicMock()
        backend_client_cls.return_value.fetch_our_stores = AsyncMock(return_value=_our_stores_fixture())

        ingest_client_cls = MagicMock()
        push_bulk_mock = AsyncMock(return_value={"created": 1, "updated": 0})
        ingest_client_cls.return_value.push_bulk = push_bulk_mock
        ingest_client_cls.return_value.fetch_excluded_addresses = AsyncMock(
            return_value=rejected_addresses or []
        )

        # generate_pitch 노드가 실제 OpenAI를 호출하지 않도록 pitch.py의 LLM 호출부만 모킹한다
        # (generate_sales_pitch() 자체의 프롬프트/폴백 로직은 실제로 타게 둔다).
        openai_call_mock = MagicMock(return_value="테스트 후보를 신규 영업 우선 후보로 추천합니다. 전환 가능성이 80%로 매우 높습니다.")

        patches = [
            patch("scripts.collection.store_registry_collector.StoreRegistryCollector", store_registry_cls),
            patch("scripts.collection.collectors.TrdarBoundaryCollector", trdar_boundary_cls),
            patch("scripts.collection.collectors.SalesEstimateCollector", sales_cls),
            patch("scripts.collection.collectors.FootTrafficCollector", traffic_cls),
            patch("scripts.collection.collectors.StoreStatsCollector", store_stats_cls),
            patch("app.sales_target.graph.BackendStoreRegistryClient", backend_client_cls),
            patch("app.sales_target.graph.SalesTargetIngestClient", ingest_client_cls),
            patch("app.sales_target.pitch._call_openai", openai_call_mock),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return push_bulk_mock, openai_call_mock

    def _initial_state(self) -> dict:
        return {
            "store_registry_api_key": "dummy",
            "seoul_api_key": "dummy",
            "backend_base_url": "http://localhost:8080",
            "backend_internal_api_key": "dummy",
            "quarter_codes": ["20261", "20262"],
            "top_n": 10,
            "source_batch_id": "test-batch-1",
            "warnings": [],
        }

    def test_graph_pauses_at_review_with_ranked_candidates(self):
        self._patch_collectors()

        result = start_sales_target_run(self._initial_state())

        self.assertIsNotNone(result["대기중_승인"])
        candidates = result["대기중_승인"]["후보_리스트"]
        # "이미가맹점카페"는 자사 매장 주소와 일치해서 제외되고, 1건만 남아야 한다.
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["bizesNm"], "루트커피(T1)")
        self.assertEqual(result["상태"], "배치 검수 완료 — 관리자 승인 대기")

    def test_approve_proceeds_to_finalize_and_pushes_to_backend(self):
        push_bulk_mock, openai_call_mock = self._patch_collectors()
        started = start_sales_target_run(self._initial_state())
        thread_id = started["thread_id"]

        result = continue_sales_target_run(thread_id, approved=True)

        self.assertEqual(result["상태"], "완료 — BE 반영됨")
        self.assertIsNone(result["대기중_승인"])
        # generate_pitch 노드가 후보 1건에 대해 LLM을 정확히 1번 호출했는지, 그 결과가 ranked에
        # 반영됐다가 finalize에서 BE로 넘어가는 push_bulk 페이로드까지 이어지는지 확인한다.
        openai_call_mock.assert_called_once()
        self.assertIn("루트커피(T1)", openai_call_mock.call_args.args[0])
        push_bulk_mock.assert_awaited_once()
        call_args = push_bulk_mock.call_args
        self.assertEqual(call_args.args[0], "test-batch-1")
        items = call_args.args[1]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["businessName"], "루트커피(T1)")
        self.assertEqual(
            items[0]["salesPitch"],
            "테스트 후보를 신규 영업 우선 후보로 추천합니다. 전환 가능성이 80%로 매우 높습니다.",
        )

    def test_reject_routes_to_discard_without_pushing_to_backend(self):
        push_bulk_mock, openai_call_mock = self._patch_collectors()
        started = start_sales_target_run(self._initial_state())
        thread_id = started["thread_id"]

        result = continue_sales_target_run(thread_id, approved=False)

        # review 노드가 먼저 "종료: 반려됨"을 반환하지만, 그 뒤 discard 노드가 이어서 실행되며
        # 상태를 다시 덮어쓴다 — 최종 상태는 discard의 메시지가 맞다.
        self.assertEqual(result["상태"], "종료: 반려되어 BE에 반영하지 않음")
        push_bulk_mock.assert_not_awaited()
        # 반려된 배치는 generate_pitch 노드 자체를 안 타므로 LLM 호출(비용 발생) 자체가 없어야 한다.
        openai_call_mock.assert_not_called()

    def test_read_after_start_reflects_pending_state(self):
        self._patch_collectors()
        started = start_sales_target_run(self._initial_state())

        read = read_sales_target_run(started["thread_id"])

        self.assertIsNotNone(read["대기중_승인"])
        self.assertEqual(read["thread_id"], started["thread_id"])

    def test_google_places_api_key_present_updates_review_score(self):
        # 6단계(리뷰 2단계 반영): google_places_api_key가 있으면 fetch_review_stats()가 호출되고
        # review_score가 NaN이 아닌 실값으로 채워져야 한다.
        self._patch_collectors()
        review_stats = pd.DataFrame({
            "rdnmAdr": ["서울 마포구 성지길 5"],
            "review_count": [80],
            "avg_rating": [4.7],
            "days_since_latest_review": [2.0],
            "review_growth_trend": [-1.0],
        })
        fetch_mock = AsyncMock(return_value=review_stats)
        with patch("app.sales_target.graph.fetch_review_stats", fetch_mock):
            state = self._initial_state()
            state["google_places_api_key"] = "test-google-key"
            result = start_sales_target_run(state)

        fetch_mock.assert_awaited_once()
        candidate = result["대기중_승인"]["후보_리스트"][0]
        # 매칭된 후보 1건뿐이라 percentile은 항상 100 -> review_activity_score = 100.0.
        self.assertAlmostEqual(candidate["review_score"], 100.0)
        self.assertNotIn(
            "GOOGLE_PLACES_API_KEY 미설정 — 리뷰 활성도 점수 없이 성장률·유동인구·유사도 3개 지표로만 최종 점수 산출됨",
            result["대기중_승인"]["주의사항"],
        )

    def test_google_places_api_key_missing_adds_warning_and_leaves_review_score_empty(self):
        self._patch_collectors()

        result = start_sales_target_run(self._initial_state())

        candidate = result["대기중_승인"]["후보_리스트"][0]
        self.assertTrue(math.isnan(candidate["review_score"]))
        self.assertIn(
            "GOOGLE_PLACES_API_KEY 미설정 — 리뷰 활성도 점수 없이 성장률·유동인구·유사도 3개 지표로만 최종 점수 산출됨",
            result["대기중_승인"]["주의사항"],
        )

    def test_openai_key_present_uses_review_matching_agent_instead_of_deterministic_fetch(self):
        # OPENAI_API_KEY도 있으면 collect_review_stats_agentic()이 호출되고, 결정론적
        # fetch_review_stats()는 호출되지 않아야 한다(고도화 설계 3절 — 구현 3).
        self._patch_collectors()
        review_stats = pd.DataFrame({
            "rdnmAdr": ["서울 마포구 성지길 5"],
            "review_count": [80], "avg_rating": [4.7],
            "days_since_latest_review": [2.0], "review_growth_trend": [-1.0],
        })
        agentic_mock = AsyncMock(return_value=review_stats)
        deterministic_mock = AsyncMock()
        with patch("app.sales_target.graph.collect_review_stats_agentic", agentic_mock), \
             patch("app.sales_target.graph.fetch_review_stats", deterministic_mock), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            state = self._initial_state()
            state["google_places_api_key"] = "test-google-key"
            result = start_sales_target_run(state)

        agentic_mock.assert_awaited_once()
        deterministic_mock.assert_not_awaited()
        candidate = result["대기중_승인"]["후보_리스트"][0]
        self.assertAlmostEqual(candidate["review_score"], 100.0)

    def test_critic_agent_summary_and_flags_appear_in_interrupt_payload(self):
        self._patch_collectors()
        review_batch_mock = MagicMock(return_value={
            "summary": "테스트 요약문",
            "flagged": [{"bizesNm": "루트커피(T1)", "reason": "테스트 사유"}],
        })
        with patch("app.sales_target.graph.review_batch", review_batch_mock):
            result = start_sales_target_run(self._initial_state())

        review_batch_mock.assert_called_once()
        self.assertEqual(result["대기중_승인"]["에이전트_요약"], "테스트 요약문")
        self.assertEqual(result["대기중_승인"]["주목_후보"], [{"bizesNm": "루트커피(T1)", "reason": "테스트 사유"}])

    def test_rejected_addresses_are_excluded_from_candidates(self):
        # 5단계(피드백 루프): "루트커피(T1)"의 주소가 이미 EXCLUDED 처리됐다고 BE가 응답하면,
        # 자사 매장 제외와 마찬가지로 후보 목록에서 빠지고 최종적으로 후보가 0건이어야 한다.
        self._patch_collectors(rejected_addresses=["서울 마포구 성지길 5"])

        result = start_sales_target_run(self._initial_state())

        self.assertIsNone(result["대기중_승인"])
        self.assertEqual(result["상태"], "종료: 후보 없음")

    def test_empty_candidates_ends_without_review(self):
        self._patch_collectors()
        state = self._initial_state()
        # 후보가 전부 자사 매장과 겹치면(=빈 결과) review 없이 바로 종료돼야 한다.
        with patch(
            "scripts.collection.store_registry_collector.StoreRegistryCollector"
        ) as store_registry_cls:
            store_registry_cls.return_value.fetch_sigungus = AsyncMock(
                return_value=pd.DataFrame(columns=["bizesNm", "rdnmAdr", "lon", "lat"])
            )
            result = start_sales_target_run(state)

        self.assertIsNone(result["대기중_승인"])
        self.assertEqual(result["상태"], "종료: 후보 없음")


try:
    import pgserver

    _PGSERVER_AVAILABLE = True
except ImportError:
    _PGSERVER_AVAILABLE = False


@unittest.skipUnless(
    _PGSERVER_AVAILABLE, "pgserver 미설치 — PostgresSaver 실제 검증 생략(pip install pgserver)"
)
class PostgresCheckpointerTests(SalesTargetGraphTests):
    """_checkpointer()의 Postgres 분기(SALES_TARGET_CHECKPOINT_DB_URL 설정 시)를 SqliteSaver가
    아니라 실제 Postgres에 붙여서 검증한다. `pgserver`는 root 권한 없이 임베디드 postgres를 띄우는
    테스트 전용 패키지(requirements.txt에 "테스트 전용" 주석과 함께 있다 — 프로덕션 코드는
    여전히 langgraph-checkpoint-postgres/psycopg만 있으면 된다).

    이 테스트가 실제로 잡아낸 버그: `PostgresSaver.from_conn_string()`은 `@contextmanager`로 만든
    제너레이터 기반 컨텍스트매니저다(`with Connection.connect(...) as conn: yield cls(conn)`).
    `_checkpointer()`가 `conn_cm.__enter__()`만 부르고 `conn_cm`을 지역변수로만 두면, 함수가
    리턴하는 순간 참조가 없어져 GC가 그 제너레이터를 정리하면서 `finally`(=커넥션 `__exit__`)가
    실행돼 연결이 곧바로 닫힌다. `SqliteSaver(conn)`은 평범한 생성자라 이 문제가 없어서, 위
    `SalesTargetGraphTests`(SqliteSaver 경로만 실행)로는 못 잡는 버그였다 — 실전 Postgres에
    붙여봐야만 드러났다. 수정: `_checkpointer()`가 `conn_cm`을 모듈 전역 변수(`_pg_conn_cm`)에
    담아 GC 대상에서 뺐다.

    부모 클래스(SalesTargetGraphTests)의 `_patch_collectors`/`_initial_state`/fixture들을 그대로
    재사용하되, `setUp`은 SqliteSaver로 강제하는 부모 로직 대신 이 클래스 전용 Postgres 접속
    로직으로 완전히 갈아끼운다(super().setUp() 호출 안 함).
    """

    _server = None
    _pgdata_dir = None

    @classmethod
    def setUpClass(cls):
        cls._pgdata_dir = tempfile.mkdtemp(prefix="test_sales_target_pg_checkpointer_")
        cls._server = pgserver.get_server(cls._pgdata_dir, cleanup_mode="delete")

    @classmethod
    def tearDownClass(cls):
        if cls._server is not None:
            cls._server.cleanup()

    def setUp(self):
        # 부모의 setUp(SqliteSaver + SALES_TARGET_CHECKPOINT_DB_URL="")은 의도적으로 안 부른다.
        url_patcher = patch.object(graph_module, "SALES_TARGET_CHECKPOINT_DB_URL", self._server.get_uri())
        url_patcher.start()
        self.addCleanup(url_patcher.stop)
        self._reconnect()

    def _reconnect(self):
        """체크포인터/그래프 lru_cache를 비우고 완전히 새 PostgresSaver 커넥션을 맺는다 — 별도
        워커 프로세스가 승인 요청을 이어받는 상황을 흉내낸다. 핵심은 같은 커넥션 객체를 재사용해서
        통과하는 게 아니라, Postgres에 실제로 저장된 체크포인트를 새 커넥션이 읽어내야 한다는 것."""
        graph_module._checkpointer.cache_clear()
        graph_module.get_graph.cache_clear()

    def test_review_state_survives_reconnect_and_resumes_via_postgres(self):
        push_bulk_mock, openai_call_mock = self._patch_collectors()

        started = start_sales_target_run(self._initial_state())
        self.assertIsNotNone(started["대기중_승인"])
        thread_id = started["thread_id"]

        checkpointer_before = graph_module._checkpointer()
        self._reconnect()
        checkpointer_after = graph_module._checkpointer()
        self.assertIsNot(checkpointer_before, checkpointer_after)  # 진짜 새 커넥션인지 확인

        reread = read_sales_target_run(thread_id)
        self.assertIsNotNone(reread["대기중_승인"])
        self.assertEqual(reread["대기중_승인"]["후보_리스트"][0]["bizesNm"], "루트커피(T1)")

        self._reconnect()
        result = continue_sales_target_run(thread_id, approved=True)
        self.assertEqual(result["상태"], "완료 — BE 반영됨")
        self.assertIsNone(result["대기중_승인"])
        push_bulk_mock.assert_awaited_once()
        self.assertIsNotNone(push_bulk_mock.call_args.args[1][0]["salesPitch"])

    def test_reject_via_postgres_after_reconnect(self):
        push_bulk_mock, _ = self._patch_collectors()
        started = start_sales_target_run(self._initial_state())
        thread_id = started["thread_id"]

        self._reconnect()
        result = continue_sales_target_run(thread_id, approved=False)

        self.assertEqual(result["상태"], "종료: 반려되어 BE에 반영하지 않음")
        push_bulk_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
