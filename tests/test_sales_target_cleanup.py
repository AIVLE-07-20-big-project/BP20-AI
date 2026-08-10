# 대상 경로(AI 레포): tests/test_sales_target_cleanup.py
#
# app/sales_target/cleanup.py 검증. 그래프/콜렉터는 전혀 안 거친다 — cleanup.py는 checkpointer의
# list()/delete_thread()만 다루는 순수 로직이라, checkpointer.put()으로 체크포인트를 직접 심어서
# 테스트한다(그래프를 실제로 돌리는 것보다 훨씬 빠르고, 상태 조합을 자유롭게 만들 수 있다).
#
# SqliteSaver와 PostgresSaver 양쪽에서 동일한 테스트 바디를 돌린다 — delete_thread()의 실제 SQL
# 구현이 백엔드마다 다르므로(tests/test_sales_target_graph.py의 PostgresCheckpointerTests와
# 같은 이유), 둘 다 확인해야 안심할 수 있다. 각 테스트 메서드마다 완전히 새 DB를 만든다 — 이
# 테스트들은 find_deletable_threads()가 반환하는 정확한 thread_id 목록을 비교하므로, 이전
# 테스트의 잔여 데이터가 섞이면 안 된다.

import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver

from app.sales_target.cleanup import delete_old_terminal_threads, find_deletable_threads


def _put_checkpoint(checkpointer, thread_id: str, status: str, ts: datetime) -> None:
    checkpoint = empty_checkpoint()
    checkpoint["ts"] = ts.isoformat()
    checkpoint["channel_values"] = {"status": status}
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpointer.put(config, checkpoint, {"source": "loop", "step": 1, "parents": {}}, {})


class _FindDeletableThreadsTestsMixin:
    """서브클래스가 setUp에서 self.checkpointer를 준비한다(SqliteSaver 또는 PostgresSaver)."""

    now = datetime(2026, 8, 3, tzinfo=timezone.utc)

    def test_terminal_and_old_thread_is_deletable(self):
        _put_checkpoint(self.checkpointer, "t-old-done", "완료 — BE 반영됨", self.now - timedelta(days=40))

        result = find_deletable_threads(self.checkpointer, older_than_days=30, now=self.now)

        self.assertEqual(result, ["t-old-done"])

    def test_terminal_but_recent_thread_is_kept(self):
        _put_checkpoint(
            self.checkpointer, "t-recent-done", "종료: 반려되어 BE에 반영하지 않음", self.now - timedelta(days=5)
        )

        result = find_deletable_threads(self.checkpointer, older_than_days=30, now=self.now)

        self.assertEqual(result, [])

    def test_non_terminal_old_thread_is_never_deleted(self):
        # 오래 방치된 승인 대기 thread라도 cleanup은 지우지 않는다 — "N일 경과 시 자동 반려"는
        # BE(SalesTargetBatchService.autoRejectStaleBatches)가 담당하는 별도 정책이다.
        _put_checkpoint(
            self.checkpointer, "t-pending", "스코어링 완료 — 관리자 승인 대기", self.now - timedelta(days=100)
        )

        result = find_deletable_threads(self.checkpointer, older_than_days=30, now=self.now)

        self.assertEqual(result, [])

    def test_only_latest_checkpoint_per_thread_is_considered(self):
        # 같은 thread에 여러 체크포인트가 쌓여도(review 이전/이후) 최신 것만 기준으로 판단해야
        # 한다 — 오래 전 "승인 대기" 체크포인트가 남아있다고 착각해서 지우면 안 되고, 반대로
        # 최근에 끝났는데 예전 체크포인트 때문에 안 지워져도 안 된다.
        _put_checkpoint(
            self.checkpointer, "t-multi", "스코어링 완료 — 관리자 승인 대기", self.now - timedelta(days=100)
        )
        _put_checkpoint(self.checkpointer, "t-multi", "완료 — BE 반영됨", self.now - timedelta(days=40))

        result = find_deletable_threads(self.checkpointer, older_than_days=30, now=self.now)

        self.assertEqual(result, ["t-multi"])

    def test_delete_old_terminal_threads_actually_removes_data(self):
        _put_checkpoint(self.checkpointer, "t-old-done", "완료 — BE 반영됨", self.now - timedelta(days=40))
        _put_checkpoint(self.checkpointer, "t-recent-done", "완료 — BE 반영됨", self.now - timedelta(days=5))

        deleted = delete_old_terminal_threads(self.checkpointer, older_than_days=30, now=self.now)

        self.assertEqual(deleted, ["t-old-done"])
        remaining = {tup.config["configurable"]["thread_id"] for tup in self.checkpointer.list(None)}
        self.assertNotIn("t-old-done", remaining)
        self.assertIn("t-recent-done", remaining)

    def test_negative_older_than_days_raises(self):
        with self.assertRaises(ValueError):
            find_deletable_threads(self.checkpointer, older_than_days=-1)


class SqliteFindDeletableThreadsTests(_FindDeletableThreadsTestsMixin, unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        conn = sqlite3.connect(f"{self._tmpdir.name}/cleanup_test.sqlite3", check_same_thread=False)
        self.checkpointer = SqliteSaver(conn)
        self.checkpointer.setup()


try:
    import pgserver

    _PGSERVER_AVAILABLE = True
except ImportError:
    _PGSERVER_AVAILABLE = False


@unittest.skipUnless(
    _PGSERVER_AVAILABLE, "pgserver 미설치 — PostgresSaver 실제 검증 생략(pip install pgserver)"
)
class PostgresFindDeletableThreadsTests(_FindDeletableThreadsTestsMixin, unittest.TestCase):

    def setUp(self):
        pgdata_dir = tempfile.mkdtemp(prefix="test_sales_target_cleanup_pg_")
        self.addCleanup(shutil.rmtree, pgdata_dir, True)
        server = pgserver.get_server(pgdata_dir, cleanup_mode="delete")
        self.addCleanup(server.cleanup)

        from langgraph.checkpoint.postgres import PostgresSaver

        conn_cm = PostgresSaver.from_conn_string(server.get_uri())
        self.checkpointer = conn_cm.__enter__()
        self.checkpointer.setup()
        self.addCleanup(conn_cm.__exit__, None, None, None)


if __name__ == "__main__":
    unittest.main()
