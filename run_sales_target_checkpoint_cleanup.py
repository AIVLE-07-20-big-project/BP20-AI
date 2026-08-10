# 대상 경로(AI 레포): run_sales_target_checkpoint_cleanup.py (레포 루트, 신규)
#
# app/sales_target/cleanup.py의 정리 로직을 수동으로 즉시 실행하고 싶을 때 쓰는 CLI.
# 실제 운영에서는 app/tasks/sales_target.py의 Celery beat 태스크(매일 새벽 4시)가 자동으로
# 돌지만, 배포 직후 검증하거나 특정 시점에 즉시 한 번 정리하고 싶을 때 이 스크립트를 쓴다.
#
# 실행 방법:
#   python run_sales_target_checkpoint_cleanup.py                  # 기본 30일(또는 환경변수) 기준
#   python run_sales_target_checkpoint_cleanup.py --older-than-days 7
#   python run_sales_target_checkpoint_cleanup.py --dry-run         # 삭제 없이 대상 목록만 확인

from __future__ import annotations

import argparse

from app.core.config import SALES_TARGET_CHECKPOINT_RETENTION_DAYS
from app.sales_target.cleanup import delete_old_terminal_threads, find_deletable_threads
from app.sales_target.graph import get_checkpointer


def main(older_than_days: int, dry_run: bool) -> None:
    checkpointer = get_checkpointer()

    if dry_run:
        thread_ids = find_deletable_threads(checkpointer, older_than_days=older_than_days)
        print(f"[dry-run] 삭제 대상 {len(thread_ids)}건 (실제로 지우지 않음)")
    else:
        thread_ids = delete_old_terminal_threads(checkpointer, older_than_days=older_than_days)
        print(f"삭제 완료: {len(thread_ids)}건")

    for thread_id in thread_ids:
        print(f"  - {thread_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--older-than-days", type=int, default=SALES_TARGET_CHECKPOINT_RETENTION_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.older_than_days, args.dry_run)
