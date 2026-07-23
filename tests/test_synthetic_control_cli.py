"""Synthetic Control CLI의 최소 회귀 검증."""
from io import StringIO
from unittest.mock import patch

from scripts.response_strategy import synthetic_control


def test_cli_baseline_invokes_main_without_name_error():
    expected = {"판정": "판정불가", "사유": "테스트"}
    with (
        patch.object(synthetic_control.sys, "argv", ["synthetic_control.py", "baseline", "1", "A"]),
        patch.object(synthetic_control, "segment_baseline", return_value=expected),
        patch("sys.stdout", new_callable=StringIO) as stdout,
    ):
        synthetic_control.main()

    assert '"판정": "판정불가"' in stdout.getvalue()
