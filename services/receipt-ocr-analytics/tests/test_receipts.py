from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app

SAMPLE_IMAGE = Path(__file__).resolve().parent.parent / "fake_receipt_gaon_beans_v2.png"


def test_parse_missing_file_returns_422():
    """file 파트를 아예 안 보내면 FastAPI가 자체적으로 422를 반환한다."""
    client = TestClient(app)
    response = client.post("/api/v1/receipts/parse")

    assert response.status_code == 422


def test_parse_invalid_image_returns_422():
    """이미지가 아닌 파일을 보내면(디코딩 실패) 422와 에러 메시지를 반환한다.
    cv2.imread가 PaddleOCR 모델 로딩보다 먼저 실행되므로, 이 테스트는 모델 없이도 빠르게 통과한다.
    """
    client = TestClient(app)
    response = client.post(
        "/api/v1/receipts/parse",
        files={"file": ("not-an-image.txt", b"this is not a valid image", "text/plain")},
    )

    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.integration
def test_parse_valid_receipt_returns_structured_result():
    """실제 PaddleOCR 모델을 로딩해서 샘플 영수증 이미지를 정상적으로 인식하는지 확인한다.
    모델 로딩 때문에 느려서(수십 초~수 분) 기본 실행에서는 제외된다.
    실행: pytest -m integration
    """
    assert SAMPLE_IMAGE.exists(), f"샘플 이미지가 없습니다: {SAMPLE_IMAGE}"

    client = TestClient(app)
    with open(SAMPLE_IMAGE, "rb") as f:
        response = client.post(
            "/api/v1/receipts/parse",
            files={"file": ("receipt.png", f, "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    result = body["result"]

    assert result["totalAmount"] > 0
    assert isinstance(result["items"], list)
