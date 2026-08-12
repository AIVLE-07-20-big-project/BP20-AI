"""PaddleOCR 모델 구성.

Docker 이미지 빌드 시 모델을 미리 내려받는 단계와 런타임 OCR 엔진이
동일한 모델 설정을 사용하도록 이 모듈에서 구성을 한 곳에 관리한다.
"""
from __future__ import annotations


PADDLE_OCR_MODEL_NAMES = (
    "PP-OCRv5_mobile_det",
    "korean_PP-OCRv5_mobile_rec",
)


def create_paddle_ocr():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="korean",
        # CPU 환경에서 사용할 경량 탐지·인식 모델을 명시한다.
        text_detection_model_name=PADDLE_OCR_MODEL_NAMES[0],
        text_recognition_model_name=PADDLE_OCR_MODEL_NAMES[1],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # PaddlePaddle 3.3.x CPU + oneDNN 조합 오류를 방지한다.
        enable_mkldnn=False,
    )

