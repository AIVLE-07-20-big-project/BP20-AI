# FastAPI + Celery 워커/비트가 공유하는 이미지(BE compose가 command로 프로세스를 구분)
FROM python:3.11-slim

WORKDIR /app

# OpenCV/PaddleOCR 런타임 의존 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# torch 기본 wheel은 CUDA 번들이라 무거움 — CPU 전용으로 고정
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 운영 Task가 시작될 때 PaddleOCR 모델을 다시 다운로드하지 않도록
# 런타임과 동일한 모델 구성으로 빌드 단계에서 공식 모델을 캐시에 저장한다.
COPY app/ocr/model_config.py /tmp/paddleocr_model_config.py
RUN PYTHONPATH=/tmp python -c "from paddleocr_model_config import create_paddle_ocr; create_paddle_ocr()" \
    && test -d /root/.paddlex/official_models/PP-OCRv5_mobile_det \
    && test -d /root/.paddlex/official_models/korean_PP-OCRv5_mobile_rec \
    && rm /tmp/paddleocr_model_config.py

COPY app ./app
COPY rag ./rag
COPY scripts ./scripts

# data/ · model/ 은 이미지에 안 담는다 — BE compose가 바인드 마운트로 제공

EXPOSE 8000

# BE compose가 api/worker/beat 컨테이너마다 CONTAINER_ROLE을 지정해야 healthcheck가 맞게 갈라진다.
ENV CONTAINER_ROLE=api
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -m scripts.docker_healthcheck || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
