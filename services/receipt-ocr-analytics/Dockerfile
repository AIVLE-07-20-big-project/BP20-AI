FROM python:3.11-slim

WORKDIR /app

# OpenCV 등이 필요로 하는 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 8000번은 다른 팀 서비스와 충돌 가능성이 있어 8001로 변경.
# 필요하면 PORT 환경변수로 실행 시점에 오버라이드 가능.
ENV PORT=8001
EXPOSE 8001

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
