"""
AI 상품 이미지 생성 - FastAPI 서비스

백엔드(Spring Boot)가 이 서비스를 HTTP로 호출해서 사용하는 구조.
실행: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os

from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.responses import Response
from dotenv import load_dotenv

from image_pipeline import generate_product_image, CATEGORY_PROMPTS, InvalidCategoryError

load_dotenv()

app = FastAPI(
    title="AI 상품 이미지 생성 서비스",
    description="상품 사진과 메뉴명을 받아, 배경을 자동으로 교체한 상품 이미지를 생성합니다.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """서버 상태 확인용 (백엔드/인프라에서 헬스체크 용도로 호출)"""
    api_key_set = bool(os.environ.get("OPENAI_API_KEY"))
    return {"status": "ok", "openai_key_configured": api_key_set}


@app.get("/api/v1/product-images/categories")
def get_categories():
    """지원하는 메뉴(카테고리) 목록 조회"""
    return {"categories": list(CATEGORY_PROMPTS.keys())}


@app.post("/api/v1/product-images/generate")
async def generate(
    file: UploadFile = File(..., description="원본 상품 사진 (jpg/png 등)"),
    category: str = Form(..., description="메뉴명. /categories 에서 조회 가능한 값 중 하나"),
):
    """
    상품 사진 + 메뉴명을 받아 배경이 합성된 최종 이미지를 반환한다.

    응답: image/png 바이너리 (성공 시)
    """
    if category not in CATEGORY_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 카테고리입니다: '{category}'. "
                   f"지원 목록: {list(CATEGORY_PROMPTS.keys())}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        result_bytes = generate_product_image(image_bytes, category)
    except InvalidCategoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # OpenAI API 오류, rembg 처리 오류 등
        raise HTTPException(status_code=500, detail=f"이미지 생성 중 오류가 발생했습니다: {e}")

    return Response(content=result_bytes, media_type="image/png")
