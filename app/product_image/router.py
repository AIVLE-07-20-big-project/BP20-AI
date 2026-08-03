from fastapi import APIRouter, UploadFile, Form, HTTPException, File
from fastapi.responses import Response

from .image_pipeline import generate_product_image, CATEGORY_PROMPTS, InvalidCategoryError

router = APIRouter(prefix="/api/v1/product-images", tags=["ProductImage"])


@router.get("/categories")
def get_categories():
    """지원하는 메뉴(카테고리) 목록 조회"""
    return {"categories": list(CATEGORY_PROMPTS.keys())}


@router.post("/generate")
async def generate(
    file: UploadFile = File(..., description="원본 상품 사진 (jpg/png 등)"),
    category: str = Form(..., description="메뉴명. /categories 에서 조회 가능한 값 중 하나"),
):
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
