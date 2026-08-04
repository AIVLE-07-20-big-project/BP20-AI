from typing import Optional

from fastapi import APIRouter, UploadFile, Form, HTTPException, File
from fastapi.responses import Response

from .image_pipeline import generate_product_image, CATEGORY_PROMPTS, InvalidCategoryError

router = APIRouter(prefix="/api/v1/product-images", tags=["ProductImage"])


@router.get("/categories")
def get_categories():
    """기본 제공 카테고리 프리셋 목록(참고용). category는 여기 없는 값도 자유 입력 가능."""
    return {"categories": list(CATEGORY_PROMPTS.keys())}


@router.get("/categories/{category}/prompt")
def get_default_prompt(category: str):
    """특정 카테고리의 기본(프리셋) 프롬프트 조회.
    프론트에서 '프롬프트 수정' 폼을 열 때 초기값 채우는 용도."""
    return {"category": category, "prompt": CATEGORY_PROMPTS.get(category, "")}


@router.post("/generate")
async def generate(
    file: UploadFile = File(..., description="원본 상품 사진 (jpg/png 등)"),
    category: str = Form(..., description="메뉴명 (자유 입력 가능)"),
    prompt: Optional[str] = Form(
        None,
        description="이 메뉴에 사용할 커스텀 프롬프트. 비우면 카테고리 프리셋 또는 기본 템플릿 사용.",
    ),
):
    """
    상품 사진 + 메뉴명(+선택적 커스텀 프롬프트)을 받아 배경이 합성된 최종 이미지를 반환한다.

    응답: image/png 바이너리 (성공 시)
    """
    if not category or not category.strip():
        raise HTTPException(status_code=400, detail="카테고리(메뉴명)를 입력해주세요.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        result_bytes = generate_product_image(image_bytes, category, prompt)
    except InvalidCategoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 생성 중 오류가 발생했습니다: {e}")

    return Response(content=result_bytes, media_type="image/png")