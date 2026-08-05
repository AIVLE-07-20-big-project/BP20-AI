import os
import uuid
import base64
import tempfile
from typing import Optional

from PIL import Image
from rembg import remove
from openai import OpenAI

TARGET_SIZE = (1024, 1024)  # gpt-image-1이 지원하는 정사각형 크기

# 카테고리별 "기본 프롬프트" 프리셋(참고용/초기값).
# 실제 사용 프롬프트는 요청에 custom_prompt가 오면 그게 우선한다.
CATEGORY_PROMPTS = {
    "아메리카노": "Place this on a warm wooden cafe table. Gentle steam rising from the hot coffee. "
                 "Soft morning light, natural shadow. Photo realistic, professional coffee shop photography.",
    "카페라떼": "Place this on a warm wooden cafe table with soft latte foam visible on top. "
               "Soft natural morning light, natural shadow. Photo realistic, professional coffee shop photography.",
    "카푸치노": "Place this on a warm wooden cafe table, thick milk foam with a light cocoa dusting on top. "
               "Soft cafe lighting, natural shadow. Photo realistic, professional coffee shop photography.",
    "바닐라라떼": "Place this on a warm wooden cafe table, delicate latte foam. "
                "Soft cozy cafe lighting, natural shadow. Photo realistic, professional coffee shop photography.",
    "카라멜마키아토": "Place this on a warm wooden cafe table, caramel drizzle visible on top. "
                    "Soft warm cafe lighting, natural shadow. Photo realistic, professional coffee shop photography.",
    "카페모카": "Place this on a warm wooden cafe table, whipped cream and chocolate drizzle on top. "
               "Soft warm cafe lighting, natural shadow. Photo realistic, professional coffee shop photography.",
    "콜드브루": "Place this on a modern minimal table, ice cubes and condensation droplets on the glass, "
               "rich dark tone. Bright even lighting, natural shadow. Photo realistic, professional product photography.",
    "아이스티": "Place this on a bright summer-themed table, ice cubes and condensation droplets on the glass. "
               "Cool bright tone, soft even lighting, natural shadow. Photo realistic, professional product photography.",
    "레몬에이드": "Place this on a bright fresh summer table, lemon slices and ice visible, condensation on the glass. "
                "Cool vibrant tone, soft even lighting, natural shadow. Photo realistic, professional product photography.",
    "크루아상": "Place this on a rustic wooden plate, a few pastry crumbs scattered nearby. "
               "Warm morning light, bakery atmosphere, natural shadow. Photo realistic, professional bakery photography.",
    "스콘": "Place this on a rustic wooden plate, a few crumbs scattered nearby. "
           "Soft natural light, bakery atmosphere, natural shadow. Photo realistic, professional bakery photography.",
    "허니브레드": "Place this on a warm wooden table, a light honey drizzle glistening on top. "
                "Soft warm lighting, bakery atmosphere, natural shadow. Photo realistic, professional bakery photography.",
    "초코케이크": "Place this on an elegant white plate. Soft clean studio lighting, minimal background, "
                 "natural shadow. Photo realistic, professional dessert photography.",
    "티라미수": "Place this on an elegant white plate, a light dusting of cocoa powder visible on top. "
               "Soft daylight, minimal clean background, natural shadow. Photo realistic, professional dessert photography.",
    "마카롱": "Place this on a soft pastel-colored plate. Soft even daylight, minimal clean background, "
             "natural shadow. Photo realistic, professional dessert photography.",
}

# 프리셋에도 없고 custom_prompt도 없을 때 쓰는 최후 fallback
DEFAULT_PROMPT_TEMPLATE = (
    "Place this {category} on a table suitable for its category. "
    "Professional food and beverage product photography, natural lighting, "
    "natural shadow, photo realistic."
)

# custom_prompt 사용 시에도 사진 품질/스타일이 무너지지 않도록 항상 덧붙이는 접미사
QUALITY_SUFFIX = (
    " Photo realistic, professional product photography, natural lighting, natural shadow."
)


class InvalidCategoryError(Exception):
    pass


def _contains_korean(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def _translate_prompt_to_english(korean_text: str) -> str:
    """gpt-image-1은 영어 프롬프트를 훨씬 더 잘 따르고, 한글 지시문은 스타일/분위기
    묘사가 약하게 반영되거나 무시되는 경우가 많다. 그래서 이미지 생성에 넘기기 전에
    한글 프롬프트를 짧은 영어 지시문으로 변환한다."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You translate a short Korean instruction for AI product-photo "
                    "background editing into a single concise English prompt. "
                    "Keep it under 40 words. Only output the translated prompt, nothing else."
                ),
            },
            {"role": "user", "content": korean_text},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()


def _resolve_prompt(category: str, custom_prompt: Optional[str]) -> str:
    """실제 사용할 프롬프트 결정.
    우선순위: custom_prompt > 카테고리 프리셋 > 카테고리명 기반 기본 템플릿

    custom_prompt에 한글이 섞여 있으면 영어로 변환한 뒤 사용하고,
    사진 품질 관련 문구(QUALITY_SUFFIX)를 항상 덧붙인다.
    """
    if custom_prompt and custom_prompt.strip():
        text = custom_prompt.strip()
        if _contains_korean(text):
            text = _translate_prompt_to_english(text)
        return text + QUALITY_SUFFIX
    if category in CATEGORY_PROMPTS:
        return CATEGORY_PROMPTS[category]
    return DEFAULT_PROMPT_TEMPLATE.format(category=category)


def _to_square(img: Image.Image, size, fill=(0, 0, 0, 0)) -> Image.Image:
    img = img.copy()
    img.thumbnail(size)
    canvas = Image.new("RGBA", size, fill)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)
    return canvas


def _prepare_image_and_mask(image_bytes: bytes) -> tuple:
    import io

    request_id = uuid.uuid4().hex
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, f"{request_id}_image.png")
    mask_path = os.path.join(temp_dir, f"{request_id}_mask.png")

    original = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = remove(original)

    original_sq = _to_square(original, TARGET_SIZE, fill=(255, 255, 255, 255))
    mask_sq = _to_square(mask, TARGET_SIZE, fill=(0, 0, 0, 0))

    original_sq.save(image_path)
    mask_sq.save(mask_path)

    return image_path, mask_path


def _edit_with_openai(
    image_path: str,
    mask_path: str,
    category: str,
    custom_prompt: Optional[str] = None,
) -> bytes:
    if not category or not category.strip():
        raise InvalidCategoryError("카테고리(메뉴명)를 입력해주세요.")

    prompt = _resolve_prompt(category, custom_prompt)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
        result = client.images.edit(
            model="gpt-image-1",
            image=image_file,
            mask=mask_file,
            prompt=prompt,
            size="1024x1024",
            n=1,
        )

    item = result.data[0]
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    elif getattr(item, "url", None):
        import requests
        return requests.get(item.url, timeout=60).content
    else:
        raise RuntimeError(f"OpenAI 응답에서 이미지 데이터를 찾지 못했습니다: {item}")


def generate_product_image(
    image_bytes: bytes,
    category: str,
    custom_prompt: Optional[str] = None,
) -> bytes:
    """전체 파이프라인 실행: 원본 이미지 바이트 + 카테고리(+선택적 커스텀 프롬프트) -> 생성된 이미지 바이트"""
    image_path, mask_path = _prepare_image_and_mask(image_bytes)
    try:
        return _edit_with_openai(image_path, mask_path, category, custom_prompt)
    finally:
        for path in (image_path, mask_path):
            if os.path.exists(path):
                os.remove(path)