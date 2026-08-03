import os
import uuid
import base64
import tempfile

from PIL import Image
from rembg import remove
from openai import OpenAI

TARGET_SIZE = (1024, 1024)  # gpt-image-1이 지원하는 정사각형 크기

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


class InvalidCategoryError(Exception):
    pass


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
    temp_dir = tempfile.gettempdir()  # OS에 맞는 임시 폴더 경로 (Windows/Linux/Mac 모두 대응)
    image_path = os.path.join(temp_dir, f"{request_id}_image.png")
    mask_path = os.path.join(temp_dir, f"{request_id}_mask.png")

    original = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = remove(original)

    original_sq = _to_square(original, TARGET_SIZE, fill=(255, 255, 255, 255))
    mask_sq = _to_square(mask, TARGET_SIZE, fill=(0, 0, 0, 0))

    original_sq.save(image_path)
    mask_sq.save(mask_path)

    return image_path, mask_path


def _edit_with_openai(image_path: str, mask_path: str, category: str) -> bytes:
    if category not in CATEGORY_PROMPTS:
        raise InvalidCategoryError(f"지원하지 않는 카테고리입니다: {category}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = CATEGORY_PROMPTS[category]

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


def generate_product_image(image_bytes: bytes, category: str) -> bytes:
    """전체 파이프라인 실행: 원본 이미지 바이트 + 카테고리 -> 생성된 이미지 바이트"""
    image_path, mask_path = _prepare_image_and_mask(image_bytes)
    try:
        return _edit_with_openai(image_path, mask_path, category)
    finally:
        # 요청별 임시 파일은 결과 반환 후 즉시 정리 (동시 요청 간 파일 잔존 방지)
        for path in (image_path, mask_path):
            if os.path.exists(path):
                os.remove(path)
