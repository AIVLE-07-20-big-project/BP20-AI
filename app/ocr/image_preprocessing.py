"""Production receipt preprocessing shared by the OCR API.

The implementation is promoted from the experiments in ``preparation``.  It
keeps geometry correction before pixel enhancement and returns one selected
CLAHE + unsharp-mask image to avoid multiple expensive OCR passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ReceiptPreprocessConfig:
    long_side: int = 1800
    max_upscale: float = 1.5
    min_receipt_area_ratio: float = 0.20
    max_skew_degrees: float = 15.0
    denoise_strength: int = 5
    clahe_clip_limit: float = 2.0
    enhancement_passes: int = 2


@dataclass(frozen=True)
class ReceiptPreprocessMetadata:
    receipt_detected: bool
    document_rotation: int
    deskew_angle: float
    original_size: tuple[int, int]
    output_size: tuple[int, int]


def load_receipt_image(path: str | Path) -> np.ndarray:
    """Load a receipt with EXIF orientation applied."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {path}")
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def detect_receipt_corners(
    image: np.ndarray, min_area_ratio: float = 0.20
) -> np.ndarray | None:
    height, width = image.shape[:2]
    scale = min(1.0, 1400.0 / max(height, width))
    work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = work.shape[0] * work.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        if cv2.contourArea(contour) < image_area * min_area_ratio:
            break
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            return _order_points(polygon.reshape(4, 2) / scale)
    return None


def perspective_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = _order_points(corners)
    width = int(round(max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )))
    height = int(round(max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )))
    if width < 2 or height < 2:
        return image.copy()
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(_order_points(corners), destination)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)


def _horizontal_text_score(binary: np.ndarray) -> float:
    kernel_width = max(15, binary.shape[1] // 25)
    joined = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3)),
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(sum(
        width * height
        for x, y, width, height in map(cv2.boundingRect, contours)
        if width > 3 * height
    ))


def correct_document_axis(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Correct a 90-degree axis error; image-only logic cannot resolve 180 degrees."""
    height, width = image.shape[:2]
    # 영수증 영역 검출/원근 보정이 끝난 결과가 세로형이면 이미 정상 축이다.
    # 표의 긴 세로선이나 촘촘한 열 때문에 horizontal score가 잘못 높아져
    # 정상 영수증을 90도로 회전시키는 오판을 방지한다.
    if height >= width:
        return image, 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if max(gray.shape) > 1200:
        ratio = 1200 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    rotated = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
    if _horizontal_text_score(rotated) > _horizontal_text_score(binary) * 1.15:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 90
    return image, 0


def estimate_skew_angle(image: np.ndarray, max_degrees: float = 7.0) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    lines = cv2.HoughLinesP(
        binary, 1, np.pi / 180, 80,
        minLineLength=max(40, image.shape[1] // 5), maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= max_degrees:
            angles.append(float(angle))
    return float(np.median(angles)) if len(angles) >= 2 else 0.0


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def normalize_resolution(
    image: np.ndarray, long_side: int, max_upscale: float
) -> np.ndarray:
    current = max(image.shape[:2])
    scale = min(long_side / current, max_upscale)
    if abs(scale - 1.0) < 0.05:
        return image
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def enhance_clahe_sharpen(
    image: np.ndarray, config: ReceiptPreprocessConfig
) -> np.ndarray:
    if config.enhancement_passes < 1:
        raise ValueError("enhancement_passes는 1 이상이어야 합니다")
    enhanced = image
    for _ in range(config.enhancement_passes):
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=config.denoise_strength)
        clahe = cv2.createCLAHE(config.clahe_clip_limit, (8, 8)).apply(denoised)
        blurred = cv2.GaussianBlur(clahe, (0, 0), 1.0)
        sharpened = cv2.addWeighted(clahe, 1.6, blurred, -0.6, 0)
        enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return enhanced


def preprocess_receipt(
    path: str | Path,
    config: ReceiptPreprocessConfig | None = None,
) -> tuple[np.ndarray, ReceiptPreprocessMetadata]:
    config = config or ReceiptPreprocessConfig()
    image = load_receipt_image(path)
    original_size = (image.shape[1], image.shape[0])

    corners = detect_receipt_corners(image, config.min_receipt_area_ratio)
    receipt_detected = corners is not None
    if corners is not None:
        image = perspective_warp(image, corners)

    image, rotation = correct_document_axis(image)
    angle = estimate_skew_angle(image, config.max_skew_degrees)
    image = deskew(image, angle)
    image = normalize_resolution(image, config.long_side, config.max_upscale)
    image = enhance_clahe_sharpen(image, config)
    metadata = ReceiptPreprocessMetadata(
        receipt_detected=receipt_detected,
        document_rotation=rotation,
        deskew_angle=angle,
        original_size=original_size,
        output_size=(image.shape[1], image.shape[0]),
    )
    return image, metadata
