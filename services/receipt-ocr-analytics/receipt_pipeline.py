"""
영수증 OCR 자동 정산 파이프라인 (로컬 VSCode 실행용)

Colab 전용 코드(google.colab.files 업로드, !pip 등)를 제거하고
로컬 환경에 맞게 정리한 버전입니다.

사용법:
    python receipt_pipeline.py --image ./20220710_092028.jpg
"""

import datetime
import json
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------
# 0. 공통 유틸
# ------------------------------------------------------------------
def setup_korean_font():
    """
    ocr_result.png 시각화에 한글 라벨이 네모(□)로 깨지는 걸 방지.
    OS별로 기본 내장된 한글 폰트를 찾아서 matplotlib에 등록한다.
    """
    import matplotlib.font_manager as fm

    candidates = [
        "Malgun Gothic",   # Windows 기본 내장
        "AppleGothic",     # macOS 기본 내장
        "NanumGothic",     # Linux (설치돼 있는 경우)
        "Noto Sans CJK KR",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name

    print(
        "⚠️ 한글 폰트를 찾지 못해 시각화 라벨의 한글이 깨질 수 있습니다. "
        "(콘솔에 출력되는 텍스트 결과 자체는 정상입니다)"
    )
    return None


# ------------------------------------------------------------------
# 1. 텍스트 추출 (PaddleOCR)
# ------------------------------------------------------------------
def enhance_receipt_image(original_img: np.ndarray) -> np.ndarray:
    """
    영수증 인식률을 높이기 위한 전처리 파이프라인.

    이 영수증은 닷매트릭스(도트 프린터) 폰트라 글자가 점(dot)들의 조합으로
    끊어져 있습니다. adaptiveThreshold 같은 강한 이진화는 오히려 점 사이
    간격을 벌려 글자를 더 깨뜨리므로 사용하지 않고, 대신:
    1) 업스케일 (작은 글씨를 더 크게, 점들 사이 여유 공간 확보)
    2) 그레이스케일 변환
    3) 노이즈 제거
    4) 대비 향상 (CLAHE)
    5) 모폴로지 Closing (점들을 이어붙여 글자 획을 매끄럽게 복원)
    6) Otsu 전역 이진화 (배경이 균일한 흰색이라 적응형보다 안전)
    반환값은 3채널(BGR)로 다시 변환해서 PaddleOCR에 그대로 넣을 수 있게 함.
    """
    h, w = original_img.shape[:2]

    # 1) 글자가 너무 작으면(가로 1800px 미만) 업스케일 - 점들 사이 여유를 더 줌
    if w < 1800:
        scale = 1800 / w
        original_img = cv2.resize(
            original_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    # 2) 그레이스케일
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 3) 노이즈 제거 (감열지 특유의 잡티 완화, 너무 세게 하면 점이 지워지므로 약하게)
    denoised = cv2.fastNlMeansDenoising(gray, h=7)

    # 4) 대비 향상 (CLAHE: 지역별 히스토그램 평활화 - 그림자/불균일 조명에 강함)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)

    # 5) 모폴로지 Closing: 닷매트릭스 특유의 점 간격을 메꿔서 획을 연결
    #    커널을 너무 크게 하면 글자끼리 뭉치니 작게(2~3) 유지
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(contrasted, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) Otsu 전역 이진화 (영수증 배경이 균일한 흰색이라 블록 단위 적응형보다 안전)
    _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # PaddleOCR도 3채널 입력을 기대하므로 다시 BGR로 변환
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def enhance_receipt_image_light(original_img: np.ndarray) -> np.ndarray:
    """
    가벼운 전처리 버전. 이진화/모폴로지 없이 업스케일 + 약한 노이즈 제거 + 대비 향상만 적용.
    큰 숫자/금액처럼 원본 그대로도 잘 인식되던 영역을 위한 보조 패스.
    """
    h, w = original_img.shape[:2]
    if w < 1800:
        scale = 1800 / w
        original_img = cv2.resize(
            original_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)

    return cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)


def _bbox_to_xyxy(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def merge_ocr_results(results_a, results_b, iou_threshold=0.3):
    """
    두 번의 OCR 패스 결과(서로 다른 전처리)를 겹치는 영역 기준으로 병합.
    같은 위치를 가리키는 결과가 둘 다 있으면 confidence가 더 높은 쪽을 채택하고,
    한쪽에만 있는 결과는 그대로 포함시킨다.
    """
    used_b = set()
    merged = []

    for bbox_a, text_a, prob_a in results_a:
        box_a = _bbox_to_xyxy(bbox_a)
        best_match, best_iou = None, 0.0

        for j, (bbox_b, text_b, prob_b) in enumerate(results_b):
            if j in used_b:
                continue
            box_b = _bbox_to_xyxy(bbox_b)
            iou = _iou(box_a, box_b)
            if iou > best_iou:
                best_iou, best_match = iou, j

        if best_match is not None and best_iou >= iou_threshold:
            used_b.add(best_match)
            bbox_b, text_b, prob_b = results_b[best_match]
            # confidence 더 높은 쪽 채택
            if prob_b > prob_a:
                merged.append((bbox_b, text_b, prob_b))
            else:
                merged.append((bbox_a, text_a, prob_a))
        else:
            merged.append((bbox_a, text_a, prob_a))

    # results_b 중 어느 것과도 매칭 안 된 것들(예: light 버전에서만 잡힌 영역) 추가
    for j, (bbox_b, text_b, prob_b) in enumerate(results_b):
        if j not in used_b:
            merged.append((bbox_b, text_b, prob_b))

    return merged


def run_paddle_ocr(ocr_engine, img_rgb: np.ndarray):
    """
    PaddleOCR predict() 결과를 (bbox, text, prob) 튜플 리스트로 변환.
    bbox는 [[x,y],[x,y],[x,y],[x,y]] 형태(좌상단부터 시계방향)로 통일해서
    기존 merge_ocr_results / 시각화 코드를 그대로 재사용할 수 있게 한다.
    """
    results = []
    output = ocr_engine.predict(img_rgb)
    for res in output:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        polys = res.get("rec_polys", [])
        for bbox, text, prob in zip(polys, texts, scores):
            bbox_list = [[int(p[0]), int(p[1])] for p in bbox]
            results.append((bbox_list, text, float(prob)))
    return results


# PaddleOCR 엔진은 생성(모델 로딩) 비용이 커서, 요청마다 새로 만들지 않고
# 프로세스 안에서 한 번만 만들어 재사용한다 (모듈 레벨 캐시).
_OCR_ENGINE = None


def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from paddleocr import PaddleOCR
        print("PaddleOCR 모델 로딩 중 (최초 1회만 - 이후 요청부터는 재사용됩니다)...")
        _OCR_ENGINE = PaddleOCR(
            lang="korean",
            # 기본값은 PP-OCRv5_server_det(서버/GPU 환경용 고성능·고부하 모델)라
            # CPU 노트북에서 화면 버퍼링/끊김이 심하게 발생할 수 있음.
            # 모바일용 경량 모델로 명시해서 CPU 부하와 처리 시간을 크게 낮춘다.
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,  # 촬영 각도가 심하게 틀어진 경우 True로
            use_doc_unwarping=False,             # 종이가 휘어져 찍힌 경우 True로
            use_textline_orientation=False,      # 텍스트 줄이 세로/회전인 경우 True로
            enable_mkldnn=False,  # PaddlePaddle 3.3.x CPU + oneDNN 조합의 알려진 버그
                                  # (ConvertPirAttribute2RuntimeAttribute NotImplementedError) 회피
        )
    return _OCR_ENGINE


def preprocess_and_ocr(img_path: str):
    setup_korean_font()

    original_img = cv2.imread(img_path)
    if original_img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

    ocr_engine = _get_ocr_engine()

    # 패스 1: 가벼운 전처리 (큰 글자/숫자에 유리)
    light_img = enhance_receipt_image_light(original_img)
    rgb_light = cv2.cvtColor(light_img, cv2.COLOR_BGR2RGB)
    print("텍스트 추출 시작 (1/2: 원본에 가까운 버전)...")
    results_light = run_paddle_ocr(ocr_engine, rgb_light)

    # 패스 2: 강한 전처리 (닷매트릭스 작은 글자 복원에 유리)
    heavy_img = enhance_receipt_image(original_img)
    rgb_heavy = cv2.cvtColor(heavy_img, cv2.COLOR_BGR2RGB)
    print("텍스트 추출 시작 (2/2: 점 연결 보정 버전)...")
    results_heavy = run_paddle_ocr(ocr_engine, rgb_heavy)

    # 두 결과 병합 (겹치는 영역은 confidence 높은 쪽 채택)
    ocr_results = merge_ocr_results(results_light, results_heavy)

    # 시각화는 가벼운 버전(원본에 가까운 화질) 위에 그린다
    rgb_img = rgb_light

    plt.figure(figsize=(12, 12))
    plt.imshow(rgb_img)
    ax = plt.gca()

    extracted_texts = []

    for (bbox, text, prob) in ocr_results:
        if prob > 0.2:  # 임계값을 0.3 -> 0.2로 낮춰서 더 많이 잡되, 결과에서 prob도 같이 확인
            extracted_texts.append(text)

            top_left = [int(val) for val in bbox[0]]
            bottom_right = [int(val) for val in bbox[2]]

            rect = plt.Rectangle(
                (top_left[0], top_left[1]),
                bottom_right[0] - top_left[0],
                bottom_right[1] - top_left[1],
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)
            plt.text(
                top_left[0],
                top_left[1] - 5,
                f"{text} ({prob:.2f})",
                bbox=dict(facecolor="yellow", alpha=0.5),
                fontsize=9,
            )

    plt.axis("off")
    plt.savefig("ocr_result.png", bbox_inches="tight", dpi=150)
    print("OCR 시각화 결과를 ocr_result.png 로 저장했습니다.")

    image_height = rgb_img.shape[0]
    filtered_results = [(bbox, text, prob) for (bbox, text, prob) in ocr_results if prob > 0.2]

    return extracted_texts, filtered_results, image_height


# ------------------------------------------------------------------
# 2. 좌표 기반 품목(items) 및 상호명(storeName) 추출
# ------------------------------------------------------------------

# 이 키워드가 포함된 행(row)은 품목이 아니라 헤더/합계/결제 정보 등이므로
# 품목 추출 대상에서 제외한다.
NON_ITEM_ROW_KEYWORDS = [
    "합계", "총매출", "부가세", "공급가액", "면세금액", "과세금액",
    "결제금액", "받은금액", "거스름", "카드", "승인", "영수증",
    "사업자", "대표", "전화", "주소", "제품명", "상품명", "품목",
    "수량", "단가", "금액", "할인",
]

# 행 안에서 "품목명"으로 볼 수 있는 토큰 판별 - 숫자만 있거나 너무 짧으면 품목명 아님
_ITEM_NAME_MIN_LEN = 2


def _bbox_metrics(bbox):
    """bbox([[x,y]*4])에서 x/y 범위, y중심, 높이를 계산."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max, (y_min + y_max) / 2, (y_max - y_min)


def cluster_rows(ocr_results, y_overlap_ratio: float = 0.5):
    """
    (bbox, text, prob) 리스트를 같은 행(row)끼리 묶는다.
    y좌표 범위가 일정 비율 이상 겹치면 같은 행으로 판단 (완전 일치를 요구하면
    촬영 각도가 살짝만 틀어져도 다른 행으로 잘못 나뉘기 때문).
    반환값: 각 행이 x좌표 오름차순으로 정렬된 (bbox, text, prob) 리스트들의 리스트.
    """
    # y중심 기준으로 먼저 정렬해두면 근처 행부터 비교하게 되어 효율적
    items_with_metrics = []
    for bbox, text, prob in ocr_results:
        x_min, y_min, x_max, y_max, y_center, height = _bbox_metrics(bbox)
        items_with_metrics.append(
            {"bbox": bbox, "text": text, "prob": prob,
             "x_min": x_min, "y_min": y_min, "y_max": y_max, "height": height}
        )
    items_with_metrics.sort(key=lambda d: (d["y_min"] + d["y_max"]) / 2)

    rows = []  # 각 원소: {"y_min":.., "y_max":.., "boxes":[...]}
    for item in items_with_metrics:
        assigned = False
        for row in rows:
            overlap = min(row["y_max"], item["y_max"]) - max(row["y_min"], item["y_min"])
            if overlap > 0 and overlap / max(item["height"], 1) > y_overlap_ratio:
                row["boxes"].append(item)
                row["y_min"] = min(row["y_min"], item["y_min"])
                row["y_max"] = max(row["y_max"], item["y_max"])
                assigned = True
                break
        if not assigned:
            rows.append({"y_min": item["y_min"], "y_max": item["y_max"], "boxes": [item]})

    result_rows = []
    for row in rows:
        boxes = sorted(row["boxes"], key=lambda d: d["x_min"])
        result_rows.append([(b["bbox"], b["text"], b["prob"]) for b in boxes])

    # 다시 y순서(위→아래)로 정렬해서 반환
    result_rows.sort(key=lambda row: _bbox_metrics(row[0][0])[4])
    return result_rows


def extract_store_name(ocr_results, image_height: float):
    """
    영수증 상단부(이미지 높이의 상위 15%)에 있는 텍스트 중 상호명 후보를 찾는다.

    1) 숫자로만 이루어진 텍스트(전화번호, 사업자번호, 날짜 등)는 제외
    2) 숫자가 하나라도 섞인 텍스트도 제외
       (주소 "동천동 790-7", 전화번호, 사업자번호, 날짜 등은 보통 숫자가 섞여 있고,
        실제 상호명/지점명에는 거의 숫자가 나오지 않는다는 특징을 이용)
    3) "영수증" 같은 문서 제목 단어는 상호명이 아니라 문서 종류를 나타내는 것이므로 제외
       (OCR이 "영수증"을 "영"/"수증"처럼 잘라서 인식해도 걸러지도록 부분 포함 여부로 검사)
    4) 다른 후보 문자열에 포함되는(부분 문자열인) 후보는 제외
       예: "PARIS BAGUETTE"와 "BAGUETTE"가 둘 다 잡히면, 후자는 전자의 일부일 뿐이므로 제외
    5) 한글이 포함된 후보를 영문 전용 후보보다 우선한다
       (실제 매장명/지점명은 보통 한글로 인쇄되고, 영문은 브랜드 로고인 경우가 많음)
    6) 그 안에서 글자 크기(bbox 높이)가 가장 큰 것을 채택
    """
    top_boundary = image_height * 0.15
    candidates = []

    # 영수증 상단에 흔히 나오는 "문서 제목" 성격의 단어들 (상호명이 아님)
    DOCUMENT_TITLE_WORDS = ["영수증", "영수", "수증", "보관용", "고객용", "거래명세서", "거래명세표", "세금계산서"]

    for bbox, text, prob in ocr_results:
        _, y_min, _, y_max, y_center, height = _bbox_metrics(bbox)
        if y_center > top_boundary:
            continue

        clean = text.strip()
        if len(clean) < _ITEM_NAME_MIN_LEN:
            continue
        # 숫자/기호 위주 텍스트(전화번호, 사업자번호, 날짜 등)는 상호명 후보에서 제외
        has_letter = bool(re.search(r"[가-힣A-Za-z]", clean))
        if not has_letter:
            continue
        # 숫자가 하나라도 섞여있으면 주소/전화번호/사업자번호/날짜일 가능성이 높아 제외
        if re.search(r"\d", clean):
            continue
        # "영수증" 등 문서 제목 단어(조각 포함)는 상호명 후보에서 제외
        no_space = clean.replace(" ", "")
        if any(no_space == w or w in no_space or no_space in w for w in DOCUMENT_TITLE_WORDS):
            continue

        candidates.append((height, clean))

    if not candidates:
        return None

    def _is_substring_of_another(text, others):
        norm = text.replace(" ", "").upper()
        for _, other_text in others:
            if other_text == text:
                continue
            other_norm = other_text.replace(" ", "").upper()
            if norm != other_norm and norm in other_norm:
                return True
        return False

    filtered = [
        (h, t) for h, t in candidates if not _is_substring_of_another(t, candidates)
    ]
    if not filtered:  # 전부 서로의 부분 문자열이라 다 걸러졌다면(이례적), 원래 후보로 폴백
        filtered = candidates

    def _sort_key(item):
        height, text = item
        has_korean = bool(re.search(r"[가-힣]", text))
        return (has_korean, height)

    filtered.sort(key=_sort_key, reverse=True)
    return filtered[0][1]


def extract_items(ocr_results):
    """
    좌표 기반으로 행을 클러스터링한 뒤, 각 행이 "품목 행"인지 판단하고
    품목명/수량/단가/금액을 추출한다.

    현재는 헤더 컬럼 매핑 없이 패턴 기반으로 역할을 추정하는 방식(MVP):
    - 콤마 형식 숫자(예: 1,900) -> 금액(totalPrice) 후보
    - 숫자만 있고 짧은 토큰(1~2자리) -> 수량(quantity) 후보
    - 한글/영문이 섞인 토큰 -> 품목명(itemName) 후보

    품목명이 길어서 줄바꿈된 경우, "품목명만 있고 금액이 없는 행"이 먼저 나오고
    바로 다음 행에 "금액(및 수량)만 있는 행"이 이어지는 패턴이 흔하다.
    이런 경우 두 행을 하나의 품목으로 병합한다. (단, 바로 다음 행까지만 확인 -
    범위를 넓히면 서로 무관한 텍스트가 잘못 합쳐질 위험이 커지기 때문)
    """
    # 금액은 "24,800원"처럼 숫자 뒤에 '원'이 공백 없이 붙는 경우가 흔해서, 비교 전에 '원'을 떼고 검사한다.
    comma_amount_pattern = re.compile(r"^\d{1,3}(?:,\d{3})+$")
    plain_qty_pattern = re.compile(r"^\d{1,2}$")
    # "10kg", "3개"처럼 수량과 단위가 붙어 나오는 토큰을 수량+단위로 분리 인식
    qty_with_unit_pattern = re.compile(r"^(\d+(?:\.\d+)?)(kg|g|mg|ml|l|L|개|병|장|봉|팩|박스|통|캔|묶음)$")

    rows = cluster_rows(ocr_results)
    items = []
    pending_name_tokens = []  # 이름만 있고 금액이 없던 "직전 행"의 이름 토큰 (바로 다음 행과만 병합)

    for row in rows:
        texts = [text.strip() for _, text, _ in row]
        row_joined = "".join(texts)

        # 헤더/합계/결제 정보 등이 섞인 행은 품목 후보에서 제외하고,
        # 이런 행을 만나면 직전에 대기 중이던 이름 후보도 무효화(다른 섹션으로 넘어간 것이므로)
        if any(kw in row_joined for kw in NON_ITEM_ROW_KEYWORDS):
            pending_name_tokens = []
            continue

        name_tokens = []
        amount_tokens = []
        qty_tokens = []
        unit_tokens = []

        for text in texts:
            clean = text.replace(" ", "")
            amount_candidate = clean[:-1] if clean.endswith("원") else clean
            qty_unit_match = qty_with_unit_pattern.match(clean)

            if comma_amount_pattern.match(amount_candidate):
                amount_tokens.append(int(amount_candidate.replace(",", "")))
            elif qty_unit_match:
                qty_tokens.append(int(float(qty_unit_match.group(1))))
                unit_tokens.append(qty_unit_match.group(2))
            elif plain_qty_pattern.match(clean):
                qty_tokens.append(int(clean))
            elif len(clean) >= _ITEM_NAME_MIN_LEN and re.search(r"[가-힣A-Za-z]", clean):
                name_tokens.append(text)

        if name_tokens and not amount_tokens:
            # 이름만 있는 행 -> 다음 행과 합쳐질 "대기 중" 이름으로 보관하고 다음 행으로
            pending_name_tokens = name_tokens
            continue

        if not amount_tokens:
            # 이름도 금액도 없는 행(장식용 구분선, 빈 줄 등) -> 무시하되,
            # 대기 중이던 이름과는 무관하므로 대기열은 초기화
            pending_name_tokens = []
            continue

        # 이 행에 금액이 있음 -> 직전에 대기 중이던 이름과 병합해서 품목으로 확정
        combined_name_tokens = pending_name_tokens + name_tokens
        pending_name_tokens = []

        if not combined_name_tokens:
            continue  # 이름 후보가 전혀 없으면 품목으로 보기 어려움

        item_name = " ".join(combined_name_tokens)
        total_price = max(amount_tokens)  # 한 행에 금액이 여러 개 잡히면 가장 큰 값을 채택
        quantity = qty_tokens[0] if qty_tokens else 1
        unit = unit_tokens[0] if unit_tokens else None
        unit_price = total_price // quantity if quantity else None

        items.append({
            "itemName": item_name,
            "quantity": quantity,
            "unit": unit,
            "unitPrice": unit_price,
            "totalPrice": total_price,
        })

    return items


# ------------------------------------------------------------------
# 3. 추출 데이터 구조화 (정규식 기반 NLP 파서)
# ------------------------------------------------------------------

# 문서 유형별 판별 키워드.
# 위에서 아래 순서로 검사하며, 더 구체적인/배타적인 유형을 먼저 배치했다.
# (예: "세금계산서"는 "영수증"이라는 단어도 같이 포함하는 경우가 있어서
#  일반 영수증보다 먼저 검사해야 오분류를 막을 수 있음)
DOC_TYPE_KEYWORDS = [
    ("TAX_INVOICE", ["세금계산서", "공급받는자", "공급자등록번호"]),
    ("TRANSACTION_STATEMENT", ["거래명세서", "거래명세표"]),
    ("DELIVERY_SETTLEMENT", ["배달의민족", "쿠팡이츠", "요기요", "배달정산", "정산내역서"]),
    ("BANK_TRANSFER", ["이체확인증", "계좌이체", "출금계좌", "입금계좌"]),
    ("ONLINE_ORDER", ["주문번호", "주문상세", "온라인주문", "주문서"]),
    ("CARD_SALES_SLIP", ["카드매출전표", "매출전표"]),
    ("SIMPLE_RECEIPT", ["간이영수증"]),
    ("RECEIPT", ["영수증", "결제기번호", "승인번호"]),  # 가장 일반적인 유형이라 마지막에 검사
]


def classify_document_type(ocr_text_list) -> str:
    """
    OCR로 뽑힌 텍스트 리스트를 훑어서 문서 유형을 키워드 매칭으로 1차 분류.
    나중에 문서 종류별 데이터가 충분히 쌓이면 분류 모델로 교체 가능하지만,
    지금 단계에서는 키워드 매칭만으로도 충분히 실용적이다.
    """
    full_text = "".join(ocr_text_list).replace(" ", "")

    for doc_type, keywords in DOC_TYPE_KEYWORDS:
        if any(kw in full_text for kw in keywords):
            return doc_type

    return "UNKNOWN"


# 결제수단 판별 키워드. 위에서 아래로 갈수록 "더 일반적인(구체성이 낮은)" 표현이라,
# 구체적인 수단(간편결제 브랜드 등)이 먼저 매칭되면 이후 일반적인 "카드"에 덮어써지지 않도록
# _detect_payment_method에서 우선순위를 관리한다.
PAYMENT_METHOD_KEYWORDS = [
    ("카카오페이", ["카카오페이", "카카오결제", "가가오결제", "가가오페이"]),
    ("삼성페이", ["삼성페이"]),
    ("네이버페이", ["네이버페이"]),
    ("제로페이", ["제로페이"]),
    ("카드", ["카드", "일시불", "할부"]),  # 구체적 브랜드가 안 잡히면 일반 카드 결제로 처리
    ("현금", ["현금영수증", "현금결제"]),
]

# 위 목록에서 "카드"보다 구체적인(우선순위가 높은) 결제수단 이름 집합.
# 한 번 이 중 하나로 확정되면, 이후 텍스트에서 "카드"라는 일반 단어가 나와도 덮어쓰지 않는다.
_SPECIFIC_PAYMENT_METHODS = {"카카오페이", "삼성페이", "네이버페이", "제로페이"}


def _detect_payment_method(text_clean: str, current: str | None) -> str | None:
    """
    텍스트 한 줄을 보고 결제수단을 판별.
    이미 구체적인 간편결제 수단(카카오페이 등)으로 확정된 상태라면,
    이후 나오는 일반적인 "카드"라는 단어에 덮어쓰이지 않게 방지한다.
    """
    for method_name, keywords in PAYMENT_METHOD_KEYWORDS:
        if any(kw in text_clean for kw in keywords):
            if current in _SPECIFIC_PAYMENT_METHODS and method_name == "카드":
                continue  # 이미 더 구체적인 수단으로 확정됨 - 일반 카드로 덮어쓰지 않음
            return method_name
    return current


# "라벨 텍스트" 바로 다음(또는 몇 칸 이내)에 나오는 숫자를 그 라벨의 값으로 보는 방식.
# PaddleOCR 결과가 대체로 읽는 순서(위→아래, 좌→우)로 반환되기 때문에,
# "과세금액" 다음에 "9,091"이 나오는 식의 인접 패턴이 실제로 잘 맞는다.
# (진짜 좌표 기반 매칭은 이후 items[] 추출 작업 때 더 정교하게 구현 예정)
LABEL_FIELD_MAP = {
    "supplyAmount": ["공급가액", "과세금액"],
    "vat": ["부가세", "부가가치세"],
    "taxFreeAmount": ["면세금액"],
    "totalAmount": ["합계금액", "총매출액", "결제금액", "요금총액", "총액"],
}

# 라벨 다음에 오는 값은 "9,091"처럼 콤마가 있을 수도, "909"처럼 없을 수도 있어서
# 콤마 유무 상관없이 순수 숫자 토큰만 매칭한다.
_ADJACENT_NUMBER_PATTERN = re.compile(r"^\d{1,3}(?:,\d{3})*$")


def _find_adjacent_amount(ocr_text_list, label_index: int, window: int = 3):
    """
    라벨이 나온 위치(label_index) 바로 다음 몇 개 텍스트 안에서 숫자를 찾아 반환.
    "248,000원"처럼 숫자 뒤에 "원"이 공백 없이 바로 붙어있는 경우도 인식한다.
    """
    for j in range(label_index + 1, min(label_index + 1 + window, len(ocr_text_list))):
        candidate = ocr_text_list[j].replace(" ", "")
        if candidate.endswith("원"):
            candidate = candidate[:-1]
        if _ADJACENT_NUMBER_PATTERN.fullmatch(candidate):
            return int(candidate.replace(",", ""))
    return None


def native_nlp_parser(ocr_text_list, document_type: str = "RECEIPT"):
    """
    OCR 텍스트 리스트를 목표 스키마 형태로 구조화.

    ⚠️ 현재 구현 상태:
    - storeName: 아직 추출 로직 없음 (좌표 기반 작업 때 구현 예정) → 항상 None
    - items: 아직 품목 단위 추출 로직 없음 (좌표 기반 작업 때 구현 예정) → 항상 빈 리스트
    - businessNumber / transactionDate / transactionTime / paymentMethod / totalAmount: 구현됨
    - supplyAmount / vat: 라벨-인접 숫자 매칭 방식으로 신규 구현 (실패 시 None)
    """
    structured_data = {
        "documentType": document_type,
        "storeName": None,          # TODO: 좌표 기반 상호명 추출 (다음 작업)
        "businessNumber": None,
        "transactionDate": None,
        "transactionTime": None,    # 목표 스키마엔 없지만 유용한 정보라 유지
        "paymentMethod": "현금",
        "items": [],                # TODO: 좌표 기반 품목 추출 (다음 작업)
        "supplyAmount": None,
        "vat": None,
        "taxFreeAmount": None,
        "totalAmount": 0,
    }

    date_pattern = re.compile(r"(\d{4}[-/.]\d{2}[-/.]\d{2})")
    time_pattern = re.compile(r"(\d{2}:\d{2})")
    biz_pattern = re.compile(r"(\d{3}-\d{2}-\d{5})")
    # 전화번호/사업자번호/차량번호 등은 콤마(,) 천단위 구분이 없으므로,
    # "콤마로 구분된 숫자"만 금액 후보로 본다 (예: 16,200 / 1,234,567)
    amount_pattern = re.compile(r"\d{1,3}(?:,\d{3})+")

    candidate_amounts = []  # totalAmount 라벨 매칭 실패 시 쓸 fallback 후보
    detected_method = None

    for idx, text in enumerate(ocr_text_list):
        text_clean = text.replace(" ", "")

        date_match = date_pattern.search(text_clean)
        if date_match and not structured_data["transactionDate"]:
            structured_data["transactionDate"] = date_match.group(1).replace("/", "-")

        time_match = time_pattern.search(text_clean)
        if time_match and not structured_data["transactionTime"]:
            structured_data["transactionTime"] = time_match.group(1)

        biz_match = biz_pattern.search(text_clean)
        if biz_match:
            structured_data["businessNumber"] = biz_match.group(1)

        detected_method = _detect_payment_method(text_clean, detected_method)

        # 라벨(공급가액/부가세/합계금액 등) 바로 다음 숫자를 그 필드 값으로 채택
        for field, labels in LABEL_FIELD_MAP.items():
            if structured_data.get(field) in (None, 0) and any(kw in text_clean for kw in labels):
                amt = _find_adjacent_amount(ocr_text_list, idx)
                if amt is not None:
                    structured_data[field] = amt

        # 콤마 형식 금액은 fallback용으로 계속 수집
        for amt_str in amount_pattern.findall(text_clean):
            candidate_amounts.append(int(amt_str.replace(",", "")))

    if detected_method:
        structured_data["paymentMethod"] = detected_method

    if not structured_data["totalAmount"] and candidate_amounts:
        # 라벨 기반으로 totalAmount를 못 찾았으면, 콤마 형식 금액 중 최댓값으로 대체
        # (거스름돈/할인 등 하위 항목보다 총액이 보통 가장 크다는 가정)
        structured_data["totalAmount"] = max(candidate_amounts)

    return structured_data


# ------------------------------------------------------------------
# 3. 데이터 검증 및 가계부 반영
# ------------------------------------------------------------------
CATEGORY_MAP = {
    "식재료비": ["돼지고기", "양파", "소스", "쌀", "야채", "고기", "원두", "생두", "커피", "우유", "시럽", "밀가루"],
    "포장재비": ["포장", "용기", "비닐봉투", "박스", "테이프", "테이크아웃컵", "냅킨", "컵"],
    "소모품비": ["세제", "수세미", "휴지", "행주"],
    "공과금": ["전기요금", "수도요금", "전기", "수도", "가스요금"],
    "광고비": ["광고", "마케팅", "SNS"],
    "유지비": ["점검", "수리", "머신", "정비"],
    "여비교통비": ["택시", "카카오결제", "운임", "주차"],
}


def validate_and_reflect(structured_data, raw_ocr_texts):
    print("=== 데이터 검증 및 시스템 반영 시작 ===")

    if not structured_data["transactionDate"]:
        structured_data["transactionDate"] = datetime.date.today().strftime("%Y-%m-%d")
        print(f"⚠️ 날짜 인식 누락으로 현재 날짜({structured_data['transactionDate']})로 대체합니다.")

    if not structured_data["storeName"]:
        structured_data["storeName"] = "미분류 상호 (수정 필요)"

    total_amount = structured_data["totalAmount"]
    if total_amount <= 0:
        print("❌ 검증 실패: 총 금액이 0원 이하입니다. 수기 확인이 필요합니다.")
        return structured_data
    print(f"✅ 금액 검증 완료: 총 결제 금액 {total_amount:,}원 확인.")

    # 1) 공급가액 + 부가세 + 면세금액 = 총금액 검증 (필요한 값들이 인식된 경우에만)
    #    면세금액이 없으면 0으로 간주 (면세 품목이 없는 일반적인 경우)
    supply = structured_data["supplyAmount"]
    vat = structured_data["vat"]
    tax_free = structured_data["taxFreeAmount"] or 0
    if supply is not None and vat is not None:
        calculated = supply + vat + tax_free
        if calculated != total_amount:
            print(
                f"⚠️ 금액 검증 불일치: 공급가액({supply:,}) + 부가세({vat:,}) + 면세금액({tax_free:,}) "
                f"= {calculated:,} ≠ 총액({total_amount:,}). OCR 오인식 가능성이 있어 확인이 필요합니다."
            )
        else:
            print("✅ 공급가액 + 부가세 + 면세금액 = 총액 검증 통과.")

    # 2) 품목별 금액 합계 = 총금액 검증 (품목이 추출된 경우에만)
    #    영수증에 표기된 품목 가격은 보통 부가세가 이미 포함된 소비자가이므로,
    #    공급가액이 아니라 총금액(totalAmount)과 비교하는 것이 맞다.
    items = structured_data.get("items") or []
    if items:
        items_sum = sum(item["totalPrice"] for item in items)
        if items_sum != total_amount:
            print(
                f"⚠️ 품목 합계 불일치: 품목별 금액 합계({items_sum:,}) ≠ 총액({total_amount:,}). "
                f"품목 인식이 누락되었거나 잘못됐을 가능성이 있습니다."
            )
        else:
            print("✅ 품목별 금액 합계 = 총액 검증 통과.")

        # 3) 품목별 수량 × 단가 = 금액 검증
        #    현재 unitPrice는 totalPrice // quantity로 역산해서 채워지므로,
        #    나누어떨어지지 않는 경우(나머지가 남는 경우) 수량 인식이 잘못됐을 가능성을 알려준다.
        for item in items:
            quantity = item["quantity"]
            unit_price = item["unitPrice"]
            total_price = item["totalPrice"]
            if quantity and unit_price is not None and quantity * unit_price != total_price:
                print(
                    f"⚠️ 품목 금액 불일치: '{item['itemName']}' "
                    f"수량({quantity}) × 단가({unit_price:,}) = {quantity * unit_price:,} "
                    f"≠ 품목금액({total_price:,})."
                )

    matched_category = "기타운영비"
    full_text_dump = "".join(raw_ocr_texts)
    for category, keywords in CATEGORY_MAP.items():
        if any(kw in full_text_dump for kw in keywords) or any(
            kw in (structured_data["paymentMethod"] or "") for kw in keywords
        ):
            matched_category = category
            break

    structured_data["category"] = matched_category
    print(f"📂 자동 분류된 지출 카테고리: {matched_category}")
    return structured_data


# ------------------------------------------------------------------
# 참고: 이 파일 자체의 CLI(main())는 더 이상 사용하지 않습니다.
# 실제 실행 진입점은 run_full_pipeline.py 입니다.
# (OCR -> CSV 저장 -> 가계부/원가분석까지 한 번에 잇는 스크립트)
# 이 파일은 이제 그 파이프라인이 가져다 쓰는 "함수 라이브러리" 역할만 합니다.
# ------------------------------------------------------------------