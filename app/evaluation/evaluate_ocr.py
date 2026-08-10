r"""Evaluate the production receipt OCR API against hand-written ground truth.

Examples (PowerShell, from BP20_AI_repo):
    python app\evaluation\evaluate_ocr.py predict `
      --api-url http://localhost:8000/api/v1/receipts/parse
    python app\evaluation\evaluate_ocr.py evaluate
    python app\evaluation\evaluate_ocr.py run `
      --api-url http://localhost:8000/api/v1/receipts/parse
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = EVALUATION_DIR / "images"
DEFAULT_GROUND_TRUTH_DIR = EVALUATION_DIR / "ground_truth"
DEFAULT_PREDICTIONS_DIR = EVALUATION_DIR / "predictions"
DEFAULT_REPORTS_DIR = EVALUATION_DIR / "reports"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass(frozen=True)
class ItemPair:
    truth_index: int
    prediction_index: int
    name_similarity: float
    score: float


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 최상위 값은 객체여야 합니다: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def name_similarity(left: Any, right: Any) -> float:
    left_normalized = normalize_name(left)
    right_normalized = normalize_name(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def character_text(value: Any, remove_spaces: bool = False) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s", "", text) if remove_spaces else text


def levenshtein_distance(left: str, right: str) -> int:
    """Calculate character insertions, deletions and substitutions."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def integer_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def items_from(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result", payload)
    if isinstance(result, dict):
        items = result.get("items", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def total_amount_from(payload: dict[str, Any]) -> int | None:
    result = payload.get("result", payload)
    return integer_value(result.get("totalAmount")) if isinstance(result, dict) else None


def match_items(
    truth_items: list[dict[str, Any]],
    predicted_items: list[dict[str, Any]],
    match_threshold: float,
) -> list[ItemPair]:
    """Greedily choose one-to-one item matches using name and total price.

    A normal match requires the configured name similarity. A lower name score
    is accepted only when the item total is exactly equal; this helps OCR typos
    without allowing unrelated administrative rows to become true positives.
    """
    candidates: list[ItemPair] = []
    for truth_index, truth in enumerate(truth_items):
        for prediction_index, prediction in enumerate(predicted_items):
            similarity = name_similarity(truth.get("itemName"), prediction.get("itemName"))
            truth_name = normalize_name(truth.get("itemName"))
            prediction_name = normalize_name(prediction.get("itemName"))
            name_contained = bool(truth_name) and truth_name in prediction_name
            truth_total = integer_value(truth.get("totalPrice"))
            prediction_total = integer_value(prediction.get("totalPrice"))
            amount_equal = truth_total is not None and truth_total == prediction_total
            if (
                not name_contained
                and similarity < match_threshold
                and not (similarity >= 0.55 and amount_equal)
            ):
                continue
            effective_name_score = 1.0 if name_contained else similarity
            score = effective_name_score * 0.75 + (0.25 if amount_equal else 0.0)
            candidates.append(ItemPair(truth_index, prediction_index, similarity, score))

    candidates.sort(key=lambda pair: (pair.score, pair.name_similarity), reverse=True)
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[ItemPair] = []
    for pair in candidates:
        if pair.truth_index in matched_truth or pair.prediction_index in matched_predictions:
            continue
        matched_truth.add(pair.truth_index)
        matched_predictions.add(pair.prediction_index)
        matches.append(pair)
    return sorted(matches, key=lambda pair: pair.truth_index)


def build_multipart(image_path: Path) -> tuple[bytes, str]:
    boundary = f"----bp20-ocr-evaluation-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'.encode()
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.extend(image_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def call_ocr_api(image_path: Path, api_url: str, timeout: float) -> tuple[dict[str, Any], float]:
    body, content_type = build_multipart(image_path)
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OCR API HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"OCR API 연결 실패: {error.reason}") from error
    elapsed = time.perf_counter() - started_at

    # Direct AI API returns the payload itself. A wrapped backend response may
    # place it under result/data, so accept those forms as well.
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = payload["data"]
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        raise RuntimeError("OCR API 응답에 result 객체가 없습니다.")
    return payload, elapsed


def find_image(images_dir: Path, image_name: str, stem: str) -> Path | None:
    named_path = images_dir / image_name if image_name else None
    if named_path is not None and named_path.is_file():
        return named_path
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def generate_predictions(args: argparse.Namespace) -> None:
    truth_paths = sorted(args.ground_truth_dir.glob("receipt_*.json"))
    if not truth_paths:
        raise RuntimeError(f"정답 JSON이 없습니다: {args.ground_truth_dir}")
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, str]] = []
    elapsed_values: list[float] = []
    for index, truth_path in enumerate(truth_paths, start=1):
        truth = read_json(truth_path)
        image_path = find_image(args.images_dir, str(truth.get("image", "")), truth_path.stem)
        if image_path is None:
            failures.append({"receipt": truth_path.stem, "error": "평가 이미지를 찾을 수 없습니다."})
            print(f"[{index}/{len(truth_paths)}] {truth_path.stem}: 이미지 없음")
            continue
        try:
            prediction, elapsed = call_ocr_api(image_path, args.api_url, args.timeout)
            prediction["evaluationMeta"] = {
                "image": image_path.name,
                "elapsedSeconds": round(elapsed, 4),
            }
            write_json(args.predictions_dir / f"{truth_path.stem}.json", prediction)
            elapsed_values.append(elapsed)
            print(f"[{index}/{len(truth_paths)}] {truth_path.stem}: {elapsed:.2f}초")
        except Exception as error:  # noqa: BLE001 - batch should report all failures
            failures.append({"receipt": truth_path.stem, "error": str(error)})
            print(f"[{index}/{len(truth_paths)}] {truth_path.stem}: 실패 - {error}")

    summary = {
        "requestedCount": len(truth_paths),
        "successCount": len(elapsed_values),
        "failureCount": len(failures),
        "averageSeconds": round(statistics.mean(elapsed_values), 4) if elapsed_values else None,
        "failures": failures,
    }
    write_json(args.predictions_dir / "prediction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)}개 이미지의 예측 생성에 실패했습니다.")


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_predictions(args: argparse.Namespace) -> dict[str, Any]:
    truth_paths = sorted(args.ground_truth_dir.glob("receipt_*.json"))
    if not truth_paths:
        raise RuntimeError(f"정답 JSON이 없습니다: {args.ground_truth_dir}")

    aggregate = {
        "truth": 0, "predicted": 0, "matched": 0, "nameExact": 0,
        "nameNormalized": 0, "nameContained": 0, "nameFuzzy": 0,
        "nameAccepted": 0, "quantity": 0,
        "unitPrice": 0, "totalPrice": 0, "allFields": 0,
        "receiptPerfect": 0, "totalAmountCorrect": 0,
        "charEditsWithSpaces": 0, "charTruthWithSpaces": 0,
        "charEditsWithoutSpaces": 0, "charTruthWithoutSpaces": 0,
        "matchedCharEditsWithSpaces": 0, "matchedCharTruthWithSpaces": 0,
        "matchedCharEditsWithoutSpaces": 0, "matchedCharTruthWithoutSpaces": 0,
    }
    total_errors: list[int] = []
    elapsed_values: list[float] = []
    details: list[dict[str, Any]] = []
    missing_predictions: list[str] = []

    for truth_path in truth_paths:
        prediction_path = args.predictions_dir / truth_path.name
        if not prediction_path.is_file():
            missing_predictions.append(truth_path.stem)
            continue
        truth = read_json(truth_path)
        prediction = read_json(prediction_path)
        truth_items = items_from(truth)
        predicted_items = items_from(prediction)
        matches = match_items(truth_items, predicted_items, args.match_threshold)
        matched_truth = {pair.truth_index for pair in matches}
        matched_predictions = {pair.prediction_index for pair in matches}

        aggregate["truth"] += len(truth_items)
        aggregate["predicted"] += len(predicted_items)
        aggregate["matched"] += len(matches)
        matched_details = []
        receipt_all_fields = len(truth_items) == len(predicted_items) == len(matches)

        for pair in matches:
            expected = truth_items[pair.truth_index]
            actual = predicted_items[pair.prediction_index]
            exact_name = str(expected.get("itemName", "")).strip() == str(actual.get("itemName", "")).strip()
            expected_normalized_name = normalize_name(expected.get("itemName"))
            actual_normalized_name = normalize_name(actual.get("itemName"))
            normalized_name = expected_normalized_name == actual_normalized_name
            contained_name = bool(expected_normalized_name) and expected_normalized_name in actual_normalized_name
            fuzzy_name = pair.name_similarity >= args.fuzzy_threshold
            # 한두 글자의 OCR 오타는 동일 상품을 검출한 것으로 인정한다.
            # 완전 일치/포함 여부는 진단용 세부 지표로 별도 유지한다.
            accepted_name = contained_name or fuzzy_name
            quantity_equal = integer_value(expected.get("quantity")) == integer_value(actual.get("quantity"))
            unit_price_equal = integer_value(expected.get("unitPrice")) == integer_value(actual.get("unitPrice"))
            total_price_equal = integer_value(expected.get("totalPrice")) == integer_value(actual.get("totalPrice"))
            all_fields_equal = accepted_name and quantity_equal and unit_price_equal and total_price_equal

            expected_with_spaces = character_text(expected.get("itemName"))
            actual_with_spaces = character_text(actual.get("itemName"))
            expected_without_spaces = character_text(expected.get("itemName"), remove_spaces=True)
            actual_without_spaces = character_text(actual.get("itemName"), remove_spaces=True)
            aggregate["charEditsWithSpaces"] += levenshtein_distance(
                expected_with_spaces, actual_with_spaces
            )
            aggregate["charTruthWithSpaces"] += len(expected_with_spaces)
            aggregate["charEditsWithoutSpaces"] += levenshtein_distance(
                expected_without_spaces, actual_without_spaces
            )
            aggregate["charTruthWithoutSpaces"] += len(expected_without_spaces)
            aggregate["matchedCharEditsWithSpaces"] += levenshtein_distance(
                expected_with_spaces, actual_with_spaces
            )
            aggregate["matchedCharTruthWithSpaces"] += len(expected_with_spaces)
            aggregate["matchedCharEditsWithoutSpaces"] += levenshtein_distance(
                expected_without_spaces, actual_without_spaces
            )
            aggregate["matchedCharTruthWithoutSpaces"] += len(expected_without_spaces)

            aggregate["nameExact"] += int(exact_name)
            aggregate["nameNormalized"] += int(normalized_name)
            aggregate["nameContained"] += int(contained_name)
            aggregate["nameFuzzy"] += int(fuzzy_name)
            aggregate["nameAccepted"] += int(accepted_name)
            aggregate["quantity"] += int(quantity_equal)
            aggregate["unitPrice"] += int(unit_price_equal)
            aggregate["totalPrice"] += int(total_price_equal)
            aggregate["allFields"] += int(all_fields_equal)
            receipt_all_fields = receipt_all_fields and all_fields_equal
            matched_details.append({
                "expected": expected,
                "predicted": actual,
                "nameSimilarity": round(pair.name_similarity, 4),
                "nameContained": contained_name,
                "nameAccepted": accepted_name,
                "characterErrorsWithSpaces": levenshtein_distance(
                    expected_with_spaces, actual_with_spaces
                ),
                "characterErrorsWithoutSpaces": levenshtein_distance(
                    expected_without_spaces, actual_without_spaces
                ),
                "allFieldsMatch": all_fields_equal,
            })

        # 누락 상품명은 전체가 삭제된 것으로, 오탐 상품명은 전체가 삽입된 것으로 계산한다.
        for index, item in enumerate(truth_items):
            if index in matched_truth:
                continue
            with_spaces = character_text(item.get("itemName"))
            without_spaces = character_text(item.get("itemName"), remove_spaces=True)
            aggregate["charEditsWithSpaces"] += len(with_spaces)
            aggregate["charTruthWithSpaces"] += len(with_spaces)
            aggregate["charEditsWithoutSpaces"] += len(without_spaces)
            aggregate["charTruthWithoutSpaces"] += len(without_spaces)
        for index, item in enumerate(predicted_items):
            if index in matched_predictions:
                continue
            aggregate["charEditsWithSpaces"] += len(character_text(item.get("itemName")))
            aggregate["charEditsWithoutSpaces"] += len(
                character_text(item.get("itemName"), remove_spaces=True)
            )

        aggregate["receiptPerfect"] += int(receipt_all_fields)
        truth_total = total_amount_from(truth)
        prediction_total = total_amount_from(prediction)
        total_correct = truth_total is not None and truth_total == prediction_total
        aggregate["totalAmountCorrect"] += int(total_correct)
        if truth_total is not None and prediction_total is not None:
            total_errors.append(abs(truth_total - prediction_total))

        elapsed = prediction.get("evaluationMeta", {}).get("elapsedSeconds")
        if isinstance(elapsed, (int, float)):
            elapsed_values.append(float(elapsed))

        details.append({
            "receipt": truth_path.stem,
            "itemCounts": {
                "groundTruth": len(truth_items),
                "predicted": len(predicted_items),
                "matched": len(matches),
            },
            "matchedItems": matched_details,
            "missingItems": [item for index, item in enumerate(truth_items) if index not in matched_truth],
            "falsePositiveItems": [
                item for index, item in enumerate(predicted_items) if index not in matched_predictions
            ],
            "receiptPerfectMatch": receipt_all_fields,
            "totalAmount": {
                "expected": truth_total,
                "predicted": prediction_total,
                "correct": total_correct,
                "absoluteError": (
                    abs(truth_total - prediction_total)
                    if truth_total is not None and prediction_total is not None else None
                ),
            },
        })

    evaluated_count = len(details)
    if evaluated_count == 0:
        raise RuntimeError("평가 가능한 정답/예측 JSON 쌍이 없습니다. predict를 먼저 실행하세요.")

    precision = safe_ratio(aggregate["matched"], aggregate["predicted"])
    recall = safe_ratio(aggregate["matched"], aggregate["truth"])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_count = aggregate["matched"]
    cer_with_spaces = safe_ratio(
        aggregate["charEditsWithSpaces"], aggregate["charTruthWithSpaces"]
    )
    cer_without_spaces = safe_ratio(
        aggregate["charEditsWithoutSpaces"], aggregate["charTruthWithoutSpaces"]
    )
    matched_cer_with_spaces = safe_ratio(
        aggregate["matchedCharEditsWithSpaces"], aggregate["matchedCharTruthWithSpaces"]
    )
    matched_cer_without_spaces = safe_ratio(
        aggregate["matchedCharEditsWithoutSpaces"], aggregate["matchedCharTruthWithoutSpaces"]
    )
    sorted_elapsed = sorted(elapsed_values)
    p95_index = max(0, min(len(sorted_elapsed) - 1, int(len(sorted_elapsed) * 0.95 + 0.9999) - 1))

    metrics = {
        "receiptCount": evaluated_count,
        "missingPredictions": missing_predictions,
        "configuration": {
            "itemMatchThreshold": args.match_threshold,
            "fuzzyNameThreshold": args.fuzzy_threshold,
        },
        "items": {
            "groundTruthCount": aggregate["truth"],
            "predictedCount": aggregate["predicted"],
            "matchedCount": matched_count,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "nameExactAccuracy": round(safe_ratio(aggregate["nameExact"], matched_count), 4),
            "nameNormalizedAccuracy": round(safe_ratio(aggregate["nameNormalized"], matched_count), 4),
            "nameContainmentAccuracy": round(safe_ratio(aggregate["nameContained"], matched_count), 4),
            "nameFuzzyAccuracy": round(safe_ratio(aggregate["nameFuzzy"], matched_count), 4),
            "nameAcceptedAccuracy": round(safe_ratio(aggregate["nameAccepted"], matched_count), 4),
            "quantityAccuracy": round(safe_ratio(aggregate["quantity"], matched_count), 4),
            "unitPriceAccuracy": round(safe_ratio(aggregate["unitPrice"], matched_count), 4),
            "totalPriceAccuracy": round(safe_ratio(aggregate["totalPrice"], matched_count), 4),
            "allFieldsAccuracy": round(safe_ratio(aggregate["allFields"], matched_count), 4),
            "receiptPerfectMatchRate": round(
                safe_ratio(aggregate["receiptPerfect"], evaluated_count), 4
            ),
            "itemNameCharacters": {
                "groundTruthCountWithSpaces": aggregate["charTruthWithSpaces"],
                "editCountWithSpaces": aggregate["charEditsWithSpaces"],
                "cerWithSpaces": round(cer_with_spaces, 4),
                "characterAccuracyWithSpaces": round(max(0.0, 1.0 - cer_with_spaces), 4),
                "groundTruthCountWithoutSpaces": aggregate["charTruthWithoutSpaces"],
                "editCountWithoutSpaces": aggregate["charEditsWithoutSpaces"],
                "cerWithoutSpaces": round(cer_without_spaces, 4),
                "characterAccuracyWithoutSpaces": round(max(0.0, 1.0 - cer_without_spaces), 4),
                "matchedItemsOnly": {
                    "cerWithSpaces": round(matched_cer_with_spaces, 4),
                    "characterAccuracyWithSpaces": round(
                        max(0.0, 1.0 - matched_cer_with_spaces), 4
                    ),
                    "cerWithoutSpaces": round(matched_cer_without_spaces, 4),
                    "characterAccuracyWithoutSpaces": round(
                        max(0.0, 1.0 - matched_cer_without_spaces), 4
                    ),
                },
            },
        },
        "totalAmount": {
            "accuracy": round(safe_ratio(aggregate["totalAmountCorrect"], evaluated_count), 4),
            "mae": round(statistics.mean(total_errors), 2) if total_errors else None,
        },
        "performance": {
            "averageSeconds": round(statistics.mean(elapsed_values), 4) if elapsed_values else None,
            "p95Seconds": round(sorted_elapsed[p95_index], 4) if sorted_elapsed else None,
        },
    }
    write_json(args.reports_dir / "metrics.json", metrics)
    write_json(args.reports_dir / "details.json", {"receipts": details})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"평가 결과: {args.reports_dir / 'metrics.json'}")
    print(f"상세 오류: {args.reports_dir / 'details.json'}")
    return metrics


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BP20 영수증 OCR 성능평가")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="OCR API로 예측 JSON 생성")
    add_common_paths(predict_parser)
    predict_parser.add_argument(
        "--api-url", default="http://localhost:8000/api/v1/receipts/parse"
    )
    predict_parser.add_argument("--timeout", type=float, default=180.0)

    evaluate_parser = subparsers.add_parser("evaluate", help="정답과 예측 비교")
    add_common_paths(evaluate_parser)
    evaluate_parser.add_argument("--match-threshold", type=float, default=0.72)
    evaluate_parser.add_argument("--fuzzy-threshold", type=float, default=0.85)

    run_parser = subparsers.add_parser("run", help="예측 생성 후 바로 평가")
    add_common_paths(run_parser)
    run_parser.add_argument(
        "--api-url", default="http://localhost:8000/api/v1/receipts/parse"
    )
    run_parser.add_argument("--timeout", type=float, default=180.0)
    run_parser.add_argument("--match-threshold", type=float, default=0.72)
    run_parser.add_argument("--fuzzy-threshold", type=float, default=0.85)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command in {"predict", "run"}:
            generate_predictions(args)
        if args.command in {"evaluate", "run"}:
            evaluate_predictions(args)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
