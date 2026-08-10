"""Conservative catalog-based correction for OCR product names."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable


@dataclass(frozen=True)
class CatalogProduct:
    name: str
    aliases: tuple[str, ...] = ()
    price: int | None = None


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def _decompose_hangul(value: str) -> str:
    choseong = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    jungseong = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
    jongseong = "_ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
    result: list[str] = []
    for character in _normalize(value):
        code = ord(character)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            result.extend((
                choseong[offset // 588],
                jungseong[(offset % 588) // 28],
                jongseong[offset % 28],
            ))
        else:
            result.append(character)
    return "".join(result)


def product_similarity(ocr_name: str, candidate: str) -> float:
    text_score = SequenceMatcher(None, _normalize(ocr_name), _normalize(candidate)).ratio()
    jamo_score = SequenceMatcher(None, _decompose_hangul(ocr_name), _decompose_hangul(candidate)).ratio()
    return 0.35 * text_score + 0.65 * jamo_score


def parse_catalog(payload: Iterable[Any]) -> list[CatalogProduct]:
    products: list[CatalogProduct] = []
    for entry in payload:
        if isinstance(entry, str):
            name, aliases, price = entry.strip(), (), None
        elif isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("productName") or "").strip()
            raw_aliases = entry.get("aliases") or []
            if isinstance(raw_aliases, str):
                raw_aliases = [part.strip() for part in raw_aliases.split("|")]
            aliases = tuple(str(alias).strip() for alias in raw_aliases if str(alias).strip())
            raw_price = entry.get("price")
            price = int(raw_price) if raw_price is not None else None
        else:
            continue
        if name:
            products.append(CatalogProduct(name, aliases, price))
    return products


def _item_price(item: dict[str, Any]) -> int | None:
    for field in ("unitPrice", "totalPrice"):
        value = item.get(field)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def find_catalog_match(
    raw_name: str,
    item_price: int | None,
    catalog: list[CatalogProduct],
    min_score: float = 0.70,
    min_margin: float = 0.08,
) -> tuple[CatalogProduct, float] | None:
    normalized_raw = _normalize(raw_name)
    if not normalized_raw:
        return None
    for product in catalog:
        if normalized_raw in {_normalize(product.name), *(_normalize(alias) for alias in product.aliases)}:
            return product, 1.0

    candidates: list[tuple[float, CatalogProduct]] = []
    for product in catalog:
        names = (product.name, *product.aliases)
        score = max(product_similarity(raw_name, name) for name in names)
        # Price is only a tie-breaker. A matching price cannot rescue a poor name.
        if score >= min_score and item_price is not None and product.price == item_price:
            score = min(1.0, score + 0.05)
        candidates.append((score, product))
    candidates.sort(key=lambda value: value[0], reverse=True)
    if not candidates or candidates[0][0] < min_score:
        return None
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    if candidates[0][0] - second_score < min_margin:
        return None
    return candidates[0][1], candidates[0][0]


def correct_item_names(
    items: list[dict[str, Any]],
    catalog: list[CatalogProduct],
    min_score: float = 0.70,
    min_margin: float = 0.08,
) -> list[dict[str, Any]]:
    """Return API-compatible item dictionaries with corrected ``itemName`` values."""
    if not catalog:
        return items
    corrected = []
    for item in items:
        result = dict(item)
        raw_name = str(item.get("itemName") or "").strip()
        match = find_catalog_match(raw_name, _item_price(item), catalog, min_score, min_margin)
        if match is not None:
            result["itemName"] = match[0].name
        corrected.append(result)
    return corrected
