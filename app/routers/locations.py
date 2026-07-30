from __future__ import annotations

import csv

from fastapi import APIRouter

from app.core.config import SOURCE_DATA

router = APIRouter(tags=["지역·상권"])


@router.get("/locations")
def get_locations() -> dict:
    grouped: dict[str, dict] = {}
    with (SOURCE_DATA / "area_coords.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            district_code = row.get("SIGNGU_CD", "")
            district_name = row.get("SIGNGU_CD_NM", "")
            dong_code = row.get("ADSTRD_CD", "")
            dong_name = row.get("ADSTRD_CD_NM", "")
            area_code = row.get("TRDAR_CD", "")
            area_name = row.get("TRDAR_CD_NM", "")
            if not district_code or not dong_code or not area_code:
                continue
            district = grouped.setdefault(district_code, {"code": district_code, "name": district_name, "areas": {}})
            district["areas"].setdefault(f"{dong_code}:{area_code}", {
                "code": area_code, "name": area_name,
                "region_code": dong_code, "region_name": dong_name,
            })
    regions = []
    for district in grouped.values():
        areas = sorted(district.pop("areas").values(), key=lambda item: (item["region_name"], item["name"]))
        regions.append({**district, "areas": areas})
    return {"regions": sorted(regions, key=lambda item: item["name"])}

