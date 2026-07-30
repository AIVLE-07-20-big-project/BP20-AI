from __future__ import annotations

import csv

from fastapi import APIRouter

from app.core.config import SOURCE_DATA

router = APIRouter(tags=["업종"])


@router.get("/industries")
def get_industries() -> dict:
    values = {}
    with (SOURCE_DATA / "sales_estimate.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            code = row.get("SVC_INDUTY_CD", "")
            name = row.get("SVC_INDUTY_CD_NM", "")
            if code and name:
                values[code] = {"code": code, "name": name}
    return {"industries": sorted(values.values(), key=lambda item: item["name"])}

