from pathlib import Path

import pandas as pd
import pytest

from scripts.modeling.detailed_sales_external_analysis import (
    DetailedSalesDataError,
    aggregate_daily_sales,
    analyze_external_factors,
    analyze_internal_drivers,
    build_root_cause_analysis,
    load_external_factors,
    load_noaa_isd_daily,
    load_pos_transactions,
    merge_external_factors,
    validate_aggregation,
)


def _pos_frame() -> pd.DataFrame:
    rows = []
    transaction_id = 1
    for store_id, location in ((1, "A"), (2, "B")):
        for day in pd.date_range("2023-01-01", periods=40):
            for hour in (8, 12):
                qty = 1 + int(day.day % 2)
                price = 3.0 + store_id
                rows.append({
                    "transaction_id": transaction_id,
                    "transaction_date": day.strftime("%d-%m-%Y"),
                    "transaction_time": f"{hour:02d}:00:00",
                    "store_id": store_id,
                    "store_location": location,
                    "product_id": 10 + hour,
                    "transaction_qty": qty,
                    "unit_price": price,
                    "Total_Bill": qty * price,
                    "product_category": "Coffee",
                })
                transaction_id += 1
    return pd.DataFrame(rows)


def test_pos_aggregation_preserves_totals(tmp_path: Path):
    path = tmp_path / "pos.csv"
    _pos_frame().to_csv(path, index=False)

    transactions = load_pos_transactions(path)
    daily = aggregate_daily_sales(transactions)
    validate_aggregation(transactions, daily)

    assert len(daily) == 80
    assert daily["revenue"].sum() == transactions["Total_Bill"].sum()
    assert daily["transaction_count"].sum() == len(transactions)


def test_pos_loader_accepts_korean_iso_date_format(tmp_path: Path):
    frame = _pos_frame()
    frame["transaction_date"] = pd.to_datetime(
        frame["transaction_date"],
        format="%d-%m-%Y",
    ).dt.strftime("%Y-%m-%d")
    path = tmp_path / "korean-pos.csv"
    frame.to_csv(path, index=False)

    transactions = load_pos_transactions(path)

    assert transactions["date"].min() == pd.Timestamp("2023-01-01")


def test_invalid_total_bill_is_rejected(tmp_path: Path):
    frame = _pos_frame()
    frame.loc[0, "Total_Bill"] = 999
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(DetailedSalesDataError, match="Total_Bill"):
        load_pos_transactions(path)


def test_external_factor_join_and_association_analysis(tmp_path: Path):
    pos_path = tmp_path / "pos.csv"
    _pos_frame().to_csv(pos_path, index=False)
    transactions = load_pos_transactions(pos_path)
    daily = aggregate_daily_sales(transactions)

    dates = pd.date_range("2023-01-01", periods=40)
    external = pd.DataFrame({
        "date": dates,
        "temperature": range(40),
        "rain_mm": [10 if day.day % 3 == 0 else 0 for day in dates],
        "source": "fixture",
    })
    external_path = tmp_path / "external.csv"
    external.to_csv(external_path, index=False)

    factors, columns = load_external_factors(external_path)
    merged, rates = merge_external_factors(daily, factors, columns)
    results = analyze_external_factors(merged, columns)

    assert rates == {"temperature": 1.0, "rain_mm": 1.0}
    assert {result["factor"] for result in results} == {"temperature", "rain_mm"}
    assert {result["target"] for result in results} == {
        "revenue", "transaction_count", "average_order_value",
    }
    assert all(result["evidenceType"] == "association" for result in results)


def test_duplicate_external_join_key_is_rejected(tmp_path: Path):
    path = tmp_path / "external.csv"
    pd.DataFrame({
        "date": ["2023-01-01", "2023-01-01"],
        "temperature": [1, 2],
    }).to_csv(path, index=False)

    with pytest.raises(DetailedSalesDataError, match="중복"):
        load_external_factors(path)


def test_noaa_isd_fields_are_scaled_and_converted_to_new_york_date(tmp_path: Path):
    path = tmp_path / "noaa.csv"
    pd.DataFrame([
        {
            "STATION": 72505394728,
            "DATE": "2023-01-02T02:00:00",
            "TMP": "+0057,5",
            "DEW": "-0018,5",
            "WND": "120,5,N,0021,5",
            "VIS": "016093,5,N,5",
            "SLP": "10113,5",
            "AA1": "01,0010,9,5",
            "AA2": None,
            "AA3": None,
        },
        {
            "STATION": 72505394728,
            "DATE": "2023-01-02T15:00:00",
            "TMP": "+0100,5",
            "DEW": "+0020,5",
            "WND": "180,5,N,0030,5",
            "VIS": "010000,5,N,5",
            "SLP": "10100,5",
            "AA1": "01,0005,9,5",
            "AA2": None,
            "AA3": None,
        },
    ]).to_csv(path, index=False)

    daily = load_noaa_isd_daily(path, start_date="2023-01-01", end_date="2023-01-02")

    assert daily["date"].dt.strftime("%Y-%m-%d").tolist() == ["2023-01-01", "2023-01-02"]
    assert daily["temperature_c"].tolist() == [5.7, 10.0]
    assert daily["wind_speed_ms"].tolist() == [2.1, 3.0]
    assert daily["precipitation_mm"].tolist() == [1.0, 0.5]


def test_internal_driver_decomposition_reconciles_revenue_change():
    frame = _pos_frame()
    frame["date"] = pd.to_datetime(frame["transaction_date"], format="%d-%m-%Y")

    result = analyze_internal_drivers(
        frame,
        baseline_start="2023-01-01",
        baseline_end="2023-01-20",
        current_start="2023-01-21",
        current_end="2023-02-09",
    )

    revenue_change = result["summary"]["revenueChange"]
    formula_total = sum(
        item["contributionAmount"] for item in result["revenueFormulaDrivers"]
    )
    for drivers in result["dimensionDrivers"].values():
        assert sum(item["contributionAmount"] for item in drivers) == pytest.approx(
            revenue_change, abs=0.01,
        )
    price_quantity = result["priceQuantityDrivers"]
    assert formula_total == pytest.approx(revenue_change, abs=0.01)
    assert price_quantity["reconciledChange"] == pytest.approx(revenue_change, abs=0.01)


def test_root_cause_narrative_only_uses_reliable_aligned_external_drivers():
    internal = {
        "summary": {"revenueChange": -100, "revenueChangePct": -10},
        "revenueFormulaDrivers": [
            {
                "factor": "transaction_count",
                "contributionAmount": -80,
                "contributionPct": 80,
                "evidenceType": "decomposition",
            },
            {
                "factor": "average_order_value",
                "contributionAmount": -20,
                "contributionPct": 20,
                "evidenceType": "decomposition",
            },
        ],
    }
    external = [
        {
            "factor": "subway_ridership",
            "label": "인근 지하철 이용량",
            "estimatedContributionAmount": -10,
            "estimatedEffectPct": -2,
            "confidence": "medium",
            "direction": "negative",
            "joinRate": 1.0,
        },
        {
            "factor": "event_exposure",
            "label": "인근 행사 노출도",
            "estimatedContributionAmount": -5,
            "estimatedEffectPct": -1,
            "confidence": "low",
            "direction": "negative",
            "joinRate": 0.2,
        },
    ]

    result = build_root_cause_analysis(internal, external, data_warnings=["행사 품질 낮음"])

    assert result["headline"].startswith("거래 건수")
    assert "지하철 이용량" in result["narrative"]
    assert "행사 노출도" not in result["narrative"]
    assert result["excludedExternalDrivers"] == [
        {"factor": "event_exposure", "reason": "data_quality"},
    ]
