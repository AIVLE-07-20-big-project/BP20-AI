from __future__ import annotations

import html
import json

import pandas as pd
import math

TARGET_AMOUNT = "THSMON_SELNG_AMT"
REGION_COLORS = {"대상 상권": "#1e40af", "동일 상권유형 중앙값": "#f97316", "서울 동종업종 중앙값": "#ec4899"}


def _quarter_label(code) -> str:
    code = int(code)
    return f"{code // 10}.{code % 10}Q"


def _comparison_frames(row: pd.DataFrame, df: pd.DataFrame):
    target_q = int(row["STDR_YYQU_CD"].iloc[0])
    industry = row["SVC_INDUTY_CD"].iloc[0]
    area_type = row["TRDAR_SE_CD"].iloc[0]
    target = df[
        (df["TRDAR_CD"] == row["TRDAR_CD"].iloc[0])
        & (df["SVC_INDUTY_CD"] == industry)
        & (df["STDR_YYQU_CD"] <= target_q)
    ]
    area = df[(df["SVC_INDUTY_CD"] == industry) & (df["TRDAR_SE_CD"] == area_type)
              & (df["STDR_YYQU_CD"] <= target_q)]
    industry_df = df[(df["SVC_INDUTY_CD"] == industry) & (df["STDR_YYQU_CD"] <= target_q)]
    return target, area, industry_df


def _trend_payload(row: pd.DataFrame, df: pd.DataFrame, column: str) -> dict | None:
    if column not in df.columns:
        return None
    target, area, industry = _comparison_frames(row, df)
    quarters = sorted(target["STDR_YYQU_CD"].dropna().astype(int).unique())
    if not quarters:
        return None
    def series(frame, target_only=False):
        grouped = frame.groupby("STDR_YYQU_CD")[column]
        values = grouped.last() if target_only else grouped.median()
        return [float(values[q]) if q in values.index and pd.notna(values[q]) else None for q in quarters]
    return {
        "분기": [_quarter_label(q) for q in quarters],
        "계열": {
            "대상 상권": series(target, True),
            "동일 상권유형 중앙값": series(area),
            "서울 동종업종 중앙값": series(industry),
        },
    }


def _ratio_payload(row: pd.DataFrame, df: pd.DataFrame, labels: dict[str, str], value_kind=None) -> dict | None:
    cols = list(labels)
    if not cols or any(c not in df.columns for c in cols):
        return None
    target, area, industry = _comparison_frames(row, df)
    q = int(row["STDR_YYQU_CD"].iloc[0])
    frames = {
        "대상 상권": target[target["STDR_YYQU_CD"] == q].tail(1),
        "동일 상권유형 중앙값": area[area["STDR_YYQU_CD"] == q],
        "서울 동종업종 중앙값": industry[industry["STDR_YYQU_CD"] == q],
    }
    result, raw_result = {}, {}
    for region, frame in frames.items():
        if frame.empty:
            result[region] = [None] * len(cols)
            raw_result[region] = [None] * len(cols)
            continue
        shares = frame[cols].div(frame[cols].sum(axis=1).replace(0, pd.NA), axis=0)
        values = shares.iloc[0] if region == "대상 상권" else shares.median()
        raw_values = frame[cols].iloc[0] if region == "대상 상권" else frame[cols].median()
        result[region] = [round(float(values[c]), 4) if pd.notna(values[c]) else None for c in cols]
        raw_result[region] = [float(raw_values[c]) if pd.notna(raw_values[c]) else None for c in cols]
    return {"labels": list(labels.values()), "지역별": result, "원시값": raw_result, "값유형": value_kind}


def _qoq(current, previous):
    return (float(current) - float(previous)) / float(previous) if previous not in (None, 0) and pd.notna(previous) else None


def _fmt_money(amount: float | int | None) -> str | None:
    if amount is None or pd.isna(amount):
        return None
    return f"{round(float(amount) / 10000):,} 만원"


def _fmt_pct(value: float | int | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return f"{float(value) * 100:.1f}%"


def build_simple_report(row: pd.DataFrame, df: pd.DataFrame, diag: dict) -> dict:
    sales = float(row[TARGET_AMOUNT].iloc[0]) if TARGET_AMOUNT in row.columns else None
    sales_qoq = float(row["sales_qoq"].iloc[0]) if "sales_qoq" in row.columns and pd.notna(row["sales_qoq"].iloc[0]) else None
    sales_yoy = float(row["sales_yoy"].iloc[0]) if "sales_yoy" in row.columns and pd.notna(row["sales_yoy"].iloc[0]) else None
    traffic = float(row["TOT_FLPOP_CO"].iloc[0]) if "TOT_FLPOP_CO" in row.columns else None

    day_cols = {
        "MON_FLPOP_CO": "월요일",
        "TUES_FLPOP_CO": "화요일",
        "WED_FLPOP_CO": "수요일",
        "THUR_FLPOP_CO": "목요일",
        "FRI_FLPOP_CO": "금요일",
        "SAT_FLPOP_CO": "토요일",
        "SUN_FLPOP_CO": "일요일",
    }
    hour_cols = {
        "TMZON_00_06_FLPOP_CO": "00-06시",
        "TMZON_06_11_FLPOP_CO": "06-11시",
        "TMZON_11_14_FLPOP_CO": "11-14시",
        "TMZON_14_17_FLPOP_CO": "14-17시",
        "TMZON_17_21_FLPOP_CO": "17-21시",
        "TMZON_21_24_FLPOP_CO": "21-24시",
    }

    day_values = {c: float(row[c].iloc[0]) for c in day_cols if c in row.columns and pd.notna(row[c].iloc[0])}
    hour_values = {c: float(row[c].iloc[0]) for c in hour_cols if c in row.columns and pd.notna(row[c].iloc[0])}
    top_day = max(day_values, key=day_values.get) if day_values else None
    top_hour = max(hour_values, key=hour_values.get) if hour_values else None

    same_industry = df[(df["STDR_YYQU_CD"] == row["STDR_YYQU_CD"].iloc[0]) & (df["SVC_INDUTY_CD"] == row["SVC_INDUTY_CD"].iloc[0])]
    same_area_type = df[
        (df["STDR_YYQU_CD"] == row["STDR_YYQU_CD"].iloc[0])
        & (df["SVC_INDUTY_CD"] == row["SVC_INDUTY_CD"].iloc[0])
        & (df["TRDAR_SE_CD"] == row["TRDAR_SE_CD"].iloc[0])
    ]

    industry_sales_mean = float(same_industry[TARGET_AMOUNT].median()) if TARGET_AMOUNT in same_industry.columns and not same_industry.empty else None
    area_sales_mean = float(same_area_type[TARGET_AMOUNT].median()) if TARGET_AMOUNT in same_area_type.columns and not same_area_type.empty else None
    industry_store_mean = float(same_industry["STOR_CO"].median()) if "STOR_CO" in same_industry.columns and not same_industry.empty else None
    area_store_mean = float(same_area_type["STOR_CO"].median()) if "STOR_CO" in same_area_type.columns and not same_area_type.empty else None

    explanation_parts = []
    if sales_yoy is not None:
        explanation_parts.append(f"전년동분기 대비 매출은 {_fmt_pct(sales_yoy)} 변했습니다.")
    if sales_qoq is not None:
        explanation_parts.append(f"전분기 대비 매출은 {_fmt_pct(sales_qoq)} 변했습니다.")
    if top_day and top_hour:
        explanation_parts.append(f"유동인구는 {day_cols[top_day]}과 {hour_cols[top_hour]}에 가장 많이 몰립니다.")

    target_hist, area_hist, industry_hist = _comparison_frames(row, df)
    current_q = int(row["STDR_YYQU_CD"].iloc[0])
    previous_q = (current_q // 10 - 1) * 10 + 4 if current_q % 10 == 1 else current_q - 1
    def region_value(frame, col, q, target_only=False):
        selected = frame[frame["STDR_YYQU_CD"] == q]
        if selected.empty or col not in selected:
            return None
        value = selected[col].iloc[-1] if target_only else selected[col].median()
        return float(value) if pd.notna(value) else None
    comparison_summary = {}
    for region, frame, target_only in [
        ("대상 상권", target_hist, True),
        ("동일 상권유형 중앙값", area_hist, False),
    ]:
        stores_now = region_value(frame, "STOR_CO", current_q, target_only)
        stores_prev = region_value(frame, "STOR_CO", previous_q, target_only)
        sales_now = region_value(frame, TARGET_AMOUNT, current_q, target_only)
        sales_prev = region_value(frame, TARGET_AMOUNT, previous_q, target_only)
        comparison_summary[region] = {
            "업소수": int(stores_now) if stores_now is not None else None,
            "업소수_전분기대비": _qoq(stores_now, stores_prev),
            "월평균매출": sales_now,
            "매출_전분기대비": _qoq(sales_now, sales_prev),
        }

    weekday_sales = {
        "MDWK_SELNG_AMT": "주중", "WKEND_SELNG_AMT": "주말",
    }
    weekday_count = {"MDWK_SELNG_CO": "주중", "WKEND_SELNG_CO": "주말"}
    day_sales = {c: label for c, label in {
        "MON_SELNG_AMT": "월", "TUES_SELNG_AMT": "화", "WED_SELNG_AMT": "수",
        "THUR_SELNG_AMT": "목", "FRI_SELNG_AMT": "금", "SAT_SELNG_AMT": "토",
        "SUN_SELNG_AMT": "일"}.items()}
    day_count = {c.replace("_AMT", "_CO"): label for c, label in day_sales.items()}
    time_sales = {c: label for c, label in {
        "TMZON_00_06_SELNG_AMT": "00-06", "TMZON_06_11_SELNG_AMT": "06-11",
        "TMZON_11_14_SELNG_AMT": "11-14", "TMZON_14_17_SELNG_AMT": "14-17",
        "TMZON_17_21_SELNG_AMT": "17-21", "TMZON_21_24_SELNG_AMT": "21-24"}.items()}
    time_count = {c.replace("_AMT", "_CO"): label for c, label in time_sales.items()}
    gender_traffic = {"ML_FLPOP_CO": "남성", "FML_FLPOP_CO": "여성"}
    age_traffic = {
        "AGRDE_10_FLPOP_CO": "10대", "AGRDE_20_FLPOP_CO": "20대",
        "AGRDE_30_FLPOP_CO": "30대", "AGRDE_40_FLPOP_CO": "40대",
        "AGRDE_50_FLPOP_CO": "50대", "AGRDE_60_ABOVE_FLPOP_CO": "60대 이상",
    }
    time_traffic = {c.replace("SELNG_AMT", "FLPOP_CO"): label for c, label in time_sales.items()}
    # sales_analysis.AXES 의 "성별"/"연령대" 라벨과 동일하게 맞춘다(대응방안 추천이
    # 세그먼트 매출비중을 조회할 때 이 라벨로 찾기 때문에 임의로 바꾸면 안 됨).
    gender_sales = {"ML_SELNG_AMT": "남성", "FML_SELNG_AMT": "여성"}
    age_sales = {
        "AGRDE_10_SELNG_AMT": "10대", "AGRDE_20_SELNG_AMT": "20대",
        "AGRDE_30_SELNG_AMT": "30대", "AGRDE_40_SELNG_AMT": "40대",
        "AGRDE_50_SELNG_AMT": "50대", "AGRDE_60_ABOVE_SELNG_AMT": "60대이상",
    }

    return {
        "기본정보": {
            "상권": str(row["TRDAR_CD_NM"].iloc[0]) if "TRDAR_CD_NM" in row.columns else None,
            "지역유형": str(row["TRDAR_SE_CD_NM"].iloc[0]) if "TRDAR_SE_CD_NM" in row.columns else None,
            "업종": str(row["SVC_INDUTY_CD_NM"].iloc[0]) if "SVC_INDUTY_CD_NM" in row.columns else None,
            "기준분기": int(row["STDR_YYQU_CD"].iloc[0]) if "STDR_YYQU_CD" in row.columns else None,
        },
        "간단분석 정보요약": {
            "월 평균 매출": _fmt_money(sales),
            "선택업종 업소수": int(row["STOR_CO"].iloc[0]) if "STOR_CO" in row.columns and pd.notna(row["STOR_CO"].iloc[0]) else None,
            "일 평균 유동인구": int(round(traffic)) if traffic is not None and not pd.isna(traffic) else None,
            "유동인구 많은 요일": day_cols.get(top_day) if top_day else None,
            "유동인구 많은 시간대": hour_cols.get(top_hour) if top_hour else None,
        },
        "매출분석": {
            "월 평균 매출": _fmt_money(sales),
            "동종업종 중앙값": _fmt_money(industry_sales_mean),
            "동일업종·지역유형 중앙값": _fmt_money(area_sales_mean),
            "전년동분기대비": _fmt_pct(sales_yoy),
            "전분기대비": _fmt_pct(sales_qoq),
        },
        "유동인구 분석": {
            "일 평균 유동인구": int(round(traffic)) if traffic is not None and not pd.isna(traffic) else None,
            "유동인구 많은 요일": day_cols.get(top_day) if top_day else None,
            "유동인구 많은 시간대": hour_cols.get(top_hour) if top_hour else None,
        },
        "업종분석": {
            "선택업종 업소수": int(row["STOR_CO"].iloc[0]) if "STOR_CO" in row.columns and pd.notna(row["STOR_CO"].iloc[0]) else None,
            "동종업종 중앙 업소수": round(industry_store_mean, 1) if industry_store_mean is not None else None,
            "동일업종·지역유형 중앙 업소수": round(area_store_mean, 1) if area_store_mean is not None else None,
        },
        "상단비교요약": comparison_summary,
        "추이": {
            "업소수": _trend_payload(row, df, "STOR_CO"),
            "매출액": _trend_payload(row, df, TARGET_AMOUNT),
            "매출건수": _trend_payload(row, df, "THSMON_SELNG_CO"),
            "유동인구": _trend_payload(row, df, "TOT_FLPOP_CO"),
        },
        "시기별_매출특성": {
            "주중주말": {"매출액비율": _ratio_payload(row, df, weekday_sales, "금액"),
                         "매출건수비율": _ratio_payload(row, df, weekday_count)},
            "요일별": {"매출액비율": _ratio_payload(row, df, day_sales, "금액"),
                       "매출건수비율": _ratio_payload(row, df, day_count)},
            "시간대별": {"매출액비율": _ratio_payload(row, df, time_sales, "금액"),
                        "매출건수비율": _ratio_payload(row, df, time_count)},
        },
        "유동인구_구성": {
            "성별": _ratio_payload(row, df, gender_traffic),
            "연령대별": _ratio_payload(row, df, age_traffic),
            "시간대별": _ratio_payload(row, df, time_traffic),
        },
        "고객군별_매출비중": {
            "성별": _ratio_payload(row, df, gender_sales, "금액"),
            "연령대": _ratio_payload(row, df, age_sales, "금액"),
        },
        "분석결과 해설": " ".join(explanation_parts) if explanation_parts else None,
    }


def _block(title: str, body: str) -> str:
    return f"""
    <section class="card">
      <h2>{html.escape(title)}</h2>
      <div class="body">{body}</div>
    </section>
    """


def svg_line_chart(labels: list[str], series: dict[str, list], value_formatter=None) -> str:
    width, height, left, right, top, bottom = 900, 300, 62, 24, 28, 52
    values = [float(v) for items in series.values() for v in items if v is not None and pd.notna(v)]
    if not values:
        return "<p class='empty'>표시할 추이 데이터가 없습니다.</p>"
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        lo, hi = lo * .9, hi * 1.1 if hi else 1
    pad = (hi - lo) * .12
    lo, hi = max(0, lo - pad), hi + pad
    plot_w, plot_h = width - left - right, height - top - bottom
    x = lambda i: left + (plot_w * i / max(1, len(labels) - 1))
    y = lambda v: top + plot_h * (hi - float(v)) / (hi - lo)
    pieces = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for i in range(5):
        gy = top + plot_h * i / 4
        pieces.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{width-right}" y2="{gy:.1f}" stroke="#e5e7eb"/>')
    for i, label in enumerate(labels):
        pieces.append(f'<text x="{x(i):.1f}" y="{height-18}" text-anchor="middle" class="axis-label">{html.escape(str(label))}</text>')
    for name, items in series.items():
        color = REGION_COLORS.get(name, "#64748b")
        points = [(x(i), y(v), v) for i, v in enumerate(items) if v is not None and pd.notna(v)]
        if len(points) >= 2:
            pieces.append(f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px,py,_ in points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        for px, py, value in points:
            label = value_formatter(value) if value_formatter else f"{value:,.0f}"
            pieces.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/><text x="{px:.1f}" y="{py-9:.1f}" text-anchor="middle" class="value-label">{html.escape(str(label))}</text>')
    pieces.append('</svg><div class="legend">' + ''.join(
        f'<span><i style="background:{REGION_COLORS.get(name, "#64748b")}"></i>{html.escape(name)}</span>' for name in series
    ) + '</div>')
    return ''.join(pieces)


def svg_bar_chart(labels: list[str], values: list, color="#1e40af") -> str:
    valid = [float(v) for v in values if v is not None and pd.notna(v)]
    if not valid:
        return "<p class='empty'>표시할 데이터가 없습니다.</p>"
    width, height, base = 520, 250, 210
    max_v = max(valid) or 1
    step = (width - 60) / max(1, len(labels))
    bars = [f'<svg class="chart" viewBox="0 0 {width} {height}">']
    for i, (label, value) in enumerate(zip(labels, values)):
        if value is None or pd.isna(value):
            continue
        h = 160 * float(value) / max_v
        bx = 42 + i * step + step * .18
        bars.append(f'<rect x="{bx:.1f}" y="{base-h:.1f}" width="{step*.64:.1f}" height="{h:.1f}" rx="5" fill="{color}"/><text x="{bx+step*.32:.1f}" y="{base-h-7:.1f}" text-anchor="middle" class="value-label">{float(value):,.0f}</text><text x="{bx+step*.32:.1f}" y="{base+22}" text-anchor="middle" class="axis-label">{html.escape(label)}</text>')
    return ''.join(bars) + '</svg>'


def _change_badge(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    cls, arrow = ("up", "▲") if value > 0 else (("down", "▼") if value < 0 else ("flat", "–"))
    return f'<span class="change {cls}">{arrow} {abs(value)*100:.1f}%</span>'


def _trend_table(payload: dict, formatter=None) -> str:
    labels, series = payload.get("분기", []), payload.get("계열", {})
    head = '<thead><tr><th>비교 기준</th>' + ''.join(f'<th>{html.escape(x)}</th>' for x in labels) + '<th>전분기 대비</th></tr></thead>'
    rows = []
    for name, values in series.items():
        cells = ''.join(f'<td>{html.escape(str(formatter(v) if formatter and v is not None else (f"{v:,.0f}" if v is not None else "-")))}</td>' for v in values)
        change = _qoq(values[-1], values[-2]) if len(values) > 1 else None
        rows.append(f'<tr><th><i class="dot" style="background:{REGION_COLORS.get(name)}"></i>{html.escape(name)}</th>{cells}<td>{_change_badge(change)}</td></tr>')
    return f'<div class="table-scroll"><table class="trend-table">{head}<tbody>{"".join(rows)}</tbody></table></div>'


def _ratio_table(payload: dict | None, title: str = "") -> str:
    if not payload:
        return ""
    labels, regions = payload.get("labels", []), payload.get("지역별", {})
    raw = payload.get("원시값", {})
    if not labels or not regions:
        return ""
    head = '<thead><tr><th>비교 기준</th>' + ''.join(f'<th>{html.escape(x)}</th>' for x in labels) + '</tr></thead>'
    rows = []
    for region, values in regions.items():
        cells = []
        for i, value in enumerate(values):
            if value is None:
                cells.append('<td>-</td>')
                continue
            raw_value = (raw.get(region) or [None] * len(values))[i]
            if payload.get("값유형") == "금액" and raw_value is not None:
                cells.append(f'<td><strong>{html.escape(str(_fmt_money(raw_value)))}</strong><small>{float(value)*100:.1f}%</small></td>')
            else:
                cells.append(f'<td>{float(value)*100:.1f}%</td>')
        rows.append(f'<tr><th><i class="dot" style="background:{REGION_COLORS.get(region)}"></i>{html.escape(region)}</th>{"".join(cells)}</tr>')
    return (f'<h3>{html.escape(title)}</h3>' if title else '') + f'<div class="table-scroll"><table class="ratio-table">{head}<tbody>{"".join(rows)}</tbody></table></div>'


def _metric_cards(summary: dict) -> str:
    icons = {"업소수": "▦", "업소수_전분기대비": "↗", "월평균매출": "₩", "매출_전분기대비": "％"}
    labels = {"업소수": "업소수", "업소수_전분기대비": "업소수 전분기 대비", "월평균매출": "월평균 매출액", "매출_전분기대비": "매출 전분기 대비"}
    groups = []
    for region, values in summary.items():
        cards = []
        for key in labels:
            value = values.get(key)
            shown = _fmt_money(value) if key == "월평균매출" else (_fmt_pct(value) if "대비" in key else (f"{int(value):,}개" if value is not None else "-"))
            cards.append(f'<div class="metric"><span class="metric-icon">{icons[key]}</span><small>{labels[key]}</small><strong>{html.escape(str(shown or "-"))}</strong></div>')
        groups.append(f'<div class="metric-group"><h3>{html.escape(region)}</h3><div class="metric-grid">{"".join(cards)}</div></div>')
    return '<div class="compare-summary">' + ''.join(groups) + '</div>'


def _trend_section(title: str, payload: dict | None, formatter=None, insight=None) -> str:
    if not payload:
        return ""
    chart = svg_line_chart(payload["분기"], payload["계열"], formatter)
    insight_html = f'<div class="insight"><span>🔎</span><p>{html.escape(str(insight))}</p></div>' if insight else ''
    return _block(title, chart + insight_html + _trend_table(payload, formatter))


def _kv_table(items: dict) -> str:
    rows = []
    for k, v in items.items():
        value = "없음" if v is None else html.escape(str(v))
        rows.append(f"<tr><th>{html.escape(str(k))}</th><td>{value}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _impact_body(payload: dict | None) -> str:
    if not payload:
        return "<p>없음</p>"
    body = [
        f"<p><strong>전체:</strong> {html.escape(str(payload.get('전체')))}</p>",
    ]
    axis = payload.get("축별", {})
    rows = []
    for axis_name, data in axis.items():
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(axis_name))}</th>"
            f"<td>{html.escape(str(data.get('강도')))}</td>"
            f"<td>{html.escape(str(data.get('최대변화_구간')))}</td>"
            f"<td>{html.escape(str(data.get('최대변화량')))}</td>"
            "</tr>"
        )
    body.append(
        "<table>"
        "<thead><tr><th>축</th><th>강도</th><th>최대변화 구간</th><th>최대변화량</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
    return "".join(body)


def _diag_body(payload: dict | None) -> str:
    if not payload:
        return "<p>없음</p>"

    summary_bits = []
    confidence = payload.get("분석_신뢰도", {})
    summary_bits.append(f"<p><strong>분석 상태:</strong> {html.escape(str(confidence.get('판정') or '정보 없음'))}</p>")
    if confidence.get("차단사유"):
        summary_bits.append(f"<p><strong>차단 사유:</strong> {html.escape(', '.join(confidence['차단사유']))}</p>")
    if payload.get("등급") is not None:
        summary_bits.append(f"<p><strong>등급:</strong> {html.escape(str(payload.get('등급')))}</p>")
    if payload.get("긴급도") is not None:
        summary_bits.append(f"<p><strong>긴급도:</strong> {html.escape(str(payload.get('긴급도')))}</p>")
    if payload.get("방향") is not None:
        summary_bits.append(f"<p><strong>방향:</strong> {html.escape(str(payload.get('방향')))}</p>")
    if payload.get("정체") is not None:
        summary_bits.append(f"<p><strong>정체:</strong> {html.escape(str(payload.get('정체')))}</p>")

    parts = [
        "<div class='diag-summary'>"
        + "".join(summary_bits)
        + "</div>"
    ]

    trend = payload.get("동반_변화", {})
    structure = payload.get("구조_변화", {})
    if trend:
        parts.append("<h3>동반 변화 신호</h3>" + _kv_table({
            "관측 패턴": trend.get("판정"),
            "대상 추세": _fmt_pct(trend.get("내_추세")),
            "상권 추세": _fmt_pct(trend.get("상권_추세")),
            "업종 추세": _fmt_pct(trend.get("업종_추세")),
            "해석 제한": trend.get("해석주의"),
        }))
    if structure:
        structure_rows = {}
        for name, values in structure.items():
            if isinstance(values, dict):
                change = values.get("변화율")
                structure_rows[name] = (
                    f"{values.get('처음')} → {values.get('현재')}"
                    + (f" ({_fmt_pct(change)})" if change is not None else "")
                )
            else:
                structure_rows[name] = values
        parts.append("<h3>매출 구성요소의 관측 변화</h3>" + _kv_table(structure_rows))

    for axis_name, axis_payload in payload.get("축_분해", {}).items():
        strong = ", ".join(axis_payload.get("강점", [])) or "없음"
        weak = ", ".join(axis_payload.get("약점", [])) or "없음"
        peer_basis = axis_payload.get("비교기준") or "정보 없음"
        peer_count = axis_payload.get("비교대상수")
        confidence = "낮음" if axis_payload.get("저신뢰경고") else "보통"
        parts.append(
            "<div class='diag-axis'>"
            f"<h3>{html.escape(str(axis_name))}</h3>"
            f"<p><strong>상대적으로 높은 구성비:</strong> {html.escape(str(strong))}</p>"
            f"<p><strong>상대적으로 낮은 구성비:</strong> {html.escape(str(weak))}</p>"
            f"<p><strong>비교:</strong> {html.escape(str(peer_basis))} / "
            f"{html.escape(str(peer_count))}곳 / 신뢰도 {confidence}</p>"
            "</div>"
        )

    pres = payload.get("확인과제", {})
    pres_rows = []
    for k, v in pres.items():
        if k == "권장" and isinstance(v, list):
            value = "<br>".join(html.escape(str(x)) for x in v)
        elif isinstance(v, dict):
            value = html.escape(json.dumps(v, ensure_ascii=False))
        else:
            value = html.escape(str(v))
        pres_rows.append(f"<tr><th>{html.escape(str(k))}</th><td>{value}</td></tr>")

    return (
        "".join(parts)
        + "<table>"
        + "".join(pres_rows)
        + "</table>"
    )


def _ai_body(payload: dict | None) -> str:
    if not payload:
        return "<p>학습된 AI 모델이 없습니다.</p>"
    if payload.get("error"):
        return f"<p>{html.escape(str(payload['error']))}</p>"

    structure = payload.get("구조_특이성", {})
    change = payload.get("변화_이상", {})
    quality = payload.get("데이터_품질", {})
    validation = payload.get("모델검증", {})
    comparison = payload.get("동종비교", {})
    rows = {
        "예측 사용": validation.get("예측사용"),
        "학습 건수": validation.get("학습건수"),
        "이상 판정 기준": validation.get("이상판정기준"),
        "구조 특이성": structure.get("판정"),
        "구조 이상도 백분위": structure.get("이상도_백분위"),
        "변화 이상": change.get("판정"),
        "변화 방향": change.get("방향"),
        "변화 이상도 백분위": change.get("이상도_백분위"),
        "데이터 품질": quality.get("판정"),
        "품질 경고": ", ".join(quality.get("경고", [])) or "없음",
        "일반 비교 사용 가능": quality.get("일반비교사용가능"),
        "동종 비교 기준": comparison.get("비교기준"),
        "동종 비교 대상": comparison.get("비교대상수"),
        "동종 비교 신뢰도": comparison.get("신뢰도"),
    }
    indicators = comparison.get("지표분석", [])[:6]
    driver_rows = "".join(
        "<tr>"
        f"<th>{html.escape(str(item.get('지표')))}</th>"
        f"<td>{html.escape(str(item.get('백분위')))}</td>"
        f"<td>{html.escape(str(item.get('강건_z')))}</td>"
        f"<td>{html.escape(str(item.get('판정')))}</td>"
        "</tr>"
        for item in indicators
    )
    comparison_table = (
        "<h3>동종 대비 주요 이상 지표</h3>"
        "<div class='table-scroll'><table><thead><tr><th>지표</th><th>백분위</th><th>강건 z</th><th>판정</th></tr></thead>"
        f"<tbody>{driver_rows}</tbody></table></div>"
        if driver_rows else ""
    )
    return _kv_table(rows) + comparison_table + f"<p class='muted-note'>{html.escape(str(payload.get('해석주의') or ''))}</p>"


def _external_body(payload: dict | None) -> str:
    if not payload:
        return "<p>외부요인 분석 결과가 없습니다.</p>"
    event = payload.get("문화행사", {})
    weather = payload.get("날씨", {})
    exposure = payload.get("대상분기_문화행사노출", {})
    rows = {
        "용도": "참고 정보만 제공(매출 효과 분석 아님)",
        "데이터 해상도": payload.get("데이터해상도"),
        "인과 추정 여부": payload.get("인과추정"),
        "행사 효과 분석": "사용 안 함",
        "행사 분석 제한": "분기 매출로 행사 전·중·후 효과를 구분할 수 없음",
        "날씨 효과 분석": "사용 안 함",
        "날씨 분석 제한": weather.get("이유"),
        "주변 행사 수": exposure.get("주변행사수"),
        "행사일 수 합": exposure.get("행사일수합"),
        "최근접 행사 거리(m)": exposure.get("최근접행사_m"),
    }
    return (
        _kv_table(rows)
        + f"<p class='muted'>{html.escape(str(payload.get('해석주의') or ''))}</p>"
    )


def render_html_report(report: dict) -> str:
    basic = report.get("기본정보", {})
    diag = report.get("관측_변화_분석", {})
    ai_analysis = report.get("AI_분석", {})
    external_analysis = report.get("외부환경_참고", {})
    narrative = report.get("사장님_요약")
    explanation = report.get("분석결과 해설")
    trends = report.get("추이", {})
    period_sales = report.get("시기별_매출특성", {})
    traffic_ratio = report.get("유동인구_구성", {})

    axis_bits = []
    for axis, payload in diag.get("축_분해", {}).items():
        if payload.get("강점"):
            axis_bits.append(f"{axis} {'·'.join(payload['강점'])} 상대 강세")
        if payload.get("약점"):
            axis_bits.append(f"{axis} {'·'.join(payload['약점'])} 상대 약세")
    axis_summary = ", ".join(axis_bits[:5]) or "뚜렷한 구성비 강·약점이 관측되지 않았습니다."
    trend_insight = (diag.get("동반_변화") or {}).get("판정") or explanation
    structure = diag.get("구조_변화", {})

    ratio_blocks = []
    for section_name, block in period_sales.items():
        amount = _ratio_table(block.get("매출액비율"), f"{section_name} 매출액 비율")
        count = _ratio_table(block.get("매출건수비율"), f"{section_name} 매출건수 비율")
        if amount or count:
            ratio_blocks.append(f'<div class="ratio-section">{amount}{count}</div>')
    traffic_blocks = ''.join(_ratio_table(payload, f"{name} 유동인구 비율") for name, payload in traffic_ratio.items() if payload)

    html_parts = [
        "<!doctype html>",
        '<html lang="ko">',
        "<head>",
        '<meta charset="utf-8" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        "<title>매출 분석 리포트</title>",
        """
        <style>
          * { box-sizing: border-box; }
          body { font-family: Pretendard,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin: 0; background: #f3f6fb; color: #172033; }
          .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }
          .hero { background: linear-gradient(135deg, #172554, #1e3a8a 62%, #7c2d5b); color: white; border-radius: 18px; padding: 26px 30px; box-shadow: 0 14px 35px rgba(30,64,175,.18); }
          .hero h1 { margin: 0 0 10px; font-size: 28px; }
          .hero .meta { opacity: .94; display: flex; gap: 18px; flex-wrap: wrap; font-size: 14px; }
          .section-stack { display: grid; gap: 18px; margin-top: 18px; }
          .card { background: white; border-radius: 16px; padding: 22px; border: 1px solid #e2e8f0; box-shadow: 0 7px 22px rgba(30,41,59,.06); }
          .card h2 { margin: 0 0 18px; padding-left: 12px; border-left: 5px solid #1e40af; font-size: 20px; color: #172554; }
          .card h3 { margin: 20px 0 10px; color: #334155; font-size: 15px; }
          table { width: 100%; border-collapse: collapse; font-size: 13px; }
          th, td { text-align: center; padding: 10px 9px; vertical-align: middle; border: 1px solid #e2e8f0; white-space: nowrap; }
          th { background: #eff6ff; color: #1e3a8a; font-weight: 700; }
          tbody th { text-align: left; background: #f8fafc; }
          .ratio-table td strong { display:block; color:#0f172a; font-size:12px; }
          .ratio-table td small { display:block; margin-top:3px; color:#64748b; }
          .body p { margin: 0 0 10px; line-height: 1.6; }
          .muted { color: #dbeafe; }
          pre { white-space: pre-wrap; word-break: break-word; background: #f8fafc; border-radius: 12px; padding: 14px; overflow: auto; }
          .compare-summary { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
          .metric-group { border:1px solid #dbeafe; border-radius:14px; padding:16px; background:#f8fbff; }
          .metric-group h3 { margin:0 0 13px; color:#1e3a8a; }
          .metric-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
          .metric { min-height:105px; background:white; border:1px solid #e2e8f0; border-radius:12px; padding:13px; display:grid; gap:5px; }
          .metric-icon { width:30px; height:30px; display:grid; place-items:center; border-radius:50%; color:white; background:#1e40af; }
          .metric small { color:#64748b; } .metric strong { font-size:18px; color:#0f172a; }
          .strength-card { margin-top:14px; padding:15px 18px; background:#fff1f7; border-left:5px solid #ec4899; border-radius:10px; color:#831843; }
          .ai-summary { margin:0; padding:18px 20px; background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; color:#1e3a8a; font-size:16px; line-height:1.75; }
          .chart { width:100%; min-width:720px; display:block; overflow:visible; }
          .axis-label { font-size:11px; fill:#64748b; } .value-label { font-size:10px; fill:#334155; font-weight:600; }
          .legend { display:flex; justify-content:center; gap:22px; flex-wrap:wrap; margin:6px 0 16px; font-size:12px; }
          .legend i,.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
          .insight { display:flex; gap:12px; align-items:flex-start; margin:16px 0; padding:15px 18px; background:#fff1f7; border:1px solid #fbcfe8; border-radius:12px; color:#831843; }
          .insight span { font-size:22px; } .insight p { margin:0; }
          .table-scroll { overflow-x:auto; margin-bottom:14px; }
          .change.up { color:#dc2626; } .change.down { color:#2563eb; } .change.flat { color:#64748b; }
          .ratio-section { padding-bottom:8px; border-bottom:1px dashed #cbd5e1; }
          .diag-summary { display: grid; gap: 8px; margin-bottom: 14px; padding: 14px; background: #eff6ff; border-radius: 14px; border: 1px solid #bfdbfe; }
          .diag-summary p { margin: 0; }
          .diag-axis { margin-bottom: 12px; padding: 12px; border-radius: 12px; background: #fff7ed; border: 1px solid #fed7aa; }
          .diag-axis h3 { margin: 0 0 8px; font-size: 15px; color: #9a3412; }
          .diag-axis p { margin: 0 0 8px; }
          details summary { cursor:pointer; color:#64748b; font-weight:600; }
          @media(max-width:760px){.compare-summary{grid-template-columns:1fr}.metric-grid{grid-template-columns:1fr 1fr}.wrap{padding:14px 10px 36px}.card{padding:16px}.chart{min-width:650px}.body:has(.chart){overflow-x:auto}}

          /* 다크모드 — OS 설정(prefers-color-scheme) 기본 반영 + 뷰어 토글(data-theme)이 양쪽으로 덮어씀 */
          @media (prefers-color-scheme: dark) {
            html:not([data-theme="light"]) body { background:#0d1220; color:#e7ebf7; }
            html:not([data-theme="light"]) .card { background:#161d33; border-color:#2b3457; box-shadow:0 7px 22px rgba(0,0,0,.35); }
            html:not([data-theme="light"]) .card h2 { color:#c7d2fe; border-left-color:#6d8ffa; }
            html:not([data-theme="light"]) .card h3 { color:#a4adc4; }
            html:not([data-theme="light"]) th { background:#1c2440; color:#a9c0ff; }
            html:not([data-theme="light"]) tbody th { background:#1a2036; }
            html:not([data-theme="light"]) th, html:not([data-theme="light"]) td { border-color:#2b3457; }
            html:not([data-theme="light"]) .ratio-table td strong { color:#e7ebf7; }
            html:not([data-theme="light"]) .ratio-table td small { color:#8891a3; }
            html:not([data-theme="light"]) pre { background:#1a2036; color:#c9d2e6; }
            html:not([data-theme="light"]) .metric-group { background:#161d33; border-color:#2b3457; }
            html:not([data-theme="light"]) .metric { background:#1c2440; border-color:#2b3457; }
            html:not([data-theme="light"]) .metric small { color:#8891a3; }
            html:not([data-theme="light"]) .metric strong { color:#e7ebf7; }
            html:not([data-theme="light"]) .strength-card,
            html:not([data-theme="light"]) .insight { background:#3a1c30; border-color:#5c2540; color:#f9c9de; }
            html:not([data-theme="light"]) .diag-summary { background:#132038; border-color:#2b3457; }
            html:not([data-theme="light"]) .ai-summary { background:#132038; border-color:#2b3457; color:#c7d2fe; }
            html:not([data-theme="light"]) .diag-axis { background:#2e1f10; border-color:#5c3d17; }
            html:not([data-theme="light"]) .diag-axis h3 { color:#f3b986; }
            html:not([data-theme="light"]) .change.up { color:#f2887f; }
            html:not([data-theme="light"]) .change.down { color:#89a7f2; }
            html:not([data-theme="light"]) .change.flat { color:#8891a3; }
            html:not([data-theme="light"]) .axis-label { fill:#8891a3; }
            html:not([data-theme="light"]) .value-label { fill:#c8d0e0; }
            html:not([data-theme="light"]) details summary { color:#a4adc4; }
          }
          html[data-theme="dark"] body { background:#0d1220; color:#e7ebf7; }
          html[data-theme="dark"] .card { background:#161d33; border-color:#2b3457; box-shadow:0 7px 22px rgba(0,0,0,.35); }
          html[data-theme="dark"] .card h2 { color:#c7d2fe; border-left-color:#6d8ffa; }
          html[data-theme="dark"] .card h3 { color:#a4adc4; }
          html[data-theme="dark"] th { background:#1c2440; color:#a9c0ff; }
          html[data-theme="dark"] tbody th { background:#1a2036; }
          html[data-theme="dark"] th, html[data-theme="dark"] td { border-color:#2b3457; }
          html[data-theme="dark"] .ratio-table td strong { color:#e7ebf7; }
          html[data-theme="dark"] .ratio-table td small { color:#8891a3; }
          html[data-theme="dark"] pre { background:#1a2036; color:#c9d2e6; }
          html[data-theme="dark"] .metric-group { background:#161d33; border-color:#2b3457; }
          html[data-theme="dark"] .metric { background:#1c2440; border-color:#2b3457; }
          html[data-theme="dark"] .metric small { color:#8891a3; }
          html[data-theme="dark"] .metric strong { color:#e7ebf7; }
          html[data-theme="dark"] .strength-card,
          html[data-theme="dark"] .insight { background:#3a1c30; border-color:#5c2540; color:#f9c9de; }
          html[data-theme="dark"] .diag-summary { background:#132038; border-color:#2b3457; }
          html[data-theme="dark"] .ai-summary { background:#132038; border-color:#2b3457; color:#c7d2fe; }
          html[data-theme="dark"] .diag-axis { background:#2e1f10; border-color:#5c3d17; }
          html[data-theme="dark"] .diag-axis h3 { color:#f3b986; }
          html[data-theme="dark"] .change.up { color:#f2887f; }
          html[data-theme="dark"] .change.down { color:#89a7f2; }
          html[data-theme="dark"] .change.flat { color:#8891a3; }
          html[data-theme="dark"] .axis-label { fill:#8891a3; }
          html[data-theme="dark"] .value-label { fill:#c8d0e0; }
          html[data-theme="dark"] details summary { color:#a4adc4; }
        </style>
        """,
        "</head>",
        "<body>",
        '<div class="wrap">',
        '<section class="hero">',
        f"<h1>{html.escape(str(basic.get('상권') or '매출 분석 리포트'))}</h1>",
        f'<div class="meta"><span>업종: {html.escape(str(basic.get("업종") or "-"))}</span><span>지역유형: {html.escape(str(basic.get("지역유형") or "-"))}</span><span>기준분기: {html.escape(str(basic.get("기준분기") or "-"))}</span></div>',
        f'<p class="muted" style="margin:12px 0 0;">분기 집계 기반 · 대상 상권 / 동일 상권유형 중앙값 / 서울 동종업종 중앙값 비교</p>',
        '</section>',
        '<div class="section-stack">',
        _block("AI 핵심 요약 · GPT-4.1", f'<p class="ai-summary">{html.escape(str(narrative))}</p>') if narrative else '',
        _block("핵심 지표 비교", _metric_cards(report.get("상단비교요약", {})) + f'<div class="strength-card"><strong>상권 특성 요약</strong><br>{html.escape(axis_summary)}</div>'),
        _trend_section("업소수 추이", trends.get("업소수"), lambda v: f"{v:,.0f}개", structure.get("시장_상태_해설")),
        _trend_section("매출액 추이", trends.get("매출액"), _fmt_money, explanation),
        _trend_section("매출건수 추이", trends.get("매출건수"), lambda v: f"{v:,.0f}건", structure.get("거래건수_해설")),
        _block("시기별 매출 특성", ''.join(ratio_blocks)) if ratio_blocks else '',
        _trend_section("유동인구 추이", trends.get("유동인구"), lambda v: f"{v:,.0f}명", "유동인구 변화는 매출 원인이 아니라 함께 관측된 신호입니다."),
        _block("유동인구 구성", traffic_blocks) if traffic_blocks else '',
        _block("관측 변화 및 확인 과제", _diag_body(diag)),
        _block("AI 이상 패턴 점검", _ai_body(ai_analysis)),
        _block("날씨·문화행사 참고 정보", _external_body(external_analysis)),
        '</div>',
        f'<details style="margin-top:18px"><summary>분석 원본 JSON 보기</summary>{_block("원본 JSON", f"<pre>{html.escape(json.dumps(report, ensure_ascii=False, indent=2))}</pre>")}</details>',
        '</div>',
        '</body></html>',
    ]
    return "".join(html_parts)
