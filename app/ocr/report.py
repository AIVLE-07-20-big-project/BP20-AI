# AI 가계부(이상지출·예산초과) + 원가·수익성 분석(매입단가·원가율) 통합 HTML 리포트 생성

import argparse
import os
from datetime import date, datetime, timedelta

import pandas as pd

from app.ocr.expense_analysis import check_budget_overage, detect_expense_anomalies
from app.ocr.cost_analysis import calculate_cost_rates, detect_price_changes


def load_all(data_dir: str):
    receipts = pd.read_csv(os.path.join(data_dir, "cafe_expense_receipts.csv"), parse_dates=["TransactionDate"])
    budget = pd.read_csv(os.path.join(data_dir, "cafe_budget.csv"))
    items = pd.read_csv(os.path.join(data_dir, "cafe_expense_items.csv"))
    products = pd.read_csv(os.path.join(data_dir, "cafe_products.csv"))
    orders = pd.read_csv(os.path.join(data_dir, "cafe_sales_orders.csv"), parse_dates=["OrderedDate"])
    return receipts, budget, items, products, orders


def won(n) -> str:
    return f"{n:,.0f}원"


# ------------------------------------------------------------------
# 기간 선택
# ------------------------------------------------------------------
def resolve_period(report_type: str, year: int, month: int, data_min: date, data_max: date):
    # report_type/year/month를 받아 (period_start, period_end, 라벨)을 계산
    if report_type == "monthly":
        if not year or not month:
            raise ValueError("월간 보고서는 --year와 --month를 모두 지정해야 합니다.")
        period_start = date(year, month, 1)
        period_end = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))
        label = f"{year}년 {month}월 · 월간 보고서"
    elif report_type == "yearly":
        if not year:
            raise ValueError("연간 보고서는 --year를 지정해야 합니다.")
        period_start = date(year, 1, 1)
        period_end = date(year, 12, 31)
        label = f"{year}년 · 연간 보고서"
    elif report_type == "full":
        period_start = data_min
        period_end = data_max
        label = "전체 기간 보고서"
    else:
        raise ValueError(f"알 수 없는 report_type: {report_type}")

    period_start = max(period_start, data_min)
    period_end = min(period_end, data_max)
    return period_start, period_end, label


# ------------------------------------------------------------------
# 기존 4개 섹션 (요약/이상지출/예산초과/매입단가/원가율) - 렌더링만 담당
# ------------------------------------------------------------------
def render_summary_cards(orders_in_period: pd.DataFrame, receipts_in_period: pd.DataFrame) -> str:
    total_sales = orders_in_period["TotalAmount"].sum()
    total_expense = receipts_in_period["TotalAmount"].sum()
    net = total_sales - total_expense
    net_class = "figure--loss" if net < 0 else ""

    return f"""
    <section class="docket">
      <div class="docket__item">
        <span class="docket__label">총매출</span>
        <span class="docket__figure">{won(total_sales)}</span>
      </div>
      <div class="docket__item">
        <span class="docket__label">총지출</span>
        <span class="docket__figure">{won(total_expense)}</span>
      </div>
      <div class="docket__item docket__item--net">
        <span class="docket__label">순이익</span>
        <span class="docket__figure {net_class}">{won(net)}</span>
      </div>
    </section>
    """


def render_anomaly_section(anomalies: pd.DataFrame) -> str:
    if anomalies.empty:
        rows = '<tr><td colspan="4" class="empty-row">이상 지출로 탐지된 항목이 없습니다.</td></tr>'
    else:
        rows = ""
        for _, r in anomalies.iterrows():
            stamp = '<span class="stamp stamp--surge">급증</span>' if r["direction"] == "급증" else \
                    '<span class="stamp stamp--drop">급감</span>'
            rows += f"""
            <tr>
              <td>{r['week']}</td>
              <td>{r['category']}</td>
              <td class="num">{won(r['weeklyAmount'])}</td>
              <td class="num">평균 {won(r['categoryAvg'])} (Z={r['zScore']}) {stamp}</td>
            </tr>"""

    return f"""
    <section class="ledger-section">
      <h2><span class="eyebrow">01</span>이상 지출 탐지</h2>
      <p class="section-note">카테고리별 주간 지출을 평소 평균과 비교해, 통계적으로 눈에 띄게 벗어난 주를 표시합니다. (통계는 전체 데이터 기준, 표시는 선택한 기간만)</p>
      <table class="ledger-table">
        <thead><tr><th>주간</th><th>카테고리</th><th class="num">해당 주 지출</th><th class="num">비교</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    """


def render_budget_section(overage: pd.DataFrame) -> str:
    if overage.empty:
        rows = '<tr><td colspan="5" class="empty-row">예산을 초과한 항목이 없습니다.</td></tr>'
    else:
        rows = ""
        for _, r in overage.iterrows():
            if pd.isna(r["overPct"]):
                # 해당 월/카테고리에 예산 자체가 등록되어 있지 않은 경우
                # (BudgetAmount=0이라 초과율 계산이 불가능 - nan% 같은 깨진 표시를 막기 위해 별도 처리)
                rows += f"""
                <tr>
                  <td>{r['YearMonth']}</td>
                  <td>{r['Category']}</td>
                  <td class="num">{won(r['actualAmount'])}</td>
                  <td class="num">-</td>
                  <td><span class="stamp stamp--drop">예산 미설정</span></td>
                </tr>"""
                continue

            pct = min(r["overPct"], 200)  # 시각화 막대가 너무 길어지지 않게 상한
            rows += f"""
            <tr>
              <td>{r['YearMonth']}</td>
              <td>{r['Category']}</td>
              <td class="num">{won(r['actualAmount'])}</td>
              <td class="num">{won(r['BudgetAmount'])}</td>
              <td>
                <div class="bar-track">
                  <div class="bar-fill" style="width:{min(pct,100)}%"></div>
                </div>
                <span class="stamp stamp--over">+{r['overPct']}%</span>
              </td>
            </tr>"""

    return f"""
    <section class="ledger-section">
      <h2><span class="eyebrow">02</span>예산 초과 확인</h2>
      <p class="section-note">월별 · 카테고리별 실제 지출을 예산 목표치와 비교합니다.</p>
      <table class="ledger-table">
        <thead><tr><th>월</th><th>카테고리</th><th class="num">실제 지출</th><th class="num">예산</th><th>초과율</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    """


def render_price_change_section(changes: pd.DataFrame) -> str:
    if changes.empty:
        rows = '<tr><td colspan="4" class="empty-row">유의미한 매입 단가 변동이 없습니다.</td></tr>'
    else:
        rows = ""
        for _, r in changes.iterrows():
            direction = "상승" if r["changePct"] > 0 else "하락"
            stamp_class = "stamp--surge" if r["changePct"] > 0 else "stamp--drop"
            rows += f"""
            <tr>
              <td>{r['itemName']}</td>
              <td class="num">{won(r['previousAvgPrice'])}</td>
              <td class="num">{won(r['recentAvgPrice'])}</td>
              <td><span class="stamp {stamp_class}">{r['changePct']:+.1f}% {direction}</span></td>
            </tr>"""

    return f"""
    <section class="ledger-section">
      <h2><span class="eyebrow">03</span>매입 단가 변화 추적</h2>
      <p class="section-note">선택한 기간의 마지막 시점 기준, 최근 4주 평균 매입 단가를 이전 4주와 비교합니다 (원재료성 품목만 대상).</p>
      <table class="ledger-table">
        <thead><tr><th>품목</th><th class="num">이전 평균</th><th class="num">최근 평균</th><th>변동</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    """


def render_cost_rate_section(cost_rates: pd.DataFrame) -> str:
    rows = ""
    for _, r in cost_rates.iterrows():
        if pd.isna(r["costRatePct"]):
            rows += f"""
            <tr class="row--muted">
              <td>{r['productName']}</td>
              <td class="num">{won(r['salePrice'])}</td>
              <td colspan="2" class="muted-note">{r['note']}</td>
            </tr>"""
        else:
            rate = r["costRatePct"]
            level_class = "bar-fill--high" if rate >= 20 else ("bar-fill--mid" if rate >= 15 else "")
            rows += f"""
            <tr>
              <td>{r['productName']}</td>
              <td class="num">{won(r['salePrice'])}</td>
              <td class="num">{won(r['costPerServing'])}</td>
              <td>
                <div class="bar-track">
                  <div class="bar-fill {level_class}" style="width:{rate}%"></div>
                </div>
                <span class="rate-label">{rate}%</span>
              </td>
            </tr>"""

    return f"""
    <section class="ledger-section">
      <h2><span class="eyebrow">04</span>메뉴별 원가율</h2>
      <p class="section-note">선택한 기간의 마지막 시점까지 중 가장 최근 매입 단가를 기준으로 계산했습니다. 원재료 매입 기록이 없는 메뉴는 계산에서 제외됩니다.</p>
      <table class="ledger-table">
        <thead><tr><th>메뉴</th><th class="num">판매가</th><th class="num">원가</th><th>원가율</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    """


# ------------------------------------------------------------------
# 그래프 섹션 (연간/총기간 보고서 전용)
# ------------------------------------------------------------------
def _svg_bar_chart(labels, series_a, series_b, label_a="매출", label_b="지출") -> str:
    # 매출/지출 그룹 막대그래프 + 각 계열의 추이를 잇는 꺾은선을 함께 그린다
    n = len(labels)
    if n == 0:
        return '<p class="section-note">표시할 데이터가 없습니다.</p>'

    width = max(560, min(60 * n + 140, 900))
    height = 300
    pad_left, pad_right, pad_top, pad_bottom = 70, 20, 30, 46
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom
    group_w = chart_w / n
    bar_w = max(group_w * 0.32, 4)

    max_val = max(max(series_a, default=0), max(series_b, default=0), 1)

    def y_of(v):
        return pad_top + chart_h - (v / max_val * chart_h)

    parts = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        val = max_val * frac
        yy = y_of(val)
        parts.append(
            f'<line x1="{pad_left}" y1="{yy:.1f}" x2="{width-pad_right}" y2="{yy:.1f}" '
            f'stroke="var(--rule)" stroke-width="1" stroke-dasharray="2,3"/>'
        )
        parts.append(
            f'<text x="{pad_left-8}" y="{yy+4:.1f}" font-size="10" text-anchor="end" '
            f'fill="var(--ink-soft)">{val/10000:.0f}만</text>'
        )

    a_points = []  # 매출 막대 중앙 상단 좌표 (추이선 연결용)
    b_points = []  # 지출 막대 중앙 상단 좌표

    for i, label in enumerate(labels):
        gx = pad_left + i * group_w
        a_val, b_val = series_a[i], series_b[i]
        ax = gx + group_w * 0.5 - bar_w - 2
        bx = gx + group_w * 0.5 + 2
        a_h = chart_h - (y_of(a_val) - pad_top)
        b_h = chart_h - (y_of(b_val) - pad_top)
        parts.append(f'<rect x="{ax:.1f}" y="{y_of(a_val):.1f}" width="{bar_w:.1f}" height="{max(a_h,0):.1f}" fill="var(--brass)" opacity="0.55"/>')
        parts.append(f'<rect x="{bx:.1f}" y="{y_of(b_val):.1f}" width="{bar_w:.1f}" height="{max(b_h,0):.1f}" fill="var(--red)" opacity="0.55"/>')
        parts.append(
            f'<text x="{gx+group_w/2:.1f}" y="{height-pad_bottom+18}" font-size="10" '
            f'text-anchor="middle" fill="var(--ink-soft)">{label}</text>'
        )
        a_points.append((ax + bar_w / 2, y_of(a_val)))
        b_points.append((bx + bar_w / 2, y_of(b_val)))

    # 막대 위에 매출/지출 각각의 추이를 잇는 꺾은선 + 점 마커를 얹는다
    a_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in a_points)
    b_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in b_points)
    parts.append(f'<polyline points="{a_path}" fill="none" stroke="var(--brass)" stroke-width="2.5"/>')
    parts.append(f'<polyline points="{b_path}" fill="none" stroke="var(--red)" stroke-width="2.5"/>')
    for x, y in a_points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--brass)" stroke="var(--card)" stroke-width="1"/>')
    for x, y in b_points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--red)" stroke="var(--card)" stroke-width="1"/>')

    legend = (
        f'<rect x="{pad_left}" y="4" width="10" height="10" fill="var(--brass)"/>'
        f'<text x="{pad_left+14}" y="13" font-size="11" fill="var(--ink)">{label_a}</text>'
        f'<rect x="{pad_left+70}" y="4" width="10" height="10" fill="var(--red)"/>'
        f'<text x="{pad_left+84}" y="13" font-size="11" fill="var(--ink)">{label_b}</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;max-width:{width}px;">{legend}{"".join(parts)}</svg>'
    )


def _svg_line_chart(labels, values, color_var="--brass") -> str:
    # 단일 지표(예: 원두 단가) 추이를 꺾은선 그래프로 그린다
    n = len(labels)
    if n == 0:
        return '<p class="section-note">표시할 데이터가 없습니다.</p>'

    width = max(560, min(60 * n + 140, 900))
    height = 220
    pad_left, pad_right, pad_top, pad_bottom = 70, 20, 20, 40
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    max_val = max(values) if values else 1
    min_val = min(values) if values else 0
    span = max(max_val - min_val, 1)

    def x_of(i):
        return pad_left + (i / max(n - 1, 1)) * chart_w

    def y_of(v):
        return pad_top + chart_h - ((v - min_val) / span * chart_h)

    parts = []
    for frac in (0, 0.5, 1.0):
        val = min_val + span * frac
        yy = y_of(val)
        parts.append(
            f'<line x1="{pad_left}" y1="{yy:.1f}" x2="{width-pad_right}" y2="{yy:.1f}" '
            f'stroke="var(--rule)" stroke-width="1" stroke-dasharray="2,3"/>'
        )
        parts.append(
            f'<text x="{pad_left-8}" y="{yy+4:.1f}" font-size="10" text-anchor="end" '
            f'fill="var(--ink-soft)">{val:,.0f}</text>'
        )

    points = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<polyline points="{points}" fill="none" stroke="var({color_var})" stroke-width="2.5"/>')
    for i, v in enumerate(values):
        parts.append(f'<circle cx="{x_of(i):.1f}" cy="{y_of(v):.1f}" r="3" fill="var({color_var})"/>')

    step = max(1, n // 12)
    for i, label in enumerate(labels):
        if i % step != 0 and i != n - 1:
            continue
        parts.append(
            f'<text x="{x_of(i):.1f}" y="{height-pad_bottom+18}" font-size="10" '
            f'text-anchor="middle" fill="var(--ink-soft)">{label}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;max-width:{width}px;">{"".join(parts)}</svg>'
    )


def render_trend_chart_section(orders_in_period: pd.DataFrame, receipts_in_period: pd.DataFrame,
                                freq: str, section_no: str, title: str) -> str:
    # freq='M'(월별) 또는 'Q'(분기별) 단위로 매출/지출 추이 그래프를 그린다
    # pandas 2.2+ 부터 'M'/'Q' 별칭이 폐지되어 'ME'/'QE'(월말/분기말 기준) 사용
    pandas_freq = "ME" if freq == "M" else "QE"
    rev = orders_in_period.set_index("OrderedDate")["TotalAmount"].resample(pandas_freq).sum()
    exp = receipts_in_period.set_index("TransactionDate")["TotalAmount"].resample(pandas_freq).sum()
    idx = sorted(set(rev.index) | set(exp.index))

    if freq == "M":
        labels = [ts.strftime("%y-%m") for ts in idx]
    else:
        labels = [f"{ts.year}Q{(ts.month-1)//3+1}" for ts in idx]

    rev_vals = [float(rev.get(ts, 0)) for ts in idx]
    exp_vals = [float(exp.get(ts, 0)) for ts in idx]

    chart_svg = _svg_bar_chart(labels, rev_vals, exp_vals, "매출", "지출")

    return f"""
    <section class="ledger-section">
      <h2><span class="eyebrow">{section_no}</span>{title}</h2>
      <p class="section-note">선택한 기간 내 매출과 지출을 {"월" if freq == "M" else "분기"} 단위로 비교합니다.</p>
      {chart_svg}
    </section>
    """


def _svg_multi_line_chart(labels, series_dict: dict) -> str:
    # 여러 품목의 지수(index) 추이를 한 그래프에 겹쳐 그린다
    n = len(labels)
    if n == 0 or not series_dict:
        return '<p class="section-note">표시할 데이터가 없습니다.</p>'

    palette = ["#6B4226", "#B23A2E", "#3B6E8F", "#7A8B4A", "#A9752E", "#5C4A72", "#2F6B4F"]

    width = max(560, min(60 * n + 140, 900))
    height = 260
    pad_left, pad_right, pad_top, pad_bottom = 70, 20, 20, 60
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    all_vals = [v for series in series_dict.values() for v in series if v is not None]
    if not all_vals:
        return '<p class="section-note">표시할 데이터가 없습니다.</p>'
    max_val = max(all_vals + [100])
    min_val = min(all_vals + [100])
    span = max(max_val - min_val, 1)

    def x_of(i):
        return pad_left + (i / max(n - 1, 1)) * chart_w

    def y_of(v):
        return pad_top + chart_h - ((v - min_val) / span * chart_h)

    parts = []
    for frac in (0, 0.5, 1.0):
        val = min_val + span * frac
        yy = y_of(val)
        parts.append(
            f'<line x1="{pad_left}" y1="{yy:.1f}" x2="{width-pad_right}" y2="{yy:.1f}" '
            f'stroke="var(--rule)" stroke-width="1" stroke-dasharray="2,3"/>'
        )
        parts.append(
            f'<text x="{pad_left-8}" y="{yy+4:.1f}" font-size="10" text-anchor="end" '
            f'fill="var(--ink-soft)">{val:,.0f}</text>'
        )
    # 기준선(100 = 첫 관측월 대비 변동 없음)
    base_y = y_of(100)
    parts.append(
        f'<line x1="{pad_left}" y1="{base_y:.1f}" x2="{width-pad_right}" y2="{base_y:.1f}" '
        f'stroke="var(--ink-soft)" stroke-width="1" stroke-dasharray="4,2"/>'
    )

    axis_label_y = pad_top + chart_h + 18  # x축(월) 라벨의 y좌표
    legend_top = axis_label_y + 28         # 범례 시작 y좌표 - x축 라벨과 겹치지 않도록 충분히 띄움
    legend_x = pad_left
    for idx, (name, series) in enumerate(series_dict.items()):
        color = palette[idx % len(palette)]

        # None(결측)으로 끊긴 구간을 나눠서 각각 폴리라인으로 그림
        segment = []
        for i, v in enumerate(series):
            if v is None:
                if len(segment) >= 2:
                    pts = " ".join(f"{x_of(j):.1f},{y_of(val):.1f}" for j, val in segment)
                    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
                segment = []
            else:
                segment.append((i, v))
        if len(segment) >= 2:
            pts = " ".join(f"{x_of(j):.1f},{y_of(val):.1f}" for j, val in segment)
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i, v in enumerate(series):
            if v is not None:
                parts.append(f'<circle cx="{x_of(i):.1f}" cy="{y_of(v):.1f}" r="2.5" fill="{color}"/>')

        # 범례 (한 줄에 3개씩, 넘치면 다음 줄로)
        col = idx % 3
        row = idx // 3
        lx = legend_x + col * 150
        ly = legend_top + row * 20
        parts.append(f'<rect x="{lx}" y="{ly-9}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx+14}" y="{ly}" font-size="11" fill="var(--ink)">{name}</text>')

    step = max(1, n // 12)
    for i, label in enumerate(labels):
        if i % step != 0 and i != n - 1:
            continue
        parts.append(
            f'<text x="{x_of(i):.1f}" y="{axis_label_y}" font-size="10" '
            f'text-anchor="middle" fill="var(--ink-soft)">{label}</text>'
        )

    legend_rows = (len(series_dict) - 1) // 3 + 1
    total_height = legend_top + (legend_rows - 1) * 20 + 16  # 마지막 범례 줄 아래 여백 포함

    return (
        f'<svg viewBox="0 0 {width} {total_height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;max-width:{width}px;">{"".join(parts)}</svg>'
    )


def render_purchase_price_index_section(receipts: pd.DataFrame, items: pd.DataFrame,
                                         period_start: date, period_end: date, section_no: str) -> str:
    # 원두뿐 아니라 원재료성(Unit이 있는) 매입 품목 전체의 단가 추이를 보여준다
    merged = items.merge(receipts[["ReceiptID", "TransactionDate"]], on="ReceiptID")
    merged = merged[merged["Unit"].notna()]  # 서비스성 지출(광고비/공과금 등) 제외, 원재료만
    merged = merged[
        (merged["TransactionDate"] >= pd.Timestamp(period_start))
        & (merged["TransactionDate"] <= pd.Timestamp(period_end))
    ]

    if merged.empty:
        chart_svg = '<p class="section-note">매입 기록이 없습니다.</p>'
    else:
        monthly = (
            merged.groupby(["ItemName", pd.Grouper(key="TransactionDate", freq="ME")])["UnitPrice"]
            .mean()
            .reset_index()
        )
        all_months = sorted(monthly["TransactionDate"].unique())
        month_labels = [pd.Timestamp(ts).strftime("%y-%m") for ts in all_months]

        series_dict = {}
        for name in monthly["ItemName"].unique():
            sub = monthly[monthly["ItemName"] == name].set_index("TransactionDate")["UnitPrice"]
            sub = sub.reindex(all_months)
            valid = sub.dropna()
            if valid.empty:
                continue
            base = valid.iloc[0]
            indexed = (sub / base * 100)
            series_dict[name] = [None if pd.isna(v) else round(float(v), 1) for v in indexed]

        chart_svg = _svg_multi_line_chart(month_labels, series_dict)

    return f"""
    <section class="ledger-section">
      <h2><span class="eyebrow">{section_no}</span>월별 매입단가 추이 (품목별 지수)</h2>
      <p class="section-note">원재료성 매입 품목 전체를 대상으로, 각 품목의 첫 관측월 단가를 100으로 놓고 그 대비 변동률을 비교합니다. (점선=100, 기준선)</p>
      {chart_svg}
    </section>
    """

    return f"""
    <section class="ledger-section">
      <h2><span class="eyebrow">{section_no}</span>월별 원두 매입단가 추이</h2>
      <p class="section-note">원두(생두) 월평균 매입 단가(원/kg)의 추이입니다.</p>
      {chart_svg}
    </section>
    """


CSS = """
:root {
  --paper: #E7EDE0;
  --rule: #A9BFA0;
  --ink: #16211B;
  --ink-soft: #4A5A4E;
  --red: #B23A2E;
  --brass: #6B4226;
  --card: #FAF8F1;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  background-image:
    repeating-linear-gradient(var(--paper), var(--paper) 27px, var(--rule) 28px);
  color: var(--ink);
  font-family: 'IBM Plex Sans KR', 'Noto Sans KR', sans-serif;
  padding: 48px 20px 80px;
}
.masthead {
  max-width: 880px;
  margin: 0 auto 32px;
  padding-bottom: 20px;
  border-bottom: 3px double var(--ink);
}
.masthead__eyebrow {
  font-family: 'Special Elite', monospace;
  font-size: 13px;
  letter-spacing: 0.12em;
  color: var(--brass);
  text-transform: uppercase;
}
.masthead h1 {
  font-family: 'Special Elite', monospace;
  font-size: 32px;
  margin: 6px 0 4px;
}
.masthead__meta {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  color: var(--ink-soft);
}
.docket {
  max-width: 880px;
  margin: 0 auto 40px;
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 2px;
  display: flex;
  box-shadow: 3px 3px 0 rgba(22,33,27,0.08);
}
.docket__item {
  flex: 1;
  padding: 20px 24px;
  border-right: 1px dashed var(--rule);
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.docket__item:last-child { border-right: none; }
.docket__label {
  font-size: 12px;
  letter-spacing: 0.08em;
  color: var(--ink-soft);
}
.docket__figure {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 26px;
  font-weight: 600;
}
.figure--loss { color: var(--red); }
.ledger-section {
  max-width: 880px;
  margin: 0 auto 36px;
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 2px;
  padding: 24px 28px 20px;
  box-shadow: 3px 3px 0 rgba(22,33,27,0.06);
}
.ledger-section h2 {
  font-family: 'Special Elite', monospace;
  font-size: 20px;
  margin: 0 0 6px;
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.eyebrow {
  font-size: 13px;
  color: var(--brass);
  border: 1px solid var(--brass);
  padding: 1px 7px;
  border-radius: 2px;
}
.section-note {
  font-size: 13px;
  color: var(--ink-soft);
  margin: 0 0 16px;
}
.ledger-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.ledger-table th {
  text-align: left;
  font-size: 12px;
  letter-spacing: 0.04em;
  color: var(--ink-soft);
  border-bottom: 1px solid var(--ink);
  padding: 6px 8px;
}
.ledger-table td {
  padding: 9px 8px;
  border-bottom: 1px dashed var(--rule);
  vertical-align: middle;
}
.ledger-table .num { font-family: 'IBM Plex Mono', monospace; text-align: right; }
.empty-row, .muted-note {
  color: var(--ink-soft);
  font-style: italic;
  text-align: center;
}
.row--muted td { color: var(--ink-soft); }
.stamp {
  display: inline-block;
  font-family: 'Special Elite', monospace;
  font-size: 11px;
  padding: 2px 8px;
  border: 2px solid var(--red);
  color: var(--red);
  border-radius: 3px;
  transform: rotate(-4deg);
  margin-left: 6px;
  white-space: nowrap;
}
.stamp--drop { border-color: var(--brass); color: var(--brass); }
.stamp--over { border-color: var(--red); color: var(--red); }
.bar-track {
  display: inline-block;
  width: 110px;
  height: 8px;
  background: var(--rule);
  border-radius: 4px;
  overflow: hidden;
  vertical-align: middle;
}
.bar-fill {
  height: 100%;
  background: var(--brass);
}
.bar-fill--mid { background: #C08A3E; }
.bar-fill--high { background: var(--red); }
.rate-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  margin-left: 8px;
}
footer {
  max-width: 880px;
  margin: 24px auto 0;
  font-size: 12px;
  color: var(--ink-soft);
  text-align: center;
}
@media (max-width: 640px) {
  .docket { flex-direction: column; }
  .docket__item { border-right: none; border-bottom: 1px dashed var(--rule); }
  .ledger-table { font-size: 12px; }
  .bar-track { width: 70px; }
}
"""


def build_html(data_dir: str, store_name: str = "온기카페",
               report_type: str = "full", year: int = None, month: int = None) -> str:
    # CLI/로컬 실행용 - CSV 폴더를 읽어서 build_html_from_frames로 위임
    receipts, budget, items, products, orders = load_all(data_dir)
    return build_html_from_frames(receipts, budget, items, products, orders, store_name, report_type, year, month)


def build_html_from_frames(receipts: pd.DataFrame, budget: pd.DataFrame, items: pd.DataFrame,
                            products: pd.DataFrame, orders: pd.DataFrame, store_name: str = "온기카페",
                            report_type: str = "full", year: int = None, month: int = None) -> str:
    # 이미 메모리에 있는 DataFrame들로 리포트를 생성한다
    # 매출(orders) 데이터가 아직 없는 경우(예: Java 백엔드에 Order 엔티티 미구현) 지출 데이터만으로 기간을 계산한다.
    # dropna()를 거치는 이유: 날짜를 파싱하지 못한 값(NaT)이 섞여 있으면 min()/max()가 NaN을 반환해서
    # 바로 뒤의 .date() 호출이 "'float' object has no attribute 'date'"로 죽는 문제를 막기 위함.
    valid_receipt_dates = receipts["TransactionDate"].dropna()
    if valid_receipt_dates.empty:
        raise ValueError(
            "리포트를 생성할 지출(영수증) 데이터가 없습니다. "
            "storeId가 맞는지, 해당 매장에 저장된 영수증이 있는지 확인해주세요."
        )

    valid_order_dates = orders["OrderedDate"].dropna() if not orders.empty else orders["OrderedDate"]

    if valid_order_dates.empty:
        data_min = valid_receipt_dates.min().date()
        data_max = valid_receipt_dates.max().date()
    else:
        data_min = min(valid_receipt_dates.min().date(), valid_order_dates.min().date())
        data_max = max(valid_receipt_dates.max().date(), valid_order_dates.max().date())

    period_start, period_end, period_label = resolve_period(report_type, year, month, data_min, data_max)
    ps_ts, pe_ts = pd.Timestamp(period_start), pd.Timestamp(period_end)

    # ---- 기간 내 데이터만 필터링 (요약 카드 / 그래프용) ----
    receipts_in_period = receipts[(receipts["TransactionDate"] >= ps_ts) & (receipts["TransactionDate"] <= pe_ts)]
    orders_in_period = orders[(orders["OrderedDate"] >= ps_ts) & (orders["OrderedDate"] <= pe_ts)]

    # ---- 이상지출/예산초과: 통계는 전체 데이터 기준으로 계산 후, 결과만 기간에 맞춰 표시 ----
    anomalies_all = detect_expense_anomalies(receipts, z_thresh=1.3)
    if not anomalies_all.empty:
        anomalies_all["week"] = pd.to_datetime(anomalies_all["week"])
        anomalies_in_period = anomalies_all[
            (anomalies_all["week"] >= ps_ts) & (anomalies_all["week"] <= pe_ts)
        ].copy()
        anomalies_in_period["week"] = anomalies_in_period["week"].dt.date
    else:
        anomalies_in_period = anomalies_all

    overage_all = check_budget_overage(receipts, budget)
    if not overage_all.empty:
        start_ym, end_ym = period_start.strftime("%Y-%m"), period_end.strftime("%Y-%m")
        overage_in_period = overage_all[
            (overage_all["YearMonth"] >= start_ym) & (overage_all["YearMonth"] <= end_ym)
        ]
    else:
        overage_in_period = overage_all

    # ---- 매입단가추적/원가율: 선택 기간의 "끝 시점"까지의 데이터만으로 계산 ----
    changes = detect_price_changes(receipts, items, end_date=period_end)
    cost_rates = calculate_cost_rates(receipts, items, products, end_date=period_end)

    sections = [
        render_summary_cards(orders_in_period, receipts_in_period),
        render_anomaly_section(anomalies_in_period),
        render_budget_section(overage_in_period),
        render_price_change_section(changes),
        render_cost_rate_section(cost_rates),
    ]

    if report_type == "yearly":
        sections.append(render_trend_chart_section(
            orders_in_period, receipts_in_period, "M", "05", "월별 매출·지출 추이"
        ))
        sections.append(render_purchase_price_index_section(
            receipts_in_period, items, period_start, period_end, "06"
        ))
    elif report_type == "full":
        sections.append(render_trend_chart_section(
            orders_in_period, receipts_in_period, "M", "05", "월별 매출·지출 추이"
        ))
        sections.append(render_trend_chart_section(
            orders_in_period, receipts_in_period, "Q", "06", "분기별 매출·지출 추이"
        ))
        sections.append(render_purchase_price_index_section(
            receipts_in_period, items, period_start, period_end, "07"
        ))

    body = f"""
    <div class="masthead">
      <div class="masthead__eyebrow">경영 장부 · MANAGEMENT LEDGER</div>
      <h1>{store_name} 가계부 · 원가 리포트</h1>
      <div class="masthead__meta">{period_label} · 집계 기간 {period_start} ~ {period_end} · 생성일 {datetime.now().strftime('%Y-%m-%d')}</div>
    </div>
    {''.join(sections)}
    <footer>본 리포트는 규칙 기반 통계 계산으로 자동 생성되었습니다. (에이전트/LLM 미사용)</footer>
    """

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{store_name} 가계부 · 원가 리포트</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Special+Elite&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="AI 가계부 + 원가분석 통합 HTML 리포트 생성")
    parser.add_argument("--data-dir", default="cafe_synthetic_data")
    parser.add_argument("--output", default=None, help="출력 파일 경로 (기본값: report-type에 따라 자동 지정)")
    parser.add_argument("--store-name", default="온기카페")
    parser.add_argument(
        "--report-type", choices=["monthly", "yearly", "full"], default="full",
        help="monthly(월간) / yearly(연간) / full(총기간) 중 선택"
    )
    parser.add_argument("--year", type=int, default=None, help="monthly/yearly 보고서에 필요한 연도")
    parser.add_argument("--month", type=int, default=None, help="monthly 보고서에 필요한 월(1~12)")
    args = parser.parse_args()

    html = build_html(
        args.data_dir, args.store_name,
        report_type=args.report_type, year=args.year, month=args.month,
    )

    output = args.output
    if output is None:
        if args.report_type == "monthly":
            output = f"cafe_report_{args.year}-{args.month:02d}.html"
        elif args.report_type == "yearly":
            output = f"cafe_report_{args.year}.html"
        else:
            output = "cafe_report_full.html"

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"리포트 생성 완료: {output}")


if __name__ == "__main__":
    main()
