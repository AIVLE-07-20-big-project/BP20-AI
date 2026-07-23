"""
업종 기반 검색 트렌드 조회 스크립트

실행하면 업종을 입력받아, 해당 업종의 대표 메뉴들에 대한
네이버 Search Trend(검색어트렌드) 검색지수를 조회하고 결과를 보여준다.

사전 준비:
- .env 파일에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 설정 (.env.example 참고)
- pip install -r requirements.txt
"""

import os
import json
import time
from datetime import date, timedelta, datetime

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # 같은 폴더의 .env 파일이 있으면 자동으로 환경변수로 불러옴

# ----------------------------------------------------------------------
# 1. 업종별 대표 메뉴 사전
#    - 필요에 따라 계속 추가/수정 가능한 구조
#    - 그룹 5개(NCP 제약)를 넘는 업종은 자동으로 배치 분할됨
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 1. 업종별 메뉴 "후보군" 사전
#    - 여기 있는 메뉴 전부를 진짜로 쓰는 게 아니라, 이 후보들 중 실제 검색량이
#      높은 상위 N개(TOP_N_MENUS)를 select_top_menus()가 API로 골라냄
#    - 후보 수가 많을수록 선정 결과가 더 현실적이지만 API 호출도 늘어남
#    - 그룹 5개(NCP 제약)를 넘는 후보군은 자동으로 배치 분할됨
# ----------------------------------------------------------------------
CANDIDATE_MENU_MAP = {
    "카페": [
        {"groupName": "아메리카노", "keywords": ["아메리카노", "아아", "아이스아메리카노"]},
        {"groupName": "카페라떼", "keywords": ["카페라떼", "라떼"]},
        {"groupName": "카푸치노", "keywords": ["카푸치노"]},
        {"groupName": "바닐라라떼", "keywords": ["바닐라라떼"]},
        {"groupName": "카라멜마키아토", "keywords": ["카라멜마키아토"]},
        {"groupName": "아이스티", "keywords": ["아이스티"]},
        {"groupName": "레몬에이드", "keywords": ["레몬에이드"]},
        {"groupName": "크루아상", "keywords": ["크루아상"]},
        {"groupName": "스콘", "keywords": ["스콘"]},
        {"groupName": "초코케이크", "keywords": ["초코케이크", "초콜릿케이크"]},
        {"groupName": "콜드브루", "keywords": ["콜드브루"]},
        {"groupName": "카페모카", "keywords": ["카페모카"]},
        {"groupName": "티라미수", "keywords": ["티라미수"]},
        {"groupName": "마카롱", "keywords": ["마카롱"]},
        {"groupName": "허니브레드", "keywords": ["허니브레드"]},
    ],
    "한식당": [
        {"groupName": "김치찌개", "keywords": ["김치찌개"]},
        {"groupName": "된장찌개", "keywords": ["된장찌개"]},
        {"groupName": "제육볶음", "keywords": ["제육볶음"]},
        {"groupName": "비빔밥", "keywords": ["비빔밥"]},
        {"groupName": "냉면", "keywords": ["냉면", "물냉면", "비빔냉면"]},
        {"groupName": "갈비탕", "keywords": ["갈비탕"]},
        {"groupName": "삼겹살", "keywords": ["삼겹살"]},
        {"groupName": "순두부찌개", "keywords": ["순두부찌개"]},
        {"groupName": "불고기", "keywords": ["불고기"]},
        {"groupName": "육개장", "keywords": ["육개장"]},
        {"groupName": "잡채", "keywords": ["잡채"]},
    ],
    "분식집": [
        {"groupName": "떡볶이", "keywords": ["떡볶이"]},
        {"groupName": "순대", "keywords": ["순대"]},
        {"groupName": "튀김", "keywords": ["튀김"]},
        {"groupName": "김밥", "keywords": ["김밥"]},
        {"groupName": "라면", "keywords": ["라면", "분식 라면"]},
        {"groupName": "어묵", "keywords": ["어묵"]},
        {"groupName": "만두", "keywords": ["만두"]},
        {"groupName": "쫄면", "keywords": ["쫄면"]},
        {"groupName": "계란찜", "keywords": ["계란찜"]},
    ],
    "치킨집": [
        {"groupName": "후라이드치킨", "keywords": ["후라이드치킨", "후라이드"]},
        {"groupName": "양념치킨", "keywords": ["양념치킨"]},
        {"groupName": "반반치킨", "keywords": ["반반치킨"]},
        {"groupName": "치킨무", "keywords": ["치킨무"]},
        {"groupName": "핫윙", "keywords": ["핫윙", "치킨 윙"]},
        {"groupName": "마늘치킨", "keywords": ["마늘치킨"]},
        {"groupName": "파닭", "keywords": ["파닭"]},
        {"groupName": "순살치킨", "keywords": ["순살치킨"]},
    ],
    "중식당": [
        {"groupName": "짜장면", "keywords": ["짜장면"]},
        {"groupName": "짬뽕", "keywords": ["짬뽕"]},
        {"groupName": "탕수육", "keywords": ["탕수육"]},
        {"groupName": "볶음밥", "keywords": ["볶음밥", "중식 볶음밥"]},
        {"groupName": "마파두부", "keywords": ["마파두부"]},
        {"groupName": "유산슬", "keywords": ["유산슬"]},
        {"groupName": "깐풍기", "keywords": ["깐풍기"]},
        {"groupName": "팔보채", "keywords": ["팔보채"]},
    ],
    "일식당": [
        {"groupName": "초밥", "keywords": ["초밥", "스시"]},
        {"groupName": "우동", "keywords": ["우동"]},
        {"groupName": "라멘", "keywords": ["라멘"]},
        {"groupName": "돈카츠", "keywords": ["돈카츠", "돈까스"]},
        {"groupName": "규동", "keywords": ["규동"]},
        {"groupName": "가라아게", "keywords": ["가라아게"]},
        {"groupName": "사시미", "keywords": ["사시미", "회"]},
        {"groupName": "우나기동", "keywords": ["우나기동", "장어덮밥"]},
    ],
    "피자·양식점": [
        {"groupName": "피자", "keywords": ["피자"]},
        {"groupName": "파스타", "keywords": ["파스타"]},
        {"groupName": "리조또", "keywords": ["리조또"]},
        {"groupName": "스테이크", "keywords": ["스테이크"]},
        {"groupName": "샐러드", "keywords": ["샐러드"]},
        {"groupName": "크림파스타", "keywords": ["크림파스타"]},
        {"groupName": "마르게리타피자", "keywords": ["마르게리타피자", "마르게리타"]},
        {"groupName": "감바스", "keywords": ["감바스"]},
    ],
}

TOP_N_MENUS = 10  # 후보군 중 최종 선정할 메뉴 개수
MENU_SELECTION_CACHE_FILE = "menu_selection_cache.json"
MENU_SELECTION_CACHE_MAX_AGE_DAYS = 7  # 상위 메뉴 순위는 매번 새로 뽑을 필요 없이 주 단위로 갱신

BATCH_SIZE = 5  # NCP 제약: 한 번 요청에 최대 5그룹

# 저검색량(신뢰도 낮음) 판정 기준: 조회 기간 중 검색지수가 0인 날의 비율이
# 이 값 이상이면 "검색 신호가 거의 없는 상품"으로 보고 추천 대상에서 제외한다.
# (STEP 5 검증에서 카라멜마키아토가 이 케이스였음 - 결측 비율이 높아 신호가 노이즈에 묻힘)
LOW_VOLUME_MISSING_THRESHOLD = 0.4  # 40% 이상 결측이면 제외

# ----------------------------------------------------------------------
# 2. 인증 정보 & 엔드포인트
# ----------------------------------------------------------------------
CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")

BASE_URL = "https://naveropenapi.apigw.ntruss.com/datalab/v1/search"
HEADERS = {
    "x-ncp-apigw-api-key-id": CLIENT_ID,
    "x-ncp-apigw-api-key": CLIENT_SECRET,
    "Content-Type": "application/json",
}


def choose_industry():
    """사용자에게 업종을 입력받고, 해당 업종의 메뉴 '후보' 리스트를 반환
    (실제로 쓸 상위 N개는 이후 select_top_menus()가 API로 골라냄)"""
    print("=== 지원하는 업종 목록 ===")
    for name in CANDIDATE_MENU_MAP:
        print(f"  - {name}")

    while True:
        choice = input("\n업종을 입력하세요: ").strip()
        if choice in CANDIDATE_MENU_MAP:
            return choice, CANDIDATE_MENU_MAP[choice]
        print(f"'{choice}'는 지원 목록에 없습니다. 위 목록 중에서 정확히 입력해주세요.")


def get_batches(menu_groups):
    return [menu_groups[i:i + BATCH_SIZE] for i in range(0, len(menu_groups), BATCH_SIZE)]


def fetch_batch(batch, start_date, end_date, time_unit="date"):
    body = {
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "timeUnit": time_unit,
        "keywordGroups": batch,
    }
    response = requests.post(BASE_URL, headers=HEADERS, data=json.dumps(body))
    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()
    return response.json()["results"]


def _load_menu_selection_cache() -> dict:
    if not os.path.exists(MENU_SELECTION_CACHE_FILE):
        return {}
    with open(MENU_SELECTION_CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_menu_selection_cache(cache: dict):
    with open(MENU_SELECTION_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def print_table(rows: list, col_widths: dict, col_labels: dict):
    """딕셔너리 리스트를 '|'로 구분된 정렬된 표 형태로 출력"""
    keys = list(col_labels.keys())
    header = " | ".join(col_labels[k].rjust(col_widths[k]) for k in keys)
    print(header)
    print("-+-".join("-" * col_widths[k] for k in keys))
    for row in rows:
        print(" | ".join(str(row[k]).rjust(col_widths[k]) for k in keys))


def select_top_menus(industry: str, candidates: list, top_n: int = TOP_N_MENUS) -> list:
    """후보 메뉴 전체를 Search Trend API로 조회해 평균 검색지수 상위 N개를 선정.
    업종당 결과는 파일 캐시(MENU_SELECTION_CACHE_FILE)에 저장해, 며칠 이내 재실행 시
    API를 다시 호출하지 않고 재사용한다 (검색 트렌드 순위는 하루 단위로 요동칠 필요가 없음)."""
    cache = _load_menu_selection_cache()
    cached_entry = cache.get(industry)

    if cached_entry:
        cached_at = datetime.fromisoformat(cached_entry["cached_at"])
        age_days = (datetime.now() - cached_at).days
        if age_days < MENU_SELECTION_CACHE_MAX_AGE_DAYS:
            print(f"'{industry}' 상위 메뉴 선정 결과 캐시 사용 (생성 {age_days}일 전, "
                  f"{MENU_SELECTION_CACHE_MAX_AGE_DAYS}일 이내라 재사용)")
            return cached_entry["menu_groups"]
        print(f"'{industry}' 캐시가 {age_days}일 지나 만료됨 -> 새로 선정")
    else:
        print(f"'{industry}' 상위 메뉴 선정 캐시 없음 -> 새로 선정")

    print(f"후보 {len(candidates)}개 중 검색량 상위 {top_n}개를 API로 선정합니다...")
    end_date = date.today()
    start_date = end_date - timedelta(days=90)  # 최근 3개월 평균 검색량 기준으로 순위 산정

    batches = get_batches(candidates)
    avg_scores = []
    for i, batch in enumerate(batches, 1):
        group_names = [g["groupName"] for g in batch]
        print(f"  [선정용 배치 {i}/{len(batches)}] 조회 중... ({group_names})")
        results = fetch_batch(batch, start_date, end_date)
        for group_result in results:
            ratios = [point["ratio"] for point in group_result["data"]]
            avg_score = sum(ratios) / len(ratios) if ratios else 0
            avg_scores.append({"groupName": group_result["title"], "평균검색지수": avg_score})
        if i < len(batches):
            time.sleep(1)

    ranked = sorted(avg_scores, key=lambda x: x["평균검색지수"], reverse=True)
    top_names = {item["groupName"] for item in ranked[:top_n]}
    selected_groups = [c for c in candidates if c["groupName"] in top_names]

    print(f"\n선정 결과 (검색량 상위 {top_n}개):")
    table_rows = [
        {"순위": rank, "메뉴": item["groupName"], "평균검색지수": f"{item['평균검색지수']:.1f}"}
        for rank, item in enumerate(ranked[:top_n], 1)
    ]
    print_table(table_rows,
                col_widths={"순위": 4, "메뉴": 12, "평균검색지수": 10},
                col_labels={"순위": "순위", "메뉴": "메뉴", "평균검색지수": "평균검색지수"})

    cache[industry] = {
        "cached_at": datetime.now().isoformat(),
        "menu_groups": selected_groups,
        "ranking": ranked,
    }
    _save_menu_selection_cache(cache)

    return selected_groups


def fetch_long_term_quarterly(menu_groups, years=10):
    """지정한 메뉴들의 장기(기본 10년) 분기별 검색지수 추이를 조회
    (네이버 API는 timeUnit='quarter'를 지원하지 않아 월 단위로 받아 분기 평균으로 재집계)"""
    end_date = date.today()
    start_year = max(2016, end_date.year - years)  # API는 2016-01-01 이후만 지원
    start_date = date(start_year, end_date.month, 1)

    batches = get_batches(menu_groups)
    all_rows = []
    for i, batch in enumerate(batches, 1):
        group_names = [g["groupName"] for g in batch]
        print(f"[장기 추이 배치 {i}/{len(batches)}] 조회 중... ({group_names})")
        results = fetch_batch(batch, start_date, end_date, time_unit="month")
        for group_result in results:
            for point in group_result["data"]:
                all_rows.append({
                    "date": point["period"],
                    "menu": group_result["title"],
                    "검색지수": point["ratio"],
                })
        if i < len(batches):
            time.sleep(1)

    long_df = pd.DataFrame(all_rows)
    if long_df.empty:
        return long_df
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df["분기"] = long_df["date"].dt.year.astype(str) + "-" + long_df["date"].dt.quarter.astype(str)
    quarterly = long_df.groupby(["menu", "분기"], sort=False)["검색지수"].mean().reset_index()
    return quarterly


def fetch_industry_trend(menu_groups, start_date, end_date):
    all_rows = []
    batches = get_batches(menu_groups)

    for i, batch in enumerate(batches, 1):
        group_names = [g["groupName"] for g in batch]
        print(f"[배치 {i}/{len(batches)}] 조회 중... ({group_names})")
        results = fetch_batch(batch, start_date, end_date)

        for group_result in results:
            for point in group_result["data"]:
                all_rows.append({
                    "date": point["period"],
                    "menu": group_result["title"],
                    "검색지수": point["ratio"],
                })

        if i < len(batches):
            time.sleep(1)

    return pd.DataFrame(all_rows)


def classify_signal(change_pct):
    """가이드라인 STEP 4 룰 엔진 기준으로 증가율을 신호로 변환"""
    if pd.isna(change_pct):
        return "판단불가"
    if change_pct >= 30:
        return "🔴 급증"
    if change_pct >= 10:
        return "🟠 증가"
    if change_pct <= -10:
        return "🔵 감소"
    return "⚪ 평이"


def summarize(df):
    """메뉴별 최근 7일 평균 검색지수와 그 이전 대비 증가율을 계산해 출력"""
    print("\n" + "=" * 60)
    print("결과 읽는 법")
    print("=" * 60)
    print("""
- 검색지수: 조회 기간(최근 3개월) 중 검색량이 가장 많았던 날을 100으로 두고,
  그 대비 상대적인 비율로 나타낸 값 (0~100 사이). 절대 검색 횟수가 아님.
  예) 아메리카노 78.4 = "이 메뉴가 가장 많이 검색된 날의 78.4% 수준으로 검색되고 있다"는 뜻.
  메뉴끼리 지수를 서로 비교하는 것도 가능 (아메리카노 78.4 > 스콘 46.3 = 아메리카노가 더 자주 검색됨).

- 최근7일평균 / 이전14일평균: 값 자체의 절대 크기보다 "방향"이 중요함.

- 증가율(%): (최근 7일 평균 - 이전 14일 평균) / 이전 14일 평균 × 100
  → 최근 들어 검색이 느는 중인지 주는 중인지를 나타냄.

- 신호 판정 기준 (가이드라인 STEP 4 기준):
    증가율 +30% 이상   → 🔴 급증   (재고 +20% 권장)
    증가율 +10%~+30%   → 🟠 증가   (재고 +10% 권장)
    증가율 -10%~+10%   → ⚪ 평이   (기존 유지)
    증가율 -10% 이하   → 🔵 감소   (재고 축소 검토)
""")
    print("=" * 60)

    result_rows = []
    skipped = []
    for menu in df["menu"].unique():
        sub = df[df["menu"] == menu].sort_values("date")

        missing_rate = (sub["검색지수"] == 0).mean()
        if len(sub) < 14:
            skipped.append((menu, len(sub), missing_rate, "표본 부족"))
            continue
        if missing_rate >= LOW_VOLUME_MISSING_THRESHOLD:
            skipped.append((menu, len(sub), missing_rate, "저검색량"))
            continue

        recent_7 = sub.tail(7)["검색지수"].mean()
        prev_14 = sub.tail(21).head(14)["검색지수"].mean()
        change = ((recent_7 - prev_14) / prev_14 * 100) if prev_14 > 0 else float("nan")
        result_rows.append({
            "메뉴": menu,
            "최근7일평균": round(recent_7, 1),
            "이전14일평균": round(prev_14, 1),
            "증가율(%)": round(change, 1),
            "신호": classify_signal(change),
        })

    summary_df = pd.DataFrame(result_rows).sort_values("증가율(%)", ascending=False)
    print("\n=== 메뉴별 검색지수 요약 (최근7일평균 / 이전14일평균 값의 범위: 0~100) ===")
    table_rows = [
        {"메뉴": row["메뉴"], "최근7일평균": f"{row['최근7일평균']:.1f}",
         "이전14일평균": f"{row['이전14일평균']:.1f}", "증가율(%)": f"{row['증가율(%)']:+.1f}",
         "신호": row["신호"]}
        for _, row in summary_df.iterrows()
    ]
    print_table(table_rows,
                col_widths={"메뉴": 10, "최근7일평균": 10, "이전14일평균": 10, "증가율(%)": 8, "신호": 6},
                col_labels={"메뉴": "메뉴", "최근7일평균": "최근7일평균", "이전14일평균": "이전14일평균",
                            "증가율(%)": "증가율(%)", "신호": "신호"})

    if skipped:
        print(f"\n--- 결과에서 제외된 메뉴 (결측 비율 기준: {LOW_VOLUME_MISSING_THRESHOLD:.0%} 이상이면 제외) ---")
        for menu, n, missing_rate, reason in skipped:
            if reason == "표본 부족":
                print(f"  - {menu}: 조회 기간 내 검색 데이터가 {n}일치밖에 없어 신뢰할 만한 평균 계산 불가")
            else:
                print(f"  - {menu}: 결측(검색 0) 비율 {missing_rate:.0%} → 검색량이 너무 적어 신호가 노이즈에 묻힘. "
                      f"이 메뉴는 이 기능 추천 대상에서 제외 권장")

    return summary_df, skipped


def svg_line_chart(quarter_labels: list, values: list, color: str, title: str) -> str:
    """분기별 검색지수 추이를 간단한 SVG 꺾은선 그래프로 생성"""
    width, height = 600, 160
    pad_left, pad_right, pad_top, pad_bottom = 40, 16, 16, 24

    if not values:
        return ""

    v_min, v_max = min(values), max(values)
    v_range = (v_max - v_min) or 1

    n = len(values)
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def x_at(i):
        return pad_left + (i / max(n - 1, 1)) * plot_w

    def y_at(v):
        return pad_top + plot_h - ((v - v_min) / v_range) * plot_h

    points = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(values))

    # x축 라벨: 너무 많으면 겹치므로 1년(4분기) 간격으로만 표시
    label_step = max(1, n // 8)
    x_labels = ""
    for i, label in enumerate(quarter_labels):
        if i % label_step == 0 or i == n - 1:
            x_labels += (f'<text x="{x_at(i):.1f}" y="{height - 6}" font-size="9" '
                         f'fill="#9ca3af" text-anchor="middle">{label}</text>')

    return f"""
    <div style="margin-top:16px;">
      <div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:4px;">{title}</div>
      <svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;background:#fafafa;border-radius:6px;">
        <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" />
        {"".join(f'<circle cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" r="2.5" fill="{color}" />' for i, v in enumerate(values))}
        {x_labels}
        <text x="{pad_left}" y="{pad_top}" font-size="9" fill="#9ca3af">{v_max:.0f}</text>
        <text x="{pad_left}" y="{pad_top + plot_h}" font-size="9" fill="#9ca3af">{v_min:.0f}</text>
      </svg>
    </div>"""


def generate_daily_report_html(industry: str, summary_df: pd.DataFrame, skipped: list,
                                start_date=None, end_date=None, long_term_trend=None) -> str:
    """가이드라인 STEP 6 기준 리포트를 앱/웹 화면에 바로 띄울 수 있는 HTML로 생성"""
    today_str = date.today().strftime("%Y-%m-%d")

    def natural_language_summary() -> str:
        """가이드라인 원본 스펙 예시 형태의 자연어 요약 문장 생성
        예: "최근 '딸기 케이크', '빙수' 검색량이 증가하고 있습니다. 이번 주말 해당 상품 재고를
             평소보다 20% 늘리는 것을 추천합니다." """
        surge_names = summary_df[summary_df["신호"] == "🔴 급증"]["메뉴"].tolist()
        up_names = summary_df[summary_df["신호"] == "🟠 증가"]["메뉴"].tolist()
        down_names = summary_df[summary_df["신호"] == "🔵 감소"]["메뉴"].tolist()

        sentences = []
        if surge_names:
            names_str = ", ".join(f"'{n}'" for n in surge_names)
            sentences.append(
                f"최근 {names_str} 검색량이 크게 증가하고 있습니다. "
                f"이번 주말 해당 상품 재고를 평소보다 20% 늘리는 것을 추천합니다."
            )
        if up_names:
            names_str = ", ".join(f"'{n}'" for n in up_names)
            sentences.append(
                f"{names_str} 검색량도 증가 추세입니다. "
                f"재고를 평소보다 10% 늘리는 것을 추천합니다."
            )
        if down_names:
            names_str = ", ".join(f"'{n}'" for n in down_names)
            sentences.append(
                f"반면 {names_str} 검색량은 감소하고 있어, 재고 축소나 프로모션 검토를 추천합니다."
            )

        if not sentences:
            return "이번 주는 검색량에 특이한 변화를 보인 메뉴가 없습니다. 기존 재고를 유지하세요."
        return " ".join(sentences)

    summary_sentence = natural_language_summary()

    SIGNAL_STYLE = {
        "🔴 급증": ("#e11d48", "#fef2f2", "급증"),
        "🟠 증가": ("#ea580c", "#fff7ed", "증가"),
        "⚪ 평이": ("#6b7280", "#f9fafb", "평이"),
        "🔵 감소": ("#2563eb", "#eff6ff", "감소"),
    }

    def badge(signal):
        color, bg, label = SIGNAL_STYLE.get(signal, ("#6b7280", "#f9fafb", signal))
        return (f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
                f'font-size:12px;font-weight:600;color:{color};background:{bg};'
                f'border:1px solid {color}33;">{label}</span>')

    def action_text(signal):
        return {
            "🔴 급증": "재고를 평소보다 20% 늘리는 것을 권장합니다.",
            "🟠 증가": "재고를 평소보다 10% 늘리는 것을 권장합니다.",
            "⚪ 평이": "특이 변화 없음, 기존 재고를 유지하세요.",
            "🔵 감소": "재고를 축소하거나 프로모션을 검토해볼 시점입니다.",
        }.get(signal, "")

    rows_html = ""
    for _, row in summary_df.iterrows():
        change = row["증가율(%)"]
        # 증가율 숫자 색상을 원부호가 아니라 '신호' 판정과 동일하게 맞춤
        # (평이한데 숫자만 빨갛게 보여 착시를 주는 것 방지)
        change_color, _, _ = SIGNAL_STYLE.get(row["신호"], ("#111827", "#f9fafb", ""))
        rows_html += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;font-weight:600;">{row['메뉴']}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;">{badge(row['신호'])}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:right;color:{change_color};font-weight:600;">
            {change:+.1f}%
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;color:#4b5563;font-size:13px;">
            {action_text(row['신호'])}
          </td>
        </tr>"""

    excluded = skipped  # 사유(표본 부족 / 저검색량) 관계없이 제외된 항목은 모두 안내
    excluded_html = ""
    if excluded:
        items_str = "".join(
            f'<li>{menu}: {"조회 기간 내 검색 데이터가 부족함" if reason == "표본 부족" else f"검색량이 너무 적음(결측 {mr:.0%})"}</li>'
            for menu, n, mr, reason in excluded
        )
        excluded_html = f"""
        <div style="margin-top:20px;padding:12px 16px;background:#f9fafb;border-left:3px solid #d1d5db;
                   color:#6b7280;font-size:13px;border-radius:4px;">
          <strong>이번 리포트에서 제외된 메뉴</strong>
          <ul style="margin:6px 0 0 0;padding-left:18px;">{items_str}</ul>
        </div>"""

    # 신호가 있는(평이가 아닌) 메뉴에 한해 장기 추이 그래프 생성
    long_term_charts_html = ""
    if long_term_trend is not None and not long_term_trend.empty:
        flagged = summary_df[summary_df["신호"] != "⚪ 평이"]
        chart_blocks = ""
        for _, row in flagged.iterrows():
            menu = row["메뉴"]
            color, _, _ = SIGNAL_STYLE.get(row["신호"], ("#111827", "#f9fafb", ""))
            menu_trend = long_term_trend[long_term_trend["menu"] == menu].sort_values("분기")
            if menu_trend.empty:
                continue
            chart_blocks += svg_line_chart(
                quarter_labels=menu_trend["분기"].tolist(),
                values=menu_trend["검색지수"].tolist(),
                color=color,
                title=f"{menu} - 분기별 검색지수 장기 추이",
            )
        if chart_blocks:
            long_term_charts_html = f"""
      <div style="margin-top:24px;padding-top:20px;border-top:1px solid #e5e7eb;">
        <div style="font-size:14px;font-weight:700;color:#111827;margin-bottom:4px;">📈 신호 메뉴 장기 추이</div>
        <div style="font-size:12px;color:#9ca3af;margin-bottom:8px;">
          최근 신호가 감지된 메뉴가 과거에도 반복되는 패턴인지, 이례적인 변화인지 참고하세요.
        </div>
        {chart_blocks}
      </div>"""

        # 평이한(신호 없는) 메뉴 - 드롭다운으로 하나씩 선택해서 보는 섹션
        normal_rows = summary_df[summary_df["신호"] == "⚪ 평이"]
        normal_chart_divs = ""
        options_html = ""
        for idx, (_, row) in enumerate(normal_rows.iterrows()):
            menu = row["메뉴"]
            menu_trend = long_term_trend[long_term_trend["menu"] == menu].sort_values("분기")
            if menu_trend.empty:
                continue
            display = "block" if idx == 0 else "none"
            chart_svg = svg_line_chart(
                quarter_labels=menu_trend["분기"].tolist(),
                values=menu_trend["검색지수"].tolist(),
                color="#6b7280",
                title=f"{menu} - 분기별 검색지수 장기 추이",
            )
            normal_chart_divs += f'<div class="normal-chart" data-menu="{menu}" style="display:{display};">{chart_svg}</div>'
            options_html += f'<option value="{menu}">{menu}</option>'

        if normal_chart_divs:
            long_term_charts_html += f"""
      <div style="margin-top:24px;padding-top:20px;border-top:1px solid #e5e7eb;">
        <div style="font-size:14px;font-weight:700;color:#111827;margin-bottom:4px;">📊 그 외 메뉴 장기 추이 보기</div>
        <div style="font-size:12px;color:#9ca3af;margin-bottom:8px;">
          특이 신호는 없지만, 참고용으로 다른 메뉴의 장기 추이도 확인할 수 있습니다.
        </div>
        <select onchange="document.querySelectorAll('.normal-chart').forEach(function(el){{
                   el.style.display = (el.getAttribute('data-menu') === this.value) ? 'block' : 'none';
                 }}, this)"
                style="padding:6px 10px;border:1px solid #d1d5db;border-radius:6px;font-size:13px;
                       color:#374151;background:#fff;">
          {options_html}
        </select>
        {normal_chart_divs}
      </div>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{industry} 온라인 검색 기반 수요 리포트</title>
</head>
<body style="margin:0;padding:32px;background:#f3f4f6;font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;">
  <div style="max-width:640px;margin:0 auto;background:#ffffff;border-radius:12px;
              box-shadow:0 1px 3px rgba(0,0,0,0.1);overflow:hidden;">
    <div style="padding:24px 28px;background:#111827;color:#ffffff;">
      <div style="font-size:13px;color:#9ca3af;margin-bottom:4px;">{today_str}</div>
      <div style="font-size:20px;font-weight:700;">{industry} 온라인 검색 기반 수요 리포트</div>
      {f'<div style="font-size:12px;color:#9ca3af;margin-top:4px;">조회 기간: {start_date} ~ {end_date}</div>' if start_date and end_date else ''}
    </div>
    <div style="padding:24px 28px;">
      <p style="margin:0 0 20px 0;padding:16px 18px;background:#f0f9ff;border-left:4px solid #0284c7;
                 border-radius:6px;font-size:15px;line-height:1.6;color:#0c4a6e;">
        {summary_sentence}
      </p>
      <details style="margin:0 0 20px 0;font-size:13px;color:#6b7280;">
        <summary style="cursor:pointer;font-weight:600;color:#374151;">이 리포트 읽는 법</summary>
        <div style="margin-top:8px;line-height:1.7;padding:12px 14px;background:#f9fafb;border-radius:6px;">
          · <strong>증가율</strong> = (최근 7일 평균 검색지수 - 이전 14일 평균 검색지수) / 이전 14일 평균 × 100<br>
          · <strong>검색지수</strong>는 조회 기간 중 최고 검색일을 100으로 둔 상대값이며, 절대 검색 횟수가 아닙니다.<br>
          · <strong>신호 기준</strong>: 🔴 급증(+30%↑) · 🟠 증가(+10~30%) · ⚪ 평이(-10~10%) · 🔵 감소(-10%↓)
        </div>
      </details>
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="border-bottom:2px solid #111827;">
            <th style="padding:8px 12px;text-align:left;font-size:13px;color:#6b7280;">메뉴</th>
            <th style="padding:8px 12px;text-align:left;font-size:13px;color:#6b7280;">신호</th>
            <th style="padding:8px 12px;text-align:right;font-size:13px;color:#6b7280;">증가율</th>
            <th style="padding:8px 12px;text-align:left;font-size:13px;color:#6b7280;">권장 조치</th>
          </tr>
        </thead>
        <tbody>{rows_html}
        </tbody>
      </table>
      {excluded_html}
      {long_term_charts_html}
    </div>
  </div>
</body>
</html>"""
    return html


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("환경변수 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 을 먼저 설정해주세요.")

    industry, candidates = choose_industry()

    menu_groups = select_top_menus(industry, candidates, top_n=TOP_N_MENUS)

    end_date = date.today()
    start_date = end_date - timedelta(days=90)  # 기본: 최근 3개월

    print(f"\n'{industry}' 업종 대표 메뉴 {len(menu_groups)}개 조회 시작 "
          f"({start_date} ~ {end_date})")

    trend_df = fetch_industry_trend(menu_groups, start_date, end_date)

    filename = f"search_trend_{industry}.csv"
    trend_df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n원본 데이터 저장 완료: {filename}")

    summary_df, skipped = summarize(trend_df)

    # 신호가 있는(평이가 아닌) 메뉴만 골라 장기 추이 조회 (전체 메뉴 다 조회하면 호출 낭비이므로)
    # 표시 여부와 관계없이 전체 메뉴의 10년 장기 추이를 한 번에 조회
    # (신호 메뉴는 항상 노출, 평이한 메뉴는 리포트에서 드롭다운으로 선택해 볼 수 있게 함)
    print(f"\n전체 메뉴 {len(menu_groups)}개의 10년 장기 추이 조회 중...")
    long_term_trend = fetch_long_term_quarterly(menu_groups, years=10)

    html_report = generate_daily_report_html(industry, summary_df, skipped, start_date, end_date,
                                               long_term_trend)
    report_filename = f"daily_report_{industry}_{date.today().strftime('%Y%m%d')}.html"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"\nHTML 리포트 저장 완료: {report_filename}")
    print("(파일을 더블클릭하거나 브라우저로 열면 바로 확인 가능)")


if __name__ == "__main__":
    main()