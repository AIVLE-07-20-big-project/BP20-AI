import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SOURCE = DATA / "source"
PROCESSED = DATA / "processed"

sales = pd.read_csv(SOURCE / "sales_estimate.csv")
store = pd.read_csv(SOURCE / "store_stats.csv")
pop = pd.read_csv(SOURCE / "foot_traffic.csv")

df = sales.merge(store, on=["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD"], how="left", suffixes=("", "_store"))
df = df.merge(pop, on=["TRDAR_CD", "STDR_YYQU_CD"], how="left", suffixes=("", "_pop"))

repop_path = SOURCE / "resident_population.csv"
if repop_path.exists():
    repop = pd.read_csv(repop_path)
    df = df.merge(repop, on=["TRDAR_CD", "STDR_YYQU_CD"], how="left", suffixes=("", "_repop"))
    print("상주인구 데이터 병합 완료:", repop_path.name)
else:
    print("상주인구 파일이 없어 건너뜁니다:", repop_path.name)

workpop_path = SOURCE / "workplace_population.csv"
if workpop_path.exists():
    workpop = pd.read_csv(workpop_path)
    df = df.merge(workpop, on=["TRDAR_CD", "STDR_YYQU_CD"], how="left", suffixes=("", "_workpop"))
    print("직장인구 데이터 병합 완료:", workpop_path.name)
else:
    print("직장인구 파일이 없어 건너뜁니다:", workpop_path.name)

weather_path = SOURCE / "weather_seoul_quarterly.csv"
if weather_path.exists():
    weather = pd.read_csv(weather_path)
    if "STDR_YYQU_CD" in weather.columns:
        weather = weather.copy()
        weather_drop = [
            c for c in weather.columns
            if c.lower() in {"stn_id", "stn_ko", "stn_en", "info"}
            or weather[c].isna().all()
        ]
        if weather_drop:
            weather = weather.drop(columns=weather_drop)
        df = df.merge(weather, on="STDR_YYQU_CD", how="left", suffixes=("", "_weather"))
        print("날씨 데이터 병합 완료:", weather_path.name)
    else:
        print("날씨 파일은 있으나 STDR_YYQU_CD가 없어 병합하지 못했습니다.")
else:
    print("날씨 파일이 없어 건너뜁니다:", weather_path.name)

print("최종 shape:", df.shape)
print("\n점포 데이터 결측 비율:")
print(df[["STOR_CO", "OPBIZ_RT", "CLSBIZ_RT"]].isna().mean())
print("\n유동인구 결측 비율:")
print(df[["TOT_FLPOP_CO"]].isna().mean())
if "TOT_REPOP_CO" in df.columns:
    print("\n상주인구 결측 비율:")
    print(df[["TOT_REPOP_CO"]].isna().mean())
if "TOT_WRC_POPLTN_CO" in df.columns:
    print("\n직장인구 결측 비율:")
    print(df[["TOT_WRC_POPLTN_CO"]].isna().mean())

PROCESSED.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED / "merged_sales_analysis.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: merged_sales_analysis.csv")
