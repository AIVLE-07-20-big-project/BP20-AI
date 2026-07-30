from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from .trend_logic import get_supported_industries, build_report

router = APIRouter(prefix="/api/v1/online-trend", tags=["OnlineTrend"])


@router.get("/industries")
def get_industries():
    """지원하는 업종 목록 조회"""
    return {"industries": get_supported_industries()}


@router.get("/report", response_class=HTMLResponse)
def get_report(industry: str):
    try:
        result = build_report(industry)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 생성 중 오류가 발생했습니다: {e}")

    return result["html"]


@router.get("/report/summary")
def get_report_summary(industry: str):
    try:
        result = build_report(industry)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 생성 중 오류가 발생했습니다: {e}")

    return {
        "industry": result["industry"],
        "period": result["period"],
        "summary": result["summary"],
    }
