"""
검색어 기반 수요예측 - API 라우터

원래 대화형 CLI였던 기능을 API 엔드포인트로 전환.
"""

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
    """
    업종별 검색 트렌드 리포트를 생성해 HTML로 반환한다.

    예) GET /api/v1/online-trend/report?industry=카페
    """
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
    """
    HTML이 아니라 JSON 요약 데이터만 필요할 때 사용 (백엔드 연동용).

    예) GET /api/v1/online-trend/report/summary?industry=카페
    """
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
