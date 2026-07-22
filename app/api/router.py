from fastapi import APIRouter
from api.v1.endpoints import review, dashboard

api_router = APIRouter()
api_router.include_router(review.router, prefix="/review", tags=["Review ABSA"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])