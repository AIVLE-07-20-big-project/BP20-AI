from fastapi import APIRouter
from app.api.v1.endpoints import review

api_router = APIRouter()

api_router.include_router(review.router, prefix="/v1/review", tags=["Review ABSA"])