from fastapi import APIRouter
from api.v1.endpoints import review

api_router = APIRouter()
api_router.include_router(review.router, prefix="/review", tags=["Review ABSA"])