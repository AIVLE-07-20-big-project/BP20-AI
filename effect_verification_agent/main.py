from fastapi import FastAPI

from app.api.effect_verification import router as effect_router

app = FastAPI(
    title="Effect Verification Agent",
    version="1.0.0"
)

app.include_router(effect_router)

@app.get("/")
def root():
    return {
        "message": "Effect Verification Agent is running!"
    }